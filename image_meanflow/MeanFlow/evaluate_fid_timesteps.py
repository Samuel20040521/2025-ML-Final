import os
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import shutil

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from scipy import linalg
from glob import glob

from sit import SiT_models
from meanflow_sampler import meanflow_sampler


def generate_timestep_list(N=40, gamma=0.5):
    """Generate non-uniform timesteps using the specified formula."""
    k = torch.arange(0, N+1)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]


def run_sampling_and_fid(args, model, vae, device, rank, num_steps, use_uniform=True):
    """Run sampling and FID computation for a given timestep schedule."""
    # Reset seed to ensure same latents across different runs
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    
    mode_str = "uniform" if use_uniform else "non-uniform"
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Running evaluation for N={num_steps}, mode={mode_str}")
        print(f"{'='*60}")
    
    # Setup directories
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"meanflow-{model_string_name}-{ckpt_string_name}-size-{args.resolution}-" \
                  f"cfg-{args.cfg_scale}-steps-{num_steps}-{mode_str}-seed-{args.global_seed}"
    sample_dir = os.path.join(args.output_dir, folder_name)
    img_folder = os.path.join(sample_dir, "images")
    
    if rank == 0:
        os.makedirs(img_folder, exist_ok=True)
        print(f"Saving .png samples at {sample_dir}")
    
    dist.barrier()
    
    # Determine timesteps
    latent_size = args.resolution // 8
    if use_uniform:
        timesteps = None  # meanflow_sampler defaults to uniform
    else:
        timesteps = generate_timestep_list(N=num_steps, gamma=0.5).to(device)
        timesteps = timesteps.flip(0)  # Reverse for sampling from t=1 to t=0
    print(f"Using timesteps: {timesteps}" if timesteps is not None else "Using uniform timesteps.")

    # Sampling
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    
    if rank == 0:
        print(f"Sampling {total_samples} images (per GPU: {samples_needed_this_gpu})...")
    
    pbar = range(iterations)
    pbar = tqdm(pbar, desc=f"Sampling ({mode_str})") if rank == 0 else pbar
    total = 0
    
    # Pre-create tensors to avoid repeated allocation
    latents_scale = torch.tensor(
        [0.18125, 0.18125, 0.18125, 0.18125],
        device=device
    ).view(1, 4, 1, 1)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.],
        device=device
    ).view(1, 4, 1, 1)
    
    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        
        with torch.no_grad():
            samples = meanflow_sampler(
                model=model, 
                latents=z,
                y=y,
                cfg_scale=args.cfg_scale,
                num_steps=num_steps,
                timesteps=timesteps
            ).to(torch.float32)
            
            # Decode latents to images
            samples = vae.decode((samples - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save images (done outside torch.no_grad to allow GPU to process next batch)
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
        
        total += global_batch_size
    
    dist.barrier()
    
    # FID Calculation (only on rank 0)
    fid = None
    if rank == 0:
        print("Computing FID...")
        
        try:
            if args.resolution == 256 and args.fid_statistics_file:
                print(f"Computing FID using statistics from {args.fid_statistics_file}...")
                
                from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
                from torch.utils.data import Dataset, DataLoader
                
                # Load reference statistics
                stats = np.load(args.fid_statistics_file)
                mu_ref = stats['mu']
                sigma_ref = stats['sigma']
                
                # Extract features from generated samples
                feat_extractor = FeatureExtractorInceptionV3(
                    name='inception-v3-compat',
                    features_list=['2048']
                )
                if torch.cuda.is_available():
                    feat_extractor = feat_extractor.cuda()
                feat_extractor.eval()
                
                # Dataset for loading images
                class SimpleImageDataset(Dataset):
                    def __init__(self, img_paths):
                        self.img_paths = sorted(img_paths)
                    
                    def __len__(self):
                        return len(self.img_paths)
                    
                    def __getitem__(self, idx):
                        img = Image.open(self.img_paths[idx]).convert('RGB')
                        img_array = np.array(img)
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        return img_tensor
                
                img_paths = glob(os.path.join(img_folder, '*.png'))
                dataset = SimpleImageDataset(img_paths)
                dataloader = DataLoader(dataset, batch_size=50, num_workers=4, pin_memory=True)
                
                # Extract features
                features = []
                for batch in tqdm(dataloader, desc="Extracting features for FID"):
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    with torch.no_grad():
                        feats = feat_extractor(batch)
                        # Handle both dict and tuple returns
                        if isinstance(feats, dict):
                            feats = feats['2048']
                        elif isinstance(feats, tuple):
                            feats = feats[0]  # First element is usually the features
                    features.append(feats.cpu().numpy())
                
                features = np.concatenate(features, axis=0)
                mu_gen = np.mean(features, axis=0)
                sigma_gen = np.cov(features, rowvar=False)
                
                # Calculate FID
                diff = mu_gen - mu_ref
                covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_ref), disp=False)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                fid = diff.dot(diff) + np.trace(sigma_gen + sigma_ref - 2 * covmean)
                
                print(f"FID ({mode_str}, N={num_steps}): {fid:.2f}")
            else:
                print("FID statistics file not provided or resolution not 256. Skipping FID computation.")
        
        except Exception as e:
            print(f"Error computing FID: {e}")
            import traceback
            traceback.print_exc()
    
    dist.barrier()
    
    # Cleanup images to save space (optional)
    if args.cleanup_images and rank == 0:
        print(f"Cleaning up images in {img_folder}...")
        shutil.rmtree(img_folder)
    
    return fid


def main(args):
    """
    Run sampling and evaluation for multiple timestep schedules.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
        print(f"Output directory: {args.output_dir}")

    # Load model:
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        **block_kwargs,
    ).to(device)
    
    # Load checkpoint
    state_dict = torch.load(args.ckpt, map_location=f'cuda:{device}')
    model.load_state_dict(state_dict)
    model.eval()
    
    if rank == 0:
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    
    # Steps to evaluate
    N_values = args.timestep_counts
    if rank == 0:
        print(f"Evaluating timestep counts: {N_values}")
    
    results = {
        "N": N_values,
        "uniform_fid": [],
        "non_uniform_fid": []
    }
    
    for N in N_values:
        # Uniform
        fid_uni = run_sampling_and_fid(args, model, vae, device, rank, N, use_uniform=True)
        if rank == 0:
            results["uniform_fid"].append(fid_uni)
        
        # Non-uniform
        fid_non_uni = run_sampling_and_fid(args, model, vae, device, rank, N, use_uniform=False)
        if rank == 0:
            results["non_uniform_fid"].append(fid_non_uni)
        
        # Save intermediate results
        if rank == 0:
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            print(f"Intermediate results saved.")
    
    # Plotting (only on rank 0)
    if rank == 0:
        print("\nGenerating comparison plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(results["N"], results["uniform_fid"], 'o-', label='Uniform Timesteps', linewidth=2, markersize=8)
        plt.plot(results["N"], results["non_uniform_fid"], 's-', label='Non-Uniform Timesteps (γ=0.5)', linewidth=2, markersize=8)
        plt.xlabel('Number of Steps (N)', fontsize=12)
        plt.ylabel('FID', fontsize=12)
        plt.title('FID vs Number of Steps (ImageNet 256x256)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "fid_comparison.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for i, N in enumerate(N_values):
            print(f"N={N:3d}: Uniform FID={results['uniform_fid'][i]:.2f}, "
                  f"Non-Uniform FID={results['non_uniform_fid'][i]:.2f}")
        print("="*60)
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--global-seed", type=int, default=0)
    
    # Model checkpoint
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a MeanFlow checkpoint.")
    parser.add_argument("--output-dir", type=str, default="fid_evaluation_results")
    
    # Model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    
    # Sampling
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    
    # Timestep schedules to evaluate
    parser.add_argument("--timestep-counts", type=int, nargs='+', 
                       default=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                       help="List of timestep counts to evaluate")
    
    # FID computation
    parser.add_argument("--fid-statistics-file", type=str, required=True,
                       help="Path to pre-computed FID statistics file (e.g., imagenet_256_val.npz)")
    
    # Cleanup
    parser.add_argument("--cleanup-images", action="store_true",
                       help="Delete generated images after computing FID to save space")
    
    args = parser.parse_args()
    
    main(args)
