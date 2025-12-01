import os
import json
import math
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import numpy as np

import torch
import torch.distributed as dist
import torch_fidelity

from torchcfm.models.unet.unet import UNetModelWrapper

def generate_timestep_list(N=40, gamma=0.5, device='cpu'):
    k = torch.arange(0, N+1, device=device)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]

def integrate_model_with_traj(model, x0, timesteps, device):
    """
    Manual Euler integration with custom timesteps.
    timesteps: Tensor of shape [T+1]
    """
    xs = [x0]
    vs = []
    x = x0

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i+1]
        dt = t_next - t_curr
        
        # Expand t for batch
        t_input = t_curr.view(1, 1, 1, 1).expand(x.size(0), 1, 1, 1)
        
        v = model(t_input, x)
        vs.append(v)
        x = x + dt * v
        xs.append(x)

    xs = torch.stack(xs, dim=1)
    vs = torch.stack(vs, dim=1)
    return xs, vs

def run_sampling_and_fid(args, model, device, num_steps, use_uniform=True, rank=0, world_size=1):
    # Reset seed to ensure same latents across different runs (per rank)
    torch.manual_seed(args.global_seed + rank)
    
    mode_str = "uniform" if use_uniform else "non-uniform"
    if rank == 0:
        print(f"\nRunning evaluation for N={num_steps}, mode={mode_str}")
    
    # Setup directories
    sample_dir = os.path.join(args.output_dir, f"samples_N{num_steps}_{mode_str}")
    img_folder = os.path.join(sample_dir, "images")
    
    if rank == 0:
        os.makedirs(img_folder, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Determine timesteps
    if use_uniform:
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
    else:
        timesteps = generate_timestep_list(N=num_steps, gamma=0.5, device=device)
    
    # Sampling workload distribution
    total_samples = args.num_fid_samples
    samples_per_gpu = int(math.ceil(total_samples / world_size))
    
    n = args.per_proc_batch_size
    iterations = int(math.ceil(samples_per_gpu / n))
    
    if rank == 0:
        print(f"Sampling {total_samples} images across {world_size} GPUs ({samples_per_gpu} per GPU)...")
    
    iterator = range(iterations)
    if rank == 0:
        iterator = tqdm(iterator)
        
    total_generated = 0
    for _ in iterator:
        current_batch_size = min(n, samples_per_gpu - total_generated)
        if current_batch_size <= 0:
            break
            
        z = torch.randn(current_batch_size, 3, 32, 32, device=device)
        
        with torch.no_grad():
            xs, _ = integrate_model_with_traj(
                model=model,
                x0=z,
                timesteps=timesteps,
                device=device
            )
            xT = xs[:, -1]
            
            # Post-process images
            imgs = (xT * 127.5 + 128).clamp(0, 255).permute(0, 2, 3, 1)
            imgs = imgs.to("cpu", dtype=torch.uint8).numpy()
            
            for i, img in enumerate(imgs):
                index = (rank * samples_per_gpu) + total_generated + i
                if index < total_samples:
                    Image.fromarray(img).save(f"{img_folder}/{index:06d}.png")
        
        total_generated += current_batch_size
        
    if world_size > 1:
        dist.barrier()
        
    # FID Calculation (Only on Rank 0)
    fid = 0.0
    if rank == 0:
        print("Computing FID...")
        # Allow torch.load cached stats with PyTorch>=2.6 safe globals
        try:
            import torch.serialization as _ts
            import numpy as _np
            if hasattr(_ts, "add_safe_globals"):
                _ts.add_safe_globals([_np._core.multiarray._reconstruct])
        except Exception:
            pass

        metrics_args = {
            'input1': img_folder,
            'input2': 'cifar10-train' if args.fid_ref == 'train' else 'cifar10-test',
            'cuda': True,
            'isc': False, 
            'fid': True,
            'kid': False,
            'verbose': False,
        }
        
        metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)
        fid = metrics_dict.get('frechet_inception_distance', None)
        print(f"FID ({mode_str}, N={num_steps}): {fid:.2f}")
        
        # Optional: Cleanup images to save space
        # shutil.rmtree(img_folder)
    
    if world_size > 1:
        fid_tensor = torch.tensor([fid if fid is not None else 0.0], device=device)
        dist.broadcast(fid_tensor, src=0)
        fid = fid_tensor.item()
        
    return fid

def main():
    parser = argparse.ArgumentParser()
    # Model/ckpt
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt).")
    parser.add_argument("--input-dir", type=str, default="./results", help="Base directory for checkpoints")
    parser.add_argument("--model", type=str, default="icfm", help="Model name")
    parser.add_argument("--step", type=int, default=400000, help="Training step")
    parser.add_argument("--num_channel", type=int, default=128, help="Base channel of UNet")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="fid_comparison_results")
    
    # Sampling
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--per-proc-batch-size", type=int, default=100)
    parser.add_argument("--num-fid-samples", type=int, default=10000)
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"])
    
    # DDP
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    # Initialize DDP
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Initialized DDP with world_size={world_size}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    
    # Load checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(args.input_dir, args.model,
                                 f"{args.model}_cifar10_weights_step_{args.step}.pt")
                                 
    if rank == 0:
        print(f"Loading checkpoint: {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("ema_model", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Steps to evaluate
    N_values = [1] + list(range(10, 101, 10)) # 1, 10, 20, ..., 100
    
    results = {
        "N": N_values,
        "uniform_fid": [],
        "non_uniform_fid": []
    }
    
    for N in N_values:
        # Uniform
        fid_uni = run_sampling_and_fid(args, model, device, N, use_uniform=True, rank=rank, world_size=world_size)
        if rank == 0:
            results["uniform_fid"].append(fid_uni)
        
        # Non-uniform
        fid_non_uni = run_sampling_and_fid(args, model, device, N, use_uniform=False, rank=rank, world_size=world_size)
        if rank == 0:
            results["non_uniform_fid"].append(fid_non_uni)
        
        # Save intermediate results (Rank 0 only)
        if rank == 0:
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            
    # Plotting (Rank 0 only)
    if rank == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(results["N"], results["uniform_fid"], 'o-', label='Uniform Timesteps')
        plt.plot(results["N"], results["non_uniform_fid"], 's-', label='Non-Uniform Timesteps')
        plt.xlabel('Number of Steps (N)')
        plt.ylabel('FID')
        plt.title('FID vs Number of Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "fid_comparison.png"))
        print(f"Plot saved to {os.path.join(args.output_dir, 'fid_comparison.png')}")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
