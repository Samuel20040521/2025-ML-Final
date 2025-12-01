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
import torch_fidelity

from unet import SongUNet
from meanflow_sampler import meanflow_sampler

def generate_timestep_list(N=40, gamma=0.5):
    k = torch.arange(0, N+1)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]

def run_sampling_and_fid(args, model, device, num_steps, use_uniform=True):
    # Reset seed to ensure same latents across different runs
    torch.manual_seed(args.seed)
    
    mode_str = "uniform" if use_uniform else "non-uniform"
    print(f"\nRunning evaluation for N={num_steps}, mode={mode_str}")
    
    # Setup directories
    sample_dir = os.path.join(args.output_dir, f"samples_N{num_steps}_{mode_str}")
    img_folder = os.path.join(sample_dir, "images")
    os.makedirs(img_folder, exist_ok=True)
    
    # Determine timesteps
    if use_uniform:
        timesteps = None # meanflow_sampler defaults to uniform
    else:
        timesteps = generate_timestep_list(N=num_steps, gamma=0.5).to(device)
        # Ensure 0 to 1 range (generate_timestep_list returns 0 to 1)
    
    # Sampling
    n = args.batch_size
    total_samples = args.num_fid_samples
    iterations = int(math.ceil(total_samples / n))
    
    print(f"Sampling {total_samples} images...")
    total = 0
    for _ in tqdm(range(iterations)):
        current_batch_size = min(n, total_samples - total)
        if current_batch_size <= 0:
            break
            
        z = torch.randn(current_batch_size, 3, 32, 32, device=device)
        
        with torch.no_grad():
            samples = meanflow_sampler(
                model=model, 
                latents=z,
                cfg_scale=1.0,
                num_steps=num_steps,
                timesteps=timesteps
            )
            
            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            for i, sample in enumerate(samples):
                index = total + i
                Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
        
        total += current_batch_size
        
    # FID Calculation
    print("Computing FID...")
    metrics_args = {
        'input1': img_folder,
        'input2': 'cifar10-train' if args.fid_ref == 'train' else 'cifar10-test',
        'cuda': True,
        'isc': False, # Skip IS for speed if only FID is needed, or enable if user wants
        'fid': True,
        'kid': False,
        'verbose': False,
    }
    
    metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)
    fid = metrics_dict.get('frechet_inception_distance', None)
    print(f"FID ({mode_str}, N={num_steps}): {fid:.2f}")
    
    # Cleanup images to save space (optional, maybe keep them?)
    # shutil.rmtree(img_folder) 
    
    return fid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="fid_evaluation_results")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_fid_samples", type=int, default=50000) # Reduced default for speed, user can increase
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Load Model
    print("Loading model...")
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,
    ).to(device)
    
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
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
        fid_uni = run_sampling_and_fid(args, model, device, N, use_uniform=True)
        results["uniform_fid"].append(fid_uni)
        
        # Non-uniform
        fid_non_uni = run_sampling_and_fid(args, model, device, N, use_uniform=False)
        results["non_uniform_fid"].append(fid_non_uni)
        
        # Save intermediate results
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
    # Plotting
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

if __name__ == "__main__":
    main()
