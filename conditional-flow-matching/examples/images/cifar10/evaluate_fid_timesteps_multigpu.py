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
import torch_fidelity

from unet import SongUNet
from meanflow_sampler import meanflow_sampler

def generate_timestep_list(N=40, gamma=0.5):
    k = torch.arange(0, N+1)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]

def run_sampling_and_fid(args, model, device, num_steps, use_uniform=True, rank=0, world_size=1):
    # Reset seed to ensure same latents across different runs (per rank)
    # Each rank gets a different seed offset to ensure diversity across GPUs
    # but consistency across runs for the same rank
    torch.manual_seed(args.seed + rank)
    
    mode_str = "uniform" if use_uniform else "non-uniform"
    if rank == 0:
        print(f"\nRunning evaluation for N={num_steps}, mode={mode_str}")
    
    # Setup directories
    sample_dir = os.path.join(args.output_dir, f"samples_N{num_steps}_{mode_str}")
    img_folder = os.path.join(sample_dir, "images")
    
    if rank == 0:
        os.makedirs(img_folder, exist_ok=True)
    
    # Wait for directory creation
    if world_size > 1:
        dist.barrier()
    
    # Determine timesteps
    if use_uniform:
        timesteps = None # meanflow_sampler defaults to uniform
    else:
        timesteps = generate_timestep_list(N=num_steps, gamma=0.5).to(device)
    
    # Sampling workload distribution
    total_samples = args.num_fid_samples
    samples_per_gpu = int(math.ceil(total_samples / world_size))
    
    # Adjust last GPU if total doesn't divide evenly (though ceil usually handles over-provisioning)
    # Simple approach: each GPU does samples_per_gpu, we might generate slightly more than total_samples
    
    n = args.batch_size
    iterations = int(math.ceil(samples_per_gpu / n))
    
    if rank == 0:
        print(f"Sampling {total_samples} images across {world_size} GPUs ({samples_per_gpu} per GPU)...")
    
    # Use tqdm only on rank 0 to avoid clutter
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
                # Unique index across all GPUs
                # Global index = (rank * samples_per_gpu) + local_index
                index = (rank * samples_per_gpu) + total_generated + i
                
                # Only save if we haven't exceeded the requested total (optional, but cleaner)
                if index < total_samples:
                    Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
        
        total_generated += current_batch_size
        
    # Wait for all GPUs to finish sampling
    if world_size > 1:
        dist.barrier()
        
    # FID Calculation (Only on Rank 0)
    fid = 0.0
    if rank == 0:
        print("Computing FID...")
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
    
    # Broadcast FID to all ranks (optional, mostly for return consistency)
    if world_size > 1:
        fid_tensor = torch.tensor([fid if fid is not None else 0.0], device=device)
        dist.broadcast(fid_tensor, src=0)
        fid = fid_tensor.item()
        
    return fid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="fid_evaluation_results")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    # DDP args
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # Initialize Distributed Process Group
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
    
    # Load Model
    if rank == 0:
        print("Loading model...")
        
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,
    ).to(device)
    
    # Load checkpoint
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
        fid_uni = run_sampling_and_fid(args, model, device, N, use_uniform=True, rank=rank, world_size=world_size)
        if rank == 0:
            results["uniform_fid"].append(fid_uni)
        
        # Non-uniform
        fid_non_uni = run_sampling_and_fid(args, model, device, N, use_uniform=False, rank=rank, world_size=world_size)
        if rank == 0:
            results["non_uniform_fid"].append(fid_non_uni)
        
        # Save intermediate results (Rank 0 only)
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
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
