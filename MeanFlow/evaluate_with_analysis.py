import os
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch_fidelity

from unet import SongUNet
from meanflow_sampler import meanflow_sampler


def generate_timestep_list(N=40, gamma=0.5):
    """Generate non-uniform timesteps using the specified formula."""
    k = torch.arange(0, N+1)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    # Reverse to go from 1 to 0 (MeanFlow convention)
    t = torch.flip(t, [0])
    return t  # shape [N+1]


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two velocity fields."""
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos_sim


def meanflow_sampler_with_analysis(
    model, 
    latents,
    cfg_scale=1.0,
    num_steps=1,
    time_steps=None,
    use_nonuniform=False,
    gamma=0.5,
):
    """
    MeanFlow sampler with velocity field analysis
    Returns: samples, analysis_dict
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    # Storage for analysis
    timestep_list = []
    angular_diff_mean = []
    angular_diff_std = []
    angular_diff_per_sample = []
    cosine_sim_mean = []
    velocity_norm_mean = []
    velocity_norm_std = []
    
    prev_velocity = None
    
    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        u = model(latents, noise_labels_r=r, noise_labels_t=t)
        
        # Compute velocity norm
        vel_norm = u.view(batch_size, -1).norm(dim=1)
        velocity_norm_mean.append(vel_norm.mean().item())
        velocity_norm_std.append(vel_norm.std().item())
        timestep_list.append(1.0)
        
        # x_0 = x_1 - u(x_1, 0, 1)
        x0 = latents - u
        
    else:
        z = latents
        # Generate timesteps
        if time_steps is None:
            if use_nonuniform:
                time_steps = generate_timestep_list(N=num_steps, gamma=gamma).to(device)
            else:
                time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        else:
            time_steps = time_steps.to(device)
        
        for i in range(num_steps):
            t_cur = time_steps[i].item()
            t_next = time_steps[i + 1].item()
            
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)

            u = model(z, noise_labels_r=r, noise_labels_t=t)
            
            # Compute velocity norm
            vel_norm = u.view(batch_size, -1).norm(dim=1)
            velocity_norm_mean.append(vel_norm.mean().item())
            velocity_norm_std.append(vel_norm.std().item())
            
            # Compute angular difference if we have a previous velocity
            if prev_velocity is not None:
                cos_sim = compute_cosine_similarity(prev_velocity, u)
                # Convert to angles (in degrees)
                angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / np.pi
                
                angular_diff_mean.append(angles.mean().item())
                angular_diff_std.append(angles.std().item())
                angular_diff_per_sample.append(angles.cpu())
                cosine_sim_mean.append(cos_sim.mean().item())
            
            prev_velocity = u.clone()
            timestep_list.append(t_cur)
            
            # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
            z = z - (t_cur - t_next) * u
        
        x0 = z
    
    analysis_dict = {
        'timesteps': timestep_list,
        'angular_diff_mean': angular_diff_mean,
        'angular_diff_std': angular_diff_std,
        'angular_diff_per_sample': angular_diff_per_sample,
        'cosine_sim_mean': cosine_sim_mean,
        'velocity_norm_mean': velocity_norm_mean,
        'velocity_norm_std': velocity_norm_std,
    }
    
    return x0, analysis_dict


def aggregate_analysis(all_analyses):
    """Aggregate analysis across multiple batches."""
    if not all_analyses:
        return None
    
    # All batches should have same timesteps
    timesteps = all_analyses[0]['timesteps']
    num_timesteps = len(timesteps)
    
    aggregated = {
        'timesteps': timesteps,
        'angular_diff_mean': [],
        'angular_diff_std': [],
        'cosine_sim_mean': [],
        'velocity_norm_mean': [],
        'velocity_norm_std': [],
        'all_angular_diffs': [],  # All per-sample angular differences
    }
    
    # For each timestep, collect statistics across all batches
    for t_idx in range(num_timesteps):
        # Collect velocity norms
        vel_norms = [a['velocity_norm_mean'][t_idx] for a in all_analyses]
        vel_norm_stds = [a['velocity_norm_std'][t_idx] for a in all_analyses]
        
        aggregated['velocity_norm_mean'].append(np.mean(vel_norms))
        aggregated['velocity_norm_std'].append(np.mean(vel_norm_stds))
        
        # Collect angular differences (skip first timestep which has no previous)
        if t_idx > 0:
            ang_idx = t_idx - 1
            ang_means = [a['angular_diff_mean'][ang_idx] for a in all_analyses if ang_idx < len(a['angular_diff_mean'])]
            ang_stds = [a['angular_diff_std'][ang_idx] for a in all_analyses if ang_idx < len(a['angular_diff_std'])]
            cos_means = [a['cosine_sim_mean'][ang_idx] for a in all_analyses if ang_idx < len(a['cosine_sim_mean'])]
            
            # Collect all per-sample angular differences
            all_per_sample = [a['angular_diff_per_sample'][ang_idx] for a in all_analyses if ang_idx < len(a['angular_diff_per_sample'])]
            if all_per_sample:
                all_per_sample_tensor = torch.cat(all_per_sample)
                aggregated['all_angular_diffs'].append(all_per_sample_tensor)
            
            aggregated['angular_diff_mean'].append(np.mean(ang_means))
            aggregated['angular_diff_std'].append(np.mean(ang_stds))
            aggregated['cosine_sim_mean'].append(np.mean(cos_means))
    
    return aggregated


def plot_velocity_analysis(analysis, output_path, num_steps, use_nonuniform=False, gamma=None):
    """Plot velocity field analysis."""
    if analysis is None or len(analysis['timesteps']) <= 1:
        print("Not enough data for analysis plot")
        return
    
    timesteps = analysis['timesteps']
    angular_mean = analysis['angular_diff_mean']
    angular_std = analysis['angular_diff_std']
    cosine_mean = analysis['cosine_sim_mean']
    vel_norm_mean = analysis['velocity_norm_mean']
    vel_norm_std = analysis['velocity_norm_std']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    timestep_type = f"Non-uniform (γ={gamma})" if use_nonuniform else "Uniform"
    fig.suptitle(f'Velocity Field Analysis (MeanFlow, {num_steps} steps, {timestep_type})', fontsize=14)
    
    # Plot 1: Angular difference vs timestep
    if len(angular_mean) > 0:
        ax1 = axes[0, 0]
        # Timesteps for angular diff (exclude first)
        ang_timesteps = timesteps[1:]
        ax1.plot(ang_timesteps, angular_mean, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(ang_timesteps,
                         [m - s for m, s in zip(angular_mean, angular_std)],
                         [m + s for m, s in zip(angular_mean, angular_std)],
                         alpha=0.3, color='blue', label='± 1 Std')
        ax1.set_xlabel('Time t', fontsize=12)
        ax1.set_ylabel('Angular Difference (degrees)', fontsize=12)
        ax1.set_title('Angular Difference Between Consecutive Velocities', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Cosine similarity vs timestep
    if len(cosine_mean) > 0:
        ax2 = axes[0, 1]
        ang_timesteps = timesteps[1:]
        ax2.plot(ang_timesteps, cosine_mean, 'g-', linewidth=2)
        ax2.set_xlabel('Time t', fontsize=12)
        ax2.set_ylabel('Cosine Similarity', fontsize=12)
        ax2.set_title('Cosine Similarity Between Consecutive Velocities', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-1.1, 1.1])
    
    # Plot 3: Velocity norm vs timestep
    ax3 = axes[1, 0]
    ax3.plot(timesteps, vel_norm_mean, 'r-', linewidth=2, label='Mean')
    ax3.fill_between(timesteps,
                     [m - s for m, s in zip(vel_norm_mean, vel_norm_std)],
                     [m + s for m, s in zip(vel_norm_mean, vel_norm_std)],
                     alpha=0.3, color='red', label='± 1 Std')
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Velocity L2 Norm', fontsize=12)
    ax3.set_title('Mean Velocity Magnitude', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Angular difference distribution
    if len(analysis['all_angular_diffs']) > 0:
        ax4 = axes[1, 1]
        all_angles = torch.cat(analysis['all_angular_diffs']).numpy()
        ax4.hist(all_angles, bins=50, edgecolor='black', alpha=0.7, color='blue')
        ax4.axvline(x=np.mean(all_angles), color='r', linestyle='--', 
                    linewidth=2, label=f'Mean={np.mean(all_angles):.2f}°')
        ax4.set_xlabel('Angular Difference (degrees)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Angular Differences', fontsize=12)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved velocity analysis plot to {output_path}")


def save_analysis_csv(analysis, output_path):
    """Save analysis data to CSV."""
    if analysis is None or len(analysis['timesteps']) <= 1:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestep', 'velocity_norm_mean', 'velocity_norm_std', 
                        'angular_diff_mean', 'angular_diff_std', 'cosine_sim_mean'])
        
        timesteps = analysis['timesteps']
        for i, t in enumerate(timesteps):
            row = [t, analysis['velocity_norm_mean'][i], analysis['velocity_norm_std'][i]]
            
            # Angular diff starts from second timestep
            if i > 0:
                ang_idx = i - 1
                row.extend([
                    analysis['angular_diff_mean'][ang_idx],
                    analysis['angular_diff_std'][ang_idx],
                    analysis['cosine_sim_mean'][ang_idx]
                ])
            else:
                row.extend(['', '', ''])
            
            writer.writerow(row)
    
    print(f"Saved analysis data to {output_path}")


def main(args):
    """
    Run sampling and evaluation for unconditional CIFAR-10 with velocity field analysis.
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
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,  # Unconditional
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=True)
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create folder to save samples:
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"meanflow-cifar10-unconditional-{ckpt_string_name}-" \
                  f"steps-{args.num_steps}-seed-{args.global_seed}"
    eval_fid_dir = f"{args.sample_dir}/{folder_name}"
    img_folder = os.path.join(eval_fid_dir, 'images')
    if rank == 0:
        os.makedirs(eval_fid_dir, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        print(f"Saving .png samples at {eval_fid_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using {args.num_steps}-step sampling")
        if args.use_nonuniform:
            print(f"Using NON-UNIFORM timesteps with gamma={args.gamma}")
        else:
            print(f"Using UNIFORM timesteps")
        if args.analyze_velocity:
            print(f"Velocity field analysis enabled")
    
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    # Storage for velocity analysis
    all_analyses = []
    
    for _ in pbar:
        # Sample noise at full resolution for CIFAR-10
        z = torch.randn(n, 3, 32, 32, device=device)

        # Sample images:
        with torch.no_grad():
            if args.analyze_velocity and args.num_steps > 1:
                samples, analysis = meanflow_sampler_with_analysis(
                    model=model, 
                    latents=z,
                    cfg_scale=1.0,
                    num_steps=args.num_steps,
                    use_nonuniform=args.use_nonuniform,
                    gamma=args.gamma,
                )
                all_analyses.append(analysis)
            else:
                samples = meanflow_sampler(
                    model=model, 
                    latents=z,
                    cfg_scale=1.0,
                    num_steps=args.num_steps,
                )
            
            # Convert to [0, 255] range
            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    
    # Aggregate and plot velocity analysis (only on rank 0)
    if rank == 0 and args.analyze_velocity and len(all_analyses) > 0:
        print(f"Aggregating velocity field analysis from {len(all_analyses)} batches...")
        aggregated = aggregate_analysis(all_analyses)
        
        if aggregated is not None:
            # Plot analysis
            plot_path = os.path.join(eval_fid_dir, "velocity_analysis.png")
            plot_velocity_analysis(aggregated, plot_path, args.num_steps, args.use_nonuniform, args.gamma)
            
            # Save CSV data
            csv_path = os.path.join(eval_fid_dir, "velocity_analysis.csv")
            save_analysis_csv(aggregated, csv_path)
            
            # Print summary statistics
            if len(aggregated['angular_diff_mean']) > 0:
                print(f"\n=== Velocity Field Analysis ===")
                print(f"Mean angular difference: {np.mean(aggregated['angular_diff_mean']):.2f}°")
                print(f"Std of angular difference: {np.std(aggregated['angular_diff_mean']):.2f}°")
                print(f"Min angular difference: {np.min(aggregated['angular_diff_mean']):.2f}°")
                print(f"Max angular difference: {np.max(aggregated['angular_diff_mean']):.2f}°")
                print(f"Mean cosine similarity: {np.mean(aggregated['cosine_sim_mean']):.4f}")
                print(f"Mean velocity norm: {np.mean(aggregated['velocity_norm_mean']):.4f}")
    
    # Calculate FID and IS metrics (only on rank 0)
    if rank == 0 and args.compute_metrics:
        print(f"\nComputing evaluation metrics...")
        
        metrics_args = {
            'input1': img_folder,
            'input2': 'cifar10-train' if args.fid_ref == 'train' else 'cifar10-test',
            'cuda': True,
            'isc': True,
            'fid': True,
            'kid': False,
            'verbose': True,
        }
        
        metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)
        
        fid = metrics_dict.get('frechet_inception_distance', None)
        is_mean = metrics_dict.get('inception_score_mean', None)
        is_std = metrics_dict.get('inception_score_std', None)
        
        print(f"\n===== Evaluation Results =====")
        if fid is not None:
            print(f"FID: {fid:.2f}")
        if is_mean is not None:
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
            
        # Save results
        results = {
            'fid': fid,
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'num_samples': total_samples,
            'num_steps': args.num_steps,
            'checkpoint': args.ckpt,
        }
        
        metrics_file = os.path.join(eval_fid_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
        
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)
    # logging/saving:
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a MeanFlow checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # sampling
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--num-steps", type=int, default=1, help="Number of sampling steps")
    
    # Evaluation metrics
    parser.add_argument("--compute-metrics", action="store_true", help="Compute FID and IS after sampling")
    parser.add_argument("--fid-ref", type=str, default="train", choices=["train", "test"],
                       help="Reference dataset for FID computation")
    
    # Velocity field analysis
    parser.add_argument("--analyze-velocity", action="store_true", help="Analyze velocity field during sampling")
    
    # Non-uniform timesteps
    parser.add_argument("--use-nonuniform", action="store_true", help="Use non-uniform timesteps")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter for non-uniform timesteps (default: 0.5)")

    args = parser.parse_args()
    
    main(args)
