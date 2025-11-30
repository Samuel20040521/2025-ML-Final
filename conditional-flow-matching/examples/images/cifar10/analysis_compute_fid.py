import os
import json
import math
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_fidelity
import numpy as np

from torchdiffeq import odeint
try:
    from torchdyn.core import NeuralODE
    HAS_TORCHDYN = True
except Exception:
    HAS_TORCHDYN = False

from torchcfm.models.unet.unet import UNetModelWrapper


def integrate_model_with_traj(model, x0, steps, device):
    """
    Manual Euler integration that returns trajectories.
    Returns:
        xs: [B, T+1, C, H, W]
        vs: [B, T,   C, H, W]
    """
    ts = torch.linspace(0, 1, steps + 1, device=device)
    xs = [x0]
    vs = []

    x = x0
    dt = 1.0 / steps

    for i in range(steps):
        t = ts[i].expand(x.size(0), 1, 1, 1)
        v = model(t, x)
        vs.append(v)
        x = x + dt * v
        xs.append(x)

    xs = torch.stack(xs, dim=1)  # [B, T+1, C, H, W]
    vs = torch.stack(vs, dim=1)  # [B, T, C, H, W]
    return xs, vs


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two velocity fields."""
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos_sim


def compute_wasserstein2_distance(v1, v2):
    """
    Compute 2-Wasserstein distance between consecutive velocity fields.
    For Gaussians, W2^2 = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2(Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})
    
    As an approximation, we compute the empirical 2-Wasserstein distance:
    W2^2 ≈ ||mean(v1) - mean(v2)||^2 + ||std(v1) - std(v2)||^2
    
    Args:
        v1, v2: [B, C, H, W] velocity tensors
    Returns:
        w2_dist: scalar 2-Wasserstein distance
    """
    # Flatten to [B, D]
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    
    # Compute means
    mu1 = v1_flat.mean(dim=0)  # [D]
    mu2 = v2_flat.mean(dim=0)  # [D]
    
    # Compute standard deviations
    std1 = v1_flat.std(dim=0)  # [D]
    std2 = v2_flat.std(dim=0)  # [D]
    
    # Empirical W2 distance
    mean_diff = (mu1 - mu2).norm().item()
    std_diff = (std1 - std2).norm().item()
    
    w2_dist = math.sqrt(mean_diff**2 + std_diff**2)
    
    return w2_dist


def analyze_trajectories(vs):
    """
    Analyze velocity field trajectories.
    vs: [B, T, C, H, W]
    
    Returns:
        analysis_dict with statistics per timestep
    """
    B, T, C, H, W = vs.shape
    
    # Compute velocity magnitudes for each timestep
    vel_norms = vs.view(B, T, -1).norm(dim=2)  # [B, T]
    velocity_norm_mean = vel_norms.mean(dim=0).cpu().numpy()  # [T]
    velocity_norm_std = vel_norms.std(dim=0).cpu().numpy()  # [T]
    
    # Compute angular differences between consecutive timesteps
    angular_diff_mean = []
    angular_diff_std = []
    cosine_sim_mean = []
    all_angular_diffs = []
    wasserstein2_dists = []
    
    for t in range(T - 1):
        v_cur = vs[:, t]      # [B, C, H, W]
        v_next = vs[:, t + 1]  # [B, C, H, W]
        
        # Compute cosine similarity
        cos_sim = compute_cosine_similarity(v_cur, v_next)
        
        # Convert to angles (in degrees)
        angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / np.pi
        
        # Compute 2-Wasserstein distance
        w2_dist = compute_wasserstein2_distance(v_cur, v_next)
        
        angular_diff_mean.append(angles.mean().item())
        angular_diff_std.append(angles.std().item())
        cosine_sim_mean.append(cos_sim.mean().item())
        all_angular_diffs.append(angles.cpu())
        wasserstein2_dists.append(w2_dist)
    
    analysis_dict = {
        'velocity_norm_mean': velocity_norm_mean.tolist(),
        'velocity_norm_std': velocity_norm_std.tolist(),
        'angular_diff_mean': angular_diff_mean,
        'angular_diff_std': angular_diff_std,
        'cosine_sim_mean': cosine_sim_mean,
        'all_angular_diffs': all_angular_diffs,
        'wasserstein2_dists': wasserstein2_dists,
    }
    
    return analysis_dict


def aggregate_analyses(all_analyses, num_steps):
    """Aggregate analysis across multiple batches."""
    if not all_analyses:
        return None
    
    timesteps = np.linspace(0, 1, num_steps + 1)
    
    aggregated = {
        'timesteps': timesteps.tolist(),
        'angular_diff_mean': [],
        'angular_diff_std': [],
        'cosine_sim_mean': [],
        'velocity_norm_mean': [],
        'velocity_norm_std': [],
        'all_angular_diffs': [],
        'wasserstein2_mean': [],
        'wasserstein2_std': [],
    }
    
    # Aggregate velocity norms (T timesteps)
    for t_idx in range(num_steps):
        vel_norms = [a['velocity_norm_mean'][t_idx] for a in all_analyses]
        vel_norm_stds = [a['velocity_norm_std'][t_idx] for a in all_analyses]
        
        aggregated['velocity_norm_mean'].append(np.mean(vel_norms))
        aggregated['velocity_norm_std'].append(np.mean(vel_norm_stds))
    
    # Aggregate angular differences (T-1 differences)
    for ang_idx in range(num_steps - 1):
        ang_means = [a['angular_diff_mean'][ang_idx] for a in all_analyses]
        ang_stds = [a['angular_diff_std'][ang_idx] for a in all_analyses]
        cos_means = [a['cosine_sim_mean'][ang_idx] for a in all_analyses]
        w2_dists = [a['wasserstein2_dists'][ang_idx] for a in all_analyses]
        
        # Collect all per-sample angular differences
        all_per_sample = [a['all_angular_diffs'][ang_idx] for a in all_analyses]
        if all_per_sample:
            all_per_sample_tensor = torch.cat(all_per_sample)
            aggregated['all_angular_diffs'].append(all_per_sample_tensor)
        
        aggregated['angular_diff_mean'].append(np.mean(ang_means))
        aggregated['angular_diff_std'].append(np.mean(ang_stds))
        aggregated['cosine_sim_mean'].append(np.mean(cos_means))
        aggregated['wasserstein2_mean'].append(np.mean(w2_dists))
        aggregated['wasserstein2_std'].append(np.std(w2_dists))
    
    return aggregated


def plot_velocity_analysis(analysis, output_path, num_steps):
    """Plot velocity field analysis matching evaluate_with_analysis.py style."""
    if analysis is None or len(analysis['timesteps']) <= 1:
        print("Not enough data for analysis plot")
        return
    
    timesteps = analysis['timesteps']
    angular_mean = analysis['angular_diff_mean']
    angular_std = analysis['angular_diff_std']
    cosine_mean = analysis['cosine_sim_mean']
    vel_norm_mean = analysis['velocity_norm_mean']
    vel_norm_std = analysis['velocity_norm_std']
    w2_mean = analysis.get('wasserstein2_mean', [])
    w2_std = analysis.get('wasserstein2_std', [])
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f'Velocity Field Analysis (CFM, {num_steps} steps)', fontsize=14)

    # For angular difference and cosine similarity, use timesteps[1:-1] to match length
    ang_timesteps = timesteps[1:-1]

    # Plot 1: Angular difference vs timestep
    if len(angular_mean) > 0:
        ax1 = axes[0, 0]
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
        ax2.plot(ang_timesteps, cosine_mean, 'g-', linewidth=2)
        ax2.set_xlabel('Time t', fontsize=12)
        ax2.set_ylabel('Cosine Similarity', fontsize=12)
        ax2.set_title('Cosine Similarity Between Consecutive Velocities', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-1.1, 1.1])

    # Plot 3: Velocity norm vs timestep
    ax3 = axes[1, 0]
    vel_timesteps = timesteps[:-1]
    ax3.plot(vel_timesteps, vel_norm_mean, 'r-', linewidth=2, label='Mean')
    ax3.fill_between(vel_timesteps,
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
    
    # Plot 5: 2-Wasserstein distance vs timestep
    if len(w2_mean) > 0:
        ax5 = axes[2, 0]
        ax5.plot(ang_timesteps, w2_mean, 'm-', linewidth=2, label='Mean')
        ax5.fill_between(ang_timesteps,
                         [m - s for m, s in zip(w2_mean, w2_std)],
                         [m + s for m, s in zip(w2_mean, w2_std)],
                         alpha=0.3, color='magenta', label='± 1 Std')
        ax5.set_xlabel('Time t', fontsize=12)
        ax5.set_ylabel('2-Wasserstein Distance', fontsize=12)
        ax5.set_title('2-Wasserstein Distance Between Consecutive Velocities', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # Plot 6: Empty or statistics summary
    ax6 = axes[2, 1]
    ax6.axis('off')
    if len(w2_mean) > 0:
        summary_text = f"Summary Statistics:\n\n"
        summary_text += f"Angular Difference:\n"
        summary_text += f"  Mean: {np.mean(angular_mean):.2f}°\n"
        summary_text += f"  Std: {np.std(angular_mean):.2f}°\n\n"
        summary_text += f"2-Wasserstein Distance:\n"
        summary_text += f"  Mean: {np.mean(w2_mean):.4f}\n"
        summary_text += f"  Std: {np.std(w2_mean):.4f}\n\n"
        summary_text += f"Velocity Norm:\n"
        summary_text += f"  Mean: {np.mean(vel_norm_mean):.4f}\n"
        summary_text += f"  Std: {np.std(vel_norm_mean):.4f}"
        ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

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
                        'angular_diff_mean', 'angular_diff_std', 'cosine_sim_mean',
                        'wasserstein2_mean', 'wasserstein2_std'])
        
        timesteps = analysis['timesteps']
        w2_mean = analysis.get('wasserstein2_mean', [])
        w2_std = analysis.get('wasserstein2_std', [])
        
        # Write T rows (for velocity norms)
        for i in range(len(analysis['velocity_norm_mean'])):
            t = timesteps[i]
            row = [t, analysis['velocity_norm_mean'][i], analysis['velocity_norm_std'][i]]
            
            # Angular diff and W2 have T-1 values
            if i < len(analysis['angular_diff_mean']):
                row.extend([
                    analysis['angular_diff_mean'][i],
                    analysis['angular_diff_std'][i],
                    analysis['cosine_sim_mean'][i],
                    w2_mean[i] if i < len(w2_mean) else '',
                    w2_std[i] if i < len(w2_std) else '',
                ])
            else:
                row.extend(['', '', '', '', ''])
            
            writer.writerow(row)
    
    print(f"Saved analysis data to {output_path}")


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling requires at least one GPU"
    torch.set_grad_enabled(False)

    # Setup DDP (or single-GPU fallback)
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Build model
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
    if args.ckpt.endswith(".pt"):
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(args.input_dir, args.model,
                                 f"{args.model}_cifar10_weights_step_{args.step}.pt")
    if rank == 0:
        print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{device}")
    state_dict = checkpoint.get("ema_model", checkpoint)
    # Handle potential 'module.' prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Output folders
    ckpt_string_name = os.path.basename(ckpt_path).replace(".pt", "")
    folder_name = f"cfm-cifar10-unconditional-{ckpt_string_name}-steps-{args.integration_steps}-seed-{args.global_seed}"
    eval_dir = os.path.join(args.sample_dir, folder_name)
    img_folder = os.path.join(eval_dir, "images")
    if rank == 0:
        os.makedirs(img_folder, exist_ok=True)
        print(f"Saving .png samples at {eval_dir}")
        print(f"Integration steps: {args.integration_steps}")
        if args.analyze_velocity:
            print(f"Velocity field analysis enabled")
    if use_ddp:
        dist.barrier()

    # Work split
    n = args.per_proc_batch_size
    global_batch = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch) * global_batch)
    samples_this_rank = total_samples // world_size
    assert samples_this_rank % n == 0
    iters = samples_this_rank // n
    pbar = tqdm(range(iters)) if rank == 0 else range(iters)

    total = 0
    
    # Storage for velocity analysis
    all_analyses = []

    for _ in pbar:
        z = torch.randn(n, 3, 32, 32, device=device)

        with torch.no_grad():
            # Use manual Euler + trajectory version
            xs, vs = integrate_model_with_traj(
                model=model,
                x0=z,
                steps=args.integration_steps,
                device=device,
            )

            xT = xs[:, -1]  # Final result [B, 3, 32, 32]
            
            # Analyze velocity field if requested
            if args.analyze_velocity:
                analysis = analyze_trajectories(vs)
                all_analyses.append(analysis)

            # Output sample images
            imgs = (xT * 127.5 + 128).clamp(0, 255).permute(0, 2, 3, 1)
            imgs = imgs.to("cpu", dtype=torch.uint8).numpy()
            for i, img in enumerate(imgs):
                index = i * world_size + rank + total
                Image.fromarray(img).save(os.path.join(img_folder, f"{index:06d}.png"))

        total += global_batch

    if use_ddp:
        dist.barrier()
    
    # Aggregate and plot velocity analysis (only on rank 0)
    if rank == 0 and args.analyze_velocity and len(all_analyses) > 0:
        print(f"Aggregating velocity field analysis from {len(all_analyses)} batches...")
        aggregated = aggregate_analyses(all_analyses, args.integration_steps)
        
        if aggregated is not None:
            # Plot analysis
            plot_path = os.path.join(eval_dir, "velocity_analysis.png")
            plot_velocity_analysis(aggregated, plot_path, args.integration_steps)
            
            # Save CSV data
            csv_path = os.path.join(eval_dir, "velocity_analysis.csv")
            save_analysis_csv(aggregated, csv_path)
            
            # Print summary statistics
            if len(aggregated['angular_diff_mean']) > 0:
                print(f"\n=== Velocity Field Analysis ===")
                print(f"Angular Difference:")
                print(f"  Mean: {np.mean(aggregated['angular_diff_mean']):.2f}°")
                print(f"  Std:  {np.std(aggregated['angular_diff_mean']):.2f}°")
                print(f"  Min:  {np.min(aggregated['angular_diff_mean']):.2f}°")
                print(f"  Max:  {np.max(aggregated['angular_diff_mean']):.2f}°")
                print(f"Cosine Similarity:")
                print(f"  Mean: {np.mean(aggregated['cosine_sim_mean']):.4f}")
                print(f"Velocity Norm:")
                print(f"  Mean: {np.mean(aggregated['velocity_norm_mean']):.4f}")
                if 'wasserstein2_mean' in aggregated and len(aggregated['wasserstein2_mean']) > 0:
                    print(f"2-Wasserstein Distance:")
                    print(f"  Mean: {np.mean(aggregated['wasserstein2_mean']):.4f}")
                    print(f"  Std:  {np.std(aggregated['wasserstein2_mean']):.4f}")
                    print(f"  Min:  {np.min(aggregated['wasserstein2_mean']):.4f}")
                    print(f"  Max:  {np.max(aggregated['wasserstein2_mean']):.4f}")

    # Compute metrics (rank 0)
    if rank == 0 and args.compute_metrics:
        print("Computing evaluation metrics with torch-fidelity...")
        # Allow torch.load cached stats with PyTorch>=2.6 safe globals
        try:
            import torch.serialization as _ts
            import numpy as _np
            if hasattr(_ts, "add_safe_globals"):
                _ts.add_safe_globals([_np._core.multiarray._reconstruct])
        except Exception:
            pass

        metrics_args = {
            "input1": img_folder,
            "input2": "cifar10-train" if args.fid_ref == "train" else "cifar10-test",
            "cuda": True,
            "isc": True,
            "fid": True,
            "kid": False,
            "verbose": True,
        }
        metrics = torch_fidelity.calculate_metrics(**metrics_args)
        fid = metrics.get("frechet_inception_distance")
        is_mean = metrics.get("inception_score_mean")
        is_std = metrics.get("inception_score_std")

        print("\n===== Evaluation Results =====")
        if fid is not None:
            print(f"FID: {fid:.2f}")
        if is_mean is not None:
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

        results = {
            "fid": fid,
            "inception_score_mean": is_mean,
            "inception_score_std": is_std,
            "num_samples": total_samples,
            "integration_steps": args.integration_steps,
            "checkpoint": ckpt_path,
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {os.path.join(eval_dir, 'metrics.json')}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model/ckpt
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt). If empty, uses --input-dir/--model/--step.")
    parser.add_argument("--input-dir", type=str, default="./results", help="Base directory for checkpoints")
    parser.add_argument("--model", type=str, default="icfm", help="Model name for path composition")
    parser.add_argument("--step", type=int, default=400000, help="Training step for path composition")
    parser.add_argument("--num_channel", type=int, default=128, help="Base channel of UNet")

    # Sampling/integration
    parser.add_argument("--integration_steps", type=int, default=100, help="Number of integration steps")

    # DDP and generation
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--per-proc-batch-size", type=int, default=512)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--sample-dir", type=str, default="samples_torchfid")

    # Metrics
    parser.add_argument("--compute_metrics", action="store_true")
    parser.set_defaults(compute_metrics=True)
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"])
    
    # Velocity field analysis
    parser.add_argument("--analyze-velocity", action="store_true", help="Analyze velocity field during sampling")
    parser.set_defaults(analyze_velocity=True)

    args = parser.parse_args()
    main(args)