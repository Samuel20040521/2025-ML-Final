import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from models.model_configs import instantiate_model
import numpy as np
import os
from pathlib import Path


def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print("Loaded model weights")
    return model


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two velocity fields."""
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos_sim, cos_sim.mean().item()


def model_forward(model, x, t, device):
    """Match the original eval_loop.py forward pass (no CFG, unconditional)."""
    t_batch = torch.zeros(x.shape[0], device=device) + t
    with torch.cuda.amp.autocast(), torch.no_grad():
        # Unconditional: use empty extra dict, same as original when cfg_scale=0
        result = model(x, t_batch, extra={})
    return result.to(dtype=torch.float32)


def generate_samples_with_analysis(
    model,
    num_samples=16,
    image_size=32,
    device="cuda",
    step_size=0.01,
):
    """Generate samples using midpoint method (matching original ODESolver behavior)."""
    model.eval()
    
    num_steps = int(1.0 / step_size)
    print(f"Using {num_steps} ODE steps (step_size={step_size})")
    print(f"Total model evaluations: {num_steps * 2} (midpoint method)")
    
    # Start from random noise at t=0 (same as original)
    x = torch.randn(num_samples, 3, image_size, image_size, dtype=torch.float32, device=device)
    
    # Storage for analysis
    eval_times = []
    cosine_similarities = []
    cosine_similarities_per_sample = []
    velocities_norm = []
    eval_types = []
    
    prev_velocity = None
    eval_count = 0
    
    with torch.no_grad():
        for i in range(num_steps):
            t = i * step_size
            t_next = (i + 1) * step_size
            dt = step_size
            t_mid = t + dt / 2
            
            # === First evaluation: velocity at (x, t) ===
            v1 = model_forward(model, x, t, device)
            
            eval_times.append(t)
            vel_norm = v1.view(num_samples, -1).norm(dim=1).mean().item()
            velocities_norm.append(vel_norm)
            eval_types.append('start')
            
            if prev_velocity is not None:
                cos_per_sample, cos_mean = compute_cosine_similarity(prev_velocity, v1)
                cosine_similarities.append(cos_mean)
                cosine_similarities_per_sample.append(cos_per_sample.cpu())
            
            prev_velocity = v1.clone()
            eval_count += 1
            
            # === Midpoint: x_mid = x + (dt/2) * v1 ===
            x_mid = x + (dt / 2) * v1
            
            # === Second evaluation: velocity at (x_mid, t_mid) ===
            v2 = model_forward(model, x_mid, t_mid, device)
            
            eval_times.append(t_mid)
            vel_norm = v2.view(num_samples, -1).norm(dim=1).mean().item()
            velocities_norm.append(vel_norm)
            eval_types.append('midpoint')
            
            cos_per_sample, cos_mean = compute_cosine_similarity(prev_velocity, v2)
            cosine_similarities.append(cos_mean)
            cosine_similarities_per_sample.append(cos_per_sample.cpu())
            
            prev_velocity = v2.clone()
            eval_count += 1
            
            # === Full step: x_next = x + dt * v2 ===
            x = x + dt * v2
    
    print(f"Total evaluations recorded: {eval_count}")
    
    # Post-process samples (same as original eval_loop.py)
    # Scaling to [0, 1] from [-1, 1]
    samples = torch.clamp(x * 0.5 + 0.5, min=0.0, max=1.0)
    samples = torch.floor(samples * 255)
    samples = samples.to(torch.float32) / 255.0
    
    analysis = {
        "eval_times": eval_times,
        "cosine_similarities": cosine_similarities,
        "cosine_similarities_per_sample": torch.stack(cosine_similarities_per_sample),
        "velocities_norm": velocities_norm,
        "eval_types": eval_types,
        "num_evaluations": eval_count,
    }
    
    return samples, analysis


def run_multi_batch_analysis(
    model,
    total_samples=960,
    batch_size=32,
    image_size=32,
    device="cuda",
    step_size=0.01,
):
    """Run inference over multiple batches and aggregate cosine similarity analysis."""
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    num_steps = int(1.0 / step_size)
    num_evals = num_steps * 2 - 1  # Number of cosine similarity comparisons
    
    print(f"Running {num_batches} batches of {batch_size} samples each")
    print(f"Total samples: {num_batches * batch_size}")
    print(f"ODE steps: {num_steps}, Total evaluations per batch: {num_steps * 2}")
    
    # Aggregated storage across all batches
    # Shape: (num_batches, num_evals) for mean cosine sim per step
    all_batch_cos_sims = []
    # Shape: (num_batches, num_evals, batch_size) for per-sample cosine sim
    all_per_sample_cos_sims = []
    # Velocity norms
    all_vel_norms = []
    
    all_samples = []
    eval_times = None
    eval_types = None
    
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx + 1}/{num_batches}...")
        
        samples, analysis = generate_samples_with_analysis(
            model,
            num_samples=batch_size,
            image_size=image_size,
            device=device,
            step_size=step_size,
        )
        
        all_samples.append(samples.cpu())
        all_batch_cos_sims.append(analysis["cosine_similarities"])
        all_per_sample_cos_sims.append(analysis["cosine_similarities_per_sample"])
        all_vel_norms.append(analysis["velocities_norm"])
        
        # Store eval_times and eval_types from first batch (same for all)
        if eval_times is None:
            eval_times = analysis["eval_times"]
            eval_types = analysis["eval_types"]
    
    # Convert to tensors for analysis
    # all_batch_cos_sims: (num_batches, num_evals)
    all_batch_cos_sims = torch.tensor(all_batch_cos_sims)
    # all_per_sample_cos_sims: (num_batches, num_evals, batch_size)
    all_per_sample_cos_sims = torch.stack(all_per_sample_cos_sims)
    # all_vel_norms: (num_batches, num_steps * 2)
    all_vel_norms = torch.tensor(all_vel_norms)
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    aggregated_analysis = {
        "eval_times": eval_times,
        "eval_types": eval_types,
        # Per-step statistics across all batches
        "cos_sim_per_step_mean": all_batch_cos_sims.mean(dim=0).tolist(),  # (num_evals,)
        "cos_sim_per_step_std": all_batch_cos_sims.std(dim=0).tolist(),    # (num_evals,)
        # All per-sample cosine similarities: (num_batches * batch_size, num_evals)
        "all_per_sample_cos_sims": all_per_sample_cos_sims.permute(1, 0, 2).reshape(num_evals, -1),
        # Velocity norms
        "vel_norm_per_step_mean": all_vel_norms.mean(dim=0).tolist(),
        "vel_norm_per_step_std": all_vel_norms.std(dim=0).tolist(),
        # Metadata
        "num_batches": num_batches,
        "batch_size": batch_size,
        "total_samples": num_batches * batch_size,
        "num_steps": num_steps,
        "num_evaluations": num_steps * 2,
    }
    
    return all_samples, aggregated_analysis


def plot_multi_batch_analysis(analysis, output_dir):
    """Plot and save aggregated cosine similarity analysis from multiple batches."""
    
    eval_times = analysis["eval_times"][1:]  # Skip first (no previous velocity)
    cos_mean = analysis["cos_sim_per_step_mean"]
    cos_std = analysis["cos_sim_per_step_std"]
    vel_mean = analysis["vel_norm_per_step_mean"]
    vel_std = analysis["vel_norm_per_step_std"]
    all_cos_sims = analysis["all_per_sample_cos_sims"]  # (num_evals, total_samples)
    eval_types = analysis["eval_types"]
    
    num_batches = analysis["num_batches"]
    total_samples = analysis["total_samples"]
    
    # === Convert ALL cosine similarities to angles (in degrees) ===
    all_cos_flat = all_cos_sims.flatten().numpy()
    all_angles = np.degrees(np.arccos(np.clip(all_cos_flat, -1.0, 1.0)))
    
    # Per-step angle statistics
    all_angles_tensor = torch.from_numpy(
        np.degrees(np.arccos(np.clip(all_cos_sims.numpy(), -1.0, 1.0)))
    )
    angle_per_step_mean = all_angles_tensor.mean(dim=1).numpy()
    angle_per_step_std = all_angles_tensor.std(dim=1).numpy()
    
    # Calculate y-axis range
    angle_min = all_angles.min()
    angle_max = all_angles.max()
    angle_margin = (angle_max - angle_min) * 0.05
    angle_ylim = (max(0, angle_min - angle_margin), angle_max + angle_margin)
    
    # === Create figure with 3x2 subplots (ALL using angles) ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f"Velocity Field Analysis ({num_batches} batches, {total_samples} total samples)", fontsize=14)
    
    # Plot 1: Mean angular difference over evaluation index
    ax1 = axes[0, 0]
    x_axis = range(len(angle_per_step_mean))
    ax1.fill_between(x_axis, 
                     angle_per_step_mean - angle_per_step_std,
                     angle_per_step_mean + angle_per_step_std,
                     alpha=0.3, color='blue')
    ax1.plot(x_axis, angle_per_step_mean, 'b-', linewidth=1.5, label='Mean ± Std')
    ax1.set_xlabel('Evaluation Index')
    ax1.set_ylabel('Angular Difference (degrees)')
    ax1.set_title(f'Mean Angular Difference Across {num_batches} Batches')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(angle_ylim)
    ax1.legend()
    
    # Plot 2: Angular difference vs time
    ax2 = axes[0, 1]
    start_indices = [i for i, t in enumerate(eval_types[1:]) if t == 'start']
    mid_indices = [i for i, t in enumerate(eval_types[1:]) if t == 'midpoint']
    
    start_times = [eval_times[i] for i in start_indices if i < len(angle_per_step_mean)]
    start_angles = [angle_per_step_mean[i] for i in start_indices if i < len(angle_per_step_mean)]
    start_angle_std = [angle_per_step_std[i] for i in start_indices if i < len(angle_per_step_std)]
    
    mid_times = [eval_times[i] for i in mid_indices if i < len(angle_per_step_mean)]
    mid_angles = [angle_per_step_mean[i] for i in mid_indices if i < len(angle_per_step_mean)]
    mid_angle_std = [angle_per_step_std[i] for i in mid_indices if i < len(angle_per_step_std)]
    
    ax2.errorbar(start_times, start_angles, yerr=start_angle_std, fmt='o', color='blue', 
                 markersize=3, alpha=0.6, label='Start eval', capsize=2)
    ax2.errorbar(mid_times, mid_angles, yerr=mid_angle_std, fmt='o', color='red',
                 markersize=3, alpha=0.6, label='Midpoint eval', capsize=2)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Angular Difference (degrees)')
    ax2.set_title('Angular Difference vs Time (Mean ± Std)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(angle_ylim)
    
    # Plot 3: Angular difference over evaluation index (green version)
    ax3 = axes[1, 0]
    ax3.fill_between(x_axis,
                     angle_per_step_mean - angle_per_step_std,
                     angle_per_step_mean + angle_per_step_std,
                     alpha=0.3, color='green')
    ax3.plot(x_axis, angle_per_step_mean, 'g-', linewidth=1.5, label='Mean ± Std')
    ax3.set_xlabel('Evaluation Index')
    ax3.set_ylabel('Angular Difference (degrees)')
    ax3.set_title('Angular Difference Between Consecutive Velocities')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(angle_ylim)
    ax3.legend()
    
    # Plot 4: Angular difference vs time (green version)
    ax4 = axes[1, 1]
    ax4.errorbar(start_times, start_angles, yerr=start_angle_std, fmt='o', color='blue',
                 markersize=3, alpha=0.6, label='Start eval', capsize=2)
    ax4.errorbar(mid_times, mid_angles, yerr=mid_angle_std, fmt='o', color='red',
                 markersize=3, alpha=0.6, label='Midpoint eval', capsize=2)
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Angular Difference (degrees)')
    ax4.set_title('Angular Difference vs Time (Mean ± Std)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(angle_ylim)
    ax4.legend()
    
    # Plot 5: Velocity norm with std band
    ax5 = axes[2, 0]
    x_axis_vel = range(len(vel_mean))
    vel_min = min([m - s for m, s in zip(vel_mean, vel_std)])
    vel_max = max([m + s for m, s in zip(vel_mean, vel_std)])
    vel_margin = (vel_max - vel_min) * 0.05
    vel_ylim = (max(0, vel_min - vel_margin), vel_max + vel_margin)
    
    ax5.fill_between(x_axis_vel,
                     [m - s for m, s in zip(vel_mean, vel_std)],
                     [m + s for m, s in zip(vel_mean, vel_std)],
                     alpha=0.3, color='red')
    ax5.plot(x_axis_vel, vel_mean, 'r-', linewidth=1.5, label='Mean ± Std')
    ax5.set_xlabel('Evaluation Index')
    ax5.set_ylabel('Velocity L2 Norm')
    ax5.set_title(f'Velocity Magnitude Across {num_batches} Batches')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(vel_ylim)
    ax5.legend()
    
    # Plot 6: Histogram of angular differences
    ax6 = axes[2, 1]
    ax6.hist(all_angles, bins=100, edgecolor='black', alpha=0.7, color='green')
    ax6.set_xlabel('Angular Difference (degrees)')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Distribution of Angular Differences\n'
                  f'Mean: {all_angles.mean():.2f}°, Std: {all_angles.std():.2f}°')
    ax6.axvline(x=all_angles.mean(), color='r', linestyle='--', 
                label=f'Mean={all_angles.mean():.2f}°')
    ax6.set_xlim(angle_ylim)
    ax6.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "angular_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved analysis plot to {plot_path}")
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, "summary.txt")
    n = len(cos_mean)
    early_angles = all_angles_tensor[:n//3].flatten()
    mid_angles_phase = all_angles_tensor[n//3:2*n//3].flatten()
    late_angles = all_angles_tensor[2*n//3:].flatten()
    
    with open(summary_path, 'w') as f:
        f.write(f"=== Multi-Batch Velocity Field Analysis ===\n")
        f.write(f"Total batches: {num_batches}\n")
        f.write(f"Batch size: {analysis['batch_size']}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"ODE steps: {analysis['num_steps']}\n")
        f.write(f"Total evaluations per sample: {analysis['num_evaluations']}\n")
        f.write(f"\nCosine Similarity:\n")
        f.write(f"  Mean: {all_cos_flat.mean():.4f}\n")
        f.write(f"  Std:  {all_cos_flat.std():.4f}\n")
        f.write(f"  Min:  {all_cos_flat.min():.4f}\n")
        f.write(f"  Max:  {all_cos_flat.max():.4f}\n")
        f.write(f"\nAngular Difference:\n")
        f.write(f"  Mean: {all_angles.mean():.2f}°\n")
        f.write(f"  Std:  {all_angles.std():.2f}°\n")
        f.write(f"  Min: {all_angles.min():.2f}°\n")
        f.write(f"  Max: {all_angles.max():.2f}°\n")
        f.write(f"\nPhase-wise Angular Difference:\n")
        f.write(f"  Early (eval 0-{n//3}):    mean={early_angles.mean():.2f}°, std={early_angles.std():.2f}°\n")
        f.write(f"  Mid   (eval {n//3}-{2*n//3}):  mean={mid_angles_phase.mean():.2f}°, std={mid_angles_phase.std():.2f}°\n")
        f.write(f"  Late  (eval {2*n//3}-{n}): mean={late_angles.mean():.2f}°, std={late_angles.std():.2f}°\n")
    
    print(f"Saved summary to {summary_path}")
    
    # Print summary to console
    with open(summary_path, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser("CIFAR-10 Flow Matching Inference with Multi-Batch Cosine Similarity Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--total_samples", type=int, default=960, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for each inference run")
    parser.add_argument("--output", default="./inference_output", help="Output directory for all generated files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--save_all_samples", action="store_true", help="Save all generated samples (can be large)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Create output directory
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize model (same as training)
    model = instantiate_model(
        architechture="cifar10",
        is_discrete=False,
        use_ema=False,
    )
    
    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint)
    model.to(device)
    
    # Run multi-batch analysis
    print(f"Generating {args.total_samples} samples in batches of {args.batch_size}...")
    all_samples, analysis = run_multi_batch_analysis(
        model,
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        device=device,
        step_size=args.step_size,
    )
    
    # Save sample grid (first 64 or fewer)
    num_preview = min(64, all_samples.shape[0])
    nrow = int(num_preview ** 0.5)
    preview_path = os.path.join(output_dir, "samples_preview.png")
    save_image(all_samples[:num_preview], preview_path, nrow=nrow)
    print(f"Saved {num_preview} sample preview to {preview_path}")
    
    # Optionally save all samples
    if args.save_all_samples:
        all_samples_path = os.path.join(output_dir, "samples_all.png")
        nrow_all = int(all_samples.shape[0] ** 0.5)
        save_image(all_samples, all_samples_path, nrow=nrow_all)
        print(f"Saved all {all_samples.shape[0]} samples to {all_samples_path}")
    
    # Plot analysis
    plot_multi_batch_analysis(analysis, output_dir)
    
    print(f"\n=== All outputs saved to: {output_dir} ===")


if __name__ == "__main__":
    main()
