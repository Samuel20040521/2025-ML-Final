import torch
import argparse
import numpy as np
import os
import sys
from torchvision.utils import save_image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flow_matching.examples.image.models.model_configs import instantiate_model


def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print("Loaded model weights")
    return model


def model_forward(model, x, t, device):
    """Forward pass through the model at time t."""
    t_batch = torch.zeros(x.shape[0], device=device) + t
    with torch.cuda.amp.autocast(), torch.no_grad():
        result = model(x, t_batch, extra={})
    return result.to(dtype=torch.float32)


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two velocity fields."""
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    cos_sim = torch.nn.functional.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos_sim


def generate_timestep_samples(
    model,
    timesteps,
    num_samples=16,
    image_size=32,
    device="cuda",
    seed=None,
):
    """
    Generate samples at different timesteps using UNIFORM timesteps.
    Flows through the ODE using the provided timestep intervals.
    
    Args:
        model: The flow matching model
        timesteps: List of timesteps to generate samples at (must be sorted, start with 0)
        num_samples: Number of samples to generate
        image_size: Size of the images
        device: Device to run on
        seed: Random seed for reproducibility
        
    Returns:
        samples_dict: Dictionary mapping timestep -> generated samples
        analysis_dict: Dictionary containing angular difference analysis
    """
    model.eval()
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    
    # Ensure timesteps start at 0 and are sorted
    timesteps = sorted(timesteps)
    if timesteps[0] != 0:
        timesteps = [0] + timesteps
    
    # Start from random noise at t=0
    x = torch.randn(num_samples, 3, image_size, image_size, dtype=torch.float32, device=device)
    
    samples_dict = {}
    
    # Storage for angular difference analysis
    timestep_list = []
    angular_diff_mean = []
    angular_diff_std = []
    cosine_sim_mean = []
    velocity_norm_mean = []
    
    print(f"Generating samples using {len(timesteps)} uniform timesteps...")
    print(f"Timesteps: {[f'{t:.3f}' for t in timesteps]}")
    
    prev_velocity = None
    
    with torch.no_grad():
        for i, current_t in enumerate(timesteps):
            print(f"  At timestep t={current_t:.3f}...")
            
            # Save current state (post-processed for visualization)
            samples_vis = torch.clamp(x * 0.5 + 0.5, min=0.0, max=1.0)
            samples_vis = torch.floor(samples_vis * 255)
            samples_vis = samples_vis.to(torch.float32) / 255.0
            samples_dict[current_t] = samples_vis.cpu()
            
            # If not the last timestep, flow to the next one
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
                dt = next_t - current_t
                t_mid = current_t + dt / 2
                
                # Midpoint method for this interval
                v1 = model_forward(model, x, current_t, device)
                
                # Compute velocity norm
                vel_norm = v1.view(num_samples, -1).norm(dim=1).mean().item()
                
                # Compute angular difference if we have a previous velocity
                
                x_mid = x + (dt / 2) * v1
                v2 = model_forward(model, x_mid, t_mid, device)
                if prev_velocity is not None:
                    cos_sim = compute_cosine_similarity(prev_velocity, v2)
                    # Convert to angles (in degrees)
                    angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / np.pi
                    
                    timestep_list.append(current_t)
                    angular_diff_mean.append(angles.mean().item())
                    angular_diff_std.append(angles.std().item())
                    cosine_sim_mean.append(cos_sim.mean().item())
                    velocity_norm_mean.append(vel_norm)
                
                prev_velocity = v2.clone()
                x = x + dt * v2
                
                print(f"    Flowed from t={current_t:.3f} to t={next_t:.3f} (dt={dt:.3f})")
    
    analysis_dict = {
        'timesteps': timestep_list,
        'angular_diff_mean': angular_diff_mean,
        'angular_diff_std': angular_diff_std,
        'cosine_sim_mean': cosine_sim_mean,
        'velocity_norm_mean': velocity_norm_mean,
    }
    
    return samples_dict, analysis_dict


import json

def save_analysis_json(analysis_dict, output_path):
    """Save analysis dictionary to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(analysis_dict, f, indent=4)
    print(f"Saved analysis data to {output_path}")

def plot_angular_difference(analysis_dict, output_path):
    """Plot angular difference between consecutive velocities."""
    timesteps = analysis_dict['timesteps']
    angular_mean = analysis_dict['angular_diff_mean']
    angular_std = analysis_dict['angular_diff_std']
    cosine_mean = analysis_dict['cosine_sim_mean']
    vel_norm = analysis_dict['velocity_norm_mean']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Velocity Field Analysis (Uniform Timesteps)', fontsize=14)
    
    # Plot 1: Angular difference vs timestep
    ax1 = axes[0, 0]
    ax1.plot(timesteps, angular_mean, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(timesteps,
                     [m - s for m, s in zip(angular_mean, angular_std)],
                     [m + s for m, s in zip(angular_mean, angular_std)],
                     alpha=0.3, color='blue', label='± 1 Std')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Angular Difference (degrees)', fontsize=12)
    ax1.set_title('Angular Difference Between Consecutive Velocities', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cosine similarity vs timestep
    ax2 = axes[0, 1]
    ax2.plot(timesteps, cosine_mean, 'g-', linewidth=2)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Cosine Similarity Between Consecutive Velocities', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1.1, 1.1])
    
    # Plot 3: Velocity norm vs timestep
    ax3 = axes[1, 0]
    ax3.plot(timesteps, vel_norm, 'r-', linewidth=2)
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel('Velocity L2 Norm', fontsize=12)
    ax3.set_title('Mean Velocity Magnitude', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Angular difference distribution
    ax4 = axes[1, 1]
    ax4.hist(angular_mean, bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax4.axvline(x=np.mean(angular_mean), color='r', linestyle='--', 
                linewidth=2, label=f'Mean={np.mean(angular_mean):.2f}°')
    ax4.set_xlabel('Angular Difference (degrees)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Angular Differences', fontsize=12)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved angular difference plot to {output_path}")
    
    # Print summary statistics
    print(f"\n=== Angular Difference Statistics ===")
    print(f"Mean angular difference: {np.mean(angular_mean):.2f}°")
    print(f"Std of angular difference: {np.std(angular_mean):.2f}°")
    print(f"Min angular difference: {np.min(angular_mean):.2f}°")
    print(f"Max angular difference: {np.max(angular_mean):.2f}°")
    print(f"Mean cosine similarity: {np.mean(cosine_mean):.4f}")


def create_timestep_grid(samples_dict, output_path, max_samples_per_row=8):
    """Create a grid showing samples at different timesteps."""
    timesteps = sorted(samples_dict.keys())
    num_samples = samples_dict[timesteps[0]].shape[0]
    samples_per_row = min(num_samples, max_samples_per_row)
    
    # Create a large grid: rows = timesteps, columns = samples
    all_images = []
    for t in timesteps:
        all_images.append(samples_dict[t][:samples_per_row])
    
    # Stack all images
    grid_tensor = torch.cat(all_images, dim=0)
    
    # Save the grid
    save_image(grid_tensor, output_path, nrow=samples_per_row)
    print(f"Saved timestep grid to {output_path}")


def create_timestep_comparison(samples_dict, output_path, sample_idx=0):
    """Create a figure showing one sample at different timesteps."""
    timesteps = sorted(samples_dict.keys())
    num_timesteps = len(timesteps)
    
    # Calculate grid dimensions
    ncols = min(10, num_timesteps)
    nrows = (num_timesteps + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Sample Evolution Across Timesteps (Sample {sample_idx})", fontsize=14)
    
    for idx, t in enumerate(timesteps):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Get the sample image
        img = samples_dict[t][sample_idx].permute(1, 2, 0).numpy()
        
        ax.imshow(img)
        ax.set_title(f"t={t:.2f}", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_timesteps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved timestep comparison to {output_path}")


def generate_timestep_list(num_steps=24):
    """Generate uniform timestep list from 0 to 1."""
    # Generate uniform timesteps from 0 to 1
    timesteps = np.linspace(0, 1, num_steps + 1)
    return timesteps.tolist()


def main():
    parser = argparse.ArgumentParser("Generate samples using uniform timesteps")
    parser.add_argument("--checkpoint", default="checkpoint-1799.pth", help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of uniform timesteps (default: 24)")
    parser.add_argument("--output_dir", default="./uniform_timestep_samples", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples_per_row", type=int, default=8, help="Max samples per row in grid")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Initialize model
    model = instantiate_model(
        architechture="cifar10",
        is_discrete=False,
        use_ema=False,
    )
    
    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint)
    model.to(device)
    
    # Generate uniform timestep list
    timesteps = generate_timestep_list(args.num_steps)
    print(f"Generating samples for {len(timesteps)} uniform timesteps (num_steps={args.num_steps}):")
    print(f"Step size: {1.0/args.num_steps:.4f}")
    print(f"Timesteps: {[f'{t:.3f}' for t in timesteps]}")
    
    # Generate samples at different timesteps
    samples_dict, analysis_dict = generate_timestep_samples(
        model,
        timesteps,
        num_samples=args.num_samples,
        device=device,
        seed=args.seed,
    )
    
    # Plot angular difference analysis
    angular_plot_path = os.path.join(args.output_dir, "angular_difference.png")
    plot_angular_difference(analysis_dict, angular_plot_path)
    
    # Save analysis data to JSON
    json_path = os.path.join(args.output_dir, "uniform_analysis.json")
    save_analysis_json(analysis_dict, json_path)
    
    # Save full grid
    grid_path = os.path.join(args.output_dir, "timestep_grid.png")
    create_timestep_grid(samples_dict, grid_path, args.max_samples_per_row)
    
    # Save comparison for first few samples
    for sample_idx in range(min(4, args.num_samples)):
        comparison_path = os.path.join(args.output_dir, f"timestep_comparison_sample_{sample_idx}.png")
        create_timestep_comparison(samples_dict, comparison_path, sample_idx)
    
    # Save individual timestep samples
    individual_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    
    for t in timesteps:
        t_path = os.path.join(individual_dir, f"timestep_{t:.3f}.png")
        save_image(samples_dict[t], t_path, nrow=int(np.sqrt(args.num_samples)))
    
    print(f"\nAll samples saved to {args.output_dir}")
    print(f"  - Angular difference plot: angular_difference.png")
    print(f"  - Full grid: timestep_grid.png")
    print(f"  - Comparisons: timestep_comparison_sample_*.png")
    print(f"  - Individual timesteps: individual/timestep_*.png")


if __name__ == "__main__":
    main()
