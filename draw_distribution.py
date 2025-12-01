import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_timestep_list(N=40, gamma=0.5):
    k = torch.arange(0, N+1)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]

if __name__ == "__main__":
    N = 40
    gamma = 0.5
    timesteps = generate_timestep_list(N=N, gamma=gamma)
    
    # Convert to numpy for plotting
    t_np = timesteps.numpy()
    s_np = np.linspace(0, 1, N+1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 1.5))

    # Plot 1: t vs s (The mapping function)
    axes[0].plot(s_np, t_np, 'b.-', label=f'gamma={gamma}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Uniform (gamma=1)')
    axes[0].set_xlabel('Uniform Progress s (k/N)')
    axes[0].set_ylabel('Timestep t')
    axes[0].set_title(f'Timestep Schedule (N={N}, gamma={gamma})')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Distribution of points on the timeline
    axes[1].scatter(t_np, np.zeros_like(t_np), alpha=0.6, marker='|', s=100)
    axes[1].set_yticks([])
    axes[1].set_xlabel('Timestep t')
    axes[1].set_title(f'Distribution of Timesteps on [0, 1], gamma={gamma}')
    axes[1].grid(True, axis='x')
    
    plt.tight_layout(pad=1.0)
    output_file = f'timestep_distribution_gamma_{gamma}.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

