import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_analysis(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_comparison(uni_data, nonuni_data, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Uniform vs Non-Uniform Timestep Analysis', fontsize=16)

    # 1. Angular Difference
    ax1 = axes[0]
    
    # Uniform
    t_uni = uni_data['timesteps']
    ang_uni = uni_data['angular_diff_mean']
    # Note: angular difference is calculated between steps, so len(ang) = len(t) - 1 usually, 
    # or the script might have aligned them. Let's check the data length.
    # In the generation scripts:
    # timestep_list.append(current_t) -> this is the start of the interval
    # So ang[i] corresponds to interval starting at t[i]
    
    ax1.plot(t_uni, ang_uni, 'b-o', label='Uniform', markersize=4, alpha=0.7)
    
    # Non-Uniform
    t_non = nonuni_data['timesteps']
    ang_non = nonuni_data['angular_diff_mean']
    ax1.plot(t_non, ang_non, 'r-s', label='Non-Uniform', markersize=4, alpha=0.7)
    
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Angular Difference (degrees)')
    ax1.set_title('Angular Difference')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cosine Similarity
    ax2 = axes[1]
    cos_uni = uni_data['cosine_sim_mean']
    ax2.plot(t_uni, cos_uni, 'b-o', label='Uniform', markersize=4, alpha=0.7)
    
    cos_non = nonuni_data['cosine_sim_mean']
    ax2.plot(t_non, cos_non, 'r-s', label='Non-Uniform', markersize=4, alpha=0.7)
    
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # ax2.set_ylim([0.9, 1.0]) # Zoom in if needed

    # 3. Velocity Norm
    ax3 = axes[2]
    vel_uni = uni_data['velocity_norm_mean']
    # Velocity norm is calculated at the start of the interval in the script?
    # In script: vel_norm = v1...mean().item() where v1 is at current_t
    # And it is appended to velocity_norm_mean list.
    # But wait, in the script:
    # if prev_velocity is not None: ... velocity_norm_mean.append(vel_norm)
    # So it skips the first timestep (t=0) for the list?
    # Let's check the script logic again.
    # v1 = model(x, current_t)
    # ...
    # if prev_velocity is not None: ... velocity_norm_mean.append(vel_norm)
    # So yes, the lists in JSON correspond to t[1:] effectively?
    # Actually timestep_list.append(current_t) happens inside the if block too.
    # So all lists (timesteps, ang, cos, vel) should be aligned and have length N-1.
    
    ax3.plot(t_uni, vel_uni, 'b-o', label='Uniform', markersize=4, alpha=0.7)
    
    vel_non = nonuni_data['velocity_norm_mean']
    ax3.plot(t_non, vel_non, 'r-s', label='Non-Uniform', markersize=4, alpha=0.7)
    
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Velocity Norm')
    ax3.set_title('Velocity Norm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")

def main():
    uni_json = "uniform_timestep_samples/uniform_analysis.json"
    nonuni_json = "timestep_samples/nonuniform_analysis.json"
    
    if not os.path.exists(uni_json) or not os.path.exists(nonuni_json):
        print("Error: Analysis JSON files not found.")
        print(f"Checked: {uni_json} and {nonuni_json}")
        return

    uni_data = load_analysis(uni_json)
    nonuni_data = load_analysis(nonuni_json)
    
    plot_comparison(uni_data, nonuni_data, "uniform_vs_nonuniform_analysis.png")

if __name__ == "__main__":
    main()
