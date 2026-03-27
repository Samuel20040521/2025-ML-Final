import os
import math
import argparse
import csv
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from torchcfm.models.unet.unet import UNetModelWrapper


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("ema_model", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_cosine_similarity(v1, v2):
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    return F.cosine_similarity(v1_flat, v2_flat, dim=1)


@torch.no_grad()
def midpoint_with_analysis(model, batch_size, steps, device):
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    dt = 1.0 / steps
    t = 0.0

    timesteps = []
    angular_diff_mean = []
    angular_diff_std = []
    cosine_sim_mean = []
    velocity_norm_mean = []
    all_angular_diffs = []

    prev_velocity = None

    for _ in range(steps):
        t_tensor = torch.full((batch_size, 1, 1, 1), t, device=device)
        v1 = model(t_tensor, x)
        vel_norm = v1.view(batch_size, -1).norm(dim=1).mean().item()
        velocity_norm_mean.append(vel_norm)

        t_mid = t + 0.5 * dt
        t_mid_tensor = torch.full((batch_size, 1, 1, 1), t_mid, device=device)
        x_mid = x + 0.5 * dt * v1
        v2 = model(t_mid_tensor, x_mid)

        if prev_velocity is not None:
            cos_sim = compute_cosine_similarity(prev_velocity, v2)
            angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / math.pi
            timesteps.append(t)
            angular_diff_mean.append(angles.mean().item())
            angular_diff_std.append(angles.std().item())
            cosine_sim_mean.append(cos_sim.mean().item())
            all_angular_diffs.append(angles.cpu())

        prev_velocity = v2.clone()
        x = x + dt * v2
        t += dt

    analysis = {
        "timesteps": timesteps,
        "angular_diff_mean": angular_diff_mean,
        "angular_diff_std": angular_diff_std,
        "cosine_sim_mean": cosine_sim_mean,
        "velocity_norm_mean": velocity_norm_mean,
        "all_angular_diffs": all_angular_diffs,
    }
    return x, analysis


def aggregate(analyses, steps):
    if not analyses:
        return None

    # Velocity norms are recorded per step (length=steps), angular diffs per step-1
    agg = {
        "timesteps": None,
        "angular_diff_mean": [],
        "angular_diff_std": [],
        "cosine_sim_mean": [],
        "velocity_norm_mean": [],
        "velocity_norm_std": [],
        "all_angular_diffs": [],
    }

    # Timesteps from first analysis
    agg["timesteps"] = analyses[0]["timesteps"]

    # Velocity norms
    vel_stacks = [torch.tensor(a["velocity_norm_mean"]) for a in analyses]
    vel_stack = torch.stack(vel_stacks, dim=0)
    agg["velocity_norm_mean"] = vel_stack.mean(dim=0).tolist()
    agg["velocity_norm_std"] = vel_stack.std(dim=0).tolist()

    # Angular + cosine
    for i in range(steps - 1):
        ang_means = [a["angular_diff_mean"][i] for a in analyses]
        ang_stds = [a["angular_diff_std"][i] for a in analyses]
        cos_means = [a["cosine_sim_mean"][i] for a in analyses]
        per_sample = [a["all_angular_diffs"][i] for a in analyses]
        agg["angular_diff_mean"].append(float(np.mean(ang_means)))
        agg["angular_diff_std"].append(float(np.mean(ang_stds)))
        agg["cosine_sim_mean"].append(float(np.mean(cos_means)))
        agg["all_angular_diffs"].append(torch.cat(per_sample))

    return agg


def plot_analysis(analysis, output_png, steps):
    if analysis is None or len(analysis["timesteps"]) == 0:
        return

    t = analysis["timesteps"]
    ang_mean = analysis["angular_diff_mean"]
    ang_std = analysis["angular_diff_std"]
    cos_mean = analysis["cosine_sim_mean"]
    vel_mean = analysis["velocity_norm_mean"]
    vel_std = analysis["velocity_norm_std"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Velocity Field Analysis (midpoint, {steps} steps)")

    if len(ang_mean) > 0:
        ax1 = axes[0, 0]
        ax1.plot(t, ang_mean, "b-", linewidth=2, label="Mean")
        ax1.fill_between(t,
                         [m - s for m, s in zip(ang_mean, ang_std)],
                         [m + s for m, s in zip(ang_mean, ang_std)],
                         alpha=0.3, color="blue", label="± 1 Std")
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("Angular Difference (degrees)")
        ax1.set_title("Angular Difference Between Consecutive Velocities")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    if len(cos_mean) > 0:
        ax2 = axes[0, 1]
        ax2.plot(t, cos_mean, "g-", linewidth=2)
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("Cosine Similarity")
        ax2.set_title("Cosine Similarity Between Consecutive Velocities")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-1.1, 1.1])

    ax3 = axes[1, 0]
    vel_t = np.linspace(0, 1, steps)
    ax3.plot(vel_t, vel_mean, "r-", linewidth=2, label="Mean")
    ax3.fill_between(vel_t,
                     [m - s for m, s in zip(vel_mean, vel_std)],
                     [m + s for m, s in zip(vel_mean, vel_std)],
                     alpha=0.3, color="red", label="± 1 Std")
    ax3.set_xlabel("Time t")
    ax3.set_ylabel("Velocity L2 Norm")
    ax3.set_title("Mean Velocity Magnitude")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    if len(analysis["all_angular_diffs"]) > 0:
        ax4 = axes[1, 1]
        all_angles = torch.cat(analysis["all_angular_diffs"]).numpy()
        ax4.hist(all_angles, bins=60, edgecolor="black", alpha=0.7, color="blue")
        ax4.axvline(x=np.mean(all_angles), color="r", linestyle="--",
                    linewidth=2, label=f"Mean={np.mean(all_angles):.2f}°")
        ax4.set_xlabel("Angular Difference (degrees)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Distribution of Angular Differences")
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def save_csv(analysis, output_csv, steps):
    if analysis is None:
        return
    t = analysis["timesteps"]
    vel_t = np.linspace(0, 1, steps)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "velocity_norm_mean", "velocity_norm_std",
                    "angular_diff_mean", "angular_diff_std", "cosine_sim_mean"])
        for i in range(steps):
            vel_row = [vel_t[i], analysis["velocity_norm_mean"][i], analysis["velocity_norm_std"][i]]
            if i < len(t):
                vel_row.extend([
                    analysis["angular_diff_mean"][i],
                    analysis["angular_diff_std"][i],
                    analysis["cosine_sim_mean"][i],
                ])
            else:
                vel_row.extend(["", "", ""])
            w.writerow(vel_row)


def main():
    p = argparse.ArgumentParser("Plot angular difference and magnitude across timesteps (midpoint)")
    # Checkpoint
    p.add_argument("--ckpt", type=str, default="results/otcfm/otcfm_cifar10_weights_step_400000.pt", help="Path to .pt checkpoint. If empty, uses --input-dir/--model/--step")
    p.add_argument("--input-dir", type=str, default="./results")
    p.add_argument("--model", type=str, default="otcfm")
    p.add_argument("--step", type=int, default=400000)
    # Model
    p.add_argument("--num_channel", type=int, default=128)
    # Analysis
    p.add_argument("--integration_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--batches", type=int, default=4, help="Number of batches to aggregate")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="samples_torchfid/timestep_analysis")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Requires a CUDA device"
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

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

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(args.input_dir, args.model, f"{args.model}_cifar10_weights_step_{args.step}.pt")

    model = load_checkpoint(model, ckpt_path, device)

    os.makedirs(args.output_dir, exist_ok=True)

    analyses = []
    for _ in range(args.batches):
        _, analysis = midpoint_with_analysis(model, args.batch_size, args.integration_steps, device)
        analyses.append(analysis)

    agg = aggregate(analyses, args.integration_steps)
    png_path = os.path.join(args.output_dir, "velocity_analysis_midpoint.png")
    csv_path = os.path.join(args.output_dir, "velocity_analysis_midpoint.csv")
    plot_analysis(agg, png_path, args.integration_steps)
    save_csv(agg, csv_path, args.integration_steps)
    print(f"Saved plot to {png_path}\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
