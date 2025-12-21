# import sys
# import time

# import torchvision.utils

# sys.dont_write_bytecode = True

# from argparse import ArgumentParser, Namespace
# from tqdm import tqdm
# from PIL import Image
# import os

# import torch
# from model import Model
# from model.generation import generate

# from lutils.configuration import Configuration
# from lutils.dict_wrapper import dict2namespace
# from lutils.logger import Logger
# from lutils.logging import to_image


# def parse_args() -> Namespace:
#     parser = ArgumentParser()

#     parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
#     parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
#     parser.add_argument("--output", type=str, required=True, help="Path to the output directory to write images to.")
#     parser.add_argument("--num_images", type=int, required=True, help="Number of images to generate.")
#     parser.add_argument("--nrows", type=int, default=8, help="nrows in make_grid.")
#     parser.add_argument("--method", type=str, required=True, help="Generation method of format solver-steps.")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

#     return parser.parse_args()


# def main(args: Namespace):
#     device = torch.device("cuda:0")

#     config = Configuration(args.config)

#     extended_run_name = "{}_run-{}".format(config["name"], "evaluation")
#     logger = Logger(
#         project="straighter_flow_matching",
#         run_name=extended_run_name,
#         use_wandb=False,
#         config=config,
#         rank=0)

#     # Setup model
#     logger.info("Building the model and loading from ckpt")
#     model = Model(config=dict2namespace(config))
#     model.load_from_ckpt(args.ckpt)
#     model.to(device)
#     if model.ae is not None:
#         model.ae.cpu()
#         del model.ae.encoder
#     model.eval()

#     # Generate images
#     logger.info("Generating images")
#     num_generated = 0
#     bar = tqdm(total=args.num_images)
#     os.makedirs(args.output, exist_ok=True)
#     samples = []
#     while num_generated < args.num_images:
#         num_samples = min(args.batch_size, args.num_images - num_generated)
#         generated_images, _ = generate(
#             model=model,
#             batch_size=num_samples,
#             device=device,
#             dtype=torch.float32,
#             source=config["evaluation"]["source"],
#             odesolver=args.method.split("-")[0],
#             num_steps=int(args.method.split("-")[1]),
#             first_step=None,
#             return_source=False,
#             decode=True,
#         )

#         samples.append(to_image(generated_images).cpu())
#         bar.update(num_samples)
#         num_generated += num_samples
#     bar.close()
#     samples = torch.cat(samples, dim=0)

#     # Saving images
#     samples_grid = torchvision.utils.make_grid(samples, nrow=args.nrows)
#     samples_image = Image.fromarray(samples_grid.permute(1, 2, 0).numpy())
#     output_samples_path = os.path.join(args.output, f"samples_{args.method}.png")
#     samples_image.save(output_samples_path)
#     logger.info(f"Samples saved at {output_samples_path}")

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

sys.dont_write_bytecode = True

# --- Project Imports ---
from model import Model
from lutils.configuration import Configuration
from lutils.dict_wrapper import dict2namespace
from lutils.logger import Logger

# ==========================================
# 1. Utility Functions for Analysis
# ==========================================

def model_forward(model, x, t, device):
    """
    Evaluates the velocity field v(x, t).
    """
    t_batch = torch.zeros(x.shape[0], device=device) + t
    with torch.no_grad():
        # FIX: Removed 'extra={}' which caused the TypeError
        result = model(x, t_batch) 
    return result.to(dtype=torch.float32)

def compute_cosine_similarity(v1, v2):
    """Returns the cosine similarity for each sample (Shape: [Batch])"""
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos_sim

def compute_velocity_norm(v):
    """Returns the L2 norm for each sample (Shape: [Batch])"""
    return v.view(v.shape[0], -1).norm(dim=1)

# ==========================================
# 2. ODE Solver Loop (Midpoint Method)
# ==========================================

def generate_samples_with_analysis(model, num_samples, image_shape, device, num_steps):
    """
    Performs sampling using the Midpoint method and records velocity data.
    """
    model.eval()
    
    # Initialize noise
    x = torch.randn(num_samples, *image_shape, dtype=torch.float32, device=device)
    
    dt = 1.0 / num_steps
    eval_times = []
    step_cos_sims = [] 
    step_vel_norms = []
    
    prev_velocity = None
    
    with torch.no_grad():
        for i in range(num_steps):
            t = i * dt
            t_mid = t + dt / 2
            
            # --- 1. Start Point (x, t) ---
            v1 = model_forward(model, x, t, device)
            
            # Record analysis data at the start of the step
            eval_times.append(t)
            step_vel_norms.append(compute_velocity_norm(v1))
            
            if prev_velocity is not None:
                cos = compute_cosine_similarity(prev_velocity, v1)
                step_cos_sims.append(cos)
            
            prev_velocity = v1.clone()
            
            # --- Midpoint Step ---
            x_mid = x + (dt / 2) * v1
            
            # --- 2. Mid Point (x_mid, t_mid) ---
            v2 = model_forward(model, x_mid, t_mid, device)
            
            # Full Step (using midpoint velocity)
            x = x + dt * v2

    # Record final point (t=1.0)
    t = 1.0 
    v_final = model_forward(model, x, t, device)
    eval_times.append(t)
    step_vel_norms.append(compute_velocity_norm(v_final))
    
    if prev_velocity is not None:
         cos = compute_cosine_similarity(prev_velocity, v_final)
         step_cos_sims.append(cos)

    # Stack results
    cos_sims_stack = torch.stack(step_cos_sims, dim=1)  
    vel_norms_stack = torch.stack(step_vel_norms, dim=1) 
    
    return cos_sims_stack, vel_norms_stack, eval_times

# ==========================================
# 3. Plotting Function
# ==========================================

def plot_results(results_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard distinct colors

    # --- Plot 1: Angular Difference ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (steps, data) in enumerate(results_dict.items()):
        angle_mean = data["angle_mean"]
        angle_std = data["angle_std"]
        # Use times corresponding to the intervals (drop the last time point)
        times_plot = np.array(data["times"])[1:]
        
        # Ensure lengths match
        min_len = min(len(times_plot), len(angle_mean))
        times_plot = times_plot[:min_len]
        angle_mean = angle_mean[:min_len]
        angle_std = angle_std[:min_len]
            
        color = colors[idx % len(colors)]
        label = f"Steps={steps}"
        
        ax.plot(times_plot, angle_mean, label=label, color=color, linewidth=2)
        ax.fill_between(times_plot, 
                        np.maximum(0, angle_mean - angle_std), 
                        angle_mean + angle_std, 
                        color=color, alpha=0.15)

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Angular Difference (Degrees)", fontsize=12)
    ax.set_title("Trajectory Straightness (Lower is Straighter)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 10)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, "angular_difference.png"), dpi=150)
    plt.close()

    # --- Plot 2: Velocity Magnitude ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (steps, data) in enumerate(results_dict.items()):
        vel_mean = data["vel_mean"]
        vel_std = data["vel_std"]
        times = np.array(data["times"])
        
        min_len = min(len(vel_mean), len(times))
        
        color = colors[idx % len(colors)]
        ax.plot(times[:min_len], vel_mean[:min_len], label=f"Steps={steps}", color=color, linewidth=2)
        ax.fill_between(times[:min_len], 
                        vel_mean[:min_len] - vel_std[:min_len], 
                        vel_mean[:min_len] + vel_std[:min_len], 
                        color=color, alpha=0.15)
        
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Velocity Norm", fontsize=12)
    ax.set_title("Velocity Magnitude vs Time", fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, "velocity_magnitude.png"), dpi=150)
    plt.close()

# ==========================================
# 4. Main Program
# ==========================================

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # Analysis specific args
    parser.add_argument("--total_samples", type=int, default=128, help="Total samples to analyze")
    parser.add_argument("--target_steps", type=str, default="20,50,100", help="Comma separated steps")
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Dummy args
    parser.add_argument("--num_images", type=int, default=0)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--nrows", type=int, default=8)

    return parser.parse_args()

def main(args: Namespace):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = Configuration(args.config)
    
    logger = Logger(
        project="straighter_flow_matching",
        run_name="analysis",
        use_wandb=False,
        config=config,
        rank=0)
    
    # --- Setup Model ---
    logger.info("Building model for analysis...")
    model = Model(config=dict2namespace(config))
    model.load_from_ckpt(args.ckpt)
    model.to(device)
    if model.ae is not None:
        model.ae.cpu()
        del model.ae.encoder
    model.eval()
    
    # --- Image Shape Detection ---
    if "data" in config and "image_size" in config["data"]:
        s = config["data"]["image_size"]
    else:
        s = 32
        
    if "model" in config and "input_channels" in config["model"]:
        c = config["model"]["input_channels"]
    else:
        c = 3
    
    image_shape = (c, s, s)
    logger.info(f"Using image shape: {image_shape}")

    # Parse target steps
    target_steps_list = [int(x) for x in args.target_steps.split(",")]
    final_results = {}
    
    # --- Analysis Loop ---
    for steps in target_steps_list:
        logger.info(f"Analyzing {steps} steps...")
        
        all_cos = []
        all_vel = []
        
        num_done = 0
        pbar = tqdm(total=args.total_samples)
        
        while num_done < args.total_samples:
            bs = min(args.batch_size, args.total_samples - num_done)
            
            cos, vel, times = generate_samples_with_analysis(
                model, bs, image_shape, device, steps
            )
            
            all_cos.append(cos.cpu())
            all_vel.append(vel.cpu())
            
            num_done += bs
            pbar.update(bs)
        
        pbar.close()
        
        # Aggregate
        global_cos = torch.cat(all_cos, dim=0)
        global_vel = torch.cat(all_vel, dim=0)
        
        # Calculate Stats
        global_angles = torch.rad2deg(torch.acos(torch.clamp(global_cos, -1.0, 1.0)))
        
        final_results[steps] = {
            "angle_mean": global_angles.mean(dim=0).numpy(),
            "angle_std": global_angles.std(dim=0).numpy(),
            "vel_mean": global_vel.mean(dim=0).numpy(),
            "vel_std": global_vel.std(dim=0).numpy(),
            "times": times
        }

    logger.info("Generating plots...")
    plot_results(final_results, args.output)
    logger.info(f"Done. Results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)