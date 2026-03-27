import os
import json
import math
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import numpy as np

import torch
import torch.distributed as dist
import torch_fidelity
from torchdiffeq import odeint

from torchcfm.models.unet.unet import UNetModelWrapper

def generate_timestep_list(N=40, gamma=0.5, device='cpu'):
    k = torch.arange(0, N+1, device=device)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]

def integrate_model_with_traj(model, x0, timesteps, device, method='euler'):
    """
    Integration with custom timesteps using torchdiffeq.
    timesteps: Tensor of shape [T+1]
    """
    def ode_func(t, x):
        # Expand t for batch
        t_input = t.view(1, 1, 1, 1).expand(x.size(0), 1, 1, 1)
        return model(t_input, x)

    # Perform integration
    traj = odeint(
        ode_func,
        x0,
        timesteps,
        method=method,
    )
    
    # traj shape is [T+1, B, C, H, W] -> [B, T+1, C, H, W]
    xs = traj.permute(1, 0, 2, 3, 4)
    return xs, None

def run_sampling_and_fid(args, model, device, num_steps, method='euler', use_uniform=True, rank=0, world_size=1):
    # Reset seed to ensure same latents across different runs (per rank)
    torch.manual_seed(args.global_seed + rank)
    
    mode_str = "uniform" if use_uniform else "non-uniform"
    if rank == 0:
        print(f"\nRunning evaluation for N={num_steps}, method={method}, mode={mode_str}")
    
    # Setup directories
    sample_dir = os.path.join(args.output_dir, f"samples_N{num_steps}_{method}_{mode_str}")
    img_folder = os.path.join(sample_dir, "images")
    
    if rank == 0:
        os.makedirs(img_folder, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Determine timesteps
    if use_uniform:
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
    else:
        timesteps = generate_timestep_list(N=num_steps, gamma=0.5, device=device)
    
    # Sampling workload distribution
    total_samples = args.num_fid_samples
    samples_per_gpu = int(math.ceil(total_samples / world_size))
    
    n = args.per_proc_batch_size
    iterations = int(math.ceil(samples_per_gpu / n))
    
    if rank == 0:
        print(f"Sampling {total_samples} images across {world_size} GPUs ({samples_per_gpu} per GPU)...")
    
    iterator = range(iterations)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"Sampling (N={num_steps}, {method}, {mode_str})", unit="batch")
        
    total_generated = 0
    for _ in iterator:
        current_batch_size = min(n, samples_per_gpu - total_generated)
        if current_batch_size <= 0:
            break
            
        z = torch.randn(current_batch_size, 3, 32, 32, device=device)
        
        with torch.no_grad():
            xs, _ = integrate_model_with_traj(
                model=model,
                x0=z,
                timesteps=timesteps,
                device=device,
                method=method
            )
            xT = xs[:, -1]
            
            # Post-process images
            imgs = (xT * 127.5 + 128).clamp(0, 255).permute(0, 2, 3, 1)
            imgs = imgs.to("cpu", dtype=torch.uint8).numpy()
            
            for i, img in enumerate(imgs):
                index = (rank * samples_per_gpu) + total_generated + i
                if index < total_samples:
                    Image.fromarray(img).save(f"{img_folder}/{index:06d}.png")
        
        total_generated += current_batch_size
        
    if world_size > 1:
        dist.barrier()
        
    # FID Calculation (Only on Rank 0)
    fid = 0.0
    if rank == 0:
        print("Computing FID...")
        # Allow torch.load cached stats with PyTorch>=2.6 safe globals
        try:
            import torch.serialization as _ts
            import numpy as _np
            if hasattr(_ts, "add_safe_globals"):
                safe_globals = [_np.ndarray, _np.dtype]
                
                # Add numpy dtypes which are often pickled
                try:
                    safe_globals.extend([
                        _np.dtypes.Float32DType,
                        _np.dtypes.Float64DType,
                        _np.dtypes.Int32DType,
                        _np.dtypes.Int64DType,
                        _np.dtypes.UInt8DType,
                    ])
                except AttributeError:
                    pass

                # Attempt to add numpy reconstruction function if accessible
                try:
                    safe_globals.append(_np._core.multiarray._reconstruct)
                except AttributeError:
                    try:
                        safe_globals.append(_np.core.multiarray._reconstruct)
                    except AttributeError:
                        pass
                _ts.add_safe_globals(safe_globals)
        except Exception as e:
            print(f"Warning: Could not add safe globals: {e}")

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
        print(f"FID ({mode_str}, {method}, N={num_steps}): {fid:.2f}")
        
        # Optional: Cleanup images to save space
        # shutil.rmtree(img_folder)
    
    if world_size > 1:
        fid_tensor = torch.tensor([fid if fid is not None else 0.0], device=device)
        dist.broadcast(fid_tensor, src=0)
        fid = fid_tensor.item()
        
    return fid

def main():
    parser = argparse.ArgumentParser()
    # Model/ckpt
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (.pt).")
    parser.add_argument("--input-dir", type=str, default="./results", help="Base directory for checkpoints")
    parser.add_argument("--model", type=str, default="icfm", help="Model name")
    parser.add_argument("--step", type=int, default=400000, help="Training step")
    parser.add_argument("--num_channel", type=int, default=128, help="Base channel of UNet")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="fid_comparison_results_nfe")
    
    # Sampling
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--per-proc-batch-size", type=int, default=100)
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"])
    
    # DDP
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    # Initialize DDP
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
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
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
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(args.input_dir, args.model,
                                 f"{args.model}_cifar10_weights_step_{args.step}.pt")
                                 
    if rank == 0:
        print(f"Loading checkpoint: {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("ema_model", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Solvers and NFEs to evaluate
    solvers = ['euler', 'rk4', 'midpoint']
    target_nfes = [20, 50, 80]
    
    results = {
        "nfe": target_nfes,
        "data": {} # solver -> { "uniform": [], "non_uniform": [] }
    }
    
    for solver in solvers:
        results["data"][solver] = {"uniform": [], "non_uniform": []}
        
        # Determine cost per step
        if solver == 'euler':
            cost = 1
        elif solver == 'midpoint':
            cost = 2
        elif solver == 'rk4':
            cost = 4
        else:
            cost = 1 # Default/Unknown
            
        for nfe in target_nfes:
            # Calculate steps
            steps = max(1, nfe // cost)
            
            # Uniform
            fid_uni = run_sampling_and_fid(args, model, device, steps, method=solver, use_uniform=True, rank=rank, world_size=world_size)
            if rank == 0:
                results["data"][solver]["uniform"].append(fid_uni)
            
            # Non-uniform
            fid_non_uni = run_sampling_and_fid(args, model, device, steps, method=solver, use_uniform=False, rank=rank, world_size=world_size)
            if rank == 0:
                results["data"][solver]["non_uniform"].append(fid_non_uni)
        
        # Save intermediate results (Rank 0 only)
        if rank == 0:
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            
    # Plotting (Rank 0 only)
    if rank == 0:
        plt.figure(figsize=(12, 8))
        
        markers = {'euler': 'o', 'rk4': 's', 'midpoint': '^'}
        colors = {'euler': 'b', 'rk4': 'r', 'midpoint': 'g'}
        
        for solver in solvers:
            # Plot Uniform
            plt.plot(results["nfe"], results["data"][solver]["uniform"], 
                     marker=markers.get(solver, 'o'), linestyle='-', color=colors.get(solver, 'k'), 
                     label=f'{solver} (Uniform)')
            
            # Plot Non-Uniform
            plt.plot(results["nfe"], results["data"][solver]["non_uniform"], 
                     marker=markers.get(solver, 'o'), linestyle='--', color=colors.get(solver, 'k'), 
                     label=f'{solver} (Non-Uniform)')
            
        plt.xlabel('Number of Function Evaluations (NFE)')
        plt.ylabel('FID')
        plt.title('FID vs NFE for Different Solvers')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "fid_comparison_nfe.png"))
        print(f"Plot saved to {os.path.join(args.output_dir, 'fid_comparison_nfe.png')}")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
