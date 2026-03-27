import os
import json
import math
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.distributed as dist
import torch_fidelity
import numpy as np

from torchdiffeq import odeint
try:
    from torchdyn.core import NeuralODE
    HAS_TORCHDYN = True
except Exception:
    HAS_TORCHDYN = False

from torchcfm.models.unet.unet import UNetModelWrapper


def integrate_model(model, x0, method, steps, tol, device):
    """Integrate the UNet vector field from t=0 to t=1 and return final state."""
    if method == "euler":
        if not HAS_TORCHDYN:
            raise RuntimeError("torchdyn is required for Euler integration. Install torchdyn or use dopri5.")
        node = NeuralODE(model, solver="euler")
        t_span = torch.linspace(0, 1, steps + 1, device=device)
        traj = node.trajectory(x0, t_span=t_span)
        xT = traj[-1]
    else:
        # Use torchdiffeq (e.g., dopri5)
        t_span = torch.linspace(0, 1, 2, device=device)
        traj = odeint(model, x0, t_span, rtol=tol, atol=tol, method=method)
        xT = traj[-1]
    return xT


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
    # model = UNetModelWrapper(
    #     dim=(3, 32, 32),
    #     num_res_blocks=2,
    #     num_channels=args.num_channel,
    #     channel_mult=[1, 2, 2, 2],
    #     num_heads=4,
    #     num_head_channels=64,
    #     attention_resolutions="16",
    #     dropout=0.1,
    # ).to(device)

    # Load checkpoint
    if args.ckpt.endswith(".pt"):
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(args.input_dir, args.model,
                                 f"{args.model}_cifar10_weights_step_{args.step}.pt")
    # print(f"Checkpoint path: {ckpt_path}")
    # if rank == 0:
    #     print(f"Loading checkpoint: {ckpt_path}")
    # checkpoint = torch.load(ckpt_path, map_location=f"cuda:{device}")
    # state_dict = checkpoint.get("ema_model", checkpoint)
    # # Handle potential 'module.' prefix
    # if any(k.startswith("module.") for k in state_dict.keys()):
    #     state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict)
    # model.eval()

    # # Output folders
    ckpt_string_name = os.path.basename(ckpt_path).replace(".pt", "")
    folder_name = f"cfm-cifar10-unconditional-{ckpt_string_name}-steps-{args.integration_steps}-seed-{args.global_seed}"
    eval_dir = os.path.join(args.sample_dir, folder_name)
    img_folder = os.path.join(eval_dir, "images")
    # if rank == 0:
    #     os.makedirs(img_folder, exist_ok=True)
    #     print(f"Saving .png samples at {eval_dir}")
    #     print(f"Integration method: {args.integration_method}")
    #     if args.integration_method == "euler":
    #         print(f"Euler steps: {args.integration_steps}")
    #     else:
    #         print(f"ODE tol: {args.tol}")
    # if use_ddp:
    #     dist.barrier()

    # # Work split
    n = args.per_proc_batch_size
    global_batch = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch) * global_batch)
    # samples_this_rank = total_samples // world_size
    # assert samples_this_rank % n == 0
    # iters = samples_this_rank // n
    # pbar = tqdm(range(iters)) if rank == 0 else range(iters)

    # total = 0
    # for _ in pbar:
    #     z = torch.randn(n, 3, 32, 32, device=device)
    #     with torch.no_grad():
    #         xT = integrate_model(
    #             model=model,
    #             x0=z,
    #             method=args.integration_method,
    #             steps=args.integration_steps,
    #             tol=args.tol,
    #             device=device,
    #         )
    #         # Map from [-1,1] to [0,255]
    #         imgs = (xT * 127.5 + 128).clamp(0, 255).permute(0, 2, 3, 1)
    #         imgs = imgs.to("cpu", dtype=torch.uint8).numpy()
    #         for i, img in enumerate(imgs):
    #             index = i * world_size + rank + total
    #             Image.fromarray(img).save(os.path.join(img_folder, f"{index:06d}.png"))
    #     total += global_batch

    if use_ddp:
        dist.barrier()
    
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
            "integration_method": args.integration_method,
            "integration_steps": args.integration_steps,
            "tol": args.tol,
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
    parser.add_argument("--ckpt", type=str, default="results/otcfm/otcfm_cifar10_weights_step_400000.pt", help="Path to checkpoint (.pt). If empty, uses --input-dir/--model/--step.")
    parser.add_argument("--input-dir", type=str, default="./results", help="Base directory for checkpoints")
    parser.add_argument("--model", type=str, default="otcfm", help="Model name for path composition")
    parser.add_argument("--step", type=int, default=400000, help="Training step for path composition")
    parser.add_argument("--num_channel", type=int, default=128, help="Base channel of UNet")

    # Sampling/integration
    parser.add_argument("--integration_method", type=str, default="dopri5", help="dopri5 or euler")
    parser.add_argument("--integration_steps", type=int, default=100, help="Euler steps (only for euler)")
    parser.add_argument("--tol", type=float, default=1e-5, help="rtol=atol for dopri5")

    # DDP and generation
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--per-proc-batch-size", type=int, default=512)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--sample-dir", type=str, default="samples_torchfid")

    # Metrics
    parser.add_argument("--compute_metrics", action="store_true")
    ##default compute_metrics is true
    parser.set_defaults(compute_metrics=True)
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"]) 

    args = parser.parse_args()
    main(args)
