import os
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import shutil

import torch
import torch_fidelity

from unet import SongUNet
from meanflow_sampler import meanflow_sampler

def generate_timestep_list(N=40, gamma=0.5):
    k = torch.arange(0, N+1)
    s = k / float(N)  # [0,1] uniform
    t = 0.5 + 0.5 * torch.sign(s - 0.5) * torch.abs(2*s - 1)**gamma
    return t  # shape [N+1]

def run_sampling_and_fid(args, model, device, num_steps, seed, use_uniform=True, delete_images=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mode_str = "uniform" if use_uniform else "non-uniform"
    print(f"\nRunning evaluation for N={num_steps}, mode={mode_str}, seed={seed}")

    sample_dir = os.path.join(args.output_dir, f"samples_N{num_steps}_{mode_str}_seed{seed}")
    img_folder = os.path.join(sample_dir, "images")
    os.makedirs(img_folder, exist_ok=True)

    if use_uniform:
        timesteps = None
    else:
        timesteps = generate_timestep_list(N=num_steps, gamma=0.5).to(device)

    n = args.batch_size
    total_samples = args.num_fid_samples
    iterations = int(math.ceil(total_samples / n))

    print(f"Sampling {total_samples} images...")
    total = 0
    for _ in tqdm(range(iterations)):
        current_batch_size = min(n, total_samples - total)
        if current_batch_size <= 0:
            break

        z = torch.randn(current_batch_size, 3, 32, 32, device=device)

        with torch.no_grad():
            samples = meanflow_sampler(
                model=model,
                latents=z,
                cfg_scale=1.0,
                num_steps=num_steps,
                timesteps=timesteps
            )

            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                index = total + i
                Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")

        total += current_batch_size

    print("Computing FID...")
    metrics_args = {
        "input1": img_folder,
        "input2": "cifar10-train" if args.fid_ref == "train" else "cifar10-test",
        "cuda": True,
        "isc": False,
        "fid": True,
        "kid": False,
        "verbose": False,
    }

    metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)
    fid = metrics_dict.get("frechet_inception_distance", None)
    print(f"FID ({mode_str}, N={num_steps}, seed={seed}): {fid:.2f}")

    if delete_images:
        shutil.rmtree(sample_dir, ignore_errors=True)

    return fid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="fid_evaluation_results")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    parser.add_argument("--fid_ref", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_start", type=int, default=20, help="Starting N value (inclusive)")
    parser.add_argument("--n_end", type=int, default=100, help="Ending N value (inclusive)")
    parser.add_argument("--n_step", type=int, default=20, help="Step size for N sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="Seeds to repeat for each N")
    parser.add_argument("--keep_images", action="store_true", help="Keep generated images instead of deleting them")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    
    # Load Model
    print("Loading model...")
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,
    ).to(device)
    
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    N_values = list(range(args.n_start, args.n_end + 1, args.n_step))
    delete_images = not args.keep_images

    results = {
        "config": {
            "ckpt": args.ckpt,
            "fid_ref": args.fid_ref,
            "batch_size": args.batch_size,
            "num_fid_samples": args.num_fid_samples,
            "N_values": N_values,
            "seeds": args.seeds,
            "delete_images": delete_images,
        },
        "per_seed": {},
        "summary": [],
    }

    for N in N_values:
        results["per_seed"][str(N)] = {"uniform": {}, "non_uniform": {}}

        uniform_fids = []
        non_uniform_fids = []

        for seed in args.seeds:
            fid_uni = run_sampling_and_fid(
                args=args,
                model=model,
                device=device,
                num_steps=N,
                seed=seed,
                use_uniform=True,
                delete_images=delete_images,
            )
            results["per_seed"][str(N)]["uniform"][str(seed)] = fid_uni
            uniform_fids.append(fid_uni)

            fid_non_uni = run_sampling_and_fid(
                args=args,
                model=model,
                device=device,
                num_steps=N,
                seed=seed,
                use_uniform=False,
                delete_images=delete_images,
            )
            results["per_seed"][str(N)]["non_uniform"][str(seed)] = fid_non_uni
            non_uniform_fids.append(fid_non_uni)

            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)

        uniform_mean = float(np.mean(uniform_fids))
        uniform_std = float(np.std(uniform_fids))
        non_uniform_mean = float(np.mean(non_uniform_fids))
        non_uniform_std = float(np.std(non_uniform_fids))

        results["summary"].append(
            {
                "N": N,
                "mode": "uniform",
                "mean_fid": uniform_mean,
                "std_fid": uniform_std,
                "seeds": args.seeds,
            }
        )
        results["summary"].append(
            {
                "N": N,
                "mode": "non_uniform",
                "mean_fid": non_uniform_mean,
                "std_fid": non_uniform_std,
                "seeds": args.seeds,
            }
        )

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

    plt.figure(figsize=(10, 6))
    uniform_curve = [item["mean_fid"] for item in results["summary"] if item["mode"] == "uniform"]
    non_uniform_curve = [item["mean_fid"] for item in results["summary"] if item["mode"] == "non_uniform"]

    plt.plot(N_values, uniform_curve, "o-", label="Uniform Timesteps (mean)")
    plt.plot(N_values, non_uniform_curve, "s-", label="Non-Uniform Timesteps (mean)")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("FID")
    plt.title("FID vs Number of Steps (mean over seeds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "fid_comparison.png"))
    print(f"Plot saved to {os.path.join(args.output_dir, 'fid_comparison.png')}")

if __name__ == "__main__":
    main()
