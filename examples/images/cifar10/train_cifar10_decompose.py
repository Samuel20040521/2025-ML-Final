# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
# Authors: Kilian Fatras, Alexander Tong
# Modified: use UNetModel_Decompose_Wrapper and a decomposed loss (norm + angle + vector).

import copy
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from absl import app, flags
from torchvision import datasets, transforms
from tqdm import trange

from utils_cifar import ema, generate_samples, infiniteloop
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
# 這裡假設你的 UNetModel_Decompose_Wrapper 就放在 torchcfm.models.unet.unet 裡
from torchcfm.models.unet.unet import UNetModel_Decompose_Wrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results_decompose/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation / checkpoint
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

# Decomposition loss weights
flags.DEFINE_float("lambda_vec", 1.0, help="weight for vector reconstruction loss")
flags.DEFINE_float("lambda_r", 0.1, help="weight for magnitude loss")
flags.DEFINE_float("lambda_dir", 0.1, help="weight for direction (angular) loss")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def build_flow_matcher():
    sigma = 0.0
    if FLAGS.model == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )


def build_dataloader():
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    return infiniteloop(dataloader)


def build_model():
    # dim = (C, H, W)
    dim = (3, 32, 32)
    net_model = UNetModel_Decompose_Wrapper(
        dim=dim,
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    ema_model = copy.deepcopy(net_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to "
            "statistics computation in DataParallel. Single GPU is recommended if possible."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    return net_model, ema_model


def decompose_loss(v_pred, r_pred, d_pred, u_target):
    """
    v_pred: [B, C, H, W]  (model output vector field = r * d_pred)
    r_pred: [B, 1, H, W]        (pixel-wise magnitude)
    d_pred: [B, C, H, W]  (unit direction field)
    u_target: [B, C, H, W] (ground-truth flow from FM)
    """

    B, C, H, W = u_target.shape
    eps = 1e-8

    # 1. vector-level MSE (跟原本一樣的 flow matching loss)
    loss_vec = F.mse_loss(v_pred, u_target)

    # 2. magnitude target: 把每個 pixel 的向量長度取 norm，再對 (H, W) 做平均 -> 得到一個 per-sample scalar
    u_norm_per_pix = u_target.norm(dim=1)  # [B, H, W]
    
    r_pred_flat = r_pred.squeeze(1)  # [B, H, W]
    loss_r = F.mse_loss(r_pred_flat, u_norm_per_pix)
    # 3. direction target: 單位向量場
    u_unit = u_target / (u_norm_per_pix.unsqueeze(1) + eps)  # [B, C, H, W]

    # cosine similarity on each pixel, 再平均 (1 - cos)
    cos_sim = (d_pred * u_unit).sum(dim=1)  # [B, H, W]
    loss_dir = 1.0 - cos_sim.mean()
    
    
    return loss_vec, loss_r, loss_dir


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )
    print(
        "lambda_vec, lambda_r, lambda_dir:",
        FLAGS.lambda_vec,
        FLAGS.lambda_r,
        FLAGS.lambda_dir,
    )

    # DATA
    datalooper = build_dataloader()

    # MODELS
    net_model, ema_model = build_model()
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # FLOW MATCHER
    FM = build_flow_matcher()

    # OUTPUT DIR
    savedir = os.path.join(FLAGS.output_dir, FLAGS.model + "_decompose/")
    os.makedirs(savedir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(savedir, "training_log.txt")
    json_log_file = os.path.join(savedir, "training_stats.json")
    
    # Initialize log files
    training_stats = {
        "config": {
            "model": FLAGS.model,
            "lr": FLAGS.lr,
            "total_steps": FLAGS.total_steps,
            "batch_size": FLAGS.batch_size,
            "warmup": FLAGS.warmup,
            "ema_decay": FLAGS.ema_decay,
            "grad_clip": FLAGS.grad_clip,
            "lambda_vec": FLAGS.lambda_vec,
            "lambda_r": FLAGS.lambda_r,
            "lambda_dir": FLAGS.lambda_dir,
            "num_channel": FLAGS.num_channel,
            "save_step": FLAGS.save_step,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "training_history": []
    }
    
    with open(log_file, "w") as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Model: {FLAGS.model}\n")
        f.write(f"  Learning rate: {FLAGS.lr}\n")
        f.write(f"  Total steps: {FLAGS.total_steps}\n")
        f.write(f"  Batch size: {FLAGS.batch_size}\n")
        f.write(f"  Lambda weights - vec: {FLAGS.lambda_vec}, r: {FLAGS.lambda_r}, dir: {FLAGS.lambda_dir}\n")
        f.write(f"\n{'='*80}\n\n")

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()

            # Sample x1 ~ data, x0 ~ N(0, I)
            x1 = next(datalooper).to(device)      # [B, 3, 32, 32]
            x0 = torch.randn_like(x1)             # [B, 3, 32, 32]

            # Conditional flow matching: sample t, x_t, u_t
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)  # xt, ut: [B, C, H, W]

            # Model: predict decomposed flow
            v_pred, r_pred, d_pred = net_model(t, xt)  # v: [B,C,H,W], r:[B,1], d:[B,C,H,W]

            # Decomposition loss
            loss_vec, loss_r, loss_dir = decompose_loss(v_pred, r_pred, d_pred, ut)

            loss = (
                FLAGS.lambda_vec * loss_vec
                + FLAGS.lambda_r * loss_r
                + FLAGS.lambda_dir * loss_dir
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            pbar.set_postfix(
                dict(
                    loss=loss.item(),
                    loss_vec=loss_vec.item(),
                    loss_r=loss_r.item(),
                    loss_dir=loss_dir.item(),
                )
            )

            # Log every 100 steps
            if step % 100 == 0:
                stats = {
                    "step": step,
                    "loss": loss.item(),
                    "loss_vec": loss_vec.item(),
                    "loss_r": loss_r.item(),
                    "loss_dir": loss_dir.item(),
                    "lr": sched.get_last_lr()[0]
                }
                training_stats["training_history"].append(stats)
                
                # Append to text log
                with open(log_file, "a") as f:
                    f.write(f"Step {step:>6d} | Loss: {loss.item():.6f} | "
                           f"Vec: {loss_vec.item():.6f} | R: {loss_r.item():.6f} | "
                           f"Dir: {loss_dir.item():.6f} | LR: {stats['lr']:.6e}\n")
                
                # Save JSON every 1000 steps
                if step % 1000 == 0:
                    with open(json_log_file, "w") as f:
                        json.dump(training_stats, f, indent=2)

            # Sampling & checkpoint
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > 0:
                generate_samples(
                    net_model, FLAGS.parallel, savedir, step, net_="normal"
                )
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(
                        savedir, f"{FLAGS.model}_cifar10_decompose_weights_step_{step}.pt"
                    ),
                )
    
    # Final save of training stats
    training_stats["config"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(json_log_file, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    app.run(train)
