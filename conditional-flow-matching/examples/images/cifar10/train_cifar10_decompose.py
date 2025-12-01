# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
# Authors: Kilian Fatras, Alexander Tong
# Modified: Bottleneck Split Architecture with Hybrid Loss Strategy

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
# [重要] 確保這裡 import 到的是你剛剛修改過包含 BottleneckEnergyHead 的新模型
from torchcfm.models.unet.unet import UNetModel_Decompose_Wrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results_bottleneck_final/", help="output_directory")
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

# === Decomposition Loss Weights (Recommended Strategy) ===
# lambda_vec: 1.0   -> 總重建 Loss (裁判)，確保最終合成結果正確，並提供對能量強區域的注意力。
# lambda_energy: 0.1 -> 能量 Loss (營養師)，因為 scalar 很好學，設小一點避免搶走梯度。
# lambda_shape: 1.0 -> 形狀 Loss (教練)，專門糾正方向和紋理，解決模糊問題。
flags.DEFINE_float("lambda_vec", 1.0, help="weight for total reconstruction loss")
flags.DEFINE_float("lambda_energy", 0.1, help="weight for energy (scalar) loss")
flags.DEFINE_float("lambda_shape", 1.0, help="weight for shape (spatial) loss")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")


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


def decompose_loss_bottleneck(v_pred, energy_pred, shape_pred, u_target):
    """
    計算解耦後的 Loss。
    
    Args:
        v_pred:      [B, C, H, W]  (Combined output = shape * energy)
        energy_pred: [B, C]        (Predicted Scalars per channel)
        shape_pred:  [B, C, H, W]  (Predicted Spatial Map, normalized along H,W)
        u_target:    [B, C, H, W]  (Ground Truth Flow)
    """
    
    # 1. Vector-level MSE (The "Referee" Loss)
    # 確保最終合成結果與目標一致，這隱含了對高能量區域的關注。
    loss_vec = F.mse_loss(v_pred, u_target)

    # --- Prepare Ground Truth Decomposition ---
    
    # Target Energy: 計算 GT 在空間維度 (H, W) 上的總能量
    # [B, C, H, W] -> [B, C]
    target_energy = torch.norm(u_target, p=2, dim=(2, 3))
    
    # Target Shape: 計算 GT 的空間歸一化形狀
    # 加上 eps 避免全黑背景導致除以零
    eps = 1e-8
    # Broadcast energy: [B, C] -> [B, C, 1, 1]
    target_shape = u_target / (target_energy.view(*target_energy.shape, 1, 1) + eps)
    
    # --- Calculate Component Losses ---

    # 2. Energy Loss (Scalar MSE) (The "Volume" Loss)
    # 比較預測的總能量 vs 真實的總能量
    loss_energy = F.mse_loss(energy_pred, target_energy)

    # 3. Shape Loss (Spatial Cosine Similarity) (The "Structure" Loss)
    # 由於 shape_pred 和 target_shape 都是在 (H, W) 空間上的 Unit Vectors
    # Dot Product 就等於 Cosine Similarity (不需要再除 norm)
    # Sum over H, W dimensions -> [B, C]
    spatial_cosine = torch.sum(shape_pred * target_shape, dim=(2, 3))
    
    # 我們希望 Cosine 越大越好 (接近 1)，所以 Loss = 1 - Mean(Cosine)
    loss_shape = 1.0 - spatial_cosine.mean()
    
    return loss_vec, loss_energy, loss_shape


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )
    print(
        "Weights -> Vec: {}, Energy: {}, Shape: {}".format(
            FLAGS.lambda_vec, FLAGS.lambda_energy, FLAGS.lambda_shape
        )
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
    savedir = os.path.join(FLAGS.output_dir, FLAGS.model + "_final/")
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
            "lambda_energy": FLAGS.lambda_energy,
            "lambda_shape": FLAGS.lambda_shape,
            "num_channel": FLAGS.num_channel,
            "save_step": FLAGS.save_step,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "training_history": []
    }
    
    with open(log_file, "w") as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Model: {FLAGS.model} (Bottleneck Split)\n")
        f.write(f"  Lambda weights - vec: {FLAGS.lambda_vec}, energy: {FLAGS.lambda_energy}, shape: {FLAGS.lambda_shape}\n")
        f.write(f"\n{'='*80}\n\n")

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()

            # Sample x1 ~ data, x0 ~ N(0, I)
            x1 = next(datalooper).to(device)      # [B, 3, 32, 32]
            x0 = torch.randn_like(x1)             # [B, 3, 32, 32]

            # Conditional flow matching: sample t, x_t, u_t
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)  # xt, ut: [B, C, H, W]

            # Model Forward: predict decomposed components
            # Returns: v_pred (combined), energy_pred (scalar), shape_pred (map)
            v_pred, energy_pred, shape_pred = net_model(t, xt) 

            # Calculate decomposed losses
            loss_vec, loss_energy, loss_shape = decompose_loss_bottleneck(v_pred, energy_pred, shape_pred, ut)

            # Weighted Sum
            loss = (
                FLAGS.lambda_vec * loss_vec
                + FLAGS.lambda_energy * loss_energy
                + FLAGS.lambda_shape * loss_shape
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            pbar.set_postfix(
                dict(
                    loss=loss.item(),
                    l_vec=loss_vec.item(),
                    l_egy=loss_energy.item(),
                    l_shp=loss_shape.item(),
                )
            )

            # Log every 100 steps
            if step % 100 == 0:
                stats = {
                    "step": step,
                    "loss": loss.item(),
                    "loss_vec": loss_vec.item(),
                    "loss_energy": loss_energy.item(),
                    "loss_shape": loss_shape.item(),
                    "lr": sched.get_last_lr()[0]
                }
                training_stats["training_history"].append(stats)
                
                with open(log_file, "a") as f:
                    f.write(f"Step {step:>6d} | Loss: {loss.item():.6f} | "
                           f"Vec: {loss_vec.item():.5f} | Egy: {loss_energy.item():.5f} | "
                           f"Shp: {loss_shape.item():.5f} | LR: {stats['lr']:.2e}\n")
                
                # Save JSON less frequently
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
                        "args": FLAGS.flag_values_dict(), # Save args for easier resume
                    },
                    os.path.join(
                        savedir, f"{FLAGS.model}_cifar10_decompose_weights_step_{step}.pt"
                    ),
                )
    
    training_stats["config"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(json_log_file, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    app.run(train)