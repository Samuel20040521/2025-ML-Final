# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import math

import torch
import torch.nn.functional as F
from absl import app, flags
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop
import matplotlib.pyplot as plt
import numpy as np

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results_my/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 50001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer(
    "log_step",
    100,
    help="frequency of logging metrics (loss, Wasserstein distance)",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def compute_wasserstein2_distance(v1, v2):
    """
    Compute 2-Wasserstein distance between two velocity fields.
    
    Args:
        v1, v2: [B, C, H, W] velocity tensors
    Returns:
        w2_dist: scalar 2-Wasserstein distance
    """
    # Flatten to [B, D]
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    
    # Compute means
    mu1 = v1_flat.mean(dim=0)  # [D]
    mu2 = v2_flat.mean(dim=0)  # [D]
    
    # Compute standard deviations
    std1 = v1_flat.std(dim=0)  # [D]
    std2 = v2_flat.std(dim=0)  # [D]
    
    # Empirical W2 distance
    mean_diff = (mu1 - mu2).norm().item()
    std_diff = (std1 - std2).norm().item()
    
    w2_dist = math.sqrt(mean_diff**2 + std_diff**2)
    
    return w2_dist


def plot_training_metrics(steps, losses, w2_distances, savedir):
    """Plot training loss and Wasserstein distances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    ax1.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Wasserstein Distance
    ax2 = axes[1]
    ax2.plot(steps, w2_distances, 'm-', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('2-Wasserstein Distance', fontsize=12)
    ax2.set_title('2-Wasserstein Distance (predicted vs target velocity)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
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

    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    # Tracking for plotting
    train_steps = []
    train_losses = []
    train_w2_distances = []

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # Log metrics periodically
            if step % FLAGS.log_step == 0:
                with torch.no_grad():
                    w2_dist = compute_wasserstein2_distance(vt, ut)
                    train_steps.append(step)
                    train_losses.append(loss.item())
                    train_w2_distances.append(w2_dist)
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'W2': f'{w2_dist:.4f}'
                    })

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                
                # Plot and save metrics
                if len(train_steps) > 0:
                    plot_training_metrics(train_steps, train_losses, train_w2_distances, savedir)
                    
                    # Save metrics to numpy file
                    np.savez(
                        os.path.join(savedir, 'training_metrics.npz'),
                        steps=np.array(train_steps),
                        losses=np.array(train_losses),
                        w2_distances=np.array(train_w2_distances)
                    )
                
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )
    
    # Final plot
    if len(train_steps) > 0:
        plot_training_metrics(train_steps, train_losses, train_w2_distances, savedir)
        np.savez(
            os.path.join(savedir, 'training_metrics.npz'),
            steps=np.array(train_steps),
            losses=np.array(train_losses),
            w2_distances=np.array(train_w2_distances)
        )
        print(f"\nTraining metrics saved to {savedir}training_metrics.png and training_metrics.npz")


if __name__ == "__main__":
    app.run(train)
