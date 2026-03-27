# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import json
from collections import defaultdict

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

flags.DEFINE_string("model", "icfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 64, help="batch size")  # Lipman et al uses 128
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
    help="frequency of logging metrics per timestep",
)
flags.DEFINE_integer(
    "write_step",
    200,
    help="frequency of writing timestep metrics to txt file",
)
flags.DEFINE_integer(
    "num_timestep_bins",
    10,
    help="number of bins to aggregate timestep metrics",
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two velocity fields."""
    v1_flat = v1.view(v1.shape[0], -1)
    v2_flat = v2.view(v2.shape[0], -1)
    cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=1)
    return cos_sim.mean().item()


def plot_timestep_metrics(timestep_logs, savedir):
    """Plot loss and 1-cosine similarity vs timestep bins."""
    if not timestep_logs['timesteps']:
        return
    
    timesteps = timestep_logs['timesteps']
    losses = timestep_logs['losses']
    one_minus_cos = timestep_logs['one_minus_cos']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss per timestep
    ax1 = axes[0]
    ax1.plot(timesteps, losses, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Loss vs Timestep', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    
    # Plot 2: 1 - cosine similarity per timestep
    ax2 = axes[1]
    ax2.plot(timesteps, one_minus_cos, 'ro-', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xlabel('Timestep t', fontsize=12)
    ax2.set_ylabel('1 - Cosine Similarity', fontsize=12)
    ax2.set_title('1 - Cosine Similarity vs Timestep', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'timestep_metrics.png'), dpi=150, bbox_inches='tight')
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

    # Direct batch-level metric tracking
    # Store timesteps, losses, and cosine similarities from each batch
    batch_timesteps = []
    batch_losses = []
    batch_cos_sims = []
    
    # Global tracking
    global_step_log = []
    global_loss_log = []
    
    # Open txt file for logging timestep metrics
    timestep_log_file = open(os.path.join(savedir, 'timestep_metrics.txt'), 'w')
    timestep_log_file.write("# Timestep Metrics Log\n")
    timestep_log_file.write("# Format: step | avg_loss | avg_1-cos_sim\n")
    timestep_log_file.write("#" + "="*70 + "\n")
    timestep_log_file.flush()

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

            # Log metrics per timestep
            if step % FLAGS.log_step == 0:
                with torch.no_grad():
                    # Compute per-sample losses and cosine similarities
                    per_sample_loss = ((vt - ut) ** 2).view(vt.shape[0], -1).mean(dim=1)
                    
                    # Compute per-sample cosine similarity
                    vt_flat = vt.view(vt.shape[0], -1)
                    ut_flat = ut.view(ut.shape[0], -1)
                    per_sample_cos_sim = F.cosine_similarity(vt_flat, ut_flat, dim=1)
                    per_sample_one_minus_cos = 1.0 - per_sample_cos_sim
                    
                    # Store batch data directly
                    t_values = t.squeeze().cpu().numpy()
                    if t_values.ndim == 0:
                        t_values = np.array([t_values.item()])
                    
                    batch_timesteps.extend(t_values.tolist())
                    batch_losses.extend(per_sample_loss.cpu().numpy().tolist())
                    batch_cos_sims.extend(per_sample_one_minus_cos.cpu().numpy().tolist())
                    
                    # Global tracking
                    global_step_log.append(step)
                    global_loss_log.append(loss.item())
                    
                    # Display metrics
                    t_mean = t.mean().item()
                    avg_cos_sim = per_sample_cos_sim.mean().item()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        't': f'{t_mean:.3f}',
                        '1-cos': f'{1.0 - avg_cos_sim:.4f}'
                    })
            
            # Write timestep metrics to txt every write_step
            if step % FLAGS.write_step == 0 and step > 0:
                with torch.no_grad():
                    curr_cos_sim = compute_cosine_similarity(vt, ut)
                    curr_1_minus_cos = 1.0 - curr_cos_sim
                    curr_loss = loss.item()
                    
                    timestep_log_file.write(
                        f"Step {step}: loss={curr_loss:.6f}, 1-cos={curr_1_minus_cos:.6f}\n"
                    )
                    timestep_log_file.flush()

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                
                # Aggregate and plot timestep metrics
                if batch_timesteps:
                    # Create bins for visualization
                    num_bins = FLAGS.num_timestep_bins
                    timestep_bins = np.linspace(0, 1, num_bins + 1)
                    
                    # Bin the data for plotting
                    timestep_centers = []
                    avg_losses = []
                    avg_one_minus_cos = []
                    
                    for i in range(num_bins):
                        bin_mask = (np.array(batch_timesteps) >= timestep_bins[i]) & (np.array(batch_timesteps) < timestep_bins[i+1])
                        if i == num_bins - 1:  # Include right edge for last bin
                            bin_mask = (np.array(batch_timesteps) >= timestep_bins[i]) & (np.array(batch_timesteps) <= timestep_bins[i+1])
                        
                        if bin_mask.sum() > 0:
                            timestep_centers.append((timestep_bins[i] + timestep_bins[i+1]) / 2)
                            avg_losses.append(np.mean(np.array(batch_losses)[bin_mask]))
                            avg_one_minus_cos.append(np.mean(np.array(batch_cos_sims)[bin_mask]))
                    
                    if timestep_centers:
                        timestep_log = {
                            'timesteps': timestep_centers,
                            'losses': avg_losses,
                            'one_minus_cos': avg_one_minus_cos
                        }
                        
                        # Plot
                        plot_timestep_metrics(timestep_log, savedir)
                        
                        # Save detailed logs
                        np.savez(
                            os.path.join(savedir, 'training_logs.npz'),
                            global_steps=np.array(global_step_log),
                            global_losses=np.array(global_loss_log),
                            batch_timesteps=np.array(batch_timesteps),
                            batch_losses=np.array(batch_losses),
                            batch_cos_sims=np.array(batch_cos_sims),
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
    
    # Final save and close txt file
    if batch_timesteps:
        # Create bins for final visualization
        num_bins = FLAGS.num_timestep_bins
        timestep_bins = np.linspace(0, 1, num_bins + 1)
        
        timestep_centers = []
        avg_losses = []
        avg_one_minus_cos = []
        
        for i in range(num_bins):
            bin_mask = (np.array(batch_timesteps) >= timestep_bins[i]) & (np.array(batch_timesteps) < timestep_bins[i+1])
            if i == num_bins - 1:  # Include right edge for last bin
                bin_mask = (np.array(batch_timesteps) >= timestep_bins[i]) & (np.array(batch_timesteps) <= timestep_bins[i+1])
            
            if bin_mask.sum() > 0:
                timestep_centers.append((timestep_bins[i] + timestep_bins[i+1]) / 2)
                avg_losses.append(np.mean(np.array(batch_losses)[bin_mask]))
                avg_one_minus_cos.append(np.mean(np.array(batch_cos_sims)[bin_mask]))
        
        if timestep_centers:
            timestep_log = {
                'timesteps': timestep_centers,
                'losses': avg_losses,
                'one_minus_cos': avg_one_minus_cos
            }
            
            plot_timestep_metrics(timestep_log, savedir)
            
            np.savez(
                os.path.join(savedir, 'training_logs.npz'),
                global_steps=np.array(global_step_log),
                global_losses=np.array(global_loss_log),
                batch_timesteps=np.array(batch_timesteps),
                batch_losses=np.array(batch_losses),
                batch_cos_sims=np.array(batch_cos_sims),
            )
            
            print(f"\nTimestep metrics saved to {savedir}timestep_metrics.png and timestep_metrics.txt")
    
    timestep_log_file.close()


if __name__ == "__main__":
    app.run(train)
