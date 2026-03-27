# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
# Modified for UNetModel_Decompose_Wrapper (Bottleneck Split Architecture)

import sys
import os
import torch
from absl import flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

# [修改 1] 改用新的分解模型 Wrapper
from torchcfm.models.unet.unet import UNetModel_Decompose_Wrapper

FLAGS = flags.FLAGS

# UNet Configs
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# FID Configs
# [注意] 這裡預設路徑改成你訓練時設定的輸出資料夾，例如 ./results_decompose_bottleneck_final
flags.DEFINE_string("input_dir", "./results_decompose_bottleneck", help="directory containing checkpoints")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use (dopri5, euler, rk4)")
flags.DEFINE_integer("step", 400000, help="training step checkpoint to load")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate for FID")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 100, help="Batch size to compute FID (adjust based on GPU VRAM)")

# Parse flags
FLAGS(sys.argv)

# Define device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print(f"Initializing Decompose Model on {device}...")

# [修改 2] 定義模型為 UNetModel_Decompose_Wrapper
new_net = UNetModel_Decompose_Wrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)

# [修改 3] 建構正確的 Checkpoint 路徑
# 訓練腳本存檔格式為: {model}_cifar10_decompose_weights_step_{step}.pt
# 請確保這裡的檔名格式與你訓練時的 save_image/torch.save 一致
ckpt_filename = f"{FLAGS.model}_cifar10_decompose_weights_step_{FLAGS.step}.pt"

# 如果你之前的訓練腳本沒有加 _decompose，請改回 f"{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
# 為了保險，這裡做個檢查：
PATH = os.path.join(FLAGS.input_dir, f"{FLAGS.model}_decompose", ckpt_filename)
if not os.path.exists(PATH):
    # 嘗試另一種路徑結構 (視你的 output_dir 設定而定)
    PATH = os.path.join(FLAGS.input_dir, ckpt_filename)

print(f"Loading checkpoint from: {PATH}")

checkpoint = torch.load(PATH, map_location=device)

# Load State Dict (Handle DataParallel if necessary)
if "ema_model" in checkpoint:
    state_dict = checkpoint["ema_model"]
    print("Loaded EMA weights.")
else:
    state_dict = checkpoint["net_model"]
    print("Loaded standard weights (no EMA found).")

try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    # Handle 'module.' prefix from DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    new_net.load_state_dict(new_state_dict)

# [關鍵] 設定為 eval 模式
# 在 UNetModel_Decompose 中，eval() 模式下 forward 只會回傳 v (向量場)
# 這樣 ODE Solver 才能正常運作
new_net.eval()


# Define the integration method wrapper
# Torchdiffeq 需要的格式是 forward(t, x)，而我們的模型是 forward(t, x) -> 完美匹配
if FLAGS.integration_method == "euler":
    # NeuralODE wrapper (optional, mainly for fixed step solvers)
    node = NeuralODE(new_net, solver=FLAGS.integration_method)

def gen_1_img(unused_latent):
    """
    Generator function for cleanfid.
    Generates a batch of images.
    """
    with torch.no_grad():
        # 1. Sample Noise x0
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        
        # 2. Integrate ODE
        if FLAGS.integration_method == "euler":
            # Fixed step size using TorchDyn
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
            traj = traj[-1, :] 
        else:
            # Adaptive step size using TorchDiffEq (dopri5)
            # t_span 只需起點和終點
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                new_net,
                x,
                t_span,
                rtol=FLAGS.tol,
                atol=FLAGS.tol,
                method=FLAGS.integration_method,
            )
            traj = traj[-1, :]

    # 3. Post-process to [0, 255] uint8
    # Flow Matching output is usually roughly [-1, 1], mapping to [0, 255]
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img


print(f"Start computing FID on {FLAGS.num_gen} samples...")
print(f"Batch size: {FLAGS.batch_size_fid}")

score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train", # Compare against training set (standard for CIFAR-10)
    mode="legacy_tensorflow",
)

print("\n" + "="*50)
print(f"FID Score: {score}")
print("="*50)