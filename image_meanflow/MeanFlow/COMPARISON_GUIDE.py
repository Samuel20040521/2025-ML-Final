"""
Quick comparison of evaluation scripts for MeanFlow models.

This file provides a side-by-side comparison of evaluation approaches
for CIFAR-10 vs ImageNet256.
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MeanFlow Evaluation Scripts Comparison                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌────────────────────────┬─────────────────────────┬──────────────────────────┐
│ Feature                │ CIFAR-10                │ ImageNet256              │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Script Location        │ /MeanFlow/              │ /image_meanflow/MeanFlow/│
│ Main Script            │ evaluate_fid_timesteps  │ evaluate_fid_timesteps   │
│                        │ .py                     │ .py                      │
│ Image Size             │ 32×32                   │ 256×256                  │
│ Color Space            │ RGB (pixel)             │ RGB (VAE latent)         │
│ Model Type             │ UNet                    │ SiT (Transformer)        │
│ Latent Diffusion       │ No                      │ Yes (VAE)                │
│ Classes                │ 10                      │ 1000                     │
│ CFG Support            │ No (unconditional)      │ Yes                      │
│ Default Batch Size     │ 500                     │ 16                       │
│ Default Samples        │ 50,000                  │ 50,000                   │
│ GPU Memory (batch=16)  │ ~3 GB                   │ ~10 GB                   │
│ Time per N (1 GPU)     │ ~2-3 min                │ ~15-20 min               │
│ DDP Support            │ No                      │ Yes (required)           │
└────────────────────────┴─────────────────────────┴──────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                              Usage Examples                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

CIFAR-10 Evaluation:
────────────────────────────────────────────────────────────────────────────────
python evaluate_fid_timesteps.py \\
    --ckpt unet_cifar10_meanflow_100k_ema.pt \\
    --output_dir fid_evaluation_results \\
    --batch_size 500 \\
    --num_fid_samples 50000 \\
    --fid_ref train \\
    --device cuda

Features:
  ✓ Single GPU execution
  ✓ Direct pixel-space generation
  ✓ Fast sampling (~2-3 min per N)
  ✓ Unconditional generation
  ✓ FID against CIFAR-10 train or test set

ImageNet256 Evaluation:
────────────────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \\
    --ckpt sit_b_2_meanflow_ema.pt \\
    --output-dir fid_timestep_evaluation \\
    --per-proc-batch-size 16 \\
    --num-fid-samples 50000 \\
    --cfg-scale 1.5 \\
    --fid-statistics-file fid_stats/adm_in256_stats.npz

Features:
  ✓ Multi-GPU distributed execution (DDP)
  ✓ VAE latent space diffusion
  ✓ Slower sampling (~15-20 min per N on 1 GPU)
  ✓ Class-conditional with CFG
  ✓ FID against ImageNet validation set

╔══════════════════════════════════════════════════════════════════════════════╗
║                         Timestep Schedule Settings                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Both scripts support:
  • Uniform timesteps:     Equal spacing from t=1 to t=0
  • Non-uniform timesteps: Gamma-based warping (more steps at boundaries)

Formula for non-uniform (γ=0.5 by default):
  s_k = k/N
  t_k = 0.5 + 0.5·sign(s-0.5)·|2s-1|^γ
  Then flip to go from 1→0

Default N values: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

╔══════════════════════════════════════════════════════════════════════════════╗
║                            Output Structure                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Both scripts generate:
  results.json              - FID scores for all N values
  fid_comparison.png        - Uniform vs Non-uniform comparison plot
  meanflow-steps-N-*/       - Per-N sample directories
    ├── images/             - Generated PNG files
    └── metrics.json        - torch-fidelity metrics

╔══════════════════════════════════════════════════════════════════════════════╗
║                         Performance Comparison                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

                        CIFAR-10        ImageNet256 (1 GPU)    ImageNet256 (4 GPUs)
                        ─────────       ───────────────────    ────────────────────
Batch Size              500             16                     64 (16×4)
Samples/sec             ~2000           ~60                    ~240
Time per N              2-3 min         15-20 min              4-5 min
Total (11 N values)     30-40 min       3-4 hours              45-60 min
Memory per GPU          ~3 GB           ~10 GB                 ~10 GB
Disk (no cleanup)       ~5 GB           ~150 GB                ~150 GB
Disk (with cleanup)     ~500 MB         ~5 GB                  ~5 GB

╔══════════════════════════════════════════════════════════════════════════════╗
║                          Key Differences                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Model Architecture
   CIFAR-10:     SongUNet (pixel-space U-Net)
   ImageNet256:  SiT (transformer in VAE latent space)

2. Execution Mode
   CIFAR-10:     Single process (python script.py)
   ImageNet256:  DDP required (torchrun --nproc_per_node=N script.py)

3. Generation Pipeline
   CIFAR-10:     z → model → x (direct)
   ImageNet256:  z → model → latent → VAE.decode → x

4. Conditioning
   CIFAR-10:     Unconditional
   ImageNet256:  Class-conditional with CFG scale

5. FID Reference
   CIFAR-10:     torch-fidelity builtin 'cifar10-train' or 'cifar10-test'
   ImageNet256:  External .npz file (ADM statistics)

6. Memory Scaling
   CIFAR-10:     Linear with batch size
   ImageNet256:  Linear with batch size + constant VAE overhead

╔══════════════════════════════════════════════════════════════════════════════╗
║                           Quick Start Commands                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

CIFAR-10 Quick Test (5K samples, 4 N values):
────────────────────────────────────────────────────────────────────────────────
cd /MeanFlow
python evaluate_fid_timesteps.py \\
    --ckpt unet_cifar10_meanflow_100k_ema.pt \\
    --num_fid_samples 5000 \\
    --batch_size 500

ImageNet256 Quick Test (5K samples, 4 N values):
────────────────────────────────────────────────────────────────────────────────
cd /image_meanflow/MeanFlow
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \\
    --ckpt sit_b_2_meanflow_ema.pt \\
    --num-fid-samples 5000 \\
    --per-proc-batch-size 16 \\
    --n-values "1,25,50,100" \\
    --cleanup-images

CIFAR-10 Full Evaluation (50K samples, 11 N values):
────────────────────────────────────────────────────────────────────────────────
cd /MeanFlow
python evaluate_fid_timesteps.py \\
    --ckpt unet_cifar10_meanflow_100k_ema.pt \\
    --num_fid_samples 50000 \\
    --batch_size 500

ImageNet256 Full Evaluation (50K samples, 11 N values, 4 GPUs):
────────────────────────────────────────────────────────────────────────────────
cd /image_meanflow/MeanFlow
torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \\
    --ckpt sit_b_2_meanflow_ema.pt \\
    --num-fid-samples 50000 \\
    --per-proc-batch-size 16 \\
    --cfg-scale 1.5

╔══════════════════════════════════════════════════════════════════════════════╗
║                          Common Issues & Solutions                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Issue: OOM on ImageNet256
  Solution: Reduce --per-proc-batch-size to 8 or 4

Issue: DDP fails to initialize
  Solution: Check NCCL, use single GPU, or set environment:
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500

Issue: Missing FID statistics for ImageNet256
  Solution: Download from:
    wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/\\
         ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \\
         -O fid_stats/adm_in256_stats.npz

Issue: Disk space full
  Solution: Add --cleanup-images flag to delete images after FID computation

Issue: Slow evaluation
  Solution: Use more GPUs or reduce --num-fid-samples for testing

╔══════════════════════════════════════════════════════════════════════════════╗
║                              File Locations                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

CIFAR-10:
  Script:     /MeanFlow/evaluate_fid_timesteps.py
  Checkpoint: /MeanFlow/unet_cifar10_meanflow_100k_ema.pt
  Model:      /MeanFlow/unet.py (SongUNet)
  Sampler:    /MeanFlow/meanflow_sampler.py

ImageNet256:
  Script:     /image_meanflow/MeanFlow/evaluate_fid_timesteps.py
  Checkpoint: /image_meanflow/MeanFlow/sit_b_2_meanflow_ema.pt
  Model:      /image_meanflow/MeanFlow/sit.py (SiT)
  Sampler:    /image_meanflow/MeanFlow/meanflow_sampler.py
  FID Stats:  /image_meanflow/MeanFlow/fid_stats/adm_in256_stats.npz
  VAE:        Downloaded from HuggingFace (stabilityai/sd-vae-ft-ema)

""")

if __name__ == "__main__":
    print("\nFor detailed documentation, see:")
    print("  - CIFAR-10: /MeanFlow/README.md (if exists)")
    print("  - ImageNet256: /image_meanflow/MeanFlow/README_FID_EVALUATION.md")
