# ImageNet256 FID Evaluation - Quick Start Guide

## What Was Created

I've created a complete evaluation pipeline for ImageNet256 MeanFlow models that matches what you did for CIFAR-10. Here's what's included:

### 1. Main Evaluation Script
**File:** `evaluate_fid_timesteps.py`

This script:
- ✅ Generates 50K samples using MeanFlow on ImageNet256
- ✅ Tests multiple timestep counts (N = 1, 10, 20, ..., 100)
- ✅ Compares uniform vs non-uniform timestep schedules
- ✅ Computes FID against ImageNet validation statistics
- ✅ Supports multi-GPU distributed training (DDP)
- ✅ Generates comparison plots automatically
- ✅ Saves detailed results in JSON format

### 2. Helper Scripts
**File:** `run_fid_evaluation.sh`
- Interactive launcher with confirmation prompts
- Shows configuration before running
- Handles common errors

**File:** `COMPARISON_GUIDE.py`
- Side-by-side comparison of CIFAR-10 vs ImageNet256
- Performance benchmarks
- Common issues and solutions

### 3. Documentation
**File:** `README_FID_EVALUATION.md`
- Complete usage guide
- All command-line arguments explained
- Performance estimates
- Troubleshooting section
- Example commands for different scenarios

## Quick Start

### Option 1: Interactive Mode
```bash
cd /home/dereklin1205/University/ML/2025-ML-Final/image_meanflow/MeanFlow
bash run_fid_evaluation.sh
```

### Option 2: Direct Command (Single GPU)
```bash
cd /home/dereklin1205/University/ML/2025-ML-Final/image_meanflow/MeanFlow

torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --output-dir fid_timestep_evaluation \
    --per-proc-batch-size 16 \
    --num-fid-samples 50000 \
    --cfg-scale 1.5 \
    --fid-statistics-file fid_stats/adm_in256_stats.npz
```

### Option 3: Multi-GPU (4 GPUs, Faster)
```bash
torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --per-proc-batch-size 16 \
    --num-fid-samples 50000 \
    --cfg-scale 1.5
```

### Option 4: Quick Test (Fast, 5K samples)
```bash
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --num-fid-samples 5000 \
    --n-values "1,25,50,100" \
    --cleanup-images
```

## Key Features

### Same as CIFAR-10 Evaluation
- ✅ Multiple timestep counts (N values)
- ✅ Uniform vs non-uniform comparison
- ✅ Gamma parameter for non-uniform scheduling (default: 0.5)
- ✅ Automatic FID computation
- ✅ JSON results file
- ✅ Comparison plots

### Additional for ImageNet256
- ✅ Multi-GPU distributed sampling (DDP)
- ✅ VAE latent space diffusion
- ✅ Class-conditional generation
- ✅ Classifier-free guidance (CFG) support
- ✅ Higher resolution (256×256 vs 32×32)

## Important Arguments

```bash
--ckpt                    # Path to checkpoint (REQUIRED)
--num-fid-samples 50000   # Total samples (standard: 50K)
--per-proc-batch-size 16  # Batch per GPU (reduce if OOM)
--cfg-scale 1.5           # Guidance scale (1.0-4.0)
--n-values "1,10,20,..."  # Which N to test
--gamma 0.5               # Non-uniform schedule param
--cleanup-images          # Save disk space
--fid-statistics-file     # Reference stats file
```

## Output

```
fid_timestep_evaluation/
├── results.json                    # FID for all N values
├── fid_comparison.png              # Plot: uniform vs non-uniform
└── meanflow-steps-{N}-{mode}-.../ 
    ├── images/                     # 50K generated images
    └── metrics.json                # Detailed metrics
```

## Expected Results Format

`results.json`:
```json
{
    "N": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "uniform_fid": [12.34, 8.56, 7.89, ...],
    "non_uniform_fid": [12.34, 8.23, 7.45, ...],
    "gamma": 0.5,
    "cfg_scale": 1.5,
    "num_samples": 50000
}
```

## Performance Estimates

| Configuration | Time per N | Total Time (11 N values) | GPU Memory |
|---------------|-----------|-------------------------|------------|
| 1 GPU (A100)  | 15-20 min | 3-4 hours              | ~10 GB     |
| 4 GPUs (A100) | 4-5 min   | 45-60 min              | ~10 GB ea. |

## Prerequisites

### 1. Download FID Statistics
```bash
mkdir -p fid_stats
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
    -O fid_stats/adm_in256_stats.npz
```

### 2. Check Checkpoint Exists
```bash
ls -lh sit_b_2_meanflow_ema.pt
```

### 3. Install Dependencies
```bash
pip install torch torchvision diffusers torch-fidelity matplotlib
```

## Troubleshooting

### OOM Error
```bash
# Reduce batch size
--per-proc-batch-size 8  # or 4
```

### DDP Fails
```bash
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

### Disk Space Issue
```bash
# Add cleanup flag
--cleanup-images
```

### Want Faster Testing
```bash
# Use fewer samples and N values
--num-fid-samples 5000 \
--n-values "1,25,50,100"
```

## Comparison with CIFAR-10

| Feature | CIFAR-10 | ImageNet256 |
|---------|----------|-------------|
| Image Size | 32×32 | 256×256 |
| Model | UNet (pixel) | SiT (latent) |
| VAE | No | Yes |
| DDP | No | Yes (required) |
| CFG | No | Yes |
| Classes | 10 | 1000 |
| Memory | ~3 GB | ~10 GB |
| Speed (1 GPU) | 2-3 min/N | 15-20 min/N |

## Testing Different Scenarios

### Different CFG Scales
```bash
for cfg in 1.0 1.5 2.0 2.5; do
    torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
        --ckpt sit_b_2_meanflow_ema.pt \
        --cfg-scale $cfg \
        --output-dir fid_cfg_${cfg}
done
```

### Different Gamma Values
```bash
for gamma in 0.3 0.5 0.7 1.0; do
    torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
        --ckpt sit_b_2_meanflow_ema.pt \
        --gamma $gamma \
        --output-dir fid_gamma_${gamma}
done
```

### Minimal Test Run
```bash
# Just 1 N value for debugging
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --n-values "50" \
    --num-fid-samples 1000
```

## View Comparison Guide

```bash
cd /home/dereklin1205/University/ML/2025-ML-Final/image_meanflow/MeanFlow
python COMPARISON_GUIDE.py
```

This shows a detailed side-by-side comparison of CIFAR-10 vs ImageNet256 evaluation.

## Files Created

```
image_meanflow/MeanFlow/
├── evaluate_fid_timesteps.py        # Main evaluation script
├── run_fid_evaluation.sh            # Interactive launcher
├── README_FID_EVALUATION.md         # Complete documentation
├── COMPARISON_GUIDE.py              # CIFAR-10 vs ImageNet comparison
└── GETTING_STARTED.md               # This file
```

## Next Steps

1. **Download FID statistics** (if not already present)
2. **Verify checkpoint exists** (`sit_b_2_meanflow_ema.pt`)
3. **Run quick test** with 5K samples to verify setup
4. **Run full evaluation** with 50K samples for final results
5. **Analyze results** in `results.json` and `fid_comparison.png`

## Get Help

- For detailed arguments: `python evaluate_fid_timesteps.py --help`
- For comparison with CIFAR-10: `python COMPARISON_GUIDE.py`
- For complete guide: Read `README_FID_EVALUATION.md`

---

**Ready to run?** Start with the quick test:

```bash
cd /home/dereklin1205/University/ML/2025-ML-Final/image_meanflow/MeanFlow
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --num-fid-samples 5000 \
    --n-values "1,25,50,100" \
    --cleanup-images
```

This should take ~30-45 minutes and verify everything works correctly!
