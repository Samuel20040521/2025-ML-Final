# ImageNet256 FID Evaluation with Different Timestep Schedules

This script evaluates FID (Fréchet Inception Distance) scores for the MeanFlow model on ImageNet 256×256 using different numbers of timesteps and comparing uniform vs non-uniform timestep schedules.

## Overview

The evaluation:
- Generates 50,000 samples (default) using the MeanFlow sampler
- Tests various timestep counts: N ∈ {1, 10, 20, 30, ..., 100}
- Compares two timestep scheduling strategies:
  - **Uniform**: Equal spacing from t=1 to t=0
  - **Non-uniform**: Gamma-based spacing with more steps near boundaries
- Computes FID against ImageNet256 reference statistics
- Generates comparison plots and detailed results

## Requirements

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install diffusers transformers
pip install torch-fidelity
pip install matplotlib pillow tqdm
```

### Pre-requisites
1. **Model Checkpoint**: Download or have a trained MeanFlow checkpoint
   - Expected file: `sit_b_2_meanflow_ema.pt`
   
2. **FID Reference Statistics**: ImageNet256 validation set statistics
   - Expected file: `fid_stats/adm_in256_stats.npz`
   - Download from: [ADM ImageNet-256 stats](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)

## Usage

### Quick Start (Interactive)
```bash
bash run_fid_evaluation.sh
```

This will prompt you with the configuration and ask for confirmation before starting.

### Direct Command (Single GPU)
```bash
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --output-dir fid_timestep_evaluation \
    --per-proc-batch-size 16 \
    --num-fid-samples 50000 \
    --cfg-scale 1.5 \
    --gamma 0.5 \
    --n-values "1,10,20,30,40,50,60,70,80,90,100" \
    --fid-statistics-file fid_stats/adm_in256_stats.npz
```

### Multi-GPU (Recommended for faster evaluation)
```bash
torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --output-dir fid_timestep_evaluation \
    --per-proc-batch-size 16 \
    --num-fid-samples 50000 \
    --cfg-scale 1.5
```

With 4 GPUs, total batch size = 4 × 16 = 64 samples per iteration.

### Quick Test (Fewer samples, faster)
```bash
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --output-dir fid_test \
    --per-proc-batch-size 16 \
    --num-fid-samples 5000 \
    --n-values "1,10,50,100" \
    --cfg-scale 1.5
```

## Command-Line Arguments

### Required Arguments
- `--ckpt`: Path to the MeanFlow model checkpoint file

### Model Configuration
- `--model`: Model architecture (default: `SiT-XL/2`)
  - Options: `SiT-XL/2`, `SiT-L/2`, `SiT-B/2`, etc.
- `--num-classes`: Number of ImageNet classes (default: 1000)
- `--resolution`: Image resolution (default: 256)

### Sampling Configuration
- `--per-proc-batch-size`: Batch size per GPU (default: 16)
  - Reduce if you encounter OOM errors
- `--num-fid-samples`: Total samples to generate (default: 50000)
  - Standard FID evaluation uses 50,000 samples
- `--cfg-scale`: Classifier-free guidance scale (default: 1.5)
  - Higher values → more class-conditional, potentially higher quality
  - Typical range: 1.0-4.0

### Timestep Configuration
- `--n-values`: Comma-separated timestep counts to evaluate
  - Default: `"1,10,20,30,40,50,60,70,80,90,100"`
  - Example: `"1,5,10,25,50,100"` for faster evaluation
- `--gamma`: Gamma parameter for non-uniform timesteps (default: 0.5)
  - Controls the distribution of timesteps
  - γ = 0.5 gives more steps near t=0 and t=1
  - γ = 1.0 gives uniform spacing

### Evaluation Settings
- `--fid-statistics-file`: Path to reference FID statistics
  - Default: `fid_stats/adm_in256_stats.npz`
- `--output-dir`: Directory for saving results (default: `fid_timestep_evaluation`)
- `--cleanup-images`: Delete generated images after FID computation
  - Saves disk space (each N value generates ~12GB of images)
- `--global-seed`: Random seed for reproducibility (default: 0)

## Output Structure

```
fid_timestep_evaluation/
├── results.json                          # FID scores for all N values
├── fid_comparison.png                    # Comparison plot
└── meanflow-steps-{N}-{mode}-cfg-{cfg}-seed-{seed}/
    ├── images/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...                          # 50,000 images (if not cleaned up)
    └── metrics.json                      # Detailed metrics for this N
```

### Results File (`results.json`)
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

## Non-Uniform Timestep Formula

The non-uniform timesteps are generated using:

```
s_k = k / N                                    (uniform in [0,1])
t_k = 0.5 + 0.5 · sign(s - 0.5) · |2s - 1|^γ  (warped)
```

Then reversed to go from t=1 to t=0.

- **γ < 1**: More timesteps near the boundaries (t=0 and t=1)
- **γ = 1**: Uniform spacing
- **γ > 1**: More timesteps in the middle

Default γ=0.5 concentrates steps near the start and end of the diffusion process.

## Performance Estimates

### Single GPU (A100 80GB)
- Batch size: 16
- Samples per N: 50,000
- Time per N: ~15-20 minutes
- Total time (11 N values): ~3-4 hours
- Disk space (with images): ~150 GB
- Disk space (with `--cleanup-images`): ~5 GB

### 4 GPUs (A100 80GB)
- Batch size: 16 × 4 = 64
- Time per N: ~4-5 minutes
- Total time (11 N values): ~45-60 minutes

### Memory Requirements
- Model: ~2 GB
- VAE: ~500 MB
- Per-sample memory: ~4 MB (latent) + ~12 MB (decoded)
- Total GPU memory: ~8-12 GB per GPU (batch_size=16)

If you encounter OOM:
- Reduce `--per-proc-batch-size` to 8 or 4
- Use fewer GPUs
- Close other GPU-consuming processes

## Tips and Best Practices

### For Quick Experiments
```bash
# Test with fewer samples and timesteps
torchrun --nproc_per_node=1 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --num-fid-samples 5000 \
    --n-values "1,25,50,100" \
    --cleanup-images
```

### For Publication-Quality Results
```bash
# Full evaluation with 50K samples
torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
    --ckpt sit_b_2_meanflow_ema.pt \
    --num-fid-samples 50000 \
    --n-values "1,10,20,30,40,50,60,70,80,90,100" \
    --cfg-scale 1.5
```

### Testing Different CFG Scales
```bash
# Compare different guidance scales
for cfg in 1.0 1.5 2.0 2.5; do
    torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
        --ckpt sit_b_2_meanflow_ema.pt \
        --cfg-scale $cfg \
        --output-dir fid_cfg_${cfg}
done
```

### Testing Different Gamma Values
```bash
# Compare different non-uniform schedules
for gamma in 0.3 0.5 0.7 1.0; do
    torchrun --nproc_per_node=4 evaluate_fid_timesteps.py \
        --ckpt sit_b_2_meanflow_ema.pt \
        --gamma $gamma \
        --output-dir fid_gamma_${gamma}
done
```

## Troubleshooting

### Issue: FID statistics file not found
```
Solution: Download the reference statistics:
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
    -O fid_stats/adm_in256_stats.npz
```

### Issue: CUDA Out of Memory
```
Solution 1: Reduce batch size
--per-proc-batch-size 8

Solution 2: Use fewer GPUs
torchrun --nproc_per_node=1 ...

Solution 3: Enable cleanup to free disk space between runs
--cleanup-images
```

### Issue: DDP initialization fails
```
Ensure NCCL environment is properly configured:
export MASTER_ADDR=localhost
export MASTER_PORT=29500

Or use single GPU:
python evaluate_fid_timesteps.py ... (without torchrun)
```

### Issue: Slow FID computation
```
FID computation is CPU-bound. To speed up:
1. Use more CPU cores (torch-fidelity uses multiprocessing)
2. Reduce --num-fid-samples for testing (though less accurate)
3. Use --cleanup-images to reduce I/O overhead
```

## Comparison with CIFAR-10 Evaluation

Similar script exists for CIFAR-10: `/MeanFlow/evaluate_fid_timesteps.py`

Key differences:
- **Image size**: 256×256 vs 32×32
- **VAE**: Uses latent diffusion vs pixel-space
- **Classes**: 1000 ImageNet classes vs 10 CIFAR classes
- **Memory**: ~10GB vs ~3GB per GPU
- **Speed**: ~4× slower due to larger images

## Citation

If you use this evaluation script, please cite the MeanFlow paper:

```bibtex
@article{meanflow2024,
  title={MeanFlow: Mean Teacher Flow Matching},
  author={Your Authors},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- [MeanFlow Paper](link-to-paper)
- [torch-fidelity](https://github.com/toshas/torch-fidelity)
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema)
- [ADM ImageNet Statistics](https://github.com/openai/guided-diffusion)
