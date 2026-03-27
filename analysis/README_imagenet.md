# ImageNet Flow Matching Analysis

This script performs velocity field analysis on ImageNet32 and ImageNet64 flow matching models.

## Setup

### 1. Download Checkpoint

Download a pre-trained ImageNet checkpoint:

```bash
# For ImageNet32
bash download_imagenet_ckpt.sh 32

# For ImageNet64
bash download_imagenet_ckpt.sh 64
```

Or download manually from:
- **ImageNet32**: https://huggingface.co/cvg-unibe/loom-cfm_imagenet32/blob/main/model.pth
- **ImageNet64**: https://huggingface.co/cvg-unibe/loom-cfm_imagenet64/blob/main/model.pth

### 2. Install Dependencies

Make sure you have the required packages:

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Basic Usage

```bash
# ImageNet32 analysis
python analysis_imagenet.py \
    --checkpoint checkpoint-imagenet32.pth \
    --dataset imagenet32

# ImageNet64 analysis
python analysis_imagenet.py \
    --checkpoint checkpoint-imagenet64.pth \
    --dataset imagenet64
```

### Full Options

```bash
python analysis_imagenet.py \
    --checkpoint <path-to-checkpoint> \
    --dataset <imagenet32|imagenet64> \
    --total_samples 512 \
    --batch_size 16 \
    --output ./imagenet_output \
    --device cuda \
    --step_size 0.01 \
    --save_all_samples
```

### Arguments

- `--checkpoint`: Path to the model checkpoint file (required)
- `--dataset`: Which dataset/architecture to use: `imagenet32` or `imagenet64` (default: `imagenet64`)
- `--total_samples`: Total number of samples to generate (default: 512)
- `--batch_size`: Batch size for generation (default: 16)
- `--output`: Output directory (default: `./imagenet_analysis_<dataset>`)
- `--device`: Device to use (default: `cuda`)
- `--step_size`: ODE solver step size (default: 0.01, i.e., 100 steps)
- `--save_all_samples`: Save all generated samples (not just preview)

## Output

The script generates:

1. **samples_preview.png**: Preview grid of first 64 generated images
2. **angular_analysis.png**: 6-panel visualization showing:
   - Mean angular difference over evaluation steps
   - Angular difference vs time
   - Velocity field magnitude analysis
   - Distribution histogram of angular differences
3. **summary.txt**: Statistical summary including:
   - Cosine similarity statistics
   - Angular difference statistics (in degrees)
   - Phase-wise analysis (early/mid/late)

## How It Works

The analysis script:

1. **Loads the pre-trained ImageNet flow matching model**
2. **Generates samples using midpoint ODE solver**:
   - Starts from Gaussian noise
   - Iteratively evolves through the flow from t=0 to t=1
   - Uses midpoint method for numerical stability
3. **Tracks velocity field changes**:
   - Computes cosine similarity between consecutive velocity predictions
   - Converts to angular differences for interpretability
   - Records velocity magnitudes
4. **Aggregates statistics across multiple batches**:
   - Computes mean and standard deviation
   - Analyzes temporal patterns
   - Generates comprehensive visualizations

## Comparison with CIFAR-10

Key differences from the CIFAR-10 analysis:

- **Image Size**: ImageNet uses 32×32 or 64×64 images (vs 32×32 for CIFAR-10)
- **Model Architecture**: Uses `imagenet` config with more channels and attention heads
- **Batch Size**: Default reduced to 16 (vs 32) due to larger images
- **Sample Count**: Default 512 samples (vs 960) for faster analysis

## Memory Requirements

Approximate GPU memory usage:

- **ImageNet32**: ~4-6 GB (batch_size=16)
- **ImageNet64**: ~8-12 GB (batch_size=16)

If you encounter OOM errors, reduce `--batch_size`.

## Example Commands

```bash
# Quick test with minimal samples
python analysis_imagenet.py \
    --checkpoint checkpoint-imagenet32.pth \
    --dataset imagenet32 \
    --total_samples 128 \
    --batch_size 16

# Full analysis with all samples saved
python analysis_imagenet.py \
    --checkpoint checkpoint-imagenet64.pth \
    --dataset imagenet64 \
    --total_samples 512 \
    --batch_size 16 \
    --save_all_samples \
    --output ./imagenet64_full_analysis

# Custom step size (fewer ODE steps)
python analysis_imagenet.py \
    --checkpoint checkpoint-imagenet32.pth \
    --dataset imagenet32 \
    --step_size 0.02  # 50 steps instead of 100
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--total_samples`
- Use `--device cpu` (much slower)

### Model Loading Error
- Ensure checkpoint matches the dataset (imagenet32 checkpoint with `--dataset imagenet32`)
- Verify checkpoint file is complete and not corrupted

### Import Errors
- Make sure you're running from the repository root or the analysis directory
- Ensure `flow_matching` package is in the parent directory
