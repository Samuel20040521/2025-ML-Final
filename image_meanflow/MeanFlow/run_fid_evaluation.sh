#!/bin/bash

# ImageNet256 FID Evaluation with Different Timestep Schedules
# 
# This script evaluates FID scores for various timestep counts (N) using both
# uniform and non-uniform timestep schedules on ImageNet 256x256.

echo "=============================================================================="
echo "  ImageNet256 MeanFlow - FID Evaluation for Different Timestep Schedules"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Generate 50,000 samples for each timestep count"
echo "  2. Compute FID using ImageNet256 test set statistics"
echo "  3. Compare uniform vs non-uniform timestep schedules"
echo "  4. Generate comparison plots"
echo ""

# Default settings
CHECKPOINT="sit_b_2_meanflow_ema.pt"
OUTPUT_DIR="fid_timestep_evaluation"
NUM_GPUS=1
PER_GPU_BATCH=16
NUM_SAMPLES=50000
CFG_SCALE=1.5
GAMMA=0.5
N_VALUES="1,10,20,30,40,50,60,70,80,90,100"
FID_STATS="fid_stats/adm_in256_stats.npz"

echo "Default Configuration:"
echo "  Checkpoint:        $CHECKPOINT"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Number of GPUs:    $NUM_GPUS"
echo "  Batch Size/GPU:    $PER_GPU_BATCH"
echo "  Total Samples:     $NUM_SAMPLES"
echo "  CFG Scale:         $CFG_SCALE"
echo "  Gamma:             $GAMMA"
echo "  N values:          $N_VALUES"
echo "  FID Statistics:    $FID_STATS"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file '$CHECKPOINT' not found!"
    echo "Please download the checkpoint first or specify the correct path."
    exit 1
fi

# Check if FID statistics exist
if [ ! -f "$FID_STATS" ]; then
    echo "WARNING: FID statistics file '$FID_STATS' not found!"
    echo "You may need to compute or download the reference statistics first."
    echo ""
fi

echo "To customize settings, you can run the script directly:"
echo ""
echo "torchrun --nproc_per_node=NUM_GPUS evaluate_fid_timesteps.py \\"
echo "    --ckpt $CHECKPOINT \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --per-proc-batch-size $PER_GPU_BATCH \\"
echo "    --num-fid-samples $NUM_SAMPLES \\"
echo "    --cfg-scale $CFG_SCALE \\"
echo "    --gamma $GAMMA \\"
echo "    --n-values \"$N_VALUES\" \\"
echo "    --fid-statistics-file $FID_STATS \\"
echo "    --cleanup-images  # Optional: delete images after FID computation"
echo ""

read -p "Do you want to start the evaluation now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Starting evaluation..."
    echo ""
    
    torchrun --nproc_per_node=$NUM_GPUS evaluate_fid_timesteps.py \
        --ckpt "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --per-proc-batch-size $PER_GPU_BATCH \
        --num-fid-samples $NUM_SAMPLES \
        --cfg-scale $CFG_SCALE \
        --gamma $GAMMA \
        --n-values "$N_VALUES" \
        --fid-statistics-file "$FID_STATS"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=============================================================================="
        echo "Evaluation completed successfully!"
        echo "=============================================================================="
        echo ""
        echo "Results saved to: $OUTPUT_DIR/"
        echo "  - results.json: FID scores for all timestep counts"
        echo "  - fid_comparison.png: Visualization comparing uniform vs non-uniform"
        echo ""
        echo "Sample directories:"
        ls -d "$OUTPUT_DIR"/meanflow-steps-* 2>/dev/null | head -5
        echo ""
    else
        echo ""
        echo "ERROR: Evaluation failed!"
        exit 1
    fi
else
    echo "Evaluation cancelled."
fi
