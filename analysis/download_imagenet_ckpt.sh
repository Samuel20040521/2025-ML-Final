#!/bin/bash

# Script to download ImageNet checkpoints from HuggingFace

echo "=== ImageNet Checkpoint Downloader ==="
echo ""
echo "Available checkpoints:"
echo "  1) ImageNet32"
echo "  2) ImageNet64"
echo ""

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <32|64>"
    echo "Example: $0 32  # Download ImageNet32 checkpoint"
    exit 1
fi

DATASET=$1

if [ "$DATASET" == "32" ]; then
    URL="https://huggingface.co/cvg-unibe/loom-cfm_imagenet32/resolve/main/model.pth"
    OUTPUT="checkpoint-imagenet32.pth"
elif [ "$DATASET" == "64" ]; then
    URL="https://huggingface.co/cvg-unibe/loom-cfm_imagenet64/resolve/main/model.pth"
    OUTPUT="checkpoint-imagenet64.pth"
else
    echo "Error: Invalid argument. Use '32' or '64'"
    exit 1
fi

echo "Downloading ImageNet${DATASET} checkpoint..."
echo "URL: $URL"
echo "Output: $OUTPUT"
echo ""

# Download using wget or curl
if command -v wget &> /dev/null; then
    wget -O "$OUTPUT" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT" "$URL"
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Download completed successfully!"
    echo "Checkpoint saved to: $OUTPUT"
    echo ""
    echo "To run analysis, use:"
    echo "  python analysis_imagenet.py --checkpoint $OUTPUT --dataset imagenet${DATASET}"
else
    echo ""
    echo "✗ Download failed!"
    exit 1
fi
