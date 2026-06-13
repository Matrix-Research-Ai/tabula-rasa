# GPU Training Setup — Tabula Rasa
# ================================
# Run this on a rented cloud GPU (RunPod, Vast.ai, etc.)
#
# Quick start:
#   1. Rent a GPU pod (RTX 4080/4090, any PyTorch template)
#   2. Upload or git-clone this project
#   3. Run: bash setup_gpu.sh

set -e

echo "=== Tabula Rasa GPU Setup ==="
echo ""

# Create output directory
mkdir -p /output

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"

# Generate training data (pre-tokenized mmap)
echo ""
echo "=== Preparing datasets ==="
python3 prepare_dataset.py
python3 prepare_dataset.py add
python3 prepare_dataset.py sub
python3 prepare_dataset.py mul
python3 prepare_dataset.py div

# Run training
echo ""
echo "=== Starting GPU training ==="
echo "Output will be in /output/"
python3 -u train_gpu.py 2>&1 | tee /output/training_log.txt

# Package results
echo ""
echo "=== Results ==="
ls -lh /output/*.zip
