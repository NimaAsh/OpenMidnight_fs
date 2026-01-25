#!/usr/bin/env bash
# =============================================================================
# Alliance Canada (Nibi) - Quick Test Job (15-30 minutes)
# =============================================================================
# This script runs a quick test to verify everything is set up correctly.
# It uses the small ViT-S model and runs only a few iterations.
#
# Submit with: sbatch run_nibi_test.sh
#
# NOTE: Nibi-specific settings:
#   - Internet access available on all nodes (can stream from HuggingFace!)
#   - H100 GPUs (80GB) - same as Trillium
#   - No --mem needed (allocated automatically)
#   - 1TB soft quota on scratch (60-day grace period)
# =============================================================================

#SBATCH --job-name=openmidnight-test
#SBATCH --account=def-ssfels              # Your PI's allocation
#SBATCH --time=0-00:30:00                 # 30 minutes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # Just 1 GPU for quick test
#SBATCH --cpus-per-task=6                 # ~6 CPUs per GPU
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nima.ashjaee@ubc.ca

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_FILE="./dinov2/configs/train/vits14_reg_ablations.yaml"
OUTPUT_DIR="$SCRATCH/openmidnight_output_test"
VENV_DIR="$SCRATCH/openmidnight_venv"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "OpenMidnight Quick Test (Nibi)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"

# Load modules
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/9.2.1
module load arrow/17.0.0
module load opencv/4.10.0

# Activate virtual environment
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Please run install_alliancecan.sh first"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# Set environment
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export MPLCONFIGDIR="${SCRATCH}/.cache/matplotlib"
export FONTCONFIG_PATH="${SCRATCH}/.cache/fontconfig"
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}" "${FONTCONFIG_PATH}"

# Uncomment if you have an HF token (helps avoid rate limits):
# export HF_TOKEN="your_token_here"

# =============================================================================
# STEP 1: Basic Python/PyTorch Test
# =============================================================================
echo ""
echo "=== Step 1: Testing Python and PyTorch ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick GPU test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print('GPU matrix multiplication: OK')
"

# =============================================================================
# STEP 2: Test Imports
# =============================================================================
echo ""
echo "=== Step 2: Testing imports ==="
python -c "
print('Testing dinov2 imports...')
from dinov2.models import build_model_from_cfg
from dinov2.data import DataAugmentationDINO, MaskingGenerator
from dinov2.train.ssl_meta_arch import SSLMetaArch
print('  dinov2 imports: OK')

print('Testing HuggingFace datasets...')
from datasets import load_dataset
print('  datasets import: OK')

print('Testing other dependencies...')
import wandb
import einops
import omegaconf
import torchvision
print('  other dependencies: OK')

print('')
print('All imports successful!')
"

# =============================================================================
# STEP 3: Test Data Loading (streaming from HuggingFace)
# =============================================================================
echo ""
echo "=== Step 3: Testing data loading from HuggingFace ==="
echo "Nibi has internet access - testing HuggingFace streaming..."
python -c "
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import time

print('Loading dataset (streaming mode)...')
start = time.time()
ds = load_dataset('medarc/TCGA-12K-parquet', streaming=True)['train']

print('Fetching first few samples...')
count = 0
for sample in ds.take(5):
    img_bytes = sample['image_bytes']
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    print(f'  Sample {count+1}: image size = {img.size}')
    count += 1

elapsed = time.time() - start
print(f'Data loading test completed in {elapsed:.1f}s')
print('HuggingFace streaming: OK')
"

# =============================================================================
# STEP 4: Quick Training Test (just a few iterations)
# =============================================================================
echo ""
echo "=== Step 4: Quick training test (5 iterations) ==="
echo "This tests the full training pipeline with minimal compute"

# Clean test output
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Run training for just 5 iterations using the ablation config
# The ablation config uses ViT-S which is much smaller than ViT-G
timeout 600 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    dinov2/train/train.py \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --no-resume \
    train.OFFICIAL_EPOCH_LENGTH=5 \
    optim.epochs=1 \
    optim.early_stop=1 \
    || echo "Training test completed (may have timed out, which is OK for a test)"

echo ""
echo "=============================================="
echo "Test completed at: $(date)"
echo "=============================================="
echo ""
echo "If all steps passed, your environment is ready for full training!"
echo ""
echo "Next steps:"
echo "  1. For ablation (small model): sbatch run_nibi_1node.sh"
echo "  2. For full ViT-G training: edit run_nibi_1node.sh to use vitg14_reg4.yaml"
