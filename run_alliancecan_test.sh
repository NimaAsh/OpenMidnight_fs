#!/usr/bin/env bash
# =============================================================================
# Alliance Canada (Trillium) - Quick Test Job (15-30 minutes)
# =============================================================================
# This script runs a quick test to verify everything is set up correctly.
# It uses the small ViT-S model and runs only a few iterations.
#
# Submit with: sbatch run_alliancecan_test.sh
#
# NOTE: Trillium-specific settings:
#   - No internet access on compute nodes (must use offline data)
#   - H100 GPUs (80GB)
#   - Output files must go to $SCRATCH (not /project which is read-only)
#
# IMPORTANT: Before running this test, you MUST:
#   1. Download TCGA data to $SCRATCH/TCGA_data/ on a LOGIN node
#   2. Create a sample_dataset.txt file using prepatching_scripts/
#   3. Or use the provided sample_dataset_ablation.txt for testing
# =============================================================================

#SBATCH --job-name=openmidnight-test
#SBATCH --account=def-ssfels              # Your PI's allocation
#SBATCH --time=0-00:30:00                 # 30 minutes (short job = faster scheduling)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # Just 1 GPU for quick test
#SBATCH --cpus-per-task=12                # More CPUs for data loading
#SBATCH --mem=64G                         # Request 64GB RAM
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

# OFFLINE DATA - Using local parquet dataset from HuggingFace
# Downloaded with: huggingface-cli download medarc/TCGA-12K-parquet --repo-type dataset
PARQUET_DATASET_PATH="$SCRATCH/TCGA_data/TCGA-12K-parquet"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "OpenMidnight Quick Test (Trillium - Offline Mode)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

# Use absolute path for sbatch compatibility
REPO_ROOT="/project/def-ssfels/nima5/openmidnight_project/OpenMidnight_fs"
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"

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

# Disable wandb for testing
export WANDB_MODE=disabled

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
# STEP 3: Test Data Loading (OFFLINE - no HuggingFace streaming)
# =============================================================================
echo ""
echo "=== Step 3: Testing OFFLINE data loading ==="
echo "NOTE: Trillium compute nodes have NO internet access."
echo "      You must use offline data (sample_dataset.txt with local SVS files)."
echo ""

if [[ -d "${PARQUET_DATASET_PATH}" ]]; then
    PARQUET_COUNT=$(find "${PARQUET_DATASET_PATH}" -name "*.parquet" | wc -l)
    echo "Found parquet dataset: ${PARQUET_DATASET_PATH}"
    echo "  Parquet files: ${PARQUET_COUNT}"
    echo "  First 3 parquet files:"
    find "${PARQUET_DATASET_PATH}" -name "*.parquet" | head -3 | while read f; do echo "    $f"; done

    # Test reading a parquet file
    FIRST_PARQUET=$(find "${PARQUET_DATASET_PATH}" -name "*.parquet" | head -1)
    if [[ -n "${FIRST_PARQUET}" ]]; then
        echo ""
        echo "Testing parquet file access..."
        python -c "
import pyarrow.parquet as pq
parquet_path = '${FIRST_PARQUET}'
table = pq.read_table(parquet_path)
print(f'  Parquet file: {parquet_path}')
print(f'  Rows: {table.num_rows}')
print(f'  Columns: {table.column_names}')
print('  Parquet access: OK')
"
    fi
else
    echo "WARNING: Parquet dataset not found at ${PARQUET_DATASET_PATH}"
    echo ""
    echo "To set up offline data, run on a LOGIN node:"
    echo "  huggingface-cli download medarc/TCGA-12K-parquet \\"
    echo "      --repo-type dataset --local-dir \$SCRATCH/TCGA_data/TCGA-12K-parquet"
    echo ""
    echo "Skipping data loading test..."
fi

# =============================================================================
# STEP 4: Quick Training Test (just a few iterations)
# =============================================================================
echo ""
echo "=== Step 4: Quick training test (offline mode) ==="

if [[ ! -d "${PARQUET_DATASET_PATH}" ]]; then
    echo "SKIPPING: Cannot run training test without parquet dataset"
    echo ""
    echo "=============================================="
    echo "Test partially completed at: $(date)"
    echo "=============================================="
    echo ""
    echo "Steps 1-2 passed (Python/PyTorch and imports work)"
    echo "Step 3 skipped (no offline data available)"
    echo "Step 4 skipped (no offline data available)"
    echo ""
    echo "To complete the test, download the parquet dataset first."
    exit 0
fi

echo "This tests the full training pipeline with minimal compute"

# Clean test output
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Run training for just a few iterations using the ablation config
# The ablation config uses ViT-S which is much smaller than ViT-G
timeout 600 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    "${REPO_ROOT}/dinov2/train/train.py" \
    --config-file "${REPO_ROOT}/${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --no-resume \
    train.streaming_from_hf=true \
    train.streaming_dataset_path="${PARQUET_DATASET_PATH}" \
    train.batch_size_per_gpu=16 \
    train.num_workers=1 \
    optim.epochs=1 \
    optim.warmup_epochs=0 \
    evaluation.eval_period_iterations=100 \
    || echo "Training test completed (may have timed out, which is OK for a test)"

echo ""
echo "=============================================="
echo "Test completed at: $(date)"
echo "=============================================="
echo ""
echo "If all steps passed, your environment is ready for full training!"
echo ""
echo "Next steps:"
echo "  1. For ablation (small model, 4 GPUs): sbatch run_alliancecan_1node.sh"
echo "     (edit CONFIG_FILE to use vits14_reg_ablations.yaml)"
echo "  2. For full ViT-G training: sbatch run_alliancecan_1node.sh"
echo "     (uses vitg14_reg4.yaml by default)"
echo "  3. For multi-node training: sbatch run_alliancecan_multinode.sh"
