#!/usr/bin/env bash
# =============================================================================
# Alliance Canada (Trillium) - Single Node Multi-GPU Training
# =============================================================================
# Submit with: sbatch run_alliancecan_1node.sh
#
# This script follows Alliance Canada best practices:
# - Uses modules for Python/CUDA instead of conda/uv
# - Uses virtualenv in $SCRATCH for persistence
# - Sets proper environment variables for distributed training
#
# Trillium-specific notes:
#   - No --mem option (186 GiB per GPU, 745 GiB for whole node)
#   - Output files must go to $SCRATCH (not /project which is read-only)
#   - H100 GPUs with 80GB memory
#   - NO INTERNET ACCESS on compute nodes - must use offline data!
#
# Data Setup (run on LOGIN node before submitting):
#   1. Download TCGA-12K dataset to $SCRATCH/TCGA_data/
#   2. Run prepatching script to create sample_dataset.txt
#   3. Set DATASET_PATH and SAMPLE_LIST_PATH below
#   4. Use train.streaming_from_hf=false in torchrun command
#
# =============================================================================

#SBATCH --job-name=openmidnight
#SBATCH --account=def-ssfels              # Your PI's allocation
#SBATCH --time=3-00:00:00                 # 3 days (max 7 days)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8            # 8x H100 GPUs
#SBATCH --cpus-per-task=96                # ~12 cores per GPU (96/8) * 8
#SBATCH --mem=512G                        # 64GB RAM per GPU * 8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nima.ashjaee@ubc.ca

set -euo pipefail

# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================
# For ablation/testing (small model, faster):
# CONFIG_FILE="./dinov2/configs/train/vits14_reg_ablations.yaml"
# For full reproduction (large model):
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"

OUTPUT_DIR="$SCRATCH/openmidnight_output_8gpu"
RESUME="True"   # "True" to resume from checkpoint, "False" to start fresh

# Virtualenv location
VENV_DIR="$SCRATCH/openmidnight_venv"

# -----------------------------------------------------------------------------
# OFFLINE DATA CONFIGURATION (Using local parquet dataset)
# -----------------------------------------------------------------------------
# Since Trillium compute nodes have NO internet access, we use the locally
# downloaded parquet dataset from HuggingFace (medarc/TCGA-12K-parquet).
#
# To prepare the data (run on LOGIN node):
#   huggingface-cli download medarc/TCGA-12K-parquet \
#       --repo-type dataset --local-dir $SCRATCH/TCGA_data/TCGA-12K-parquet
#
PARQUET_DATASET_PATH="$SCRATCH/TCGA_data/TCGA-12K-parquet"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "OpenMidnight Training (Trillium - Offline Mode)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

# Use absolute path for sbatch compatibility
REPO_ROOT="/project/6012563/nima5/openmidnight_proj/OpenMidnight_fs"
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

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

# Extract number of GPUs (handle formats like "h100:8" or just "8")
_GPUS_RAW=${SLURM_GPUS_PER_NODE:-4}
NPROC_PER_NODE=${_GPUS_RAW##*:}  # Remove "h100:" prefix if present

# Environment variables for optimal performance
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=WARN

# HuggingFace cache (even for offline, some models may be cached)
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

# Fix cache warnings
export MPLCONFIGDIR="${SCRATCH}/.cache/matplotlib"
export FONTCONFIG_PATH="${SCRATCH}/.cache/fontconfig"
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_PATH}"

# Load secrets from .env file (WANDB_API_KEY, etc.)
# Try SCRATCH first (more reliable on compute nodes), then HOME
ENV_FILE_SCRATCH="${SCRATCH}/.openmidnight_env"
ENV_FILE_HOME="${HOME}/.openmidnight_env"

if [[ -f "${ENV_FILE_SCRATCH}" ]]; then
    echo "Loading environment from ${ENV_FILE_SCRATCH}"
    source "${ENV_FILE_SCRATCH}"
elif [[ -f "${ENV_FILE_HOME}" ]]; then
    echo "Loading environment from ${ENV_FILE_HOME}"
    source "${ENV_FILE_HOME}"
else
    echo "WARNING: No .openmidnight_env found in SCRATCH or HOME"
    echo "Create it with: echo 'export WANDB_API_KEY=\"your_key\"' > \$SCRATCH/.openmidnight_env"
    echo "Running wandb in offline mode - sync later with: wandb sync <run_dir>"
    export WANDB_MODE=offline
fi

# Set Python path
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# =============================================================================
# VERIFY OFFLINE DATA EXISTS
# =============================================================================
if [[ ! -d "${PARQUET_DATASET_PATH}" ]]; then
    echo "ERROR: Parquet dataset not found at ${PARQUET_DATASET_PATH}"
    echo ""
    echo "Since Trillium compute nodes have NO internet access, you must:"
    echo "  Download the dataset on a LOGIN node:"
    echo "    huggingface-cli download medarc/TCGA-12K-parquet \\"
    echo "        --repo-type dataset --local-dir \$SCRATCH/TCGA_data/TCGA-12K-parquet"
    exit 1
fi
PARQUET_COUNT=$(find "${PARQUET_DATASET_PATH}" -name "*.parquet" | wc -l)
echo "Using offline parquet dataset: ${PARQUET_DATASET_PATH}"
echo "  Parquet files found: ${PARQUET_COUNT}"

# =============================================================================
# HANDLE RESUME LOGIC
# =============================================================================
if [[ "${RESUME}" == "True" ]]; then
    echo "Resume enabled; preserving ${OUTPUT_DIR}"
    RESUME_FLAG=""
else
    echo "Resume disabled; cleaning ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    RESUME_FLAG="--no-resume"
fi

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================
echo ""
echo "Configuration:"
echo "  CONFIG_FILE: ${CONFIG_FILE}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  SAMPLE_LIST_PATH: ${SAMPLE_LIST_PATH}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Print GPU information
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Check GPU memory is actually available
echo ""
echo "Checking GPU memory..."
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU {i}: {props.name}, Total: {total/1e9:.1f}GB, Free: {free/1e9:.1f}GB')
    if free < 10e9:
        print(f'  WARNING: GPU {i} has less than 10GB free!')
"

# =============================================================================
# LAUNCH TRAINING (OFFLINE MODE - NO HF STREAMING)
# =============================================================================
echo ""
echo "Starting training (offline mode)..."
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/dinov2/train/train.py" \
    --config-file "${REPO_ROOT}/${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${RESUME_FLAG} \
    train.streaming_from_hf=true \
    train.streaming_dataset_path="${PARQUET_DATASET_PATH}" \
    train.num_workers=2

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
