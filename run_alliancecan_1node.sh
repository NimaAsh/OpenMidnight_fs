#!/usr/bin/env bash
# =============================================================================
# Alliance Canada - Single Node Multi-GPU Training
# =============================================================================
# Submit with: sbatch run_alliancecan_1node.sh
#
# This script follows Alliance Canada best practices:
# - Uses modules for Python/CUDA instead of conda/uv
# - Creates virtualenv in $SLURM_TMPDIR for best performance
# - Uses --no-index to install from Alliance Canada wheelhouse
# - Sets proper environment variables for distributed training
#
# Common clusters and their GPU types:
#   - Narval:  A100 (40GB/80GB)  - account format: def-<pi> or rrg-<pi>
#   - Beluga:  V100 (16GB)       - account format: def-<pi> or rrg-<pi>
#   - Cedar:   V100 (32GB), P100 - account format: def-<pi> or rrg-<pi>
#   - Graham:  V100, T4, P100    - account format: def-<pi> or rrg-<pi>
# =============================================================================

#SBATCH --job-name=openmidnight
#SBATCH --account=def-YOURPI              # CHANGE THIS to your allocation
#SBATCH --time=3-00:00:00                 # 3 days (max 7 days on most clusters)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4                 # Number of GPUs
#SBATCH --cpus-per-task=32                # CPUs (~6-8 per GPU recommended)
#SBATCH --mem=128G                        # Memory per node
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your@email.com        # CHANGE THIS

# Uncomment for Narval A100-80GB (if needed)
# #SBATCH --constraint=a100_80g

set -euo pipefail

# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="./output_alliancecan"
RESUME="True"   # "True" to resume from checkpoint, "False" to start fresh

# Choose virtualenv location:
# Option 1: Persistent venv in scratch (survives job restarts, but slower I/O)
VENV_DIR="${HOME}/scratch/openmidnight_venv"
# Option 2: Ephemeral venv in SLURM_TMPDIR (faster I/O, recreated each job)
# VENV_DIR="${SLURM_TMPDIR}/env"
# USE_TMPDIR_VENV="true"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

# Create output directories
mkdir -p slurms
mkdir -p "${OUTPUT_DIR}"

# Get project root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"

# Load modules (Alliance Canada recommended approach)
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/9.2.1
module load arrow/17.0.0
module load opencv/4.10.0

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP
# =============================================================================
# Option A: Use persistent venv from scratch
if [[ "${USE_TMPDIR_VENV:-false}" != "true" ]]; then
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "ERROR: Virtual environment not found at ${VENV_DIR}"
        echo "Please run install_alliancecan.sh first"
        exit 1
    fi
    source "${VENV_DIR}/bin/activate"
else
    # Option B: Create ephemeral venv in SLURM_TMPDIR (Alliance Canada recommended for performance)
    echo "Creating virtualenv in SLURM_TMPDIR..."
    virtualenv --no-download "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch torchvision
    pip install --no-index -r "${REPO_ROOT}/requirements_alliancecan.txt" 2>/dev/null || \
        pip install -r "${REPO_ROOT}/requirements_alliancecan.txt"
fi

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))  # Deterministic port based on job ID

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}

# Environment variables for optimal performance
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))
export TORCH_NCCL_ASYNC_HANDLING=1  # Recommended by Alliance Canada for distributed training
export NCCL_BLOCKING_WAIT=1

# Hugging Face cache configuration (use scratch for better I/O)
export HF_HOME="${SCRATCH:-$HOME/scratch}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

# Uncomment and set if you have an HF token:
# export HF_TOKEN="your_token_here"

# Set Python path
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# =============================================================================
# LAUNCH TRAINING
# =============================================================================
echo ""
echo "Starting training..."
echo ""

# Use torchrun for single-node multi-GPU training
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    dinov2/train/train.py \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${RESUME_FLAG}

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
