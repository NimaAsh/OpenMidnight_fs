#!/usr/bin/env bash
# =============================================================================
# Alliance Canada (Nibi) - Single Node Multi-GPU Training
# =============================================================================
# Submit with: sbatch run_nibi_1node.sh
#
# Nibi cluster specs:
#   - 36 GPU nodes with 8x H100 SXM (80GB) each, connected via NVLink
#   - 112 cores per GPU node, 2TB memory
#   - Internet access on all nodes (can stream from HuggingFace)
#   - 1TB soft quota on scratch (60-day grace period)
#
# For tightly coupled multi-node jobs, add: #SBATCH --switches=1
# =============================================================================

#SBATCH --job-name=openmidnight
#SBATCH --account=def-ssfels              # Your PI's allocation
#SBATCH --time=3-00:00:00                 # 3 days (max 7 days)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8                 # Nibi has 8x H100 per GPU node
#SBATCH --cpus-per-task=112               # Full node (112 cores for GPU nodes)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nima.ashjaee@ubc.ca

set -euo pipefail

# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================
# For ablation/testing (small model, faster):
CONFIG_FILE="./dinov2/configs/train/vits14_reg_ablations.yaml"
# For full reproduction (large model):
# CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"

OUTPUT_DIR="$SCRATCH/openmidnight_output"
RESUME="True"   # "True" to resume from checkpoint, "False" to start fresh

# Virtualenv location
VENV_DIR="$SCRATCH/openmidnight_venv"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "OpenMidnight Training (Nibi)"
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

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-8}

# Environment variables for optimal performance
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=WARN

# HuggingFace cache (Nibi has internet access!)
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

# Fix cache warnings
export MPLCONFIGDIR="${SCRATCH}/.cache/matplotlib"
export FONTCONFIG_PATH="${SCRATCH}/.cache/fontconfig"
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_PATH}"

# Uncomment if you have an HF token:
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
