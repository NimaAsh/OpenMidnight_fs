#!/usr/bin/env bash
# =============================================================================
# Alliance Canada (Nibi) - Full ViT-G Training (Similar to run_1node.sh)
# =============================================================================
# Submit with: sbatch run_nibi_full.sh
#
# This script runs the FULL ViT-G/14 model training (vitg14_reg4.yaml)
# For ablation/testing with smaller ViT-S model, use run_nibi_1node.sh
#
# Nibi cluster specs:
#   - 36 GPU nodes with 8x H100 SXM (80GB) each, connected via NVLink
#   - 112 cores per GPU node, 2TB memory
#   - Internet access on all nodes (can stream from HuggingFace)
#
# ViT-G is much larger than ViT-S, so we use all 8 GPUs per node
# =============================================================================

#SBATCH --job-name=openmidnight-vitg
#SBATCH --account=def-ssfels              # Your PI's allocation
#SBATCH --time=4-00:00:00                 # 4 days (ViT-G takes longer)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8            # All 8x H100 GPUs
#SBATCH --cpus-per-task=112               # All 112 cores
#SBATCH --mem=0                           # Request all available memory
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nima.ashjaee@ubc.ca

set -euo pipefail

# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================
# Full reproduction (large ViT-G model) - same as run_1node.sh
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"

OUTPUT_DIR="$SCRATCH/openmidnight_output_vitg14"
RESUME="False"   # "True" to resume from checkpoint, "False" to start fresh

# Virtualenv location
VENV_DIR="$SCRATCH/openmidnight_venv"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "OpenMidnight Full Training - ViT-G/14 (Nibi)"
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
_GPUS_RAW=${SLURM_GPUS_PER_NODE:-8}
NPROC_PER_NODE=${_GPUS_RAW##*:}  # Remove "h100:" prefix if present

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

# Load secrets from .env file (WANDB_API_KEY, HF_TOKEN, etc.)
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

# Verify and login to HuggingFace if token is available
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN found, logging into HuggingFace..."
    # Also export as HUGGING_FACE_HUB_TOKEN for older versions
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    # Login via CLI to store credentials
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
    echo "HuggingFace login successful"
else
    echo "WARNING: HF_TOKEN not set - may hit rate limits with HuggingFace"
fi

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
# LAUNCH TRAINING
# =============================================================================
echo ""
echo "Starting ViT-G/14 training..."
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${REPO_ROOT}/dinov2/train/train.py" \
    --config-file "${REPO_ROOT}/${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${RESUME_FLAG}

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
