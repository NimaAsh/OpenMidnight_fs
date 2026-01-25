#!/usr/bin/env bash
# =============================================================================
# Alliance Canada - Multi-Node Distributed Training
# =============================================================================
# Submit with: sbatch run_alliancecan_multinode.sh
#
# This script follows Alliance Canada best practices for multi-node training:
# - Uses srun to launch torchrun on each node
# - Creates virtualenv on each node in $SLURM_TMPDIR for best I/O
# - Sets TORCH_NCCL_ASYNC_HANDLING=1 as recommended
# - Proper NCCL configuration for InfiniBand networks
#
# NOTE: Multi-node training works best on clusters with InfiniBand:
# - Narval and Cedar have InfiniBand for fast GPU-to-GPU communication
# =============================================================================

#SBATCH --job-name=openmidnight-multi
#SBATCH --account=def-YOURPI              # CHANGE THIS to your allocation
#SBATCH --time=3-00:00:00                 # 3 days (max 7 days)
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=1               # One task per node (torchrun handles processes)
#SBATCH --gpus-per-node=4                 # GPUs per node
#SBATCH --cpus-per-task=32                # CPUs per node (~6-8 per GPU)
#SBATCH --mem=128G                        # Memory per node
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your@email.com        # CHANGE THIS

# Uncomment for Narval A100-80GB
# #SBATCH --constraint=a100_80g

set -euo pipefail

# =============================================================================
# USER CONFIGURATION
# =============================================================================
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="./output_alliancecan_multi"
RESUME="True"
VENV_DIR="${HOME}/scratch/openmidnight_venv"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "=============================================="

# Create output directories
mkdir -p slurms
mkdir -p "${OUTPUT_DIR}"

# Get project root
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

# =============================================================================
# CREATE VIRTUALENV ON EACH NODE (Alliance Canada recommended for performance)
# =============================================================================
echo "Creating virtualenv on each node in SLURM_TMPDIR..."

srun --ntasks="${SLURM_NNODES}" --tasks-per-node=1 bash << VENV_EOF
virtualenv --no-download \$SLURM_TMPDIR/env
source \$SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
# Install from requirements file or persistent venv
if [[ -f "${REPO_ROOT}/requirements_alliancecan.txt" ]]; then
    pip install --no-index -r "${REPO_ROOT}/requirements_alliancecan.txt" 2>/dev/null || \
        pip install -r "${REPO_ROOT}/requirements_alliancecan.txt"
elif [[ -d "${VENV_DIR}" ]]; then
    # Fallback: copy site-packages from persistent venv
    cp -r "${VENV_DIR}"/lib/python*/site-packages/* \$SLURM_TMPDIR/env/lib/python*/site-packages/ 2>/dev/null || true
fi
VENV_EOF

# Activate on main node
source $SLURM_TMPDIR/env/bin/activate

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

NNODES=${SLURM_NNODES}
NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-4}

# Environment variables for optimal distributed performance
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))
export TORCH_NCCL_ASYNC_HANDLING=1  # Alliance Canada recommended

# For InfiniBand networks (Narval, Cedar)
export NCCL_IB_DISABLE=0

# Hugging Face cache configuration
export HF_HOME="${SCRATCH:-$HOME/scratch}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

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
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo ""
echo "Configuration:"
echo "  CONFIG_FILE: ${CONFIG_FILE}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  NNODES: ${NNODES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  WORLD_SIZE: ${WORLD_SIZE}"
echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# =============================================================================
# LAUNCH DISTRIBUTED TRAINING
# =============================================================================
# Use srun to launch torchrun on each node
# torchrun will spawn NPROC_PER_NODE processes per node and handle distributed setup

echo "Starting distributed training across ${NNODES} nodes..."
echo ""

srun --ntasks="${NNODES}" --tasks-per-node=1 bash -c "
    source \$SLURM_TMPDIR/env/bin/activate

    # Get this node's rank from SLURM
    NODE_RANK=\${SLURM_NODEID}

    echo \"Node \${NODE_RANK} (\$(hostname)) starting torchrun...\"

    torchrun \\
        --nnodes=${NNODES} \\
        --nproc_per_node=${NPROC_PER_NODE} \\
        --node_rank=\${NODE_RANK} \\
        --master_addr=${MASTER_ADDR} \\
        --master_port=${MASTER_PORT} \\
        dinov2/train/train.py \\
        --config-file ${CONFIG_FILE} \\
        --output-dir ${OUTPUT_DIR} \\
        ${RESUME_FLAG}
"

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
