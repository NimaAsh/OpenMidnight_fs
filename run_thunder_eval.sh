#!/usr/bin/env bash
# =============================================================================
# THUNDER Benchmark Evaluation for OpenMidnight
# =============================================================================
# This script evaluates OpenMidnight on THUNDER benchmark tasks
#
# For INTERACTIVE jobs (recommended for debugging):
#   salloc --account=def-ssfels --time=2:00:00 --gpus-per-node=1 --cpus-per-task=8 --mem=64G
#   bash run_thunder_eval.sh
#
# For BATCH jobs (recommended for full evaluation):
#   sbatch run_thunder_eval.sh
#
# THUNDER tasks available:
#   - linear_probing: Train a linear classifier on frozen features
#   - knn: k-Nearest Neighbors classification
#   - simple_shot: Few-shot classification (the problematic one!)
#   - image_retrieval: Image retrieval benchmark
#   - transformation_invariance: Test robustness to transforms
#   - adversarial_attack: Test adversarial robustness
# =============================================================================

#SBATCH --job-name=thunder-eval
#SBATCH --account=def-ssfels
#SBATCH --time=6:00:00                    # 6 hours should be enough for most tasks
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # Single GPU is sufficient for eval
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=thunder-%x-%j.out
#SBATCH --error=thunder-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nima.ashjaee@ubc.ca

set -euo pipefail

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Choose evaluation mode:
#   - "quick": Run on 1-2 small datasets for testing
#   - "standard": Run on key benchmark datasets
#   - "full": Run on ALL THUNDER datasets (takes a long time)
EVAL_MODE="${EVAL_MODE:-quick}"

# Tasks to evaluate (space-separated)
# Options: linear_probing knn simple_shot image_retrieval
TASKS="${TASKS:-linear_probing knn simple_shot}"

# Model to evaluate
MODEL_NAME="openmidnight"

# Where to store THUNDER data (datasets, embeddings, results)
THUNDER_BASE="${SCRATCH}/thunder_data"

# Your trained checkpoint (leave empty to use HuggingFace version)
# CHECKPOINT_PATH="${SCRATCH}/openmidnight_output_vitg14/checkpoints/teacher_checkpoint.pth"
CHECKPOINT_PATH=""

# Virtual environment
VENV_DIR="${SCRATCH}/openmidnight_venv"

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
# Quick mode: Just test on 1-2 small datasets
DATASETS_QUICK="mhist"

# Standard mode: Key benchmark datasets
DATASETS_STANDARD="mhist bach break_his crc patch_camelyon"

# Full mode: All THUNDER histopathology datasets
DATASETS_FULL="mhist bach break_his crc patch_camelyon bracs consep gleason monusac panda"

# SPIDER datasets (separate - these are newer)
DATASETS_SPIDER="spider_breast spider_colorectal spider_skin spider_thorax"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "=============================================="
echo "THUNDER Benchmark Evaluation"
echo "Mode: ${EVAL_MODE}"
echo "Tasks: ${TASKS}"
echo "Start time: $(date)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Node: ${SLURMD_NODENAME}"
fi
echo "=============================================="

# Handle interactive vs batch job
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_ROOT="/project/6012563/nima5/openmidnight_proj/OpenMidnight_fs"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"

# Load modules (only if on compute node)
# IMPORTANT: opencv must be loaded BEFORE activating venv on Alliance Canada
if command -v module &> /dev/null; then
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load cuda/12.2
    module load cudnn/9.2.1
    module load arrow/17.0.0
    module load opencv/4.10.0  # Required for thunder-bench
fi

# Activate virtual environment
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Please run install_alliancecan.sh first, or install thunder with:"
    echo "  pip install thunder-bench"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# Check if thunder is installed
if ! python -c "import thunder" 2>/dev/null; then
    echo "Installing thunder-bench..."
    pip install thunder-bench
fi

# Set THUNDER environment variables
export THUNDER_BASE_DATA_FOLDER="${THUNDER_BASE}"
mkdir -p "${THUNDER_BASE}"

# HuggingFace cache
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "THUNDER data folder: ${THUNDER_BASE}"
echo "HuggingFace cache: ${HF_HOME}"

# =============================================================================
# SELECT DATASETS BASED ON MODE
# =============================================================================
case "${EVAL_MODE}" in
    quick)
        DATASETS="${DATASETS_QUICK}"
        ;;
    standard)
        DATASETS="${DATASETS_STANDARD}"
        ;;
    full)
        DATASETS="${DATASETS_FULL}"
        ;;
    spider)
        DATASETS="${DATASETS_SPIDER}"
        ;;
    *)
        echo "ERROR: Unknown EVAL_MODE: ${EVAL_MODE}"
        echo "Valid options: quick, standard, full, spider"
        exit 1
        ;;
esac

echo "Datasets to evaluate: ${DATASETS}"
echo ""

# =============================================================================
# RUN EVALUATIONS
# =============================================================================
RESULTS_DIR="${THUNDER_BASE}/results/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

# Log file
LOG_FILE="${RESULTS_DIR}/evaluation.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Function to run a single evaluation
run_eval() {
    local dataset=$1
    local task=$2

    echo "=============================================="
    echo "Evaluating: ${MODEL_NAME} on ${dataset} - ${task}"
    echo "Time: $(date)"
    echo "=============================================="

    # Run thunder benchmark
    # Using embedding_pre_loading mode for efficiency (extracts embeddings once)
    python -c "
from thunder import benchmark
import sys

try:
    benchmark(
        '${MODEL_NAME}',
        dataset='${dataset}',
        task='${task}',
        loading_mode='embedding_pre_loading',
    )
    print(f'SUCCESS: ${MODEL_NAME} on ${dataset} - ${task}')
except Exception as e:
    print(f'ERROR: ${MODEL_NAME} on ${dataset} - ${task}: {e}')
    sys.exit(1)
"

    local status=$?
    if [[ $status -eq 0 ]]; then
        echo "✓ Completed: ${dataset} - ${task}"
    else
        echo "✗ Failed: ${dataset} - ${task}"
    fi
    echo ""

    return $status
}

# Track results
TOTAL=0
PASSED=0
FAILED=0
FAILED_LIST=""

# Run all evaluations
for dataset in ${DATASETS}; do
    for task in ${TASKS}; do
        TOTAL=$((TOTAL + 1))

        if run_eval "${dataset}" "${task}"; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST="${FAILED_LIST}\n  - ${dataset}/${task}"
        fi
    done
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "Total evaluations: ${TOTAL}"
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
if [[ -n "${FAILED_LIST}" ]]; then
    echo -e "Failed evaluations:${FAILED_LIST}"
fi
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "THUNDER results also in: ${THUNDER_BASE}/results/"
echo "End time: $(date)"
echo "=============================================="

# Exit with error if any failed
[[ ${FAILED} -eq 0 ]]
