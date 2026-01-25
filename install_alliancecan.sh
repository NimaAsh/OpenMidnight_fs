#!/usr/bin/env bash
# =============================================================================
# Alliance Canada / Compute Canada - First-time Environment Setup
# =============================================================================
# This script sets up the Python virtual environment for OpenMidnight on
# Alliance Canada clusters (Narval, Beluga, Cedar, Graham).
#
# USAGE (run this ONCE from a login node):
#   chmod +x install_alliancecan.sh
#   ./install_alliancecan.sh
#
# The virtual environment will be created in ~/scratch/openmidnight_venv
# which persists across jobs and has better I/O performance than $HOME.
#
# IMPORTANT: Do NOT create virtualenvs in $SCRATCH if they may be purged.
# For long-term storage, consider ~/projects/ instead.
# =============================================================================
set -euo pipefail

# Configuration - adjust these as needed
PYTHON_VERSION="python/3.11"
VENV_DIR="${VENV_DIR:-$SCRATCH/openmidnight_venv}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "OpenMidnight - Alliance Canada Setup"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Virtual env:  ${VENV_DIR}"
echo ""

# Load required modules
echo "[1/6] Loading modules..."
module purge
module load StdEnv/2023
module load ${PYTHON_VERSION}
module load cuda/12.2
module load cudnn/9.2.1
module load arrow/17.0.0
module load opencv/4.10.0

echo "Loaded modules:"
module list

# Create virtual environment using virtualenv (Alliance Canada recommended)
echo ""
echo "[2/6] Creating virtual environment..."
if [[ -d "${VENV_DIR}" ]]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
    read -p "Do you want to remove it and create a fresh one? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
        virtualenv --no-download "${VENV_DIR}"
    fi
else
    virtualenv --no-download "${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip using Alliance Canada wheels
echo ""
echo "[3/6] Upgrading pip..."
pip install --no-index --upgrade pip

# Install PyTorch and related packages from Alliance Canada wheels
# IMPORTANT: Use --no-index to get optimized wheels from the cluster
# Note: For H100 GPUs, torch >= 2.5.1 is required
echo ""
echo "[4/6] Installing PyTorch from Alliance Canada wheels..."
pip install --no-index torch torchvision torchaudio

# Install packages available from Alliance Canada wheelhouse
echo ""
echo "[5/6] Installing packages from Alliance Canada wheelhouse..."
pip install --no-index \
    numpy \
    scipy \
    scikit-learn \
    scikit-image \
    matplotlib \
    Pillow \
    h5py \
    pandas \
    pyarrow \
    tensorboard \
    protobuf || echo "Some wheelhouse packages not available, will install from PyPI"

# Install remaining dependencies from PyPI
# These are packages not available in the Alliance Canada wheelhouse
echo ""
echo "[6/6] Installing remaining dependencies from PyPI..."
cd "${PROJECT_ROOT}"

pip install \
    einops \
    fvcore \
    iopath \
    omegaconf \
    opencv-python \
    openslide-bin \
    openslide-python \
    submitit \
    torchmetrics \
    wandb \
    "huggingface-hub>=0.34.0" \
    xformers \
    imagesize \
    monai \
    lightning \
    "jsonargparse[omegaconf]" \
    loguru \
    transformers \
    onnxruntime \
    onnx \
    toolz \
    rich \
    nibabel \
    timm \
    hestcore \
    datasets

# Install the project in editable mode
pip install -e . --no-deps

# Loosen transformers' hub pin so hub 1.x works
python -c "
try:
    import transformers.dependency_versions_table as t
    from pathlib import Path
    p = Path(t.__file__)
    txt = p.read_text()
    if 'huggingface-hub>=0.34.0,<1.0' in txt:
        p.write_text(txt.replace('huggingface-hub>=0.34.0,<1.0', 'huggingface-hub>=0.34.0'))
        print('Patched transformers hub version constraint')
    else:
        print('Transformers hub constraint already patched or different version')
except Exception as e:
    print(f'Could not patch transformers: {e}')
"

# Create a requirements.txt for reproducibility (for use with $SLURM_TMPDIR installs)
pip freeze --local > "${PROJECT_ROOT}/requirements_alliancecan.txt"
echo "Saved package list to requirements_alliancecan.txt"

# Create a convenience activation script
cat > "${PROJECT_ROOT}/activate_alliancecan.sh" << 'ACTIVATE_EOF'
#!/usr/bin/env bash
# Source this script to activate the OpenMidnight environment on Alliance Canada
# Usage: source activate_alliancecan.sh

VENV_DIR="${VENV_DIR:-$HOME/scratch/openmidnight_venv}"

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/9.2.1
module load arrow/17.0.0
module load opencv/4.10.0

if [[ -d "${VENV_DIR}" ]]; then
    source "${VENV_DIR}/bin/activate"
    echo "OpenMidnight environment activated"
    echo "Python: $(which python)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Please run install_alliancecan.sh first"
fi
ACTIVATE_EOF
chmod +x "${PROJECT_ROOT}/activate_alliancecan.sh"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment manually, run:"
echo "  source ${PROJECT_ROOT}/activate_alliancecan.sh"
echo ""
echo "To submit a training job, use one of:"
echo "  sbatch run_alliancecan_1node.sh      # Single node, multi-GPU"
echo "  sbatch run_alliancecan_multinode.sh  # Multi-node distributed"
echo ""
echo "Remember to:"
echo "  1. Edit the SBATCH --account directive in the job scripts"
echo "  2. Run 'wandb login' if you want wandb logging"
echo ""
