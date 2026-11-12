#!/bin/bash
# =============================================================================
# Build, check or repair the Python environment the HPC jobs use.
# =============================================================================
# Optional: every job already checks the environment before it starts and
# repairs whatever is broken (hpc/common.sh, ensure_env). Run this by hand to
# build the environment ahead of the first job, or to check one:
#
#     bash /data/Warmflash/KR/micropattern_pipeline/hpc/setup_env.sh /scratch/$USER/micropattern_venv
#     FRESH_VENV=true bash .../setup_env.sh /scratch/$USER/micropattern_venv   # rebuild from scratch
#
# Installs exactly as the jobs do: PyTorch + torchvision force-reinstalled from
# the CUDA 12.8 index (Blackwell sm_120 kernels; cu126 and older cannot use the
# gpu-b6000 GPUs), then requirements.txt with torch pinned to that build.
# The GPU itself is tested at the start of every segmentation job.
# =============================================================================
set -euo pipefail

VENV_DIR="${1:-/scratch/$USER/micropattern_venv}"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.10.7}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SETTINGS="(command line)"

# shellcheck disable=SC1091
source "$PIPELINE_DIR/hpc/common.sh"

if type module >/dev/null 2>&1; then
    module load "$PYTHON_MODULE"
    module load "$CUDA_MODULE"
fi

echo "Pipeline    : $PIPELINE_DIR"
echo "Venv        : $VENV_DIR"
echo "Python      : $(python3 --version 2>&1) ($PYTHON_MODULE)"
echo "Torch index : $TORCH_INDEX"
ensure_env

echo ""
echo "Done. Set this in your experiment's hpc_job.env (GUI step 5):"
echo "    VENV_DIR=\"$VENV_DIR\""
