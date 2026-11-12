#!/bin/bash
# =============================================================================
# 2D nuclear segmentation on THIS computer: check -> load -> segment -> extract
# =============================================================================
# The local twin of hpc/run_pipeline_2dseg.lsf. Segments each image on its DAPI
# maximum-intensity projection (2-D Cellpose-SAM), using the config.yaml and
# plate maps you saved from the GUI.
#
#     bash run_local_2dseg.sh /path/to/exp_040
#
# Optional settings, given in front of the command:
#     SEQ="0-9"              only these Seq indices        (default: all images)
#     WELLS="1-3"            only these wells              (default: all wells)
#     DATASETS="zmax"        only these dataset names      (default: all in config)
#     FORCE=true             redo every step, ignoring finished images
#     CONDA_ENV=cellpose_sam the conda environment with cellpose + torch
#   e.g.  SEQ="0-4" DATASETS="zmax" bash run_local_2dseg.sh /path/to/exp_040
#
# Safe to re-run: finished images are skipped (per-image checkpoints). Uses the
# GPU when PyTorch can see one, otherwise the CPU (much slower).
# Afterwards run the analysis from GUI step 4, or:
#     python main.py manifest   --config /path/to/exp_040/config.yaml
#     python main.py downstream --config /path/to/exp_040/config.yaml
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIR="${1:-}"
[ -n "$EXP_DIR" ] || { echo "Usage: bash $0 /path/to/experiment_folder" >&2; exit 1; }
EXP_DIR="$(cd "$EXP_DIR" && pwd)"
CONFIG="$EXP_DIR/config.yaml"
[ -f "$CONFIG" ] || { echo "ERROR: no config.yaml in $EXP_DIR - save step 3 in the GUI first." >&2; exit 1; }

# --- Python environment --------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-cellpose_sam}"
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: conda environment '$CONDA_ENV' not found." >&2; exit 1; }
fi
python -c "import cellpose, torch" 2>/dev/null ||
    { echo "ERROR: $(command -v python) has no cellpose/torch. Set CONDA_ENV=<env with cellpose>." >&2; exit 1; }
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none - using the CPU (slow)')"

# --- Which images ----------------------------------------------------------------
ARGS=(--config "$CONFIG")
# shellcheck disable=SC2206
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets $DATASETS)
SELECT=()
# shellcheck disable=SC2206
[ -n "${SEQ:-}" ] && SELECT+=(--seq $SEQ)
# shellcheck disable=SC2206
[ -n "${WELLS:-}" ] && SELECT+=(--wells $WELLS)
PROCESS=(process "${ARGS[@]}" --seg-mode 2d --steps load segment extract "${SELECT[@]}")
[ "${FORCE:-false}" = "true" ] && PROCESS+=(--force)

cd "$EXP_DIR"     # pipeline.log lands in the experiment folder
echo ""
echo "=== Check: config + plate maps vs files ==="
python "$HERE/main.py" check "${ARGS[@]}"
echo ""
echo "=== Segmentation (2d): load segment extract  $(date) ==="
python "$HERE/main.py" "${PROCESS[@]}"
echo ""
echo "Segmentation finished $(date). Per-image results are next to each image."
echo "Next: GUI step 4 'Run downstream analysis' (or main.py manifest + downstream)."
