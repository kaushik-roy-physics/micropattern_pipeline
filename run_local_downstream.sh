#!/bin/bash
# =============================================================================
# Downstream analysis on THIS computer: manifest -> downstream (CPU only)
# =============================================================================
# The local twin of hpc/run_downstream.lsf. Needs the segmentation results
# (run_local_2dseg.sh or the HPC GPU job) and writes the tables, figures and
# report.html into the analysis folder (downstream.results_dir in config.yaml;
# with the default 'auto' it is analysis_2d / analysis_3d, after the results used).
#
#     bash run_local_downstream.sh /path/to/exp_040
#
# Optional settings, given in front of the command:
#     DATASETS="zmax"            only these dataset names     (default: all in config)
#     SEQ="0-9"                  only these Seq indices        (default: all images)
#     WELLS="1-3"                only these wells              (default: all wells)
#     STEPS="profiles compare"   only these steps              (default: all; see below)
#     FORCE=true                 redo every step, ignoring checkpoints
#     FORCE_STEPS="thresholds"   redo these steps only
#     SEG_MODE="3d"              analyze only 2d or 3d results (default: downstream.seg_mode)
#     OUT=/path/to/folder        write somewhere other than downstream.results_dir
#     CONDA_ENV=cellpose_sam     the conda environment with the pipeline's packages
#   e.g.  DATASETS="zmax" bash run_local_downstream.sh /path/to/exp_040
#
# List the steps:  python main.py downstream --list-steps
# Safe to re-run: steps whose inputs and settings did not change are skipped.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIR="${1:-}"
[ -n "$EXP_DIR" ] || { echo "Usage: bash $0 /path/to/experiment_folder" >&2; exit 1; }
EXP_DIR="$(cd "$EXP_DIR" && pwd)"
CONFIG="$EXP_DIR/config.yaml"
[ -f "$CONFIG" ] || { echo "ERROR: no config.yaml in $EXP_DIR. Save step 3 in the GUI first." >&2; exit 1; }

# --- Python environment --------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-cellpose_sam}"
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: conda environment '$CONDA_ENV' not found." >&2; exit 1; }
fi
python -c "import pandas, scipy, matplotlib, sklearn" 2>/dev/null ||
    { echo "ERROR: $(command -v python) lacks the pipeline's packages. Set CONDA_ENV=<env>." >&2; exit 1; }

# --- What to run -----------------------------------------------------------------
ARGS=(--config "$CONFIG")
# shellcheck disable=SC2206
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets $DATASETS)
SELECT=()
# shellcheck disable=SC2206
[ -n "${SEQ:-}" ] && SELECT+=(--seq $SEQ)
# shellcheck disable=SC2206
[ -n "${WELLS:-}" ] && SELECT+=(--wells $WELLS)
DOWN=(downstream "${ARGS[@]}" "${SELECT[@]}")
# shellcheck disable=SC2206
[ -n "${STEPS:-}" ] && DOWN+=(--steps $STEPS)
[ "${FORCE:-false}" = "true" ] && DOWN+=(--force)
# shellcheck disable=SC2206
[ -n "${FORCE_STEPS:-}" ] && DOWN+=(--force-step $FORCE_STEPS)
[ -n "${OUT:-}" ] && DOWN+=(--out "$OUT")
[ -n "${SEG_MODE:-}" ] && DOWN+=(--seg-mode "$SEG_MODE")

cd "$EXP_DIR"     # pipeline.log lands in the experiment folder
echo ""
echo "=== Manifest: plate maps -> image/well/condition table  $(date) ==="
python "$HERE/main.py" manifest "${ARGS[@]}"
echo ""
echo "=== Downstream analysis  $(date) ==="
python "$HERE/main.py" "${DOWN[@]}"
echo ""
echo "Downstream finished $(date). Open report.html in the analysis folder."
