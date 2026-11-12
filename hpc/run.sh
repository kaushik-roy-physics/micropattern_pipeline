#!/bin/bash
# =============================================================================
# run.sh: run, re-run or check the pipeline jobs of ONE experiment on the HPC
# =============================================================================
# Type it from the experiment folder (the one holding config.yaml and
# hpc_job.env). It only submits the generic job files with bsub; nothing is
# edited. Settings come from hpc_job.env (GUI step 5); the options below change
# them for this one submission only.
#
#   cd /data/Warmflash/KR/exp_040
#   bash /data/Warmflash/KR/micropattern_pipeline/hpc/run.sh <what>
#
# <what>
#   all              segmentation (GPU), then the analysis (CPU) by itself   [default]
#   seg              segmentation only (finished images are skipped)
#   analysis         the analysis only: after changing config.yaml or a plate map
#   redo <step> ...  redo these analysis steps and everything after them,
#                    e.g. `redo colony` (new overlays), `redo profiles`, `redo report`
#   redo-all         redo every analysis step, ignoring checkpoints
#   reseg            segment every image again (segment + extract), then the analysis
#   test             dry run on this node: prints what each job would do; no GPU, no bsub
#   status           your jobs, and the last lines of the newest logs here
#   steps            list the analysis steps
#
# Options, written in front of the command:
#   SEQ="0-4"        only these Seq/Point numbers   (e.g. a quick test)
#   WELLS="1-3"      only these wells               (or "C1-C4")
#   DATASETS="zmax"  only these datasets (names from config.yaml)
#   SEG="2d"         only the 2-D (or 3-D) segmentation job
#   e.g.   SEQ="0-4" bash .../hpc/run.sh            # try everything on 5 images
#          DATASETS="zstacks" bash .../hpc/run.sh reseg
#
# Everything is checkpointed: re-submitting skips what is done, and a changed
# setting re-runs what depends on it by itself. `redo` is for when nothing in
# the settings changed but you want a step done again anyway.
# =============================================================================
set -euo pipefail

PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$PWD"
WHAT="${1:-all}"; shift || true

die() { echo "ERROR: $*" >&2; exit 1; }

usage() { sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
case "$WHAT" in -h|--help|help) usage; exit 0 ;; esac

[ -f "$EXP/hpc_job.env" ] || die "no hpc_job.env in $EXP.
Go to the experiment folder first (cd /path/to/exp), and save step 5 in the GUI."
[ -f "$EXP/config.yaml" ] || [ "$WHAT" = "status" ] || die "no config.yaml in $EXP (GUI step 3)."

# Read hpc_job.env for the dataset lists (no other side effects).
# shellcheck disable=SC1090
eval "$(tr -d '\r' < "$EXP/hpc_job.env" | grep -E '^(DATASETS_2D|DATASETS_3D)=')"
D2="${DATASETS_2D:-}"; D3="${DATASETS_3D:-}"

# One-off settings → MP_* variables, which the job files apply over hpc_job.env.
[ -n "${SEQ:-}" ] && export MP_SEQ="$SEQ"
[ -n "${WELLS:-}" ] && export MP_WELLS="$WELLS"
if [ -n "${DATASETS:-}" ]; then
    export MP_DOWNSTREAM_DATASETS="$DATASETS"
    keep2="" keep3=""
    for d in ${DATASETS//,/ }; do
        [[ " ${D2//,/ } " == *" $d "* ]] && keep2="$keep2 $d"
        [[ " ${D3//,/ } " == *" $d "* ]] && keep3="$keep3 $d"
    done
    D2="${keep2# }"; D3="${keep3# }"
    if [ -z "$D2$D3" ]; then
        D2="$DATASETS"     # not listed under DATASETS_2D/3D: segment them in 2-D
    fi
    export MP_DATASETS_2D="$D2" MP_DATASETS_3D="$D3"
fi
case "${SEG:-}" in
    2d|2D) D3="" ;;
    3d|3D) D2="" ;;
    "") ;;
    *) die "SEG must be 2d or 3d (got '$SEG')" ;;
esac
# hpc_job.env with neither list set: one 2-D job on every dataset.
if [ -z "$D2$D3" ] && [ -z "${DATASETS_2D:-}${DATASETS_3D:-}" ]; then D2="(all)"; fi

submit() {           # submit <lsf file> ; prints the job id
    local out
    out=$(cd "$EXP" && bsub < "$PIPE/hpc/$1") || die "bsub failed: $out"
    echo "$out" >&2
    echo "$out" | grep -o '<[0-9]\+>' | head -1 | tr -d '<>'
}

seg_jobs() {          # submit the segmentation job(s); THEN_ANALYSIS decides the chain
    local n=0
    [ -n "$D2" ] && { echo "Segmentation, 2-D: ${D2}"; submit run_pipeline_2dseg.lsf >/dev/null; n=$((n+1)); }
    [ -n "$D3" ] && { echo "Segmentation, 3-D: ${D3}"; submit run_pipeline_3dseg.lsf >/dev/null; n=$((n+1)); }
    [ "$n" -gt 0 ] || die "no dataset to segment (DATASETS_2D / DATASETS_3D in hpc_job.env are empty)."
}

case "$WHAT" in
    all)
        export MP_THEN_ANALYSIS="true"
        seg_jobs
        echo ""
        echo "The analysis starts by itself when segmentation has finished successfully."
        ;;
    seg)
        export MP_THEN_ANALYSIS="false"
        seg_jobs
        ;;
    reseg)
        export MP_THEN_ANALYSIS="true" MP_SEG_FORCE_STEPS="segment extract"
        seg_jobs
        echo ""
        echo "Every selected image is segmented again; the analysis follows by itself and"
        echo "re-detects the colonies on the new masks."
        ;;
    analysis)
        echo "Analysis (CPU):"; submit run_downstream.lsf >/dev/null
        ;;
    redo)
        [ $# -gt 0 ] || die "redo needs step names, e.g.  bash $0 redo profiles   (list: bash $0 steps)"
        export MP_DOWNSTREAM_FORCE_STEPS="$*"
        echo "Analysis, redoing: $* (and every step after them)"; submit run_downstream.lsf >/dev/null
        ;;
    redo-all)
        export MP_DOWNSTREAM_FORCE="true"
        echo "Analysis, every step redone:"; submit run_downstream.lsf >/dev/null
        ;;
    test)
        export DRY_RUN=1 EXP_DIR="$EXP"
        for job in run_pipeline_2dseg.lsf run_pipeline_3dseg.lsf; do
            if { [ "$job" = run_pipeline_2dseg.lsf ] && [ -n "$D2" ]; } ||
               { [ "$job" = run_pipeline_3dseg.lsf ] && [ -n "$D3" ]; }; then
                echo "################ $job (dry run) ################"
                bash "$PIPE/hpc/$job" || true
            fi
        done
        echo "################ run_downstream.lsf (dry run) ################"
        bash "$PIPE/hpc/run_downstream.lsf" || true
        ;;
    status)
        command -v bjobs >/dev/null && bjobs -w 2>&1 | head -30 || true
        for f in $(ls -t "$EXP"/lsf_*.log 2>/dev/null | head -2); do
            echo ""; echo "==> $(basename "$f") (last 15 lines)"; tail -15 "$f"
        done
        ;;
    steps)
        python3 "$PIPE/main.py" downstream --list-steps 2>/dev/null ||
            echo "colony signaling assemble qc thresholds boundary profiles aggregate compare multimarker coexpression signaling_summary effects report"
        ;;
    *)
        echo "Unknown command '$WHAT'." >&2; usage >&2; exit 1 ;;
esac
[ "$WHAT" != test ] && [ "$WHAT" != status ] && [ "$WHAT" != steps ] &&
    echo "Watch with: bjobs   |   logs: lsf_*.log and pipeline.log in $EXP"
exit 0
