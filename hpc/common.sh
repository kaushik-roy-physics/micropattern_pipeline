#!/bin/bash
# =============================================================================
# Shared body of the HPC job scripts. Not submitted directly: each .lsf file
# reads the experiment's hpc_job.env and then sources this file.
#
# Everything experiment-specific comes from hpc_job.env (see
# templates/hpc_job.env); everything here is generic.
#
# DRY_RUN=1 prints the commands instead of running them, which lets you test a
# settings file on the login node without using any GPU time:
#
#     cd /path/to/exp_033
#     DRY_RUN=1 bash /path/to/micropattern_pipeline/hpc/run_pipeline_2dseg.lsf
# =============================================================================

# --- helpers -----------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

is_true() { case "${1,,}" in true|yes|1|on) return 0 ;; *) return 1 ;; esac; }

# Run (or, with DRY_RUN=1, just print) a command.
run() {
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi
}

# Split a space/comma separated setting into the named array. (No `local -n`:
# the login node's bash 4.2 predates namerefs.)
to_array() {
    local _txt="${2//,/ }"
    read -r -a "$1" <<< "$_txt"
}

# Always executed, also in a dry run: loading a module changes nothing on disk,
# and the environment check that follows needs it (the venv's python3 links
# against the module's libpython).
load_module() {
    echo "+ module load $1"
    module load "$1"
}

# --- the Python environment ------------------------------------------------
#
# Every job checks the venv before it runs (hpc/check_env.py: a few seconds, no
# downloads) and repairs only what is broken, so a working environment costs no
# install time while a broken one is fixed the way the original
# run_3d_segmentation.lsf did it:
#
#   * PyTorch: uninstall, then force-reinstall torch + torchvision from the
#     CUDA 12.8 index (TORCH_INDEX). cu128 is the first build with Blackwell
#     (sm_120) kernels; cu126 and older import fine but cannot use the GPU, and a
#     plain `pip install torch` would leave such a wheel in place.
#   * Everything else: pip install -r requirements.txt without its torch lines, so
#     pip keeps the cu128 build (a plain PyPI torch is now built for CUDA 13.0,
#     which needs a newer GPU driver). The check afterwards confirms it.
#
# FRESH_VENV=true in hpc_job.env deletes and rebuilds the venv from scratch
# (the old behavior: slow, but the last resort). A venv built with another
# Python than PYTHON_MODULE is rebuilt automatically. Two jobs starting at once
# take turns (flock), so they never install into the same venv together.

TORCH_INDEX_DEFAULT="https://download.pytorch.org/whl/cu128"

_install_torch() {
    run pip uninstall -y torch torchvision || true
    run pip install torch torchvision --index-url "${TORCH_INDEX:-$TORCH_INDEX_DEFAULT}" \
        --force-reinstall --quiet
}

_install_requirements() {
    local reqs
    reqs="$(mktemp)"
    grep -viE '^[[:space:]]*(torch|torchvision)([^a-z0-9_.-]|$)' \
        "$PIPELINE_DIR/requirements.txt" > "$reqs" || true
    run pip install -r "$reqs" --quiet
    rm -f "$reqs"
}

# CUDA version of the torch build TORCH_INDEX serves: .../cu128 -> 12.8.
_index_cuda() {
    local cu
    cu=$(echo "${TORCH_INDEX:-$TORCH_INDEX_DEFAULT}" | grep -oE 'cu[0-9]{3}' | tail -1)
    cu=${cu#cu}
    [ -n "$cu" ] && echo "${cu:0:${#cu}-1}.${cu: -1}" || echo "12.8"
}

_check_env() {
    python3 "$PIPELINE_DIR/hpc/check_env.py" "$PIPELINE_DIR/requirements.txt" \
        --cuda "${TORCH_CUDA:-$(_index_cuda)}" --arch "${GPU_ARCH:-sm_120}"
}

ensure_env() {
    [ -n "${VENV_DIR:-}" ] || die "VENV_DIR is not set in ${SETTINGS:-hpc_job.env}"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        if [ -f "$VENV_DIR/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$VENV_DIR/bin/activate"
            _check_env || echo "(dry run: the job would repair the environment first. On the
login node this check is only indicative: it is older than the compute nodes, so a
torch import error here is expected.)"
        else
            echo "(dry run: $VENV_DIR does not exist yet; the job would build it)"
        fi
        return 0
    fi

    mkdir -p "$(dirname "$VENV_DIR")"
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$VENV_DIR.lock"
        if ! flock -n 9; then
            echo "Another job is using $VENV_DIR (installing or checking): waiting for it..."
            flock -w 7200 9 || die "timed out waiting for $VENV_DIR.lock"
        fi
    fi

    if is_true "${FRESH_VENV:-false}" && [ -d "$VENV_DIR" ]; then
        echo "FRESH_VENV=true: deleting $VENV_DIR and building it again."
        rm -rf "$VENV_DIR"
    fi
    if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
        local want have
        want=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
        have=$(sed -n 's/^version[_info]* *= *\([0-9]*\.[0-9]*\).*/\1/p' "$VENV_DIR/pyvenv.cfg" | head -1)
        if [ -n "$have" ] && [ "$have" != "$want" ]; then
            echo "$VENV_DIR was built with Python $have, the loaded module is $want: rebuilding."
            rm -rf "$VENV_DIR"
        fi
    fi
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "Creating the Python environment $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        run pip install --upgrade pip --quiet
        _install_torch
        _install_requirements
    else
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi

    local status=0
    _check_env || status=$?
    if [ "$status" -ne 0 ]; then
        run pip install --upgrade pip --quiet
        if [ "$status" -eq 2 ]; then _install_torch; fi
        _install_requirements
        _check_env || die "the Python environment is still not right after reinstalling (see
the lines above). Set FRESH_VENV=\"true\" in hpc_job.env to rebuild it from scratch."
    fi

    if command -v flock >/dev/null 2>&1; then
        flock -u 9
        exec 9>&-
    fi
}

# --- setup -------------------------------------------------------------------

pl_setup() {
    local kind=$1            # gpu | cpu

    [ -n "${PIPELINE_DIR:-}" ] || die "PIPELINE_DIR is not set in $SETTINGS"
    [ -f "$PIPELINE_DIR/main.py" ] || die "PIPELINE_DIR=$PIPELINE_DIR does not contain main.py"

    CONFIG="${CONFIG:-config.yaml}"
    case "$CONFIG" in /*) ;; *) CONFIG="$EXP_DIR/$CONFIG" ;; esac
    [ -f "$CONFIG" ] || die "config not found: $CONFIG"

    cd "$EXP_DIR"

    echo "============================================================"
    echo " Job        : ${LSB_JOBNAME:-local} (${LSB_JOBID:-no job id})  [$kind]"
    echo " Host       : $(hostname)"
    echo " Experiment : $EXP_DIR"
    echo " Settings   : $SETTINGS"
    echo " Config     : $CONFIG"
    echo " Pipeline   : $PIPELINE_DIR"
    echo " Python env : ${VENV_DIR:-<none>}"
    echo " Started    : $(date)"
    [ "${DRY_RUN:-0}" = "1" ] && echo " DRY RUN    : commands are printed, not executed"
    local ovr
    ovr=$( (compgen -v MP_ || true) | sed 's/^MP_//' | tr '\n' ' ')
    [ -n "$ovr" ] && echo " One-off    : $ovr(from the command line; win over hpc_job.env)"
    echo "============================================================"

    if type module >/dev/null 2>&1; then
        [ -n "${PYTHON_MODULE:-}" ] && load_module "$PYTHON_MODULE"
        if [ "$kind" = "gpu" ] && [ -n "${CUDA_MODULE:-}" ]; then
            load_module "$CUDA_MODULE"
        fi
    else
        echo "(no 'module' command here: skipping module loads)"
    fi

    ensure_env

    PY=(python3 "$PIPELINE_DIR/main.py")
    COMMON=(--config "$CONFIG" --log-level "${LOG_LEVEL:-INFO}")
    SELECT=()
    local seq wells
    to_array seq "${SEQ:-}"
    to_array wells "${WELLS:-}"
    [ ${#seq[@]} -gt 0 ] && SELECT+=(--seq ${seq[@]+"${seq[@]}"})
    [ ${#wells[@]} -gt 0 ] && SELECT+=(--wells ${wells[@]+"${wells[@]}"})
    return 0
}

verify_gpu() {
    echo "Verifying PyTorch GPU access..."
    [ "${DRY_RUN:-0}" = "1" ] && { echo "(dry run: skipped)"; return 0; }
    python3 - <<'PYEOF' || die "the GPU test failed (see above). If the wheel looks right,
set FRESH_VENV=\"true\" in hpc_job.env and resubmit to rebuild the environment."
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"Built for CUDA  : {torch.version.cuda}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("No GPU visible to PyTorch - aborting (was the job sent to a GPU queue?).")
print(f"GPU             : {torch.cuda.get_device_name(0)}")
cc = torch.cuda.get_device_capability(0)
print(f"Compute cap.    : sm_{cc[0]}{cc[1]}")
print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# is_available() stays True even on an incompatible build: the failure only
# shows up on a real kernel launch, so run one to fail fast before the pipeline.
x = torch.randn(1024, 1024, device="cuda"); torch.mm(x, x); torch.cuda.synchronize()
print("GPU compute     : OK")
PYEOF
}

# --- segmentation -> analysis, without job-id bookkeeping -------------------
#
# Each segmentation job leaves a marker file in the experiment folder while it
# runs. When it finishes successfully it removes its marker and, if no other
# segmentation job of this experiment is still pending or running, submits
# run_downstream.lsf. With a 2-D and a 3-D job, whichever finishes last starts
# the analysis; a job that fails starts nothing (resubmit it, and it will).
# The check runs under a lock (mkdir is atomic on network file systems too), so
# two jobs finishing together cannot both, or neither, submit.

SEG_MARK_PREFIX=".seg_job_"

_job_alive() {             # 0 when LSF reports the job pending or running
    local stat
    command -v bjobs >/dev/null 2>&1 || return 1
    stat=$(bjobs -noheader "$1" 2>/dev/null | awk '{print $3}' | head -1)
    case "$stat" in PEND|RUN|PSUSP|USUSP|SSUSP|WAIT|PROV) return 0 ;; *) return 1 ;; esac
}

seg_mark_start() {
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    [ -n "${LSB_JOBID:-}" ] || return 0
    echo "$1 $(date '+%F %T')" > "$EXP_DIR/${SEG_MARK_PREFIX}${LSB_JOBID}"
    # a killed or failed job: drop the marker but start nothing
    trap 'rm -f "$EXP_DIR/${SEG_MARK_PREFIX}${LSB_JOBID}"' EXIT
}

chain_downstream() {
    local lock="$EXP_DIR/.seg_chain.lock" waited=0 other id alive=""
    if ! is_true "${THEN_ANALYSIS:-true}"; then
        echo "THEN_ANALYSIS=false: not starting the analysis. Submit it with:"
        echo "    cd '$EXP_DIR' && bsub < $PIPELINE_DIR/hpc/run_downstream.lsf"
        return 0
    fi
    if [ "${DRY_RUN:-0}" = "1" ] || [ -z "${LSB_JOBID:-}" ]; then
        echo "(not an LSF job: the analysis is not submitted automatically)"
        return 0
    fi
    until mkdir "$lock" 2>/dev/null; do
        sleep 2; waited=$((waited + 2))
        [ "$waited" -ge 120 ] && { echo "stale $lock: taking it"; rm -rf "$lock"; }
    done
    rm -f "$EXP_DIR/${SEG_MARK_PREFIX}${LSB_JOBID}"
    trap - EXIT
    for other in "$EXP_DIR/${SEG_MARK_PREFIX}"*; do
        [ -e "$other" ] || continue
        id=${other##*"${SEG_MARK_PREFIX}"}
        if _job_alive "$id"; then alive="$alive $id"; else rm -f "$other"; fi
    done
    if [ -n "$alive" ]; then
        echo "Segmentation job(s)$alive of this experiment are still running: the last one"
        echo "to finish starts the analysis."
    else
        echo "Submitting the analysis (run_downstream.lsf) ..."
        (cd "$EXP_DIR" && bsub < "$PIPELINE_DIR/hpc/run_downstream.lsf") ||
            echo "WARNING: could not submit the analysis from this node. Submit it with:
    cd '$EXP_DIR' && bsub < $PIPELINE_DIR/hpc/run_downstream.lsf"
    fi
    rmdir "$lock" 2>/dev/null || true
}

# --- the two job types -------------------------------------------------------

run_segmentation() {
    local mode=$1            # 2d | 3d
    local ds_setting steps force_steps datasets
    if [ "$mode" = "3d" ]; then ds_setting="${DATASETS_3D:-}"; else ds_setting="${DATASETS_2D:-}"; fi

    pl_setup gpu
    seg_mark_start "$mode"
    verify_gpu

    to_array steps "${SEG_STEPS:-load segment extract}"
    to_array force_steps "${SEG_FORCE_STEPS:-}"
    to_array datasets "$ds_setting"

    local args=(process ${COMMON[@]+"${COMMON[@]}"} --seg-mode "$mode" --steps ${steps[@]+"${steps[@]}"} ${SELECT[@]+"${SELECT[@]}"})
    [ ${#datasets[@]} -gt 0 ] && args+=(--datasets ${datasets[@]+"${datasets[@]}"})
    is_true "${FORCE:-false}" && args+=(--force)
    [ ${#force_steps[@]} -gt 0 ] && args+=(--force-step ${force_steps[@]+"${force_steps[@]}"})

    echo ""
    echo "=== Segmentation ($mode): ${steps[*]} ==="
    run ${PY[@]+"${PY[@]}"} ${args[@]+"${args[@]}"}
    echo ""
    echo "Segmentation finished $(date). Per-image results are next to each image."
    chain_downstream
}

run_downstream_job() {
    local datasets steps force_steps
    pl_setup cpu

    to_array datasets "${DOWNSTREAM_DATASETS:-}"
    to_array steps "${DOWNSTREAM_STEPS:-}"
    to_array force_steps "${DOWNSTREAM_FORCE_STEPS:-}"
    local ds_args=() seg_args=()
    [ -n "${DOWNSTREAM_SEG_MODE:-}" ] && seg_args=(--seg-mode "$DOWNSTREAM_SEG_MODE")
    [ ${#datasets[@]} -gt 0 ] && ds_args=(--datasets ${datasets[@]+"${datasets[@]}"})

    if is_true "${RUN_CHECK:-true}"; then
        echo ""; echo "=== check ==="
        run ${PY[@]+"${PY[@]}"} check ${COMMON[@]+"${COMMON[@]}"} ${ds_args[@]+"${ds_args[@]}"} ${SELECT[@]+"${SELECT[@]}"} \
            || die "check found problems (see above): fix the plate map / config and resubmit."
    fi
    if is_true "${RUN_MANIFEST:-true}"; then
        echo ""; echo "=== manifest ==="
        run ${PY[@]+"${PY[@]}"} manifest ${COMMON[@]+"${COMMON[@]}"} ${ds_args[@]+"${ds_args[@]}"}
    fi

    local args=(downstream ${COMMON[@]+"${COMMON[@]}"} ${ds_args[@]+"${ds_args[@]}"} ${seg_args[@]+"${seg_args[@]}"} ${SELECT[@]+"${SELECT[@]}"})
    [ ${#steps[@]} -gt 0 ] && args+=(--steps ${steps[@]+"${steps[@]}"})
    is_true "${DOWNSTREAM_FORCE:-false}" && args+=(--force)
    [ ${#force_steps[@]} -gt 0 ] && args+=(--force-step ${force_steps[@]+"${force_steps[@]}"})
    [ -n "${DOWNSTREAM_OUT:-}" ] && args+=(--out "$DOWNSTREAM_OUT")

    echo ""; echo "=== downstream ==="
    run ${PY[@]+"${PY[@]}"} ${args[@]+"${args[@]}"}
    echo ""
    echo "Downstream finished $(date). Open report.html in the analysis folder"
    echo "(its name is in the 'downstream:' lines above, e.g. analysis_2d)."
}
