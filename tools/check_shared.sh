#!/usr/bin/env bash
# Check that the files micropattern_pipeline and analysis_pipeline_v2 share are
# still identical. Run it before committing a change to any file listed below,
# then make the same change (and commit) in the other repository.
#
#   tools/check_shared.sh                 # other repo assumed next to this one
#   tools/check_shared.sh /path/to/other  # or say where it is
#   tools/check_shared.sh --diff          # also print the differences
#
# Exit status 0 when every shared file matches, 1 otherwise.
# This script is identical in both repositories.
set -u

SHARED_FILES="
src/pipeline_config.py
src/provenance.py
src/seg_status.py
src/selection.py
src/image_processing/__init__.py
src/image_processing/directory_walker.py
src/image_processing/image_loader.py
src/image_processing/intensity_extractor.py
src/image_processing/io.py
src/image_processing/segmentation.py
src/image_processing/utils.py
src/experiment/__init__.py
src/experiment/manifest.py
src/experiment/parser.py
src/experiment/plate.py
src/experiment/plate_map.py
src/experiment/template.py
src/downstream/controls.py
src/downstream/data.py
src/downstream/registry.py
src/downstream/runner.py
src/downstream/scope.py
src/downstream/style.py
src/downstream/thresholds.py
src/downstream/plots/distributions.py
src/downstream/signaling/__init__.py
src/downstream/signaling/catalog.py
src/downstream/signaling/measure.py
src/downstream/signaling/readouts.py
hpc/check_env.py
tests/test_gui_core.py
tests/test_provenance.py
tests/test_selection.py
tools/check_shared.sh
.gitattributes
.githooks/pre-commit
requirements.txt
requirements-gui.txt
requirements-test.txt
run_local_2dseg.sh
run_local_downstream.sh
"

here="$(cd "$(dirname "$0")/.." && pwd)"
show_diff=0
other=""
for arg in "$@"; do
    case "$arg" in
        --diff) show_diff=1 ;;
        *) other="$arg" ;;
    esac
done
if [ -z "$other" ]; then
    case "$(basename "$here")" in
        micropattern_pipeline*) other="$here/../analysis_pipeline_v2" ;;
        *) other="$here/../micropattern_pipeline" ;;
    esac
fi
if [ ! -d "$other" ]; then
    echo "other repository not found: $other (pass its path)" >&2
    exit 2
fi

n_diff=0
for f in $SHARED_FILES; do
    if [ ! -f "$here/$f" ] || [ ! -f "$other/$f" ]; then
        echo "MISSING  $f"
        n_diff=$((n_diff + 1))
    elif ! cmp -s "$here/$f" "$other/$f"; then
        echo "DIFFERS  $f"
        n_diff=$((n_diff + 1))
        [ "$show_diff" = 1 ] && diff -u "$other/$f" "$here/$f" | sed 's/^/    /'
    fi
done

if [ "$n_diff" = 0 ]; then
    echo "all shared files match $(cd "$other" && pwd)"
    exit 0
fi
echo "$n_diff shared file(s) differ from $(cd "$other" && pwd)"
exit 1
