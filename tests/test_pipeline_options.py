"""
The 2026-11 options: leaving markers out by name, skipping optional analyses,
choosing 2-D / 3-D results (and naming the analysis folder after them), stale
per-image checkpoints, the comment-preserving GUI writers and the HPC
one-off overrides. Synthetic data only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT))

import synthetic as syn  # noqa: E402


def _run(cfg: Path, *extra) -> int:
    import main as pipeline_main
    args = ["--config", str(cfg), "--log-level", "WARNING"]
    assert pipeline_main.main(["manifest"] + args) == 0
    code = pipeline_main.main(["downstream"] + args + list(extra))
    logging.getLogger().handlers.clear()
    return code


def _set_mode(data: Path, mode: str) -> None:
    """Pretend the segmentation of every image in *data* was made in *mode*."""
    for res in data.iterdir():
        if res.is_dir() and (res / "nuclear_masks.tif").exists():
            (res / "segmentation_metadata.json").write_text(json.dumps({"mode": mode}))


def _edit(cfg: Path, **down) -> None:
    d = yaml.safe_load(cfg.read_text())
    d["downstream"].update(down)
    cfg.write_text(yaml.safe_dump(d, sort_keys=False))


# ---------------------------------------------------------------------------
# 1. Markers left out by name
# ---------------------------------------------------------------------------

def test_exclude_marker_by_name_keeps_dapi_out():
    root = Path(tempfile.mkdtemp())
    cfg = syn.make_experiment(root, colonies_per_well=3,
                              downstream={"results_dir": "auto", "exclude_markers": ["FLAT"],
                                          "skip_steps": ["coexpression", "multimarker",
                                                         "boundary", "effects"]})
    assert _run(cfg) == 0
    inten = pd.read_parquet(root / "analysis_2d" / "intensities.parquet")
    markers = set(inten["marker"].astype(str))
    assert markers == {"EDGE", "CENTER"}, markers          # FLAT out, DAPI still out
    print("ok  exclude_markers: FLAT left out by name, DAPI stays out without being listed")


# ---------------------------------------------------------------------------
# 2. Optional analyses
# ---------------------------------------------------------------------------

def test_skip_steps_run_less_and_set_aside_old_outputs():
    from downstream import steps
    try:
        steps.resolve_skips(["qc"])
        raise AssertionError("skipping a core step must fail")
    except ValueError:
        pass
    assert steps.resolve_skips(["boundary", "compare"]) == {"boundary", "compare", "effects"}

    root = Path(tempfile.mkdtemp())
    cfg = syn.make_experiment(root, colonies_per_well=3, downstream={"results_dir": "./out"})
    assert _run(cfg) == 0
    out = root / "out"
    assert (out / "tables" / "boundary_colonies.csv").exists()
    assert any((out / "figures" / "coexpression").rglob("*.png"))
    _edit(cfg, skip_steps=["boundary", "coexpression"])
    assert _run(cfg) == 0
    assert not (out / "tables" / "boundary_colonies.csv").exists()
    assert not (out / "figures" / "coexpression").exists()
    assert (out / "_skipped" / "boundary" / "tables" / "boundary_colonies.csv").exists()
    assert not (out / ".checkpoints" / "boundary.json").exists()
    assert list((out / "tables").glob("effects_*.csv"))           # still runs, on compare
    html = (out / "report.html").read_text()
    assert "Boundaries and peaks" not in html and "Co-expression" not in html
    # Switching them back on runs them again.
    _edit(cfg, skip_steps=[])
    assert _run(cfg) == 0
    assert (out / "tables" / "boundary_colonies.csv").exists()
    print("ok  skip_steps: core protected, skipped outputs set aside, report clean, re-enable works")


# ---------------------------------------------------------------------------
# 3. Which segmentation results, and where they go
# ---------------------------------------------------------------------------

def test_auto_folder_and_actual_mode_label():
    root = Path(tempfile.mkdtemp())
    skip = ["coexpression", "multimarker", "boundary", "effects", "compare"]
    cfg = syn.make_experiment(root, colonies_per_well=3,
                              downstream={"results_dir": "auto", "skip_steps": skip})
    _set_mode(root / "zmax", "3d")             # masks say 3-D, the config says 2-D
    assert _run(cfg) == 0
    out = root / "analysis_3d"
    assert out.exists() and not (root / "analysis_2d").exists()
    cells = pd.read_parquet(out / "cells.parquet")
    assert set(cells["seg_mode"].astype(str)) == {"3d"}
    print("ok  results_dir auto → analysis_3d, and cells labeled by the masks' real mode")


def test_seg_mode_choice_picks_datasets():
    import seg_status
    from pipeline_config import load_config, resolve_datasets
    root = Path(tempfile.mkdtemp())
    cfg = syn.make_experiment(root, colonies_per_well=2)
    # a second, 3-D dataset: a copy of the first
    import shutil
    shutil.copytree(root / "zmax", root / "zst")
    _set_mode(root / "zmax", "2d")
    _set_mode(root / "zst", "3d")
    (root / "empty").mkdir()
    (root / "empty" / "Point0000_x.tif").touch()
    d = yaml.safe_load(cfg.read_text())
    d["datasets"] += [{"name": "zst", "directory": "./zst", "segmentation": {"mode": "3d"}},
                      {"name": "empty", "directory": "./empty"}]
    cfg.write_text(yaml.safe_dump(d, sort_keys=False))
    c = load_config(cfg)
    dss = resolve_datasets(c)
    stats = [seg_status.dataset_status(ds) for ds in dss]
    by = {s.name: s for s in stats}
    assert by["zmax"].modes == ["2d"] and by["zst"].modes == ["3d"] and by["empty"].n_ready == 0
    keep, modes = seg_status.select(dss, stats, "auto")
    assert [k.name for k in keep] == ["zmax", "zst"] and modes == ["2d", "3d"]
    keep, modes = seg_status.select(dss, stats, "3d")
    assert [k.name for k in keep] == ["zst"] and modes == ["3d"]
    assert seg_status.auto_dir_name(["3d", "2d"]) == "analysis_2d_3d"
    # mismatch is reported
    by["zst"].configured = "2d"
    assert by["zst"].mismatch
    print("ok  seg_mode: datasets without results dropped; 2d/3d/auto pick the right ones")


def test_stale_checkpoints_follow_new_masks():
    from downstream import colony
    from image_processing import intensity_extractor as ie
    root = Path(tempfile.mkdtemp())
    syn.make_experiment(root, colonies_per_well=1)
    res = next(p for p in (root / "zmax").iterdir() if p.is_dir())
    params = colony.settings({"colony": {"expected_diameter_um": 500}})
    colony.process_image(res, params, 0)
    digest = colony.settings_hash(params)
    assert colony.is_current(res, digest) and ie._extraction_current(res)
    time.sleep(1.1)
    os.utime(res / "nuclear_masks.tif")           # re-segmented
    assert not colony.is_current(res, digest)
    assert not ie._extraction_current(res)
    print("ok  new masks make the colony geometry and the intensities stale")


# ---------------------------------------------------------------------------
# 4. GUI writers keep what is on disk
# ---------------------------------------------------------------------------

def test_config_written_into_existing_file():
    import core
    t = core.ConfigTemplate()
    existing = t.render(t.merged({}))
    existing = existing.replace("    bin_um: 10                   # µm grid",
                                "    bin_um: 10                   # MY NOTE: matches nucleus size\n"
                                "    # MY BLOCK NOTE\n    # µm grid", 1)
    existing = existing.replace("  skip_steps: []", "  skip_steps: []  # SKIP NOTE")
    vals = t.merged(yaml.safe_load(existing))
    assert t.render_into(vals, existing) == existing           # untouched: byte for byte
    vals["downstream"]["profiles"]["n_norm_bins"] = 30
    out = t.render_into(vals, existing)
    assert "MY NOTE" in out and "MY BLOCK NOTE" in out and "SKIP NOTE" in out
    assert yaml.safe_load(out)["downstream"]["profiles"]["n_norm_bins"] == 30
    # an old file without the newer options gets them, with their explanation
    old = "\n".join(l for l in existing.splitlines() if not l.strip().startswith(
        ("seg_mode:", "skip_steps:")))
    out = t.render_into(t.merged(yaml.safe_load(old)), old)
    assert "seg_mode: auto" in out and "Optional analyses to leave out" in out
    assert core.changed_keys(yaml.safe_load(old), yaml.safe_load(out)) == [
        "downstream.seg_mode (new option)", "downstream.skip_steps (new option)"]
    print("ok  config.yaml saved into the existing file: comments kept, new options explained")


def test_plate_map_comments_survive_an_edit():
    import core
    text = ("plate: 18well\nplate_id: p1\nmode: predefined\npoints_per_well: 2\n"
            "stains:\n  default: {0: DAPI, 1: SOX2}\n"
            "wells:\n  # well 3 re-seeded on day 2\n  1-3: {treatment: A}   # same mix\n"
            "  4: {treatment: B, class: control}\n")
    raw = yaml.safe_load(text)
    st = core.plate_map_from_yaml(raw)
    assert core.plate_map_text(st, "x", text) == text
    st["wells"][-1]["treatment"] = "C"
    out = core.plate_map_text(st, "x", text)
    assert "# well 3 re-seeded on day 2" in out and "# same mix" in out
    assert yaml.safe_load(out)["wells"][4]["treatment"] == "C"
    assert core.parse_env('SEQ="${SEQ:-0-4}"\nX="a"') == {"SEQ": "0-4", "X": "a"}
    print("ok  plate_map.yaml: comments survive an edit; env defaults read through ${X:-…}")


# ---------------------------------------------------------------------------
# 5. HPC: one-off overrides and the run.sh dry run
# ---------------------------------------------------------------------------

def test_hpc_one_off_overrides_and_dry_run():
    exp = Path(tempfile.mkdtemp())
    (exp / "config.yaml").write_text("datasets: []\n")
    env = (ROOT / "templates" / "hpc_job.env").read_text()
    env = env.replace('PIPELINE_DIR="/data/Warmflash/KR/micropattern_pipeline"',
                      f'PIPELINE_DIR="{ROOT}"')
    env = env.replace('VENV_DIR="/scratch/$USER/micropattern_venv"',
                      f'VENV_DIR="{exp}/novenv"')
    (exp / "hpc_job.env").write_text(env)
    base = {k: v for k, v in os.environ.items() if not k.startswith("MP_")}
    run = lambda cmd, extra: subprocess.run(  # noqa: E731
        ["bash", "-c", cmd], cwd=exp, capture_output=True, text=True,
        env={**base, **extra}, timeout=120)
    r = run(f"bash {ROOT}/hpc/run_downstream.lsf",
            {"DRY_RUN": "1", "EXP_DIR": str(exp), "MP_SEQ": "0-2",
             "MP_DOWNSTREAM_FORCE_STEPS": "profiles"})
    assert r.returncode == 0, r.stderr
    assert "--seq 0-2" in r.stdout and "--force-step profiles" in r.stdout, r.stdout
    assert "One-off" in r.stdout
    r = run(f"bash {ROOT}/hpc/run.sh test", {"SEQ": "1-3"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("--seq 1-3") >= 2, r.stdout       # segmentation and analysis
    r = run(f"bash {ROOT}/hpc/run.sh bogus", {})
    assert r.returncode != 0
    print("ok  HPC: MP_* one-off settings win over hpc_job.env; run.sh test is a dry run")


if __name__ == "__main__":
    failures = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except Exception as exc:                          # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print("\n" + ("all tests passed" if not failures else f"{failures} test(s) failed"))
    sys.exit(1 if failures else 0)
