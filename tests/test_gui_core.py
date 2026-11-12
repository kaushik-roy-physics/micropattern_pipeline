"""
Tests for the GUI's file writers (gui/core.py): no Streamlit needed.

The rule these lock in: whatever the GUI writes, the pipeline reads back as
exactly what was entered. Plate maps go through the real PlateMap loader,
configs through yaml + the template, and the HPC env file through bash.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gui"))

import core  # noqa: E402
from experiment.plate_map import PlateMap  # noqa: E402


def folder(names) -> Path:
    d = Path(tempfile.mkdtemp()) / "acq"
    d.mkdir()
    for n in names:
        (d / n).touch()
    return d


def load_written(state) -> PlateMap:
    d = Path(tempfile.mkdtemp())
    core.write_text(d / "plate_map.yaml", core.render_plate_map(state))
    return PlateMap(d / "plate_map.yaml")


def test_coerce_keeps_numbers_numbers():
    assert core.coerce("10") == 10 and core.coerce("0.5") == 0.5 and core.coerce("1e-3") == 0.001
    assert core.coerce("DrugX") == "DrugX" and core.coerce("") is None and core.coerce(float("nan")) is None
    assert core.coerce(3.0) == 3 and core.coerce("C10") == "C10"
    print("ok  table cells become numbers where they look like numbers")


def test_predefined_equal_roundtrip():
    d = folder([f"P_Seq{i:04d}.nd2" for i in range(12)])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    st["mapping"] = "equal"
    st["points_per_well"] = 4
    st["wells"] = [{"well": "1", "treatment": "Ctrl", "class": "control"},
                   {"well": "2", "treatment": "X", "dose_uM": "10"},
                   {"well": "3", "treatment": "X", "dose_uM": "10"}]
    st["columns"] = ["treatment", "dose_uM"]
    pm = load_written(st)
    assert pm.well_for_seq(5) == 2 and pm.conditions_for_well(3) == {"treatment": "X", "dose_uM": 10}
    assert "2-3:" in core.render_plate_map(st)                    # identical wells compressed
    rep = core.validate_plate_map(st, scan)
    assert rep.ok and not rep.errors, rep.errors
    back = core.plate_map_from_yaml(yaml.safe_load(core.render_plate_map(st)), scan)
    assert core.render_plate_map(back) == core.render_plate_map(st)
    print("ok  predefined plate map: written, read by PlateMap, re-opened identically")


def test_validation_catches_unmapped_and_missing():
    d = folder([f"P_Seq{i:04d}.nd2" for i in range(12)])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    st["mapping"] = "equal"
    st["points_per_well"] = 4
    st["wells"] = [{"well": "1", "treatment": "Ctrl"}]           # wells 2, 3 missing
    rep = core.validate_plate_map(st, scan)
    assert not rep.ok and any("no row" in e for e in rep.errors)
    assert any("negative control" in w for w in rep.warnings)
    st["mapping"], st["variable"] = "variable", [{"well": 1, "count": 4}]
    rep = core.validate_plate_map(st, scan)
    assert any("map to no well" in e for e in rep.errors)
    print("ok  validation reports unmapped images, wells without conditions, no control")


def test_new_predefined_map_starts_empty():
    d = folder([f"P_Seq{i:04d}.nd2" for i in range(12)])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    # Seq filenames carry only the acquisition order: no guessed well list, and the
    # image count per well is typed in the wells table (variable mapping).
    assert st["mode"] == "predefined" and st["mapping"] == "variable" and st["wells"] == []
    st["variable"] = [{"well": 1, "count": 5}, {"well": 2, "count": 7}]
    st["wells"] = [{"well": "1", "treatment": "a"}, {"well": "2", "treatment": "b"}]
    assert load_written(st).well_for_seq(5) == 2
    print("ok  a new predefined plate map starts with an empty wells table")


def test_variable_and_ranges():
    d = folder([f"P_Seq{i:04d}.nd2" for i in range(10)])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    st["mapping"], st["variable"] = "variable", [{"well": 1, "count": 3}, {"well": 2, "count": 7}]
    st["wells"] = [{"well": "1", "treatment": "a"}, {"well": "2", "treatment": "b"}]
    assert load_written(st).well_for_seq(3) == 2
    st["mapping"], st["ranges"] = "ranges", [{"well": 1, "first": 0, "last": 4},
                                            {"well": 2, "first": 5, "last": 9}]
    pm = load_written(st)
    assert pm.well_for_seq(4) == 1 and pm.well_for_seq(5) == 2
    assert core.validate_plate_map(st, scan).ok
    print("ok  variable counts and explicit Seq ranges map correctly")


def test_spinning_disk_labels():
    d = folder([f"Well{l}_Point0000.nd2" for l in ("C01", "C02", "C03", "D10")])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    assert st["mode"] == "spinning_disk" and st["plate"] == "96well"
    st["wells"] = [{"well": "C1", "treatment": "a"}, {"well": "c2", "treatment": "a"},
                   {"well": "C3", "treatment": "b"}, {"well": "D10", "treatment": "a"}]
    text = core.render_plate_map(st)
    assert "C1-C2:" in text and "D10:" in text
    assert core.validate_plate_map(st, scan).ok
    back = core.plate_map_from_yaml(yaml.safe_load(text), scan)
    assert [r["well"] for r in back["wells"]] == ["C1", "C2", "C3", "D10"]
    print("ok  spinning-disk labels compress by row and round-trip")


def test_generate_and_stains():
    d = folder([f"Well0{w}_Point0000.nd2" for w in (1, 2, 5)])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    assert st["mode"] == "generate" and st["plate"] == "8well"
    st["stains"] = {"default": {0: "DAPI", 1: "SOX2"}, "panelB": {0: "DAPI", 1: "BRA"}}
    st["wells"] = [{"well": "1", "treatment": "m", "class": "pluripotent", "positive_for": "SOX2"},
                   {"well": "2", "treatment": "b", "control_for": "SOX2"},
                   {"well": "5", "treatment": "b", "stain": "panelB"}]
    pm = load_written(st)
    assert pm.positive_for_well(1) == "SOX2" and pm.control_for_well(2) == "SOX2"
    assert pm.stains_for_well(5)[1] == "BRA"
    st["wells"][2]["stain"] = "nope"
    assert any("not defined" in e for e in core.validate_plate_map(st, scan).errors)
    print("ok  generate mode, stain sets and per-marker control keys")


def test_stain_sweeps():
    d = folder([f"P_Seq{i:04d}.nd2" for i in range(8)])
    scan = core.scan_folder(d, d.parent)
    st = core.new_plate_map(scan)
    st["stains"] = {"default": {0: "DAPI", 1: "A"}, "s2": {0: "DAPI", 1: "B"}}
    st["sweeps"] = [{"stain": "default", "first": 0, "last": 3, "points_per_well": 2},
                    {"stain": "s2", "first": 4, "last": 7, "points_per_well": 2}]
    st["wells"] = [{"well": k} | {"treatment": "t"} for k in core.default_well_keys(st, scan)]
    pm = load_written(st)
    assert pm.well_for_seq(4) == 1 and pm.resolve_stain(1, 5)[0] == "s2"
    assert core.validate_plate_map(st, scan).ok
    print("ok  stain sweeps: blocks restart at well 1")


def test_config_template_roundtrip():
    t = core.ConfigTemplate()
    vals = t.merged({"datasets": [{"name": "a", "directory": "./a",
                                   "segmentation": {"mode": "3d"}}],
                     "downstream": {"markers": {"ISL1": {"scope": "plate"}},
                                    "my_extra": [1, 2]},
                     "defaults": {"input": {"select": {"seq": "0-9, 15", "wells": None}}}})
    text = t.render(vals)
    assert yaml.safe_load(text) == vals
    assert "# the control's P99" in text or "P99" in text           # comments survive
    assert not text.startswith("# config.yaml: written by the pipeline GUI (plain")
    assert t.help("downstream", "thresholds", "fpr")
    print("ok  config.yaml keeps the template's comments and reads back exactly")


def test_env_file_roundtrip_through_bash():
    vals = core.env_defaults()
    vals.update(SEQ="0-9, 15", DATASETS_2D="a b", PIPELINE_DIR="/data/x y/pipe")
    text = core.render_env(vals)
    assert core.parse_env(text)["SEQ"] == "0-9, 15"
    d = Path(tempfile.mkdtemp())
    core.write_text(d / "hpc_job.env", text)
    assert b"\r" not in (d / "hpc_job.env").read_bytes()
    out = subprocess.run(["bash", "-c", f'source "{d}/hpc_job.env"; echo "$SEQ|$DATASETS_2D|$PIPELINE_DIR"'],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "0-9, 15|a b|/data/x y/pipe", out
    print("ok  hpc_job.env is valid bash with Unix line endings")


def test_hpc_path_translation():
    assert core.to_hpc_path("/mnt/share/KR/exp_033", "/mnt/share", "/data/Warmflash") == \
        "/data/Warmflash/KR/exp_033"
    assert core.to_hpc_path("/mnt/share", "/mnt/share", "/data/Warmflash/") == "/data/Warmflash"
    assert core.guess_share_root(Path("/run/gvfs/smb-share:server=x,share=warmflash/KR/e")) \
        .endswith("share=warmflash")
    print("ok  laptop paths on the share translate to HPC paths")


def test_find_dataset_folders():
    exp = Path(tempfile.mkdtemp())
    for sub in ("zmax", "zstack"):
        (exp / sub).mkdir()
        (exp / sub / f"{sub}_Seq0000.nd2").touch()
    (exp / "analysis").mkdir()
    (exp / "analysis" / "fig.tif").touch()
    found = core.find_dataset_folders(exp)
    assert [f.rel for f in found] == ["./zmax", "./zstack"]
    print("ok  dataset folders found, analysis output ignored")


if __name__ == "__main__":
    failures = 0
    for test_name, func in sorted(globals().items()):
        if test_name.startswith("test_") and callable(func):
            try:
                func()
            except Exception as exc:                    # noqa: BLE001
                failures += 1
                print(f"FAIL {test_name}: {exc!r}")
    print("\n" + ("all tests passed" if not failures else f"{failures} test(s) failed"))
    sys.exit(1 if failures else 0)
