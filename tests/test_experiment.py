"""
Tests for the annotation layer: plate geometry, Seq→well mapping, stain resolution.

The rule these lock in: **nothing is inferred from a filename except the well
number or Seq index the microscope wrote there.** Which markers an image carries
comes from ``plate_map.yaml`` and nowhere else.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiment.plate_map import PlateMap  # noqa: E402


def write_map(text: str) -> Path:
    """Write a plate map into a throwaway directory and return its path."""
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "plate_map.yaml"
    path.write_text(text)
    return path


BASIC = """
plate: 18well
plate_id: test
mode: predefined
points_per_well: 5
stains:
  default: {0: DAPI, 1: GM130, 2: pS6}
  setB:    {0: DAPI, 1: Tom20, 2: pS6}
wells:
  1:   {treatment: Control}
  2-4: {treatment: DrugX, stain: setB}
"""


def test_seq_to_well_and_conditions():
    pm = PlateMap(write_map(BASIC))
    assert pm.well_for_seq(0) == 1
    assert pm.well_for_seq(4) == 1
    assert pm.well_for_seq(5) == 2
    assert pm.well_for_seq(14) == 3
    assert pm.conditions_for_well(2) == {"treatment": "DrugX"}   # 'stain' is not a condition
    print("ok  Seq→well mapping and condition columns")


def test_stain_from_well_key():
    pm = PlateMap(write_map(BASIC))
    name, channels = pm.resolve_stain(well=1, seq_index=0)
    assert name == "default" and channels[1] == "GM130"
    name, channels = pm.resolve_stain(well=3, seq_index=12)
    assert name == "setB" and channels[1] == "Tom20"
    print("ok  stain set comes from the well's 'stain:' key")


def test_stain_id_is_an_alias_for_stain():
    pm = PlateMap(write_map(BASIC.replace("stain: setB", "stain_id: setB")))
    name, channels = pm.resolve_stain(well=3, seq_index=12)
    assert name == "setB" and channels[1] == "Tom20"
    assert pm.conditions_for_well(2) == {"treatment": "DrugX"}   # not a condition either
    clash = BASIC.replace("stain: setB", "stain: setB, stain_id: default")
    try:
        PlateMap(write_map(clash))
    except ValueError as exc:
        assert "stain_id" in str(exc)
    else:
        raise AssertionError("conflicting 'stain' and 'stain_id' were accepted")
    print("ok  'stain_id:' (old plate maps) is read as 'stain:'; a conflicting pair is an error")


SWEEPS = """
plate: 18well
plate_id: test
mode: predefined
stains:
  stain1: {0: DAPI, 1: ISL1,  2: NKX2.5}
  stain2: {0: DAPI, 1: HCN4,  2: MEF2C}
stain_sweeps:
  stain1: {seq_range: [0, 17],  points_per_well: 6}
  stain2: {seq_range: [18, 35], points_per_well: 6}
wells:
  1-3: {treatment: Control}
"""


def test_stain_sweeps_are_local_to_their_block():
    """Each sweep re-visits the plate, so its Seq indices restart at its own start."""
    pm = PlateMap(write_map(SWEEPS))

    assert pm.sweep_for_seq(0) == "stain1"
    assert pm.sweep_for_seq(17) == "stain1"
    assert pm.sweep_for_seq(18) == "stain2"

    # stain1: Seq 0-5 → well 1, 6-11 → well 2, 12-17 → well 3
    assert [pm.well_for_seq(s) for s in (0, 6, 12, 17)] == [1, 2, 3, 3]
    # stain2 restarts: Seq 18-23 → well 1 (NOT well 4)
    assert [pm.well_for_seq(s) for s in (18, 24, 30, 35)] == [1, 2, 3, 3]

    name, channels = pm.resolve_stain(well=1, seq_index=18)
    assert name == "stain2" and channels[1] == "HCN4"
    name, channels = pm.resolve_stain(well=1, seq_index=0)
    assert name == "stain1" and channels[1] == "ISL1"
    print("ok  stain_sweeps: same well, two panels, indices local to each block")


def test_overlapping_sweeps_are_rejected():
    bad = SWEEPS.replace("seq_range: [18, 35]", "seq_range: [17, 35]")
    try:
        PlateMap(write_map(bad))
    except ValueError as exc:
        assert "overlap" in str(exc)
        print("ok  overlapping stain sweeps are refused")
        return
    raise AssertionError("overlapping sweeps should have raised")


def test_sweep_must_name_a_declared_stain():
    bad = SWEEPS.replace("  stain2: {seq_range: [18, 35]", "  typo: {seq_range: [18, 35]")
    try:
        PlateMap(write_map(bad))
    except ValueError as exc:
        assert "not" in str(exc) and "stains" in str(exc)
        print("ok  a sweep naming an undeclared stain set is refused")
        return
    raise AssertionError("undeclared stain set should have raised")


def test_no_marker_names_are_read_from_filenames():
    """A filename full of marker names must not influence the stain set."""
    pm = PlateMap(write_map(BASIC))
    name, channels = pm.resolve_stain(well=1, seq_index=0)
    # Well 1 says 'default'; a file called '..._isl1,nkx25,mef2c_Seq0000.nd2'
    # changes nothing, because filenames carry no marker meaning here.
    assert name == "default"
    assert channels == {0: "DAPI", 1: "GM130", 2: "pS6"}
    print("ok  marker names in a filename are ignored: the plate map decides")


def test_image_discovery_prunes_the_right_directories():
    """Only real inputs are discovered: exports and outputs are never ingested."""
    from experiment.manifest import _iter_images, PLATE_MAP_NAME
    from image_processing.directory_walker import OUTPUT_MARKER

    root = Path(tempfile.mkdtemp())
    (root / "img_a.nd2").touch()
    (root / "img_b.nd2").touch()

    # img_a's own result directory: recognized by name, no marker needed.
    (root / "img_a").mkdir()
    (root / "img_a" / "ch0.tif").touch()
    # A result directory that does carry the marker (belt and braces).
    marked = root / "something_else"
    marked.mkdir()
    (marked / OUTPUT_MARKER).touch()
    (marked / "stray.nd2").touch()
    # Exported images: must be excluded by name, not ingested as inputs.
    (root / "jpegs").mkdir()
    (root / "jpegs" / "export.nd2").touch()
    # A dot-directory.
    (root / ".cache").mkdir()
    (root / ".cache" / "hidden.nd2").touch()
    # A sub-plate with its own map: belongs to its own plate, not this one.
    sub = root / "plate2"
    sub.mkdir()
    (sub / PLATE_MAP_NAME).touch()
    (sub / "other.nd2").touch()

    found = sorted(p.name for p in _iter_images(root, {".nd2"}, exclude_dirs=["jpegs"]))
    assert found == ["img_a.nd2", "img_b.nd2"], found
    print("ok  discovery skips result dirs, exports, dot-dirs and sub-plates")


SPINNING_DISK = """
plate: 96well
plate_id: sd_test
mode: spinning_disk
stains:
  default: {0: DAPI, 1: SOX2, 2: BRA, 3: CDX2}
wells:
  A1:      {treatment: mTeSR, class: pluripotent, positive_for: [SOX2]}
  'C1-C3': {treatment: BMP50, control_for: [SOX2]}
  D12:     {treatment: BMP10}
"""


def test_spinning_disk_alphanumeric_wells():
    """Alphanumeric labels resolve to canonical row-major well numbers."""
    pm = PlateMap(write_map(SPINNING_DISK))
    assert pm.is_spinning_disk and pm.mode == "spinning_disk"
    assert (pm.rows, pm.cols) == (8, 12)
    # A1 -> 1; C1..C3 -> 25, 26, 27; D12 -> 48.
    assert sorted(pm.wells) == [1, 25, 26, 27, 48]
    assert pm.wells[26]["treatment"] == "BMP50"
    print("ok  spinning-disk labels and same-row ranges resolve to well numbers")


def test_reserved_keys_are_not_conditions():
    """control_for/positive_for must never leak into the condition columns."""
    pm = PlateMap(write_map(SPINNING_DISK))
    conds = pm.conditions_for_well(1)
    assert set(conds) == {"treatment", "class"}, conds
    assert pm.positive_for_well(1) == "SOX2"
    assert pm.control_for_well(1) == ""
    assert pm.control_for_well(25) == "SOX2"
    assert pm.declares_marker_controls
    assert pm.declared_control_markers() == {"SOX2"}
    print("ok  control_for/positive_for are reserved, not condition columns")


def test_control_declaration_normalisation():
    """A bare string, a list and 'all' all normalize to the stored form."""
    pm = PlateMap(write_map("""
plate: 8well
mode: generate
stains: {default: {0: DAPI, 1: A}}
wells:
  1: {treatment: X, control_for: all}
  2: {treatment: Y, control_for: SOX2}
  3: {treatment: Z, control_for: [SOX2, BRA]}
"""))
    assert pm.control_for_well(1) == "all"
    assert pm.control_for_well(2) == "SOX2"
    assert pm.control_for_well(3) == "SOX2;BRA"
    assert pm.control_for_well(99) == ""
    print("ok  control_for accepts a bare name, a list and 'all'")


PREDEFINED_PARTIAL = """
plate: 18well
plate_id: pitx2
mode: predefined
points_per_well: 2
stains: {default: {0: DAPI, 1: ISL1}}
wells:
  1: {treatment: A}
  2: {treatment: B, class: control}
  3: {treatment: C}
"""


def test_predefined_wells_are_acquisition_order():
    """Seq filenames carry no plate position, so wells are numbered by visit order."""
    pm = PlateMap(write_map(PREDEFINED_PARTIAL))
    assert pm.well_numbering == "sequential"
    assert not pm.has_plate_positions

    # …unless the map says the keys really are plate wells.
    opted_in = PlateMap(write_map(PREDEFINED_PARTIAL + "well_numbering: plate\n"))
    assert opted_in.has_plate_positions

    # Filename-derived modes always know where the well was.
    assert PlateMap(write_map(SPINNING_DISK)).has_plate_positions
    assert PlateMap(write_map(BASIC.replace("mode: predefined",
                                            "mode: generate"))).has_plate_positions
    print("ok  predefined wells are sequential; other modes are plate positions")


def test_manifest_row_col_follow_well_numbering():
    """(row, col) is emitted only when the well number is a real plate coordinate."""
    from experiment import manifest as mf

    tmp = Path(tempfile.mkdtemp())
    pm_path = tmp / "plate_map.yaml"
    pm_path.write_text(PREDEFINED_PARTIAL)
    img = tmp / "Point0002_Seq0002.nd2"
    img.touch()

    # points_per_well: 2 → Seq 2 belongs to the 2nd well imaged.
    rec = mf._build_row(img, PlateMap(pm_path), tmp)
    assert rec["well"] == 2 and rec["treatment"] == "B"
    assert "row" not in rec and "col" not in rec, "a fabricated plate position leaked in"

    pm_path.write_text(PREDEFINED_PARTIAL + "well_numbering: plate\n")
    rec = mf._build_row(img, PlateMap(pm_path), tmp)
    assert (rec["row"], rec["col"]) == (1, 2)   # well 2 of an 18-well (3×6) plate
    print("ok  row/col appear only under plate numbering")


def test_template_detects_mode_and_wells():
    """The generator reads mode, wells and counts off real filenames."""
    from experiment import template

    tmp = Path(tempfile.mkdtemp()) / "zmax_stain1"
    tmp.mkdir()
    for label in ("C10", "C11", "D02"):
        for point in range(3):
            (tmp / f"Well{label}_Point000{point}_ChannelDAPI,FITC_Seq000{point}.nd2").touch()
    result = template.scan(tmp)
    assert result.mode == "spinning_disk", result.mode
    assert result.labels == ["C10", "C11", "D2"], result.labels   # D02 normalized
    assert result.points_per_well["C10"] == 3

    path = template.write_template(tmp, {})
    assert path is not None and path.exists()
    # Round-trip: what it wrote must parse, and must not be overwritten silently.
    pm = PlateMap(path)
    assert pm.is_spinning_disk and sorted(pm.wells) == [34, 35, 38]
    assert template.write_template(tmp, {}) is None
    print("ok  generator detects spinning-disk mode, wells and per-well counts")


def test_template_detects_the_other_two_modes():
    from experiment import template

    gen = Path(tempfile.mkdtemp()) / "acq"
    gen.mkdir()
    for well in (1, 2):
        for point in range(2):
            (gen / f"Well0{well}_Point000{point}.nd2").touch()
    assert template.scan(gen).mode == "generate"
    assert template.scan(gen).wells == [1, 2]

    pre = Path(tempfile.mkdtemp()) / "acq"
    pre.mkdir()
    for seq in range(6):
        (pre / f"Point000{seq}_Seq000{seq}.nd2").touch()
    scanned = template.scan(pre)
    assert scanned.mode == "predefined"
    assert scanned.seq_indices == [0, 1, 2, 3, 4, 5]
    print("ok  generator distinguishes generate / predefined from filenames")


if __name__ == "__main__":
    failures = 0
    for test_name, func in sorted(globals().items()):
        if test_name.startswith("test_") and callable(func):
            try:
                func()
            except Exception as exc:                    # noqa: BLE001
                failures += 1
                print(f"FAIL {test_name}: {exc}")
    print("\n" + ("all tests passed" if not failures else f"{failures} test(s) failed"))
    sys.exit(1 if failures else 0)
