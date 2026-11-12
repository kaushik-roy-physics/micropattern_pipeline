"""
Tests for running on a subset of images (``--seq`` / ``--wells`` / ``input.select``).

The rules these lock in: no selection means every image; Seq numbers are the
ones written in the filename; wells resolve through the plate map in predefined
mode; and a typo fails loudly instead of silently selecting nothing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selection import Selection, parse_index_spec, parse_well_spec  # noqa: E402


def make_folder(names, plate_map: str | None = None) -> Path:
    d = Path(tempfile.mkdtemp()) / "acq"
    d.mkdir()
    for n in names:
        (d / n).touch()
    if plate_map:
        (d / "plate_map.yaml").write_text(plate_map)
    return d


SEQ_FILES = [f"Point{i:04d}_ChannelDAPI,FITC_Seq{i:04d}.nd2" for i in range(12)]


def test_spec_parsing():
    assert parse_index_spec(None) is None
    assert parse_index_spec("all") is None
    assert parse_index_spec("") is None
    assert parse_index_spec("0-3, 7") == {0, 1, 2, 3, 7}
    assert parse_index_spec(["0-1", 5]) == {0, 1, 5}
    assert parse_index_spec("2 4;6") == {2, 4, 6}
    nums, labels = parse_well_spec("1-2, C1-C3, d05")
    assert nums == {1, 2} and labels == {"C1", "C2", "C3", "D5"}
    for bad in ("3-x", "abc", "5-2", "C1-D3"):
        try:
            (parse_well_spec if bad[0].isalpha() else parse_index_spec)(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")
    print("ok  selection specs parse ranges, lists, labels and reject typos")


def test_no_selection_keeps_everything():
    d = make_folder(SEQ_FILES)
    paths = sorted(d.iterdir())
    sel = Selection.from_config({})
    assert not sel.active
    assert sel.filter_paths(paths, d) == paths
    print("ok  no selection → every image")


def test_seq_selection_uses_filename_numbers():
    d = make_folder(SEQ_FILES)
    paths = sorted(d.iterdir())
    kept = Selection(seq="1-3, 10").filter_paths(paths, d)
    assert [p.name.split("_Seq")[1] for p in kept] == ["0001.nd2", "0002.nd2",
                                                       "0003.nd2", "0010.nd2"]
    print("ok  --seq selects by the Seq number in the filename")


def test_cli_overrides_config():
    cfg = {"input": {"select": {"seq": "0-5"}}}
    assert Selection.from_config(cfg).seq == set(range(6))
    assert Selection.from_config(cfg, seq=["7"]).seq == {7}
    print("ok  command-line selection overrides input.select")


def test_wells_in_predefined_mode_use_plate_map():
    pm = "plate: 18well\nmode: predefined\npoints_per_well: 4\nstains: {default: {0: DAPI}}\n" \
         "wells: {1-3: {treatment: X}}\n"
    d = make_folder(SEQ_FILES, pm)
    paths = sorted(p for p in d.iterdir() if p.suffix == ".nd2")
    kept = Selection(wells="2").filter_paths(paths, d)
    assert len(kept) == 4 and "Seq0004" in kept[0].name and "Seq0007" in kept[-1].name
    # Without a plate map the well of a Seq file is unknowable → refuse.
    d2 = make_folder(SEQ_FILES)
    try:
        Selection(wells="2").filter_paths(sorted(d2.iterdir()), d2)
    except ValueError:
        pass
    else:
        raise AssertionError("predefined well selection without a plate map must fail")
    print("ok  --wells in predefined mode resolves through the plate map")


def test_wells_generate_and_spinning_disk():
    gen = make_folder([f"Well{w:02d}_Point{p:04d}.nd2" for w in (1, 2, 3) for p in range(2)])
    kept = Selection(wells="1, 3").filter_paths(sorted(gen.iterdir()), gen)
    assert len(kept) == 4 and not any("Well02" in p.name for p in kept)
    sd = make_folder([f"Well{lab}_Point0000.nd2" for lab in ("C01", "C02", "C10", "D01")])
    kept = Selection(wells="C1-C2").filter_paths(sorted(sd.iterdir()), sd)
    assert sorted(p.name[:7] for p in kept) == ["WellC01", "WellC02"]
    print("ok  --wells works on Well07 and WellC10 filenames")


def test_seq_and_wells_combine_with_and():
    pm = "plate: 18well\nmode: predefined\npoints_per_well: 4\nstains: {default: {0: DAPI}}\n" \
         "wells: {1-3: {treatment: X}}\n"
    d = make_folder(SEQ_FILES, pm)
    paths = sorted(p for p in d.iterdir() if p.suffix == ".nd2")
    kept = Selection(seq="0-5", wells="2").filter_paths(paths, d)
    assert [p.name[-11:-4] for p in kept] == ["Seq0004", "Seq0005"]
    print("ok  seq and wells together must both match")


def test_manifest_filter():
    man = pd.DataFrame({"image_path": [f"a_Seq{i:04d}.nd2" for i in range(6)],
                        "seq_index": range(6), "well": [1, 1, 2, 2, 3, 3]})
    assert list(Selection(seq="1-2").filter_manifest(man)["seq_index"]) == [1, 2]
    assert list(Selection(wells="3").filter_manifest(man)["seq_index"]) == [4, 5]
    assert len(Selection().filter_manifest(man)) == 6
    print("ok  downstream manifest rows are filtered the same way")


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
