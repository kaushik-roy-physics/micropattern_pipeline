"""
Everything the GUI does that is not drawing widgets.

Kept free of Streamlit so it can be tested on its own (``tests/test_gui_core.py``)
and so the rules live in one place. The GUI never re-implements the pipeline:
folders are scanned with :mod:`experiment.template`, plate maps are validated by
loading them with :class:`experiment.plate_map.PlateMap`, and the final word is
always ``main.py check``. This module only translates between the tables a
person edits and the YAML files the pipeline reads.

Files written here:

* ``<dataset>/plate_map.yaml``: :func:`render_plate_map`
* ``<experiment>/config.yaml``: :class:`ConfigTemplate` fills the commented
  template in ``templates/config.yaml``, so every option keeps its explanation
* ``<experiment>/hpc_job.env``: :func:`render_env` fills ``templates/hpc_job.env``

All three are written with Unix line endings, whatever the laptop's OS, because
the HPC reads them.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = PIPELINE_DIR / "templates"
if str(PIPELINE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR / "src"))

from experiment import parser, plate  # noqa: E402
from experiment.manifest import PLATE_MAP_NAME  # noqa: E402
from pipeline_config import deep_merge  # noqa: E402

CONFIG_NAME = "config.yaml"
ENV_NAME = "hpc_job.env"
PREFS_PATH = Path.home() / ".micropattern_pipeline_gui.json"

PLATES = {"8well": (2, 4), "18well": (3, 6), "96well": (8, 12)}
MODES = ["predefined", "generate", "spinning_disk"]
MODE_HELP = {
    "predefined": "JOBS 'Predefined Points'; filenames end in Seq0042 (acquisition order)",
    "generate": "JOBS 'Generate Points'; filenames start with Well07 (well number)",
    "spinning_disk": "Spinning-disk confocal; filenames carry a well label like WellC10",
}
CONTROL_CLASSES = ["control", "pluripotent", "negative_control"]

# Plate-map well columns with a structural meaning (everything else is a condition).
CLASS_COL, STAIN_COL, POS_COL, CTRL_COL = "class", "stain", "positive_for", "control_for"
SPECIAL_COLS = [CLASS_COL, STAIN_COL, POS_COL, CTRL_COL]
WELL_COL, COUNT_COL = "well", "images"


# ===========================================================================
# Small helpers
# ===========================================================================

def coerce(value):
    """Table cell → YAML value. ``"10"`` → 10, ``"0.5"`` → 0.5, ``""`` → None.

    Numbers must stay numbers so that a dose or a timepoint can be a trend axis.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool, int, float)):
        return int(value) if isinstance(value, float) and value.is_integer() else value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
        return float(text)
    return text


def split_list(value) -> list[str]:
    """``"SOX2, BRA"`` / ``["SOX2"]`` → ``["SOX2", "BRA"]``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        if isinstance(value, float) and math.isnan(value):
            return []
        items = re.split(r"[,;]", str(value))
    return [str(i).strip() for i in items if str(i).strip()]


def flow(value) -> str:
    """One-line YAML for any value (``{a: 1}``, ``[1, 2]``, ``null``, ``'0-9, 15'``)."""
    text = yaml.safe_dump([value], default_flow_style=True, width=10**9,
                          sort_keys=False, allow_unicode=True).strip()
    return text[1:-1]


def write_text(path: Path, text: str) -> None:
    """Write with Unix newlines: these files are read on the HPC."""
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def well_label(well: int, cols: int) -> str:
    row, col = plate.well_to_rowcol(int(well), cols)
    return f"{chr(ord('A') + row - 1)}{col}"


def load_prefs() -> dict:
    try:
        return json.loads(PREFS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_prefs(prefs: dict) -> None:
    try:
        PREFS_PATH.write_text(json.dumps(prefs, indent=2))
    except OSError:
        pass


# ===========================================================================
# Experiment folder scanning
# ===========================================================================

@dataclass
class FolderScan:
    """What one acquisition folder holds, for the GUI's tables."""
    name: str
    path: Path
    rel: str
    n_images: int = 0
    mode: str | None = None
    seq: list[int] = field(default_factory=list)
    wells: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    per_well: dict = field(default_factory=dict)
    channels: list[str] = field(default_factory=list)
    has_plate_map: bool = False
    images: list[Path] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.n_images:
            return "no images"
        if self.mode == "predefined" and self.seq:
            return f"{self.n_images} images, Seq {min(self.seq)}-{max(self.seq)}"
        if self.labels:
            return f"{self.n_images} images in {len(self.labels)} wells"
        if self.wells:
            return f"{self.n_images} images in {len(self.wells)} wells"
        return f"{self.n_images} images"


def _image_exts(cfg_input: dict | None = None) -> set[str]:
    from experiment.manifest import _normalize_extensions
    return _normalize_extensions((cfg_input or {}).get("file_types"))


def _has_images_shallow(folder: Path, exts: set[str]) -> bool:
    from image_processing.io import get_extension
    try:
        with os.scandir(folder) as it:
            return any(e.is_file() and get_extension(Path(e.path)) in exts for e in it)
    except OSError:
        return False


def scan_folder(folder: Path, exp_dir: Path, cfg_input: dict | None = None) -> FolderScan:
    from experiment import template
    folder = Path(folder)
    rel = os.path.relpath(folder, exp_dir).replace("\\", "/")
    rel = "." if rel == "." else f"./{rel}"
    name = folder.name if rel != "." else Path(exp_dir).name
    out = FolderScan(name=name, path=folder, rel=rel,
                     has_plate_map=(folder / PLATE_MAP_NAME).exists())
    s = template.scan(folder, {"input": cfg_input or {}})
    out.images = s.images
    out.n_images = len(s.images)
    out.mode = s.mode
    out.seq = s.seq_indices
    out.wells = s.wells
    out.labels = s.labels
    out.per_well = s.points_per_well
    out.channels = s.channels
    return out


def find_dataset_folders(exp_dir: Path, cfg_input: dict | None = None) -> list[FolderScan]:
    """Acquisition folders in an experiment: the folder itself and/or its sub-folders.

    Only folders holding image files directly are offered; pipeline output
    folders, ``exclude_dirs`` and dot-folders are skipped.
    """
    exp_dir = Path(exp_dir)
    cfg_input = cfg_input or {}
    exts = _image_exts(cfg_input)
    exclude = {str(d).strip("/") for d in (cfg_input.get("exclude_dirs")
                                           or ["analysis", "jpegs", "pngs", "figures"])}
    found: list[FolderScan] = []
    if _has_images_shallow(exp_dir, exts):
        found.append(scan_folder(exp_dir, exp_dir, cfg_input))
        return found                  # sub-folders are this folder's results
    try:
        subdirs = sorted(d for d in exp_dir.iterdir() if d.is_dir())
    except OSError:
        return found
    for d in subdirs:
        if d.name.startswith(".") or d.name in exclude:
            continue
        if _has_images_shallow(d, exts):
            found.append(scan_folder(d, exp_dir, cfg_input))
    return found


# ===========================================================================
# Plate map: tables <-> plate_map.yaml
# ===========================================================================
#
# The plate-map state is a plain dict (it lives in Streamlit's session state):
#
#   plate, rows, cols, plate_id, mode, well_numbering,
#   mapping: equal | variable | ranges        (predefined only)
#   points_per_well: int
#   variable: [{well, count}]                  (acquisition order)
#   ranges:   [{well, first, last}]
#   stains:   {set_name: {channel: marker}}
#   sweeps:   [{stain, first, last, points_per_well}]
#   columns:  [condition column names]
#   wells:    [{well: key, <column>: value, class, stain, positive_for, control_for}]
#
# `well` keys are strings: "3" for numbered plates, "C10" for spinning disk.

def new_plate_map(scan: FolderScan | None, plate_id: str | None = None) -> dict:
    """A starting plate map built from what the folder shows."""
    mode = (scan.mode if scan and scan.mode else "predefined")
    plate_name, rows, cols = "18well", 3, 6
    if scan and scan.labels:
        r = max(plate.label_to_rowcol(l)[0] for l in scan.labels)
        c = max(plate.label_to_rowcol(l)[1] for l in scan.labels)
        plate_name = "96well" if (r > 3 or c > 6) else "18well"
    elif scan and scan.wells:
        hi = max(scan.wells)
        plate_name = "8well" if hi <= 8 else "18well" if hi <= 18 else "96well"
    rows, cols = PLATES[plate_name]

    channels = scan.channels if scan and scan.channels else ["DAPI", "Marker1", "Marker2", "Marker3"]
    state = {
        "plate": plate_name, "rows": rows, "cols": cols,
        "plate_id": plate_id or (scan.name if scan else "plate1"),
        "mode": mode, "well_numbering": "sequential" if mode == "predefined" else "plate",
        "mapping": "variable", "points_per_well": 1, "variable": [], "ranges": [],
        "stains": {"default": {i: str(n) for i, n in enumerate(channels)}},
        "sweeps": [], "columns": ["treatment"], "wells": [],
    }
    if mode == "predefined" and scan and scan.seq:
        from experiment.template import _guess_points_per_well
        state["points_per_well"] = _guess_points_per_well(len(scan.seq))
    # Predefined points: the filenames carry only the acquisition order, so any
    # well list would be a guess. The table starts empty and the user adds the
    # wells with their image counts. Well07 / spinning-disk filenames name the
    # well, so those wells are listed from the files.
    if mode != "predefined":
        state["wells"] = [{WELL_COL: k} for k in default_well_keys(state, scan)]
    return state


def default_well_keys(state: dict, scan: FolderScan | None) -> list[str]:
    """The wells a plate map should list, given the mapping and the images present."""
    mode = state["mode"]
    if mode == "spinning_disk":
        return list(scan.labels) if scan else []
    if mode == "generate":
        return [str(w) for w in scan.wells] if scan else []
    # predefined: whatever the Seq mapping produces
    if state.get("sweeps"):
        keys: set[int] = set()
        for sw in state["sweeps"]:
            n = int(sw.get("last", 0)) - int(sw.get("first", 0)) + 1
            ppw = max(int(sw.get("points_per_well") or 1), 1)
            keys.update(range(1, math.ceil(n / ppw) + 1))
        return [str(k) for k in sorted(keys)]
    if state["mapping"] == "variable":
        return [str(int(r["well"])) for r in state["variable"] if r.get("well") not in (None, "")]
    if state["mapping"] == "ranges":
        return [str(int(r["well"])) for r in state["ranges"] if r.get("well") not in (None, "")]
    n = len(scan.seq) if scan and scan.seq else 0
    ppw = max(int(state.get("points_per_well") or 1), 1)
    return [str(k) for k in range(1, math.ceil(n / ppw) + 1)] if n else []


def plate_map_from_yaml(raw: dict, scan: FolderScan | None = None) -> dict:
    """Read an existing plate_map.yaml (already parsed) into the editable state."""
    state = new_plate_map(scan)
    state["wells"] = []
    pname = raw.get("plate")
    geom = raw.get("geometry") or {}
    rows, cols = plate.resolve_geometry(pname, geom.get("rows"), geom.get("cols"))
    digits = "".join(ch for ch in str(pname or "") if ch.isdigit())
    state["plate"] = f"{digits}well" if f"{digits}well" in PLATES and not geom else "custom"
    state["rows"], state["cols"] = rows, cols
    state["plate_id"] = str(raw.get("plate_id") or (scan.name if scan else "plate1"))
    state["mode"] = str(raw.get("mode", state["mode"])).lower()
    if state["mode"] in parser.SPINNING_DISK_MODES:
        state["mode"] = "spinning_disk"
    state["well_numbering"] = str(raw.get("well_numbering")
                                  or ("sequential" if state["mode"] == "predefined" else "plate"))

    ppw, imaged, seq_ranges = raw.get("points_per_well"), raw.get("imaged_wells"), raw.get("seq_ranges")
    if seq_ranges:
        state["mapping"] = "ranges"
        state["ranges"] = [{"well": int(w), "first": int(r[0]), "last": int(r[1])}
                           for w, r in seq_ranges.items()]
    elif isinstance(ppw, dict) or imaged is not None or isinstance(ppw, list):
        state["mapping"] = "variable"
        if isinstance(ppw, dict):
            order = [int(w) for w in imaged] if imaged else sorted(int(w) for w in ppw)
            counts = {int(w): int(n) for w, n in ppw.items()}
        else:
            order = [int(w) for w in (imaged or [])]
            counts = ({w: int(n) for w, n in zip(order, ppw)} if isinstance(ppw, list)
                      else {w: int(ppw or 1) for w in order})
        state["variable"] = [{"well": w, "count": counts.get(w, 0)} for w in order]
    elif isinstance(ppw, int):
        state["mapping"], state["points_per_well"] = "equal", ppw

    if raw.get("stains"):
        state["stains"] = {str(k): {int(c): str(m) for c, m in (v or {}).items()}
                           for k, v in raw["stains"].items()}
    state["sweeps"] = [
        {"stain": str(name), "first": int(spec["seq_range"][0]), "last": int(spec["seq_range"][1]),
         "points_per_well": int(spec.get("points_per_well") or 1)}
        for name, spec in (raw.get("stain_sweeps") or {}).items()
    ]

    columns: list[str] = []
    rows_out: dict[str, dict] = {}
    for key, cond in (raw.get("wells") or {}).items():
        cond = dict(cond or {})
        if state["mode"] == "spinning_disk":
            numbers = plate.expand_label_key(key, cols)
            keys = [well_label(n, cols) for n in numbers]
        else:
            from experiment.plate_map import _expand_well_key
            keys = [str(n) for n in _expand_well_key(key)]
        for k in keys:
            row = {WELL_COL: k}
            for c, v in cond.items():
                if c in (POS_COL, CTRL_COL):
                    v = ", ".join(split_list(v)) if not isinstance(v, str) else v
                row[c] = v
                if c not in SPECIAL_COLS and c not in columns:
                    columns.append(c)
            rows_out[k] = row
    state["columns"] = columns or ["treatment"]
    state["wells"] = sorted(rows_out.values(), key=lambda r: well_sort_key(r[WELL_COL]))
    return state


def well_sort_key(key: str):
    key = str(key)
    if key[:1].isalpha():
        r, c = plate.label_to_rowcol(key)
        return (r, c)
    try:
        return (0, int(key))
    except ValueError:
        return (999, key)


def _well_condition(row: dict, columns: list[str]) -> dict:
    """One table row → the YAML mapping for that well (empty cells omitted)."""
    out: dict = {}
    for c in columns:
        v = coerce(row.get(c))
        if v is not None:
            out[c] = v
    cls = coerce(row.get(CLASS_COL))
    if cls is not None:
        out[CLASS_COL] = cls
    stain = coerce(row.get(STAIN_COL))
    if stain is not None and str(stain) != "default":
        out[STAIN_COL] = str(stain)
    for c in (POS_COL, CTRL_COL):
        items = split_list(row.get(c))
        if items:
            out[c] = "all" if [i.lower() for i in items] == ["all"] else items
    return out


def compress_wells(state: dict) -> list[tuple[str, dict]]:
    """Wells with identical conditions in a run → one range key (``"2-4"`` / ``"C1-C3"``)."""
    label_mode = state["mode"] == "spinning_disk"
    rows = [r for r in state["wells"] if str(r.get(WELL_COL, "")).strip()]
    rows = sorted(rows, key=lambda r: well_sort_key(r[WELL_COL]))
    out: list[tuple[str, dict]] = []
    run_start = run_end = None
    run_cond = None

    def flush():
        if run_start is None:
            return
        key = run_start if run_start == run_end else f"{run_start}-{run_end}"
        if not label_mode and run_start == run_end:
            key = int(key)                    # plain numbers stay numbers in YAML
        out.append((key, run_cond))

    def adjacent(a: str, b: str) -> bool:
        if label_mode:
            ra, ca = plate.label_to_rowcol(a)
            rb, cb = plate.label_to_rowcol(b)
            return ra == rb and cb == ca + 1
        return int(b) == int(a) + 1

    for r in rows:
        key = str(r[WELL_COL]).strip().upper() if label_mode else str(int(coerce(r[WELL_COL])))
        if label_mode:
            rr, cc = plate.label_to_rowcol(key)
            key = f"{chr(ord('A') + rr - 1)}{cc}"
        cond = _well_condition(r, state["columns"])
        if run_start is not None and cond == run_cond and adjacent(run_end, key):
            run_end = key
            continue
        flush()
        run_start = run_end = key
        run_cond = cond
    flush()
    return out


def plate_map_dict(state: dict) -> dict:
    """The plate map as the pipeline will read it."""
    d: dict = {}
    if state["plate"] == "custom":
        d["geometry"] = {"rows": int(state["rows"]), "cols": int(state["cols"])}
    else:
        d["plate"] = state["plate"]
    d["plate_id"] = state["plate_id"]
    d["mode"] = state["mode"]
    if state["mode"] == "predefined":
        if state.get("well_numbering") == "plate":
            d["well_numbering"] = "plate"
        if not state.get("sweeps"):
            if state["mapping"] == "equal":
                d["points_per_well"] = int(state["points_per_well"])
            elif state["mapping"] == "variable":
                rows = [r for r in state["variable"] if coerce(r.get("well")) is not None]
                d["imaged_wells"] = [int(coerce(r["well"])) for r in rows]
                d["points_per_well"] = [int(coerce(r.get("count")) or 0) for r in rows]
            else:
                d["seq_ranges"] = {int(coerce(r["well"])): [int(coerce(r["first"])), int(coerce(r["last"]))]
                                   for r in state["ranges"] if coerce(r.get("well")) is not None}
    d["stains"] = {name: {int(c): m for c, m in sorted(chs.items()) if str(m).strip()}
                   for name, chs in state["stains"].items()}
    if state.get("sweeps"):
        d["stain_sweeps"] = {
            sw["stain"]: {"seq_range": [int(sw["first"]), int(sw["last"])],
                          "points_per_well": int(sw["points_per_well"])}
            for sw in state["sweeps"] if sw.get("stain")}
    d["wells"] = {k: v for k, v in compress_wells(state)}
    return d


def render_plate_map(state: dict, folder_name: str = "") -> str:
    """plate_map.yaml text, with short comments explaining each block."""
    d = plate_map_dict(state)
    L: list[str] = []
    add = L.append
    add("# =============================================================================")
    add(f"# plate_map.yaml: {folder_name or state['plate_id']}")
    add("# Written by the pipeline GUI (gui/app.py). Safe to edit by hand; the GUI can")
    add("# re-open it. Run `python main.py check` (or the GUI's Check step) afterwards.")
    add("# =============================================================================")
    add("")
    if "geometry" in d:
        add(f"geometry: {flow(d['geometry'])}      # rows x columns of the plate")
    else:
        add(f"plate: {d['plate']}                  # 8well | 18well | 96well")
    add(f"plate_id: {flow(d['plate_id'])}          # same value in every folder of ONE physical plate")
    add(f"mode: {d['mode']}                  # {MODE_HELP.get(d['mode'], '')}")
    if "well_numbering" in d:
        add("well_numbering: plate           # well keys below are real plate positions")
    add("")
    if d["mode"] == "predefined" and "stain_sweeps" not in d:
        add("# --- Seq -> well: which acquisition points belong to which well -------------")
        if d["mode"] == "predefined" and state.get("well_numbering") != "plate":
            add("# Wells are numbered in ACQUISITION ORDER: well 1 = the first well imaged.")
        for k in ("imaged_wells", "points_per_well", "seq_ranges"):
            if k in d:
                add(f"{k}: {flow(d[k])}")
        add("")
    add("# --- Stain sets: channel index (0-based) -> marker name --------------------")
    add("# 'default' applies to every well that does not name another set with `stain:`.")
    add("stains:")
    for name, chs in d["stains"].items():
        add(f"  {flow(name)}: {flow(chs)}")
    add("")
    if "stain_sweeps" in d:
        add("# --- Several stain sets in ONE folder: each Seq block is one stain's sweep --")
        add("# Seq indices are local to each block (its first point is well 1).")
        add("stain_sweeps:")
        for name, spec in d["stain_sweeps"].items():
            add(f"  {flow(name)}: {flow(spec)}")
        add("")
    add("# --- Well -> experimental condition ------------------------------------------")
    add("# Reserved keys: stain (stain set), class (negative control),")
    add("# positive_for / control_for (per-marker control exceptions).")
    add("# Every other key is a condition column you can group, facet and plot by.")
    add("wells:")
    for key, cond in d["wells"].items():
        add(f"  {flow(key)}: {flow(cond)}")
    if not d["wells"]:
        add("  {}")
    return "\n".join(L) + "\n"


def plate_map_text(state: dict, folder_name: str, existing_text: str | None) -> str:
    """plate_map.yaml for *state*, written into the file on disk when there is one.

    Hand comments in a plate map are often the only record of why a well or a
    stain was changed, so an unchanged plate map keeps its file byte for byte
    and an edited one keeps every comment whose line survives. Falls back to the
    plain rendering when that cannot be done exactly.
    """
    text = render_plate_map(state, folder_name)
    if not existing_text:
        return text
    try:
        if plate_map_equal(yaml.safe_load(existing_text) or {}, plate_map_dict(state)):
            return existing_text
    except yaml.YAMLError:
        return text
    return ConfigTemplate(text=text).render_into(yaml.safe_load(text), existing_text)


def plate_map_equal(a: dict, b: dict) -> bool:
    """Same plate map, whatever the layout of the YAML (well ranges expanded)."""
    def norm(d):
        d = json.loads(json.dumps(d or {}, default=str))
        wells = {}
        for key, cond in (d.get("wells") or {}).items():
            try:
                if str(d.get("mode", "")).lower() in parser.SPINNING_DISK_MODES:
                    _, cols = plate.resolve_geometry(d.get("plate"), (d.get("geometry") or {}).get(
                        "rows"), (d.get("geometry") or {}).get("cols"))
                    keys = [well_label(n, cols) for n in plate.expand_label_key(key, cols)]
                else:
                    from experiment.plate_map import _expand_well_key
                    keys = [str(n) for n in _expand_well_key(key)]
            except Exception:                           # noqa: BLE001
                keys = [str(key)]
            for k in keys:
                wells[k] = cond
        d["wells"] = wells
        for k in ("stains",):
            if k in d:
                d[k] = {str(n): {str(c): m for c, m in (v or {}).items()}
                        for n, v in (d[k] or {}).items()}
        return d
    return norm(a) == norm(b)


@dataclass
class PlateMapReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    per_well: list[dict]         # well, images, seq range, stain set, condition?
    unmapped: list[str]


def validate_plate_map(state: dict, scan: FolderScan | None) -> PlateMapReport:
    """Load the would-be plate map with the pipeline's own PlateMap and map every image."""
    from experiment.plate_map import PlateMap

    errors: list[str] = []
    warnings: list[str] = []
    try:
        text = render_plate_map(state)
        yaml.safe_load(text)
    except Exception as exc:                            # noqa: BLE001
        return PlateMapReport(False, [f"Could not build the plate map: {exc}"], [], [], [])

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / PLATE_MAP_NAME
        write_text(p, text)
        try:
            pm = PlateMap(p)
        except Exception as exc:                        # noqa: BLE001
            return PlateMapReport(False, [str(exc).replace(str(p), "plate_map.yaml")], [], [], [])

    if not any(pm.stains.values()):
        errors.append("No stain set has any channel → marker names.")
    for name in {str(r.get(STAIN_COL)) for r in state["wells"] if coerce(r.get(STAIN_COL))}:
        if name not in pm.stains:
            errors.append(f"Stain set '{name}' is used in the wells table but not defined.")

    per: dict[int, dict] = {}
    unmapped: list[str] = []
    for img in (scan.images if scan else []):
        info = parser.parse_image(img, pm.mode)
        if pm.is_spinning_disk:
            lab = info["well_label"]
            well = plate.label_to_well(lab, pm.cols) if lab else None
        elif pm.mode == "generate":
            well = info["well"]
        else:
            well = pm.well_for_seq(info["seq_index"])
        if well is None:
            unmapped.append(img.name)
            continue
        e = per.setdefault(well, {"seq": [], "n": 0, "stains": set()})
        e["n"] += 1
        if info["seq_index"] is not None:
            e["seq"].append(info["seq_index"])
        e["stains"].add(pm.resolve_stain(well, info["seq_index"])[0])

    label_mode = pm.is_spinning_disk
    rows = []
    for well in sorted(set(per) | set(pm.wells)):
        e = per.get(well, {"seq": [], "n": 0, "stains": set()})
        key = well_label(well, pm.cols) if label_mode else str(well)
        cond = pm.conditions_for_well(well)
        rows.append({
            "well": key, "images": e["n"],
            "Seq": f"{min(e['seq'])}-{max(e['seq'])}" if e["seq"] else "",
            "stain set": ", ".join(sorted(e["stains"])) or pm.stain_set_name(well),
            "condition": ", ".join(f"{k}={v}" for k, v in cond.items()) or
                         ("(none)" if well not in pm.wells else ""),
        })
    if unmapped:
        errors.append(f"{len(unmapped)} image(s) map to no well, e.g. {', '.join(unmapped[:3])}. "
                      "Check the Seq → well mapping.")
    missing = sorted(w for w in per if w not in pm.wells)
    if missing:
        errors.append("Wells with images but no row in the wells table: "
                      + ", ".join(well_label(w, pm.cols) if label_mode else str(w) for w in missing))
    unused = sorted(w for w in pm.wells if w not in per)
    if unused and scan and scan.images:
        warnings.append("Wells listed but with no images: "
                        + ", ".join(well_label(w, pm.cols) if label_mode else str(w) for w in unused))
    has_ctrl = any(str(c.get(CLASS_COL, "")).strip() for c in pm.wells.values()) or pm.declares_marker_controls
    if not has_ctrl:
        warnings.append("No well is marked as a negative control (class column). Without one, "
                        "positivity thresholds fall back to a mixture-model fit, which may fail.")
    return PlateMapReport(not errors, errors, warnings, rows, unmapped)


# ===========================================================================
# config.yaml: fill the commented template
# ===========================================================================

# A mapping key line: `key: value  # comment`. Keys may be plain words, numbers
# or ranges (plate-map wells: `1:`, `2-4:`, `C1-C3:`) or quoted.
_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>'[^']*'|\"[^\"]*\"|[A-Za-z0-9_][\w.\-+]*)"
                      r":(?P<rest>(?:\s.*)?)$")


def _key_value(text: str):
    """The key YAML reads from *text*: '1' → 1, '2-4' → '2-4', "'x'" → 'x'."""
    try:
        d = yaml.safe_load(f"{text}: 0")
        return next(iter(d)) if isinstance(d, dict) and d else text
    except yaml.YAMLError:
        return text


def _plain(values):
    """*values* as YAML reads them back (int keys stay ints, unlike a JSON round trip)."""
    return yaml.safe_load(yaml.safe_dump(json.loads(json.dumps(values, default=str))
                                         if not _has_nonstr_keys(values) else values,
                                         allow_unicode=True))


def _has_nonstr_keys(node) -> bool:
    if isinstance(node, dict):
        return any(not isinstance(k, str) or _has_nonstr_keys(v) for k, v in node.items())
    if isinstance(node, list):
        return any(_has_nonstr_keys(v) for v in node)
    return False


def _split_comment(rest: str) -> tuple[str, str, int]:
    """``"  0.4   # help"`` → (value, comment, column of '#'): '#' inside quotes ignored."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            return rest[:i].strip(), rest[i + 1:].strip(), i
    return rest.strip(), "", -1


@dataclass
class _Entry:
    line: int
    path: tuple
    indent: int
    is_leaf: bool
    comment: str
    help: str


class ConfigTemplate:
    """The commented ``templates/config.yaml``: defaults, help text, and a writer.

    Writing substitutes new values into the template line by line, so every
    option keeps its explanatory comment. Keys the template does not know (e.g.
    from a hand-edited config) are inserted into their section. If the result
    does not read back as exactly the requested values, a plain YAML dump is
    written instead: correctness before comments.
    """

    def __init__(self, path: Path = TEMPLATES / CONFIG_NAME, text: str | None = None):
        self.text = Path(path).read_text(encoding="utf-8") if text is None else text
        self.lines = self.text.splitlines()
        self.defaults = yaml.safe_load(self.text)
        self.entries: dict[tuple, _Entry] = {}
        self.section_end: dict[tuple, int] = {}
        self._parse()

    def _parse(self) -> None:
        stack: list[tuple[int, str]] = []
        pending: list[str] = []
        for i, line in enumerate(self.lines):
            s = line.strip()
            if not s:
                pending = []
                continue
            if s.startswith("#"):
                txt = s.lstrip("#").strip()
                if txt and not set(txt) <= set("=--"):
                    pending.append(txt)
                continue
            if s.startswith("- "):
                continue
            m = _LINE_RE.match(line)
            if not m:
                pending = []
                continue
            indent = len(m["indent"])
            while stack and stack[-1][0] >= indent:
                stack.pop()
            key = _key_value(m["key"])
            path = tuple(k for _, k in stack) + (key,)
            value, comment, _ = _split_comment(m["rest"])
            is_leaf = value != ""
            self.entries[path] = _Entry(i, path, indent, is_leaf, comment,
                                        " ".join(pending + ([comment] if comment else [])))
            for depth in range(1, len(path) + 1):
                self.section_end[path[:depth]] = i
            if not is_leaf:
                stack.append((indent, key))
            pending = []

    def help(self, *path) -> str:
        e = self.entries.get(tuple(path))
        return e.help if e else ""

    def merged(self, user_cfg: dict | None) -> dict:
        """Template defaults with a (possibly partial) user config laid over them."""
        return deep_merge(self.defaults, user_cfg or {})

    def render(self, values: dict) -> str:
        values = {k: v for k, v in values.items() if not str(k).startswith("_")}
        lines = list(self.lines)
        inserts: dict[int, list[str]] = {}

        def get(path):
            node = values
            for k in path:
                if not isinstance(node, dict) or k not in node:
                    return _MISSING
                node = node[k]
            return node

        # 1. substitute every leaf the template has
        for path, e in self.entries.items():
            if not e.is_leaf:
                continue
            v = get(path)
            if v is _MISSING:
                continue
            if path == ("datasets",):
                block = [f"datasets:{'':<25}# one entry per acquisition folder"]
                block += [f"  - {flow(ds)}" for ds in (v or [])]
                if not v:
                    block[0] = "datasets: []"
                lines[e.line] = "\n".join(block)
                continue
            line = lines[e.line]
            m = _LINE_RE.match(line)
            _, comment, hash_col = _split_comment(m["rest"])
            head = f"{m['indent']}{m['key']}: {flow(v)}"
            if comment:
                col = len(m["indent"]) + len(m["key"]) + 1 + hash_col
                head = f"{head:<{col}}" if len(head) < col else head + "  "
                head = f"{head}# {comment}"
            lines[e.line] = head

        # 2. keys the template lacks → insert at the end of their section
        def walk(node, path):
            for k, v in node.items():
                p = path + (k,)
                if p in self.entries:
                    if isinstance(v, dict) and not self.entries[p].is_leaf:
                        walk(v, p)
                    continue
                parent = path
                while parent and parent not in self.section_end:
                    parent = parent[:-1]
                if not parent:
                    inserts.setdefault(len(lines) - 1, []).append(f"{k}: {flow(v)}")
                else:
                    indent = " " * (self.entries[parent].indent + 2)
                    inserts.setdefault(self.section_end[parent], []).append(f"{indent}{k}: {flow(v)}")

        walk(values, ())
        out: list[str] = []
        for i, line in enumerate(lines):
            out.append(line)
            out.extend(inserts.get(i, []))
        text = "\n".join(out) + "\n"

        try:
            ok = yaml.safe_load(text) == _plain(values)
        except yaml.YAMLError:
            ok = False
        if not ok:
            text = ("# config.yaml: written by the pipeline GUI (plain form; the commented\n"
                    "# template could not represent these values).\n"
                    + yaml.safe_dump(values, sort_keys=False, allow_unicode=True))
        return text


    # -- writing into an existing file ------------------------------------

    def render_into(self, values: dict, existing_text: str | None) -> str:
        """config.yaml text for *values*, written into the file already on disk.

        Re-rendering from the template would replace every comment someone typed
        into their config (why a Seq was left out, why a control was dropped).
        So when a config exists, it is the base: a value that did not change keeps
        its line exactly as it was, a changed value is substituted on its own
        line with its comment kept, and an option the file lacks is added from
        the template, with the template's explanation, at the end of its section.
        Falls back to the plain template rendering if the result would not read
        back as exactly *values*.
        """
        values = {k: v for k, v in values.items() if not str(k).startswith("_")}
        if not existing_text or not existing_text.strip():
            return self.render(values)
        try:
            base = ConfigTemplate(text=existing_text)
            out = base._merge_into(values, self)
            if yaml.safe_load(out) == _plain(values):
                return out
        except Exception:                                   # noqa: BLE001
            pass
        return self.render(values)

    def _value_at(self, path: tuple):
        node = self.defaults_raw
        for k in path:
            if not isinstance(node, dict) or k not in node:
                return _MISSING
            node = node[k]
        return node

    @property
    def defaults_raw(self) -> dict:
        return self.defaults or {}

    def _block(self, path: tuple) -> list[str]:
        """This file's lines for *path*: its preceding comments, the key, its body."""
        e = self.entries[path]
        start = e.line
        while start > 0 and self.lines[start - 1].strip().startswith("#"):
            start -= 1
        end = self.section_end.get(path, e.line) if not e.is_leaf else e.line
        # trailing comment lines that belong to the last key (continuations)
        while end + 1 < len(self.lines) and self.lines[end + 1].strip().startswith("#") \
                and len(self.lines[end + 1]) - len(self.lines[end + 1].lstrip()) > e.indent:
            end += 1
        return self.lines[start:end + 1]

    def _last_line(self, path: tuple) -> int:
        """Last line of *path*'s block in this file (its body and continuations)."""
        e = self.entries[path]
        end = e.line if e.is_leaf else self.section_end.get(path, e.line)
        while end + 1 < len(self.lines) and self.lines[end + 1].strip().startswith("#") \
                and len(self.lines[end + 1]) - len(self.lines[end + 1].lstrip()) > e.indent:
            end += 1
        return end

    def _insert_at(self, path: tuple, rendered: "ConfigTemplate") -> int:
        """Where a key this file lacks goes: after the sibling that precedes it in
        the template, else at the start of its section, else at the end."""
        parent = path[:-1]
        order = [q for q in rendered.entries if q[:-1] == parent]
        before = order[:order.index(path)] if path in order else order
        for q in reversed(before):
            if q in self.entries:
                return self._last_line(q)
        if parent in self.entries:
            return self.entries[parent].line
        return len(self.lines) - 1

    def _merge_into(self, values: dict, template: "ConfigTemplate") -> str:
        lines = list(self.lines)
        inserts: dict[int, list[str]] = {}
        rendered = ConfigTemplate(text=template.render(values))   # template text, new values

        def get(path):
            node = values
            for k in path:
                if not isinstance(node, dict) or k not in node:
                    return _MISSING
                node = node[k]
            return node

        def same(a, b) -> bool:
            return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True,
                                                                          default=str)

        # 1. values already in this file
        for path, e in self.entries.items():
            v = get(path)
            if v is _MISSING:
                continue
            old = self._value_at(path)
            if path == ("datasets",):
                if same(old, v):
                    continue
                # Entry lines ('  - {...}  # note'): keep each unchanged entry's line,
                # comments around them stay where they are.
                item_rows, i = [], e.line + 1
                while i < len(lines) and lines[i] is not None and (
                        lines[i].startswith(" ") or lines[i].strip().startswith("#")
                        or not lines[i].strip()):
                    if lines[i].lstrip().startswith("- "):
                        item_rows.append(i)
                    elif lines[i].strip() and not lines[i].strip().startswith("#"):
                        break
                    i += 1
                known = []
                for i in item_rows:
                    body, _, _ = _split_comment(lines[i].lstrip()[2:])
                    try:
                        known.append((yaml.safe_load(body), lines[i]))
                    except yaml.YAMLError:
                        pass
                block = []
                for ds in (v or []):
                    hit = next((k for k in known if same(k[0], ds)), None)
                    if hit is not None:
                        known.remove(hit)
                        block.append(hit[1])
                    else:
                        block.append(f"  - {flow(ds)}")
                if not v:
                    lines[e.line] = "datasets: []"
                elif not e.is_leaf or not item_rows:
                    m = _LINE_RE.match(lines[e.line])
                    _, comment, _ = _split_comment(m["rest"])
                    lines[e.line] = "datasets:" + (f"{'':<25}# {comment}" if comment else "")
                for k, i in enumerate(item_rows):
                    lines[i] = None
                if block:
                    at = item_rows[0] if item_rows else e.line
                    if item_rows:
                        lines[at] = "\n".join(block)
                    else:
                        lines[at] = lines[at] + "\n" + "\n".join(block)
                continue
            if not e.is_leaf:
                if isinstance(v, dict):
                    continue
                # a section in the file, a scalar now: replace the key line
                lines[e.line] = f"{' ' * e.indent}{path[-1]}: {flow(v)}"
                for i in range(e.line + 1, self.section_end.get(path, e.line) + 1):
                    lines[i] = None
                continue
            if same(old, v):
                continue                                   # untouched: keep the line verbatim
            m = _LINE_RE.match(lines[e.line])
            _, comment, hash_col = _split_comment(m["rest"])
            head = f"{m['indent']}{m['key']}: {flow(v)}"
            if comment:
                col = len(m["indent"]) + len(m["key"]) + 1 + hash_col
                head = f"{head:<{col}}" if len(head) < col else head + "  "
                head = f"{head}# {comment}"
            lines[e.line] = head

        # 1b. keys this file has that the values no longer do (a removed well):
        # their whole block goes, unless an ancestor is already gone
        for path, e in self.entries.items():
            if get(path) is not _MISSING or path[:-1] and get(path[:-1]) is _MISSING:
                continue
            if path[:-1] and not isinstance(get(path[:-1]), dict):
                continue
            first = e.line
            while first > 0 and lines[first - 1] is not None \
                    and lines[first - 1].strip().startswith("#") \
                    and len(lines[first - 1]) - len(lines[first - 1].lstrip()) == e.indent \
                    and first - 1 > (self.entries[path[:-1]].line if path[:-1] else -1):
                first -= 1                     # its own comment lines, same indent
            for i in range(first, self._last_line(path) + 1):
                lines[i] = None

        # 2. options this file lacks: the template's block (with its comments)
        def walk(node, path):
            for k, v in node.items():
                p = path + (k,)
                if p in self.entries:
                    if isinstance(v, dict) and not self.entries[p].is_leaf:
                        walk(v, p)
                    continue
                if p in rendered.entries:
                    block = rendered._block(p)
                else:
                    indent = " " * (2 * len(path))
                    block = [f"{indent}{k}: {flow(v)}"]
                inserts.setdefault(self._insert_at(p, rendered), []).extend(block)

        walk(values, ())
        out: list[str] = []
        for i, line in enumerate(lines):
            if line is not None:
                out.append(line)
            out.extend(inserts.get(i, []))
        text = "\n".join(out)
        return text if text.endswith("\n") else text + "\n"


class _Missing:
    pass


_MISSING = _Missing()


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_in(d: dict, path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def set_in(d: dict, path, value) -> None:
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


# ===========================================================================
# hpc_job.env
# ===========================================================================

_ENV_RE = re.compile(r'^(?P<key>[A-Z_][A-Z0-9_]*)=(?P<val>"[^"]*"|\S*)(?P<rest>.*)$')


def env_defaults(path: Path = TEMPLATES / ENV_NAME) -> dict:
    return parse_env(Path(path).read_text(encoding="utf-8"))


def parse_env(text: str) -> dict:
    out = {}
    for line in text.replace("\r", "").splitlines():
        m = _ENV_RE.match(line.strip())
        if m:
            val = m["val"].strip('"')
            # "${SEQ:-0-4}" (a hand-written override pattern) → its default "0-4"
            dm = re.fullmatch(r"\$\{\w+:-(.*)\}", val)
            out[m["key"]] = dm.group(1) if dm else val
    return out


def render_env(values: dict, template: Path = TEMPLATES / ENV_NAME) -> str:
    """Fill the commented env template. Values are double-quoted for bash."""
    lines = []
    seen = set()
    for line in Path(template).read_text(encoding="utf-8").splitlines():
        m = _ENV_RE.match(line)
        if m and m["key"] in values:
            val = str(values[m["key"]] if values[m["key"]] is not None else "")
            val = val.replace('"', "").replace("\n", " ")
            line = f'{m["key"]}="{val}"{m["rest"]}'
            seen.add(m["key"])
        lines.append(line)
    extra = [k for k in values if k not in seen]
    if extra:
        lines.append("")
        lines.append("# --- other settings ---")
        lines += [f'{k}="{values[k]}"' for k in extra]
    return "\n".join(lines) + "\n"


# ===========================================================================
# Laptop path <-> HPC path
# ===========================================================================

def folder_shortcuts() -> dict[str, str]:
    """Places to jump to in the folder browser: home plus mounted drives and shares.

    Network shares mounted from the Linux file manager live under
    /run/user/<uid>/gvfs, which is hard to reach by clicking up and down.
    """
    out = {"Home folder": str(Path.home())}
    cands: list[Path] = []
    if sys.platform == "win32":
        cands = [Path(f"{c}:\\") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    elif sys.platform == "darwin":
        cands = [Path("/Volumes")]
    else:
        user = os.environ.get("USER", "")
        for base in (Path(f"/run/user/{os.getuid()}/gvfs"), Path("/media") / user,
                     Path("/run/media") / user, Path("/mnt")):
            try:
                cands += sorted(d for d in base.iterdir() if d.is_dir())
            except OSError:
                pass
    if sys.platform == "darwin":
        try:
            cands = sorted(d for d in cands[0].iterdir() if d.is_dir())
        except OSError:
            cands = []
    for d in cands:
        try:
            if not d.is_dir():
                continue
        except OSError:
            continue
        m = re.match(r"smb-share:server=([^,]+),share=([^,]+)", d.name)
        label = f"{m.group(2)} on {m.group(1)}" if m else str(d)
        out[label] = str(d)
    return out


def guess_share_root(path: Path) -> str:
    """The local mount point of the lab share, guessed from a path inside it."""
    parts = Path(path).resolve().parts
    for i, part in enumerate(parts):
        if "warmflash" in part.lower():
            return str(Path(*parts[: i + 1]))
    return str(Path(path).anchor)


def to_hpc_path(local: Path | str, local_root: str, hpc_root: str) -> str:
    """Translate a laptop path on the share to the same path as the HPC sees it."""
    local_s = str(Path(local).resolve()) if local else ""
    root_s = str(Path(local_root).resolve()) if local_root else ""
    if not root_s:
        return local_s.replace("\\", "/")
    try:
        rel = Path(local_s).relative_to(root_s)
    except ValueError:
        return local_s.replace("\\", "/")
    rel_s = rel.as_posix()
    return hpc_root.rstrip("/") + ("/" + rel_s if rel_s != "." else "")


# ===========================================================================
# Running the pipeline from the GUI
# ===========================================================================

def main_command(*args: str) -> list[str]:
    return [sys.executable, "-u", str(PIPELINE_DIR / "main.py"), *args]


def stream_command(cmd: list[str], cwd: Path):
    """Run a command, yielding its output line by line; the last item is the exit code."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    yield proc.wait()


def has_segmentation(folder: Path) -> bool:
    """True when at least one image in this dataset folder has been segmented."""
    try:
        with os.scandir(folder) as it:
            return any(e.is_dir() and (Path(e.path) / "nuclear_masks.tif").exists() for e in it)
    except OSError:
        return False


def optional_steps() -> list[str]:
    try:
        from downstream.steps import OPTIONAL_STEPS
        return list(OPTIONAL_STEPS)
    except Exception:                                   # noqa: BLE001
        return ["boundary", "coexpression", "multimarker", "compare", "effects",
                "signaling_summary"]


def downstream_step_names() -> list[str]:
    try:
        from downstream.registry import default_step_names
        import downstream.steps  # noqa: F401  (registers built-in steps)
        return list(default_step_names())
    except Exception:                                   # noqa: BLE001
        return ["colony", "signaling", "assemble", "qc", "thresholds", "profiles",
                "coexpression", "signaling_summary", "aggregate", "compare", "multimarker",
                "report"]


def changed_keys(old: dict | None, new: dict | None, path: tuple = ()) -> list[str]:
    """Dotted paths whose value differs between two configs (for 'what will change')."""
    old, new = old or {}, new or {}
    out: list[str] = []
    for k in list(dict.fromkeys(list(old) + list(new))):
        if str(k).startswith("_"):
            continue
        a, b = old.get(k, _MISSING), new.get(k, _MISSING)
        if isinstance(a, dict) and isinstance(b, dict):
            out += changed_keys(a, b, path + (k,))
        elif json.dumps(a, sort_keys=True, default=str) != json.dumps(b, sort_keys=True,
                                                                      default=str):
            tag = " (new option)" if a is _MISSING else ""
            out.append(".".join(map(str, path + (k,))) + tag)
    return out


def segmentation_status(exp_dir: Path, cfg: dict) -> list:
    """Per dataset: which segmentation results are on disk (see src/seg_status.py)."""
    import seg_status
    from pipeline_config import resolve_datasets
    c = copy.deepcopy(cfg)
    c["_root"] = str(Path(exp_dir).resolve())
    try:
        datasets = resolve_datasets(c)
    except ValueError:
        return []
    return [seg_status.dataset_status(ds) for ds in datasets]


def analysis_folder(cfg: dict, statuses: list) -> str:
    """The folder downstream will write to, as main.py resolves it."""
    import seg_status
    out = get_in(cfg, ["downstream", "results_dir"], "auto")
    if not seg_status.is_auto(out):
        return str(out)
    choice = seg_status.normalize_choice(get_in(cfg, ["downstream", "seg_mode"], "auto"))
    modes = set()
    for st in statuses:
        if not st.n_ready:
            continue
        have = set(st.modes) or {st.configured}
        if choice in ("2d", "3d"):
            have &= {choice}
        modes |= have
    if not modes:
        modes = {choice} if choice in ("2d", "3d") else {
            st.configured for st in statuses} or {"2d"}
    return "./" + seg_status.auto_dir_name(sorted(modes))


def collect_plate_maps(exp_dir: Path, datasets: list[dict]) -> dict[str, dict]:
    """{dataset name: parsed plate_map.yaml} for every dataset that has one."""
    out = {}
    for ds in datasets:
        p = (Path(exp_dir) / ds["directory"]).resolve() / PLATE_MAP_NAME
        if p.exists():
            try:
                out[ds["name"]] = read_yaml(p)
            except Exception:                           # noqa: BLE001
                pass
    return out


def plate_map_vocabulary(maps: dict[str, dict]) -> dict:
    """Columns, their values, and markers declared across plate maps: for dropdowns."""
    columns: dict[str, set] = {}
    markers: list[str] = []
    for raw in maps.values():
        for cond in (raw.get("wells") or {}).values():
            for k, v in (cond or {}).items():
                if k in (STAIN_COL, POS_COL, CTRL_COL):
                    continue
                columns.setdefault(k, set()).add(v)
        for chs in (raw.get("stains") or {}).values():
            for m in (chs or {}).values():
                if str(m) not in markers:
                    markers.append(str(m))
    numeric = sorted(c for c, vals in columns.items()
                     if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                     for v in vals))
    return {"columns": {c: sorted(map(str, v)) for c, v in columns.items()},
            "numeric": numeric, "markers": markers}


__all__ = [n for n in dir() if not n.startswith("_")]
