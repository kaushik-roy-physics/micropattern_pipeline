"""
Generate a ``plate_map.yaml`` template, pre-filled from the images on disk.

The plate map is the one file the pipeline cannot infer, and the one a new lab
member has to write before anything else works. This module removes most of the
guesswork by *looking at the folder first*: the acquisition mode is read off the
filenames, the wells (or Seq indices) are the ones actually present, the
points-per-well is measured rather than assumed, and the channel names come from
``image_metadata.json`` when ``load`` has already run.

What it cannot know it says so, in a comment, next to the line you have to edit.
Everything it does know is written as a real value with the alternatives left
commented out underneath, so the file doubles as documentation of the schema.

Nothing here overwrites an existing plate map unless asked: a plate map is
hand-authored knowledge and losing one costs an afternoon.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from image_processing.image_loader import METADATA_NAME, result_dir_for

from . import parser, plate
from .manifest import PLATE_MAP_NAME, _iter_images, _normalize_extensions

logger = logging.getLogger(__name__)

# Plate geometries offered in the header comment.
_KNOWN_PLATES = ["8well", "18well", "96well"]


class Scan:
    """What the folder itself reveals about an acquisition."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.images: list[Path] = []
        self.mode: str | None = None
        self.wells: list[int] = []
        self.labels: list[str] = []
        self.seq_indices: list[int] = []
        self.points_per_well: dict[int | str, int] = {}
        self.channels: list[str] = []
        self.siblings: list[Path] = []


def detect_mode(images) -> str | None:
    """Infer the acquisition mode from filenames.

    Order matters: ``WellC10`` also contains ``Well`` followed by no digits, and
    ``Well07`` must not be read as a label. The label regex is anchored on a row
    *letter*, so testing it first is unambiguous.
    """
    n_label = sum(1 for p in images if parser.parse_well_label(p.name))
    n_well = sum(1 for p in images if parser.parse_well(p.name))
    n_seq = sum(1 for p in images if parser.parse_seq(p.name) is not None)

    if n_label and n_label >= max(n_well, n_seq):
        return "spinning_disk"
    if n_well and n_well >= n_seq:
        return "generate"
    if n_seq:
        return "predefined"
    return None


def scan(directory: Path, cfg: dict | None = None) -> Scan:
    """Inspect *directory* and report everything usable for a template."""
    cfg = cfg or {}
    out = Scan(directory)
    extensions = _normalize_extensions((cfg.get("input") or {}).get("file_types"))
    exclude = (cfg.get("input") or {}).get("exclude_dirs") or []
    out.images = sorted(_iter_images(out.directory, extensions, exclude))
    if not out.images:
        return out

    out.mode = detect_mode(out.images)

    if out.mode == "spinning_disk":
        labels = [parser.parse_well_label(p.name) for p in out.images]
        out.labels = sorted({lab for lab in labels if lab}, key=_label_sort_key)
        counts = Counter(lab for lab in labels if lab)
        out.points_per_well = dict(counts)
    elif out.mode == "generate":
        wells = [parser.parse_well(p.name) for p in out.images]
        out.wells = sorted({w for w in wells if w is not None})
        out.points_per_well = dict(Counter(w for w in wells if w is not None))
    elif out.mode == "predefined":
        out.seq_indices = sorted(s for s in (parser.parse_seq(p.name) for p in out.images)
                                 if s is not None)

    out.channels = _channel_names(out.images)
    out.siblings = _sibling_folders(out.directory)
    return out


def _label_sort_key(label: str):
    row, col = plate.label_to_rowcol(label)
    return (row, col)


def _channel_names(images) -> list[str]:
    """Channel names from the first image's ``image_metadata.json``, if 'load' ran."""
    for img in images[:20]:
        meta_path = result_dir_for(img) / METADATA_NAME
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        names = meta.get("channel_names")
        if names:
            return [str(n) for n in names]
    return []


def _sibling_folders(directory: Path) -> list[Path]:
    """Sibling directories that look like other panels of the same experiment.

    Used only to raise the ``plate_id`` question: sibling panel folders become
    separate datasets, and a marker can only be pooled across them (``scope:
    plate``) when they declare the *same* explicit ``plate_id``. Because the
    default is the folder name, forgetting that makes pooling silently do nothing.
    """
    parent = directory.parent
    out: list[Path] = []
    try:
        siblings = sorted(parent.iterdir())
    except OSError:
        return out
    for d in siblings:
        # Every probe here can raise on a directory we may not read: an
        # unreadable neighbor must never stop a template being written.
        try:
            if not d.is_dir() or d == directory or d.name.startswith("."):
                continue
            if (d / PLATE_MAP_NAME).exists() or any(d.glob("*.nd2")):
                out.append(d)
        except OSError:
            continue
    return out


def _infer_geometry(scan_result: Scan) -> str:
    """Best guess at the plate line, as YAML."""
    if scan_result.labels:
        rows = {plate.label_to_rowcol(lab)[0] for lab in scan_result.labels}
        cols = {plate.label_to_rowcol(lab)[1] for lab in scan_result.labels}
        if max(cols) > 6 or max(rows) > 3:
            return "96well"
        return "18well"
    highest = max(scan_result.wells) if scan_result.wells else 0
    for name, limit in (("8well", 8), ("18well", 18), ("96well", 96)):
        if highest and highest <= limit:
            return name
    return "18well"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(scan_result: Scan, plate_id: str | None = None,
           mode: str | None = None) -> str:
    """Build the commented template text for a scanned folder."""
    s = scan_result
    mode = mode or s.mode or "generate"
    plate_name = _infer_geometry(s)
    plate_id = plate_id or s.directory.name

    lines: list[str] = []
    add = lines.append

    add("# ===========================================================================")
    add(f"# plate_map.yaml: {s.directory.name}")
    add("# ===========================================================================")
    add("# This file is the ONLY place the pipeline learns what your images mean.")
    add("# Nothing is inferred from filenames beyond the well number or Seq index the")
    add("# microscope wrote: which markers an image carries is whatever you say here.")
    add("#")
    add("# It was generated by scanning the folder, so the values below reflect the")
    add("# images actually present. Lines marked  <-- EDIT  are guesses or blanks that")
    add("# only you can fill in. Run `python main.py check` when you are done: it")
    add("# validates every image against this file without processing anything.")
    add("#")
    if s.images:
        add(f"# Scanned: {len(s.images)} image(s), acquisition mode detected as '{mode}'.")
    else:
        add("# Scanned: NO images found; every value below is a placeholder.")
    add("")

    # -- geometry ----------------------------------------------------------
    add("# --- Plate geometry --------------------------------------------------------")
    add(f"# Built-in: {', '.join(_KNOWN_PLATES)}. Anything else: delete this line and use")
    add("#   geometry: {rows: 4, cols: 6}")
    add("# Wells are numbered the way NIKON Elements numbers them: row-major, 1-based.")
    add(f"plate: {plate_name}")
    add("")

    add("# --- Plate identity --------------------------------------------------------")
    add("# Defaults to the folder name. Set it EXPLICITLY when several folders hold")
    add("# different antibody panels of the SAME physical plate: they must share a")
    add("# plate_id for a marker to be poolable across them (downstream")
    add("# `markers: {ISL1: {scope: plate}}`). Different plate_id = different batch,")
    add("# and thresholds are never shared across batches.")
    if s.siblings:
        names = ", ".join(d.name for d in s.siblings[:4])
        add(f"# NOTE: sibling folder(s) found: {names}")
        add("#       If those are other panels of this same plate, give them all the")
        add("#       same plate_id below.")
    add(f"plate_id: {plate_id}                # <-- EDIT if this plate spans several folders")
    add("")

    # -- mode --------------------------------------------------------------
    add("# --- Acquisition mode ------------------------------------------------------")
    add("#   generate      NIKON JOBS 'Generate Points'. Filenames carry the well")
    add("#                 NUMBER, e.g. Well07_....nd2 -> well 7.")
    add("#   predefined    JOBS 'Predefined Points'. Filenames carry a global 0-based")
    add("#                 point index, e.g. ...Seq0042.nd2 -> sequence 42. You must")
    add("#                 map sequence -> well below.")
    add("#   spinning_disk Spinning-Disk confocal. Filenames carry an ALPHANUMERIC well")
    add("#                 label, e.g. WellC10_....nd2 -> row C, column 10.")
    add(f"mode: {mode}")
    add("")

    lines.extend(_render_mapping(s, mode))
    lines.extend(_render_stains(s))
    lines.extend(_render_wells(s, mode))
    return "\n".join(lines) + "\n"


def _render_mapping(s: Scan, mode: str) -> list[str]:
    """The Seq -> well block, which only 'predefined' mode needs."""
    lines: list[str] = []
    add = lines.append
    if mode != "predefined":
        add("# (No sequence mapping needed: this mode reads the well from the filename.)")
        add("")
        return lines

    seq = s.seq_indices
    add("# --- Sequence -> well ------------------------------------------------------")
    add("# The Seq index is ACQUISITION ORDER. Nothing in these filenames records")
    add("# WHERE on the plate a point came from, so 'well' here means simply the Nth")
    add("# well you imaged: well 1 is the first, well 2 the second, and so on. You do")
    add("# NOT have to look up which physical wells of the plate you used: imaging")
    add("# 13 wells of an 18-well plate, in any order, is still wells 1-13 below.")
    add("#")
    add("# Three ways to say how many points went to each, most explicit first -")
    add("# use exactly one:")
    add("#")
    add("#   seq_ranges       explicit well: [first, last]  (inclusive, 0-based)")
    add("#   imaged_wells     the wells you visited, in order, + points_per_well")
    add("#   points_per_well  a bare integer; assumes wells 1..K, equal counts")
    add("#")
    add("# If you DO want the plate-layout and edge-effect figures, add")
    add("#   well_numbering: plate")
    add("# and then key everything below by real plate well numbers instead.")
    add("#")
    if seq:
        add(f"# Found Seq {min(seq)}-{max(seq)} ({len(seq)} image(s)"
            f"{', with gaps' if len(seq) != max(seq) - min(seq) + 1 else ''}).")
        guess = _guess_points_per_well(len(seq))
        add(f"# A constant {guess} points/well would give {len(seq) // guess} wells.")
        add("")
        add(f"points_per_well: {guess}          # <-- EDIT: confirm against your JOBS setup")
        add("# imaged_wells: [1, 5, 9]      # …the wells you actually imaged, in order")
        add("# points_per_well: {1: 5, 5: 8, 9: 3}   # …or a variable count per well")
        add("# seq_ranges:                  # …or spell the ranges out completely")
        add("#   1: [0, 4]")
        add("#   5: [5, 12]")
    else:
        add("# No Seq indices were found in the filenames: fill this in by hand.")
        add("points_per_well: 5              # <-- EDIT")
    add("")
    return lines


def _guess_points_per_well(n_images: int) -> int:
    """A plausible constant points-per-well: the largest small divisor of the count."""
    for candidate in (12, 10, 9, 8, 6, 5, 4, 3, 2):
        if n_images % candidate == 0:
            return candidate
    return 1


def _render_stains(s: Scan) -> list[str]:
    lines: list[str] = []
    add = lines.append
    add("# --- Stain sets: channel index -> marker name ------------------------------")
    add("# 'default' applies to every well that does not name another set with")
    add("# `stain: <name>`. Channel indices are 0-based and must match the order the")
    add("# channels were saved in. DAPI is excluded from analysis automatically.")
    if s.channels:
        add(f"# Channel names read from image_metadata.json: {', '.join(s.channels)}")
        add("# They are the FLUOROPHORE names: replace them with your ANTIBODY names.")
        mapping = ", ".join(f"{i}: {name}" for i, name in enumerate(s.channels))
    else:
        add("# ('load' has not run yet, so the channel list below is a placeholder.)")
        mapping = "0: DAPI, 1: MarkerA, 2: MarkerB, 3: MarkerC"
    add("stains:")
    add(f"  default: {{{mapping}}}   # <-- EDIT: real antibody names")
    add("  # setB:  {0: DAPI, 1: MarkerA, 2: MarkerD, 3: MarkerE}")
    add("")
    add("# --- Several stain sets acquired into ONE folder (optional) ----------------")
    add("# Only needed when panels are NOT separated into their own directories. Each")
    add("# block is that panel's own sweep of the plate, so its Seq indices are local")
    add("# to the block and it carries its own well mapping. Blocks may not overlap.")
    add("# stain_sweeps:")
    add("#   default: {seq_range: [0, 59],   points_per_well: 6}")
    add("#   setB:    {seq_range: [60, 119], points_per_well: 6}")
    add("")
    return lines


def _render_wells(s: Scan, mode: str) -> list[str]:
    lines: list[str] = []
    add = lines.append
    label_mode = mode == "spinning_disk"

    add("# --- Well -> experimental condition ----------------------------------------")
    add("# Every key you invent here (treatment, dose_uM, day, timepoint, …) becomes a")
    add("# column you can group, facet and plot by downstream. Invent freely; use the")
    add("# SAME spelling everywhere.")
    add("#")
    if label_mode:
        add("# Keys are ALPHANUMERIC well labels ('C10'), because mode is spinning_disk.")
        add("# Same-row column ranges work: 'C1-C4' means C1, C2, C3, C4.")
    elif mode == "predefined":
        add("# Keys are the wells IN THE ORDER YOU IMAGED THEM: 1 is the first well")
        add("# visited, not well 1 of the plate. Ranges work: '2-4' means the 2nd, 3rd")
        add("# and 4th wells imaged. List one entry per well, in acquisition order.")
    else:
        add("# Keys are well NUMBERS read from the filenames. Ranges work: '2-4' means")
        add("# wells 2, 3 and 4.")
    add("#")
    add("# FOUR RESERVED KEYS are structural, not conditions:")
    add("#")
    add("#   stain:        picks a named entry from 'stains' above.")
    add("#")
    add("#   class:        marks this well as the negative control. The downstream")
    add("#                 config decides which values count (default: control,")
    add("#                 pluripotent, negative_control). Positivity thresholds are")
    add("#                 the P99 of the control population, so this is the single")
    add("#                 most important annotation in the file.")
    add("#")
    add("#   positive_for: markers this well is the POSITIVE control for. It stays the")
    add("#                 negative control for everything else. Use this when one")
    add("#                 condition is high for one marker and negative for the rest -")
    add("#                 e.g. a pluripotent well is SOX2-high but BRA/CDX2-negative.")
    add("#                 Without it, SOX2's threshold is computed from SOX2-positive")
    add("#                 cells and nothing is ever called positive.")
    add("#")
    add("#   control_for:  markers this well IS the negative control for. Use 'all' for")
    add("#                 every marker. An explicit marker name here beats a blanket")
    add("#                 declaration elsewhere, so you annotate only the exception.")
    add("#")
    add("# The two-line idiom for the pluripotent/SOX2 case:")
    add("#")
    add("#   wells:")
    add("#     1-6: {treatment: mTeSR, class: pluripotent, positive_for: [SOX2]}")
    add("#     7-9: {treatment: BMP50, control_for: [SOX2]}")
    add("#")
    add("# Everything else on the plate keeps inheriting the pluripotent control.")
    add("")
    add("wells:")

    entries = s.labels if label_mode else [str(w) for w in s.wells]
    if entries:
        add(f"  # {len(entries)} well(s) have images. Counts are images per well.")
        for i, key in enumerate(entries):
            count = s.points_per_well.get(key if label_mode else int(key), 0)
            marker = "  # <-- EDIT" if i == 0 else ""
            control = ", class: control" if i == 0 else ""
            quoted = f"'{key}'" if label_mode else key
            add(f"  {quoted}: {{treatment: CHANGE_ME{control}}}"
                f"   # {count} image(s){marker}")
    elif mode == "predefined":
        # Filenames carry no well, but the Seq count does imply how many wells were
        # visited. Emitting one pre-numbered line per well, each labeled with the
        # Seq range it covers, turns this from "work out your own numbering" into
        # "fill in the condition next to the range you recognize".
        seq = s.seq_indices
        guess = _guess_points_per_well(len(seq)) if seq else 0
        n_wells = len(seq) // guess if guess else 0
        if n_wells:
            add(f"  # {len(seq)} image(s) / {guess} per well = {n_wells} wells, listed in")
            add("  # acquisition order. The Seq range on each line is what you imaged.")
            for i in range(1, n_wells + 1):
                first = min(seq) + (i - 1) * guess
                edit = "  <-- EDIT" if i == 1 else ""
                control = ", class: control" if i == 1 else ""
                add(f"  {i}: {{treatment: CHANGE_ME{control}}}"
                    f"   # Seq {first:04d}-{first + guess - 1:04d}{edit}")
        else:
            add("  # No Seq indices found: list one entry per well imaged, in order.")
            add("  1: {treatment: Control, class: control}   # <-- EDIT")
            add("  2: {treatment: CHANGE_ME}")
    else:
        add("  # No images found: list your wells by hand.")
        add("  1: {treatment: Control, class: control}   # <-- EDIT")
        add("  2: {treatment: CHANGE_ME}")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_template(directory, cfg: dict | None = None, plate_id: str | None = None,
                   mode: str | None = None, force: bool = False) -> Path | None:
    """Scan *directory* and write ``plate_map.yaml`` into it.

    Returns the path written, or ``None`` when an existing map was left alone.
    """
    directory = Path(directory)
    out_path = directory / PLATE_MAP_NAME
    if out_path.exists() and not force:
        logger.warning("%s already exists, not overwriting (use --force to replace).",
                       out_path)
        return None
    if not directory.exists():
        logger.error("%s does not exist.", directory)
        return None

    result = scan(directory, cfg)
    if result.mode is None and mode is None and result.images:
        logger.warning(
            "%s: could not detect an acquisition mode from %d filename(s). "
            "Defaulting to 'generate': set `mode:` by hand or pass --mode.",
            directory.name, len(result.images),
        )

    _log_scan(result, mode or result.mode)
    out_path.write_text(render(result, plate_id=plate_id, mode=mode))
    logger.info("wrote %s", out_path)
    return out_path


def _log_scan(s: Scan, mode: str | None) -> None:
    name = s.directory.name
    if not s.images:
        logger.warning("[%s] no images found: writing a placeholder template.", name)
        return
    logger.info("[%s] %d image(s), mode: %s", name, len(s.images), mode or "unknown")
    if s.labels:
        logger.info("[%s] wells seen: %s", name, ", ".join(s.labels[:12])
                    + (" …" if len(s.labels) > 12 else ""))
    elif s.wells:
        logger.info("[%s] wells seen: %s", name,
                    ", ".join(str(w) for w in s.wells[:12])
                    + (" …" if len(s.wells) > 12 else ""))
    elif s.seq_indices:
        logger.info("[%s] Seq %d-%d (%d image(s))", name,
                    min(s.seq_indices), max(s.seq_indices), len(s.seq_indices))
    if s.channels:
        logger.info("[%s] channels: %s", name, ", ".join(s.channels))
    else:
        logger.info("[%s] channels: unknown (run 'process --steps load' first)", name)
    if s.siblings:
        logger.info("[%s] NOTE: sibling folder(s) %s; if those are other panels of "
                    "this same plate, give them all the same plate_id to allow "
                    "cross-panel gate pooling.", name,
                    ", ".join(d.name for d in s.siblings[:4]))


__all__ = ["write_template", "render", "scan", "detect_mode", "Scan"]
