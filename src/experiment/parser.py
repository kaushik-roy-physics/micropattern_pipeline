"""
Extract well / sequence identity from NIKON JOBS filenames.

Three acquisition schemes are supported (set per plate via ``mode`` in the plate
map):

* ``generate``     : 'Generate Points' in JOBS on our BAF microscopes. Filenames
  carry the well *number* directly, e.g. ``Well07_...nd2`` → well 7.
* ``predefined``   : 'Predefined Points'. Filenames carry a global, 0-based point
  index, e.g. ``...Seq0042.nd2`` → sequence 42 (Fiji-stitched exports that keep
  only ``Point0042`` are read the same way). The point→well mapping is
  resolved later from ``points_per_well`` / ``seq_ranges`` in the plate map.
* ``spinning_disk``: Spinning-Disk confocal ('GeneratePoints' of NIKON JOBS).
  Filenames carry an *alphanumeric* well label ``Well{row}{col}`` with an
  alphabetic row and a 1-based (usually zero-padded) column, e.g.
  ``WellC10_...nd2`` → row C, col 10. The label is resolved to a canonical
  1-based well number by the plate map (which knows the plate's column count).

The regexes are deliberately tolerant of leading zeros and surrounding tokens.
"""

from __future__ import annotations

import re
from pathlib import Path

# 'Well' followed by an (optionally zero-padded) number, anywhere in the stem.
_WELL_RE = re.compile(r"Well0*(\d+)", re.IGNORECASE)
# 'Seq' followed by an (optionally zero-padded) number: the global point index.
_SEQ_RE = re.compile(r"Seq0*(\d+)", re.IGNORECASE)
# Fiji-stitched exports carry the same 0-based point index as 'Point####'
# (the raw ND2 is 'Point0003_..._Seq0003'; the stitched TIFF keeps only
# 'Point0003'). Used only when no 'Seq' token is present, so a raw ND2 that has
# both is always read from its Seq.
_POINT_RE = re.compile(r"Point0*(\d+)", re.IGNORECASE)
# 'Well' followed by a row letter + an (optionally zero-padded) column number,
# e.g. 'WellC10' → ('C', 10). Anchored so 'Well07' (generate mode) never matches.
_WELL_LABEL_RE = re.compile(r"Well([A-Za-z])0*(\d+)", re.IGNORECASE)

# Accepted spellings of the Spinning-Disk mode.
SPINNING_DISK_MODES = {"spinning_disk", "spinningdisk", "spinning-disk"}


def parse_well(name: str) -> int | None:
    """Return the well number from a 'generate'-mode filename, or ``None``."""
    m = _WELL_RE.search(str(name))
    return int(m.group(1)) if m else None


def parse_seq(name: str) -> int | None:
    """Return the 0-based sequence/point index from a filename, or ``None``.

    ``Seq####`` wins; ``Point####`` is the fallback for stitched exports.
    """
    m = _SEQ_RE.search(str(name)) or _POINT_RE.search(str(name))
    return int(m.group(1)) if m else None


def parse_well_label(name: str) -> str | None:
    """Return the normalized alphanumeric well label (e.g. ``"C10"``), or ``None``.

    Normalized = uppercase row letter + column as a bare integer (leading zeros
    stripped), so ``WellE06`` and ``WellE6`` both yield ``"E6"``. This canonical
    form matches how the plate map resolves labels to well numbers.
    """
    m = _WELL_LABEL_RE.search(str(name))
    if not m:
        return None
    return f"{m.group(1).upper()}{int(m.group(2))}"


def parse_image(path, mode: str) -> dict:
    """
    Parse one filename for the given acquisition ``mode``.

    Returns a dict with:
      * ``well``      : well number for 'generate', else ``None``;
      * ``seq_index`` : 0-based point index for 'predefined', else ``None``;
      * ``well_label``: alphanumeric label for 'spinning_disk', else ``None``.
    """
    name = Path(path).name
    mode = (mode or "").lower()
    if mode == "generate":
        return {"well": parse_well(name), "seq_index": None, "well_label": None}
    if mode == "predefined":
        return {"well": None, "seq_index": parse_seq(name), "well_label": None}
    if mode in SPINNING_DISK_MODES:
        return {"well": None, "seq_index": None, "well_label": parse_well_label(name)}
    raise ValueError(
        f"Unknown acquisition mode {mode!r} "
        f"(use 'generate', 'predefined', or 'spinning_disk')."
    )


__all__ = ["parse_well", "parse_seq", "parse_well_label", "parse_image",
           "SPINNING_DISK_MODES"]
