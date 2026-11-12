"""
Plate geometry and vectorized well ↔ (row, col) mapping.

Wells follow the **NIKON Elements** numbering convention: sequential, row-major,
1-based. For a plate with ``C`` columns, well ``w`` sits at::

    row = (w - 1) // C + 1
    col = (w - 1) %  C + 1

Built-in Ibidi geometries (as numbered by NIKON Elements):

============  ======  ======  ==============================================
Plate         rows    cols    numbering
============  ======  ======  ==============================================
8-well        2       4       01-04 (row 1), 05-08 (row 2)
18-well       3       6       01-06, 07-12, 13-18
96-well       8       12      01-12, 13-24, …, 85-96
============  ======  ======  ==============================================

Anything else can be driven by an explicit ``rows`` / ``cols`` override.
"""

from __future__ import annotations

import re

import numpy as np

# Alphanumeric well label: a single row letter + an (optionally zero-padded)
# column number, e.g. 'C10' → ('C', 10). Used by the Spinning-Disk scheme.
_LABEL_RE = re.compile(r"^\s*([A-Za-z])0*(\d+)\s*$")

# Keyed by the digits in the plate name ("8", "18", "96") → (rows, cols).
_BY_DIGITS: dict[str, tuple[int, int]] = {
    "8": (2, 4),
    "18": (3, 6),
    "96": (8, 12),
}


def resolve_geometry(
    plate: str | int | None = None,
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[int, int]:
    """
    Resolve (rows, cols) from a named plate and/or an explicit override.

    An explicit ``rows`` *and* ``cols`` always win; otherwise the plate name is
    matched by its digits ("8well", "18-well", 96 … all work).
    """
    if rows and cols:
        return int(rows), int(cols)
    digits = "".join(ch for ch in str(plate) if ch.isdigit()) if plate is not None else ""
    if digits in _BY_DIGITS:
        return _BY_DIGITS[digits]
    raise ValueError(
        f"Unknown plate {plate!r}. Use one of "
        f"{sorted(_BY_DIGITS)} (e.g. '18well') or set geometry.rows/geometry.cols."
    )


def n_wells(rows: int, cols: int) -> int:
    return int(rows) * int(cols)


def well_to_rowcol(well, cols: int):
    """
    Map 1-based well number(s) to 1-based (row, col). Vectorized: ``well`` may be
    a scalar or any array-like; the return mirrors the input shape.
    """
    w = np.asarray(well, dtype=np.int64)
    row = (w - 1) // int(cols) + 1
    col = (w - 1) % int(cols) + 1
    if np.isscalar(well) or w.ndim == 0:
        return int(row), int(col)
    return row, col


def label_to_rowcol(label) -> tuple[int, int]:
    """Map an alphanumeric well label to 1-based ``(row, col)``.

    ``'A1'`` → (1, 1); ``'C10'`` → (3, 10); ``'E06'`` → (5, 6). The row letter is
    case-insensitive; leading zeros on the column are ignored. Raises ``ValueError``
    on anything that is not ``<letter><digits>``.
    """
    m = _LABEL_RE.match(str(label))
    if not m:
        raise ValueError(
            f"Invalid well label {label!r}: expected a row letter followed by a "
            f"column number (e.g. 'C10')."
        )
    row = ord(m.group(1).upper()) - ord("A") + 1
    col = int(m.group(2))
    return row, col


def label_to_well(label, cols: int) -> int:
    """Map an alphanumeric well label to a canonical, 1-based, row-major well number.

    Consistent with :func:`well_to_rowcol` for the same ``cols``: for a 96-well
    plate (12 cols), ``'C10'`` → row 3, col 10 → ``(3-1)*12 + 10 = 34``.
    """
    row, col = label_to_rowcol(label)
    return (row - 1) * int(cols) + col


def expand_label_key(key, cols: int) -> list[int]:
    """Expand a Spinning-Disk plate-map well key into canonical well number(s).

    Accepts a single label (``'C10'``) or a same-row column range (``'C10-C12'``
    → C10, C11, C12). Range endpoints must share a row letter.
    """
    s = str(key).strip()
    if "-" in s:
        lo_s, hi_s = (part.strip() for part in s.split("-", 1))
        lo_row, lo_col = label_to_rowcol(lo_s)
        hi_row, hi_col = label_to_rowcol(hi_s)
        if lo_row != hi_row:
            raise ValueError(
                f"Well range {s!r} must stay within one row "
                f"(got rows {lo_s[0]} and {hi_s[0]})."
            )
        if hi_col < lo_col:
            raise ValueError(f"Invalid well range {s!r} (end column < start column).")
        return [(lo_row - 1) * int(cols) + c for c in range(lo_col, hi_col + 1)]
    return [label_to_well(s, cols)]


def validate_wells(wells, rows: int, cols: int) -> None:
    """Raise if any well number falls outside ``1 .. rows*cols``."""
    total = n_wells(rows, cols)
    arr = np.asarray(list(wells), dtype=np.int64)
    if arr.size == 0:
        return
    bad = np.unique(arr[(arr < 1) | (arr > total)])
    if bad.size:
        raise ValueError(
            f"Well number(s) {bad.tolist()} out of range for a "
            f"{rows}×{cols} plate (valid: 1-{total})."
        )


__all__ = ["resolve_geometry", "n_wells", "well_to_rowcol", "validate_wells",
           "label_to_rowcol", "label_to_well", "expand_label_key"]
