"""
Signaling readouts relative to a reference population.

A nuclear β-catenin level or a SMAD2/3 N:C means most next to the same readout
in a known state, usually pluripotent colonies on the same plate. For every
signaling marker and scale scope (by default plate × panel, the same scope the
fate markers share an intensity scale over)::

    reference = median over reference colonies of (each colony's median value)
    relative  = value / reference                 (1 = the reference state)

The colony is the unit here as everywhere: one huge colony does not outweigh
three small ones. The reference wells are ``signaling.reference_column`` /
``reference_values``, by default the negative-control declaration
(``conditions.control_column`` / ``control_values``).

When a scope has no reference colonies, the reference of the same plate (other
panels) is used and the log says so. When a marker has no reference anywhere
(including ``reference_values: []``, no normalization wanted), it is shown
unnormalized for every scope, so one marker never mixes units: N:C as the plain
ratio, a nuclear readout as its intensity in a.u. (raw, background-subtracted
or ÷ nuclear stain, per ``signaling.nuclear_space``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import catalog
from .readouts import READOUTS

logger = logging.getLogger(__name__)

TABLE = "signaling_reference.csv"
SRC_SCOPE, SRC_PLATE, SRC_NONE = "scope", "plate (other panels)", "none"


def reference_spec(cfg: dict) -> tuple[str, list[str]]:
    """``(column, values)`` defining the reference wells."""
    sig = cfg.get("signaling") or {}
    cond = cfg.get("conditions") or {}
    column = sig.get("reference_column") or cond.get("control_column") or "class"
    values = sig.get("reference_values")
    if values is None:
        values = cond.get("control_values") or []
    values = [values] if isinstance(values, (str, int, float)) else list(values)
    return str(column), [str(v) for v in values]


def reference_mask(cells: pd.DataFrame, cfg: dict) -> np.ndarray:
    column, values = reference_spec(cfg)
    if column not in cells.columns or not values:
        return np.zeros(len(cells), bool)
    return cells[column].astype(str).isin(set(values)).to_numpy()


def _plate_key(scope_key: str, frame: pd.DataFrame) -> str:
    cols = [c for c in ("plate_id", "seg_mode") if c in frame.columns]
    row = frame.iloc[0]
    return "|".join(str(row[c]) for c in cols)


def reference_table(long: pd.DataFrame, cells: pd.DataFrame, cfg: dict,
                    matches: dict) -> pd.DataFrame:
    """One row per signaling marker × scope: the reference level and where it came from.

    *long* holds ``cell_uid, marker, value, field_id, scope_key`` (and
    ``plate_id`` / ``seg_mode`` when present) for in-colony nuclei.
    """
    sig = catalog.settings(cfg)
    is_ref = reference_mask(cells, cfg)
    column, values = reference_spec(cfg)
    rows = []
    for marker, match in matches.items():
        block = long[long["marker"].astype(str) == marker]
        if block.empty:
            continue
        ref_rows = is_ref[block["cell_uid"].to_numpy()]
        present = sorted(cells.loc[is_ref, column].astype(str).unique()) \
            if column in cells.columns and is_ref.any() else []
        per_colony = (block[ref_rows].groupby(["scope_key", "field_id"], observed=True)["value"]
                      .median().dropna().reset_index())
        by_plate = {}
        for key, part in block.groupby("scope_key", observed=True):
            by_plate.setdefault(_plate_key(key, part), []).append(str(key))
        for key, part in block.groupby("scope_key", observed=True):
            key = str(key)
            own = per_colony[per_colony["scope_key"].astype(str) == key]
            source = SRC_SCOPE
            if own.empty:
                siblings = by_plate.get(_plate_key(key, part), [])
                own = per_colony[per_colony["scope_key"].astype(str).isin(siblings)]
                source = SRC_PLATE if len(own) else SRC_NONE
            level = float(own["value"].median()) if len(own) else np.nan
            if np.isfinite(level) and level <= 0:
                level, source = np.nan, SRC_NONE
            rows.append({"marker": marker, "scope_key": key, "readout": match.readout,
                         "space": READOUTS[match.readout].space(sig),
                         "reference_column": column,
                         "reference_values": "/".join(present or values),
                         "reference": level, "n_reference_colonies": int(len(own)),
                         "source": source})
    table = pd.DataFrame(rows, columns=["marker", "scope_key", "readout", "space",
                                        "reference_column", "reference_values", "reference",
                                        "n_reference_colonies", "source", "relative"])
    if table.empty:
        return table
    # One unit per marker: relative only if every scope of the marker has a reference.
    complete = table.groupby("marker")["reference"].transform(lambda s: bool(np.isfinite(s).all()))
    table["relative"] = complete.astype(bool)
    for marker, part in table.groupby("marker"):
        if not part["relative"].iloc[0] and not values:
            logger.info("signaling: %s shown unnormalized (no reference wells configured).",
                        marker)
        elif not part["relative"].iloc[0]:
            missing = part.loc[~np.isfinite(part["reference"]), "scope_key"].tolist()
            logger.warning("signaling: %s has no reference colonies (%s in %s) for %s; it is "
                           "shown unnormalized everywhere so its units stay the same.",
                           marker, column, values, ", ".join(map(str, missing)) or "any scope")
        else:
            for _, r in part[part["source"] == SRC_PLATE].iterrows():
                logger.warning("signaling: %s [%s] has no reference colonies in its own panel; "
                               "using the reference of the same plate.", marker, r["scope_key"])
            for _, r in part.iterrows():
                logger.info("signaling: %s [%s] reference %s = %.4g (%d colonies, %s)",
                            marker, r["scope_key"], r["reference_values"], r["reference"],
                            r["n_reference_colonies"], r["source"])
    return table


def relative_values(long: pd.DataFrame, table: pd.DataFrame,
                    scaled: pd.Series) -> pd.Series:
    """The analysis value of every row: relative to reference for signaling markers.

    Rows of standard markers keep *scaled* (the shared-scale value). A signaling
    marker without a complete reference keeps its own value: the plain ratio for
    N:C, the intensity in a.u. for a nuclear readout.
    """
    out = scaled.astype(np.float64).copy()
    if table is None or table.empty:
        return out
    marker = long["marker"].astype(str)
    idx = pd.MultiIndex.from_frame(table[["marker", "scope_key"]].astype(str))
    ref = pd.Series(table["reference"].to_numpy(np.float64), index=idx)
    rel = pd.Series(table["relative"].astype(bool).to_numpy(), index=idx)
    keys = pd.MultiIndex.from_arrays([marker, long["scope_key"].astype(str)])
    ref_v = ref.reindex(keys).to_numpy(np.float64)
    rel_v = np.nan_to_num(rel.astype(float).reindex(keys).to_numpy(np.float64)) > 0
    sig_rows = marker.isin(set(table["marker"].astype(str))).to_numpy()
    value = long["value"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[sig_rows & rel_v] = (value / ref_v)[sig_rows & rel_v]
    plain = sig_rows & ~rel_v
    out[plain] = value[plain]
    return out


def unit_labels(table: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """``marker → (axis label, short unit)`` per signaling marker, saying what 1 means."""
    labels = {}
    if table is None or table.empty:
        return labels
    for marker, part in table.groupby("marker"):
        r = part.iloc[0]
        base = READOUTS[r["readout"]].label(str(marker))
        short = "N:C" if r["readout"] == catalog.NC else "nuclear"
        if bool(r["relative"]):
            ref = str(r["reference_values"])
            labels[str(marker)] = (f"{base} (÷ {ref})", f"{short} ÷ {ref}")
        elif r["readout"] == catalog.NC:
            labels[str(marker)] = (base, short)
        else:
            unit = {"nuc_bgsub": "background-subtracted, a.u.", "raw": "raw, a.u.",
                    "dapi_norm": "÷ nuclear stain"}.get(str(r["space"]), "a.u.")
            labels[str(marker)] = (f"{base} ({unit})", f"{short}, {unit}")
    return labels


def absolute_markers(table: pd.DataFrame) -> list[str]:
    """Signaling markers shown in their own intensity units (not 0-1, not ÷ reference)."""
    if table is None or table.empty:
        return []
    rows = (~table["relative"].astype(bool)) & (table["readout"] != catalog.NC)
    return sorted(set(table.loc[rows, "marker"].astype(str)))


def reference_markers(table: pd.DataFrame) -> list[str]:
    """Signaling markers shown relative to a reference (drawn with a line at 1)."""
    if table is None or table.empty:
        return []
    return sorted(set(table.loc[table["relative"].astype(bool), "marker"].astype(str)))


__all__ = ["TABLE", "reference_spec", "reference_mask", "reference_table", "relative_values",
           "unit_labels", "reference_markers", "absolute_markers"]
