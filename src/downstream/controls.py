"""
Which wells are the negative control: resolved **per marker**.

The naive model is one boolean per well: this well is a control, therefore it is
the negative control for everything measured in it. That breaks the moment a
condition is a negative control for some markers and a positive control for
others, which is the normal situation in a pluripotency experiment:

============  ===================  ==========================
marker        mTeSR (pluripotent)  BMP ≥10 ng/ml
============  ===================  ==========================
SOX2          **positive** ctrl    **negative** ctrl
BRA           negative ctrl        positive ctrl
CDX2          negative ctrl        positive ctrl
============  ===================  ==========================

Gate SOX2 off the pluripotent well and its threshold lands at the P99 of a
saturated population: nothing is called positive and the failure is silent.
This is not an edge case: it is exactly Table S1 of the lab's own
``Cell_fate_classification.pdf``, whose Eq. (2) reads ``t = Negative control for
SOX2``: the paper's control has always been marker-indexed.

Declaring it without annotating every well
------------------------------------------
The blanket declaration still does the work; you only ever write down the
**exceptions**. Resolution runs most-specific-first:

1. **Plate map, explicit.** A well carrying ``control_for: [SOX2]`` is SOX2's
   negative control. An explicit mention beats a blanket one, so the wells that
   said ``control_for: all`` (or carry the control ``class``) are demoted for
   SOX2 alone and keep serving every other marker.
2. **Config.** ``conditions.marker_controls.SOX2.control_values`` names the well
   class to use instead, for reanalysis without touching a plate map. Skipped
   when the plate map already spoke, and a warning says so.
3. **Blanket, minus positive controls.** The usual ``control_column`` /
   ``control_values`` class, minus any well declaring ``positive_for: [SOX2]``.
   That one key is the whole fix for the table above::

       wells:
         1-6: {treatment: mTeSR, class: pluripotent, positive_for: [SOX2]}
         7-9: {treatment: BMP50, control_for: [SOX2]}

Everything else on the plate keeps inheriting the pluripotent control, and no
marker name is hardcoded anywhere in this package.

When a marker ends up with no control at all, the mask is empty and
:mod:`downstream.thresholds` falls through to its bimodality-validated GMM: the
existing behavior, reached honestly rather than by silently gating off the wrong
population. :meth:`Controls.source_for` records which rule fired so
``thresholds.csv`` can be audited.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reserved plate-map well keys, carried through the manifest as ';'-joined strings.
CONTROL_FOR = "control_for"
POSITIVE_FOR = "positive_for"

# Token meaning "every marker in this well's panel".
ALL = "all"
SEP = ";"

# Provenance labels recorded in thresholds.csv.
SRC_PLATE_MAP = "plate_map:control_for"
SRC_CONFIG = "config:marker_controls"
SRC_CLASS = "conditions.control_column"
SRC_NONE = "none"


def split_tokens(value) -> set[str]:
    """Parse one ``control_for`` / ``positive_for`` cell into a lowercase token set."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return set()
    return {t.strip().lower() for t in text.split(SEP) if t.strip()}


def normalize_declaration(value) -> str:
    """Normalize a plate-map ``control_for``/``positive_for`` value to a stored string.

    Accepts a bare string (``SOX2``), a list (``[SOX2, BRA]``) or the keyword
    ``all``; returns ``"SOX2;BRA"``. Order is preserved so the manifest reads the
    way it was written.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return ALL if value else ""
    items = value if isinstance(value, (list, tuple, set)) else [value]
    seen: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.append(text)
    return SEP.join(seen)


class Controls:
    """Per-marker negative-control masks over a ``cells`` table.

    Masks are boolean arrays positionally aligned with ``cells`` (the same
    ``cell_uid``-as-row-position invariant the rest of the package relies on), so
    a caller indexes them with ``cell_uid`` exactly like the old single mask.
    """

    def __init__(self, cells: pd.DataFrame, cfg: dict):
        self.cfg = cfg or {}
        self.cond_cfg = self.cfg.get("conditions") or {}
        self._n = len(cells)
        self._cells = cells
        self._cache: dict[str, np.ndarray] = {}
        self._sources: dict[str, str] = {}

        # -- the blanket class declaration ---------------------------------
        column = self.cond_cfg.get("control_column", "class")
        values = {str(v).lower() for v in (self.cond_cfg.get("control_values")
                                           or ["control"])}
        self.control_column = column
        self.control_values = values
        if column in cells.columns:
            self._class_mask = (cells[column].astype(str).str.lower()
                                .isin(values).to_numpy())
        else:
            self._class_mask = np.zeros(self._n, dtype=bool)

        # -- the two reserved plate-map columns ----------------------------
        self._control_for = self._token_index(cells, CONTROL_FOR)
        self._positive_for = self._token_index(cells, POSITIVE_FOR)

        self._marker_controls = {
            str(k): dict(v or {}) for k, v in
            (self.cond_cfg.get("marker_controls") or {}).items()
        }

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _token_index(cells: pd.DataFrame, column: str):
        """``(codes, token_sets)`` for a declaration column, or ``None`` when absent.

        Factorized once so a per-marker lookup is a take over a handful of unique
        strings rather than a string operation over every cell.
        """
        if column not in cells.columns:
            return None
        codes, uniques = pd.factorize(cells[column].astype(str), use_na_sentinel=False)
        return codes, [split_tokens(u) for u in uniques]

    def _declared(self, index, marker: str, include_all: bool) -> np.ndarray:
        """Rows whose declaration names *marker* (and optionally the ``all`` token)."""
        if index is None:
            return np.zeros(self._n, dtype=bool)
        codes, token_sets = index
        wanted = marker.strip().lower()
        hit = np.array([(wanted in tokens) or (include_all and ALL in tokens)
                        for tokens in token_sets], dtype=bool)
        if not hit.any():
            return np.zeros(self._n, dtype=bool)
        return hit[codes]

    def _config_mask(self, marker: str) -> np.ndarray | None:
        spec = self._marker_controls.get(marker)
        if not spec:
            return None
        column = str(spec.get("control_column", self.control_column))
        values = {str(v).lower() for v in (spec.get("control_values")
                                           or self.control_values)}
        if column not in self._cells.columns:
            logger.warning(
                "conditions.marker_controls.%s names column %r, which is not in the "
                "manifest: falling back to the default control for this marker.",
                marker, column,
            )
            return None
        return self._cells[column].astype(str).str.lower().isin(values).to_numpy()

    # -- the public question ----------------------------------------------

    def mask_for(self, marker: str) -> np.ndarray:
        """Boolean mask of cells that are *marker*'s negative control."""
        marker = str(marker)
        if marker in self._cache:
            return self._cache[marker]

        named = self._declared(self._control_for, marker, include_all=False)
        config = self._config_mask(marker)

        if named.any():
            mask, source = named, SRC_PLATE_MAP
            if config is not None:
                logger.warning(
                    "%s: plate map declares control_for: %s and config declares "
                    "conditions.marker_controls.%s: the plate map wins. Remove one.",
                    marker, marker, marker,
                )
        elif config is not None:
            mask, source = config, SRC_CONFIG
        else:
            blanket = self._class_mask | self._declared(self._control_for, marker,
                                                        include_all=True)
            demoted = self._declared(self._positive_for, marker, include_all=True)
            mask = blanket & ~demoted
            source = SRC_CLASS if mask.any() else SRC_NONE
            if demoted.any() and source == SRC_CLASS:
                source += " − positive_for"
            elif demoted.any():
                logger.warning(
                    "%s: every candidate control well declares positive_for: %s, so "
                    "the marker has no negative control left. It will fall through to "
                    "the GMM fallback. Declare control_for: [%s] on the well that is "
                    "its negative control.",
                    marker, marker, marker,
                )

        self._cache[marker] = mask
        self._sources[marker] = source
        return mask

    def source_for(self, marker: str) -> str:
        """Which rule supplied *marker*'s control (populated by :meth:`mask_for`)."""
        if str(marker) not in self._sources:
            self.mask_for(marker)
        return self._sources[str(marker)]

    def any_mask(self) -> np.ndarray:
        """Cells that are a control for *something*: for whole-plate figures.

        Marker-agnostic on purpose: a plate-position heatmap or a condition
        palette wants "this well is some kind of control", not a per-marker
        answer. Per-marker figures use :meth:`mask_for` instead.
        """
        mask = self._class_mask.copy()
        if self._control_for is not None:
            codes, token_sets = self._control_for
            hit = np.array([bool(tokens) for tokens in token_sets], dtype=bool)
            if hit.any():
                mask |= hit[codes]
        return mask

    def is_marker_specific(self) -> bool:
        """True when anything in this experiment actually differentiates by marker.

        The declaration *columns* exist in every manifest, so their presence proves
        nothing: only a non-empty token set does.
        """
        if self._marker_controls:
            return True
        return any(any(tokens for tokens in index[1])
                   for index in (self._control_for, self._positive_for)
                   if index is not None)

    def summary(self, markers) -> pd.DataFrame:
        """One row per marker: cell/well counts and the rule that supplied them.

        Written to ``control_assignment.csv``: a pooled or marker-specific gate
        is only trustworthy if you can see which wells fed it.
        """
        rows: list[dict] = []
        well_col = "well" if "well" in self._cells.columns else None
        cond_col = "condition" if "condition" in self._cells.columns else None
        for marker in markers:
            mask = self.mask_for(marker)
            rec = {
                "marker": marker,
                "n_control_cells": int(mask.sum()),
                "n_control_wells": (int(self._cells.loc[mask, well_col].nunique())
                                    if well_col else np.nan),
                "control_source": self.source_for(marker),
            }
            if well_col:
                wells = sorted(self._cells.loc[mask, well_col].unique().tolist())
                rec["control_wells"] = ", ".join(str(w) for w in wells)
            if cond_col:
                conds = sorted(str(c) for c in self._cells.loc[mask, cond_col].unique())
                rec["control_conditions"] = ", ".join(conds)
            rows.append(rec)
        return pd.DataFrame(rows)


def control_mask(cells: pd.DataFrame, cond_cfg: dict) -> np.ndarray:
    """Marker-agnostic control mask: the pre-per-marker behavior, kept for callers
    that genuinely want "is this well any kind of control" (figures, palettes)."""
    return Controls(cells, {"conditions": cond_cfg or {}}).any_mask()


__all__ = ["Controls", "control_mask", "split_tokens", "normalize_declaration",
           "CONTROL_FOR", "POSITIVE_FOR", "ALL", "SEP",
           "SRC_PLATE_MAP", "SRC_CONFIG", "SRC_CLASS", "SRC_NONE"]
