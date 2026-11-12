"""
Rollups: cell → field → condition.

**The imaged field is the replication unit.** A well holds thousands of cells, so
treating cells as replicates makes every comparison "significant" no matter how
small the effect. Every error bar, confidence interval and test produced by this
package therefore uses the per-field value as the observation, with ``n`` = the
number of fields. Cell-level distributions are still plotted (they are the
biology), but they never set an error bar.

Three tables come out of here and everything downstream reads them:

``fields.csv``      one row per (field × marker): n cells, median/mean intensity,
                    positive fraction, plus the field's well and conditions.
``conditions.csv``  one row per (condition × marker): mean over fields ± SD, SEM
                    and a bootstrap CI, the field count, and the pooled cell count.
``wells.csv``       the same at well level: the unit you pipette, and the level
                    at which plate-position artifacts show up.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import controls, scope
from .data import Data
from .stats import bootstrap_ci

logger = logging.getLogger(__name__)

# Field-level identity carried into every rollup.
FIELD_KEYS = ["field_id", "dataset", "plate_id", "seg_mode", "stain_set", "well", "row", "col"]


def condition_columns(cells: pd.DataFrame, cond_cfg: dict) -> list[str]:
    """The plate-map columns that define a condition (config-declared)."""
    keys = cond_cfg.get("condition_key") or ["treatment"]
    keys = [keys] if isinstance(keys, str) else list(keys)
    present = [k for k in keys if k in cells.columns]
    missing = [k for k in keys if k not in cells.columns]
    if missing:
        logger.warning("conditions.condition_key names absent column(s) %s: ignoring. "
                       "Available: %s", missing, sorted(set(cells.columns)))
    if not present:
        raise ValueError(
            f"None of conditions.condition_key={keys} exist in the manifest. "
            f"Set it to one or more of: {sorted(set(cells.columns))}"
        )
    _warn_pooled_class(cells, cond_cfg, present)
    return present


_POOLED_WARNED: set = set()


def _warn_pooled_class(cells: pd.DataFrame, cond_cfg: dict, keys: list[str]) -> None:
    """Warn (once) when wells of different ``class`` fall into one condition.

    A pluripotent reference well with ``dose: 0`` and a differentiated 0 ng/mL
    well share every condition_key value, so they are silently averaged into one
    curve unless the control column is part of the key.
    """
    col = cond_cfg.get("control_column") or "class"
    if col in keys or col not in cells.columns:
        return
    classes = cells[col].astype(object).where(cells[col].notna(), None).astype(str)
    mixed = classes.groupby(condition_label(cells, keys), observed=True).nunique()
    mixed = sorted(mixed[mixed > 1].index.astype(str))
    tag = (col, tuple(keys), tuple(mixed))
    if mixed and tag not in _POOLED_WARNED:
        _POOLED_WARNED.add(tag)
        logger.warning("Condition(s) %s pool wells with different '%s' values (e.g. a "
                       "pluripotent reference with differentiated wells). Add '%s' to "
                       "conditions.condition_key to keep them apart.", mixed, col, col)


def condition_label(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Compose the condition label from one or several columns.

    With several columns, missing values are left out, so ``[class, dose]`` reads
    ``pluripotent | 0`` for a reference well and just ``100`` for the others.
    """
    if len(keys) == 1:
        return df[keys[0]].astype(str)
    label = pd.Series("", index=df.index, dtype=object)
    for key in keys:
        part = df[key].astype(str).where(df[key].notna(), "")
        sep = np.where((label != "") & (part != ""), " | ", "")
        label = label + sep + part
    return label.where(label != "", "nan").astype(str)


def field_table(data: Data, gated: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (field × marker) summary: the replicate-level table."""
    cond_cfg = cfg.get("conditions") or {}
    keys = condition_columns(data.cells, cond_cfg)

    long = gated.copy()
    long["field_id"] = data.cell_column("field_id", long["cell_uid"].to_numpy())

    grouped = long.groupby(["field_id", "marker"], observed=True)
    out = grouped.agg(
        n_cells=("value", "size"),
        median=("value", "median"),
        mean=("value", "mean"),
        p25=("value", lambda s: np.nanpercentile(s, 25) if s.notna().any() else np.nan),
        p75=("value", lambda s: np.nanpercentile(s, 75) if s.notna().any() else np.nan),
        positive_fraction=("positive", "mean"),
        # min_count=1 so "no gate for this marker" stays NaN instead of summing to 0.
        n_positive=("positive", lambda s: s.sum(min_count=1)),
    ).reset_index()

    # Field metadata: one row per field, taken from the first cell of that field.
    # The control-declaring columns travel along so figures can gray out controls
    # without re-reading the cell table.
    control_col = cond_cfg.get("control_column", "class")
    candidates = [control_col, controls.CONTROL_FOR, controls.POSITIVE_FOR]
    extra = [c for c in dict.fromkeys(candidates)
             if c in data.cells.columns and c not in keys]
    meta_cols = [c for c in FIELD_KEYS if c in data.cells.columns and c != "field_id"]
    meta = (data.cells.groupby("field_id", observed=True)[meta_cols + keys + extra]
            .first().reset_index())
    meta["condition"] = condition_label(meta, keys)

    out = out.merge(meta, on="field_id", how="left")

    # Each row's gate scope, as the single key every per-marker rollup and test
    # groups by. Markers may sit at different scope levels, so one shared tuple of
    # scope columns cannot express this: see :mod:`downstream.scope`.
    available = [c for c in scope.UNION_ORDER if c in out.columns]
    out["scope_key"] = scope.scope_keys(out, cfg, available)

    logger.info("fields: %d field × marker rows over %d field(s)",
                len(out), out["field_id"].nunique())
    return out


def _component_columns(fields: pd.DataFrame, cfg: dict) -> list[str]:
    """The plate-map columns the condition label was composed from, if present."""
    try:
        return condition_columns(fields, cfg.get("conditions") or {})
    except ValueError:          # a reloaded table may carry only `condition`
        return []


def well_table(fields: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (well × marker) summary: fields aggregated within each well."""
    keys = [c for c in ("dataset", "plate_id", "stain_set", "seg_mode", "scope_key",
                        "well", "row", "col") if c in fields.columns]
    keys += [c for c in _component_columns(fields, cfg) if c in fields.columns]
    keys += [c for c in ("condition", "marker") if c in fields.columns]
    grouped = fields.groupby(keys, observed=True, dropna=False)
    out = grouped.agg(
        n_fields=("field_id", "nunique"),
        n_cells=("n_cells", "sum"),
        median=("median", "mean"),
        median_sd=("median", "std"),
        positive_fraction=("positive_fraction", "mean"),
        positive_fraction_sd=("positive_fraction", "std"),
    ).reset_index()
    # Field-to-field heterogeneity within the well (item 6).
    out["median_cv"] = out["median_sd"] / out["median"].replace(0, np.nan)
    logger.info("wells: %d well × marker rows", len(out))
    return out


def condition_table(fields: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (condition × marker) summary with field-level error bars and CIs."""
    stats_cfg = cfg.get("stats") or {}
    n_boot = int(stats_cfg.get("n_bootstrap", 2000))
    seed = int(stats_cfg.get("random_state", 0))
    min_fields = int(stats_cfg.get("min_fields", 2))

    # Group by each marker's OWN gate scope, not by the union of everyone's scope
    # columns. A marker pooled across panels (`markers: {ISL1: {scope: plate}}`)
    # has one threshold and must get one row per condition, aggregating every
    # field the gate was estimated over; grouping on the `stain_set` its rows
    # still carry would split that condition back into one row per panel, each
    # with a third of the fields behind its CI. The pooled columns are marked `*`,
    # the same convention thresholds.csv uses.
    fields = scope.mark_pooled(fields, cfg)

    # The condition's component columns ride along as their own columns. They are
    # functionally determined by `condition` (it is their join), so adding them to
    # the group keys cannot change the grouping, but it makes the table pivotable
    # by factor and lets figures facet on one of them.
    keys = [c for c in ("dataset", "plate_id", "stain_set", "seg_mode", "scope_key")
            if c in fields.columns]
    keys += [c for c in _component_columns(fields, cfg) if c in fields.columns]
    keys += [c for c in ("condition", "marker") if c in fields.columns]
    rows: list[dict] = []
    for key_vals, grp in fields.groupby(keys, observed=True, dropna=False):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        rec = dict(zip(keys, key_vals))
        med = grp["median"].to_numpy(dtype=float)
        frac = grp["positive_fraction"].to_numpy(dtype=float)
        rec.update(
            n_fields=int(len(grp)),
            n_wells=int(grp["well"].nunique()) if "well" in grp else np.nan,
            n_cells=int(grp["n_cells"].sum()),
            median=float(np.nanmean(med)) if np.isfinite(med).any() else np.nan,
            median_sd=float(np.nanstd(med, ddof=1)) if np.isfinite(med).sum() > 1 else np.nan,
            positive_fraction=float(np.nanmean(frac)) if np.isfinite(frac).any() else np.nan,
            positive_fraction_sd=(float(np.nanstd(frac, ddof=1))
                                  if np.isfinite(frac).sum() > 1 else np.nan),
        )
        rec["median_sem"] = (rec["median_sd"] / np.sqrt(rec["n_fields"])
                             if np.isfinite(rec["median_sd"]) else np.nan)
        rec["positive_fraction_sem"] = (rec["positive_fraction_sd"] / np.sqrt(rec["n_fields"])
                                        if np.isfinite(rec["positive_fraction_sd"]) else np.nan)
        if rec["n_fields"] >= min_fields:
            lo, hi = bootstrap_ci(frac, n_boot=n_boot, seed=seed)
            rec["positive_fraction_ci_lo"], rec["positive_fraction_ci_hi"] = lo, hi
            lo, hi = bootstrap_ci(med, n_boot=n_boot, seed=seed)
            rec["median_ci_lo"], rec["median_ci_hi"] = lo, hi
        else:
            rec.update(positive_fraction_ci_lo=np.nan, positive_fraction_ci_hi=np.nan,
                       median_ci_lo=np.nan, median_ci_hi=np.nan)
        rows.append(rec)

    out = pd.DataFrame(rows)
    logger.info("conditions: %d condition × marker rows (median %d fields each)",
                len(out), int(out["n_fields"].median()) if len(out) else 0)
    return out


__all__ = ["field_table", "well_table", "condition_table", "condition_columns",
           "condition_label", "FIELD_KEYS"]
