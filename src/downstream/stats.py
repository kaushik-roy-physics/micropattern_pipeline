"""
Statistics on field-level values.

Everything here takes an array of **per-field** observations, never per-cell
values: see :mod:`downstream.aggregate` for why. All routines are vectorized
over the bootstrap dimension (no Python resampling loop).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0, statistic=np.mean) -> tuple[float, float]:
    """Percentile bootstrap CI of *statistic* over field-level *values*.

    Resampling is done as one ``(n_boot, n)`` index matrix, so the whole bootstrap
    is a single vectorized reduction.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def cohens_d(a, b) -> float:
    """Standardized difference between two field-level samples."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def mann_whitney(a, b) -> float:
    """Two-sided Mann-Whitney U p-value; NaN when either sample is too small."""
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def benjamini_hochberg(pvalues) -> np.ndarray:
    """BH-adjusted q-values, NaNs preserved in place."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    n = ranked.size
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    adjusted = np.empty_like(q)
    adjusted[order] = np.clip(q, 0, 1)
    out[finite] = adjusted
    return out


def compare_to_reference(fields: pd.DataFrame, value_col: str, reference: str,
                         group_col: str = "condition", extra_keys=("marker",),
                         seed: int = 0) -> pd.DataFrame:
    """Effect size + test of every condition against a reference, per marker.

    Field-level values on both sides; p-values are BH-adjusted within each marker.
    """
    keys = [k for k in extra_keys if k in fields.columns]
    rows: list[dict] = []
    for key_vals, grp in fields.groupby(keys, observed=True, dropna=False) if keys else [((), fields)]:
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        ref = grp.loc[grp[group_col].astype(str) == str(reference), value_col].to_numpy(float)
        for name, sub in grp.groupby(group_col, observed=True):
            if str(name) == str(reference):
                continue
            vals = sub[value_col].to_numpy(float)
            rows.append({
                **dict(zip(keys, key_vals)),
                group_col: name, "value": value_col,
                "n_fields": int(np.isfinite(vals).sum()),
                "n_fields_reference": int(np.isfinite(ref).sum()),
                "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                "mean_reference": float(np.nanmean(ref)) if np.isfinite(ref).any() else np.nan,
                "difference": (float(np.nanmean(vals) - np.nanmean(ref))
                               if np.isfinite(vals).any() and np.isfinite(ref).any() else np.nan),
                "cohens_d": cohens_d(vals, ref),
                "p_value": mann_whitney(vals, ref),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = np.nan
        for _, idx in out.groupby(keys, observed=True, dropna=False).groups.items() if keys else [(None, out.index)]:
            out.loc[idx, "q_value"] = benjamini_hochberg(out.loc[idx, "p_value"])
    return out


def factor_effects(fields: pd.DataFrame, value_col: str, factors: list[str],
                   extra_keys=("marker",), seed: int = 0) -> pd.DataFrame:
    """Marginal effect of each boolean/2-level factor on a field-level readout.

    For a combinatorial screen (``bmp``, ``noggin``, ``fgf8``, … as separate
    plate-map columns) this is the natural first look: the average difference a
    factor makes across every background it appears in.
    """
    keys = [k for k in extra_keys if k in fields.columns]
    rows: list[dict] = []
    groups = fields.groupby(keys, observed=True, dropna=False) if keys else [((), fields)]
    for key_vals, grp in groups:
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        for factor in factors:
            if factor not in grp.columns:
                continue
            levels = pd.unique(grp[factor].dropna())
            if len(levels) != 2:
                continue
            lo, hi = sorted(levels, key=lambda v: str(v))
            a = grp.loc[grp[factor] == hi, value_col].to_numpy(float)
            b = grp.loc[grp[factor] == lo, value_col].to_numpy(float)
            rows.append({
                **dict(zip(keys, key_vals)),
                "factor": factor, "level_high": str(hi), "level_low": str(lo),
                "value": value_col,
                "n_fields_high": int(np.isfinite(a).sum()),
                "n_fields_low": int(np.isfinite(b).sum()),
                "mean_high": float(np.nanmean(a)) if np.isfinite(a).any() else np.nan,
                "mean_low": float(np.nanmean(b)) if np.isfinite(b).any() else np.nan,
                "effect": (float(np.nanmean(a) - np.nanmean(b))
                           if np.isfinite(a).any() and np.isfinite(b).any() else np.nan),
                "cohens_d": cohens_d(a, b),
                "p_value": mann_whitney(a, b),
            })
    out = pd.DataFrame(rows)
    if len(out):
        # Multiplicity is corrected within each (scope × marker) family (the set
        # of factors tested on one readout), not globally across unrelated ones.
        if keys:
            out["q_value"] = np.nan
            for _, idx in out.groupby(keys, observed=True, dropna=False).groups.items():
                out.loc[idx, "q_value"] = benjamini_hochberg(out.loc[idx, "p_value"])
        else:
            out["q_value"] = benjamini_hochberg(out["p_value"])
    return out


def detect_factors(cells: pd.DataFrame, candidates: list[str] | None = None,
                   max_levels: int = 2) -> list[str]:
    """Plate-map columns that look like experimental factors (exactly 2 levels)."""
    from .data import ID_COLUMNS
    cols = candidates or [c for c in cells.columns if c not in ID_COLUMNS]
    out = []
    for col in cols:
        if col not in cells.columns:
            continue
        levels = pd.unique(cells[col].dropna())
        if 2 <= len(levels) <= max_levels:
            out.append(col)
    return out


__all__ = ["bootstrap_ci", "cohens_d", "mann_whitney", "benjamini_hochberg",
           "compare_to_reference", "factor_effects", "detect_factors"]
