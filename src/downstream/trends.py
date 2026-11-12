"""
Local trends of radial profiles: at each position, is a marker going up or down?

The multi-marker figures stack every marker of a colony (or well) on one
distance axis, each on its own y-range. Reading "at 80 µm from the edge, ISL1
falls while SOX2 rises" off stacked curves is possible but slow, so each
profile also gets a **trend strip**: the local slope of the curve, divided by its
own standard error.

For bin *i* with ``window`` *k*::

    slope_i = (m[i+k] − m[i−k]) / (x[i+k] − x[i−k])
    z_i     = (m[i+k] − m[i−k]) / sqrt(se[i+k]² + se[i−k]²)

(one-sided at the ends of the curve). ``z`` is scale-free: a marker that spans
0.05 of the shared scale and one that spans 0.8 are compared on how clearly they
change, not on how much, which is what lets markers with different y-ranges sit
in one strip **without rescaling any colony**. ``|z| ≥ z_crit`` is marked as a
trend (``direction`` +1 / −1, else 0).

The ``se`` is the one the profile already carries: over nuclei for a single
colony's curve (how well that colony's own curve is determined), over colonies
for a well mean. The z-values are descriptive, not a test: they are not
corrected for the many positions and markers looked at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "window_bins": 1,         # compare bin i-k with bin i+k
    "z_crit": 1.96,           # |z| at or above this is marked as a trend
    "z_display_max": 4.0,     # color scale of the strip saturates here
    "colony_figures": True,   # one stacked figure per colony (µm grid)
    "well_figures": True,     # one stacked figure per well (normalized grid)
}


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("multimarker") or {}).items() if v is not None})
    if int(out["window_bins"]) < 1:
        raise ValueError("multimarker.window_bins must be at least 1")
    if float(out["z_crit"]) <= 0:
        raise ValueError("multimarker.z_crit must be positive")
    return out


def local_trend(x, mean, se, window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(slope, z)`` at every point of one curve (NaN where it cannot be formed)."""
    x = np.asarray(x, np.float64)
    m = np.asarray(mean, np.float64)
    s = np.asarray(se, np.float64)
    n = x.size
    slope = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - window), min(n - 1, i + window)
        if hi <= lo or not (np.isfinite(m[lo]) and np.isfinite(m[hi])):
            continue
        d = m[hi] - m[lo]
        dx = x[hi] - x[lo]
        if dx > 0:
            slope[i] = d / dx
        err = np.sqrt(s[lo] ** 2 + s[hi] ** 2)
        if np.isfinite(err) and err > 0:
            z[i] = d / err
    return slope, z


def trend_table(curves: pd.DataFrame, by: list[str], prm: dict,
                mean: str = "mean", se: str = "sem") -> pd.DataFrame:
    """Add ``slope``, ``z`` and ``direction`` to every curve point.

    One curve per *by* × ``marker`` × ``grid``, ordered by ``bin``. A gap in the
    curve (a NaN bin) stays a gap: the slope is never bridged across it.
    """
    window = int(prm["window_bins"])
    crit = float(prm["z_crit"])
    keys = [k for k in dict.fromkeys(list(by) + ["marker", "grid"]) if k in curves]
    parts = []
    for _, part in curves.groupby(keys, observed=True, dropna=False, sort=False):
        part = part.sort_values("bin").copy()
        slope, z = local_trend(part["bin_center"], part[mean], part[se], window)
        part["slope"] = slope
        part["z"] = z
        parts.append(part)
    if not parts:
        return curves.assign(slope=np.nan, z=np.nan, direction=0)
    out = pd.concat(parts, ignore_index=True)
    zz = out["z"].to_numpy(np.float64)
    out["direction"] = np.where(np.isfinite(zz) & (np.abs(zz) >= crit), np.sign(zz), 0).astype(int)
    return out


__all__ = ["DEFAULTS", "settings", "local_trend", "trend_table"]
