"""
Border roughness and angular registration: is a gradient real, or an average?

A radial profile averages over every angle. If a border is sharp at the scale of
cells but its radius wanders around the colony, the average turns the step into
a ramp, and every colony added makes it smoother. So, per colony and boundary
marker:

1. **Border radius against angle.** The colony is cut into ``n_sectors`` angular
   sectors and the border position x50 is fitted in each (gate calls, binomial
   logistic). The slope is shared by all sectors of the colony and only the
   position is free per sector, which keeps a sector of ~150 nuclei well
   determined; slope and positions are re-estimated alternately (the slope from
   the registered nuclei, below).
2. **Roughness.** Around the angle the sector positions are modeled as
   ``m + a cos θ + b sin θ + residual``. The first harmonic, ``decenter_um =
   √(a² + b²)``, is what a pattern (or a fitted colony center) slightly off
   center produces; it is reported but is not roughness. ``roughness_um`` is the
   SD of the residual with the sectors' own fitting noise subtracted
   (``√(max(0, var(residual) − mean(se²)))``); ``raw_sd_um`` is the plain SD of the
   sector positions.
3. **Registration.** Each nucleus' distance is re-expressed relative to its own
   sector's border, ``d − x50(sector)``, and the logistic is fitted again on the
   pooled, registered nuclei. ``width_registered_um`` against the ordinary
   ``width_unregistered_um`` (same nuclei, ``d − mean border``) answers the
   question: a registered width much smaller than the unregistered one means the
   apparent gradient is mostly the border's wandering, not a graded transition
   in any one place (``width_ratio`` = registered / unregistered).

The registered and unregistered profiles (positive fraction and intensity
against distance from the border) are also returned per colony, for averaging
across colonies by condition.

Sector resolution: with 16 sectors a border 250 µm from the center is sampled
every ~100 µm of arc, so roughness at shorter wavelengths (cell scale) is not
seen here; it stays inside each sector's fitted width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .boundary import LN81, _sigmoid, fit_logistic, transition_segment

DEFAULTS = {
    "enabled": True,
    "n_sectors": 16,              # angular sectors for border radius vs angle
    "min_sector_nuclei": 40,      # nuclei in a sector's fitting window, else no position there
    "min_sectors": 8,             # sectors with a position, else no roughness for the colony
    "window_widths": 2.0,         # fit window: x50 ± this many widths (+ 20 µm)
    "profile_half_um": 150.0,     # registered profiles span ± this around the border
    "profile_bin_um": 10.0,
    "iterations": 3,              # slope ⇄ positions alternations
}


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in ((cfg.get("boundary") or {}).get("roughness") or {}).items()
                if v is not None})
    if int(out["n_sectors"]) < 6:
        raise ValueError("boundary.roughness.n_sectors must be at least 6")
    return out


def _position_fixed_slope(d, y, c: float):
    """MLE of x50 for binary *y* against *d* with slope *c* fixed; ``(x50, se)``.

    One-parameter Newton on the intercept; se from the Fisher information,
    ``1 / (|c| √Σ p(1 − p))``.
    """
    if d.size < 3 or y.sum() == 0 or y.sum() == y.size:
        return np.nan, np.nan
    a = -c * float(np.median(d))
    for _ in range(40):
        p = np.clip(_sigmoid(a + c * d), 1e-9, 1 - 1e-9)
        info = float(np.sum(p * (1 - p)))
        if info <= 0:
            return np.nan, np.nan
        step = float(np.sum(y - p)) / info
        a += step
        if abs(step) < 1e-8:
            break
    p = _sigmoid(a + c * d)
    info = float(np.sum(p * (1 - p)))
    return -a / c, (1.0 / (abs(c) * np.sqrt(info)) if info > 0 else np.nan)


def _binned_fit(dd, y, bin_um, k_max):
    """Free-slope logistic on binned binary data; ``(x50, c)`` or ``None``."""
    if dd.size < 20:
        return None
    lo, hi = np.floor(dd.min() / bin_um), np.ceil(dd.max() / bin_um)
    edges = np.arange(lo, hi + 1) * bin_um
    if edges.size < 4:
        return None
    idx = np.clip(np.digitize(dd, edges) - 1, 0, edges.size - 2)
    n = np.bincount(idx, minlength=edges.size - 1).astype(float)
    k = np.bincount(idx, weights=y, minlength=edges.size - 1)
    return fit_logistic(0.5 * (edges[:-1] + edges[1:]), n, k, k_max)


def harmonic_roughness(theta_s, x_s, se_s):
    """``(mean, decenter, roughness, raw_sd)`` from sector positions.

    Weighted fit of ``m + a cos θ + b sin θ``; roughness is the residual SD with
    the mean sector variance (se²) removed.
    """
    ok = np.isfinite(x_s) & np.isfinite(se_s) & (se_s > 0)
    th, x, se = theta_s[ok], x_s[ok], se_s[ok]
    raw_sd = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    if x.size < 4:
        return float(np.mean(x)) if x.size else np.nan, np.nan, np.nan, raw_sd
    X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    w = 1.0 / se ** 2
    beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], x * np.sqrt(w), rcond=None)
    resid = x - X @ beta
    var = float(np.sum(resid ** 2) / max(x.size - 3, 1))
    rough = float(np.sqrt(max(0.0, var - float(np.mean(se ** 2)))))
    return float(beta[0]), float(np.hypot(beta[1], beta[2])), rough, raw_sd


def colony_roughness(d, theta, positive: np.ndarray, values: np.ndarray, x50: float, c: float,
                     prm: dict, k_max: float):
    """Roughness and registration of one marker in one colony.

    *x50*, *c*: the colony's accepted whole-colony logistic (gate calls).
    Returns ``(metrics, sectors, profiles)``: a dict, a per-sector DataFrame and a
    long DataFrame of registered/unregistered binned profiles.
    """
    S = int(prm["n_sectors"])
    edges = np.linspace(-np.pi, np.pi, S + 1)
    sector = np.clip(np.digitize(theta, edges) - 1, 0, S - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    known = np.isfinite(positive)
    shown = known.copy()                          # profiles: every nucleus with a call
    y_all = (positive > 0.5).astype(float)
    # Same monotone stretch the whole-colony fit used (boundary.transition_segment):
    # beyond it a domain can sag again, which would widen every fit here.
    bw = float(prm["profile_bin_um"])
    nb = int(np.ceil(np.nanmax(d) / bw)) + 1
    b = np.clip((d // bw).astype(int), 0, nb - 1)
    seg = transition_segment(np.bincount(b[known], minlength=nb),
                             np.bincount(b[known], weights=y_all[known], minlength=nb), 8.0)
    if seg is not None:
        idx = np.flatnonzero(seg)
        known = known & (d >= idx[0] * bw) & (d < (idx[-1] + 1) * bw)
    x_s = np.full(S, np.nan)
    se_s = np.full(S, np.nan)
    n_s = np.zeros(S, int)
    slope = c
    for _ in range(int(prm["iterations"])):
        half = float(prm["window_widths"]) * LN81 / abs(slope) + 20.0
        for s in range(S):
            sel = known & (sector == s) & (np.abs(d - x50) <= half)
            n_s[s] = int(sel.sum())
            if n_s[s] < int(prm["min_sector_nuclei"]):
                x_s[s] = se_s[s] = np.nan
                continue
            x_s[s], se_s[s] = _position_fixed_slope(d[sel], y_all[sel], slope)
        good = np.isfinite(x_s) & (np.abs(x_s - x50) <= half)
        x_s[~good], se_s[~good] = np.nan, np.nan
        if good.sum() < int(prm["min_sectors"]):
            break
        local = x_s[sector]
        sel = known & np.isfinite(local) & (np.abs(d - local) <= half)
        refit = _binned_fit(d[sel] - local[sel], y_all[sel], float(prm["profile_bin_um"]), k_max)
        if refit is None:
            break
        slope = refit[1]

    n_good = int(np.isfinite(x_s).sum())
    sectors = pd.DataFrame({"sector": np.arange(S), "theta": centers, "x50_um": x_s,
                            "se_um": se_s, "n_nuclei": n_s})
    metrics = {"n_sectors_ok": float(n_good)}
    profiles = pd.DataFrame()
    if n_good < int(prm["min_sectors"]):
        return metrics, sectors, profiles
    mean, decenter, rough, raw_sd = harmonic_roughness(centers, x_s, se_s)
    metrics.update({"mean_border_um": mean, "decenter_um": decenter, "roughness_um": rough,
                    "raw_sd_um": raw_sd})
    # SE of the angular statistics: resample the sectors themselves.
    rng = np.random.default_rng(int(prm.get("random_state", 0)))
    have = np.flatnonzero(np.isfinite(x_s))
    boots = []
    for _ in range(int(prm.get("n_boot", 200))):
        pick = rng.choice(have, have.size, replace=True)
        boots.append(harmonic_roughness(centers[pick], x_s[pick], se_s[pick]))
    boots = np.asarray(boots, np.float64)
    se = {}
    for j, name in enumerate(("mean_border_um", "decenter_um", "roughness_um", "raw_sd_um")):
        col = boots[:, j][np.isfinite(boots[:, j])]
        se[name] = float(np.std(col, ddof=1)) if col.size > 2 else np.nan
    metrics["_se"] = se

    half_prof = float(prm["profile_half_um"])
    local = x_s[sector]
    keep = known & np.isfinite(local)
    frames = {"unregistered": d - np.nanmean(x_s), "registered": d - local}
    # Widths are fitted in the transition window only (a wider window reaches the
    # shoulders: gate calls can bump at the very edge) and on the same nuclei in
    # both frames, so the two widths differ only by registration. The profiles are
    # binned over the wider display window.
    half_fit = float(prm["window_widths"]) * LN81 / abs(slope) + 20.0
    fit_win = keep & (np.abs(frames["unregistered"]) <= half_fit) & \
        (np.abs(frames["registered"]) <= half_fit)
    win = shown & np.isfinite(local) & (np.abs(frames["unregistered"]) <= half_prof) & \
        (np.abs(frames["registered"]) <= half_prof)
    widths, rows = {}, []
    edges_p = np.arange(-half_prof, half_prof + bw, bw)
    for frame, dd in frames.items():
        f = _binned_fit(dd[fit_win], y_all[fit_win], bw, k_max)
        widths[frame] = LN81 / abs(f[1]) if f is not None else np.nan
        idx = np.digitize(dd[win], edges_p) - 1
        ok = (idx >= 0) & (idx < edges_p.size - 1)
        idx, yy = idx[ok], y_all[win][ok]
        vv = values[win][ok]
        n = np.bincount(idx, minlength=edges_p.size - 1).astype(float)
        k = np.bincount(idx, weights=yy, minlength=edges_p.size - 1)
        has_v = np.isfinite(vv)
        nv = np.bincount(idx[has_v], minlength=edges_p.size - 1).astype(float)
        sv = np.bincount(idx[has_v], weights=vv[has_v], minlength=edges_p.size - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(pd.DataFrame({"frame": frame,
                                      "x": 0.5 * (edges_p[:-1] + edges_p[1:]), "n": n,
                                      "positive_fraction": np.where(n > 0, k / n, np.nan),
                                      "mean": np.where(nv > 0, sv / nv, np.nan)}))
    profiles = pd.concat(rows, ignore_index=True)
    metrics.update({"width_unregistered_um": widths["unregistered"],
                    "width_registered_um": widths["registered"]})
    if np.isfinite(widths["unregistered"]) and widths["unregistered"] > 0:
        metrics["width_ratio"] = widths["registered"] / widths["unregistered"]
    return metrics, sectors, profiles


__all__ = ["DEFAULTS", "settings", "colony_roughness", "harmonic_roughness"]
