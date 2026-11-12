"""
Boundaries and peaks: a few numbers per colony, each with an uncertainty.

A radial profile is the right object for testing, but the claims people make
about a pattern are scalars: *where* a domain ends, *how sharp* the transition
is, *where* a signaling peak sits relative to it. This module measures them per
colony so they can be compared across conditions with colonies (or wells) as
the replicates.

Per marker (``kind = marker``)
    A logistic fitted across the transition,
    ``f(d) = 1 / (1 + exp(-k (d - x50)))`` with d the distance from the edge:

    * ``readout = positive``: a binomial fit to the gate calls (the fate switch).
      Needs a positivity gate.
    * ``readout = intensity``: ``lo + (hi - lo) f(d)`` fitted to the binned mean
      on the shared scale. Works for any marker.

    Metrics: ``x50_um`` (the 50 % point, the boundary position), ``width_um``
    (10 % to 90 % of the fitted range, ``2 ln 9 / |k|``), ``width_cells`` (the
    same in nearest-neighbor spacings of that colony's nuclei) and ``direction``
    (+1: the marker rises toward the center, -1: it is an edge domain).

Per facing pair (``kind = pair``)
    Two markers of one stain set whose domains face each other (one edge-high,
    one center-high), e.g. two mutually repressive fate markers:

    * ``crossing_um``: where the two fitted curves cross, each on its own 0-1
      range. With logistics this is closed form,
      ``(kA xA - kB xB) / (kA - kB)``. The operational boundary.
    * ``overlap_um``: x50 of the edge-high marker minus x50 of the center-high
      one. Positive: a band where both are on; negative: a gap where neither is.
    * ``dp_fraction``: the fraction of nuclei positive for both within
      ``window_um`` of the crossing, ``dp_expected``: what independence would
      give there (product of the two marginal fractions), and their ratio
      ``dp_enrichment``. A border sharpening by mutual repression loses
      double-positive nuclei.

Per peak marker (``kind = peak``; signaling markers by default)
    The interior local maximum of the smoothed profile, ignoring the outermost
    ``edge_exclude_um`` (the colony edge is usually the global maximum and is not
    an interior peak): ``peak_um``, the minimum between the edge and the peak
    ``trough_um``, ``peak_ratio = peak / trough`` (scale-free, so comparable
    across stain panels), and ``offset_um`` = peak minus each facing-pair
    crossing and minus each marker's x50 in the same colony.

Uncertainty per colony
    Nuclei of one colony are spatially correlated, so resampling nuclei would
    understate the error. The colony is cut into ``n_sectors`` angular sectors
    and the sectors are resampled with replacement (a block bootstrap):
    every quantity, including the offsets, is recomputed jointly in each
    replicate, and ``se`` is the SD over replicates. That SE describes how well
    this one colony determines the number; it includes the boundary's variation
    around the colony and is *not* an error bar for comparing conditions, which
    comes from the spread across colonies (or wells) in the summary.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy import stats as sps

logger = logging.getLogger(__name__)

LN81 = 2.0 * np.log(9.0)          # 10 % → 90 % of a logistic spans LN81 / |k|

DEFAULTS = {
    "enabled": True,
    "pairs": "auto",              # auto | [[A, B], ...] (one stain set each)
    "peak_markers": "auto",       # auto = signaling markers | [names] | []
    "bin_um": 10.0,               # fitting bins (finer than the profile grid)
    "n_sectors": 12,              # angular blocks of the bootstrap
    "n_boot": 200,                # bootstrap replicates per colony
    "window_um": 50.0,            # ± around the crossing for the double-positive fraction
    "edge_exclude_um": 25.0,      # the peak search ignores this outermost band
    "peak_smooth_um": 15.0,       # Gaussian smoothing (σ) of the peak profile
    "min_prominence": 0.05,       # a peak must rise this fraction of the profile's range...
    "min_peak_ratio": 1.05,       # ...and at least this far above its trough (peak / trough)
    "min_amplitude": 0.1,         # an intensity transition spans at least this (shared scale)
    "min_support": 0.8,           # a colony value counts when ≥ this fraction of replicates find it
    "min_bin_nuclei": 8,          # bins thinner than this are not used
    "min_nuclei": 200,            # fewer nuclei with a value → no fit for that marker
    "random_state": 0,
    # Refit every colony's point estimates with other bins: does binning shape the result?
    # fixed:<µm> = equal-width bins; quantile:<n> = n bins of equal nucleus count. [] = off.
    "bin_check": ["fixed:5", "fixed:20", "quantile:20"],
}


def parse_scheme(text) -> tuple[str, str, float]:
    """``"fixed:5"`` → ``("fixed:5", "fixed", 5.0)``; ``"quantile:20"`` likewise."""
    kind, _, num = str(text).strip().lower().partition(":")
    try:
        val = float(num)
    except ValueError:
        val = np.nan
    if kind not in ("fixed", "quantile") or not val > 0:
        raise ValueError(f"boundary.bin_check: {text!r} is not fixed:<µm> or quantile:<n>")
    if kind == "quantile":
        val = float(int(val))
    return f"{kind}:{val:g}", kind, val


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("boundary") or {}).items() if v is not None})
    if int(out["n_sectors"]) < 4:
        raise ValueError("boundary.n_sectors must be at least 4")
    if float(out["bin_um"]) <= 0:
        raise ValueError("boundary.bin_um must be positive")
    out["bin_check"] = [parse_scheme(x)[0] for x in (out.get("bin_check") or [])]
    pairs = out["pairs"]
    if not (pairs == "auto" or isinstance(pairs, (list, tuple))):
        raise ValueError("boundary.pairs must be 'auto' or a list of [A, B] pairs")
    if isinstance(pairs, (list, tuple)):
        out["pairs"] = [tuple(map(str, p)) for p in pairs]
        if any(len(p) != 2 for p in out["pairs"]):
            raise ValueError("boundary.pairs: every entry is two marker names, [A, B]")
    return out


# ---------------------------------------------------------------------------
# Fits on binned data (fast enough to repeat inside the bootstrap)
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))       # overflow-free logistic


def fit_logistic(x, n, k, k_max: float):
    """Binomial logistic ``logit p = a + c x`` on binned counts (IRLS).

    Returns ``(x50, c)`` or ``None``. ``|c|`` is capped at *k_max*: a transition
    sharper than the bins cannot be resolved, and a perfectly separated step
    would otherwise run the slope to infinity.
    """
    x, n, k = (np.asarray(v, np.float64) for v in (x, n, k))
    ok = n > 0
    x, n, k = x[ok], n[ok], k[ok]
    if x.size < 3 or k.sum() == 0 or k.sum() == n.sum():
        return None
    p = np.clip((k + 0.5) / (n + 1.0), 1e-3, 1 - 1e-3)
    c, a = np.polyfit(x, np.log(p / (1 - p)), 1, w=np.sqrt(n))
    capped = False
    for _ in range(50):
        if abs(c) > k_max:
            c, capped = np.sign(c) * k_max, True
        eta = a + c * x
        mu = np.clip(_sigmoid(eta), 1e-9, 1 - 1e-9)
        w = n * mu * (1 - mu)
        z = eta + (k - n * mu) / np.maximum(w, 1e-12)
        if capped:                                  # slope fixed: refit the intercept only
            a_new, c_new = float(np.sum(w * (z - c * x)) / np.sum(w)), c
        else:
            X = np.column_stack([np.ones_like(x), x])
            A = X.T @ (X * w[:, None])
            try:
                a_new, c_new = np.linalg.solve(A, X.T @ (w * z))
            except np.linalg.LinAlgError:
                return None
        done = abs(a_new - a) < 1e-8 and abs(c_new - c) < 1e-10
        a, c = float(a_new), float(c_new)
        if done:
            break
    if not np.isfinite(a) or not np.isfinite(c) or c == 0:
        return None
    return -a / c, float(np.clip(c, -k_max, k_max))


def fit_four_pl(x, n, mean, k_max: float, start=None):
    """``lo + (hi - lo) / (1 + exp(-c (x - x50)))`` on binned means, weights √n.

    Returns ``(x50, c, lo, hi)`` or ``None``. *start* seeds the optimizer (the
    point estimate seeds every bootstrap replicate).
    """
    x, n, y = (np.asarray(v, np.float64) for v in (x, n, mean))
    ok = (n > 0) & np.isfinite(y)
    x, n, y = x[ok], n[ok], y[ok]
    if x.size < 5:
        return None
    if start is None:
        m = max(3, x.size // 5)
        edge, center = float(np.mean(y[:m])), float(np.mean(y[-m:]))
        mid = 0.5 * (edge + center)
        cross = np.nonzero(np.diff(np.sign(y - mid)))[0]
        x0 = float(x[cross[0]]) if cross.size else float(np.median(x))
        c0 = 4.0 / max(np.ptp(x), 1.0)
        # The sign of c carries the direction; lo/hi are always the low/high plateau.
        start = (x0, c0, edge, center) if center >= edge else (x0, -c0, center, edge)

    def f(xx, x50, c, lo, hi):
        return lo + (hi - lo) * _sigmoid(c * (xx - x50))

    span = float(np.ptp(y)) or 1.0
    lb = [x.min() - 0.5 * np.ptp(x), -k_max, y.min() - span, y.min() - span]
    ub = [x.max() + 0.5 * np.ptp(x), k_max, y.max() + span, y.max() + span]
    p0 = np.clip(np.asarray(start, np.float64), np.add(lb, 1e-9), np.subtract(ub, 1e-9))
    try:
        p, _ = optimize.curve_fit(f, x, y, p0=p0, sigma=1.0 / np.sqrt(n), bounds=(lb, ub),
                                  maxfev=4000)
    except (RuntimeError, ValueError, optimize.OptimizeWarning):
        return None
    x50, c, lo, hi = map(float, p)
    if hi < lo:                                     # canonical form: hi above lo
        x50, c, lo, hi = x50, -c, hi, lo
    return x50, c, lo, hi


def crossing(xa, ca, xb, cb) -> float:
    """Where two logistics (each on its own 0-1 range) cross; NaN unless they face."""
    if not (np.isfinite(xa) and np.isfinite(xb)) or np.sign(ca) == np.sign(cb):
        return np.nan
    return (ca * xa - cb * xb) / (ca - cb)


def interior_peak(x, s, n, prm: dict):
    """Interior maximum of a binned profile (sums *s*, counts *n*).

    Kernel-weighted mean (smoothed sums / smoothed counts), so sparse bins near
    the center are not over-weighted. Returns a dict or ``None``.
    """
    sigma = float(prm["peak_smooth_um"]) / float(prm["bin_um"])
    ss = gaussian_filter1d(np.asarray(s, np.float64), sigma, mode="nearest")
    nn = gaussian_filter1d(np.asarray(n, np.float64), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        y = ss / nn
    valid = (nn >= float(prm["min_bin_nuclei"])) & np.isfinite(y)
    if not valid[0]:
        return None
    stop = int(np.argmin(valid)) if not valid.all() else valid.size   # contiguous from edge
    x, y = np.asarray(x, np.float64)[:stop], y[:stop]
    if x.size < 5:
        return None
    rng = float(np.ptp(y))
    if rng <= 0:
        return None
    start = int(np.searchsorted(x, float(prm["edge_exclude_um"])))
    idx, props = find_peaks(y, prominence=float(prm["min_prominence"]) * rng)
    keep = idx >= max(start, 1)
    idx, prom = idx[keep], props["prominences"][keep]
    if not idx.size:
        return None
    i = int(idx[np.argmax(prom)])
    xp = float(x[i])
    if 0 < i < x.size - 1:                          # quadratic refinement
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            xp = float(x[i] + 0.5 * (y0 - y2) / den * (x[1] - x[0]))
    lo = max(start, 0)
    j = lo + int(np.argmin(y[lo:i + 1])) if i > lo else i
    out = {"peak_um": xp, "peak_value": float(y[i]), "trough_um": float(x[j]),
           "trough_value": float(y[j]), "prominence": float(prom.max())}
    out["peak_ratio"] = (out["peak_value"] / out["trough_value"]
                         if out["trough_value"] > 0 else np.nan)
    if not out["peak_ratio"] >= float(prm.get("min_peak_ratio", 1.0)):
        return None                                 # a ripple, not a peak
    return out


def transition_segment(n, s, min_n: float, sigma_bins: float = 1.5):
    """Bin mask of the monotone stretch between the profile's minimum and maximum.

    A domain marker is not always a clean step: it can rise and then sag toward
    the center, a gate call can bump at the very edge. A logistic over the whole
    radius is dragged by those shoulders (wide, wrong transitions), so the fit is
    restricted to the stretch from the smoothed profile's minimum to its maximum,
    plus one bin either side. Fixed from the observed colony and reused by every
    bootstrap replicate.
    """
    n = np.asarray(n, np.float64)
    ss = gaussian_filter1d(np.asarray(s, np.float64), sigma_bins, mode="nearest")
    nn = gaussian_filter1d(n, sigma_bins, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where((n >= min_n) & (nn > 0), ss / nn, np.nan)
    if np.isfinite(y).sum() < 5:
        return None
    i_lo, i_hi = int(np.nanargmin(y)), int(np.nanargmax(y))
    a, b = sorted((i_lo, i_hi))
    mask = np.zeros(n.size, bool)
    mask[max(a - 1, 0):min(b + 2, n.size)] = True
    return mask if mask.sum() >= 5 else None


# ---------------------------------------------------------------------------
# One colony
# ---------------------------------------------------------------------------

def nn_spacing_um(xy_um: np.ndarray) -> float:
    """Median nearest-neighbor distance between nuclei (the local cell spacing)."""
    if len(xy_um) < 3:
        return np.nan
    d, _ = cKDTree(xy_um).query(xy_um, k=2)
    return float(np.median(d[:, 1]))


def _sector_tables(d, sector, values: dict, positive: dict, prm, n_sectors, edges=None):
    """Per (sector × bin) counts and sums for every marker: the bootstrap's raw material.

    Bins are ``bin_um`` wide from the edge, or the given *edges* (any widths)."""
    if edges is not None:
        edges = np.asarray(edges, np.float64)
        nbin = edges.size - 1
        b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, nbin - 1)
        x = 0.5 * (edges[:-1] + edges[1:])
    else:
        width = float(prm["bin_um"])
        nbin = int(np.ceil(max(float(np.nanmax(d)), width) / width)) + 1
        b = np.clip((d // width).astype(int), 0, nbin - 1)
        x = (np.arange(nbin) + 0.5) * width
    flat = sector * nbin + b
    size = n_sectors * nbin

    def table(mask, weights=None):
        t = np.bincount(flat[mask], weights=None if weights is None else weights[mask],
                        minlength=size)
        return t.reshape(n_sectors, nbin).astype(np.float64)

    out = {"x": x, "markers": {}}
    for m, v in values.items():
        has = np.isfinite(v)
        entry = {"n_val": table(has), "s_val": table(has, np.where(has, v, 0.0))}
        p = positive.get(m)
        if p is not None and np.isfinite(p).any():
            hp = np.isfinite(p)
            entry["n_pos"] = table(hp)
            entry["k_pos"] = table(hp, np.where(hp, p, 0.0))
        out["markers"][m] = entry
    return out


def measure_colony(d, theta, values: dict, positive: dict, pairs, peak_markers, prm,
                   spacing_um: float, rng: np.random.Generator, edges=None) -> pd.DataFrame:
    """Every boundary/peak metric of one colony with its sector-bootstrap SE.

    *values*/*positive*: ``{marker: per-nucleus array}`` (NaN where unmeasured).
    *edges*: bin edges in µm from the edge (default: ``bin_um``-wide bins); with
    unequal bins the smoothing and the width floor use the median bin width.
    Returns a long table: kind, subject, readout, metric, value, se, n_boot_ok.
    """
    if edges is not None:
        prm = dict(prm, bin_um=float(np.median(np.diff(np.asarray(edges, np.float64)))))
    S = int(prm["n_sectors"])
    sector = np.clip(((theta + np.pi) / (2 * np.pi) * S).astype(int), 0, S - 1)
    tabs = _sector_tables(d, sector, values, positive, prm, S, edges)
    x = tabs["x"]
    k_max = LN81 / (0.5 * float(prm["bin_um"]))     # narrowest resolvable width: half a bin
    window = float(prm["window_um"])
    min_n = int(prm["min_nuclei"])
    usable = {m: e for m, e in tabs["markers"].items() if e["n_val"].sum() >= min_n}
    peak_set = set(peak_markers)
    min_bin = float(prm["min_bin_nuclei"])
    seg = {}                                      # (marker, readout) → bin mask
    for m, e in usable.items():
        if m in peak_set:
            continue                              # a peak is not a domain: no logistic
        seg[(m, "intensity")] = transition_segment(e["n_val"].sum(0), e["s_val"].sum(0), min_bin)
        if "n_pos" in e:
            seg[(m, "positive")] = transition_segment(e["n_pos"].sum(0), e["k_pos"].sum(0),
                                                      min_bin)

    min_amp = float(prm["min_amplitude"])

    def valid(f, top) -> bool:
        """A transition: inside the colony, narrower than the colony radius and, for
        an intensity fit, spanning at least min_amplitude (noise on a flat colony
        fits a tiny, sharp step anywhere)."""
        if f is None or not (0 <= f[0] <= top) or LN81 / abs(f[1]) > top:
            return False
        return len(f) < 4 or (f[3] - f[2]) >= min_amp

    # Double-positive raw material: nuclei near a crossing are only known once the
    # point estimate is; per-sector sums of the window are taken then.
    def evaluate(w, seeds=None):
        """All metrics for sector weights *w* (ones = the observed colony)."""
        res, fits = {}, {}
        for m, e in usable.items():
            if m in peak_set:
                continue
            n_val, s_val = w @ e["n_val"], w @ e["s_val"]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = s_val / n_val
            thin = n_val < min_bin
            if thin.all():
                continue
            top = float(x[~thin].max())
            mask = seg.get((m, "intensity"))
            if mask is not None:
                use = mask & ~thin
                seed = (seeds or {}).get((m, "intensity"))
                f4 = fit_four_pl(x[use], n_val[use], mean[use], k_max, start=seed)
                if valid(f4, top):
                    fits[(m, "intensity")] = f4
            mask = seg.get((m, "positive"))
            if mask is not None:
                fl = fit_logistic(x[mask], (w @ e["n_pos"])[mask], (w @ e["k_pos"])[mask],
                                  k_max)
                if valid(fl, top):
                    fits[(m, "positive")] = fl
        for (m, ro), f in fits.items():
            res[("marker", m, ro, "x50_um")] = f[0]
            res[("marker", m, ro, "width_um")] = LN81 / abs(f[1])
            res[("marker", m, ro, "width_cells")] = LN81 / abs(f[1]) / spacing_um
            res[("marker", m, ro, "direction")] = float(np.sign(f[1]))
        for a, b in pairs:
            ro = "positive" if (a, "positive") in fits and (b, "positive") in fits else \
                "intensity"
            fa, fb = fits.get((a, ro)), fits.get((b, ro))
            if fa is None or fb is None or np.sign(fa[1]) == np.sign(fb[1]):
                continue
            key = f"{a} / {b}"
            res[("pair", key, ro, "crossing_um")] = crossing(fa[0], fa[1], fb[0], fb[1])
            edge_hi, center_hi = (fa, fb) if fa[1] < 0 else (fb, fa)
            res[("pair", key, ro, "overlap_um")] = edge_hi[0] - center_hi[0]
        for m in peak_markers:
            e = usable.get(m)
            if e is None:
                continue
            pk = interior_peak(x, w @ e["s_val"], w @ e["n_val"], prm)
            if pk is None:
                continue
            for metric in ("peak_um", "trough_um", "peak_ratio", "prominence"):
                res[("peak", m, "intensity", metric)] = pk[metric]
            for (kind, subject, ro, metric), v in list(res.items()):
                if kind == "pair" and metric == "crossing_um":
                    res[("peak", m, "intensity", f"offset_um vs {subject}")] = pk["peak_um"] - v
                elif kind == "marker" and metric == "x50_um" and subject != m:
                    res[("peak", m, "intensity", f"offset_um vs {subject} x50 ({ro})")] = \
                        pk["peak_um"] - v
        return res, fits

    ones = np.ones(S)
    point, fits = evaluate(ones)
    seeds = {key: f for key, f in fits.items() if key[1] == "intensity"}

    # Double positive around each crossing (needs the per-nucleus gate calls).
    dp_tables = {}
    for (kind, subject, ro, metric), v in point.items():
        if metric != "crossing_um" or not np.isfinite(v):
            continue
        a, b = subject.split(" / ")
        pa, pb = positive.get(a), positive.get(b)
        if pa is None or pb is None:
            continue
        near = (np.abs(d - v) <= window) & np.isfinite(pa) & np.isfinite(pb)
        if near.sum() < 20:
            continue
        sec = sector[near]
        A, B = pa[near] > 0.5, pb[near] > 0.5
        dp_tables[subject] = (np.bincount(sec, minlength=S).astype(float),
                              np.bincount(sec, weights=A.astype(float), minlength=S),
                              np.bincount(sec, weights=B.astype(float), minlength=S),
                              np.bincount(sec, weights=(A & B).astype(float), minlength=S))

    def add_dp(res, w):
        for subject, (n, ka, kb, kab) in dp_tables.items():
            N = w @ n
            if N <= 0:
                continue
            obs, fa, fb = (w @ kab) / N, (w @ ka) / N, (w @ kb) / N
            res[("pair", subject, "positive", "dp_fraction")] = obs
            res[("pair", subject, "positive", "dp_expected")] = fa * fb
            res[("pair", subject, "positive", "dp_enrichment")] = \
                obs / (fa * fb) if fa * fb > 0 else np.nan

    add_dp(point, ones)
    boots = []
    for _ in range(int(prm["n_boot"])):
        w = rng.multinomial(S, np.full(S, 1.0 / S)).astype(np.float64)
        r, _ = evaluate(w, seeds)
        add_dp(r, w)
        boots.append(r)
    rows = []
    for key, v in point.items():
        reps = np.array([b.get(key, np.nan) for b in boots], np.float64)
        ok = np.isfinite(reps)
        support = float(ok.mean()) if reps.size else 1.0
        rows.append({"kind": key[0], "subject": key[1], "readout": key[2], "metric": key[3],
                     "value": float(v),
                     "se": float(np.std(reps[ok], ddof=1)) if ok.sum() > 2 else np.nan,
                     "boot_support": support,
                     "ok": bool(np.isfinite(v) and support >= float(prm["min_support"]))})
    out = pd.DataFrame(rows, columns=["kind", "subject", "readout", "metric", "value", "se",
                                      "boot_support", "ok"])
    curves = {f"{m}|{ro}": f for (m, ro), f in fits.items()}
    out.attrs["fits"] = curves
    return out


CHECK_METRICS = ("x50_um", "width_um", "crossing_um", "overlap_um", "peak_um")


def bin_check(d, theta, values: dict, positive: dict, pairs, peak_markers, prm,
              spacing_um: float, schemes=None) -> pd.DataFrame:
    """Point estimates of one colony under other binnings (no bootstrap).

    One row per scheme × metric (positions and widths only: :data:`CHECK_METRICS`),
    for comparing with the default fit. ``quantile:<n>`` puts ≈ equal numbers of
    nuclei in each bin, as equal-count radial profiles do.
    """
    rows = []
    for text in (prm.get("bin_check") if schemes is None else schemes) or []:
        name, kind, val = parse_scheme(text)
        local, edges = dict(prm, n_boot=0), None
        if kind == "fixed":
            local["bin_um"] = val
        else:
            ok = np.isfinite(d)
            edges = np.unique(np.quantile(d[ok], np.linspace(0, 1, int(val) + 1)))
            if edges.size < 6:
                continue
            edges[-1] += 1e-6                     # the center-most nucleus belongs to a bin
        res = measure_colony(d, theta, values, positive, pairs, peak_markers, local,
                             spacing_um, np.random.default_rng(0), edges)
        res = res[res["metric"].isin(CHECK_METRICS)]
        if len(res):
            rows.append(res[["kind", "subject", "readout", "metric", "value"]]
                        .assign(scheme=name))
    if not rows:
        return pd.DataFrame(columns=["scheme", "kind", "subject", "readout", "metric", "value"])
    return pd.concat(rows, ignore_index=True)


def bin_check_summary(check: pd.DataFrame, colonies: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Shift of each metric under each scheme relative to the default fit.

    Per scheme × metric (× *by*): colonies compared, median shift (alt − default),
    median |shift|, median |shift| / the default's bootstrap SE, and the fraction
    of colonies whose shift is within 2 SE. Only colonies whose default value was
    accepted are compared.
    """
    keys = ["field_id", "kind", "subject", "readout", "metric"]
    base = colonies[colonies["ok"].astype(bool) & colonies["metric"].isin(CHECK_METRICS)]
    base = base[keys + ["value", "se"] + [c for c in by if c in base.columns]]
    m = check.merge(base, on=keys, suffixes=("", "_default"))
    if m.empty:
        return pd.DataFrame()
    m["shift"] = m["value"] - m["value_default"]
    with np.errstate(invalid="ignore", divide="ignore"):
        m["shift_se"] = (m["shift"] / m["se"]).abs()
    group = ["scheme"] + [c for c in by if c in m.columns] + keys[1:]
    out = m.groupby(group, observed=True, dropna=False).agg(
        n_colonies=("field_id", "nunique"), median_shift=("shift", "median"),
        median_abs_shift=("shift", lambda s: float(np.nanmedian(np.abs(s)))),
        median_abs_shift_se=("shift_se", "median"),
        within_2se=("shift_se", lambda s: float((s <= 2).mean()) if s.notna().any() else np.nan),
        median_se=("se", "median")).reset_index()
    return out


# ---------------------------------------------------------------------------
# Which pairs face each other
# ---------------------------------------------------------------------------

def auto_pairs(long: pd.DataFrame, gated_markers: set, exclude: set,
               edge_um: float = 50.0, center_rnorm: float = 0.75) -> dict:
    """``{stain_set: [(edge_high, center_high), ...]}`` among gated markers.

    A marker's direction is the sign of (positive fraction in the center − at the
    edge) over all colonies of its stain set; a pair faces when the signs differ.
    """
    out = {}
    sub = long[long["marker"].astype(str).isin(gated_markers - exclude)
               & long["positive"].notna()]
    for stain, part in sub.groupby(sub["stain_set"].astype(str)):
        direction = {}
        for m, g in part.groupby(part["marker"].astype(str)):
            edge = g.loc[g["dist_edge_um"] <= edge_um, "positive"].mean()
            center = g.loc[g["r_norm"] >= center_rnorm, "positive"].mean()
            if np.isfinite(edge) and np.isfinite(center) and abs(center - edge) >= 0.2:
                direction[m] = np.sign(center - edge)
        pairs = []
        for a, b in combinations(sorted(direction), 2):
            if direction[a] != direction[b]:
                pairs.append((a, b) if direction[a] < 0 else (b, a))
        if pairs:
            out[stain] = pairs
    return out


def unresolved_pairs(colonies: pd.DataFrame, min_support: float) -> dict:
    """``{field_id: [note, ...]}``: why a colony has no accepted crossing for a facing
    pair that other colonies of its stain set do have.

    The note names the cause: the crossing was found in too few bootstrap
    replicates (a scattered, incoherent domain), or one marker has no accepted
    transition at all (e.g. a negative control), or the two do not face each other.
    """
    if colonies.empty:
        return {}
    pair = colonies[(colonies["kind"] == "pair") & (colonies["metric"] == "crossing_um")]
    if pair.empty:
        return {}
    expected = pair.groupby(pair["stain_set"].astype(str))["subject"].unique()
    ok = colonies["ok"].astype(bool)
    x50 = colonies[ok & (colonies["kind"] == "marker") & (colonies["metric"] == "x50_um")]
    accepted = set(zip(x50["field_id"], x50["subject"].astype(str)))
    out: dict = {}
    for fid, part in colonies.groupby("field_id"):
        stain = str(part["stain_set"].iloc[0])
        for subject in expected.get(stain, []):
            rows = part[(part["kind"] == "pair") & (part["metric"] == "crossing_um")
                        & (part["subject"] == subject)]
            if rows["ok"].astype(bool).any():
                continue
            if len(rows):
                support = float(rows["boot_support"].max())
                why = (f"found in {support:.0%} of bootstrap resamples, "
                       f"needs {float(min_support):.0%}")
            else:
                lacking = [m for m in subject.split(" / ") if (fid, m) not in accepted]
                why = (f"no accepted {' or '.join(lacking)} transition" if lacking
                       else "the two transitions do not face each other")
            out.setdefault(fid, []).append(f"{subject}: no boundary ({why})")
    return out


# ---------------------------------------------------------------------------
# Across colonies
# ---------------------------------------------------------------------------

def summarize(colonies: pd.DataFrame, by: list[str], unit: str, ci: float = 95.0,
              experiment_col: str | None = None) -> pd.DataFrame:
    """Mean ± SE / t-CI across replicates per *by* × kind × subject × readout × metric.

    Only accepted colony values (``ok``) enter; ``n_colonies_rejected`` counts the
    others. With ``unit == "well"`` colonies are averaged within each well first,
    so n is the number of wells; with ``unit == "experiment"`` the well means are
    then averaged within each experiment (*experiment_col*), n = experiments.
    """
    keys = ["kind", "subject", "readout", "metric"]
    frame = colonies.copy()
    ok = frame["ok"].astype(bool) if "ok" in frame else pd.Series(True, index=frame.index)
    frame["_v"] = frame["value"].where(ok & np.isfinite(frame["value"]))
    frame["_rej"] = (~ok).astype(int)
    if frame.empty:
        return pd.DataFrame()
    exp = [experiment_col] if unit == "experiment" and experiment_col in frame else []
    if unit == "experiment" and not exp:
        raise ValueError("boundary.summarize: the experiment unit needs its column")
    if unit in ("well", "experiment") and "well_key" in frame:
        frame = (frame.groupby(by + exp + ["well_key"] + keys, observed=True, dropna=False)
                 .agg(_v=("_v", "mean"), _rej=("_rej", "sum")).reset_index())
    if exp:
        frame = (frame.groupby(by + exp + keys, observed=True, dropna=False)
                 .agg(_v=("_v", "mean"), _rej=("_rej", "sum")).reset_index())
    out = (frame.groupby(by + keys, observed=True, dropna=False)
           .agg(mean=("_v", "mean"), sd=("_v", "std"), n=("_v", "count"),
                n_colonies_rejected=("_rej", "sum"))
           .reset_index())
    with np.errstate(invalid="ignore", divide="ignore"):
        out["se"] = out["sd"] / np.sqrt(out["n"])
        t = sps.t.ppf(0.5 + ci / 200.0, np.maximum(out["n"] - 1, 1))
        half = np.where(out["n"] > 1, t * out["se"], np.nan)
    out["ci_lo"], out["ci_hi"] = out["mean"] - half, out["mean"] + half
    out["unit"] = unit
    return out


# ---------------------------------------------------------------------------
# Which metrics a summary figure shows
# ---------------------------------------------------------------------------
# Each panel: (kind, subject, metric, readout or None = positive when present,
# label, overlay metric or None).

def pair_panels(a: str, b: str) -> list[tuple]:
    pair = f"{a} / {b}"
    return [
        ("pair", pair, "crossing_um", None, "boundary: crossing (µm from edge)", None),
        ("pair", pair, "overlap_um", None, "overlap (+) / gap (−) (µm)", None),
        ("pair", pair, "dp_fraction", "positive",
         "double-positive fraction near the boundary\n(open = expected if independent)",
         "dp_expected"),
        ("marker", a, "x50_um", None, f"{a} boundary x50 (µm)", None),
        ("marker", b, "x50_um", None, f"{b} boundary x50 (µm)", None),
        ("marker", a, "width_um", None, f"{a} transition width 10-90% (µm)", None),
        ("marker", b, "width_um", None, f"{b} transition width 10-90% (µm)", None),
        ("marker", b, "width_cells", None, f"{b} transition width (cell spacings)", None),
    ]


def peak_panels(table: pd.DataFrame) -> list[tuple]:
    subject = str(table["subject"].iloc[0])
    out = [("peak", subject, "peak_um", None, "interior peak (µm from edge)", None),
           ("peak", subject, "trough_um", None, "trough before the peak (µm)", None),
           ("peak", subject, "peak_ratio", None, "peak / trough", None)]
    offsets = sorted(m for m in table["metric"].astype(str).unique() if m.startswith("offset_um"))
    # Facing-pair crossings first (the operational boundaries), then gate-based x50s.
    offsets = [m for m in offsets if " / " in m] + \
              [m for m in offsets if " / " not in m and "(positive)" in m]
    for m in offsets:
        target = m.replace("offset_um vs ", "").replace(" (positive)", "")
        out.append(("peak", subject, m, None, f"peak − {target} (µm)", None))
    return out


def roughness_panels(marker: str) -> list[tuple]:
    return [
        ("roughness", marker, "roughness_um", "positive",
         "border roughness: angular SD (µm)\nfitting noise and decentering removed", None),
        ("roughness", marker, "decenter_um", "positive",
         "border decentering (1st harmonic, µm)", None),
        ("roughness", marker, "width_unregistered_um", "positive",
         "transition width, angle-averaged (µm)", None),
        ("roughness", marker, "width_registered_um", "positive",
         "transition width, registered to the\nlocal border in each sector (µm)", None),
        ("roughness", marker, "width_ratio", "positive",
         "registered / angle-averaged width\n(≪ 1: the gradient is the border wandering)",
         None),
    ]


__all__ = ["bin_check", "bin_check_summary", "parse_scheme", "CHECK_METRICS",
           "roughness_panels", "pair_panels", "peak_panels", "DEFAULTS", "settings",
           "fit_logistic", "fit_four_pl", "crossing", "interior_peak",
           "measure_colony", "auto_pairs", "unresolved_pairs", "summarize", "nn_spacing_um", "transition_segment",
           "LN81"]
