"""
Functional statistics on radial profiles: is the *curve* different, and where?

A radial profile is a function of distance from the edge, sampled on a grid. The
tests here treat each replicate's whole curve as one observation, which is the
honest unit: neighboring grid points of one colony are not independent
measurements, so testing them one by one (and counting nuclei as ``n``) would
manufacture significance.

Functional permutation ANOVA
----------------------------
At each grid point the one-way ANOVA F compares the group means across
replicates. The **global** statistic is the mean F over the grid points every
group covers; its null distribution comes from permuting the replicate → group
labels, which are exchangeable when the groups do not differ. Two outputs:

* ``global_p``: do the groups' curves differ anywhere?
* a **where** map: the observed F at each point against the 95th percentile of
  the permutation distribution of the *maximum* F over the grid. Points above it
  are significant with the family-wise error controlled across the whole curve
  (Westfall-Young max-T), so a shaded stretch of the plot is not one of the forty
  points that would light up by chance.

When the number of distinct labelings is no larger than ``n_permutations`` they
are enumerated exactly, so a design with 2 groups × 3 wells reports the honest
p ≥ 1/20 instead of a resampled one that looks more precise than it can be.

Everything is vectorized over permutations: per chunk the group sums, counts and
sums of squares are three matrix products, so 5 000 permutations of a 50-point
grid cost milliseconds.

Pairwise tests are the same machinery restricted to two groups, with
Benjamini-Hochberg FDR control across every marker × pair in a family.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

from .radial import n_labelings
from .stats import benjamini_hochberg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permutation F
# ---------------------------------------------------------------------------

def _f_batch(X0, M, Q0, labels, k):
    """Point-wise F for each labeling row of *labels* ``(P, n)`` → ``(P, G)``."""
    onehot = (labels[:, None, :] == np.arange(k)[None, :, None]).astype(np.float64)
    S = onehot @ X0                      # (P, k, G) group sums
    C = onehot @ M                       # (P, k, G) group counts
    Q = onehot @ Q0                      # (P, k, G) group sums of squares
    N = M.sum(axis=0)                    # (G,)
    grand = X0.sum(axis=0) / np.where(N > 0, N, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(C > 0, S / C, 0.0)
        ssb = (C * (mean - grand) ** 2).sum(axis=1)
        ssw = (Q - np.where(C > 0, S * S / C, 0.0)).sum(axis=1)
        k_eff = (C > 0).sum(axis=1)
        f = (ssb / np.maximum(k_eff - 1, 1)) / (ssw / np.maximum(N - k_eff, 1))
    f[~np.isfinite(f)] = 0.0
    return f


def _labelings(codes: np.ndarray, n_perm: int, rng: np.random.Generator):
    """``(labels, exact)``: every distinct labeling if few enough, else random ones.

    Row 0 is always the observed labeling.
    """
    counts = np.bincount(codes)
    total = n_labelings(counts)
    if total <= n_perm:
        n = codes.size
        rows = []
        # Assign group 0's positions, then group 1's among the rest, …
        def assign(free, g, current):
            if g == len(counts) - 1:
                lab = current.copy()
                lab[list(free)] = g
                rows.append(lab)
                return
            for pick in itertools.combinations(free, counts[g]):
                lab = current.copy()
                lab[list(pick)] = g
                assign([i for i in free if i not in pick], g + 1, lab)
        assign(list(range(n)), 0, np.full(n, -1))
        labels = np.vstack(rows)
        # Put the observed labeling first, drop its duplicate.
        is_obs = (labels == codes).all(axis=1)
        labels = np.vstack([codes, labels[~is_obs]])
        return labels, True
    perms = np.vstack([codes] + [codes[rng.permutation(codes.size)] for _ in range(n_perm)])
    return perms, False


def functional_anova(curves: np.ndarray, groups, n_perm: int = 5000,
                     rng: np.random.Generator | None = None, chunk: int = 512) -> dict:
    """Permutation functional one-way ANOVA over replicate curves ``(n, G)``.

    Grid points where any group has fewer than two finite replicates are left
    out (``used`` is False there): no F can be formed from them.
    """
    rng = rng or np.random.default_rng(0)
    groups = np.asarray(groups).astype(str)
    names, codes = np.unique(groups, return_inverse=True)
    k = names.size
    X = np.asarray(curves, dtype=np.float64)
    M = np.isfinite(X).astype(np.float64)
    used = np.ones(X.shape[1], bool)
    for g in range(k):
        used &= M[codes == g].sum(axis=0) >= 2
    result = {"groups": list(names), "n_replicates": int(len(codes)),
              "used": used, "n_points": int(used.sum())}
    if k < 2 or used.sum() == 0:
        result.update(global_stat=np.nan, global_p=np.nan, grid_f=np.full(X.shape[1], np.nan),
                      point_p=np.full(X.shape[1], np.nan), sig_threshold=np.nan,
                      sig_mask=np.zeros(X.shape[1], bool), exact=False, n_permutations=0)
        return result

    X0 = np.where(M > 0, X, 0.0)[:, used]
    Mu = M[:, used]
    Q0 = X0 * X0
    labels, exact = _labelings(codes, n_perm, rng)

    f_all = np.vstack([_f_batch(X0, Mu, Q0, labels[i:i + chunk], k)
                       for i in range(0, len(labels), chunk)])
    obs = f_all[0]
    null = f_all if exact else f_all[1:]           # exact: the identity is one of them
    glob = null.mean(axis=1)
    obs_glob = obs.mean()
    if exact:
        p_global = float(np.mean(glob >= obs_glob - 1e-12))
        p_point = (null >= obs - 1e-12).mean(axis=0)
    else:
        p_global = float((np.sum(glob >= obs_glob - 1e-12) + 1) / (len(null) + 1))
        p_point = ((null >= obs - 1e-12).sum(axis=0) + 1) / (len(null) + 1)
    max_null = null.max(axis=1)
    thr = float(np.percentile(max_null, 95))

    grid_f = np.full(X.shape[1], np.nan)
    grid_f[used] = obs
    point_p = np.full(X.shape[1], np.nan)
    point_p[used] = p_point
    sig = np.zeros(X.shape[1], bool)
    sig[used] = obs > thr
    result.update(global_stat=float(obs_glob), global_p=p_global, grid_f=grid_f,
                  point_p=point_p, sig_threshold=thr, sig_mask=sig, exact=exact,
                  n_permutations=int(len(null)), min_p=float(1 / len(null)) if exact
                  else float(1 / (len(null) + 1)))
    return result


# ---------------------------------------------------------------------------
# Summary features
# ---------------------------------------------------------------------------

FEATURES = {
    "edge_value": "Mean level in the outermost band (first edge_band_um µm, or r ≤ 0.15)",
    "center_value": "Mean level in the innermost band (r ≥ 0.85 of the colony's own radius)",
    "mean_value": "Mean level over the whole profile (area under the curve / length)",
    "peak_um": "Distance from the edge of the profile's maximum (µm)",
    "transition_um": "Distance from the edge where the profile first crosses halfway "
                     "between its edge and center levels (µm): the width of an edge "
                     "domain, or where a center domain begins",
}


def colony_features(um_curve: np.ndarray, um_centers: np.ndarray,
                    norm_curve: np.ndarray, norm_centers: np.ndarray,
                    edge_band_um: float = 50.0) -> dict:
    """Interpretable scalars from one colony's curves (µm and normalized)."""
    out = {k: np.nan for k in FEATURES}
    fin = np.isfinite(um_curve)
    if fin.sum() >= 2:
        x, y = um_centers[fin], um_curve[fin]
        band = x <= edge_band_um
        out["edge_value"] = float(y[band].mean()) if band.any() else float(y[0])
        out["mean_value"] = float(y.mean())
        out["peak_um"] = float(x[int(np.argmax(y))])
    nf = np.isfinite(norm_curve)
    if nf.sum() >= 2:
        core = norm_centers[nf] >= 0.85
        if core.any():
            out["center_value"] = float(norm_curve[nf][core].mean())
    if fin.sum() >= 2 and np.isfinite(out["edge_value"]) and np.isfinite(out["center_value"]):
        mid = 0.5 * (out["edge_value"] + out["center_value"])
        x, y = um_centers[fin], um_curve[fin]
        above = y >= mid
        flips = np.flatnonzero(above[1:] != above[:-1])
        if flips.size:
            i = flips[0]
            # Linear interpolation between the two grid points straddling the midpoint.
            out["transition_um"] = float(x[i] + (mid - y[i]) * (x[i + 1] - x[i])
                                         / ((y[i + 1] - y[i]) or 1e-12))
    return out


def feature_tests(features: pd.DataFrame, group_col: str, unit_col: str) -> pd.DataFrame:
    """Kruskal-Wallis per marker × feature on replicate-level values, BH-corrected."""
    rows = []
    for (marker, feat), grp in features.melt(
            id_vars=[group_col, unit_col, "marker"], value_vars=list(FEATURES),
            var_name="feature").groupby(["marker", "feature"], observed=True):
        unit = grp.groupby([group_col, unit_col], observed=True)["value"].mean().reset_index()
        samples = [g["value"].dropna().to_numpy() for _, g in unit.groupby(group_col)]
        samples = [s for s in samples if s.size >= 2]    # one replicate has no spread
        if len(samples) < 2:
            stat = p = np.nan
        else:
            try:
                stat, p = sps.kruskal(*samples)
            except ValueError:          # every value identical
                stat, p = 0.0, 1.0
        rows.append({"marker": marker, "feature": feat, "kruskal_H": float(stat),
                     "p_value": float(p), "n_groups": len(samples),
                     "n_replicates": int(sum(s.size for s in samples))})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = np.nan
        ok = out["p_value"].notna()
        out.loc[ok, "q_value"] = benjamini_hochberg(out.loc[ok, "p_value"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Dissimilarity
# ---------------------------------------------------------------------------

def pairwise_l2(X: np.ndarray) -> np.ndarray:
    """RMS distance between rows over their shared finite points (vectorized).

    Rows are concatenated marker curves; a pair with no shared support is NaN.
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.isfinite(X).astype(np.float64)
    X0 = np.where(M > 0, X, 0.0)
    sq = X0 * X0
    ss = sq @ M.T + M @ sq.T - 2.0 * (X0 @ X0.T)
    cnt = M @ M.T
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.sqrt(np.clip(ss, 0, None) / cnt)
    D[cnt == 0] = np.nan
    np.fill_diagonal(D, 0.0)
    return D


def within_between(D: np.ndarray, groups) -> pd.DataFrame:
    """Mean distance between colonies of the same vs different groups."""
    groups = np.asarray(groups).astype(str)
    iu = np.triu_indices(len(groups), k=1)
    same = groups[iu[0]] == groups[iu[1]]
    d = D[iu]
    rows = []
    for g in np.unique(groups):
        sel = same & (groups[iu[0]] == g)
        rows.append({"group": g, "comparison": "within", "mean_distance": float(np.nanmean(d[sel]))
                     if sel.any() else np.nan, "n_pairs": int(sel.sum())})
    rows.append({"group": "all", "comparison": "within",
                 "mean_distance": float(np.nanmean(d[same])) if same.any() else np.nan,
                 "n_pairs": int(same.sum())})
    rows.append({"group": "all", "comparison": "between",
                 "mean_distance": float(np.nanmean(d[~same])) if (~same).any() else np.nan,
                 "n_pairs": int((~same).sum())})
    return pd.DataFrame(rows)


__all__ = ["functional_anova", "colony_features", "feature_tests", "pairwise_l2",
           "within_between", "FEATURES"]
