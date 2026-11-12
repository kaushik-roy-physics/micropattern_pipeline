"""
Spatially resolved co-expression: where in the colony do two markers go together?

A radial profile is a per-bin mean, one marker at a time, so it cannot say whether
the nuclei that are high for marker A are the ones high for marker B. This module
works from the nuclei instead: each nucleus has both values on the shared scale
(:mod:`downstream.radial`) and its distance from the colony edge, and every
readout is computed **as a function of that distance**.

Readouts per marker pair, colony and distance bin
-------------------------------------------------
``correlation``      Spearman (default) or Pearson correlation of A and B over
                     the bin's nuclei, with a Fisher-z confidence interval
                     (Spearman SE ``sqrt(1.06 / (n − 3))``, Bonett & Wright 2000).
``mean_a, mean_b``   the bin's mean of each marker: the colony's path through
                     the (A, B) plane from edge to center.
``frac_a, frac_b``   fraction of nuclei positive for each marker (needs a gate)
``frac_ab``          fraction positive for both
``frac_expected``    ``frac_a · frac_b``: what ``frac_ab`` would be if the two
                     markers were independent within the bin
``log2_enrichment``  ``log2(frac_ab / frac_expected)`` where both are > 0

Bins use their own, coarser grid (``coexpression.bin_um``, ``n_norm_bins``):
a correlation needs more nuclei than a mean, and a bin short of
``min_nuclei_per_bin`` is NaN rather than a correlation of five points.

Well level: colonies first
--------------------------
Across a well, every readout is computed **per colony, then averaged over
colonies** (n = colonies), never on pooled nuclei. Pooling would let a level
difference between colonies (one colony brighter in both markers) masquerade as
a within-colony correlation: Simpson's paradox. Correlations are averaged on the
Fisher-z scale and transformed back.

Pairs
-----
Only markers measured in the same nuclei can be paired, so pairs are formed
within each stain set (panel): every combination by default
(``coexpression.pairs: auto``), or the explicit list given. A panel with fewer
than two analysis markers yields no pairs, and the step then writes nothing.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import radial

logger = logging.getLogger(__name__)

# Must match plots.joint.PANEL_KINDS (kept here so this module needs no matplotlib).
PANEL_KINDS = ("scatter", "bands", "correlation", "double_positive", "trajectory", "map")

DEFAULTS = {
    "pairs": "auto",              # auto | [[A, B], ...]
    "method": "spearman",         # spearman | pearson
    "bin_um": 25.0,               # µm grid for the readouts (coarser than the profiles)
    "n_norm_bins": 10,            # normalized grid for the readouts
    "min_nuclei_per_bin": 20,     # fewer nuclei in a bin → NaN
    "n_bands": 4,                 # radial bands in the split scatter panels
    "max_points": 4000,           # per-colony scatter: nuclei drawn (display only)
    "points_per_colony": 1500,    # well scatter: at most this many nuclei per colony, same for each
    "colony_figures": True,       # one page per colony (µm grid)
    "well_figures": True,         # one page per well (normalized grid)
    # Each page panel also as its own slide-sized figure, under
    # figures/coexpression/panels/<level>/<kind>/. Levels: wells, colonies.
    "panels": ["wells"],
    "panel_kinds": list(PANEL_KINDS),
}
_METHODS = {"spearman", "pearson"}

# Columns of the per-bin readout tables, in order.
VALUES = ["correlation", "mean_a", "mean_b", "frac_a", "frac_b", "frac_ab",
          "frac_expected", "log2_enrichment"]


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("coexpression") or {}).items() if v is not None})
    out["method"] = str(out["method"]).lower()
    if out["method"] not in _METHODS:
        raise ValueError(f"coexpression.method must be one of {sorted(_METHODS)}")
    if float(out["bin_um"]) <= 0 or int(out["n_norm_bins"]) < 1:
        raise ValueError("coexpression.bin_um and n_norm_bins must be positive")
    if int(out["n_bands"]) < 1:
        raise ValueError("coexpression.n_bands must be at least 1")
    for key, allowed in (("panels", {"wells", "colonies"}), ("panel_kinds", set(PANEL_KINDS))):
        value = out[key]
        value = [value] if isinstance(value, str) else list(value or [])
        bad = sorted(set(map(str, value)) - allowed)
        if bad:
            raise ValueError(f"coexpression.{key}: unknown {bad}; choose from {sorted(allowed)}")
        out[key] = [str(v) for v in value]
    return out


def pair_label(a: str, b: str) -> str:
    return f"{a} vs {b}"


# ---------------------------------------------------------------------------
# Per-nucleus wide tables
# ---------------------------------------------------------------------------

def wide_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{'value', 'positive', 'gate'}``: cell_uid × marker tables from the long one.

    ``value`` is the shared-scale intensity, ``gate`` the nucleus's positivity
    threshold on that scale (NaN where the marker has no gate).
    """
    keyed = long.assign(marker=long["marker"].astype(str)).set_index(["cell_uid", "marker"])
    keyed = keyed[~keyed.index.duplicated()]
    out = {"value": keyed["scaled"].unstack(), "positive": keyed["positive"].unstack()}
    out["gate"] = (keyed["threshold_scaled"].unstack() if "threshold_scaled" in keyed
                   else out["value"] * np.nan)
    return out


def marker_pairs(values: pd.DataFrame, cells: pd.DataFrame, cfg: dict,
                 order: list[str]) -> list[tuple[str, str]]:
    """Marker pairs measured in the same nuclei, in display order.

    A marker belongs to a stain set when most of that panel's nuclei carry a
    value for it. Explicit ``coexpression.pairs`` are kept only when the two
    markers share a panel; the others are reported and dropped.
    """
    rank = {m: i for i, m in enumerate(order)}
    panel = (cells["stain_set"].astype(str).reindex(values.index) if "stain_set" in cells
             else pd.Series("all", index=values.index))
    measured: dict[str, list[str]] = {}
    for name, idx in panel.groupby(panel, observed=True).groups.items():
        have = values.loc[idx].notna().mean()
        measured[str(name)] = sorted((str(m) for m in have.index[have > 0.5]),
                                     key=lambda m: (rank.get(m, 999), m))
    together = set()
    for markers in measured.values():
        together.update(itertools.combinations(markers, 2))

    spec = (cfg.get("coexpression") or {}).get("pairs", "auto")
    if isinstance(spec, (list, tuple)) and spec:
        pairs, dropped = [], []
        for item in spec:
            a, b = (str(v) for v in item)
            if (a, b) in together or (b, a) in together:
                pairs.append((a, b))
            else:
                dropped.append(f"{a}/{b}")
        if dropped:
            logger.warning("coexpression.pairs: %s not measured in the same nuclei (no "
                           "shared stain set); left out.", ", ".join(dropped))
        return list(dict.fromkeys(pairs))
    return sorted(together, key=lambda p: (rank.get(p[0], 999), rank.get(p[1], 999), p))


# ---------------------------------------------------------------------------
# Per-colony, per-bin readouts
# ---------------------------------------------------------------------------

def grid_edges(grid: str, prm: dict, profiles_prm: dict, max_dist_um: float) -> np.ndarray:
    """The readout grid: the profile grid's extent with this module's bin widths."""
    grid_prm = dict(profiles_prm, bin_um=prm["bin_um"], n_norm_bins=prm["n_norm_bins"])
    return radial.grid_edges(grid, grid_prm, max_dist_um)


def _fisher_ci(r, n, method: str, ci: float):
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        k = 1.06 if method == "spearman" else 1.0
        se = np.sqrt(k / (n - 3))
        q = sps.norm.ppf(0.5 + ci / 200.0)
        lo, hi = np.tanh(z - q * se), np.tanh(z + q * se)
    bad = ~(n > 3)
    lo[bad] = np.nan
    hi[bad] = np.nan
    return lo, hi


def pair_bins(cells: pd.DataFrame, a: str, b: str, wide: dict, grid: str,
              edges: np.ndarray, prm: dict, ci: float = 95.0) -> pd.DataFrame:
    """One row per colony × bin for the pair (*a*, *b*) on *grid*.

    *cells* is indexed by cell_uid and holds ``field_id`` and the radial
    coordinates. Correlations are computed with grouped sums (ranks within each
    colony × bin for Spearman), so a whole experiment takes seconds.
    """
    coord = cells["dist_edge_um" if grid == radial.UM else "r_norm"]
    df = pd.DataFrame({
        "field_id": cells["field_id"], "x": coord,
        "va": wide["value"][a].reindex(cells.index), "vb": wide["value"][b].reindex(cells.index),
        "pa": wide["positive"][a].reindex(cells.index),
        "pb": wide["positive"][b].reindex(cells.index),
    })
    x = df["x"].to_numpy(np.float64)
    ok = (np.isfinite(x) & np.isfinite(df["va"].to_numpy(np.float64))
          & np.isfinite(df["vb"].to_numpy(np.float64)) & (x >= edges[0]) & (x <= edges[-1]))
    df = df[ok].copy()
    if df.empty:
        return pd.DataFrame()
    df["bin"] = np.clip(np.searchsorted(edges, df["x"].to_numpy(), side="right") - 1,
                        0, len(edges) - 2)
    keys = ["field_id", "bin"]
    g = df.groupby(keys, observed=True, dropna=False)
    if prm["method"] == "spearman":
        ra, rb = g["va"].rank(), g["vb"].rank()
    else:
        ra, rb = df["va"], df["vb"]
    df["ca"] = ra - ra.groupby([df["field_id"], df["bin"]]).transform("mean")
    df["cb"] = rb - rb.groupby([df["field_id"], df["bin"]]).transform("mean")
    df["ab"], df["aa"], df["bb"] = df["ca"] * df["cb"], df["ca"] ** 2, df["cb"] ** 2
    both = np.isfinite(df["pa"]) & np.isfinite(df["pb"])
    df["pab"] = np.where(both, df["pa"] * df["pb"], np.nan)
    df["pa"] = np.where(both, df["pa"], np.nan)
    df["pb"] = np.where(both, df["pb"], np.nan)

    out = df.groupby(keys, observed=True, dropna=False).agg(
        n_nuclei=("va", "size"), sab=("ab", "sum"), saa=("aa", "sum"), sbb=("bb", "sum"),
        mean_a=("va", "mean"), mean_b=("vb", "mean"), frac_a=("pa", "mean"),
        frac_b=("pb", "mean"), frac_ab=("pab", "mean")).reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["correlation"] = out["sab"] / np.sqrt(out["saa"] * out["sbb"])
        out["frac_expected"] = out["frac_a"] * out["frac_b"]
        ratio = out["frac_ab"] / out["frac_expected"]
        out["log2_enrichment"] = np.where((out["frac_ab"] > 0) & (out["frac_expected"] > 0),
                                          np.log2(ratio), np.nan)
    n = out["n_nuclei"].to_numpy(np.float64)
    lo, hi = _fisher_ci(out["correlation"].to_numpy(np.float64), n, prm["method"], ci)
    out["corr_ci_lo"], out["corr_ci_hi"] = lo, hi
    out = out.drop(columns=["sab", "saa", "sbb"])
    thin = n < int(prm["min_nuclei_per_bin"])
    out.loc[thin, VALUES + ["corr_ci_lo", "corr_ci_hi"]] = np.nan

    bins = out["bin"].to_numpy()
    out.insert(1, "pair", pair_label(a, b))
    out.insert(2, "marker_a", a)
    out.insert(3, "marker_b", b)
    out.insert(4, "grid", grid)
    out["bin_lo"], out["bin_hi"] = edges[bins], edges[bins + 1]
    out["bin_center"] = 0.5 * (out["bin_lo"] + out["bin_hi"])
    return out


def colony_table(cells: pd.DataFrame, pairs, wide: dict, prm: dict, profiles_prm: dict,
                 ci: float = 95.0) -> pd.DataFrame:
    """Every pair × grid × colony × bin (only colonies measured for both markers)."""
    max_dist = float(np.nanmax(cells["dist_edge_um"])) if len(cells) else 0.0
    blocks = []
    for a, b in pairs:
        for grid in radial.GRIDS:
            edges = grid_edges(grid, prm, profiles_prm, max_dist)
            part = pair_bins(cells, a, b, wide, grid, edges, prm, ci)
            if not part.empty:
                blocks.append(part)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Well level: colonies first, then their mean
# ---------------------------------------------------------------------------

def _summary(frame: pd.DataFrame, keys: list[str], value: str, ci: float,
             need: int) -> pd.DataFrame:
    """Mean and t-CI over colonies of one readout; correlations on the Fisher-z scale."""
    v = frame[value].to_numpy(np.float64)
    sub = frame.loc[np.isfinite(v), keys].copy()
    vals = v[np.isfinite(v)]
    fisher = value == "correlation"
    sub["_v"] = np.arctanh(np.clip(vals, -0.999999, 0.999999)) if fisher else vals
    g = sub.groupby(keys, observed=True, dropna=False)["_v"]
    out = g.agg(m="mean", sd="std", n="size").reset_index()
    n = out["n"].to_numpy(np.float64)
    _, lo, hi = radial._t_ci(out["m"].to_numpy(), out["sd"].to_numpy(), n, ci)
    if fisher:
        out["m"], lo, hi = np.tanh(out["m"]), np.tanh(lo), np.tanh(hi)
    # Same rule as the profiles: points reached by too few colonies are blanked,
    # unless the well never has enough anywhere (then: mean without a band).
    curve = [k for k in keys if k not in ("bin", "bin_lo", "bin_hi", "bin_center")]
    most = out.groupby(curve, observed=True, dropna=False)["n"].transform("max").to_numpy()
    thin = n < need
    out.loc[thin & (most >= need), "m"] = np.nan
    lo[thin] = np.nan
    hi[thin] = np.nan
    out = out.rename(columns={"m": value, "n": f"n_{value}"})
    out[f"{value}_ci_lo"], out[f"{value}_ci_hi"] = lo, hi
    return out.drop(columns="sd")


def well_table(colonies: pd.DataFrame, by: list[str], ci: float = 95.0,
               min_replicates: int = 3) -> pd.DataFrame:
    """Per well × pair × grid × bin: the mean over colonies of every readout."""
    if colonies.empty:
        return pd.DataFrame()
    keys = [k for k in dict.fromkeys(by + ["pair", "marker_a", "marker_b", "grid", "bin",
                                          "bin_lo", "bin_hi", "bin_center"]) if k in colonies]
    out = colonies.groupby(keys, observed=True, dropna=False).agg(
        n_colonies=("field_id", "nunique"), n_nuclei=("n_nuclei", "sum")).reset_index()
    for value in VALUES:
        out = out.merge(_summary(colonies, keys, value, ci, min_replicates), on=keys,
                        how="left")
    return out


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def colony_gate(wide: dict, marker: str, uids) -> float | None:
    """The marker's gate on the shared scale for these nuclei (one scope → one value)."""
    if marker not in wide["gate"]:
        return None
    g = wide["gate"][marker].reindex(uids).to_numpy(np.float64)
    g = g[np.isfinite(g)]
    return float(np.median(g)) if g.size else None


def equal_subsample(cells: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """The same number of nuclei from every colony (at most *cap*), for pooled scatters.

    Without this a pooled scatter is a picture of the largest colony.
    """
    sizes = cells.groupby("field_id", observed=True).size()
    if sizes.empty:
        return cells
    take = int(min(cap, sizes.min()))
    parts = [grp.iloc[rng.choice(len(grp), size=take, replace=False)]
             for _, grp in cells.groupby("field_id", observed=True)]
    return pd.concat(parts)


def classes(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Co-expression class per nucleus: 'AB', 'A', 'B', 'none' ('' when not callable)."""
    pa = np.asarray(pos_a, np.float64)
    pb = np.asarray(pos_b, np.float64)
    out = np.full(pa.shape, "", dtype=object)
    ok = np.isfinite(pa) & np.isfinite(pb)
    a, b = (pa >= 0.5) & ok, (pb >= 0.5) & ok
    out[ok] = "none"
    out[a & ~b] = "A"
    out[~a & b] = "B"
    out[a & b] = "AB"
    return out


def correlation(x, y, method: str = "spearman") -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if method == "spearman":
        return float(sps.spearmanr(x[ok], y[ok]).statistic)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


__all__ = ["DEFAULTS", "VALUES", "settings", "pair_label", "wide_tables", "marker_pairs",
           "grid_edges", "pair_bins", "colony_table", "well_table", "colony_gate",
           "equal_subsample", "classes", "correlation"]
