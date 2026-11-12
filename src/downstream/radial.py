"""
Radial profiles: from nuclei to one curve per colony, on two coordinates.

The unit of analysis
--------------------
A nucleus is a measurement; a **colony** is the biological replicate. Every curve
built here is a per-colony curve, and every error bar drawn from more than one
colony has ``n`` = colonies (or wells: see :func:`choose_unit`), never nuclei.

One shared intensity scale
--------------------------
Each nucleus's value is its marker intensity in the marker's analysis space
(``default_space`` / ``markers.<m>.space``: DAPI-normalized by default), then
rescaled with **one pair of anchors for the whole scope**: by default the plate
× panel the marker's positivity gate is estimated over (:mod:`downstream.scope`)::

    scaled = (value − P_lo) / (P_hi − P_lo)

with ``P_lo``/``P_hi`` the ``profiles.robust_percentiles`` of all in-colony nuclei
in that scope. Because every colony in the scope uses the *same* anchors, a
colony that expresses a marker at half the level reads half as high: differences
in level are kept, not normalized away, and a colony that does not express the
marker stays near 0 instead of having its noise stretched into a pattern. The
scale is not clipped, so the transform is linear and a mean of scaled values is
the scaled mean.

The assumption this makes is that the colonies of one scope were stained and
imaged alike: one antibody mix, one exposure. That is the same assumption a
positivity threshold estimated over the scope makes; markers declared
``scope: plate`` share anchors across panels exactly as they share a gate.

Two grids
---------
``um``    fixed-width bins of ``profiles.bin_um`` measured **inward from the edge**
          (0 = edge). A colony's curve stops at its own radius. This is the
          coordinate BMP-driven micropattern fates are organized on (a band of
          fixed width from the edge), and the only fair one when colony sizes
          differ.
``norm``  ``profiles.n_norm_bins`` equal bins of ``r_norm = dist_edge / R``
          (0 = edge, 1 = center), for comparing colonies of one nominal size.

In both, a colony's bin value needs ``min_nuclei_per_bin`` nuclei, and a
well/condition mean needs ``min_replicates`` colonies (or wells) at that point;
short of either the point is NaN rather than a small-n spike: the disk center is
where both run out first.
"""

from __future__ import annotations

import logging
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import scope as scope_mod

logger = logging.getLogger(__name__)

UM, NORM = "um", "norm"
GRIDS = (UM, NORM)
GRID_LABEL = {UM: "Distance from colony edge (µm)",
              NORM: "Normalized distance from edge (0 = edge, 1 = center)"}

DEFAULTS = {
    "bin_um": 10.0,
    "max_um": None,                 # None → the largest colony's radius
    "n_norm_bins": 25,
    "min_nuclei_per_bin": 10,
    "robust_percentiles": [1.0, 99.0],
    "ci": 95.0,
    "colony_binning": "quantile",   # per-colony figures: quantile | equal_area | fixed
    "colony_n_bins": 20,
    "ci_method": "t",               # per-colony figures: t | bootstrap
    "bootstrap_n": 1000,
    "min_replicates": 3,
}
_BINNINGS = {"quantile", "equal_area", "fixed"}

# Columns describing a colony (one per imaged field) carried onto every curve row.
COLONY_KEYS = ["field_id", "dataset", "plate_id", "stain_set", "seg_mode", "well",
               "row", "col", "image"]


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("profiles") or {}).items() if v is not None})
    if out["colony_binning"] not in _BINNINGS:
        raise ValueError(f"profiles.colony_binning must be one of {sorted(_BINNINGS)}")
    if out["ci_method"] not in {"t", "bootstrap"}:
        raise ValueError("profiles.ci_method must be 't' or 'bootstrap'")
    if float(out["bin_um"]) <= 0:
        raise ValueError("profiles.bin_um must be positive")
    return out


# ---------------------------------------------------------------------------
# Per-nucleus values on the shared scale
# ---------------------------------------------------------------------------

def shared_anchors(long: pd.DataFrame, pct=(1.0, 99.0)) -> pd.DataFrame:
    """One ``(lo, hi)`` pair per (marker × scope_key) from all its nuclei."""
    rows = []
    for (marker, key), grp in long.groupby(["marker", "scope_key"], observed=True):
        v = grp["value"].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        lo, hi = np.percentile(v, pct)
        if not hi > lo:
            hi = lo + max(abs(lo), 1e-9)
        rows.append({"marker": str(marker), "scope_key": str(key), "p_lo": float(lo),
                     "p_hi": float(hi), "percentiles": f"{pct[0]:g}-{pct[1]:g}",
                     "n_nuclei": int(v.size), "n_colonies": int(grp["field_id"].nunique())})
    return pd.DataFrame(rows)


def apply_scale(long: pd.DataFrame, anchors: pd.DataFrame, column: str = "value") -> pd.Series:
    """Rescale ``long[column]`` with its own (marker, scope_key) anchors.

    *column* is ``value`` for the nuclei; ``threshold`` puts each row's positivity
    gate on the same scale, so a gate can be drawn on a shared-scale axis.
    """
    idx = pd.MultiIndex.from_frame(anchors[["marker", "scope_key"]].astype(str))
    lo = pd.Series(anchors["p_lo"].to_numpy(), index=idx)
    hi = pd.Series(anchors["p_hi"].to_numpy(), index=idx)
    keys = pd.MultiIndex.from_arrays([long["marker"].astype(str), long["scope_key"].astype(str)])
    lo_v = lo.reindex(keys).to_numpy(np.float64)
    hi_v = hi.reindex(keys).to_numpy(np.float64)
    return pd.Series((long[column].to_numpy(np.float64) - lo_v) / (hi_v - lo_v),
                     index=long.index)


def nucleus_table(data, gated: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """Long (nucleus × marker) table with the scaled value and the radial coordinates.

    *gated* is the thresholded long table (value / threshold / positive) when a
    thresholds table exists; without one the value is taken straight from the
    analysis space and ``threshold`` and ``positive`` are NaN throughout.
    """
    if gated is not None:
        cols = ["cell_uid", "marker", "value", "positive"]
        long = gated[cols + (["threshold"] if "threshold" in gated else [])].copy()
        if "threshold" not in long:
            long["threshold"] = np.nan
    else:
        long = data.intensities[["cell_uid", "marker"]].copy()
        long["value"] = data.values(data.intensities, cfg)
        long["positive"] = np.nan
        long["threshold"] = np.nan

    uid = long["cell_uid"].to_numpy()
    cells = data.cells
    for col in ["field_id", "dist_edge_um", "r_norm"] + [
            c for c in scope_mod.UNION_ORDER if c in cells.columns]:
        long[col] = cells[col].to_numpy()[uid]
    available = [c for c in scope_mod.UNION_ORDER if c in long.columns]
    long["scope_key"] = scope_mod.scope_keys(long, cfg, available)
    return long


# ---------------------------------------------------------------------------
# Grids and per-colony curves
# ---------------------------------------------------------------------------

def grid_edges(grid: str, prm: dict, max_dist_um: float) -> np.ndarray:
    if grid == NORM:
        return np.linspace(0.0, 1.0, int(prm["n_norm_bins"]) + 1)
    width = float(prm["bin_um"])
    top = float(prm["max_um"]) if prm.get("max_um") else float(max_dist_um)
    n = max(1, int(np.ceil(top / width)))
    return np.arange(n + 1) * width


def _coord(long: pd.DataFrame, grid: str) -> np.ndarray:
    return long["dist_edge_um" if grid == UM else "r_norm"].to_numpy(np.float64)


def _t_ci(mean, sd, n, ci):
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = sd / np.sqrt(n)
        t = sps.t.ppf(0.5 + ci / 200.0, np.maximum(n - 1, 1))
        half = np.where(n > 1, t * sem, np.nan)
    return sem, mean - half, mean + half


def colony_curves(long: pd.DataFrame, grid: str, edges: np.ndarray, prm: dict) -> pd.DataFrame:
    """One row per (colony × marker × bin): nuclei count, mean ± CI, positive fraction."""
    x = _coord(long, grid)
    ok = np.isfinite(x) & np.isfinite(long["scaled"].to_numpy(np.float64))
    # The last edge is inclusive: the center-most nucleus of the largest colony
    # (r_norm = 1, or dist = max_um) belongs to the last bin, not to nothing.
    ok &= (x >= edges[0]) & (x <= edges[-1])
    sub = long.loc[ok, ["field_id", "marker", "scaled", "positive"]].copy()
    sub["bin"] = np.clip(np.searchsorted(edges, x[ok], side="right") - 1, 0, len(edges) - 2)

    g = sub.groupby(["field_id", "marker", "bin"], observed=True)
    out = g.agg(n_nuclei=("scaled", "size"), mean=("scaled", "mean"), sd=("scaled", "std"),
                positive_fraction=("positive", "mean"),
                n_called=("positive", "count")).reset_index()
    n = out["n_nuclei"].to_numpy(np.float64)
    sem, lo, hi = _t_ci(out["mean"].to_numpy(), out["sd"].to_numpy(), n, float(prm["ci"]))
    out["sem"], out["ci_lo"], out["ci_hi"] = sem, lo, hi

    thin = n < int(prm["min_nuclei_per_bin"])
    out.loc[thin, ["mean", "sd", "sem", "ci_lo", "ci_hi", "positive_fraction"]] = np.nan
    out.loc[out["n_called"] == 0, "positive_fraction"] = np.nan
    out = out.drop(columns="n_called")
    b = out["bin"].to_numpy()
    out.insert(2, "grid", grid)
    out["bin_lo"], out["bin_hi"] = edges[b], edges[b + 1]
    out["bin_center"] = 0.5 * (out["bin_lo"] + out["bin_hi"])
    out["marker"] = out["marker"].astype(str)
    return out


DENSITY = "nuclei per 1000 µm²"


def ring_areas(dist_um: np.ndarray | None, edges: np.ndarray, pixel_um: float = 1.0,
               radius_um: float | None = None) -> np.ndarray:
    """Colony area (µm²) between each pair of distance-from-edge *edges*.

    From the colony mask's pixel distances to the edge (*dist_um*, one per pixel
    inside the colony) when given, else for a disk of *radius_um*.
    """
    edges = np.asarray(edges, np.float64)
    if dist_um is not None:
        counts, _ = np.histogram(dist_um, bins=edges)
        return counts * float(pixel_um) ** 2
    r = float(radius_um)
    outer = np.clip(r - edges[:-1], 0, None)
    inner = np.clip(r - edges[1:], 0, None)
    return np.pi * (outer ** 2 - inner ** 2)


def density_curves(long: pd.DataFrame, edges: np.ndarray, edge_area) -> pd.DataFrame:
    """Nuclei per 1000 µm² of colony per distance-from-edge bin, per colony.

    Shaped like :func:`colony_curves` (``marker`` = :data:`DENSITY`, value in
    ``mean``) so the same replicate averaging and figures apply. *edge_area*:
    ``f(field_id, edges)`` → area per bin (µm²), e.g. from :func:`ring_areas`.
    """
    nuc = long.drop_duplicates("cell_uid")[["field_id", "dist_edge_um"]]
    nuc = nuc[np.isfinite(nuc["dist_edge_um"].to_numpy(np.float64))]
    rows = []
    for field_id, grp in nuc.groupby("field_id", observed=True):
        n, _ = np.histogram(grp["dist_edge_um"].to_numpy(np.float64), bins=edges)
        area = np.asarray(edge_area(field_id, edges), np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            dens = np.where(area > 0, 1000.0 * n / area, np.nan)
        # A bin the colony barely reaches (its last, partial ring) is noise.
        dens[area < 0.05 * np.nanmax(area)] = np.nan
        rows.append(pd.DataFrame({"field_id": field_id, "marker": DENSITY, "grid": UM,
                                  "bin": np.arange(n.size), "n_nuclei": n, "area_um2": area,
                                  "mean": dens, "positive_fraction": np.nan}))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["bin_lo"], out["bin_hi"] = edges[out["bin"]], edges[out["bin"] + 1]
    out["bin_center"] = 0.5 * (out["bin_lo"] + out["bin_hi"])
    return out


def colony_binned(long: pd.DataFrame, prm: dict, edge_quantiles=None,
                  rng: np.random.Generator | None = None, edge_area=None) -> pd.DataFrame:
    """Per-colony profile on the colony's *own* bins (for the per-colony figure).

    ``quantile`` puts ≈ the same number of nuclei in each bin; ``equal_area``
    gives each bin ≈ the same colony area, from ``edge_quantiles(field_id, q)``:
    the quantiles of the colony's pixel distances to its edge; ``fixed`` uses the
    shared µm grid width. The x coordinate is always µm from the edge. With
    *edge_area* (``f(field_id, edges)`` → µm² per bin) each bin also gets its
    colony area and nuclear density (``area_um2``, ``density`` per 1000 µm²), so
    the figure can show how wide, and how crowded, every bin is.
    """
    rng = rng or np.random.default_rng(0)
    n_bins = int(prm["colony_n_bins"])
    q = np.linspace(0, 1, n_bins + 1)
    rows: list[pd.DataFrame] = []
    for field_id, grp in long.groupby("field_id", observed=True):
        dist = grp["dist_edge_um"].to_numpy(np.float64)
        ok = np.isfinite(dist)
        if ok.sum() < n_bins:
            continue
        if prm["colony_binning"] == "quantile":
            # Quantiles over nuclei, not nucleus×marker rows (each nucleus is
            # repeated once per marker, which does not change the quantiles).
            edges = np.quantile(dist[ok], q)
        elif prm["colony_binning"] == "equal_area" and edge_quantiles is not None:
            edges = np.asarray(edge_quantiles(field_id, q), dtype=np.float64)
        else:
            width = float(prm["bin_um"])
            edges = np.arange(0, np.nanmax(dist) + width, width)
        edges = np.maximum.accumulate(edges) + np.arange(edges.size) * 1e-9
        b = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, len(edges) - 2)
        part = pd.DataFrame({"marker": grp["marker"].astype(str).to_numpy(), "bin": b,
                             "x": dist, "v": grp["scaled"].to_numpy(np.float64),
                             "pos": grp["positive"].to_numpy(np.float64)})[ok]
        part = part[np.isfinite(part["v"])]
        agg = part.groupby(["marker", "bin"]).agg(
            n_nuclei=("v", "size"), x=("x", "mean"), mean=("v", "mean"), sd=("v", "std"),
            positive_fraction=("pos", "mean")).reset_index()
        n = agg["n_nuclei"].to_numpy(np.float64)
        sem, lo, hi = _t_ci(agg["mean"].to_numpy(), agg["sd"].to_numpy(), n, float(prm["ci"]))
        if prm["ci_method"] == "bootstrap":
            lo, hi = _bootstrap_ci(part, agg, int(prm["bootstrap_n"]), float(prm["ci"]), rng)
        agg["sem"], agg["ci_lo"], agg["ci_hi"] = sem, lo, hi
        agg["bin_lo"], agg["bin_hi"] = edges[agg["bin"]], edges[agg["bin"] + 1]
        if edge_area is not None:
            area = np.asarray(edge_area(field_id, edges), np.float64)
            uniq = grp.drop_duplicates("cell_uid")["dist_edge_um"].to_numpy(np.float64) \
                if "cell_uid" in grp else dist[ok]
            nuclei, _ = np.histogram(uniq[np.isfinite(uniq)], bins=edges)
            agg["area_um2"] = area[agg["bin"]]
            with np.errstate(invalid="ignore", divide="ignore"):
                agg["density"] = np.where(agg["area_um2"] > 0,
                                          1000.0 * nuclei[agg["bin"]] / agg["area_um2"], np.nan)
        agg.insert(0, "field_id", field_id)
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["binning"] = prm["colony_binning"]
    return out


def _bootstrap_ci(part, agg, n_boot, ci, rng):
    lo = np.full(len(agg), np.nan)
    hi = np.full(len(agg), np.nan)
    groups = part.groupby(["marker", "bin"])["v"]
    index = {key: i for i, key in enumerate(zip(agg["marker"], agg["bin"]))}
    for key, values in groups:
        v = values.to_numpy(np.float64)
        if v.size < 2:
            continue
        means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
        i = index[key]
        lo[i], hi[i] = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


# ---------------------------------------------------------------------------
# Replication unit and summaries
# ---------------------------------------------------------------------------

def choose_unit(colonies: pd.DataFrame, cfg: dict) -> tuple[str, str]:
    """``(unit, note)``: whether colonies or wells are the replicate for inference.

    ``stats.replicate: auto`` (default) uses **wells** when every condition has at
    least ``stats.min_wells_for_well_unit`` wells (3), because colonies in one well
    share one seeding, one BMP4 dose and one staining reaction and are therefore
    not independent of each other. Otherwise colonies are the replicate (the
    only option when a condition sits in one well), and the note says plainly
    that well-to-well variation cannot then be separated from the condition.
    """
    st = cfg.get("stats") or {}
    mode = str(st.get("replicate", "auto")).lower()
    min_wells = int(st.get("min_wells_for_well_unit", 3))
    per = colonies.groupby("condition", observed=True)["well_key"].nunique()
    fewest = int(per.min()) if len(per) else 0
    exp_col = experiment_column(colonies, cfg)
    if exp_col:
        per_e = colonies.groupby("condition", observed=True)[exp_col].nunique()
        fewest_e = int(per_e.min()) if len(per_e) else 0
        min_exp = int(st.get("min_experiments_for_unit", 2))
    if mode in ("colony", "well", "experiment"):
        unit = mode
        if unit == "experiment" and not exp_col:
            raise ValueError("stats.replicate: experiment needs stats.experiment_column naming "
                             "a plate-map column present in the manifest")
    elif exp_col and fewest_e >= min_exp:
        unit = "experiment"
    else:
        unit = "well" if fewest >= min_wells else "colony"

    if unit == "experiment":
        return unit, (f"Replicate = experiment ('{exp_col}': every condition has ≥ "
                      f"{fewest_e} independent differentiations); colonies are averaged per "
                      f"well and wells per experiment, so n = experiments.")
    if exp_col and fewest_e < 2:
        extra = (f" Only {fewest_e} '{exp_col}' level(s) for some condition: differences "
                 f"between differentiations cannot be estimated.")
    else:
        extra = ""
    if unit == "well":
        note = (f"Replicate = well (every condition has ≥ {fewest} wells); colonies are "
                f"averaged within their well first.")
    elif fewest <= 1:
        note = ("Replicate = colony. At least one condition was imaged in a single "
                "well, so well-to-well variation (seeding, dosing, staining) cannot be "
                "separated from the condition effect: a significant difference shows "
                "the colonies differ, not that it would replicate in another well.")
    else:
        note = (f"Replicate = colony (the fewest wells per condition is {fewest}, below "
                f"the {min_wells} needed to use wells). Colonies from the same well are "
                f"not fully independent, so p-values are optimistic.")
    return unit, note + extra


def experiment_column(frame: pd.DataFrame, cfg: dict) -> str | None:
    """``stats.experiment_column`` when set and present in *frame*, else ``None``."""
    col = (cfg.get("stats") or {}).get("experiment_column")
    return str(col) if col and str(col) in frame.columns else None


def to_unit(curves: pd.DataFrame, unit: str, experiment_col: str | None = None) -> pd.DataFrame:
    """Average colony curves within each well (unit well), and the well curves
    within each experiment (unit experiment: each well weighs the same)."""
    if unit == "experiment":
        if not experiment_col or experiment_col not in curves:
            raise ValueError("to_unit: the experiment unit needs the experiment column")
        wells = to_unit(curves, "well")
        keys = [c for c in wells.columns if c in ("condition", "marker", "grid", "bin", "bin_lo",
                                                  "bin_hi", "bin_center")] + [experiment_col]
        extra = [c for c in wells.columns if c not in keys and c not in (
            "well_key", "unit_id", "mean", "positive_fraction", "n_nuclei", "n_colonies",
            "well", "row", "col", "colony", "result_dir", "radius_um", "flags")]
        agg = {"mean": ("mean", "mean"), "positive_fraction": ("positive_fraction", "mean"),
               "n_nuclei": ("n_nuclei", "sum"), "n_colonies": ("n_colonies", "sum"),
               "n_wells": ("well_key", "nunique")}
        out = wells.groupby(keys, observed=True, dropna=False).agg(**agg).reset_index()
        if extra:
            first = wells.groupby(["condition", experiment_col], observed=True,
                                  dropna=False)[extra].first()
            out = out.join(first, on=["condition", experiment_col])
        out["unit_id"] = out["condition"].astype(str) + "|" + out[experiment_col].astype(str)
        return out
    if unit != "well":
        out = curves.copy()
        out["unit_id"] = out["field_id"].astype(str)
        return out
    keys = [c for c in curves.columns if c in ("well_key", "condition", "marker", "grid",
                                               "bin", "bin_lo", "bin_hi", "bin_center")]
    extra = [c for c in curves.columns if c not in keys and c not in (
        "field_id", "image", "n_nuclei", "mean", "sd", "sem", "ci_lo", "ci_hi",
        "positive_fraction")]
    agg = {"mean": ("mean", "mean"), "positive_fraction": ("positive_fraction", "mean"),
           "n_nuclei": ("n_nuclei", "sum"), "n_colonies": ("field_id", "nunique")}
    out = curves.groupby(keys, observed=True, dropna=False).agg(**agg).reset_index()
    if extra:
        first = curves.groupby("well_key", observed=True)[extra].first()
        out = out.join(first, on="well_key")
    out["unit_id"] = out["well_key"].astype(str)
    return out


def summarize(curves: pd.DataFrame, by: list[str], value: str, prm: dict) -> pd.DataFrame:
    """Mean ± SEM / t-CI across replicate curves, per *by* × marker × grid × bin."""
    keys = by + ["marker", "grid", "bin", "bin_lo", "bin_hi", "bin_center"]
    sub = curves[np.isfinite(curves[value].to_numpy(np.float64))]
    g = sub.groupby(keys, observed=True, dropna=False)[value]
    out = g.agg(mean="mean", sd="std", n="size").reset_index()
    n = out["n"].to_numpy(np.float64)
    sem, lo, hi = _t_ci(out["mean"].to_numpy(), out["sd"].to_numpy(), n, float(prm["ci"]))
    out["sem"], out["ci_lo"], out["ci_hi"] = sem, lo, hi
    need = int(prm["min_replicates"])
    # Points reached by too few replicates are blanked (the sparse tail near the
    # center). A group that never has enough replicates anywhere, e.g. a control
    # imaged as a single colony, keeps its mean curve without a confidence band
    # instead of vanishing from the figure.
    most = out.groupby(keys[:-4], observed=True, dropna=False)["n"].transform("max")
    small_group = (most < need).to_numpy()
    thin = n < need
    out.loc[thin & ~small_group, ["mean", "sd", "sem", "ci_lo", "ci_hi"]] = np.nan
    out.loc[thin & small_group, ["sd", "sem", "ci_lo", "ci_hi"]] = np.nan
    out.insert(len(by), "readout", value)
    return out


def display_limits(curves: pd.DataFrame, value: str = "mean", pad: float = 0.05) -> dict:
    """One y-range per marker, shared by every figure that draws that marker.

    From the colony curves themselves (each point a mean over ≥ min_nuclei_per_bin
    nuclei, so one bright nucleus cannot stretch it): ``hi = max`` and ``lo = 0``
    when the minimum is below a tenth of the maximum (shared-scale markers),
    else the minimum (raw a.u., where a zero floor would flatten every curve);
    padded by *pad* of the span. CI bands may run past it and are
    clipped. Values are on the scope's shared scale, so the range is comparable
    across colonies and conditions of a marker, not across markers.
    """
    out = {}
    if curves is None or curves.empty or value not in curves:
        return out
    for marker, part in curves.groupby("marker", observed=True):
        v = part[value].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        lo, hi = float(v.min()), float(v.max())
        if 0.0 <= lo <= 0.1 * hi:
            lo = 0.0
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        out[str(marker)] = (lo - pad * span, hi + pad * span)
    return out


def n_labelings(counts) -> int:
    """Distinct ways to assign the group labels (multinomial coefficient)."""
    total = factorial(int(sum(counts)))
    for c in counts:
        total //= factorial(int(c))
    return total


__all__ = ["DENSITY", "ring_areas", "density_curves", "settings", "shared_anchors",
           "apply_scale", "nucleus_table", "grid_edges",
           "colony_curves", "colony_binned", "choose_unit", "to_unit", "summarize",
           "display_limits", "experiment_column",
           "UM", "NORM", "GRIDS", "GRID_LABEL", "COLONY_KEYS", "DEFAULTS"]
