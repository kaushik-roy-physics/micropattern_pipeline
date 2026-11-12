"""
Effect sizes, trends, variance components and power from per-colony numbers.

Every scalar the pipeline measures per colony (boundary positions and widths,
peak offsets, roughness, profile features) is a candidate answer to "how big is
the effect, how sure are we, and how many replicates would we need". This module
answers those questions the same way for all of them.

The replicate hierarchy
    nucleus → colony → well → experiment (independent differentiation)

Colonies in one well are technical replicates; wells of one differentiation share
its batch. ``stats.experiment_column`` names the plate-map column that identifies
differentiations; when every condition has at least
``stats.min_experiments_for_unit`` of them, the experiment is the replicate:
colonies are averaged per well, wells per experiment, and n = experiments. Without
it the replicate is the well or the colony (``radial.choose_unit``), and every
output says which.

Variance components
    Nested method of moments, pooled across conditions: ``sd_colony`` (colonies
    of one well), ``sd_well`` (wells of one experiment, beyond what their colonies
    explain) and ``sd_experiment`` (differentiations, beyond their wells). A level
    with no replication in the data (one well per condition, one experiment) is
    not estimable and is reported as NaN, never as zero.

Effects against a reference condition
    Difference of replicate means, Welch 95 % CI, Hedges' g, and an exact (or
    Monte Carlo) permutation p over replicates. ``reference: auto`` takes, within
    each group of the other condition columns, the lowest level of the trend
    column; otherwise the condition named in ``effects.reference``.

Trends over a numeric condition column (dose, time, diameter)
    Within each group of the other condition columns that has ≥ ``min_levels``
    levels: OLS slope of replicate values against x (log10 x when the positive
    levels span ≥ 10× and ``scale: auto``; a zero level is then left out of the
    fit and the note says so), 95 % CI from a stratified bootstrap of replicates
    within each level, p from permuting the level labels across replicates, and
    Spearman ρ. Columns in ``exponential_columns`` (time courses) also get
    ``y = c + A·exp(−x/τ)``: τ with a bootstrap CI.

Power
    The variance of a condition mean with E experiments, W wells per experiment
    and C colonies per well is σ_E²/E + σ_W²/(E·W) + σ_C²/(E·W·C). For a two-
    condition comparison with E experiments each (df = 2(E − 1)) the minimum
    detectable difference at the requested power is (t₁₋α/₂ + t_power)·√(2 var),
    and the power for a target difference comes from the noncentral t. When σ_E
    cannot be estimated (a single differentiation) it is taken as 0 and the row
    says the MDD is a lower bound; ``power.assumed_experiment_sd`` (in units of
    σ_well, or of σ_colony when σ_well is unknown) adds an explicit assumption.
"""

from __future__ import annotations

import logging
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

logger = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": True,
    "reference": "auto",          # auto | a condition label | null (no effects table)
    "trend_columns": "auto",      # auto = numeric condition columns | [names] | []
    "min_levels": 3,              # levels needed for a trend
    "scale": "auto",              # auto | linear | log
    "exponential_columns": [],    # time-like columns that also get a decay time constant
    "n_boot": 2000,
    "n_perm": 5000,
    "random_state": 0,
    "figure_families": ["pair", "peak", "roughness", "marker"],   # boundary metrics drawn
    "power": {
        "alpha": 0.05,
        "power": 0.8,
        "experiments": [2, 3, 4, 6],
        "wells_per_experiment": None,     # null → the median observed
        "colonies_per_well": None,        # null → the median observed
        "assumed_experiment_sd": None,    # × sd_well (or sd_colony) when not estimable
        "targets": [],                    # [{metric: crossing_um, delta: 20}, {metric: peak_ratio, delta_rel: 0.3}]
    },
}

# Per-colony quantities that are bookkeeping, not answers.
SKIP_METRICS = {"direction", "n_sectors_ok"}


def settings(cfg: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    user = cfg.get("effects") or {}
    for k, v in user.items():
        if v is None:
            continue
        if k == "power":
            out["power"].update({kk: vv for kk, vv in (v or {}).items()})
        else:
            out[k] = v
    if out["scale"] not in ("auto", "linear", "log"):
        raise ValueError("effects.scale must be auto, linear or log")
    return out


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------

def unit_column(unit: str, experiment_column: str | None) -> str | None:
    return {"colony": None, "well": "well_key",
            "experiment": experiment_column}.get(unit)


def unit_values(frame: pd.DataFrame, by: list[str], unit: str,
                experiment_column: str | None = None, value: str = "value") -> pd.DataFrame:
    """One row per replicate: colonies as they are; wells = mean of their colonies;
    experiments = mean of their well means (each well weighs the same).

    Returns *by* + ``unit_id`` + *value* + ``n_colonies``.
    """
    f = frame[np.isfinite(frame[value].to_numpy(np.float64))]
    if unit == "colony" or f.empty:
        out = f[by + [value]].copy()
        out["unit_id"] = f["field_id"].astype(str).to_numpy()
        out["n_colonies"] = 1
        return out
    wells = (f.groupby(by + ["well_key"] + ([experiment_column] if unit == "experiment" else []),
                       observed=True, dropna=False)
             .agg(**{value: (value, "mean"), "n_colonies": (value, "size")}).reset_index())
    if unit == "well":
        return wells.rename(columns={"well_key": "unit_id"})
    exp = (wells.groupby(by + [experiment_column], observed=True, dropna=False)
           .agg(**{value: (value, "mean"), "n_colonies": ("n_colonies", "sum")}).reset_index())
    exp["unit_id"] = exp[experiment_column].astype(str)
    return exp.drop(columns=[experiment_column])


def variance_components(frame: pd.DataFrame, cond: list[str],
                        experiment_column: str | None = None, value: str = "value") -> dict:
    """Nested method-of-moments SDs, pooled over conditions (see module doc)."""
    f = frame[np.isfinite(frame[value].to_numpy(np.float64))]
    out = {"sd_colony": np.nan, "sd_well": np.nan, "sd_experiment": np.nan,
           "n_colonies": int(len(f)), "n_wells": int(f["well_key"].nunique()) if len(f) else 0,
           "n_experiments": (int(f[experiment_column].nunique())
                             if experiment_column and experiment_column in f and len(f) else np.nan)}
    if f.empty:
        return out
    exp = [experiment_column] if experiment_column and experiment_column in f else []
    # Colonies within wells.
    g = f.groupby(cond + exp + ["well_key"], observed=True, dropna=False)[value]
    ss = (g.var(ddof=1) * (g.count() - 1)).sum()
    df = (g.count() - 1).clip(lower=0).sum()
    var_c = ss / df if df > 0 else np.nan
    out["sd_colony"] = float(np.sqrt(var_c)) if np.isfinite(var_c) else np.nan
    wells = g.agg(["mean", "count"]).reset_index()
    # Wells within experiment (within condition).
    gw = wells.groupby(cond + exp, observed=True, dropna=False)
    ss_w = (gw["mean"].var(ddof=1) * (gw["mean"].count() - 1)).sum()
    df_w = (gw["mean"].count() - 1).clip(lower=0).sum()
    var_w = np.nan
    if df_w > 0:
        inv_c = float(np.mean(1.0 / wells["count"]))
        var_w = max(0.0, ss_w / df_w - (var_c if np.isfinite(var_c) else 0.0) * inv_c)
        out["sd_well"] = float(np.sqrt(var_w))
    # Experiments within condition.
    if exp:
        em = gw.agg(mean=("mean", "mean"), n_w=("mean", "count"),
                    inv_c=("count", lambda s: float(np.mean(1.0 / s)))).reset_index()
        ge = em.groupby(cond, observed=True, dropna=False)["mean"]
        ss_e = (ge.var(ddof=1) * (ge.count() - 1)).sum()
        df_e = (ge.count() - 1).clip(lower=0).sum()
        if df_e > 0:
            vw = var_w if np.isfinite(var_w) else 0.0
            vc = var_c if np.isfinite(var_c) else 0.0
            noise = float(np.mean(vw / em["n_w"] + vc * em["inv_c"] / em["n_w"]))
            out["sd_experiment"] = float(np.sqrt(max(0.0, ss_e / df_e - noise)))
    return out


# ---------------------------------------------------------------------------
# Two-condition effects
# ---------------------------------------------------------------------------

def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int, rng) -> tuple[float, bool]:
    """Two-sided p for a difference in means; exact when the labelings are few."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n, k = pooled.size, a.size
    total = comb(n, k)
    if total <= n_perm:
        hits = sum(abs(pooled[list(idx)].mean() - np.delete(pooled, list(idx)).mean())
                   >= obs - 1e-12 for idx in combinations(range(n), k))
        return hits / total, True
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        hits += abs(p[:k].mean() - p[k:].mean()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1), False


def effect(a: np.ndarray, ref: np.ndarray, n_perm: int, rng) -> dict:
    """Difference a − ref with Welch CI, Hedges' g and a permutation p."""
    a, ref = np.asarray(a, float), np.asarray(ref, float)
    out = {"n": a.size, "n_ref": ref.size, "mean": a.mean() if a.size else np.nan,
           "mean_ref": ref.mean() if ref.size else np.nan}
    out["diff"] = out["mean"] - out["mean_ref"]
    out["diff_rel"] = out["diff"] / out["mean_ref"] if out["mean_ref"] else np.nan
    out.update(ci_lo=np.nan, ci_hi=np.nan, hedges_g=np.nan, p_perm=np.nan, exact=False)
    if a.size < 2 or ref.size < 2:
        return out
    va, vr = a.var(ddof=1), ref.var(ddof=1)
    se = np.sqrt(va / a.size + vr / ref.size)
    if se > 0:
        dfw = se ** 4 / ((va / a.size) ** 2 / (a.size - 1) + (vr / ref.size) ** 2 / (ref.size - 1))
        t = sps.t.ppf(0.975, dfw)
        out["ci_lo"], out["ci_hi"] = out["diff"] - t * se, out["diff"] + t * se
    sp = np.sqrt(((a.size - 1) * va + (ref.size - 1) * vr) / (a.size + ref.size - 2))
    if sp > 0:
        j = 1 - 3 / (4 * (a.size + ref.size) - 9)
        out["hedges_g"] = j * out["diff"] / sp
    out["p_perm"], out["exact"] = permutation_p(a, ref, n_perm, rng)
    return out


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

def choose_scale(levels, scale: str) -> str:
    pos = np.asarray([v for v in levels if v > 0], float)
    if scale != "auto":
        return scale
    return "log" if pos.size >= 2 and pos.max() / pos.min() >= 10 else "linear"


def _slope(x, y):
    if np.unique(x).size < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def trend(x: np.ndarray, y: np.ndarray, scale: str, n_boot: int, n_perm: int, rng,
          exponential: bool = False) -> dict:
    """Slope of y on x (or log10 x) with stratified-bootstrap CI and permutation p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    note = ""
    if scale == "log":
        if (x[ok] <= 0).any():
            note = "levels ≤ 0 left out of the log-scale fit"
        ok &= x > 0
    x, y = x[ok], y[ok]
    levels = np.unique(x)
    out = {"scale": scale, "n_levels": int(levels.size), "n_units": int(x.size), "note": note,
           "slope": np.nan, "slope_ci_lo": np.nan, "slope_ci_hi": np.nan, "p_perm": np.nan,
           "spearman_rho": np.nan, "tau": np.nan, "tau_ci_lo": np.nan, "tau_ci_hi": np.nan}
    if levels.size < 2:
        return out
    xt = np.log10(x) if scale == "log" else x
    out["slope"] = _slope(xt, y)
    out["intercept"] = float(np.mean(y) - out["slope"] * np.mean(xt))
    if levels.size > 2 and np.ptp(y) > 0:
        out["spearman_rho"] = float(sps.spearmanr(xt, y).statistic)
    groups = [np.flatnonzero(x == lv) for lv in levels]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        boots.append(_slope(xt[idx], y[idx]))
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    if boots.size > 10:
        out["slope_ci_lo"], out["slope_ci_hi"] = np.percentile(boots, [2.5, 97.5])
    obs = abs(out["slope"])
    hits = sum(abs(_slope(rng.permutation(xt), y)) >= obs - 1e-12 for _ in range(n_perm))
    out["p_perm"] = (hits + 1) / (n_perm + 1)
    if exponential and levels.size >= 4:
        out.update(_exponential(x, y, groups, n_boot, rng))
    return out


def _exp_fit(x, y):
    def f(t, c, a, tau):
        return c + a * np.exp(-t / tau)
    span = float(np.ptp(x)) or 1.0
    p0 = (float(y[x == x.max()].mean()), float(y[x == x.min()].mean() - y[x == x.max()].mean()),
          span / 3)
    try:
        p, _ = optimize.curve_fit(f, x, y, p0=p0, bounds=([-np.inf, -np.inf, span / 100],
                                                          [np.inf, np.inf, span * 100]),
                                  maxfev=5000)
        return float(p[2])
    except (RuntimeError, ValueError):
        return np.nan


def _exponential(x, y, groups, n_boot, rng) -> dict:
    tau = _exp_fit(x, y)
    boots = []
    for _ in range(min(n_boot, 500)):
        idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        boots.append(_exp_fit(x[idx], y[idx]))
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots.size > 10 else (np.nan, np.nan))
    return {"tau": tau, "tau_ci_lo": lo, "tau_ci_hi": hi}


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def group_mean_var(sd_e, sd_w, sd_c, E, W, C) -> float:
    terms = [(sd_e, E), (sd_w, E * W), (sd_c, E * W * C)]
    return float(sum((s ** 2) / n for s, n in terms if np.isfinite(s)))


def mdd(var_mean: float, E: int, alpha: float, power: float) -> float:
    """Minimum detectable difference between two conditions, E replicates each."""
    df = max(2 * (E - 1), 1)
    return float((sps.t.ppf(1 - alpha / 2, df) + sps.t.ppf(power, df)) * np.sqrt(2 * var_mean))


def power_for(delta: float, var_mean: float, E: int, alpha: float) -> float:
    df = max(2 * (E - 1), 1)
    if var_mean <= 0:
        return 1.0
    nc = abs(delta) / np.sqrt(2 * var_mean)
    tc = sps.t.ppf(1 - alpha / 2, df)
    p = 1 - sps.nct.cdf(tc, df, nc) + sps.nct.cdf(-tc, df, nc)
    if not np.isfinite(p):                       # scipy's nct fails far in the tail
        p = sps.norm.sf(tc - nc) + sps.norm.cdf(-tc - nc)
    return float(np.clip(p, 0.0, 1.0))


def power_table(vc: dict, prm: dict, wells: float, colonies: float, ref_mean: float) -> list[dict]:
    """MDD (and power for matching targets) per number of experiments and assumption."""
    pw = prm["power"]
    alpha, target_power = float(pw["alpha"]), float(pw["power"])
    sd_e = vc["sd_experiment"]
    scenarios = []
    if np.isfinite(sd_e):
        scenarios.append(("estimated", sd_e))
    else:
        scenarios.append(("not estimable: taken as 0 (MDD is a lower bound)", 0.0))
        k = pw.get("assumed_experiment_sd")
        if k is not None:
            base = vc["sd_well"] if np.isfinite(vc["sd_well"]) else vc["sd_colony"]
            if np.isfinite(base):
                scenarios.append((f"assumed = {float(k):g} × "
                                  f"{'sd_well' if np.isfinite(vc['sd_well']) else 'sd_colony'}",
                                  float(k) * base))
    W = float(pw.get("wells_per_experiment") or wells or 1)
    C = float(pw.get("colonies_per_well") or colonies or 1)
    rows = []
    for label, se in scenarios:
        for E in pw["experiments"]:
            v = group_mean_var(se, vc["sd_well"], vc["sd_colony"], int(E), W, C)
            row = {"sd_experiment_used": se, "experiment_sd_source": label,
                   "experiments": int(E), "wells_per_experiment": W, "colonies_per_well": C,
                   "mdd": mdd(v, int(E), alpha, target_power)}
            row["mdd_rel"] = row["mdd"] / abs(ref_mean) if ref_mean else np.nan
            rows.append(row)
    return rows


def match_targets(metric_id: str, metric: str, targets: list) -> list[dict]:
    out = []
    for t in targets or []:
        name = str(t.get("metric", ""))
        if name and (name == metric or name in metric_id):
            out.append(t)
    return out


__all__ = ["DEFAULTS", "settings", "unit_values", "unit_column", "variance_components",
           "effect", "permutation_p", "trend", "choose_scale", "group_mean_var", "mdd",
           "power_for", "power_table", "match_targets", "SKIP_METRICS"]
