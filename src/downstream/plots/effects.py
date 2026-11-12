"""
Figures of the effects step: trends, differences from a reference, power.

Each takes the effects tables (plus the per-colony numbers for the trend
points) and returns the path written or ``None``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import style
from ..effects import unit_values
from .base import grid_shape, save_figure

_UNIT_MARK = {"experiment": "D", "well": "s", "colony": "o"}


def _label(row) -> str:
    if row["family"] == "peak" or "(" in str(row["metric"]):
        return str(row["metric"])                 # peaks are intensity by definition
    ro = {"positive": " (gate calls)", "intensity": " (intensity)", "profile": ""}.get(
        row["readout"], f" ({row['readout']})")
    return f"{row['metric']}{ro}"


def _xpos(x: np.ndarray, scale: str):
    """Plot positions; on a log axis a zero level sits one decade below the lowest
    positive level and is labeled 0."""
    x = np.asarray(x, float)
    if scale != "log":
        return x, None
    pos = x[x > 0]
    zero_at = 10 ** (np.floor(np.log10(pos.min())) - 1) if pos.size else 0.1
    return np.where(x > 0, x, zero_at), zero_at


def trend_panels(scalars: pd.DataFrame, trends: pd.DataFrame, unit: str, exp_col,
                 out_path, title: str = ""):
    """One panel per trend row: colonies (small), replicates (large), fitted line."""
    if trends.empty:
        return None
    rows = list(trends.itertuples(index=False))
    nrows, ncols = grid_shape(len(rows), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows + 0.6),
                             squeeze=False)
    groups = sorted(trends["group"].astype(str).unique())
    color = {g: style.color(i) for i, g in enumerate(groups)}
    for ax, r in zip(axes.ravel(), rows):
        part = scalars[(scalars["metric_id"] == r.metric_id)]
        col = r.column
        if r.group != "all":
            for item in str(r.group).split(" | "):
                k, _, v = item.partition("=")
                part = part[part[k].astype(str) == v]
        x_col = pd.to_numeric(part[col], errors="coerce").to_numpy(float)
        xp, zero_at = _xpos(x_col, r.scale)
        c = color[str(r.group)]
        ax.scatter(xp, part["value"], s=8, color=c, alpha=0.35, lw=0, zorder=2)
        reps = unit_values(part.assign(_x=x_col), ["condition", "_x"], r.unit, exp_col)
        rx, _ = _xpos(reps["_x"].to_numpy(float), r.scale)
        if zero_at is not None:
            rx = np.where(reps["_x"].to_numpy(float) > 0, rx, zero_at)
        ax.scatter(rx, reps["value"], s=30, color=c, marker=_UNIT_MARK.get(r.unit, "o"),
                   edgecolors=style.MARKER_EDGE, linewidths=0.5, zorder=3)
        if np.isfinite(r.slope):
            lv = np.unique(x_col[np.isfinite(x_col)])
            fit_x = lv[lv > 0] if r.scale == "log" else lv
            if fit_x.size:
                xs = np.linspace(fit_x.min(), fit_x.max(), 50)
                xt = np.log10(xs) if r.scale == "log" else xs
                ax.plot(xs, r.intercept + r.slope * xt, color=style.OKABE_ITO["black"], lw=1.3,
                        zorder=4)
        if r.scale == "log":
            ax.set_xscale("log")
            levels = np.unique(x_col[np.isfinite(x_col)])
            ticks = [lv for lv in levels if lv > 0]
            if zero_at is not None and (levels <= 0).any():
                ticks = [zero_at] + ticks
            ax.set_xticks(ticks)
            ax.set_xticklabels(["0" if zero_at is not None and np.isclose(t, zero_at)
                                else f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(min(ticks) / 2, max(ticks) * 2)
        per = " per decade" if r.scale == "log" else f" per unit {col}"
        stat = (f"slope {r.slope:.3g}{per}\n95% CI [{r.slope_ci_lo:.3g}, {r.slope_ci_hi:.3g}]"
                f" · p = {r.p_perm:.3g}" if np.isfinite(r.slope) else "no slope")
        if np.isfinite(getattr(r, "tau", np.nan)):
            stat += f"\nτ = {r.tau:.3g} [{r.tau_ci_lo:.3g}, {r.tau_ci_hi:.3g}]"
        flag = " · colonies only" if r.pseudo_replicated else ""
        ax.set_title(f"{_label(r._asdict())}{flag}\n{stat}", fontsize=7.5)
        ax.set_xlabel(col, fontsize=8)
        ax.tick_params(labelsize=7)
        if r.note:
            ax.text(0.02, 0.02, r.note, transform=ax.transAxes, fontsize=6,
                    color=style.ANNOTATION)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    handles = [Line2D([], [], color=style.ANNOTATION, marker="o", ls="", ms=3, alpha=0.5,
                      label="colony"),
               Line2D([], [], color=style.ANNOTATION, marker=_UNIT_MARK.get(unit, "o"), ls="",
                      ms=6, label=f"replicate ({unit}; colonies when a level has < 2)"),
               Line2D([], [], color=style.OKABE_ITO["black"], lw=1.3, label="OLS fit")]
    handles += [Line2D([], [], color=color[g], marker="o", ls="", ms=5, label=g)
                for g in groups if len(groups) > 1]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=7.5,
               frameon=False)
    fig.suptitle(title + "\n95% CI: stratified bootstrap of replicates; p: permutation",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.6 / fig.get_figheight(), 1, 1))
    return save_figure(fig, out_path)


def effect_forest(effects: pd.DataFrame, out_path, title: str = ""):
    """Per metric: each condition's difference from its reference ± 95 % CI."""
    if effects.empty:
        return None
    metrics = list(dict.fromkeys(effects["metric_id"]))
    nrows, ncols = grid_shape(len(metrics), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows + 0.6),
                             squeeze=False)
    for ax, mid in zip(axes.ravel(), metrics):
        e = effects[effects["metric_id"] == mid].reset_index(drop=True)
        y = np.arange(len(e))[::-1]
        for yi, r in zip(y, e.itertuples(index=False)):
            hollow = bool(r.pseudo_replicated)
            if np.isfinite(r.ci_lo):
                ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=style.OKABE_ITO["black"], lw=1.2)
            ax.plot(r.diff, yi, marker=_UNIT_MARK.get(r.unit, "o"), ms=5.5,
                    color=style.OKABE_ITO["black"], mfc="white" if hollow else style.OKABE_ITO["black"])
        ax.axvline(0, color=style.GREY, lw=0.8, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(e["condition"], e["n"])], fontsize=7)
        ax.set_title(f"{_label(e.iloc[0])}\nvs {e['reference'].iloc[0]}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles = [Line2D([], [], color=style.OKABE_ITO["black"], marker="D", ls="-", ms=5,
                      label="difference ± 95% CI (Welch), n = replicates"),
               Line2D([], [], color=style.OKABE_ITO["black"], marker="o", ls="-", ms=5,
                      mfc="white", label="too few replicates: colonies (technical)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.45 / fig.get_figheight(), 1, 1))
    return save_figure(fig, out_path)


def power_curves(power: pd.DataFrame, out_path, title: str = ""):
    """Per metric: minimum detectable difference against the number of experiments."""
    if power.empty:
        return None
    metrics = list(dict.fromkeys(power["metric_id"]))
    nrows, ncols = grid_shape(len(metrics), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows + 0.7),
                             squeeze=False)
    sources = list(dict.fromkeys(power["experiment_sd_source"]))
    styles = {s: ("-" if s.startswith("estimated") or s.startswith("not") else "--")
              for s in sources}
    for ax, mid in zip(axes.ravel(), metrics):
        p = power[power["metric_id"] == mid]
        for i, (src, g) in enumerate(p.groupby("experiment_sd_source", sort=False)):
            ax.plot(g["experiments"], g["mdd"], color=style.color(i), ls=styles[src],
                    marker=style.MARKERS[i], ms=4)
            if "permutation_can_reach_alpha" in g:
                bad = ~g["permutation_can_reach_alpha"].astype(bool)
                if bad.any():                    # hollow: a permutation test cannot reach alpha
                    ax.plot(g.loc[bad, "experiments"], g.loc[bad, "mdd"], ls="none",
                            marker=style.MARKERS[i], ms=7, mfc="white", mec=style.color(i))
        if "target_delta" in p and p["target_delta"].notna().any():
            ax.axhline(float(p["target_delta"].dropna().iloc[0]), color=style.THRESHOLD,
                       lw=1.0, ls=":")
        ax.set_title(_label(p.iloc[0]), fontsize=8)
        ax.set_xlabel("independent experiments per condition", fontsize=7.5)
        ax.set_ylabel("minimum detectable difference", fontsize=7.5)
        ax.set_xticks(sorted(p["experiments"].unique()))
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, None)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles = [Line2D([], [], color=style.color(i), ls=styles[s], marker=style.MARKERS[i],
                      label=f"between-experiment SD: {s}") for i, s in enumerate(sources)]
    if "target_delta" in power and power["target_delta"].notna().any():
        handles.append(Line2D([], [], color=style.THRESHOLD, ls=":", label="target effect"))
    if ("permutation_can_reach_alpha" in power
            and not power["permutation_can_reach_alpha"].astype(bool).all()):
        handles.append(Line2D([], [], color=style.ANNOTATION,
                              ls="none", marker="o", mfc="white",
                              label="hollow: too few experiments for p < alpha "
                                    "with a permutation test"))
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, (0.25 * len(handles) + 0.2) / fig.get_figheight(), 1, 1))
    return save_figure(fig, out_path)


__all__ = ["trend_panels", "effect_forest", "power_curves"]
