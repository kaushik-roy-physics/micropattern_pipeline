"""
Boundary and peak metrics by condition: one point per colony.

Same reading as the signaling summary: x position and color = condition (grey
for controls), marker shape = stain panel. The black diamond with its bar is the
condition mean ± CI across replicates (colonies, or wells when the well is the
unit), which is the number to quote; individual colony points show how much of
that spread is colony-to-colony.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats as sps

from .. import style
from .base import grid_shape, no_data, save_figure, shorten_labels


def _select(table: pd.DataFrame, kind, subject, metric, readout) -> pd.DataFrame:
    part = table[(table["kind"] == kind) & (table["subject"].astype(str) == subject)
                 & (table["metric"] == metric)]
    if "ok" in part:
        part = part[part["ok"].astype(bool)]
    if readout is None and part["readout"].nunique() > 1:
        readout = "positive" if (part["readout"] == "positive").any() else "intensity"
    if readout is not None:
        part = part[part["readout"] == readout]
    return part


def _unit_ci(part: pd.DataFrame, unit: str, ci: float = 95.0, experiment_col=None):
    """``(mean, half-width, n, level)`` at the replicate level: experiments (means
    of their well means) or wells when the condition has two or more of them;
    otherwise the next level down, ending at colonies (technical replicates)."""
    level = "colony"
    v = part["value"]
    if (unit == "experiment" and experiment_col in part
            and part[experiment_col].nunique() >= 2):
        wells = part.groupby([experiment_col, "well_key"])["value"].mean()
        v, level = wells.groupby(level=0).mean(), "experiment"
    elif unit in ("well", "experiment") and "well_key" in part and part["well_key"].nunique() >= 2:
        v, level = part.groupby("well_key")["value"].mean(), "well"
    v = v.to_numpy(np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0, level
    if v.size == 1:
        return float(v[0]), np.nan, 1, level
    half = sps.t.ppf(0.5 + ci / 200.0, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(half), int(v.size), level


def boundary_metrics(table: pd.DataFrame, panels: list[tuple], order: list[str], unit: str,
                     out_path, title: str = "", x_label: str = "condition",
                     controls=frozenset(), stain_labels: dict | None = None,
                     experiment_col: str | None = None):
    """One panel per metric; returns the path written or ``None``."""
    panels = [p for p in panels if not _select(table, p[0], p[1], p[2], p[3]).empty]
    if table.empty or not panels:
        return None
    present = set().union(*(set(_select(table, *p[:4])["condition"].astype(str))
                            for p in panels))          # no all-empty columns
    order = [c for c in order if c in present]
    controls = set(map(str, controls))
    color, k = {}, 0
    for cond in order:
        if cond in controls:
            color[cond] = style.CONTROL
        else:
            color[cond], k = style.color(k), k + 1
    sets = sorted(table["stain_set"].astype(str).unique()) if "stain_set" in table else []
    shape = {s: style.MARKERS[i % len(style.MARKERS)] for i, s in enumerate(sets)}
    by_set = len(sets) > 1

    nrows, ncols = grid_shape(len(panels), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(3.0, 0.7 * len(order) + 1.3) * ncols,
                                                    3.0 * nrows + 0.9), squeeze=False)
    rng = np.random.default_rng(0)
    fell_back = False
    ticks = [t.replace(" | ", "\n") for t in shorten_labels(order, 28)]
    for ax, (kind, subject, metric, readout, label, overlay) in zip(axes.ravel(), panels):
        part = _select(table, kind, subject, metric, readout)
        if part.empty:
            no_data(ax)
            continue
        ns = []
        for i, cond in enumerate(order):
            c = part[part["condition"].astype(str) == cond]
            if c.empty:
                ns.append("0")
                continue
            groups = c.groupby(c["stain_set"].astype(str)) if by_set else [(None, c)]
            for stain, g in groups:
                ax.scatter(i - 0.12 + rng.uniform(-0.12, 0.12, len(g)), g["value"], s=16,
                           color=color[cond], marker=shape.get(stain, "o"),
                           edgecolors=style.MARKER_EDGE, linewidths=0.35, zorder=3)
            if overlay:
                o = _select(table, kind, subject, overlay, readout)
                o = o[o["condition"].astype(str) == cond]
                ax.scatter(i - 0.12 + rng.uniform(-0.12, 0.12, len(o)), o["value"], s=16,
                           facecolors="none", edgecolors=style.ANNOTATION, linewidths=0.7,
                           marker="o", zorder=2)
            mean, half, n, level = _unit_ci(c, unit, experiment_col=experiment_col)
            hollow = level != unit
            fell_back |= hollow
            ns.append(f"{n}" + ("" if not hollow else level[0]))
            ax.errorbar(i + 0.22, mean, yerr=None if not np.isfinite(half) else half,
                        fmt="D", color=style.OKABE_ITO["black"], ms=4.5, capsize=3, lw=1.2,
                        mfc="white" if hollow else style.OKABE_ITO["black"], zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{t}\nn={n}" for t, n in zip(ticks, ns)], fontsize=7)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel(x_label, fontsize=8)
        if metric.startswith("offset") or metric == "overlap_um":
            ax.axhline(0, color=style.GREY, lw=0.8, zorder=0)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    handles = [Line2D([], [], color=style.OKABE_ITO["black"], marker="D", ls="-", ms=4.5,
                      label=f"mean ± 95% CI across {_plural(unit)} (n = {_plural(unit)})")]
    if fell_back:
        handles.append(Line2D([], [], color=style.OKABE_ITO["black"], marker="D", ls="-",
                              ms=4.5, mfc="white",
                              label=f"too few {_plural(unit)}: next level down (n = …w wells, "
                                    f"…c colonies)"))
    handles += [
               Line2D([], [], color=style.ANNOTATION, marker="o", ls="", ms=5,
                      label="one colony (accepted fits only)")]
    if by_set:
        labels = stain_labels or {}
        for s in sets:
            name = s if s.lower().startswith(("panel", "stain")) else f"panel {s}"
            handles.append(Line2D([], [], color="white", marker=shape[s], ls="",
                                  markeredgecolor=style.ANNOTATION, markersize=6,
                                  label=name + (f": {labels[s]}" if labels.get(s) else "")))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3), fontsize=7.5,
               frameon=False)
    fig.suptitle(title, fontsize=10.5)
    rows = -(-len(handles) // 3)
    fig.tight_layout(rect=(0, min(0.3, (0.25 * rows + 0.1) / fig.get_figheight()), 1, 1))
    return save_figure(fig, out_path)




# ---------------------------------------------------------------------------
# Registered vs angle-averaged profiles
# ---------------------------------------------------------------------------

def registered_profiles(reg: pd.DataFrame, marker: str, order: list[str], out_path,
                        metrics: pd.DataFrame | None = None, controls=frozenset()):
    """Per condition: the profile against distance from the border, averaged over
    colonies, angle-averaged (dashed) and registered to each sector's own border
    (solid). Top row: fraction positive; bottom: intensity."""
    if reg.empty:
        return None
    order = [c for c in order if c in set(reg["condition"].astype(str))]
    rows = [("positive_fraction", "fraction positive"), ("mean", "intensity (shared scale)")]
    fig, axes = plt.subplots(2, len(order), figsize=(2.9 * len(order) + 0.6, 5.6),
                             squeeze=False, sharex=True, sharey="row")
    frames = (("unregistered", "angle-averaged", (0, (4, 2)), style.ANNOTATION),
              ("registered", "registered to local border", "-", style.OKABE_ITO["blue"]))
    for j, cond in enumerate(order):
        part = reg[reg["condition"].astype(str) == cond]
        n_col = part["field_id"].nunique()
        for i, (col, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for frame, label, ls, color in frames:
                f = part[part["frame"] == frame]
                g = f.groupby("x")[col]
                m, sd, n = g.mean(), g.std(), g.count()
                with np.errstate(invalid="ignore", divide="ignore"):
                    half = sps.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n)
                half = half.where(n > 1)
                ax.fill_between(m.index, m - half, m + half, color=color, alpha=0.15, lw=0)
                ax.plot(m.index, m.to_numpy(), color=color, ls=ls, lw=1.6, label=label)
            ax.axvline(0, color=style.GREY, lw=0.8, zorder=0)
            if i == 0:
                ax.set_ylim(-0.02, 1.02)
                head = f"{cond}\n{n_col} colonies"
                if metrics is not None and len(metrics):
                    mm = metrics[(metrics["condition"].astype(str) == cond)
                                 & metrics["ok"].astype(bool)]
                    w = {k: mm.loc[mm["metric"] == k, "value"].mean()
                         for k in ("width_unregistered_um", "width_registered_um",
                                   "roughness_um")}
                    if np.isfinite(w["width_registered_um"]):
                        head += (f"\nwidth {w['width_unregistered_um']:.0f} → "
                                 f"{w['width_registered_um']:.0f} µm · rough "
                                 f"{w['roughness_um']:.0f} µm")
                ax.set_title(head, fontsize=8, color=style.CONTROL if cond in controls
                             else "black")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel(f"distance from the {marker} border (µm)\n← edge · center →",
                              fontsize=8)
            ax.tick_params(labelsize=7.5)
    axes[0, 0].legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(f"{marker}: is the transition graded, or a sharp border that wanders with "
                 f"angle?\nmean over colonies ± 95% CI; per-colony widths are means of "
                 f"the logistic fits", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Where the border is, around one colony
# ---------------------------------------------------------------------------

def border_map(cmap: dict, title: str, out_path):
    """Nuclei (positive / negative for each boundary marker) with the fitted border
    of every angular sector (± SE) and the mean border circle: the QC picture
    behind the roughness numbers."""
    markers = list(cmap["markers"])
    if not markers:
        return None
    fig, axes = plt.subplots(1, len(markers), figsize=(4.4 * len(markers), 4.6),
                             squeeze=False)
    R = float(cmap["radius_um"])
    for ax, m in zip(axes[0], markers):
        info = cmap["markers"][m]
        known, pos = info["known"], info["positive"]
        x, y = cmap["x_um"], cmap["y_um"]
        ax.scatter(x[known & ~pos], y[known & ~pos], s=1.6, color=style.GREY, lw=0,
                   rasterized=True, label=f"{m} −")
        ax.scatter(x[known & pos], y[known & pos], s=2.2, color=style.OKABE_ITO["blue"], lw=0,
                   rasterized=True, label=f"{m} +")
        sec = info["sectors"]
        sec = sec[np.isfinite(sec["x50_um"])]
        rad = R - sec["x50_um"].to_numpy()
        th = sec["theta"].to_numpy()
        se = sec["se_um"].to_numpy()
        for t, rr, e in zip(th, rad, se):
            ax.plot([(rr - e) * np.cos(t), (rr + e) * np.cos(t)],
                    [(rr - e) * np.sin(t), (rr + e) * np.sin(t)],
                    color=style.OKABE_ITO["vermillion"], lw=1.4, zorder=4)
        ax.scatter(rad * np.cos(th), rad * np.sin(th), s=22, color=style.OKABE_ITO["vermillion"],
                   edgecolors="white", linewidths=0.6, zorder=5, label="border per sector ± SE")
        met = info["metrics"]
        if np.isfinite(met.get("mean_border_um", np.nan)):
            ax.add_patch(plt.Circle((0, 0), R - met["mean_border_um"], fill=False,
                                    color=style.OKABE_ITO["black"], ls="--", lw=0.9))
        ax.add_patch(plt.Circle((0, 0), R, fill=False, color=style.ANNOTATION, lw=0.7))
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("µm", fontsize=8)
        ax.tick_params(labelsize=7)
        sub = (f"roughness {met.get('roughness_um', np.nan):.0f} µm · decenter "
               f"{met.get('decenter_um', np.nan):.0f} µm\nwidth "
               f"{met.get('width_unregistered_um', np.nan):.0f} → "
               f"{met.get('width_registered_um', np.nan):.0f} µm registered")
        ax.set_title(f"{m}\n{sub}", fontsize=8.5)
        ax.legend(handles=[
            Line2D([], [], color=style.GREY, marker="o", ls="", ms=3, label=f"{m} −"),
            Line2D([], [], color=style.OKABE_ITO["blue"], marker="o", ls="", ms=3,
                   label=f"{m} +"),
            Line2D([], [], color=style.OKABE_ITO["vermillion"], marker="o", ls="-", ms=4,
                   label="border per sector ± SE")], fontsize=6.5, loc="upper right",
            frameon=False)
    fig.suptitle(f"{title}\ndashed circle = mean border", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, out_path)


def bin_check(check: pd.DataFrame, colonies: pd.DataFrame, out_path):
    """Per position/width metric: each colony's shift under other binnings.

    One panel per metric (accepted default values only); one column of points per
    scheme, the shift alt − default in µm. The grey band is ± the median bootstrap
    SE of the default fit: points inside it moved less than the measurement's own
    uncertainty, i.e. the binning does not shape the result.
    """
    keys = ["field_id", "kind", "subject", "readout", "metric"]
    base = colonies[colonies["ok"].astype(bool)][keys + ["value", "se"]]
    m = check.merge(base, on=keys, suffixes=("", "_default"))
    if m.empty:
        return None
    m["shift"] = m["value"] - m["value_default"]
    panels = list(m[keys[1:]].drop_duplicates().itertuples(index=False, name=None))
    schemes = list(dict.fromkeys(check["scheme"]))
    nrows, ncols = grid_shape(len(panels), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows + 0.5),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (kind, subject, readout, metric) in zip(axes.ravel(), panels):
        part = m[(m["kind"] == kind) & (m["subject"] == subject) & (m["readout"] == readout)
                 & (m["metric"] == metric)]
        se = float(np.nanmedian(part["se"])) if part["se"].notna().any() else np.nan
        if np.isfinite(se):
            ax.axhspan(-se, se, color=style.GREY, alpha=0.25, lw=0, zorder=0)
        ax.axhline(0, color=style.ANNOTATION, lw=0.8, zorder=1)
        for i, sch in enumerate(schemes):
            y = part.loc[part["scheme"] == sch, "shift"].to_numpy(np.float64)
            y = y[np.isfinite(y)]
            if not y.size:
                continue
            ax.scatter(i + rng.uniform(-0.15, 0.15, y.size), y, s=10, color=style.color(i),
                       marker=style.MARKERS[i % len(style.MARKERS)], alpha=0.8, lw=0, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color=style.OKABE_ITO["black"],
                    lw=1.6, zorder=4)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels([s.replace(":", " ") + (" µm" if s.startswith("fixed") else " bins")
                            for s in schemes], fontsize=7)
        ro = "" if kind == "peak" else f" ({'gate calls' if readout == 'positive' else readout})"
        ax.set_title(f"{subject}{ro}\n{metric}", fontsize=8)
        ax.set_ylabel("shift vs default (µm)", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Does the binning shape the result? Each point: one colony's value under "
                 "other bins minus its default fit.\nBar = median; grey band = ± median "
                 "bootstrap SE of the default (inside it, the shift is within measurement "
                 "noise)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    return save_figure(fig, out_path)


__all__ = ["boundary_metrics", "registered_profiles", "border_map", "bin_check"]


def _plural(unit: str) -> str:
    """colony → colonies, well → wells, experiment → experiments."""
    unit = str(unit)
    return unit[:-1] + "ies" if unit.endswith("y") else unit + "s"
