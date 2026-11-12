"""
Per-marker distribution figures.

* :func:`violin_by_condition`: the workhorse. Cell-level violin per condition
  with the **field medians overplotted as points**, so cell spread and
  field-to-field spread are visible in one panel (items 1 and 6), plus the
  positivity gate as a horizontal line.
* :func:`ecdf_by_condition`: cumulative distributions. Scale-free and free of
  bandwidth choices, which makes it the honest view for judging whether a gate
  sits where the populations actually separate.
* :func:`threshold_qc`: the trust gate for everything downstream: the pooled
  distribution, the negative control on top of it, and the chosen cut with the
  method that produced it written on the panel.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import style
from ..data import DAPI_NORM
from .base import (annotate_n, cap_for_display, condition_palette, control_conditions,
                   display_limit, grid_shape, no_data, ordered_conditions, save_figure,
                   shorten, shorten_labels)

logger = logging.getLogger(__name__)


def _axis_label(marker: str, space: str) -> str:
    return f"{marker} intensity " + ("(DAPI-normalized)" if space == DAPI_NORM else "(a.u.)")


def violin_by_condition(gated: pd.DataFrame, fields: pd.DataFrame, marker: str,
                        space: str, threshold: float | None, cfg: dict, out_path,
                        controls=None, facet: str = ""):
    """Cell-level violins + field-level medians per condition.

    *controls* names the conditions to gray out. Pass this marker's own controls
    when they are marker-specific; ``None`` falls back to the class column, which
    is the marker-agnostic answer.
    """
    vis = cfg.get("visualization") or {}
    cap_pct = float(vis.get("cap_percentile", 99.9))
    controls = control_conditions_from(fields, cfg) if controls is None else set(controls)

    sub = gated[gated["marker"] == marker]
    if sub.empty:
        return None
    order = ordered_conditions(sub["condition"].unique(), cfg, controls)

    # Deliberately NOT one color per condition: with a dozen-plus conditions a
    # qualitative palette recycles, and repeated colors imply a grouping that
    # does not exist. The x-axis already names each condition, so color is used
    # only to separate controls (gray) from test conditions (accent).
    control_set = set(map(str, controls))

    arrays, kept, counts, colors = [], [], [], []
    for cond in order:
        vals = cap_for_display(sub.loc[sub["condition"] == cond, "value"].to_numpy(), cap_pct)
        if vals.size < 3:
            continue
        arrays.append(vals)
        kept.append(cond)
        colors.append(style.CONTROL if str(cond) in control_set else style.POSITIVE)
        counts.append(int(fields.loc[(fields["marker"] == marker)
                                     & (fields["condition"] == cond), "field_id"].nunique()))
    if not arrays:
        return None
    labels = shorten_labels(kept)

    width = max(6.0, 0.62 * len(arrays) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 4.8))
    parts = ax.violinplot(arrays, showmedians=True, showextrema=False, widths=0.85)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("none")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.2)

    # Field medians: the actual replicates, jittered over their violin.
    rng = np.random.default_rng(0)
    for i, cond in enumerate(kept, start=1):
        fvals = fields.loc[(fields["marker"] == marker) & (fields["condition"] == cond),
                           "median"].to_numpy(dtype=float)
        fvals = fvals[np.isfinite(fvals)]
        if fvals.size:
            jitter = rng.uniform(-0.13, 0.13, size=fvals.size)
            ax.scatter(np.full(fvals.size, i) + jitter, fvals, s=9, color="black",
                       alpha=0.8, zorder=3, linewidths=0)

    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color=style.THRESHOLD, ls="--", lw=1.1,
                   label="positivity threshold")
        ax.legend(loc="upper right")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(_axis_label(marker, space))
    # pad clears the per-condition field counts written just above the axes.
    ax.set_title(f"{marker}: per-cell distribution with field medians (black)"
                 + (f"\n{facet}" if facet else ""), pad=16)
    annotate_n(ax, counts)
    return save_figure(fig, out_path)


def ecdf_by_condition(gated: pd.DataFrame, marker: str, space: str,
                      threshold: float | None, cfg: dict, out_path, facet: str = ""):
    """Empirical CDF per condition: bandwidth-free, gate-honest."""
    vis = cfg.get("visualization") or {}
    cap_pct = float(vis.get("cap_percentile", 99.9))
    sub = gated[gated["marker"] == marker]
    if sub.empty:
        return None
    controls = set(sub.loc[sub.get("is_control", False) == True, "condition"].unique()) \
        if "is_control" in sub.columns else set()
    order = ordered_conditions(sub["condition"].unique(), cfg, controls)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = shorten_labels(order, 26)
    for i, cond in enumerate(order):
        vals = cap_for_display(sub.loc[sub["condition"] == cond, "value"].to_numpy(), cap_pct)
        if vals.size < 5:
            continue
        vals = np.sort(vals)
        y = np.arange(1, vals.size + 1) / vals.size
        is_ctrl = cond in controls
        ax.plot(vals, y, lw=2.0 if is_ctrl else 1.3,
                color=style.CONTROL if is_ctrl else style.color(i),
                ls="--" if is_ctrl else style.LINE_STYLES[i % len(style.LINE_STYLES)],
                label=labels[i])
    if threshold is not None and np.isfinite(threshold):
        # Haloed and named in the legend: the gate shares its color with one of
        # the series lines, so it has to be distinguishable by more than hue.
        ax.axvline(threshold, color=style.THRESHOLD, ls="--", lw=1.8, zorder=5,
                   path_effects=style.halo(3.2), label="positivity gate")
    ax.set_xlabel(_axis_label(marker, space))
    ax.set_ylabel("Cumulative fraction of cells")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{marker}: cumulative distribution by condition"
                 + (f" ({facet})" if facet else ""))
    ncol = 2 if len(order) > 8 else 1
    ax.legend(ncol=ncol, bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_figure(fig, out_path)


def threshold_qc(gated: pd.DataFrame, thresholds: pd.DataFrame, cells,
                 controls, cfg: dict, out_path):
    """One panel per (scope × marker): pooled histogram, control, and the cut.

    *controls* is a :class:`downstream.controls.Controls` (a plain boolean array
    still works). Using it per marker matters here more than anywhere: this figure
    is how you check that a gate sits where it should, and drawing the wrong
    population as "control" would make a misplaced gate look correct.
    """
    from ..controls import Controls
    from ..scope import POOLED

    vis = cfg.get("visualization") or {}
    cap_pct = float(vis.get("cap_percentile", 99.9))
    rows = thresholds.reset_index(drop=True)
    if rows.empty:
        return None

    n = len(rows)
    nrows, ncols = grid_shape(n, max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows), squeeze=False)
    uid = gated["cell_uid"].to_numpy()

    for ax, (_, rec) in zip(axes.ravel(), rows.iterrows()):
        marker = rec["marker"]
        sel = (gated["marker"] == marker).to_numpy()
        # '*' marks a column the marker was pooled over, not a filter.
        for col in ("dataset", "plate_id", "stain_set", "seg_mode"):
            if col in rec.index and col in gated.columns and str(rec[col]) != POOLED:
                sel &= (gated[col].astype(str) == str(rec[col])).to_numpy()
        mask = (controls.mask_for(marker) if isinstance(controls, Controls)
                else np.asarray(controls, dtype=bool))
        ctrl_flag = mask[uid]
        vals = gated.loc[sel, "value"].to_numpy(dtype=float)
        ctrl_vals = gated.loc[sel & ctrl_flag, "value"].to_numpy(dtype=float)
        vals = cap_for_display(vals, cap_pct)
        if vals.size == 0:
            no_data(ax, f"{rec['marker']}: no data")
            continue

        # The x axis stops at the 99.5th percentile (or just past the gate): a
        # long, sparse bright tail would otherwise squeeze the populations that
        # decide the gate into a sliver at the left.
        hi = float(np.nanpercentile(vals, 99.5))
        if np.isfinite(rec["threshold"]):
            hi = max(hi, 1.15 * float(rec["threshold"]))
        hi = min(hi, float(np.nanmax(vals)))
        bins = np.linspace(float(np.nanmin(vals)), hi, 80)
        ax.hist(vals[vals <= hi], bins=bins, color=style.OKABE_ITO["sky"], alpha=0.75,
                label=f"all wells (n={vals.size})", density=True)
        ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]
        if ctrl_vals.size:
            ax.hist(ctrl_vals[ctrl_vals <= bins[-1]], bins=bins, histtype="step",
                    color=style.CONTROL, lw=1.6, density=True,
                    label=f"control (n={ctrl_vals.size})")
        if np.isfinite(rec["threshold"]):
            # The provenance of the cut ('negative-control P99', 'GMM', …) names
            # the LINE. It is not what the x-axis measures, which is intensity.
            note = str(rec["note"] or rec.get("method", "threshold"))[:40]
            ax.axvline(rec["threshold"], color=style.THRESHOLD, ls="--", lw=1.4,
                       label=f"gate = {rec['threshold']:.3g}  ({note})")

        verdict = "" if rec["verdict"] == "ok" else "  ✗ NO GATE"
        scope = " / ".join(str(rec[c]) for c in ("plate_id", "stain_set", "seg_mode")
                           if c in rec.index and str(rec[c]) != POOLED)
        n_panels = rec.get("n_panels_pooled", 1)
        if np.isfinite(n_panels or np.nan) and int(n_panels or 1) > 1:
            scope += f" ({int(n_panels)} panels pooled)"
        ax.set_title(f"{rec['marker']}: {scope}{verdict}", fontsize=9)
        space = str(rec["space"]) if "space" in rec.index else ""
        ax.set_xlabel(_axis_label(str(rec["marker"]), space), fontsize=7.5)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=6.5)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Positivity thresholds: estimation evidence (x axis cut at the 99.5th "
                 "percentile)", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, out_path)


def control_conditions_from(fields: pd.DataFrame, cfg: dict) -> set[str]:
    """Control condition names, read off the field table when it carries the flag."""
    cond_cfg = cfg.get("conditions") or {}
    column = cond_cfg.get("control_column", "class")
    values = {str(v).lower() for v in (cond_cfg.get("control_values") or ["control"])}
    if column not in fields.columns:
        return set()
    mask = fields[column].astype(str).str.lower().isin(values)
    return set(fields.loc[mask, "condition"].astype(str).unique())


__all__ = ["violin_by_condition", "ecdf_by_condition", "threshold_qc",
           "control_conditions_from"]
