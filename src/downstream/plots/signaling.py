"""
Signaling summary figure: one value per colony, by condition.

The radial profiles show *where* a pathway is on; this figure answers the dose
question at a glance: per colony, the median signaling readout over the whole
colony, the edge band and the center, relative to the reference population
(dashed line at 1) when the plate has one.

Every channel the eye can read carries one thing: x position = condition,
color = condition (grey for the reference), marker shape = stain panel. Panels
stained with different antibody mixes can differ in brightness, so showing them
apart is what keeps a panel offset from reading as a dose effect.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import style
from .base import (REFERENCE_MARKERS, no_data, reference_line, save_figure, shorten_labels,
                   value_label)

REGIONS = (("whole", "whole colony"), ("edge", "edge band"), ("center", "center"))


def signaling_summary(table: pd.DataFrame, markers: list[str], order: list[str], out_path,
                      controls=frozenset(), edge_label: str = "edge band",
                      x_label: str = "condition", stain_labels: dict | None = None,
                      center_label: str = "center"):
    """Rows = signaling markers, columns = colony region; one point per colony.

    *stain_labels* maps each ``stain_set`` to a legend text (e.g. its other
    markers); with more than one stain set in *table*, each gets its own marker
    shape and a legend entry.
    """
    if table.empty or not markers:
        return None
    regions = [(k, edge_label if k == "edge" else center_label if k == "center" else lab)
               for k, lab in REGIONS]
    sets = (sorted(table["stain_set"].astype(str).unique())
            if "stain_set" in table else [])
    by_set = len(sets) > 1
    shape = {s: style.MARKERS[i % len(style.MARKERS)] for i, s in enumerate(sets)}
    controls = set(map(str, controls))
    color = {}
    k_color = 0
    for cond in order:
        if cond in controls:
            color[cond] = style.CONTROL
        else:
            color[cond] = style.color(k_color)
            k_color += 1

    fig, axes = plt.subplots(len(markers), len(regions),
                             figsize=(max(3.2, 0.75 * len(order) + 1.2) * len(regions),
                                      3.3 * len(markers) + 0.9),
                             squeeze=False, sharey="row")
    rng = np.random.default_rng(0)
    for i, marker in enumerate(markers):
        part = table[table["marker"] == marker]
        counts = part.groupby(part["condition"].astype(str)).size()
        # Composite labels ("pluripotent | 0") wrap at the separator instead of eliding.
        ticks = [t.replace(" | ", "\n") for t in shorten_labels(order, 28)]
        ticks = [f"{t}\n(ref)" if c in controls else t for c, t in zip(order, ticks)]
        ticks = [f"{t}\nn={int(counts.get(c, 0))}" for c, t in zip(order, ticks)]
        for j, (col, title) in enumerate(regions):
            ax = axes[i, j]
            if part.empty or part[col].notna().sum() == 0:
                no_data(ax)
                continue
            for k, cond in enumerate(order):
                c = part[(part["condition"].astype(str) == cond) & part[col].notna()]
                if c.empty:
                    continue
                groups = c.groupby(c["stain_set"].astype(str)) if by_set else [(None, c)]
                for stain, g in groups:
                    v = g[col].to_numpy(np.float64)
                    ax.scatter(k + rng.uniform(-0.2, 0.2, v.size), v, s=18,
                               color=color[cond],
                               marker=shape.get(stain, style.MARKERS[0]),
                               edgecolors=style.MARKER_EDGE, linewidths=0.35, zorder=3)
                ax.plot([k - 0.32, k + 0.32], [np.median(c[col])] * 2,
                        color=style.ANNOTATION, lw=1.8, zorder=4)
            reference_line(ax, marker)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(ticks, fontsize=7.5)
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xlabel(x_label, fontsize=8.5)
            if i == 0:
                ax.set_title(title, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(value_label(marker, f"{marker} (shared scale)"), fontsize=8.5)

    handles = [Line2D([], [], color=style.ANNOTATION, lw=1.8,
                      label="median of colonies (all panels)")]
    if any(m in REFERENCE_MARKERS for m in markers):
        handles.append(Line2D([], [], color=style.REFERENCE_LINE, ls="--", lw=0.9,
                              label="reference level (= 1)"))
    if controls & set(order):
        handles.append(Line2D([], [], color=style.CONTROL, marker="o", ls="",
                              markeredgecolor=style.MARKER_EDGE, label="reference wells"))
    if by_set:
        labels = stain_labels or {}
        for s in sets:
            name = s if s.lower().startswith(("panel", "stain")) else f"panel {s}"
            text = name + (f": {labels[s]}" if labels.get(s) else "")
            handles.append(Line2D([], [], color="white", marker=shape[s], ls="",
                                  markeredgecolor=style.ANNOTATION, markersize=6.5,
                                  label=text))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Signaling readout per colony (one point = one colony)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06 + 0.03 * ((len(handles) - 1) // 3), 1, 1))
    return save_figure(fig, out_path)


__all__ = ["signaling_summary"]
