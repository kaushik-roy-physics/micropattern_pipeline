"""
Plate-layout and field-uniformity figures.

:func:`plate_heatmap` lays a well-level statistic out in the plate's real
geometry. Nothing else catches edge evaporation, a pipetting slip or a bad
column so quickly, because those artifacts are *spatial* and invisible in any
condition-ordered plot.

:func:`field_uniformity` answers item 6 directly: every field of every well as a
point, so a well whose mean hides a bimodal set of fields is obvious. The
per-well coefficient of variation is printed alongside.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import style
from .base import grid_shape, no_data, save_figure, shorten, shorten_labels

logger = logging.getLogger(__name__)


def plate_heatmap(wells: pd.DataFrame, value_col: str, marker: str, cfg: dict,
                  out_path, title: str | None = None, percent: bool = False):
    """Well-level *value_col* drawn in plate geometry, one panel per plate."""
    sub = wells[wells["marker"] == marker]
    sub = sub[np.isfinite(sub[value_col].to_numpy(dtype=float))]
    if sub.empty or "row" not in sub.columns or "col" not in sub.columns:
        return None

    plates = list(dict.fromkeys(sub["plate_id"].astype(str))) if "plate_id" in sub else ["plate"]
    nrows, ncols = grid_shape(len(plates), max_cols=2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.4 * nrows), squeeze=False)

    scale = 100.0 if percent else 1.0
    vals_all = scale * sub[value_col].to_numpy(dtype=float)
    vmin, vmax = float(np.nanpercentile(vals_all, 2)), float(np.nanpercentile(vals_all, 98))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    for ax, plate_id in zip(axes.ravel(), plates):
        part = sub[sub["plate_id"].astype(str) == plate_id] if "plate_id" in sub else sub
        n_rows = int(part["row"].max())
        n_cols = int(part["col"].max())
        grid = np.full((n_rows, n_cols), np.nan)
        grid[part["row"].astype(int) - 1, part["col"].astype(int) - 1] = \
            scale * part[value_col].to_numpy(dtype=float)

        im = ax.imshow(grid, cmap=style.CMAP_SEQUENTIAL, vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(n_cols), [str(c + 1) for c in range(n_cols)], fontsize=7)
        ax.set_yticks(range(n_rows), [chr(ord("A") + r) for r in range(n_rows)], fontsize=7)
        # Annotate only when the plate is small enough for the text to fit.
        if n_rows * n_cols <= 96:
            for r in range(n_rows):
                for c in range(n_cols):
                    if np.isfinite(grid[r, c]):
                        ax.text(c, r, f"{grid[r, c]:.0f}", ha="center", va="center",
                                fontsize=5.5,
                                color=style.text_on_cmap(grid[r, c], vmin, vmax,
                                                         style.CMAP_SEQUENTIAL))
        ax.set_title(f"{plate_id}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85,
                     label=f"{marker} {value_col}" + (" (%)" if percent else ""))

    for ax in axes.ravel()[len(plates):]:
        ax.axis("off")
    fig.suptitle(title or f"{marker}: {value_col} by plate position", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, out_path)


def field_uniformity(fields: pd.DataFrame, marker: str, value_col: str, cfg: dict,
                     out_path, controls=()):
    """Every field of every well as a point, ordered by well: the item-6 view."""
    from .base import ordered_conditions
    sub = fields[fields["marker"] == marker].copy()
    sub = sub[np.isfinite(sub[value_col].to_numpy(dtype=float))]
    if sub.empty:
        return None

    order = ordered_conditions(sub["condition"].unique(), cfg, controls)
    rank = {c: i for i, c in enumerate(order)}
    sub["_rank"] = sub["condition"].astype(str).map(rank)
    sub = sub.sort_values(["_rank", "well"])
    wells = list(dict.fromkeys(zip(sub["condition"].astype(str), sub["well"])))

    fig, ax = plt.subplots(figsize=(max(6.5, 0.34 * len(wells) + 2.2), 4.4))
    rng = np.random.default_rng(0)
    cvs = []
    for i, (cond, well) in enumerate(wells):
        vals = sub.loc[(sub["condition"].astype(str) == cond) & (sub["well"] == well),
                       value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        color = style.CONTROL if cond in set(map(str, controls)) else style.color(rank.get(cond, 0))
        # A dark outline is what makes the pale end of the palette usable: an
        # unoutlined Okabe-Ito yellow point is invisible on a white page.
        ax.scatter(np.full(vals.size, i) + rng.uniform(-0.14, 0.14, vals.size), vals,
                   s=16, color=color, alpha=0.9, linewidths=0.4,
                   edgecolors=style.MARKER_EDGE, zorder=2)
        # Haloed, so the median never merges with a point of the same color.
        ax.hlines(np.median(vals), i - 0.3, i + 0.3, color=style.MARKER_EDGE, lw=1.6,
                  zorder=3, path_effects=style.halo(3.0))
        mean = float(np.mean(vals))
        cvs.append(float(np.std(vals, ddof=1) / mean) if vals.size > 1 and mean else np.nan)

    cond_labels = shorten_labels([c for c, _ in wells], 16)
    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels([f"{lab}\nw{w}" for lab, (_, w) in zip(cond_labels, wells)],
                       rotation=90, fontsize=6)
    ax.set_ylabel(f"{marker} {value_col} (per field)")
    median_cv = float(np.nanmedian(cvs)) if cvs else np.nan
    ax.set_title(f"{marker}: field-to-field spread within each well "
                 f"(median CV = {median_cv:.2f})")
    return save_figure(fig, out_path)


def count_qc(qc: pd.DataFrame, cfg: dict, out_path):
    """Cell count and DAPI level per field, grouped by well, flagged fields marked.

    Plotted against well rather than against the raw assembly index: a flagged
    field you cannot attribute to a condition is a number you cannot act on. The
    well grouping also makes acquisition-order drift (bleaching, focus, settling)
    legible as a trend across wells instead of a diagonal smear.
    """
    if qc.empty:
        return None
    frame = qc.copy()
    if "well" in frame.columns:
        frame = frame.sort_values(["well", "field_id"], kind="stable")
    wells = list(dict.fromkeys(frame["well"])) if "well" in frame.columns else []

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    flagged = frame["flag_outlier"].to_numpy(dtype=bool)
    x = np.arange(len(frame))

    for ax, col, label in ((axes[0], "n_cells", "nuclei per image"),
                           (axes[1], "dapi_median", "median DAPI per image")):
        vals = frame[col].to_numpy(dtype=float)
        # Alternating bands delimit wells without adding a color meaning.
        if wells:
            starts = [int(np.argmax(frame["well"].to_numpy() == w)) for w in wells]
            bounds = starts + [len(frame)]
            for k in range(0, len(wells), 2):
                ax.axvspan(bounds[k] - 0.5, bounds[k + 1] - 0.5,
                           color=style.NEUTRAL_FILL, alpha=0.45, lw=0, zorder=0)
        ax.scatter(x[~flagged], vals[~flagged], s=14, color=style.OKABE_ITO["blue"],
                   linewidths=0.3, edgecolors=style.MARKER_EDGE, label="ok", zorder=2)
        if flagged.any():
            ax.scatter(x[flagged], vals[flagged], s=34, color=style.THRESHOLD,
                       marker="X", linewidths=0.5, edgecolors=style.MARKER_EDGE,
                       label="flagged", zorder=3)
        if wells:
            centers = [(bounds[k] + bounds[k + 1] - 1) / 2 for k in range(len(wells))]
            ax.set_xticks(centers)
            ax.set_xticklabels([f"w{w}" for w in wells], fontsize=6.5, rotation=90)
            ax.set_xlabel("well (colony images grouped, acquisition order)")
        else:
            ax.set_xlabel("field (assembly order)")
        ax.set_ylabel(label)
        ax.legend()

    axes[0].set_title(f"Segmentation yield: {int(qc['flag_outlier'].sum())} of "
                      f"{len(qc)} colony image(s) flagged")
    axes[1].set_title("Staining consistency")
    fig.tight_layout()
    return save_figure(fig, out_path)


__all__ = ["plate_heatmap", "field_uniformity", "count_qc"]
