"""
Figures that look at several markers of one colony (or well) at once.

``multimarker_profile``
    Every marker's radial profile stacked on one shared distance axis, each on
    its own y-range, with a trend strip underneath (:mod:`downstream.trends`):
    at each position, which markers rise and which fall.

``coexpression_page``
    One row per marker pair: per-nucleus scatter colored by distance from the
    edge, the same scatter split into radial bands, the within-bin correlation
    and double-positive fraction against distance, the (A, B) trajectory from
    edge to center and, for a single colony with gates, the map of where the
    co-expressing nuclei sit.

Both take plain DataFrames and dicts and return the path written (or ``None``).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .. import style
from .base import no_data, reference_line, save_figure, value_label, value_short

logger = logging.getLogger(__name__)

SCALE_LABEL = "Intensity (shared scale)"


def _axis(marker) -> str:
    """'ISL1 (shared scale)', or the signaling unit ('SMAD2/3 N:C (÷ pluripotent)')."""
    label = value_label(marker, "")
    return label or f"{marker} (shared scale)"


def _guides(ax) -> None:
    """Vertical guides, so one distance can be read straight down the stack."""
    ax.grid(axis="x", color=style.GRIDLINE, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


# ---------------------------------------------------------------------------
# Stacked profiles + trend strip
# ---------------------------------------------------------------------------

def multimarker_profile(markers: list[str], tracks: dict, positive: dict, trend: pd.DataFrame,
                        info: dict, out_path):
    """Stacked per-marker tracks, an optional % positive overlay, and the trend strip.

    *tracks*: ``{marker: DataFrame(x, mean, lo, hi)}``; *info* may hold
    ``traces`` (``{marker: [(x, y), ...]}``, thin lines behind the mean),
    ``x_label``, ``title``, ``band``, ``z_crit``, ``z_max`` and ``ylims``
    (``{marker: (lo, hi)}``: each track on the range that marker has in every
    figure; without it each track autoscales). *positive*:
    ``{marker: DataFrame(x, y, lo, hi)}``, all markers on one 0-1 axis.
    *trend*: long table with ``marker``, ``x``, ``z``.
    """
    markers = [m for m in markers if m in tracks and not tracks[m].empty]
    if len(markers) < 2:
        return None
    positive = {m: p for m, p in (positive or {}).items()
                if m in markers and p["y"].notna().any()}
    traces = info.get("traces") or {}
    n = len(markers)
    # Narrow and tall: each track about twice as wide as high, so a rise or fall
    # reads as a slope instead of being flattened across a wide page.
    heights = [1.7] * n + ([1.9] if positive else []) + [0.32 * n + 0.25]
    fig, axes = plt.subplots(len(heights), 1, figsize=(4.6, sum(heights) + 0.9), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.3})
    xs_all = []
    for j, marker in enumerate(markers):
        ax = axes[j]
        sty = style.series_style(j)
        for tx, ty in traces.get(marker, []):
            ax.plot(tx, ty, color=sty["color"], lw=0.6, alpha=0.35, zorder=1)
        t = tracks[marker].sort_values("x")
        xs_all.append(t["x"].to_numpy(np.float64))
        if t[["lo", "hi"]].notna().any().any():
            ax.fill_between(t["x"], t["lo"], t["hi"], color=sty["color"], alpha=0.22, lw=0,
                            zorder=2)
        ax.plot(t["x"], t["mean"], color=sty["color"], ls=sty["linestyle"],
                marker=sty["marker"], ms=2.6, zorder=3)
        unit = value_short(marker)
        ax.set_ylabel(marker + (f"\n{unit}" if unit else ""), rotation=0, ha="right",
                      va="center", fontsize=9, color=sty["color"])
        reference_line(ax, marker)
        for bx, bse, _ in info.get("borders") or []:      # boundary step: line ± SE
            if np.isfinite(bse) and bse > 0:
                ax.axvspan(bx - bse, bx + bse, color=style.ANNOTATION, alpha=0.13, lw=0,
                           zorder=0)
            ax.axvline(bx, color=style.ANNOTATION, lw=1.1, zorder=1)
        ylims = info.get("ylims") or {}
        if marker in ylims:
            ax.set_ylim(*ylims[marker])
        ax.tick_params(axis="y", labelsize=7)
        ax.locator_params(axis="y", nbins=3)
        _guides(ax)
    head = SCALE_LABEL if not any(value_short(m) for m in markers) else \
        "Shared scale (signaling: unit under the name)"
    fixed = all(m in (info.get("ylims") or {}) for m in markers)
    axes[0].set_title(head + ("\neach marker on one y-range across all figures" if fixed
                              else "\neach panel has its own y-range"),
                      fontsize=8, loc="left", color=style.ANNOTATION)

    row = n
    if positive:
        ax = axes[row]
        for j, marker in enumerate(markers):
            if marker not in positive:
                continue
            p = positive[marker].sort_values("x")
            sty = style.series_style(j)
            if p[["lo", "hi"]].notna().any().any():
                ax.fill_between(p["x"], p["lo"], p["hi"], color=sty["color"], alpha=0.15, lw=0)
            ax.plot(p["x"], p["y"], color=sty["color"], ls=sty["linestyle"],
                    marker=sty["marker"], ms=2.6, label=marker)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Fraction\npositive", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, ncol=min(4, len(positive)), loc="upper right")
        _guides(ax)
        row += 1

    _trend_strip(axes[row], markers, trend, info)
    axes[row].set_xlabel(info.get("x_label", ""))
    xs = np.concatenate(xs_all) if xs_all else np.array([0.0])
    if np.isfinite(xs).any():
        axes[row].set_xlim(min(0.0, float(np.nanmin(xs))), float(np.nanmax(xs)) * 1.02)
    height = fig.get_figheight()
    band = info.get("band", "")
    if info.get("borders"):
        band += "\n" + " · ".join(f"{lab} boundary {x:.0f} µm (grey line ± SE)"
                                  for x, _, lab in info["borders"])
    if info.get("missing"):
        band += "\n" + " · ".join(info["missing"])
    text = f"{info.get('title', '')}\n{band}"
    fig.subplots_adjust(top=1.0 - (0.35 + 0.18 * (text.count("\n") + 1)) / height)
    fig.suptitle(text, fontsize=9, y=1.0 - 0.08 / height)
    return save_figure(fig, out_path)


def _trend_strip(ax, markers: list[str], trend: pd.DataFrame, info: dict) -> None:
    z_crit = float(info.get("z_crit", 1.96))
    z_max = float(info.get("z_max", 4.0))
    sub = trend[trend["marker"].isin(markers)] if trend is not None and len(trend) else None
    if sub is None or sub.empty or not np.isfinite(sub["z"]).any():
        no_data(ax, "no trend: too few bins")
        # This is the bottom axis of a sharex stack: it carries the x labels.
        ax.tick_params(bottom=True, labelbottom=True)
        return
    xs = np.sort(sub["x"].dropna().unique())
    Z = (sub.pivot_table(index="marker", columns="x", values="z", aggfunc="first")
         .reindex(index=markers, columns=xs).to_numpy(np.float64))
    xe = _cell_edges(xs)
    ye = np.arange(len(markers) + 1)
    cmap = plt.get_cmap(style.CMAP_DIVERGING)
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(Z), cmap=cmap, vmin=-z_max, vmax=z_max,
                         shading="flat", rasterized=True)
    ax.set_facecolor(style.MASKED)
    for i in range(Z.shape[0]):
        for k in range(Z.shape[1]):
            z = Z[i, k]
            if np.isfinite(z) and abs(z) >= z_crit:
                rgba = cmap((np.clip(z, -z_max, z_max) + z_max) / (2 * z_max))
                ax.text(xs[k], i + 0.5, "▲" if z > 0 else "▼", ha="center", va="center",
                        fontsize=5.5, color=style.contrast_text(rgba))
    ax.set_ylim(len(markers), 0)
    ax.set_yticks(np.arange(len(markers)) + 0.5)
    ax.set_yticklabels(markers, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Local trend (slope / SE; ▲▼ = |z| ≥ {z_crit:g}, descriptive, unadjusted)",
                 fontsize=7.5, loc="left", color=style.ANNOTATION)
    cax = ax.inset_axes([1.015, 0.0, 0.018, 1.0])
    cb = ax.figure.colorbar(mesh, cax=cax)
    cb.set_label("z", fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Spatially resolved co-expression
# ---------------------------------------------------------------------------

def _limits(values: np.ndarray, gate=None) -> tuple[float, float]:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    if gate is not None and np.isfinite(gate):
        lo, hi = min(lo, gate), max(hi, gate)
    pad = 0.05 * max(hi - lo, 1e-6)
    return float(lo - pad), float(hi + pad)


def _gates(ax, gate_a, gate_b) -> None:
    kw = {"color": style.THRESHOLD, "lw": 0.9, "ls": "--", "path_effects": style.halo(1.8)}
    if gate_a is not None:
        ax.axvline(gate_a, **kw)
    if gate_b is not None:
        ax.axhline(gate_b, **kw)


def _scatter_by_position(ax, pts, a, b, row, norm, lims):
    cmap = plt.get_cmap(style.CMAP_POSITION)
    sc = ax.scatter(pts["va"], pts["vb"], c=pts["x"], cmap=cmap, norm=norm, s=3.5, lw=0,
                    alpha=0.6, rasterized=True)
    _gates(ax, row.get("gate_a"), row.get("gate_b"))
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_xlabel(_axis(a), fontsize=8)
    ax.set_ylabel(_axis(b), fontsize=8)
    r = row.get("r_all")
    label = row.get("band_symbol", row.get("method_symbol", "ρ"))
    if r is not None and np.isfinite(r):
        head = row.get("scatter_title", f"all nuclei (n = {len(pts):,})")
        ax.set_title(f"{head}: {label} = {r:.2f}", fontsize=7.5)
    cax = ax.inset_axes([1.02, 0.0, 0.035, 1.0])
    cb = ax.figure.colorbar(sc, cax=cax)
    cb.set_label(row["coord_label"], fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.solids.set_alpha(1)


def _band_panels(axes, pts, row, norm, lims):
    cmap = plt.get_cmap(style.CMAP_POSITION)
    edges = row["band_edges"]
    unit = row["band_unit"]
    for k, ax in enumerate(axes):
        lo, hi = edges[k], edges[k + 1]
        last = k == len(axes) - 1
        sel = (pts["x"] >= lo) & ((pts["x"] <= hi) if last else (pts["x"] < hi))
        band = pts[sel]
        ax.scatter(band["va"], band["vb"], color=cmap(norm(0.5 * (lo + hi))), s=3, lw=0,
                   alpha=0.6, rasterized=True)
        _gates(ax, row.get("gate_a"), row.get("gate_b"))
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        r = row["band_r"][k]
        fmt = "{:.0f}" if unit == "µm" else "{:.2f}"
        head = f"{fmt.format(lo)}-{fmt.format(hi)} {unit}".strip()
        symbol = row.get("band_symbol", row.get("method_symbol", "ρ"))
        stat = f"{symbol} = {r:.2f}\nn = {len(band):,}" if np.isfinite(r) else f"\nn = {len(band):,}"
        ax.set_title(f"{head}\n{stat}", fontsize=6.5)
        ax.tick_params(labelsize=6)
        if k:
            ax.set_yticklabels([])
        ax.set_xlabel(row["a"], fontsize=7)


def _correlation_panel(ax, prof, row):
    ax.axhline(0, color=style.GREY, lw=0.7)
    for tx, ty in row.get("corr_traces", []):
        ax.plot(tx, ty, color=style.color(0), lw=0.6, alpha=0.3)
    p = prof.sort_values("bin_center")
    if p["correlation"].notna().any():
        ax.fill_between(p["bin_center"], p["corr_lo"], p["corr_hi"], color=style.color(0),
                        alpha=0.22, lw=0)
        ax.plot(p["bin_center"], p["correlation"], color=style.color(0), marker="o", ms=2.8)
    else:
        no_data(ax, "too few nuclei per bin")
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel(row["coord_label"], fontsize=7.5)
    ax.set_ylabel(f"{row['method_name']} {row.get('method_symbol', 'ρ')}", fontsize=8)
    ax.set_title(row.get("corr_title", "Correlation within each bin"), fontsize=7.5)
    _guides(ax)


def _double_positive_panel(ax, prof, row):
    if not row.get("has_gates"):
        no_data(ax, "needs a gate\nfor both markers")
        ax.set_title("Double positive", fontsize=7.5)
        return
    p = prof.sort_values("bin_center")
    obs, exp_ = style.series_style(1), style.series_style(0)
    if "frac_ab_lo" in p and p["frac_ab_lo"].notna().any():
        ax.fill_between(p["bin_center"], p["frac_ab_lo"], p["frac_ab_hi"], color=obs["color"],
                        alpha=0.2, lw=0)
    ax.plot(p["bin_center"], p["frac_ab"], color=obs["color"], ls="-", marker=obs["marker"],
            ms=2.8, label="observed (both +)")
    ax.plot(p["bin_center"], p["frac_expected"], color=style.ANNOTATION, ls="--",
            marker=exp_["marker"], ms=2.4, mfc="white", label="expected if independent")
    top = np.nanmax(np.r_[p["frac_ab"].to_numpy(float), p["frac_expected"].to_numpy(float),
                          [0.05]])
    ax.set_ylim(-0.01, min(1.02, top * 1.45 + 0.02))
    ax.set_xlabel(row["coord_label"], fontsize=7.5)
    ax.set_ylabel("Fraction of nuclei", fontsize=8)
    ax.set_title("Double positive vs independence", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper right")
    _guides(ax)


def _trajectory_panel(ax, prof, row, norm):
    p = prof.sort_values("bin_center")
    p = p[np.isfinite(p["mean_a"]) & np.isfinite(p["mean_b"])]
    if len(p) < 2:
        no_data(ax, "too few bins")
        return
    xy = p[["mean_a", "mean_b"]].to_numpy(np.float64)
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    cmap = plt.get_cmap(style.CMAP_POSITION)
    pos = p["bin_center"].to_numpy(np.float64)
    lc = LineCollection(seg, cmap=cmap, norm=norm, lw=1.8)
    lc.set_array(0.5 * (pos[:-1] + pos[1:]))
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], c=pos, cmap=cmap, norm=norm, s=16,
               edgecolor=style.MARKER_EDGE, lw=0.4, zorder=3)
    ax.annotate("edge", xy[0], xytext=(4, 4), textcoords="offset points", fontsize=6.5,
                color=style.ANNOTATION)
    ax.annotate("center", xy[-1], xytext=(-4, 6), textcoords="offset points", fontsize=6.5,
                ha="right",
                color=style.ANNOTATION)
    _gates(ax, row.get("gate_a"), row.get("gate_b"))
    # Pad each axis by its own range: the two markers can differ by orders of
    # magnitude (scaled 0-1 vs raw a.u.), and one shared pad flattens the smaller.
    for k, setter in ((0, ax.set_xlim), (1, ax.set_ylim)):
        span = np.ptp(xy[:, k])
        pad = 0.08 * span if span > 0 else 0.05 * max(abs(xy[0, k]), 1e-3)
        setter(xy[:, k].min() - pad, xy[:, k].max() + pad)
    ax.set_xlabel(f"mean {row['a']}", fontsize=7.5)
    ax.set_ylabel(f"mean {row['b']}", fontsize=7.5)
    ax.set_title("Bin means, edge → center", fontsize=7.5)


def _map_panel(ax, spatial, row):
    if spatial is None or not row.get("has_gates") or spatial.empty:
        no_data(ax, "needs a gate\nfor both markers")
        ax.set_title("Where in the colony", fontsize=7.5)
        return
    handles = []
    for key in ("none", "A", "B", "AB"):
        spec = style.COEXPRESSION_CLASSES[key]
        sel = spatial["cls"] == key
        label = spec["label"].format(a=row["a"], b=row["b"])
        size = 2.0 if key == "none" else 5.0
        ax.scatter(spatial.loc[sel, "x_um"], spatial.loc[sel, "y_um"], s=size,
                   color=spec["color"], marker=spec["marker"], lw=0, rasterized=True)
        handles.append(Line2D([], [], color=spec["color"], marker=spec["marker"], ls="",
                              ms=4, label=f"{label} ({100 * sel.mean():.0f}%)"))
    radius = row.get("radius_um")
    if radius and np.isfinite(radius):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=style.ANNOTATION, lw=0.7,
                                ls="--"))
    ax.set_aspect("equal")
    ax.invert_yaxis()                        # image orientation
    ax.set_xlabel("µm", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Where in the colony", fontsize=7.5)
    ax.legend(handles=handles, fontsize=5.8, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              handletextpad=0.2)


def coexpression_page(rows: list[dict], info: dict, out_path):
    """One row of panels per marker pair; returns the path or ``None``.

    Each *row* dict carries ``a``, ``b``, ``pts`` (DataFrame ``va``, ``vb``,
    ``x``), ``profile`` (per-bin ``bin_center``, ``correlation``, ``corr_lo``,
    ``corr_hi``, ``mean_a``, ``mean_b``, ``frac_ab``, ``frac_expected``),
    ``band_edges`` and ``band_r``, the gates (``gate_a``, ``gate_b``, None when
    absent), ``has_gates``, ``coord_label``, ``coord_max``, ``band_unit``,
    ``method_name``/``method_symbol`` and optionally ``spatial`` (DataFrame
    ``x_um``, ``y_um``, ``cls``), ``radius_um`` and ``corr_traces``.
    *info* holds ``title`` and ``show_map``.
    """
    rows = [r for r in rows if r["pts"] is not None and len(r["pts"])]
    if not rows:
        return None
    n_bands = max(len(r["band_edges"]) - 1 for r in rows)
    show_map = bool(info.get("show_map"))
    # Outer columns: scatter | colorbar room | bands | correlation | double + |
    # trajectory | map. The bands share their axes, so they sit in a tight inner grid.
    widths = [3.0, 0.75, 1.5 * n_bands, 2.6, 2.6, 2.6] + ([3.6] if show_map else [])
    height = 2.9 * len(rows) + 0.7
    fig = plt.figure(figsize=(sum(widths) + 1.0, height))
    gs = fig.add_gridspec(len(rows), len(widths), width_ratios=widths, wspace=0.42,
                          hspace=0.8)
    for i, row in enumerate(rows):
        pts = row["pts"]
        norm = Normalize(0.0, float(row["coord_max"]))
        lims = (_limits(pts["va"], row.get("gate_a")), _limits(pts["vb"], row.get("gate_b")))
        ax = fig.add_subplot(gs[i, 0])
        _scatter_by_position(ax, pts, row["a"], row["b"], row, norm, lims)
        ax.text(-0.3, 1.16, f"{row['a']} vs {row['b']}", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="left")
        inner = gs[i, 2].subgridspec(1, n_bands, wspace=0.1)
        band_axes = [fig.add_subplot(inner[0, k]) for k in range(len(row["band_edges"]) - 1)]
        _band_panels(band_axes, pts, row, norm, lims)
        _correlation_panel(fig.add_subplot(gs[i, 3]), row["profile"], row)
        _double_positive_panel(fig.add_subplot(gs[i, 4]), row["profile"], row)
        _trajectory_panel(fig.add_subplot(gs[i, 5]), row["profile"], row, norm)
        if show_map:
            _map_panel(fig.add_subplot(gs[i, 6]), row.get("spatial"), row)
    fig.suptitle(info.get("title", ""), fontsize=9.5, y=1.0 - 0.1 / height)
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# One panel of the page, on its own (for slides)
# ---------------------------------------------------------------------------

# kind → (width, height) in inches; "bands" adds width per band.
PANEL_KINDS = {"scatter": (4.4, 3.6), "bands": (0.8, 3.5), "correlation": (4.0, 3.3),
               "double_positive": (4.0, 3.3), "trajectory": (4.0, 3.5), "map": (5.8, 4.2)}
PANEL_TEXT_SCALE = 1.45      # the page's 6-8 pt text → ~9-12 pt at slide size


def _scale_text(fig, factor: float) -> None:
    """Enlarge every text object (titles, labels, ticks, legends, colorbars) in place."""
    from matplotlib.text import Text
    # Tick labels go through tick_params (it also sets the template that ticks
    # created at draw time copy); everything else is scaled once, here.
    seen = set()
    for ax in fig.findobj(lambda a: hasattr(a, "xaxis") and hasattr(a, "yaxis")):
        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis.get_major_ticks()
            if ticks:
                axis.set_tick_params(labelsize=ticks[0].label1.get_fontsize() * factor)
            seen.update(id(t) for tick in ticks for t in (tick.label1, tick.label2))
    for text in fig.findobj(Text):
        if id(text) not in seen:
            seen.add(id(text))
            text.set_fontsize(text.get_fontsize() * factor)


def coexpression_panel(kind: str, row: dict, info: dict, out_path):
    """One panel of :func:`coexpression_page` as its own figure, sized for a slide.

    Same drawing code as the page, so the two can never disagree. Panels whose
    page version would only say "needs a gate" are not written.
    """
    if kind not in PANEL_KINDS:
        raise ValueError(f"unknown co-expression panel {kind!r}; one of {list(PANEL_KINDS)}")
    pts = row["pts"]
    if pts is None or not len(pts):
        return None
    if kind in ("double_positive", "map") and not row.get("has_gates"):
        return None
    if kind == "map" and (row.get("spatial") is None or row["spatial"].empty):
        return None
    norm = Normalize(0.0, float(row["coord_max"]))
    lims = (_limits(pts["va"], row.get("gate_a")), _limits(pts["vb"], row.get("gate_b")))
    w, h = PANEL_KINDS[kind]
    if kind == "bands":
        n = len(row["band_edges"]) - 1
        fig, axes = plt.subplots(1, n, figsize=(1.6 * n + w, h), sharey=True, squeeze=False)
        _band_panels(list(axes[0]), pts, row, norm, lims)
        axes[0, 0].set_ylabel(_axis(row["b"]), fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(w, h))
        if kind == "scatter":
            _scatter_by_position(ax, pts, row["a"], row["b"], row, norm, lims)
        elif kind == "correlation":
            _correlation_panel(ax, row["profile"], row)
        elif kind == "double_positive":
            _double_positive_panel(ax, row["profile"], row)
        elif kind == "trajectory":
            _trajectory_panel(ax, row["profile"], row, norm)
        else:
            _map_panel(ax, row.get("spatial"), row)
    _scale_text(fig, PANEL_TEXT_SCALE)
    import textwrap
    sub = textwrap.fill(str(info.get("title", "")), width=int(9 * fig.get_figwidth()))
    fig.suptitle(f"{row['a']} vs {row['b']}\n{sub}", fontsize=10.5)
    fig.tight_layout()
    return save_figure(fig, out_path)


__all__ = ["multimarker_profile", "coexpression_page", "coexpression_panel", "PANEL_KINDS"]
