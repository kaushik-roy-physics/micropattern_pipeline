"""
Radial-profile figures.

Conventions shared by every figure here:

* x is distance **from the colony edge** (µm on the ``um`` grid, the fraction
  of the colony radius on the ``norm`` grid), so the edge is always on the left.
* Each condition gets a color *and* a line style *and* a marker
  (:func:`downstream.style.series_style`), so figures survive grayscale.
* Bands are confidence intervals across **replicates** (colonies or wells, as the
  title says), never across nuclei.
* A grid point covered by too few replicates is a gap, not a point.
"""

from __future__ import annotations

import textwrap

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import style
from ..radial import DENSITY, GRID_LABEL, UM
from .base import (grid_shape, no_data, plural, reference_line, save_figure, shorten,
                   shorten_labels, value_label)

logger = logging.getLogger(__name__)

# Line styles for individual colonies; ':' is reserved for flagged colonies.
_SOLID_STYLES = ["-", "--", "-.", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]

READOUT_LABEL = {
    "mean": "Intensity (shared scale)",
    "positive_fraction": "Fraction of nuclei positive",
}


def _ylabel(readout: str, marker=None) -> str:
    if str(marker) == DENSITY:
        return DENSITY
    if readout == "mean" and marker is not None:
        return value_label(marker, READOUT_LABEL["mean"])
    return READOUT_LABEL.get(readout, readout)


def _colony_name(row) -> str:
    image = str(row.get("result_dir", row.get("field_id", "")))
    return Path(image).name or str(row.get("field_id"))


# ---------------------------------------------------------------------------
# One colony
# ---------------------------------------------------------------------------

def _logistic_curve(fit, xs):
    x50, k, lo, hi = fit
    return lo + (hi - lo) * 0.5 * (1.0 + np.tanh(0.5 * k * (xs - x50)))


def _annotate_boundaries(ax, borders) -> None:
    """Each boundary as a vertical line with its ±SE band (same look everywhere)."""
    for x, se, _ in borders:
        if np.isfinite(se) and se > 0:
            ax.axvspan(x - se, x + se, color=style.ANNOTATION, alpha=0.13, lw=0, zorder=0)
        ax.axvline(x, color=style.ANNOTATION, lw=1.1, zorder=1)


def colony_profile(binned: pd.DataFrame, markers: list[str], info: dict, out_path,
                   ylims: dict | None = None, annotations: dict | None = None):
    """One colony's profile per marker on its own bins: mean ± CI over nuclei.

    The CI here is *within* the colony (over its nuclei) and describes how well
    the colony's own curve is determined: it is not an error bar for inference.
    A second row shows the positive fraction when the markers have gates.
    *ylims* (``{marker: (lo, hi)}``) fixes each marker's intensity axis, so the
    same marker reads on the same range in every colony's figure.
    *annotations* (from the boundary step): ``fits`` ``{(marker, readout):
    (x50, k, lo, hi)}`` drawn dashed over the data, ``borders`` ``[(x, se,
    label)]`` drawn on every panel, ``peaks`` ``{marker: (x, se)}``.
    """
    ann = annotations or {}
    fits, borders, peaks = ann.get("fits") or {}, ann.get("borders") or [], ann.get("peaks") or {}
    fit_kw = {"color": style.OKABE_ITO["black"], "lw": 0.9, "ls": (0, (4, 2)), "zorder": 4}
    if binned.empty:
        return None
    have_pos = binned["positive_fraction"].notna().any()
    have_dens = int("density" in binned and bool(binned["density"].notna().any()))
    nrows = (2 if have_pos else 1) + have_dens
    ncols = len(markers)
    heights = [1.0] * (nrows - have_dens) + [0.45] * have_dens
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * sum(heights)),
                             squeeze=False, sharex=True,
                             gridspec_kw={"height_ratios": heights})
    if have_dens:
        # Bottom strip: one bar per bin, as wide as the bin and as tall as its nuclear
        # density, so unequal (quantile) bins and crowding are visible at a glance.
        bins = binned.drop_duplicates("bin").sort_values("bin")
        shade = [style.GREY, style.ANNOTATION]
        for j in range(ncols):
            axd = axes[-1, j]
            axd.bar(bins["bin_lo"], bins["density"], width=bins["bin_hi"] - bins["bin_lo"],
                    align="edge", color=[shade[i % 2] for i in range(len(bins))], alpha=0.6,
                    lw=0)
            axd.set_ylim(0, None)
            axd.tick_params(labelsize=7)
            if j == 0:
                axd.set_ylabel("nuclei /\n1000 µm²", fontsize=7.5)
        axes_main = axes[:-1]
    else:
        axes_main = axes
    for j, marker in enumerate(markers):
        sub = binned[binned["marker"] == marker].sort_values("bin")
        sty = style.series_style(j)
        ax = axes_main[0, j]
        if sub.empty:
            no_data(ax, f"{marker}: no data")
            continue
        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"], color=sty["color"], alpha=0.25,
                        lw=0)
        ax.plot(sub["x"], sub["mean"], color=sty["color"], ls=sty["linestyle"],
                marker=sty["marker"], ms=3)
        ax.set_title(marker)
        xs = np.linspace(0.0, float(np.nanmax(sub["x"])), 200)
        if (marker, "intensity") in fits:
            ax.plot(xs, _logistic_curve(fits[(marker, "intensity")], xs), **fit_kw)
        _annotate_boundaries(ax, borders)
        if marker in peaks:
            px, pse = peaks[marker]
            ax.axvline(px, color=sty["color"], lw=1.0, ls=":", zorder=1)
            ax.annotate(f"peak {px:.0f}" + (f" ± {pse:.0f}" if np.isfinite(pse) else "") + " µm",
                        (px, 0.98), xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=6.5, color=style.ANNOTATION)
        if ylims and marker in ylims:
            ax.set_ylim(*ylims[marker])
        if j == 0 or _ylabel("mean", marker) != _ylabel("mean"):
            ax.set_ylabel("\n".join(textwrap.wrap(_ylabel("mean", marker), 26)), fontsize=8)
        if have_pos:
            axp = axes_main[1, j]
            if sub["positive_fraction"].notna().sum() == 0:
                # Nothing to show below (signaling markers are never gated): drop the
                # empty panel; the intensity panel gets its own distance axis unless
                # the density strip below carries it.
                axp.remove()
                if not have_dens:
                    ax.tick_params(labelbottom=True)
                    ax.set_xlabel(GRID_LABEL[UM])
                continue
            axp.plot(sub["x"], sub["positive_fraction"], color=sty["color"],
                     ls=sty["linestyle"], marker=sty["marker"], ms=3)
            if (marker, "positive") in fits:
                axp.plot(xs, _logistic_curve(fits[(marker, "positive")], xs), **fit_kw)
            _annotate_boundaries(axp, borders)
            axp.set_ylim(-0.02, 1.02)
            if j == 0:
                axp.set_ylabel(_ylabel("positive_fraction"))
    for ax in axes[-1]:
        if ax.figure is not None:          # not a removed (empty) panel
            ax.set_xlabel(GRID_LABEL[UM])
    binning = str(binned["binning"].iloc[0]) if "binning" in binned else ""
    extra = ""
    if borders:
        extra = "\n" + " · ".join(f"{lab} boundary {x:.0f}" + (f" ± {se:.0f}" if np.isfinite(se)
                                                              else "") + " µm (grey line ± SE)"
                                  for x, se, lab in borders)
    if fits:
        extra += ("" if borders else "\n") + (" · " if borders else "") + "dashed = logistic fit"
    if ann.get("missing"):
        extra += "\n" + " · ".join(ann["missing"])
    if have_dens:
        extra += "\nbottom: one bar per bin (bar width = bin width in µm, height = nuclear density)"
    if info.get("caption"):
        extra += "\n" + str(info["caption"])
    fig.suptitle(f"{info.get('title', '')}\n{binning} bins · band = {info.get('ci', 95):g}% CI "
                 f"over nuclei in each bin{extra}", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Every colony of a marker, by well
# ---------------------------------------------------------------------------

def colony_gallery(curves: pd.DataFrame, marker: str, grid: str, readout: str,
                   panel_order: list, panel_titles: dict, out_path, flagged=frozenset(),
                   ylim: tuple | None = None):
    """One panel per well: each colony a thin line, the well mean bold.

    Flagged (but not excluded) colonies are drawn dotted so their influence on
    the mean is visible.
    """
    sub = curves[(curves["marker"] == marker) & (curves["grid"] == grid)]
    if sub.empty or sub[readout].notna().sum() == 0:
        return None
    n = len(panel_order)
    nrows, ncols = grid_shape(n, max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), panel_order):
        part = sub[sub["well_key"] == key]
        if part.empty:
            no_data(ax)
            continue
        for i, (colony, c) in enumerate(part.groupby("field_id", sort=True)):
            c = c.sort_values("bin")
            color = style.color(i)
            if colony in flagged:          # dotted + x: never a style a kept colony uses
                ax.plot(c["bin_center"], c[readout], color=color, lw=0.9, ls=":",
                        marker="x", ms=3, markevery=3)
            else:
                ax.plot(c["bin_center"], c[readout], color=color, lw=0.9, alpha=0.85,
                        ls=_SOLID_STYLES[i % len(_SOLID_STYLES)])
        mean = part.groupby("bin_center")[readout].mean()
        ax.plot(mean.index, mean.to_numpy(), color=style.OKABE_ITO["black"], lw=2.2,
                label="well mean")
        ax.set_title(panel_titles.get(key, str(key)), fontsize=8.5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(GRID_LABEL[grid], fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(_ylabel(readout, marker), fontsize=8)
    if readout == "positive_fraction":
        axes[0, 0].set_ylim(-0.02, 1.02)
    else:
        if ylim is not None:
            axes[0, 0].set_ylim(*ylim)
        for ax in axes.ravel()[:n]:
            reference_line(ax, marker)
    axes[0, 0].legend(fontsize=7, loc="best")
    note = " · dotted with × = flagged colony" if flagged else ""
    fig.suptitle(f"{marker}: every colony, by well (thin lines; bold = well mean){note}",
                 fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Conditions compared
# ---------------------------------------------------------------------------

def condition_profiles(summary: pd.DataFrame, markers: list[str], order: list[str],
                       grid: str, readout: str, tests: dict, unit: str, out_path,
                       controls=frozenset(), title: str = "", caption: str = ""):
    """One panel per marker: condition means ± CI across replicates.

    *tests* maps marker → the :func:`downstream.fda.functional_anova` result.
    Grid stretches where the max-F-controlled test finds a difference are marked
    by a bar under the curves; the global p sits in the panel title.
    """
    sub = summary[(summary["grid"] == grid) & (summary["readout"] == readout)]
    if sub.empty or sub["mean"].notna().sum() == 0:
        return None
    n = len(markers)
    nrows, ncols = grid_shape(n, max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)
    for ax, marker in zip(axes.ravel(), markers):
        part = sub[sub["marker"] == marker]
        if part.empty:
            no_data(ax, f"{marker}: no data")
            continue
        ns = []
        for i, cond in enumerate(order):
            c = part[part["condition"].astype(str) == cond].sort_values("bin")
            if c.empty:
                continue
            sty = style.series_style(i)
            color = style.CONTROL if cond in controls else sty["color"]
            ax.fill_between(c["bin_center"], c["ci_lo"], c["ci_hi"], color=color,
                            alpha=0.18, lw=0)
            # Condition only in the label: n differs between panels (stain sets),
            # and "0 (n=4)", "0 (n=2)" would read as different conditions.
            ax.plot(c["bin_center"], c["mean"], color=color, ls=sty["linestyle"],
                    marker=sty["marker"], ms=3, markevery=max(1, len(c) // 12),
                    label=shorten(cond))
            ns.append(f"{shorten(cond, 10)}: {int(c['n'].max())}")
        res = tests.get(marker)
        head = marker
        if res is not None and np.isfinite(res.get("global_p", np.nan)):
            p = res["global_p"]
            head += f"   global p = {p:.3g}" if p >= 1e-3 else f"   global p < 0.001"
            centers = np.asarray(res.get("centers", []), dtype=float)
            sig = np.asarray(res["sig_mask"], bool)
            if sig.size == centers.size and sig.any():
                lo, hi = ax.get_ylim()
                y = lo + 0.02 * (hi - lo)
                width = np.diff(centers).mean() if centers.size > 1 else 1.0
                for x0 in centers[sig]:
                    ax.plot([x0 - width / 2, x0 + width / 2], [y, y], color=style.ANNOTATION,
                            lw=4, solid_capstyle="butt")
                ax.plot([], [], color=style.ANNOTATION, lw=4,
                        label="differs here (max-F, FWER 5%)")
        ax.set_title(head, fontsize=9.5, pad=14)
        ax.text(0.5, 1.01, f"n {plural(unit)} · " + " · ".join(ns), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.5, color=style.ANNOTATION)
        ax.set_xlabel(GRID_LABEL[grid], fontsize=8)
        ax.set_ylabel(_ylabel(readout, marker), fontsize=8)
        if readout == "positive_fraction":
            ax.set_ylim(-0.02, 1.02)
        else:
            reference_line(ax, marker)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle((title + "\n" if title else "") +
                 f"Mean ± 95% CI across {plural(unit)} · {'µm grid' if grid == UM else 'normalized grid'}"
                 + (f"\n{caption}" if caption else ""), fontsize=10.5 if not caption else 9)
    # One legend for all panels, below them, so it never covers a curve.
    entries: dict = {}
    for ax in axes.ravel()[:n]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(lab, h)
    if entries:
        ncol = min(4, len(entries))
        rows = -(-len(entries) // ncol)
        fig.tight_layout(rect=(0, min(0.4, (0.22 * rows + 0.15) / fig.get_figheight()), 1, 1))
        fig.legend(list(entries.values()), list(entries), loc="lower center", ncol=ncol,
                   fontsize=7.5, frameon=False)
    else:
        fig.tight_layout()
    return save_figure(fig, out_path)


def pairwise_heatmap(pairwise: pd.DataFrame, out_path, title: str = ""):
    """FDR q-values of every pairwise functional test (rows: marker; columns: pair)."""
    if pairwise.empty:
        return None
    pw = pairwise.copy()
    pw["pair"] = pw["group_a"].astype(str) + "\nvs " + pw["group_b"].astype(str)
    mat = pw.pivot_table(index="marker", columns="pair", values="q_value", aggfunc="first")
    vals = -np.log10(np.clip(mat.to_numpy(float), 1e-4, 1))
    fig, ax = plt.subplots(figsize=(1.5 + 1.25 * mat.shape[1], 1.0 + 0.55 * mat.shape[0]))
    im = ax.imshow(vals, cmap=style.CMAP_SEQUENTIAL, vmin=0, vmax=4, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            q = mat.iat[i, j]
            if not np.isfinite(q):
                continue
            star = "*" if q < 0.05 else ""
            ax.text(j, i, f"{q:.2g}{star}", ha="center", va="center", fontsize=7.5,
                    color=style.text_on_cmap(vals[i, j], 0, 4))
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("−log10 q", fontsize=8)
    ax.set_title(title or "Pairwise functional tests: BH q (* q < 0.05)", fontsize=9.5)
    fig.tight_layout()
    return save_figure(fig, out_path)


def feature_strips(features: pd.DataFrame, tests: pd.DataFrame, markers: list[str],
                   feature_names: list[str], order: list[str], unit_col: str, out_path,
                   controls=frozenset()):
    """Replicate-level summary features: one point per replicate, mean ± 95% CI."""
    from scipy import stats as sps
    if features.empty:
        return None
    nrows, ncols = len(markers), len(feature_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.2 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    ticklabels = shorten_labels(order, 14)
    for i, marker in enumerate(markers):
        fm = features[features["marker"] == marker]
        for j, feat in enumerate(feature_names):
            ax = axes[i, j]
            unit = (fm.groupby(["condition", unit_col], observed=True)[feat].mean()
                    .reset_index())
            for k, cond in enumerate(order):
                v = unit.loc[unit["condition"].astype(str) == cond, feat].dropna().to_numpy()
                if v.size == 0:
                    continue
                sty = style.series_style(k)
                color = style.CONTROL if cond in controls else sty["color"]
                ax.scatter(k + rng.uniform(-0.15, 0.15, v.size), v, s=12, color=color,
                           marker=sty["marker"], edgecolors=style.MARKER_EDGE, linewidths=0.3,
                           zorder=3)
                if v.size > 1:
                    half = sps.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
                    ax.errorbar(k + 0.3, v.mean(), yerr=half, fmt="_", color=style.ANNOTATION,
                                ms=10, lw=1.2, zorder=4)
            t = tests[(tests["marker"] == marker) & (tests["feature"] == feat)]
            qtxt = ""
            if len(t) and np.isfinite(t["q_value"].iloc[0]):
                qtxt = f"\nq = {t['q_value'].iloc[0]:.2g}"
            ax.set_title(f"{marker} · {feat}{qtxt}", fontsize=7.5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(ticklabels, fontsize=6.5, rotation=45, ha="right")
            ax.tick_params(axis="y", labelsize=6.5)
    fig.suptitle("Per-replicate profile features (Kruskal-Wallis, BH q across markers × features)",
                 fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_path)


def similarity_heatmap(D: np.ndarray, labels: list[str], groups: list[str], order: list[str],
                       out_path, title: str = ""):
    """Colony × colony profile distance, colonies grouped by condition."""
    if D.size == 0:
        return None
    n = len(labels)
    size = min(14.0, 3.0 + 0.22 * n)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    vmax = np.nanpercentile(D[np.triu_indices(n, 1)], 98) if n > 1 else 1.0
    im = ax.imshow(D, cmap=style.CMAP_SEQUENTIAL + "_r", vmin=0, vmax=vmax)
    fs = max(4.0, min(7.5, 220.0 / max(n, 1)))
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=fs)
    # Condition blocks: outline + a colored, line-styled bar along the top.
    starts = [i for i in range(n) if i == 0 or groups[i] != groups[i - 1]] + [n]
    for b in range(len(starts) - 1):
        a, z = starts[b], starts[b + 1]
        k = order.index(groups[a]) if groups[a] in order else b
        sty = style.series_style(k)
        ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), z - a, z - a, fill=False,
                                   ec=style.OKABE_ITO["black"], lw=1.2))
        ax.plot([a - 0.5, z - 0.5], [-1.2, -1.2], color=sty["color"], lw=5,
                solid_capstyle="butt", clip_on=False)
        ax.text((a + z - 1) / 2, -2.0, shorten(groups[a], 18), ha="center", va="bottom",
                fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("RMS profile distance (shared scale)", fontsize=8)
    ax.set_title(title or "Colony profile dissimilarity (all markers, normalized grid)",
                 fontsize=10, pad=24)
    fig.tight_layout()
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Colony QC
# ---------------------------------------------------------------------------

def colony_qc_scatter(qc: pd.DataFrame, out_path):
    """Diameter vs nuclei per colony; flagged colonies marked and named."""
    if qc.empty or "diameter_um" not in qc:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    flagged = qc["flag_any"].astype(bool).to_numpy()
    ax.scatter(qc.loc[~flagged, "diameter_um"], qc.loc[~flagged, "n_nuclei_colony"], s=22,
               color=style.OKABE_ITO["blue"], edgecolors=style.MARKER_EDGE, linewidths=0.3,
               label="ok", marker="o")
    if flagged.any():
        f = qc[flagged]
        ax.scatter(f["diameter_um"], f["n_nuclei_colony"], s=46, color=style.THRESHOLD,
                   marker="X", edgecolors=style.MARKER_EDGE, linewidths=0.4,
                   label="flagged (label = flags)")
        for _, r in f.iterrows():
            ax.annotate(str(r["flags"]), (r["diameter_um"], r["n_nuclei_colony"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    exp = pd.to_numeric(qc.get("expected_diameter_um"), errors="coerce").dropna().unique()
    for i, e in enumerate(exp[:4]):
        ax.axvline(e, color=style.ANNOTATION, ls="--", lw=1,
                   label="expected diameter" if i == 0 else None)
    ax.set_xlabel("Fitted colony diameter (µm)")
    ax.set_ylabel("Nuclei in colony")
    ax.set_title(f"Colony geometry: {int(flagged.sum())} of {len(qc)} flagged", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_path)


def overlay_gallery(entries: list[tuple[Path, str, str]], out_path, title: str = "",
                    ncols: int = 6, thumb_px: int = 320):
    """Tile colony overlays: ``(png, caption, flags)`` per entry; flags in red text
    *and* prefixed with ✗ so they read without color."""
    entries = [e for e in entries if Path(e[0]).exists()]
    if not entries:
        return None
    import matplotlib.image as mpimg
    n = len(entries)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), squeeze=False)
    for ax, (png, caption, flags) in zip(axes.ravel(), entries):
        img = mpimg.imread(str(png))
        step = max(1, int(np.ceil(max(img.shape[:2]) / thumb_px)))
        ax.imshow(img[::step, ::step])
        ax.set_axis_off()
        ax.set_title(caption, fontsize=6.5)
        if flags:
            ax.text(0.02, 0.02, "✗ " + flags, transform=ax.transAxes, fontsize=6.5,
                    color=style.THRESHOLD, va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.85, "lw": 0, "pad": 1})
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_path)


__all__ = ["colony_profile", "colony_gallery", "condition_profiles", "pairwise_heatmap",
           "feature_strips", "similarity_heatmap", "colony_qc_scatter", "overlay_gallery"]
