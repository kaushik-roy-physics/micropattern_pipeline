"""
Example: an experiment-specific analysis that plugs into the generic pipeline.

Copy this file next to your ``config.yaml`` (e.g. as ``domain_widths.py``) and
name it in the config::

    downstream:
      extra_modules: [domain_widths]

Nothing in ``src/downstream`` changes. The step below runs after ``profiles``,
is checkpointed like the built-ins (``--force-step domain_width`` redoes it),
and its figure appears in report.html under its own heading.

What it computes: for every colony and marker, the **width of the positive
domain** measured from the edge (how far inward the fraction of positive nuclei
stays above ``THRESHOLD``), and each condition's mean with a 95 % CI across
colonies. For an edge marker (ISL1, CDX2) that is the width of the outer ring.
Set ``FROM_EDGE = False`` to measure a center domain (SOX2) from the colony's
center outward instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from downstream import register, style
from downstream.plots import save_figure, shorten

logger = logging.getLogger(__name__)

THRESHOLD = 0.5            # "positive domain" = bins where > 50 % of nuclei are positive
FROM_EDGE = True           # True: ring from the edge inward; False: core from the center outward


def domain_width(position: np.ndarray, fraction: np.ndarray) -> float:
    """Width (µm) of the run of bins above THRESHOLD that starts at position 0."""
    ok = np.isfinite(fraction)
    position, fraction = position[ok], fraction[ok]
    if position.size == 0:
        return np.nan
    order = np.argsort(position)
    position, above = position[order], fraction[order] > THRESHOLD
    if not above[0]:
        return 0.0
    run = above.size if above.all() else int(np.argmin(above))     # first bin below
    step = float(np.median(np.diff(position))) if position.size > 1 else 0.0
    return float(position[run - 1] + step / 2)                     # bin center → bin edge


@register("domain_width", after=["profiles"], config_keys=["profiles", "conditions"],
          description="Width of each marker's positive domain per colony (example step)")
def step_domain_width(ctx) -> None:
    curves = ctx.read_table("colony_profiles.csv")
    um = curves[(curves["grid"] == "um") & curves["positive_fraction"].notna()]
    if um.empty:
        logger.info("domain_width: no %% positive profiles (no gates); nothing to do.")
        return

    rows = []
    for (field_id, marker), part in um.groupby(["field_id", "marker"]):
        dist = part["bin_center"].to_numpy(float)
        # From the center: distance from the center = R − distance from the edge.
        position = dist if FROM_EDGE else float(part["radius_um"].iloc[0]) - dist
        rows.append({"field_id": field_id, "marker": marker,
                     "condition": str(part["condition"].iloc[0]), "well": part["well"].iloc[0],
                     "width_um": domain_width(position, part["positive_fraction"].to_numpy(float))})
    widths = pd.DataFrame(rows)
    ctx.write_table(widths, "domain_widths.csv")

    import matplotlib.pyplot as plt
    markers = [m for m in ctx.markers if m in set(widths["marker"])]
    conditions = list(dict.fromkeys(widths["condition"]))
    fig, axes = plt.subplots(1, len(markers), figsize=(3.2 * len(markers), 3.2), squeeze=False)
    for ax, marker in zip(axes.ravel(), markers):
        sub = widths[widths["marker"] == marker]
        for i, cond in enumerate(conditions):
            v = sub.loc[sub["condition"] == cond, "width_um"].dropna().to_numpy()
            if v.size == 0:
                continue
            sty = style.series_style(i)
            ax.scatter(np.full(v.size, i), v, color=sty["color"], marker=sty["marker"],
                       edgecolors=style.MARKER_EDGE, linewidths=0.3, zorder=3)
            if v.size > 1:
                half = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
                ax.errorbar(i + 0.25, v.mean(), yerr=half, fmt="_", color=style.ANNOTATION,
                            ms=12, lw=1.3)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([shorten(c, 14) for c in conditions], rotation=30, ha="right")
        ax.set_title(f"{marker}: > {THRESHOLD:.0%} positive")
        ax.set_ylabel(f"domain width from the {'edge' if FROM_EDGE else 'center'} (µm)")
    fig.tight_layout()
    save_figure(fig, ctx.figure("domain_width", "domain_widths.png"))
