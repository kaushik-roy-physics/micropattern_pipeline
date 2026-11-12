"""
Shared plotting plumbing: saving, condition ordering, and display-only capping.

Two conventions the whole package follows:

* **Capping is cosmetic.** A handful of saturated nuclei stretch every axis and
  flatten the biology. :func:`cap_for_display` removes cells above a high
  percentile *for plotting only*: thresholds, fractions and statistics always
  use the full data, so nothing is ever quietly excluded from a number.
* **Controls come first.** Conditions are ordered with the negative control
  leading unless the config gives an explicit order, so every figure reads left
  to right from the baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import style

logger = logging.getLogger(__name__)


def save_figure(fig, path: Path, dpi: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("figure → %s", path.name)
    return path


def cap_for_display(values: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    """Drop non-finite values and anything above *percentile* (display only)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    cap = np.nanpercentile(values, percentile)
    return values[values <= cap]


def display_limit(frame: pd.DataFrame, column: str, percentile: float = 99.9) -> float:
    vals = frame[column].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanpercentile(vals, percentile)) if vals.size else np.nan


def ordered_conditions(conditions, cfg: dict, control_names=()) -> list[str]:
    """Condition order: explicit config order, else controls first then sorted."""
    cond_cfg = cfg.get("conditions") or {}
    explicit = cond_cfg.get("order")
    present = list(dict.fromkeys(str(c) for c in conditions))
    if explicit:
        ordered = [c for c in map(str, explicit) if c in present]
        return ordered + [c for c in present if c not in ordered]
    controls = [c for c in present if c in set(map(str, control_names))]
    rest = sorted((c for c in present if c not in set(controls)), key=natural_key)
    return sorted(controls, key=natural_key) + rest


def natural_key(label) -> list:
    """Sort key reading numbers as numbers: '100' after '30', 'BMP 10 | x' after 'BMP 5 | x'."""
    import re
    parts = re.split(r"(\d+(?:\.\d+)?)", str(label))
    return [(0, float(p), "") if i % 2 else (1, 0.0, p.lower()) for i, p in enumerate(parts)]


def control_conditions(cells: pd.DataFrame, cfg: dict) -> set[str]:
    """Names of the conditions that are negative controls."""
    from ..aggregate import condition_columns, condition_label
    from ..thresholds import control_mask
    cond_cfg = cfg.get("conditions") or {}
    keys = condition_columns(cells, cond_cfg)
    mask = control_mask(cells, cond_cfg)
    if not mask.any():
        return set()
    labels = condition_label(cells.loc[mask], keys)
    return set(labels.unique())


def condition_palette(conditions, controls=()) -> dict:
    """Color per condition, controls in neutral gray."""
    controls = set(map(str, controls))
    out, i = {}, 0
    for c in conditions:
        if str(c) in controls:
            out[c] = style.CONTROL
        else:
            out[c] = style.color(i)
            i += 1
    return out


def shorten(label: str, width: int = 22) -> str:
    """Elide the MIDDLE of an over-long label, never the tail.

    Composite condition labels are ``treatment | ps_origin`` and differ in their
    LAST segment, so tail-elision is the one choice guaranteed to destroy the
    distinction: ``Activin_Noggin_CHIR | APSLM`` and ``… | PPSLM`` both collapse
    to ``Activin_Noggin_CHIR |…``. Keeping both ends keeps them apart.
    """
    label = str(label)
    if len(label) <= width:
        return label
    keep_tail = max(1, (width - 1) // 2)
    keep_head = max(1, width - 1 - keep_tail)
    return label[:keep_head] + "…" + label[len(label) - keep_tail:]


def shorten_labels(labels, width: int = 22) -> list[str]:
    """Shorten a whole axis at once, widening until nothing collides.

    :func:`shorten` cannot see its siblings, so it cannot know that the width it
    was handed makes two labels identical. This does: it retries at increasing
    widths and, in the worst case, returns the labels untouched. A crowded axis
    is a cosmetic problem; two categories that read as one is a wrong figure.
    """
    labels = [str(x) for x in labels]
    if len(set(labels)) != len(labels):
        return [shorten(x, width) for x in labels]   # already ambiguous upstream
    longest = max((len(x) for x in labels), default=width)
    for attempt in range(width, longest + 1):
        out = [shorten(x, attempt) for x in labels]
        if len(set(out)) == len(out):
            return out
    return labels


def annotate_n(ax, counts, y: float = 1.005, fontsize: float = 6.0,
               offset: int = 1) -> None:
    """Write the replicate count above each x position: n is never implicit.

    *offset* is the x coordinate of the first entry (violin plots are 1-indexed,
    bar plots 0-indexed). When every count is identical the value is stated once
    in the corner instead of repeated over each category.
    """
    counts = list(counts)
    if not counts:
        return
    if len(set(counts)) == 1:
        ax.text(1.0, y, f"n = {counts[0]} fields per condition",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=fontsize + 0.5, color=style.ANNOTATION)
        return
    for i, n in enumerate(counts):
        ax.text(i + offset, y, f"{n}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=fontsize, color=style.ANNOTATION)


def no_data(ax, message: str = "no data") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color=style.ANNOTATION)
    # tick_params, not set_xticks([]): tick locators are shared under sharex/sharey,
    # so emptying them here would strip the ticks from every sibling panel.
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)


def short_name(name: str, width: int = 18) -> str:
    """Compact image label: the Seq/Point/Well tokens when present, else a prefix.

    ``Point0003_Channel405,488,561,640_zmax`` → ``Point0003``; a raw ND2 called
    ``WellC10_Point0002_Seq0042`` → ``WellC10 Point0002 Seq0042``.
    """
    import re
    tokens = re.findall(r"(?:Well[A-Za-z]?\d+|Point\d+|Seq\d+)", str(name), re.IGNORECASE)
    if tokens:
        return " ".join(dict.fromkeys(tokens))
    return str(name)[:width]


def plural(unit: str, n: int = 2) -> str:
    """'colony' → 'colonies', 'well' → 'wells'."""
    if n == 1:
        return unit
    return unit[:-1] + "ies" if unit.endswith("y") else unit + "s"


# Markers not on the shared intensity scale (signaling readouts), set by the
# steps from tables/signaling_reference.csv: marker → (axis label, short label).
SHARED_SCALE_LABEL = "Intensity (shared scale)"
VALUE_LABELS: dict[str, tuple[str, str]] = {}
REFERENCE_MARKERS: set[str] = set()


def set_value_labels(labels: dict, reference_markers=()) -> None:
    """Register per-marker axis labels; *reference_markers* are drawn with a line at 1."""
    VALUE_LABELS.clear()
    VALUE_LABELS.update({str(k): v for k, v in (labels or {}).items()})
    REFERENCE_MARKERS.clear()
    REFERENCE_MARKERS.update(str(m) for m in reference_markers)


def value_label(marker, default: str = SHARED_SCALE_LABEL) -> str:
    """The y-axis label of a marker's intensity readout."""
    hit = VALUE_LABELS.get(str(marker))
    return hit[0] if hit else default


def value_short(marker) -> str:
    """Short unit for stacked panels ('' for shared-scale markers)."""
    hit = VALUE_LABELS.get(str(marker))
    return hit[1] if hit else ""


def reference_line(ax, marker) -> None:
    """Dashed line at 1 (= the reference population) for relative signaling readouts."""
    if str(marker) in REFERENCE_MARKERS:
        from .. import style
        ax.axhline(1.0, color=style.REFERENCE_LINE, ls="--", lw=0.9, zorder=0)


def grid_shape(n: int, max_cols: int = 4) -> tuple[int, int]:
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(n / cols))
    return rows, cols


__all__ = ["set_value_labels", "value_label", "value_short", "reference_line",
           "SHARED_SCALE_LABEL", "save_figure", "cap_for_display", "display_limit", "ordered_conditions",
           "control_conditions", "condition_palette", "shorten", "shorten_labels",
           "annotate_n", "no_data", "grid_shape"]
