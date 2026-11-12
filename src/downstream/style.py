"""
Publication figure style, and the single place any color is defined.

Every figure in the downstream package draws its colors from here. Nothing
else in the package may hard-code a hex value; ``tests/test_style.py`` enforces
that by grepping the source, so the palette can never quietly drift.

The qualitative palette is **Okabe-Ito**, which stays distinguishable under all
three common color-vision deficiencies and in grayscale. Sequential and
diverging maps are the perceptually-uniform matplotlib ones (viridis / cividis /
magma, and coolwarm-free ``RdBu_r`` only where a true zero-centered diverging map
is needed). Red-green is never the sole contrast.

Color is also never the *only* channel: :func:`series_style` pairs each color
with its own line style and marker, so a figure survives being printed in black
and white.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from cycler import cycler

# --- Okabe-Ito qualitative palette ----------------------------------------
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Cycle order: maximum separation between neighbors, dark-first so the common
# 2-3 series case is also the most legible in grayscale.
PALETTE = [OKABE_ITO[k] for k in
           ("blue", "vermillion", "green", "purple", "orange", "sky", "yellow", "black")]

# Semantic roles used across figures.
GREY = "#999999"
CONTROL = GREY                 # negative-control / reference conditions
THRESHOLD = OKABE_ITO["vermillion"]   # gate lines
POSITIVE = OKABE_ITO["blue"]
NEGATIVE = GREY
ANNOTATION = "#444444"
NEUTRAL_FILL = "#E6E6E6"       # "negative"/"other" slices in stacked compositions
MARKER_EDGE = "#333333"        # outline for scatter points: see below
HALO = "white"                 # outline that lifts a line off any colormap
MASKED = "#DDDDDD"             # cells deliberately left out of a heatmap

GRIDLINE = "#D9D9D9"           # vertical guides shared by stacked panels

# Signaling: the ring QC image and the reference level on signaling figures.
NUCLEUS_OUTLINE = OKABE_ITO["orange"]
RING_OUTLINE = OKABE_ITO["sky"]
REFERENCE_LINE = GREY          # dashed line at 1 = the reference population

# Co-expression classes of a marker pair (A, B). Okabe-Ito blue / green /
# vermillion stay apart under every common color-vision deficiency, each class
# also has its own marker shape, and double-negative nuclei recede in gray.
COEXPRESSION_CLASSES = {
    "AB": {"color": OKABE_ITO["vermillion"], "marker": "o", "label": "both +"},
    "A": {"color": OKABE_ITO["blue"], "marker": "s", "label": "{a} + only"},
    "B": {"color": OKABE_ITO["green"], "marker": "^", "label": "{b} + only"},
    "none": {"color": "#BBBBBB", "marker": ".", "label": "both -"},
}

# Sequential / diverging maps (perceptually uniform, CVD-safe).
#
# The density map is REVERSED (light = few cells). A dark-low map on a white page
# makes sparse noise the most visually salient thing in the panel, which is
# exactly backwards; light-low also keeps the figure readable when printed.
CMAP_SEQUENTIAL = "viridis"
CMAP_SEQUENTIAL_ALT = "cividis"
CMAP_DENSITY = "magma_r"
CMAP_DIVERGING = "RdBu_r"
CMAP_POSITION = "cividis"      # distance from the colony edge, on scatter points

# Line styles and markers, cycled alongside color for grayscale survival.
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (4, 2, 1, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def color(i: int) -> str:
    """Qualitative color *i*, recycled."""
    return PALETTE[i % len(PALETTE)]


def series_style(i: int) -> dict:
    """Color + line style + marker for series *i*: three redundant channels."""
    return {
        "color": PALETTE[i % len(PALETTE)],
        "linestyle": LINE_STYLES[i % len(LINE_STYLES)],
        "marker": MARKERS[i % len(MARKERS)],
    }


def category_colors(categories) -> dict:
    """Stable ``{category: color}`` map in the given order."""
    return {c: color(i) for i, c in enumerate(categories)}


# --- Contrast ---------------------------------------------------------------
# Every heatmap writes its value into the cell, and the label has to stay legible
# at BOTH ends of the ramp. Hand-written rules like ``"white" if v > 0.55 * vmax
# else "black"`` encode an assumption about which end of the colormap is dark:
# and silently invert when the map is reversed or swapped. Deciding from the
# actual luminance of the patch the text sits on cannot be got backwards.

def contrast_text(rgba) -> str:
    """``'black'`` or ``'white'``, whichever is readable on *rgba*."""
    def _linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
    r, g, b = (_linear(c) for c in to_rgb(rgba))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b   # WCAG relative luminance
    return "black" if luminance > 0.4 else "white"


def text_on_cmap(value: float, vmin: float, vmax: float,
                 cmap_name: str = CMAP_SEQUENTIAL) -> str:
    """Readable text color for a heatmap cell holding *value*."""
    span = float(vmax) - float(vmin)
    frac = 0.0 if span <= 0 else (float(value) - float(vmin)) / span
    return contrast_text(plt.get_cmap(cmap_name)(min(1.0, max(0.0, frac))))


def band_ramp(n: int = 3, cmap_name: str = CMAP_SEQUENTIAL) -> list[str]:
    """*n* ordered colors, light → dark, sampled from the sequential map.

    Ordered categories (low/mid/high) need an ordered ramp. A qualitative palette
    cannot express "more", and Okabe-Ito's vermillion is *lighter* than its blue,
    so a hand-picked triple inverts halfway up and breaks in grayscale.
    """
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [to_hex(cmap(0.45))]
    lo, hi = 0.72, 0.18                     # light end → dark end
    return [to_hex(cmap(lo + (hi - lo) * i / (n - 1))) for i in range(n)]


def halo(linewidth: float = 2.4, foreground: str = HALO) -> list:
    """Path effect outlining a line so it reads over *any* colormap.

    Gate lines are drawn on top of density maps whose colors are not known in
    advance. An outline makes the line legible against a pale bin and a
    near-black one alike, without giving up the color that identifies it.
    """
    return [patheffects.withStroke(linewidth=linewidth, foreground=foreground)]


def apply_style(dpi: int = 300, font_scale: float = 1.0) -> None:
    """Install the publication rcParams globally."""
    base = 10.0 * font_scale
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": base,
        "axes.titlesize": base * 1.15,
        "axes.labelsize": base * 1.05,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.85,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.6,
        "errorbar.capsize": 2.5,
        "axes.prop_cycle": cycler(color=PALETTE),
        "image.cmap": CMAP_SEQUENTIAL,
        "pdf.fonttype": 42,      # editable text in Illustrator
        "ps.fonttype": 42,
    })


@contextmanager
def figure_style(dpi: int = 300, font_scale: float = 1.0):
    """Apply the style for the duration of a block, then restore what was there."""
    saved = plt.rcParams.copy()
    try:
        apply_style(dpi, font_scale)
        yield
    finally:
        plt.rcParams.update(saved)


__all__ = ["OKABE_ITO", "PALETTE", "GREY", "CONTROL", "THRESHOLD", "POSITIVE",
           "NEGATIVE", "ANNOTATION", "NEUTRAL_FILL", "MARKER_EDGE", "HALO", "MASKED",
           "GRIDLINE", "COEXPRESSION_CLASSES", "CMAP_POSITION",
           "CMAP_SEQUENTIAL", "CMAP_SEQUENTIAL_ALT",
           "CMAP_DENSITY", "CMAP_DIVERGING", "LINE_STYLES", "MARKERS",
           "color", "series_style", "category_colors", "contrast_text", "text_on_cmap",
           "band_ramp", "halo", "apply_style", "figure_style"]
