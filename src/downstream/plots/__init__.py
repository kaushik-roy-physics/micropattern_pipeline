"""Figure builders for the downstream pipeline.

Every module here takes plain DataFrames and an output path and returns the path
it wrote (or ``None`` when there was nothing to draw). They hold no pipeline
state, which keeps them reusable from a notebook.

Color lives exclusively in :mod:`downstream.style`.
"""

from .base import (annotate_n, cap_for_display, condition_palette, control_conditions,
                   grid_shape, no_data, ordered_conditions, plural, reference_line,
                   save_figure, set_value_labels, shorten, short_name, value_label)
from .boundary import bin_check, border_map, boundary_metrics, registered_profiles
from .distributions import threshold_qc
from .effects import effect_forest, power_curves, trend_panels
from .joint import PANEL_KINDS, coexpression_page, coexpression_panel, multimarker_profile
from .plate import count_qc
from .signaling import signaling_summary
from .radial import (colony_gallery, colony_profile, colony_qc_scatter, condition_profiles,
                     feature_strips, overlay_gallery, pairwise_heatmap, similarity_heatmap)

__all__ = [
    "annotate_n", "cap_for_display", "condition_palette", "control_conditions",
    "grid_shape", "no_data", "ordered_conditions", "save_figure", "shorten",
    "short_name", "plural", "set_value_labels", "value_label", "reference_line",
    "threshold_qc", "count_qc",
    "colony_profile", "colony_gallery", "condition_profiles", "pairwise_heatmap",
    "feature_strips", "similarity_heatmap", "colony_qc_scatter", "overlay_gallery",
    "multimarker_profile", "coexpression_page", "coexpression_panel", "PANEL_KINDS",
    "signaling_summary", "bin_check", "boundary_metrics", "registered_profiles", "border_map",
    "trend_panels", "effect_forest", "power_curves",
]
