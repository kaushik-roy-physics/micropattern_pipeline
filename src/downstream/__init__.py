"""
Micropattern downstream analysis: from per-nucleus measurements to radial
profiles of every colony, compared across wells and conditions.

The upstream pipeline (``process``) turns images into nuclear masks and
per-nucleus intensities and stops; it knows nothing about colonies. This package
finds the colony in each image, puts every nucleus on a distance-from-the-edge
coordinate, and turns that into biology: without knowing anything about a
particular experiment.

Four ideas carry the design:

1. **The colony is the replicate.** Nuclei estimate a colony's profile; colonies
   (or wells, when there are enough of them) are what every error bar and test
   counts.
2. **Two coordinates.** Distance from the edge in µm, and the same distance as a
   fraction of the colony radius. Both are computed, tabled, plotted and tested.
3. **One shared intensity scale per plate × panel**, so differences in
   expression level survive normalization (:mod:`downstream.radial`).
4. **Steps are plugins** (:mod:`downstream.registry`), checkpointed by the config
   they depend on, so changing a bin width re-runs only what it affects.

Usage::

    python main.py downstream                     # every step
    python main.py downstream --steps colony assemble qc
    python main.py downstream --force-step profiles
    python main.py downstream --list-steps
"""

# Lazy exports: importing ``downstream.colony`` in a parallel worker must not
# drag in the runner, every step and every figure module (on a network share
# that import costs more than the work itself).
_EXPORTS = {
    "Context": ("context", "Context"), "Data": ("data", "Data"), "RAW": ("data", "RAW"),
    "DAPI_NORM": ("data", "DAPI_NORM"), "register": ("registry", "register"),
    "REGISTRY": ("registry", "REGISTRY"), "order_steps": ("registry", "order_steps"),
    "run": ("runner", "run"), "describe_steps": ("runner", "describe_steps"),
    "apply_style": ("style", "apply_style"), "figure_style": ("style", "figure_style"),
}


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        module, attr = _EXPORTS[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_EXPORTS)
