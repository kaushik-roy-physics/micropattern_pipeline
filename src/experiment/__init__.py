"""
Experiment annotation layer: maps microscope images to plate wells and the
experimental conditions they represent, *without* touching the generic
``image_processing`` core.

The flow is:

    plate_map.yaml  (you author, one per acquisition folder)
          │
          ▼
    PlateMap       : validated geometry + well→condition + stain sets
          │
          ▼
    build_manifest : walk data, parse filenames → wells, join conditions,
                      emit a single ``manifest.parquet`` / ``manifest.csv``

Downstream / project-specific analyses load the manifest and join it to the
per-image ``per_nuclei_intensities.joblib`` files by ``result_dir``.
"""

from .manifest import PLATE_MAP_NAME, build_manifest, find_plate_maps
from .plate_map import PlateMap

__all__ = ["PlateMap", "build_manifest", "find_plate_maps", "PLATE_MAP_NAME"]

from provenance import version as _version  # noqa: E402

__version__ = _version()
