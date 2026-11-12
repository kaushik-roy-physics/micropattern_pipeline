"""
Confocal image-processing pipeline.

A modular, format-agnostic pipeline for multichannel confocal microscopy:
discovery → loading/MIP → Cellpose nuclear segmentation → per-nucleus
intensity extraction. Each step is independently runnable and checkpointed.
"""

from .directory_walker import DirectoryWalker
from .image_loader import ImageLoader
from .intensity_extractor import IntensityExtractor
from .segmentation import Segmentation

__all__ = ["DirectoryWalker", "ImageLoader", "Segmentation", "IntensityExtractor"]

from provenance import version as _version  # noqa: E402

__version__ = _version()
