"""
Which segmentation results exist, and which ones an analysis uses.

Every image gets one result folder next to it (``img.nd2`` → ``img/``), and that
folder holds **one** segmentation: 2-D (on the DAPI maximum projection) or 3-D
(on the z-stack). The mode it was made in is recorded in
``segmentation_metadata.json``; the per-nucleus intensities in
``per_nuclei_intensities.joblib`` are what the analysis reads.

This module looks at those files, without opening an image, so that

* ``main.py check`` can say, per dataset, what is on disk and whether it matches
  the mode the config asks for;
* ``main.py downstream`` analyzes only datasets that have results, in the mode
  asked for (``downstream.seg_mode`` / ``--seg-mode``), and names its output
  folder after it (``analysis_2d``, ``analysis_3d``, ``analysis_2d_3d``) when
  ``downstream.results_dir`` is ``auto``;
* the GUI can show the same, and preselect what can be analyzed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SEG_METADATA = "segmentation_metadata.json"
INTENSITIES = "per_nuclei_intensities.joblib"
MASKS = "nuclear_masks.tif"
MODES = ("2d", "3d")
AUTO_DIR_STEM = "analysis"


@dataclass
class SegStatus:
    """What one dataset folder holds."""
    name: str
    configured: str                       # the mode config.yaml asks for
    n_images: int = 0
    by_mode: dict = field(default_factory=dict)   # mode -> images with masks + intensities
    n_unknown: int = 0                    # masks without a recorded mode (old runs)
    n_masks_only: int = 0                 # segmented, not yet extracted

    @property
    def n_ready(self) -> int:
        return sum(self.by_mode.values()) + self.n_unknown

    @property
    def modes(self) -> list[str]:
        return [m for m in MODES if self.by_mode.get(m)]

    @property
    def mismatch(self) -> bool:
        """Results exist, but in another mode than the config asks for."""
        return bool(self.by_mode) and self.configured not in self.by_mode

    def describe(self) -> str:
        if not self.n_images:
            return "no images"
        if not self.n_ready and not self.n_masks_only:
            return f"not segmented yet (0 of {self.n_images} images)"
        parts = [f"{k} of {self.n_images} images in {m.upper()}"
                 for m, k in sorted(self.by_mode.items())]
        if self.n_unknown:
            parts.append(f"{self.n_unknown} of unrecorded mode")
        if self.n_masks_only:
            parts.append(f"{self.n_masks_only} segmented but not yet extracted")
        return ", ".join(parts)


def _mode_of(res: Path) -> str | None:
    import json
    try:
        with open(res / SEG_METADATA) as f:
            mode = json.load(f).get("mode")
        return str(mode).lower() if mode else None
    except (OSError, ValueError):
        return None


def dataset_status(ds, seq=None, wells=None) -> SegStatus:
    """Count the images of *ds* (a resolved Dataset) with results, per mode.

    Honors the image selection (``input.select`` or *seq* / *wells*), so a test
    run on a few images is not reported as mostly missing.
    """
    from image_processing import DirectoryWalker
    from image_processing.image_loader import result_dir_for
    from selection import Selection

    st = SegStatus(ds.name, ds.seg_mode)
    try:
        images = DirectoryWalker(ds.cfg).find_images()
        images = Selection.from_config(ds.cfg, seq, wells).filter_paths(images, ds.directory)
    except Exception as exc:                                  # noqa: BLE001
        logger.debug("[%s] cannot list images: %s", ds.name, exc)
        return st
    st.n_images = len(images)
    for img in images:
        res = result_dir_for(img)
        if not (res / MASKS).exists():
            continue
        if not (res / INTENSITIES).exists():
            st.n_masks_only += 1
            continue
        mode = _mode_of(res)
        if mode in MODES:
            st.by_mode[mode] = st.by_mode.get(mode, 0) + 1
        else:
            st.n_unknown += 1
    return st


def log_status(statuses: list[SegStatus], log=logger) -> int:
    """One line per dataset; returns the number of datasets in the wrong mode."""
    bad = 0
    for st in statuses:
        log.info("[%s] segmentation on disk: %s (config: %s)", st.name, st.describe(),
                 st.configured)
        if st.mismatch:
            bad += 1
            log.warning("[%s] the config asks for %s segmentation but the results on disk "
                        "are %s. The analysis uses what is on disk and labels it so. To get "
                        "%s results, run the %s segmentation job for this dataset (it "
                        "replaces the %s masks).", st.name, st.configured.upper(),
                        "/".join(m.upper() for m in st.modes), st.configured.upper(),
                        st.configured.upper(), "/".join(m.upper() for m in st.modes))
    return bad


def normalize_choice(value) -> str:
    """``downstream.seg_mode`` → one of auto | 2d | 3d | both."""
    v = str(value or "auto").strip().lower()
    aliases = {"null": "auto", "none": "auto", "all": "both", "2d+3d": "both",
               "2d_3d": "both", "2d,3d": "both"}
    v = aliases.get(v, v)
    if v not in ("auto", "2d", "3d", "both"):
        raise ValueError(f"downstream.seg_mode must be auto, 2d, 3d or both (got {value!r}).")
    return v


def select(datasets, statuses: list[SegStatus], choice: str, explicit: bool = False):
    """The datasets an analysis uses, and the segmentation modes they bring.

    * Datasets without any results are left out (with a warning), unless that
      leaves nothing, in which case they are kept so the error comes from the
      step that needs them.
    * ``2d`` / ``3d`` keep only datasets whose results are in that mode;
      ``auto`` / ``both`` keep every dataset with results.
    """
    choice = normalize_choice(choice)
    by_name = {s.name: s for s in statuses}
    keep, dropped = [], []
    for ds in datasets:
        st = by_name.get(ds.name)
        if st is None:
            keep.append(ds)
            continue
        if not st.n_ready:
            dropped.append((ds.name, "no segmentation results yet"))
            continue
        modes = set(st.modes) or {ds.seg_mode}
        if choice in ("2d", "3d") and choice not in modes:
            dropped.append((ds.name, f"its results are {'/'.join(sorted(modes)).upper()}, "
                                     f"not {choice.upper()}"))
            continue
        keep.append(ds)
    for name, why in dropped:
        (logger.warning if explicit else logger.info)(
            "[%s] left out of this analysis: %s.", name, why)
    if not keep:
        logger.warning("No dataset has %s segmentation results: nothing to analyze yet.",
                       "any" if choice in ("auto", "both") else choice.upper())
        return list(datasets), _modes(datasets, by_name)
    return keep, _modes(keep, by_name)


def _modes(datasets, by_name) -> list[str]:
    modes: set[str] = set()
    for ds in datasets:
        st = by_name.get(ds.name)
        modes |= set(st.modes) if st and st.modes else {ds.seg_mode}
    return [m for m in MODES if m in modes]


def auto_dir_name(modes: list[str]) -> str:
    """``['2d']`` → ``analysis_2d``; ``['2d', '3d']`` → ``analysis_2d_3d``."""
    modes = [m for m in MODES if m in set(modes)] or ["2d"]
    return f"{AUTO_DIR_STEM}_{'_'.join(modes)}"


def is_auto(results_dir) -> bool:
    return results_dir is None or str(results_dir).strip().lower() in ("auto", "", "null")


__all__ = ["SegStatus", "dataset_status", "log_status", "select", "auto_dir_name",
           "is_auto", "normalize_choice", "MODES"]
