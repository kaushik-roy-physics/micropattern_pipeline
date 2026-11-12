"""
IntensityExtractor: step 3 of the pipeline.

For every marker channel listed in ``channels.marker_channels`` it maps the
channel image onto the nuclear masks and computes a per-nucleus median
intensity (robust to hot pixels). Results are written to
``per_nuclei_intensities.joblib`` containing both **raw** and **DAPI-normalized**
intensities, the per-nucleus DAPI level, mask sizes, and **nucleus centroids**
(``(y, x)`` for 2-D, ``(z, y, x)`` for 3-D) for spatial QC of outliers.

All per-label statistics are vectorized: intensities via ``scipy.ndimage``,
centroids and sizes via coordinate ``bincount``: no per-nucleus Python loop.
Works identically for 2-D masks (MIP) and 3-D masks (z-stack).

Channel health
--------------
Alongside the intensities, each channel gets three cheap scalars recorded in
``global_stats``: the numbers that say whether the *measurement* is sound before
anyone asks what it means:

``saturated_fraction``
    Fraction of pixels at the detector ceiling. A channel that clips has lost
    information no analysis can recover, and the loss is invisible downstream:
    the per-nucleus medians stay finite and plausible.
``background``
    Median over the whole channel, i.e. the pedestal the nuclei sit on.
``signal_to_background``
    Nuclear median ÷ background. **The one number to look at.** A healthy
    channel is well above 1; a value at 1 means the nuclei are indistinguishable
    from their own background: the signature of a flooded field, and in
    particular of a maximum projection containing one over-bright z-slice, where
    every pixel dimmer than the flare has been overwritten by it.

The cost is one comparison and one median per channel (the median on a bounded
subsample), which is microseconds against the seconds that segmentation takes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import tifffile
from scipy.ndimage import median as nd_median

from .utils import load_image_metadata, resolve_channel_path, segmented_mode

logger = logging.getLogger(__name__)

CHECKPOINT = ".extracted"
OUTPUT_NAME = "per_nuclei_intensities.joblib"


def _extraction_current(out_dir: Path) -> bool:
    """True when the intensities were measured on the masks now on disk.

    Checked by file time, so it costs two stat calls: masks written after the
    extraction (a re-segmentation from before checkpoints were invalidated, or
    a hand copy) make it stale.
    """
    out, masks = Path(out_dir) / OUTPUT_NAME, Path(out_dir) / "nuclear_masks.tif"
    try:
        return out.stat().st_mtime >= masks.stat().st_mtime
    except OSError:
        return False


class IntensityExtractor:
    def __init__(self, cfg: dict):
        chan = cfg.get("channels", {})
        self.dapi_index: int = int(chan.get("dapi_index", 0))
        self.marker_channels: list[int] = [int(c) for c in chan.get("marker_channels", [])]
        intensity_cfg = cfg.get("intensity", {})
        self.normalize_by_dapi: bool = bool(intensity_cfg.get("normalize_by_dapi", True))
        # Detector ceiling for the saturation check. None → the dtype maximum
        # (65535 for uint16), which is right for a true 16-bit camera. Set it
        # explicitly for a camera that digitizes fewer bits into a 16-bit
        # container: a 12-bit sensor clips at 4095 and would otherwise never be
        # seen to saturate at all.
        sat = intensity_cfg.get("saturation_value")
        self.saturation_value: float | None = float(sat) if sat is not None else None
        # Mode follows segmentation so masks and channels stay consistent.
        self.mode: str = (cfg.get("segmentation", {}).get("mode") or "2d").lower()

    def process(self, path: Path, force: bool = False) -> None:
        from .image_loader import result_dir_for
        out_dir = result_dir_for(path)
        checkpoint = out_dir / CHECKPOINT
        seg_mode = segmented_mode(out_dir)
        if checkpoint.exists() and not force and _extraction_current(out_dir):
            logger.debug("IntensityExtractor: %s already extracted.", path.name)
            return
        if seg_mode is not None and seg_mode != self.mode:
            # Channels must be read in the masks' own geometry (z-stack vs MIP).
            logger.warning("IntensityExtractor: %s masks are %s but this run is %s: "
                           "measuring in %s to match the masks.", path.name, seg_mode,
                           self.mode, seg_mode)
        mode = seg_mode or self.mode

        meta = load_image_metadata(out_dir)
        is_zstack = bool(meta.get("is_zstack", False))
        channel_names = meta.get("channel_names")

        masks_path = out_dir / "nuclear_masks.tif"
        if not masks_path.exists():
            raise FileNotFoundError(
                f"Masks missing at {masks_path}. Run the 'segment' step first."
            )
        masks = tifffile.imread(masks_path)
        labels = np.unique(masks)
        labels = labels[labels > 0]

        result: dict = {
            "n_cells": int(labels.size),
            "labels": labels.astype(np.int64),
            "mode": mode,
            "is_zstack": is_zstack,
            "dapi_index": self.dapi_index,
            "marker_channels": list(self.marker_channels),
            "centroid_axes": "zyx" if masks.ndim == 3 else "yx",
            "raw_intensities": {},
            "dapi_normalized_intensities": {},
            "global_stats": {},
            "channel_names": {},
        }

        if labels.size == 0:
            logger.warning("IntensityExtractor: no nuclei in %s", path.name)
            joblib.dump(result, out_dir / OUTPUT_NAME)
            checkpoint.touch()
            return

        # Centroids + mask sizes (vectorized over all labels at once).
        result["centroids"], result["mask_sizes"] = _centroids_and_sizes(masks, labels)

        # DAPI per nucleus (for normalization and QC).
        dapi_per_cell = None
        dapi_path = resolve_channel_path(out_dir, self.dapi_index, mode, is_zstack)
        if dapi_path.exists():
            dapi_raw = tifffile.imread(dapi_path)
            dapi_img = dapi_raw.astype(np.float32)
            dapi_per_cell = nd_median(dapi_img, labels=masks, index=labels).astype(np.float32)
            # DAPI health matters twice over: it is the segmentation channel and
            # the normalization denominator, so a clipped DAPI silently distorts
            # every dapi_norm value on top of whatever it did to the masks.
            result["global_stats"][self.dapi_index] = {
                **_stats(dapi_per_cell),
                **_channel_health(dapi_img, dapi_per_cell,
                                  _ceiling_for(dapi_raw.dtype, self.saturation_value)),
            }
        else:
            logger.warning("IntensityExtractor: DAPI channel %s missing; "
                           "normalization disabled for %s", dapi_path.name, path.name)
        result["dapi_per_cell"] = dapi_per_cell
        result["channel_names"][self.dapi_index] = _name_for(channel_names, self.dapi_index, "DAPI")

        for ch in self.marker_channels:
            ch_path = resolve_channel_path(out_dir, ch, mode, is_zstack)
            if not ch_path.exists():
                logger.warning("IntensityExtractor: channel file %s missing for %s",
                               ch_path.name, path.name)
                continue
            raw_img = tifffile.imread(ch_path)
            img = raw_img.astype(np.float32)
            per_cell = nd_median(img, labels=masks, index=labels).astype(np.float32)

            result["raw_intensities"][ch] = per_cell
            health = _channel_health(img, per_cell,
                                     _ceiling_for(raw_img.dtype, self.saturation_value))
            result["global_stats"][ch] = {**_stats(per_cell), **health}
            result["channel_names"][ch] = _name_for(channel_names, ch, f"ch{ch}")
            _warn_unhealthy(path.name, ch, result["channel_names"][ch], health)

            if self.normalize_by_dapi and dapi_per_cell is not None:
                norm = np.divide(per_cell, dapi_per_cell,
                                 out=np.zeros_like(per_cell),
                                 where=dapi_per_cell > 0).astype(np.float32)
                result["dapi_normalized_intensities"][ch] = norm

        joblib.dump(result, out_dir / OUTPUT_NAME)
        checkpoint.touch()
        logger.info(
            "IntensityExtractor [%s]: %s; %d nuclei, channels %s",
            mode, path.name, labels.size, self.marker_channels,
        )


# ---------------------------------------------------------------------------
# Vectorized geometry
# ---------------------------------------------------------------------------

def _centroids_and_sizes(masks: np.ndarray, labels: np.ndarray):
    """
    Geometric centroids and voxel/pixel counts for every label, computed from a
    single ``nonzero`` pass. Returns ``(centroids (n, ndim), sizes (n,))`` with
    rows ordered to match ``labels``.
    """
    coords = np.nonzero(masks)               # tuple of (z,)y,x index arrays
    flat_labels = masks[coords]
    n_max = int(masks.max())

    counts = np.bincount(flat_labels, minlength=n_max + 1).astype(np.float64)
    sums = [np.bincount(flat_labels, weights=c.astype(np.float64), minlength=n_max + 1)
            for c in coords]

    sel = labels                              # 1-based label ids
    safe_counts = np.where(counts[sel] > 0, counts[sel], 1.0)
    centroids = np.stack([s[sel] / safe_counts for s in sums], axis=1).astype(np.float32)
    sizes = counts[sel].astype(np.int64)
    return centroids, sizes


# ---------------------------------------------------------------------------
# Channel health (see the module docstring)
# ---------------------------------------------------------------------------

# Enough samples for a stable background median without ever materializing a
# copy of a large z-stack: a median is a quantile, and a few million draws pin it
# far tighter than the number is ever read to.
_BACKGROUND_SAMPLE = 4_000_000

# Only used for the log line: the actionable flags live downstream in `qc`,
# where they can be compared against the field's own siblings.
_SATURATED_WARN = 0.001
_SIGNAL_WARN = 1.05


def _ceiling_for(dtype, override: float | None) -> float:
    """Detector ceiling: the configured value, else the dtype maximum."""
    if override is not None:
        return float(override)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float("inf")          # float images carry no meaningful ceiling


def _channel_health(img: np.ndarray, per_cell: np.ndarray, ceiling: float) -> dict:
    """Saturation, background pedestal and nuclear signal-to-background."""
    flat = img.reshape(-1)
    if flat.size > _BACKGROUND_SAMPLE:
        flat = flat[:: max(1, flat.size // _BACKGROUND_SAMPLE)]
    background = float(np.median(flat))
    nuclear = float(np.median(per_cell)) if per_cell.size else float("nan")
    return {
        "saturation_value": float(ceiling),
        # Counted over every pixel, never a subsample: clipping is rare by
        # definition, and a subsample would be exactly how you miss a small patch.
        "saturated_fraction": float(np.count_nonzero(img >= ceiling) / img.size),
        "nuclei_saturated_fraction": (float(np.count_nonzero(per_cell >= ceiling) / per_cell.size)
                                      if per_cell.size else float("nan")),
        "background": background,
        "signal_to_background": (nuclear / background if background > 0 else float("nan")),
    }


def _warn_unhealthy(image_name: str, ch: int, marker: str, health: dict) -> None:
    """Say so at extraction time, while the image is still in front of you."""
    if health["saturated_fraction"] > _SATURATED_WARN:
        logger.warning("IntensityExtractor: %s ch%d (%s); %.1f%% of pixels at the "
                       "detector ceiling (%.0f); this channel is clipped.",
                       image_name, ch, marker,
                       100 * health["saturated_fraction"], health["saturation_value"])
    sb = health["signal_to_background"]
    if np.isfinite(sb) and sb < _SIGNAL_WARN:
        logger.warning("IntensityExtractor: %s ch%d (%s); nuclei are only %.2f× "
                       "their own background; the channel looks flooded (a bright "
                       "z-slice in a MIP does exactly this).",
                       image_name, ch, marker, sb)


def _name_for(channel_names, idx: int, default: str) -> str:
    if channel_names and 0 <= idx < len(channel_names):
        return str(channel_names[idx])
    return default


def _stats(arr: np.ndarray) -> dict:
    return {
        "n_cells": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }
