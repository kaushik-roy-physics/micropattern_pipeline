"""
Synthetic micropattern experiments with known answers.

Each "image" is a circular colony of disk-shaped nuclei on a hexagonal lattice,
with marker intensities that are exact functions of the distance from the colony
edge. What the generator writes is exactly what ``process`` would have left
behind (``nuclear_masks.tif``, ``image_metadata.json``, ``ch0.tif`` and
``per_nuclei_intensities.joblib``), so the whole downstream analysis runs on it
without a GPU, and every number it reports can be checked against the truth.

Profiles (``d`` = distance from the edge in µm)::

    EDGE   = amp · exp(−d / 60)          edge-high, like ISL1 / CDX2
    CENTER = 1 / (1 + exp(−(d − mid)/25))  center-high, like SOX2 (mid = 120 µm)
    FLAT   = 0.5                          no pattern

``amp`` is 1 by default; a condition can lower it to test that the shared scale
keeps level differences. ``mid`` moves the CENTER boundary, so a dose series, or
well and experiment offsets with a known size, can be built from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import tifffile

PIXEL_UM = 1.3
NUC_RADIUS_PX = 4
SPACING_PX = 10.0


def edge_profile(d, amp=1.0):
    return amp * np.exp(-np.asarray(d) / 60.0)


def center_profile(d, mid=120.0):
    return 1.0 / (1.0 + np.exp(-(np.asarray(d) - mid) / 25.0))


def _lattice(cy, cx, radius_px, rng):
    """Hexagonal lattice points inside a circle, jittered."""
    pts = []
    step_y = SPACING_PX * np.sqrt(3) / 2
    n = int(radius_px / step_y) + 2
    for i in range(-n, n + 1):
        y = cy + i * step_y
        off = (SPACING_PX / 2) if i % 2 else 0.0
        m = int(radius_px / SPACING_PX) + 2
        for j in range(-m, m + 1):
            x = cx + j * SPACING_PX + off
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius_px ** 2:
                pts.append((y, x))
    pts = np.asarray(pts) + rng.normal(0, 0.8, size=(len(pts), 2))
    return pts


def make_colony_image(shape=(600, 600), center=(300, 300), radius_um=250.0, amp=1.0,
                      n_strays=6, neighbor=None, seed=0, mid=120.0):
    """Return ``(masks, dapi, true_dist_um, values, labels, centroids)`` for one colony.

    *neighbor*: optional ``(cy, cx, radius_px)`` of a second colony creeping in.
    """
    rng = np.random.default_rng(seed)
    radius_px = radius_um / PIXEL_UM
    pts = _lattice(center[0], center[1], radius_px, rng)
    true_r = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    true_dist = (radius_px - true_r) * PIXEL_UM
    kind = np.zeros(len(pts), int)                       # 0 colony, 1 stray, 2 neighbor

    strays = []
    while len(strays) < n_strays:
        y, x = rng.uniform(10, shape[0] - 10), rng.uniform(10, shape[1] - 10)
        if np.hypot(y - center[0], x - center[1]) > radius_px + 40:
            strays.append((y, x))
    if strays:
        pts = np.vstack([pts, strays])
        kind = np.concatenate([kind, np.ones(len(strays), int)])
        true_dist = np.concatenate([true_dist, np.full(len(strays), np.nan)])
    if neighbor is not None:
        npts = _lattice(neighbor[0], neighbor[1], neighbor[2], rng)
        pts = np.vstack([pts, npts])
        kind = np.concatenate([kind, np.full(len(npts), 2)])
        true_dist = np.concatenate([true_dist, np.full(len(npts), np.nan)])

    lo, hi = NUC_RADIUS_PX + 1, np.asarray(shape) - NUC_RADIUS_PX - 2
    keep = ((pts[:, 0] >= lo) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi[1]))
    pts, kind, true_dist = pts[keep], kind[keep], true_dist[keep]

    masks = np.zeros(shape, np.uint32)
    yy, xx = np.mgrid[-NUC_RADIUS_PX:NUC_RADIUS_PX + 1, -NUC_RADIUS_PX:NUC_RADIUS_PX + 1]
    disk = yy ** 2 + xx ** 2 <= NUC_RADIUS_PX ** 2
    for lab, (y, x) in enumerate(pts, start=1):
        iy, ix = int(round(y)), int(round(x))
        sl = masks[iy - NUC_RADIUS_PX:iy + NUC_RADIUS_PX + 1, ix - NUC_RADIUS_PX:ix + NUC_RADIUS_PX + 1]
        sl[disk & (sl == 0)] = lab
    labels = np.arange(1, len(pts) + 1)

    d = np.nan_to_num(true_dist, nan=80.0)
    noise = lambda: rng.normal(0, 0.03, len(pts))                    # noqa: E731
    values = {1: edge_profile(d, amp) + noise(),
              2: center_profile(d, mid) + noise(),
              3: np.full(len(pts), 0.5) + noise()}
    dapi = np.where(masks > 0, 1000.0, 50.0).astype(np.float32)
    return masks, dapi, true_dist, values, labels, pts, kind


def write_result_dir(res_dir: Path, masks, dapi, values, labels, centroids,
                     pixel_um: float | None = PIXEL_UM) -> None:
    """Write what load/segment/extract would have written for one image."""
    res_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(res_dir / "nuclear_masks.tif", masks)
    tifffile.imwrite(res_dir / "ch0.tif", dapi)
    meta = {"file_format": ".tif", "height_px": masks.shape[0], "width_px": masks.shape[1],
            "n_channels": 4, "n_z": 1, "is_zstack": False}
    if pixel_um:
        meta.update(pixel_size_x_um=pixel_um, pixel_size_y_um=pixel_um)
    (res_dir / "image_metadata.json").write_text(json.dumps(meta))
    n = len(labels)
    dapi_per = np.full(n, 1000.0)
    raw = {ch: (v * 1000.0).astype(np.float64) for ch, v in values.items()}
    joblib.dump({
        "n_cells": n, "labels": labels.astype(np.int64), "mode": "2d", "is_zstack": False,
        "dapi_index": 0, "marker_channels": sorted(values), "centroid_axes": "yx",
        "raw_intensities": raw,
        "dapi_normalized_intensities": {ch: r / dapi_per for ch, r in raw.items()},
        "dapi_per_cell": dapi_per, "mask_sizes": np.full(n, 50.0),
        "centroids": np.asarray(centroids, np.float32),
        "global_stats": {}, "channel_names": {},
    }, res_dir / "per_nuclei_intensities.joblib")
    (res_dir / ".imgproc_output").touch()


GENERATOR_KEYS = {"amp", "mid", "mid_sd"}


def make_experiment(root: Path, wells=None, colonies_per_well: int = 4, seed: int = 0,
                    downstream: dict | None = None, artifacts: bool = True) -> Path:
    """A predefined-points experiment folder with config + plate map, ready for
    ``manifest`` and ``downstream``. Returns the config path.

    *wells* maps well number → a spec: ``amp`` (EDGE amplitude), ``mid`` (CENTER
    boundary, µm from the edge) and ``mid_sd`` (colony-to-colony SD of ``mid``)
    drive the images; every other key (``class``, ``experiment``, a dose, ...)
    is written to the plate map as a condition column.
    *downstream* is deep-merged into the config's ``downstream`` section.
    With *artifacts* (default), colony 0 of well 1 is clipped by the image border
    and colony 1 of well 2 has a large neighboring colony creeping in.
    """
    wells = wells or {1: {"class": "lineA", "amp": 1.0},
                      2: {"class": "lineB", "amp": 1.0},
                      3: {"class": "lineC", "amp": 0.5}}
    root = Path(root)
    data = root / "zmax"
    data.mkdir(parents=True, exist_ok=True)
    seq = 0
    rng = np.random.default_rng(seed + 10_000)
    for well, spec in wells.items():
        for k in range(colonies_per_well):
            center, neighbor = (300, 300), None
            if artifacts and well == 1 and k == 0:
                center = (300, 140)                      # rim cut by the left border
            if artifacts and well == 2 and k == 1:
                neighbor = (40, 560, 90)
            mid = float(spec.get("mid", 120.0)) + rng.normal(0, float(spec.get("mid_sd", 0.0)))
            masks, dapi, _, values, labels, cents, _ = make_colony_image(
                center=center, amp=spec.get("amp", 1.0), neighbor=neighbor, seed=seed + seq,
                mid=mid)
            stem = f"Point{seq:04d}_Channel405,488,561,640_zmax"
            (data / f"{stem}.tif").touch()
            write_result_dir(data / stem, masks, dapi, values, labels, cents)
            seq += 1

    lines = ["plate: 96well", "mode: predefined", f"points_per_well: {colonies_per_well}",
             "stains:", "  default: {0: DAPI, 1: EDGE, 2: CENTER, 3: FLAT}", "wells:"]
    for well, spec in wells.items():
        fields = {"treatment": "BMP4", **{k: v for k, v in spec.items()
                                          if k not in GENERATOR_KEYS}}
        lines.append(f"  {well}: {{" + ", ".join(f"{k}: {v}" for k, v in fields.items()) + "}")
    (data / "plate_map.yaml").write_text("\n".join(lines) + "\n")

    cfg = {
        "defaults": {"input": {"recursive": True, "file_types": [".tif"]},
                     "channels": {"dapi_index": 0, "marker_channels": [1, 2, 3]},
                     "segmentation": {"mode": "2d"}},
        "datasets": [{"name": "zmax", "directory": "./zmax"}],
        "downstream": {
            "results_dir": "./analysis",
            "conditions": {"condition_key": ["class"]},
            "colony": {"close_radius_um": 15, "min_colony_area_um2": 5000,
                       "expected_diameter_um": 500, "n_jobs": 2},
            "profiles": {"bin_um": 10, "n_norm_bins": 20, "min_nuclei_per_bin": 5},
            "compare": {"n_permutations": 999},
            "qc": {"exclude_flags": ["failed"]},
        },
    }
    if downstream:
        from pipeline_config import deep_merge
        cfg["downstream"] = deep_merge(cfg["downstream"], downstream)
    import yaml
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


__all__ = ["make_colony_image", "write_result_dir", "make_experiment", "edge_profile",
           "center_profile", "PIXEL_UM", "GENERATOR_KEYS"]
