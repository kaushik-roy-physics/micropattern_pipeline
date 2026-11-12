"""
Colony geometry: the main micropatterned colony in each image, and every
nucleus's distance from its edge.

This is the one per-image step of the downstream analysis. It reads only the
segmentation output (``nuclear_masks.tif``) and the image metadata, so it runs on
CPU after the GPU job has finished, and writes four files into each image's
result directory:

``colony_mask.tif``        uint8 mask of the main colony
``colony_nuclei.parquet``  one row per nucleus: label, centroid, ``in_colony``,
                           ``dist_edge_um``, ``r_center_um``, ``r_norm``
``colony.json``            center, radius, areas, flags, the parameters used and
                           a hash of them (the per-image checkpoint)
``colony_overlay.png``     DAPI + nuclear outlines + colony edge + fitted circle +
                           outliers: the picture to look at before believing a
                           profile

Method
------
1. **Footprint.** The union of the nuclear masks is morphologically *closed* with
   a disk of ``close_radius_um`` (bridging the gaps between nuclei) and its holes
   filled. Closing and the optional opening are done with distance transforms,
   which are exact for a disk and cost the same whatever the radius: a 23-px
   structuring element on a 4000² stitched image is where the old pipeline spent
   its time.
2. **Main colony.** Connected components of the footprint; the main one is the
   largest (or the one nearest the image center, ``main_colony: center``). Other
   components above ``min_colony_area_um2`` are neighboring colonies that crept
   into the field; their nuclei, and any stray nucleus outside the main mask, are
   flagged ``in_colony = False`` and never enter a profile.
3. **Circle fit.** An algebraic (Kåsa) fit to the colony's *true* boundary (the
   pixels where the colony meets background, never the pixels where it meets
   the image border), iteratively trimmed of boundary pixels far off the circle
   (a tail of escaped cells, a bump in the rim), then a few Gauss-Newton steps
   of the geometric fit. The mask is then cut at ``trim_outside_circle`` × R:
   cells beyond the pattern have left it and are treated as outside the colony
   (``trimmed_fraction`` records how much was cut), and the circle is refitted.
4. **Distance from the edge.** For a colony fully inside the image this is the
   Euclidean distance transform of the mask, in µm: robust to colonies that are
   not perfect circles. A colony cut by the image border has no visible edge on
   that side, and the distance transform would measure every nucleus there to
   the far, visible edge: overestimating it. When more than
   ``max_border_fraction`` of the colony's perimeter lies on the image border the
   distance is instead taken from the fitted circle (``R − r``), and the colony
   is flagged. ``edge_distance: mask | circle`` forces one method.
5. **Normalized coordinate.** ``r_norm = dist_edge_um / R`` (0 at the edge, 1 at
   the center), for comparing colonies of the same nominal size.

Pixel size is never guessed. It comes from the image metadata or from
``colony.pixel_size_um``; with neither, the image fails with a message saying so,
because every µm in the analysis would otherwise be wrong by an unknown factor.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from . import style

logger = logging.getLogger(__name__)

MASK_NAME = "colony_mask.tif"
TABLE_NAME = "colony_nuclei.parquet"
META_NAME = "colony.json"
OVERLAY_NAME = "colony_overlay.png"
OVERLAY_THUMB_NAME = "colony_overlay_thumb.png"   # image only, for the QC contact sheet
MASKS_NAME = "nuclear_masks.tif"
IMAGE_META_NAME = "image_metadata.json"

DEFAULTS = {
    "pixel_size_um": None,         # None → image metadata
    "close_radius_um": 15.0,
    "open_radius_um": 0.0,
    "min_colony_area_um2": 5000.0,
    "main_colony": "largest",      # largest | center
    "edge_distance": "auto",       # auto | mask | circle
    "max_border_fraction": 0.15,
    "trim_outside_circle": 1.1,    # cut the mask beyond this × R of the fitted circle; 0 = off
    "overlay": True,
    "overlay_max_px": 2000,        # overlays are downsampled to this longest side
}
_MAIN_CHOICES = {"largest", "center"}
_EDGE_CHOICES = {"auto", "mask", "circle"}


def settings(cfg: dict) -> dict:
    """The ``colony`` config block merged over :data:`DEFAULTS`, validated."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("colony") or {}).items() if v is not None
                or k == "pixel_size_um"})
    if str(out["main_colony"]) not in _MAIN_CHOICES:
        raise ValueError(f"colony.main_colony must be one of {sorted(_MAIN_CHOICES)}")
    if str(out["edge_distance"]) not in _EDGE_CHOICES:
        raise ValueError(f"colony.edge_distance must be one of {sorted(_EDGE_CHOICES)}")
    return out


def settings_hash(params: dict) -> str:
    """Hash of everything that changes the geometry (not the overlay switches)."""
    keys = ("pixel_size_um", "close_radius_um", "open_radius_um", "min_colony_area_um2",
            "main_colony", "edge_distance", "max_border_fraction", "trim_outside_circle")
    blob = json.dumps({k: params.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Geometry (pure functions: no IO)
# ---------------------------------------------------------------------------

def labels_and_centroids(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, centroids_yx)`` for a 2-D or 3-D label image in one pass."""
    flat = masks.ravel()
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return np.empty(0, np.int64), np.empty((0, 2), np.float64)
    lab = flat[nz].astype(np.int64)
    coords = np.unravel_index(nz, masks.shape)
    n_max = int(lab.max())
    counts = np.bincount(lab, minlength=n_max + 1).astype(np.float64)
    labels = np.flatnonzero(counts)
    labels = labels[labels > 0]
    ys = np.bincount(lab, weights=coords[-2], minlength=n_max + 1)[labels] / counts[labels]
    xs = np.bincount(lab, weights=coords[-1], minlength=n_max + 1)[labels] / counts[labels]
    return labels.astype(np.int64), np.column_stack([ys, xs])


def disk_close(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Binary closing by a disk, via two distance transforms (exact, radius-free cost)."""
    if radius_px <= 0:
        return mask.copy()
    dilated = ndi.distance_transform_edt(~mask) <= radius_px
    # Pixels outside the array are treated as foreground by the EDT, so the
    # erosion does not eat in from the image border.
    return ndi.distance_transform_edt(dilated) > radius_px


def disk_open(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Binary opening by a disk: severs bridges thinner than 2 × radius."""
    if radius_px <= 0:
        return mask.copy()
    eroded = ndi.distance_transform_edt(mask) > radius_px
    return ndi.distance_transform_edt(~eroded) <= radius_px


def true_boundary(mask: np.ndarray) -> np.ndarray:
    """Colony pixels touching background: *not* those on the image border."""
    return mask & ~ndi.binary_erosion(mask, border_value=1)


def border_fraction(mask: np.ndarray) -> float:
    """Fraction of the colony's perimeter that is the image border (cut off)."""
    on_border = (int(mask[0, :].sum()) + int(mask[-1, :].sum())
                 + int(mask[1:-1, 0].sum()) + int(mask[1:-1, -1].sum()))
    real = int(true_boundary(mask).sum())
    total = on_border + real
    return float(on_border / total) if total else 0.0


def _kasa(y: np.ndarray, x: np.ndarray):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = c + cx * cx + cy * cy
    return cy, cx, (np.sqrt(r2) if r2 > 0 else np.nan)


def fit_circle(points_yx: np.ndarray, n_gauss_newton: int = 5, max_iter: int = 10) -> dict:
    """Robust circle through boundary points ``(n, 2)`` in (y, x).

    Kåsa fit, then iterative trimming: points whose distance from the current
    circle deviates from the median by more than 2.5 robust SDs are set aside
    and the circle refitted, until the kept set stops changing. A tail of
    escaped cells or a bump in the rim pulls a plain least-squares circle
    towards it; the trimmed fit locks onto the dominant arc instead. Finished
    with a few Gauss-Newton steps on the geometric distance. Returns ``cy, cx,
    r, rms`` (pixels, rms over kept points) and ``n_points`` kept; NaNs when
    fewer than 10 points are available.
    """
    nan = {"cy": np.nan, "cx": np.nan, "r": np.nan, "rms": np.nan, "n_points": 0}
    if len(points_yx) < 10:
        return nan
    Y = points_yx[:, 0].astype(np.float64)
    X = points_yx[:, 1].astype(np.float64)
    cy, cx, r = _kasa(Y, X)
    if not np.isfinite(r):
        return nan
    keep = np.ones(Y.size, bool)
    for _ in range(max_iter):
        res = np.hypot(Y - cy, X - cx) - r
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med))
        if not mad > 0:
            break
        new_keep = np.abs(res - med) <= 2.5 * mad
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        cy, cx, r = _kasa(Y[keep], X[keep])
        if not np.isfinite(r):
            return nan
    y, x = Y[keep], X[keep]
    for _ in range(n_gauss_newton):
        d = np.hypot(y - cy, x - cx)
        d = np.where(d == 0, 1e-9, d)
        J = np.column_stack([-(y - cy) / d, -(x - cx) / d, -np.ones_like(d)])
        resid = d - r
        step, *_ = np.linalg.lstsq(J, -resid, rcond=None)
        cy, cx, r = cy + step[0], cx + step[1], r + step[2]
        if np.max(np.abs(step)) < 1e-3:
            break
    rms = float(np.sqrt(np.mean((np.hypot(y - cy, x - cx) - r) ** 2)))
    return {"cy": float(cy), "cx": float(cx), "r": float(abs(r)), "rms": rms,
            "n_points": int(len(y))}


def _circle_of(mask: np.ndarray, area: int) -> dict:
    by, bx = np.nonzero(true_boundary(mask))
    circle = fit_circle(np.column_stack([by, bx]))
    if not np.isfinite(circle["r"]):
        cy0, cx0 = ndi.center_of_mass(mask)
        circle = {"cy": float(cy0), "cx": float(cx0),
                  "r": float(np.sqrt(area / np.pi)), "rms": np.nan, "n_points": 0}
    return circle


def detect_colony(masks: np.ndarray, pixel_um: float, params: dict) -> dict:
    """Main-colony geometry for one label image (2-D, or 3-D projected to xy).

    Returns a dict with ``mask`` (bool, yx), ``nuclei`` (DataFrame) and ``meta``
    (JSON-able dict). Pure: no files are read or written.
    """
    masks_xy = masks.max(axis=0) if masks.ndim == 3 else masks
    labels, cyx = labels_and_centroids(masks)
    shape = masks_xy.shape
    px2 = pixel_um ** 2

    occ = masks_xy > 0
    footprint = disk_close(occ, float(params["close_radius_um"]) / pixel_um)
    footprint = ndi.binary_fill_holes(footprint)
    footprint = disk_open(footprint, float(params["open_radius_um"]) / pixel_um)

    comp, n_comp = ndi.label(footprint)
    meta: dict = {"pixel_size_um": float(pixel_um), "image_shape_yx": list(shape),
                  "n_nuclei_total": int(labels.size), "n_components": int(n_comp)}
    if n_comp == 0 or labels.size == 0:
        meta.update(status="empty", n_nuclei_colony=0)
        empty = pd.DataFrame({"label": labels, "y": cyx[:, 0] if len(cyx) else [],
                              "x": cyx[:, 1] if len(cyx) else []})
        for col in ("in_colony", "dist_edge_um", "r_center_um", "r_norm"):
            empty[col] = np.nan if col != "in_colony" else False
        return {"mask": np.zeros(shape, bool), "nuclei": empty, "meta": meta}

    areas = np.bincount(comp.ravel())
    areas[0] = 0
    min_area_px = float(params["min_colony_area_um2"]) / px2
    if params["main_colony"] == "center":
        cm = ndi.center_of_mass(footprint, comp, np.arange(1, n_comp + 1))
        dist = [np.hypot(c[0] - shape[0] / 2, c[1] - shape[1] / 2) for c in cm]
        big = [i for i in range(n_comp) if areas[i + 1] >= min_area_px] or list(range(n_comp))
        main_id = 1 + min(big, key=lambda i: dist[i])
    else:
        main_id = int(areas.argmax())
    mask = comp == main_id
    main_area = int(areas[main_id])
    others = areas.copy()
    others[main_id] = 0
    neighbors = others >= min_area_px
    neighbors[0] = False
    largest_other = int(others.max())

    # Circle fit on the true boundary only.
    circle = _circle_of(mask, main_area)
    # Cells that escaped the pattern, a tail, a smear, a bridge to debris, are
    # not part of the colony whose profile we want. Cut the mask at a tolerance
    # outside the fitted pattern circle and refit on what remains.
    trim = float(params.get("trim_outside_circle") or 0)
    trimmed_fraction = 0.0
    if trim > 0 and np.isfinite(circle["r"]) and circle["n_points"]:
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        inside = (yy - circle["cy"]) ** 2 + (xx - circle["cx"]) ** 2 <= (trim * circle["r"]) ** 2
        cut = mask & inside
        if cut.sum() < main_area:
            lab, n_lab = ndi.label(cut)
            if n_lab:
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                cut = lab == int(sizes.argmax())
                trimmed_fraction = 1.0 - cut.sum() / main_area
                mask, main_area = cut, int(cut.sum())
                circle = _circle_of(mask, main_area)
    frac_border = border_fraction(mask)

    method = str(params["edge_distance"])
    if method == "auto":
        method = "circle" if frac_border > float(params["max_border_fraction"]) else "mask"

    yi = np.clip(np.round(cyx[:, 0]).astype(int), 0, shape[0] - 1)
    xi = np.clip(np.round(cyx[:, 1]).astype(int), 0, shape[1] - 1)
    in_colony = mask[yi, xi]
    r_center_px = np.hypot(cyx[:, 0] - circle["cy"], cyx[:, 1] - circle["cx"])
    if method == "mask":
        dist_px = ndi.distance_transform_edt(mask)[yi, xi]
    else:
        dist_px = np.clip(circle["r"] - r_center_px, 0.0, None)
    dist_um = np.where(in_colony, dist_px * pixel_um, np.nan)
    radius_um = circle["r"] * pixel_um
    r_norm = np.clip(dist_um / radius_um, 0.0, 1.0) if radius_um > 0 else np.full_like(dist_um, np.nan)

    nuclei = pd.DataFrame({
        "label": labels,
        "y": cyx[:, 0].astype(np.float32),
        "x": cyx[:, 1].astype(np.float32),
        "in_colony": in_colony,
        "dist_edge_um": dist_um.astype(np.float32),
        "r_center_um": (r_center_px * pixel_um).astype(np.float32),
        "r_norm": r_norm.astype(np.float32),
    })

    # Nuclei carried by neighboring colonies, as opposed to stray single cells.
    in_neighbour = np.isin(comp[yi, xi], np.flatnonzero(neighbors))
    n_in = int(in_colony.sum())
    mask_area_um2 = main_area * px2
    meta.update({
        "status": "ok",
        "center_yx_px": [circle["cy"], circle["cx"]],
        "radius_um": float(radius_um),
        "diameter_um": float(2 * radius_um),
        "fit_rms_um": float(circle["rms"] * pixel_um) if np.isfinite(circle["rms"]) else None,
        "fit_rms_rel": (float(circle["rms"] / circle["r"])
                        if np.isfinite(circle["rms"]) and circle["r"] > 0 else None),
        "fit_points": int(circle["n_points"]),
        "area_um2": float(mask_area_um2),
        # Mask area over the fitted disk's area: ≈1 for a whole, round colony.
        "fill_ratio": float(main_area / (np.pi * circle["r"] ** 2)) if circle["r"] > 0 else None,
        "border_fraction": float(frac_border),
        "trimmed_fraction": float(trimmed_fraction),
        "edge_method": method,
        "n_neighbour_colonies": int(neighbors.sum()),
        "largest_neighbour_ratio": float(largest_other / main_area) if main_area else 0.0,
        "n_nuclei_colony": n_in,
        "n_nuclei_outside": int(labels.size - n_in),
        "n_nuclei_neighbours": int(in_neighbour.sum()),
        "outside_fraction": float((labels.size - n_in) / labels.size),
        "nuclei_density_per_mm2": float(n_in / (mask_area_um2 * 1e-6)) if mask_area_um2 else None,
        "max_dist_edge_um": float(np.nanmax(dist_um)) if n_in else None,
    })
    return {"mask": mask, "nuclei": nuclei, "meta": meta,
            "neighbors": np.isin(comp, np.flatnonzero(neighbors))}


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------

def pixel_size_for(result_dir: Path, params: dict) -> float:
    """µm per pixel: config override, else image metadata; never a guess."""
    meta = {}
    meta_path = Path(result_dir) / IMAGE_META_NAME
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    from_meta = meta.get("pixel_size_x_um")
    override = params.get("pixel_size_um")
    if override:
        if from_meta and abs(float(from_meta) - float(override)) / float(override) > 0.01:
            logger.warning("%s: colony.pixel_size_um=%.4g overrides the image's own %.4g µm/px",
                           Path(result_dir).name, float(override), float(from_meta))
        return float(override)
    if from_meta and float(from_meta) > 0:
        return float(from_meta)
    raise ValueError(
        f"{Path(result_dir).name}: the image metadata has no pixel size, so distances "
        f"cannot be expressed in µm. Set downstream.colony.pixel_size_um in config.yaml "
        f"(e.g. 0.65 for 20x on the Nikon)."
    )


def is_current(result_dir: Path, digest: str) -> bool:
    meta_path = Path(result_dir) / META_NAME
    if not (meta_path.exists() and (Path(result_dir) / TABLE_NAME).exists()):
        return False
    try:
        # Masks re-segmented after the colony was found make the geometry stale.
        if (Path(result_dir) / MASKS_NAME).stat().st_mtime > meta_path.stat().st_mtime:
            return False
        with open(meta_path) as f:
            return json.load(f).get("settings_hash") == digest
    except (OSError, json.JSONDecodeError):
        return False


def process_image(result_dir: Path, params: dict, dapi_index: int = 0,
                  force: bool = False) -> dict:
    """Detect the colony in one image's result directory; return its metadata.

    Skips (returning the stored metadata) when ``colony.json`` was written with
    the same settings, unless *force*.
    """
    result_dir = Path(result_dir)
    digest = settings_hash(params)
    if not force and is_current(result_dir, digest):
        with open(result_dir / META_NAME) as f:
            meta = json.load(f)
        meta["skipped"] = True
        return meta

    masks_path = result_dir / MASKS_NAME
    if not masks_path.exists():
        raise FileNotFoundError(f"{masks_path} missing: run the 'segment' step first.")
    pixel_um = pixel_size_for(result_dir, params)
    masks = tifffile.imread(masks_path)
    geo = detect_colony(masks, pixel_um, params)

    geo["nuclei"].to_parquet(result_dir / TABLE_NAME, index=False)
    tifffile.imwrite(result_dir / MASK_NAME, geo["mask"].astype(np.uint8),
                     metadata={"axes": "YX"})
    meta = dict(geo["meta"])
    meta["settings"] = {k: params.get(k) for k in DEFAULTS}
    meta["settings_hash"] = digest
    with open(result_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=2, default=float)

    if params.get("overlay", True):
        try:
            save_overlay(result_dir, masks, geo, dapi_index, int(params["overlay_max_px"]))
        except Exception:
            logger.exception("%s: colony overlay failed (geometry was saved)", result_dir.name)
    meta["skipped"] = False
    return meta


def worker(res_dir, params, dapi_index, force):
    """Picklable entry point for parallel workers: never raises.

    Returns ``(meta | None, error text | None)``. Lives here, not in
    :mod:`downstream.steps`, so a worker imports only this module.
    """
    try:
        return process_image(Path(res_dir), params, dapi_index, force), None
    except Exception as exc:                                  # noqa: BLE001
        import traceback
        return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# QC overlay
# ---------------------------------------------------------------------------

def _dapi_image(result_dir: Path, dapi_index: int, shape) -> np.ndarray | None:
    for name in (f"ch{dapi_index}_MIP.tif", f"ch{dapi_index}.tif"):
        path = result_dir / name
        if path.exists():
            img = tifffile.imread(path)
            img = img.max(axis=0) if img.ndim == 3 else img
            if img.shape == tuple(shape):
                return img.astype(np.float32)
    return None


def _outlines(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, bool)
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge & (labels > 0)


def render_overlay(dapi, masks_xy, geo, max_px: int = 2000):
    """Return a matplotlib figure of the colony QC overlay (used by the GUI too)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    mask, nuclei, meta = geo["mask"], geo["nuclei"], geo["meta"]
    step = max(1, int(np.ceil(max(mask.shape) / max_px)))
    sl = (slice(None, None, step), slice(None, None, step))

    if dapi is None:
        gray = np.full(mask.shape, 0.15, np.float32)
    else:
        lo, hi = np.percentile(dapi[sl], (1, 99.5))
        gray = np.clip((dapi - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat(gray[sl][..., None], 3, axis=2)
    outl = _outlines(masks_xy)[sl]
    rgb[outl] = to_rgb(style.OKABE_ITO["sky"])
    if "neighbors" in geo:
        nb_edge = true_boundary(geo["neighbors"])
        rgb[ndi.binary_dilation(nb_edge, iterations=step)[sl]] = to_rgb(style.OKABE_ITO["purple"])
    edge = ndi.binary_dilation(true_boundary(mask), iterations=2 * step)[sl]
    rgb[edge] = to_rgb(style.OKABE_ITO["orange"])

    fig, ax = plt.subplots(figsize=(7.5, 7.5 * mask.shape[0] / mask.shape[1]))
    ax.imshow(rgb, interpolation="nearest")
    handles = [Line2D([], [], color=style.OKABE_ITO["orange"], lw=3, label="colony edge (mask)"),
               Line2D([], [], color=style.OKABE_ITO["sky"], lw=1.5, label="nuclear outlines")]
    if meta.get("status") == "ok":
        cy, cx = meta["center_yx_px"]
        r = meta["radius_um"] / meta["pixel_size_um"]
        ax.add_patch(Circle((cx / step, cy / step), r / step, fill=False,
                            ec=style.OKABE_ITO["vermillion"], ls="--", lw=1.4))
        handles.append(Line2D([], [], color=style.OKABE_ITO["vermillion"], ls="--",
                              label=f"circle fit, Ø {meta['diameter_um']:.0f} µm"))
        out = nuclei[~nuclei["in_colony"].astype(bool)]
        if len(out):
            ax.scatter(out["x"] / step, out["y"] / step, s=14, marker="x",
                       c=style.OKABE_ITO["yellow"], linewidths=1.0)
            handles.append(Line2D([], [], color=style.OKABE_ITO["yellow"], marker="x", ls="",
                                  label=f"outside colony (n={len(out)})"))
        if meta.get("n_neighbour_colonies"):
            handles.append(Line2D([], [], color=style.OKABE_ITO["purple"], lw=2,
                                  label=f"neighboring colonies ({meta['n_neighbour_colonies']})"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85,
              facecolor="white", frameon=True)
    ax.set_axis_off()
    return fig


def save_overlay(result_dir: Path, masks: np.ndarray, geo: dict, dapi_index: int,
                 max_px: int = 2000) -> Path:
    import matplotlib.pyplot as plt
    masks_xy = masks.max(axis=0) if masks.ndim == 3 else masks
    dapi = _dapi_image(result_dir, dapi_index, masks_xy.shape)
    fig = render_overlay(dapi, masks_xy, geo, max_px)
    meta = geo["meta"]
    if meta.get("status") == "ok":
        fig.axes[0].set_title(
            f"{result_dir.name}\n{meta['n_nuclei_colony']} nuclei in colony · "
            f"edge distance from {meta['edge_method']} · "
            f"{100 * meta['border_fraction']:.0f}% of rim on image border", fontsize=8)
    out = result_dir / OVERLAY_NAME
    fig.savefig(out, dpi=150, bbox_inches="tight")
    # The contact sheet shrinks each overlay to a few hundred pixels, where the
    # title and legend become unreadable specks; it gets a copy without them.
    ax = fig.axes[0]
    ax.set_title("")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.savefig(result_dir / OVERLAY_THUMB_NAME, dpi=80, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out


__all__ = ["settings", "settings_hash", "detect_colony", "process_image", "fit_circle",
           "disk_close", "disk_open", "border_fraction", "pixel_size_for",
           "render_overlay", "labels_and_centroids", "MASK_NAME", "TABLE_NAME",
           "META_NAME", "OVERLAY_NAME", "OVERLAY_THUMB_NAME", "DEFAULTS"]
