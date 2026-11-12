"""
Per-nucleus signaling measurements in one image: nucleus, cytoplasmic ring, background.

Generic pieces, each usable on its own (2-D masks, or 3-D masks with the ring
grown plane by plane):

:func:`nucleus_core`
    The nucleus minus its 1-pixel rim (in-plane). The rim mixes nuclear and
    cytoplasmic signal at any realistic resolution.
:func:`ring_labels`
    A band ``ring_inner_um`` to ``ring_outer_um`` outside each nucleus, grown
    with :func:`skimage.segmentation.expand_labels`, so growth stops where two
    neighbors meet: no pixel is shared and no ring enters another nucleus. In
    3-D the ring grows within each z-plane, so a nucleus above or below can never
    enter it.
:func:`background_level`
    The median of cell-free pixels in a band just outside the cells
    (``background: local``, the default): from ``free_distance_um`` to
    ``free_distance_um + local_band_um`` beyond the cell-covered area (nuclei
    within ``free_distance_um`` of each other, holes filled, plus the colony mask
    when there is one). On a micropattern that is the passivated surface right
    around the colony, so vignetting, a neighbor's debris or a dim image corner
    do not enter. Gaps inside a colony (unsegmented or sparse cells) never count
    as background: they are inside the filled area. With too few band pixels it
    falls back to every cell-free pixel outside that area (``background: image``),
    then to the 0.5th percentile of the image (the camera offset).
:func:`cell_body_mask`
    Pixels clearly above the cell-free level (smoothed channel above background +
    ``body_k`` × MAD of the cell-free pixels). Ring pixels outside it are gaps
    between cells or the empty surface beyond the colony edge, and would drag
    the cytoplasm toward background and inflate N:C exactly at the rim, where
    micropattern signaling is most interesting. ``body_k: 0`` keeps every ring pixel.

Per nucleus and channel the result holds ``nuc`` (core median), ``ring`` (median
of the ring pixels inside cell bodies), ``n_ring_px``, the image ``background``
and the two readouts::

    nuc_bgsub = nuc − background
    nc_ratio  = (nuc − background) / (ring − background)

``nc_ratio`` is NaN when the ring has fewer than ``min_ring_px`` pixels or does
not clear the background (a cytoplasm at background has no defined ratio).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

TABLE_NAME = "signaling_nuclei.parquet"
META_NAME = "signaling.json"
QC_NAME = "signaling_rings.png"

DEFAULTS = {
    "ring_inner_um": 1.0,          # ring starts this far outside the nucleus
    "ring_outer_um": 4.0,          # ...and ends here
    "free_distance_um": 15.0,      # cell-free = farther than this from every nucleus
    "min_free_px": 2000,           # fewer cell-free pixels → next background method
    "background": "local",         # local (band around the cells) | image (all cell-free)
    "local_band_um": 60.0,         # width of the local band
    "body_k": 3.0,                 # ring pixels must clear background + k·MAD; 0 = keep all
    "min_ring_px": 10,             # fewer ring pixels → no N:C for that nucleus
    "qc_images": True,             # write signaling_rings.png per image
}

COLUMNS = ["label", "channel", "nuc", "ring", "n_ring_px", "background", "nuc_bgsub",
           "nc_ratio"]


def settings_hash(params: dict, channels: dict) -> str:
    keys = ("ring_inner_um", "ring_outer_um", "free_distance_um", "min_free_px", "body_k",
            "min_ring_px", "background", "local_band_um")
    blob = json.dumps({"p": {k: params.get(k) for k in keys},
                       "ch": {str(k): v for k, v in sorted(channels.items())}},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Generic pieces (pure functions)
# ---------------------------------------------------------------------------

def _footprint(ndim: int) -> tuple:
    return (1, 3, 3) if ndim == 3 else (3, 3)


def nucleus_core(labels: np.ndarray) -> np.ndarray:
    """Each label without its 1-pixel in-plane rim (a label with no interior keeps all)."""
    fp = _footprint(labels.ndim)
    same = ndi.minimum_filter(labels, fp) == ndi.maximum_filter(labels, fp)
    core = np.where(same, labels, 0)
    lost = np.setdiff1d(np.unique(labels), np.unique(core))
    lost = lost[lost > 0]
    if lost.size:                       # tiny nuclei: fall back to the whole mask
        core = np.where(np.isin(labels, lost), labels, core)
    return core


def ring_labels(labels: np.ndarray, inner_px: float, outer_px: float) -> np.ndarray:
    """Band from ``inner_px`` to ``outer_px`` outside every nucleus, split between neighbors."""
    from skimage.segmentation import expand_labels

    def one(plane):
        if not plane.any():
            return np.zeros_like(plane)
        grown_out = expand_labels(plane, outer_px)
        grown_in = expand_labels(plane, inner_px) if inner_px > 0 else plane
        return np.where(grown_in == 0, grown_out, 0)

    if labels.ndim == 3:
        return np.stack([one(z) for z in labels])
    return one(labels)


def free_mask(labels: np.ndarray, valid: np.ndarray, dist_px: float) -> np.ndarray:
    """Cell-free pixels: holding data and farther than *dist_px* from every nucleus."""
    if labels.ndim == 3:
        return np.stack([free_mask(z, v, dist_px) for z, v in zip(labels, valid)])
    return (ndi.distance_transform_edt(labels == 0) > dist_px) & valid


BACKGROUNDS = ("local", "image")


def covered_area(labels: np.ndarray, dist_px: float, colony: np.ndarray | None = None):
    """In-plane area the cells cover: within *dist_px* of a nucleus, holes filled,
    plus *colony* (2-D) when given. 2-D, from the projection of 3-D labels."""
    yx = labels.max(axis=0) if labels.ndim == 3 else labels
    near = ndi.distance_transform_edt(yx == 0) <= dist_px
    if colony is not None and colony.shape == near.shape:
        near |= colony.astype(bool)
    return ndi.binary_fill_holes(near)


def background_masks(labels: np.ndarray, valid: np.ndarray, dist_px: float, band_px: float,
                     colony: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """``{"local": band around the cells, "image": every cell-free pixel}`` (data pixels
    outside the covered area); 3-D masks broadcast the in-plane area over z."""
    cover = covered_area(labels, dist_px, colony)
    outside = ndi.distance_transform_edt(~cover)
    local, image = (outside > 0) & (outside <= band_px), ~cover
    if labels.ndim == 3:
        local, image = (np.broadcast_to(m, labels.shape) for m in (local, image))
    free = free_mask(labels, valid, dist_px)
    return {"local": local & free, "image": image & free}


def background_level(img: np.ndarray, free, valid: np.ndarray,
                     min_free_px: int, order=("local", "image")) -> tuple[float, float, str]:
    """``(level, mad, method)``: median of the first background region in *order*
    with at least *min_free_px* pixels, else the camera offset.

    *free* is a mask or a ``{method: mask}`` dict (:func:`background_masks`).
    """
    regions = free if isinstance(free, dict) else {"cell_free": free}
    names = [m for m in order if m in regions] if isinstance(free, dict) else ["cell_free"]
    for name in names:
        vals = img[regions[name]]
        if vals.size >= min_free_px:
            med = float(np.median(vals))
            return med, float(1.4826 * np.median(np.abs(vals - med))), name
    data = img[valid] if valid.any() else img.ravel()
    off = float(np.percentile(data, 0.5))
    # No cell-free pixels (a field packed with cells): the spread must describe the
    # camera noise around the offset, so it comes from the dimmest pixels. The MAD
    # of the whole image would be the spread of the cells themselves, and a cell-body
    # mask at offset + k·MAD would then reject every cytoplasmic ring pixel.
    dim = data[data <= np.percentile(data, 5)]
    mad = float(1.4826 * np.median(np.abs(dim - np.median(dim)))) if dim.size else 0.0
    return off, mad, "offset"


def cell_body_mask(img: np.ndarray, level: float, mad: float, k: float) -> np.ndarray:
    """Pixels whose smoothed (in-plane) intensity clears ``level + k·mad``."""
    if k <= 0:
        return np.ones(img.shape, bool)
    sigma = (0, 1.0, 1.0) if img.ndim == 3 else 1.0
    return ndi.gaussian_filter(img, sigma) > level + k * mad


def label_medians(img: np.ndarray, lab: np.ndarray, ids: np.ndarray):
    """``(median, n_px)`` of *img* over each label in *ids*; NaN where a label is empty."""
    n = int(max(int(lab.max()), int(ids.max()))) + 1
    m = lab > 0
    cnt = np.bincount(lab[m], minlength=n)[ids]
    med = np.full(ids.size, np.nan)
    have = cnt > 0
    if have.any():
        med[have] = ndi.median(img, labels=lab, index=ids[have])
    return med, cnt


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------

def _bbox(labels: np.ndarray, margin: int) -> tuple:
    """Crop around every nucleus plus a margin (in-plane), whole stack in z."""
    yx = labels.max(axis=0) if labels.ndim == 3 else labels
    ys, xs = np.nonzero(yx)
    if ys.size == 0:
        return tuple(slice(None) for _ in labels.shape)
    y0, y1 = max(0, ys.min() - margin), min(yx.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(yx.shape[1], xs.max() + margin + 1)
    sl = (slice(y0, y1), slice(x0, x1))
    return ((slice(None),) + sl) if labels.ndim == 3 else sl


def measure(labels: np.ndarray, images: dict[int, np.ndarray], ring_channels: set[int],
            pixel_um: float, params: dict, colony: np.ndarray | None = None
            ) -> tuple[pd.DataFrame, dict, dict]:
    """Measure every channel in *images* (``channel → array``, same shape as *labels*).

    Rings are only grown when *ring_channels* is non-empty. *colony*: the 2-D colony
    mask, if any (added to the cell-covered area). Returns the per-nucleus table,
    per-channel background details, and the label images used (for QC).
    """
    labels = labels.astype(np.int32, copy=False)
    inner = float(params["ring_inner_um"]) / pixel_um
    outer = float(params["ring_outer_um"]) / pixel_um
    free_px = float(params["free_distance_um"]) / pixel_um
    band_px = float(params.get("local_band_um", DEFAULTS["local_band_um"])) / pixel_um
    method = str(params.get("background", DEFAULTS["background"]))
    if method not in BACKGROUNDS:
        raise ValueError(f"signaling.background must be one of {BACKGROUNDS}, got {method!r}")
    order = BACKGROUNDS[BACKGROUNDS.index(method):]
    margin = int(np.ceil(max(outer, free_px + band_px) + 20.0 / pixel_um))
    sl = _bbox(labels, margin)
    lab = labels[sl]
    col = colony[sl[-2:]] if colony is not None and colony.shape == labels.shape[-2:] else None
    ids = np.unique(lab)
    ids = ids[ids > 0]
    empty = pd.DataFrame(columns=COLUMNS)
    if ids.size == 0:
        return empty, {}, {}

    core = nucleus_core(lab)
    ring = ring_labels(lab, inner, outer) if ring_channels else None
    rows, bg_info = [], {}
    ring_body_shown = None
    for ch, full in images.items():
        img = full[sl].astype(np.float32, copy=False)
        valid = img > 0                    # stitched images pad with zeros
        regions = background_masks(lab, valid, free_px, band_px, col)
        level, mad, used = background_level(img, regions, valid, int(params["min_free_px"]),
                                            order)
        bg_info[int(ch)] = {"background": level, "mad": mad, "method": used,
                            "n_free_px": int(regions.get(used, regions["image"]).sum())}
        nuc, _ = label_medians(img, core, ids)
        part = {"label": ids, "channel": np.full(ids.size, int(ch), np.int16), "nuc": nuc,
                "ring": np.full(ids.size, np.nan), "n_ring_px": np.zeros(ids.size, np.int32),
                "background": np.full(ids.size, level)}
        part["nuc_bgsub"] = nuc - level
        part["nc_ratio"] = np.full(ids.size, np.nan)
        if ring is not None and int(ch) in ring_channels:
            body = cell_body_mask(img, level, mad, float(params["body_k"]))
            ring_body = np.where(body & valid, ring, 0)
            rmed, rcnt = label_medians(img, ring_body, ids)
            part["ring"], part["n_ring_px"] = rmed, rcnt.astype(np.int32)
            cyto = rmed - level
            ok = (rcnt >= int(params["min_ring_px"])) & (cyto > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                part["nc_ratio"] = np.where(ok, (nuc - level) / cyto, np.nan)
            if ring_body_shown is None:
                ring_body_shown = (int(ch), ring_body)
        rows.append(pd.DataFrame(part))
    table = pd.concat(rows, ignore_index=True)[COLUMNS]
    maps = {"slice": sl, "core": core, "ring": ring_body_shown}
    return table, bg_info, maps


def channel_files(result_dir: Path, channels, masks_ndim: int) -> dict[int, Path]:
    """The channel image matching the masks: z-stack for 3-D masks, else the MIP / 2-D file."""
    from image_processing.utils import load_image_metadata, resolve_channel_path
    try:
        meta = load_image_metadata(Path(result_dir))
    except FileNotFoundError:
        meta = {}
    is_zstack = bool(meta.get("is_zstack", False))
    mode = "3d" if masks_ndim == 3 else "2d"
    return {int(c): resolve_channel_path(Path(result_dir), int(c), mode, is_zstack)
            for c in channels}


def is_current(result_dir: Path, digest: str) -> bool:
    meta_path = Path(result_dir) / META_NAME
    if not (meta_path.exists() and (Path(result_dir) / TABLE_NAME).exists()):
        return False
    try:
        with open(meta_path) as f:
            return json.load(f).get("settings_hash") == digest
    except (OSError, json.JSONDecodeError):
        return False


def process_image(result_dir: Path, channels: dict[int, str], params: dict,
                  pixel_um: float, force: bool = False) -> dict:
    """Measure one image's signaling channels (``channel → readout``) and write the table."""
    import tifffile
    result_dir = Path(result_dir)
    digest = settings_hash(params, channels)
    if not force and is_current(result_dir, digest):
        return {"skipped": True}
    masks = tifffile.imread(result_dir / "nuclear_masks.tif")
    colony_path = result_dir / "colony_mask.tif"
    colony = tifffile.imread(colony_path) > 0 if colony_path.exists() else None
    files = channel_files(result_dir, channels, masks.ndim)
    images = {}
    for ch, path in files.items():
        if not path.exists():
            logger.warning("%s: channel file %s missing; signaling channel %d skipped",
                           result_dir.name, path.name, ch)
            continue
        img = tifffile.imread(path)
        if img.shape != masks.shape:
            if img.ndim == masks.ndim + 1:
                img = img.max(axis=0)
            if img.shape != masks.shape:
                logger.warning("%s: channel %d is %s but the masks are %s; skipped",
                               result_dir.name, ch, img.shape, masks.shape)
                continue
        images[ch] = img
    ring_channels = {c for c, r in channels.items() if r == "nc" and c in images}
    table, bg, maps = measure(masks, images, ring_channels, pixel_um, params, colony)
    table.to_parquet(result_dir / TABLE_NAME, index=False)
    meta = {"settings_hash": digest, "channels": {str(k): v for k, v in channels.items()},
            "pixel_size_um": pixel_um, "background": {str(k): v for k, v in bg.items()},
            "n_nuclei": int(table["label"].nunique()) if len(table) else 0}
    if ring_channels and len(table):
        sub = table[table["channel"].isin(ring_channels)]
        meta["fraction_with_nc"] = float(np.isfinite(sub["nc_ratio"]).mean())
    with open(result_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=2)
    if params.get("qc_images", True) and maps.get("ring") is not None:
        ch, ring_body = maps["ring"]
        try:
            save_ring_qc(images[ch][maps["slice"]], maps["core"], ring_body,
                         result_dir / QC_NAME)
        except Exception:                               # noqa: BLE001
            logger.debug("ring QC image failed for %s", result_dir.name, exc_info=True)
    return meta


def background_table(entries) -> pd.DataFrame:
    """Per image and channel: the background, how it was found, and whether it is an
    outlier among images of the same dataset, stain set and channel (more than 5
    robust SD, 1.4826 × MAD, from their median; at least 10 % of that median apart).
    Stain sets differ in antibodies and exposure, so their backgrounds may too.

    *entries*: ``(dataset, result_dir[, stain_set])``; images without
    ``signaling.json`` are skipped.
    """
    rows = []
    for entry in entries:
        dataset, res_dir = entry[0], entry[1]
        group = str(entry[2]) if len(entry) > 2 else ""
        path = Path(res_dir) / META_NAME
        if not path.exists():
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for ch, info in (meta.get("background") or {}).items():
            rows.append({"dataset": str(dataset), "stain_set": group,
                         "result_dir": Path(res_dir).name, "channel": int(ch), **info})
    cols = ["dataset", "stain_set", "result_dir", "channel", "background", "mad", "method", "n_free_px",
            "outlier"]
    if not rows:
        return pd.DataFrame(columns=cols)
    table = pd.DataFrame(rows)
    grp = table.groupby(["dataset", "stain_set", "channel"])["background"]
    med = grp.transform("median")
    spread = 1.4826 * grp.transform(lambda s: float(np.median(np.abs(s - s.median()))))
    dev = (table["background"] - med).abs()
    table["outlier"] = (dev > 5 * spread) & (dev > 0.1 * med.abs())
    return table.reindex(columns=cols)


def worker(res_dir, channels, params, pixel_um, force):
    """Picklable entry point for parallel workers: never raises."""
    try:
        return process_image(Path(res_dir), channels, params, pixel_um, force), None
    except Exception as exc:                                  # noqa: BLE001
        import traceback
        return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


def save_ring_qc(img: np.ndarray, core: np.ndarray, ring: np.ndarray, out_path: Path,
                 size: int = 360) -> None:
    """A crop at the colony middle: nucleus cores and the rings that were measured."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    from .. import style
    from matplotlib.colors import to_rgb

    if img.ndim == 3:                     # the plane with the most nuclei
        z = int(np.argmax((core > 0).sum(axis=(1, 2))))
        img, core, ring = img[z], core[z], ring[z]
    ys, xs = np.nonzero(core)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if ys.size else (img.shape[0] // 2,
                                                                         img.shape[1] // 2)
    h = size // 2
    y0, x0 = max(0, cy - h), max(0, cx - h)
    crop = (slice(y0, y0 + size), slice(x0, x0 + size))
    im = img[crop]
    lo, hi = np.percentile(im, (1, 99.7)) if im.size else (0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax in axes:
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.set_axis_off()
    for m, color in ((find_boundaries(core[crop], mode="inner"), style.NUCLEUS_OUTLINE),
                     (find_boundaries(ring[crop], mode="inner"), style.RING_OUTLINE)):
        rgba = np.zeros((*m.shape, 4))
        rgba[m] = (*to_rgb(color), 1.0)
        axes[1].imshow(rgba)
    axes[0].set_title("signaling channel", fontsize=9)
    axes[1].set_title("nucleus core (orange), cytoplasmic ring in cell bodies (sky blue)",
                      fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


__all__ = ["DEFAULTS", "TABLE_NAME", "META_NAME", "QC_NAME", "COLUMNS", "BACKGROUNDS",
           "nucleus_core", "ring_labels", "free_mask", "covered_area", "background_masks",
           "background_level", "cell_body_mask", "label_medians",
           "measure", "process_image", "worker", "background_table", "is_current", "settings_hash",
           "channel_files", "save_ring_qc"]
