"""
Build a per-image manifest joining filenames → wells → experimental conditions.

The manifest is the single source of truth that links the generic, condition-
agnostic ``image_processing`` outputs to the biology of an experiment. It is a
table with **one row per input image**::

    plate_id | image_path | result_dir | mode | well | [row | col] | seq_index
             | <condition columns…> | stain_set | ch0_marker … chN_marker

``image_path`` and ``result_dir`` are stored *relative to the data root* (where
the manifest is written), so the table stays portable across machines. Downstream
analyses do ``manifest_dir / row.result_dir / "per_nuclei_intensities.joblib"``.

``row``/``col`` appear only when the well number is a physical plate coordinate:
that is, when the filename named the well (``generate``, ``spinning_disk``) or a
``predefined`` map opted in with ``well_numbering: plate``. Under the sequential
default of ``predefined`` mode the well is an acquisition-order index, there is
no plate position to report, and the layout figures skip themselves.

Multiple plates are handled by dropping a ``plate_map.yaml`` into each
acquisition folder; this builder finds them all and concatenates the result,
tagging every row with its ``plate_id``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from image_processing.directory_walker import OUTPUT_MARKER
from image_processing.image_loader import METADATA_NAME, result_dir_for
from image_processing.io import SUPPORTED_EXTENSIONS, get_extension

from . import parser, plate, plate_map
from .plate_map import PlateMap

logger = logging.getLogger(__name__)

PLATE_MAP_NAME = "plate_map.yaml"

# Stable leading column order; condition + stain columns are appended after.
_ID_COLUMNS = ["dataset", "plate_id", "image_path", "result_dir", "mode",
               "well", "row", "col", "seq_index"]


def find_plate_maps(root: Path) -> list[Path]:
    """Recursively locate every ``plate_map.yaml`` under ``root`` (sorted)."""
    return sorted(Path(root).rglob(PLATE_MAP_NAME))


def _normalize_extensions(file_types) -> set[str]:
    exts = file_types or SUPPORTED_EXTENSIONS
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}


def _iter_images(folder: Path, extensions: set[str], exclude_dirs=()):
    """
    Yield input image paths under ``folder``, skipping directories that cannot
    hold inputs.

    Four kinds of directory are pruned:

    * **result directories**: named after their image's stem
      (``result_dir_for``), so they are recognized from the file listing
      ``os.walk`` already returned, with no filesystem calls at all;
    * anything holding a :data:`OUTPUT_MARKER` (the fallback for the above);
    * any sub-folder owning its own ``plate_map.yaml``: it belongs to its own plate;
    * **dot-directories and names in** ``input.exclude_dirs``: exported JPEGs,
      analysis output, and similar sibling folders.

    That last one matters twice over: descending into a folder of exports is by
    far the most expensive thing this walk can do on a network share, and if
    ``file_types`` includes ``.tif`` those exports would otherwise be ingested as
    input images.
    """
    folder = Path(folder)
    exclude = {str(d).strip("/") for d in (exclude_dirs or ())}
    for dirpath, dirnames, filenames in os.walk(folder):
        here = Path(dirpath)
        image_stems = {Path(f).stem for f in filenames if get_extension(here / f) in extensions}
        keep = []
        for d in dirnames:
            if d in image_stems or d in exclude or d.startswith("."):
                continue
            if (here / d / OUTPUT_MARKER).exists() or (here / d / PLATE_MAP_NAME).exists():
                continue
            keep.append(d)
        dirnames[:] = keep
        for fname in filenames:
            fpath = here / fname
            if get_extension(fpath) in extensions:
                yield fpath


def _channel_count(result_dir: Path) -> int | None:
    """Number of channels saved for an image, from its ``image_metadata.json`` if present."""
    meta_path = result_dir / METADATA_NAME
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta.get("n_channels_saved", meta.get("n_channels")))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _build_row(img: Path, pm: PlateMap, root: Path, dataset: str | None = None) -> dict | None:
    info = parser.parse_image(img, pm.mode)
    seq_index = info["seq_index"]
    if pm.is_spinning_disk:
        label = info["well_label"]
        well = plate.label_to_well(label, pm.cols) if label else None
    elif pm.mode == "generate":
        well = info["well"]
    else:
        well = pm.well_for_seq(seq_index)

    if well is None:
        logger.warning(
            "[%s] could not map %s to a well (mode=%s, seq=%s): skipping.",
            pm.plate_id, img.name, pm.mode, seq_index,
        )
        return None

    try:
        plate.validate_wells([well], pm.rows, pm.cols)
    except ValueError as exc:
        logger.warning("[%s] %s: %s; skipping.", pm.plate_id, img.name, exc)
        return None

    res_dir = result_dir_for(img)
    stain_set, stains = pm.resolve_stain(well, seq_index)

    record: dict = {
        "dataset": dataset or pm.plate_id,
        "plate_id": pm.plate_id,
        "image_path": _relpath(img, root),
        "result_dir": _relpath(res_dir, root),
        "mode": pm.mode,
        "well": int(well),
        "seq_index": seq_index,
        "stain_set": stain_set,
    }

    # (row, col) only exist when the well number is a real plate coordinate:
    # i.e. the filename named the well. Under sequential numbering the well is an
    # acquisition-order index, so a derived row/col would be a fabricated plate
    # position; the layout figures then correctly skip themselves instead of
    # drawing a geometry nobody imaged.
    if pm.has_plate_positions:
        row_n, col_n = plate.well_to_rowcol(well, pm.cols)
        record["row"] = int(row_n)
        record["col"] = int(col_n)
    record.update(pm.conditions_for_well(well))

    # Reserved per-marker control declarations. Always emitted (empty when unused)
    # so the column set does not depend on which wells happen to declare them.
    record[plate_map.CONTROL_FOR_KEY] = pm.control_for_well(well)
    record[plate_map.POSITIVE_FOR_KEY] = pm.positive_for_well(well)

    for ch_idx, marker in stains.items():
        record[f"ch{ch_idx}_marker"] = marker

    # QC: warn if the declared stains exceed the channels actually saved.
    n_ch = _channel_count(res_dir)
    if n_ch is not None and stains and max(stains) >= n_ch:
        logger.warning(
            "[%s] %s declares channel %d but only %d channel(s) were saved.",
            pm.plate_id, img.name, max(stains), n_ch,
        )

    if well not in pm.wells:
        logger.warning(
            "[%s] well %d (from %s) has no entry in the plate map's 'wells'.",
            pm.plate_id, well, img.name,
        )
    return record


def build_manifest(
    cfg: dict,
    data_dir: str | None = None,
    out_basename: str | None = None,
    plate_map_path: str | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """
    Discover plate maps, parse every image, and write ``<basename>.parquet`` and
    ``<basename>.csv`` at the data root. Returns the manifest DataFrame.
    """
    root = Path(data_dir or cfg.get("input", {}).get("directory", ".")).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    extensions = _normalize_extensions(cfg.get("input", {}).get("file_types"))
    exclude_dirs = cfg.get("input", {}).get("exclude_dirs") or []

    if plate_map_path:
        maps = [PlateMap(plate_map_path)]
    else:
        map_paths = find_plate_maps(root)
        if not map_paths:
            raise FileNotFoundError(
                f"No '{PLATE_MAP_NAME}' found under {root}. Author one per acquisition "
                f"folder (see the example in the README)."
            )
        maps = [PlateMap(p) for p in map_paths]

    logger.info("Building manifest from %d plate map(s) under %s", len(maps), root)

    rows: list[dict] = []
    for pm in maps:
        folder = pm.path.parent
        n_before = len(rows)
        for img in _iter_images(folder, extensions, exclude_dirs):
            rec = _build_row(img, pm, root, dataset)
            if rec is not None:
                rows.append(rec)
        logger.info(
            "  [%s] %s (%s mode): %d image(s) annotated",
            pm.plate_id, _relpath(folder, root) or ".", pm.mode, len(rows) - n_before,
        )
        if pm.sweeps:
            logger.info("  [%s] stain sweeps: %s", pm.plate_id, ", ".join(
                f"{name} Seq{spec['seq_range'][0]}-{spec['seq_range'][1]}"
                for name, spec in pm.sweeps.items()))

    if not rows:
        raise RuntimeError("No images could be annotated: check filenames and plate maps.")

    df = pd.DataFrame(rows)
    df = _order_columns(df).sort_values(["plate_id", "well", "seq_index", "image_path"],
                                        na_position="last").reset_index(drop=True)

    basename = out_basename or cfg.get("experiment", {}).get("manifest_basename", "manifest")
    parquet_path = root / f"{basename}.parquet"
    csv_path = root / f"{basename}.csv"
    _write_parquet(df, parquet_path)
    df.to_csv(csv_path, index=False)

    logger.info("Manifest written: %d row(s) → %s (+ .csv)", len(df), parquet_path)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relpath(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    ids = [c for c in _ID_COLUMNS if c in df.columns]
    marker_cols = sorted(
        (c for c in df.columns if c.startswith("ch") and c.endswith("_marker")),
        key=lambda c: int(c[2:-7]),
    )
    middle = [c for c in df.columns if c not in ids and c not in marker_cols and c != "stain_set"]
    tail = (["stain_set"] if "stain_set" in df.columns else []) + marker_cols
    return df[ids + middle + tail]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:
        logger.warning("Could not write parquet (%s). CSV is still written.", exc)


__all__ = ["build_manifest", "find_plate_maps", "PLATE_MAP_NAME"]
