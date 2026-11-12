"""
``assemble``: the only IO-heavy downstream step.

Walks every dataset's manifest, loads each image's ``per_nuclei_intensities.joblib``,
maps channel index → marker through that image's own panel, and writes the two
tables described in :mod:`downstream.data`. Everything after this step reads
parquet and runs in seconds.

Design notes
------------
* **One pair of tables per experiment**, not per dataset: ``dataset`` is a column,
  so pooling across a 2-D and a 3-D folder, or across antibody panels, is a
  ``groupby`` rather than a bespoke merge.
* **Per-image panels.** The channel→marker map is read per manifest row, so a
  folder holding several panels assembles correctly and a marker that appears in
  two panels keeps one name (canonicalized via ``marker_aliases``) while
  remaining distinguishable by ``stain_set``.
* **No per-cell Python.** Per-image arrays are collected into lists and
  concatenated once; condition columns are gathered with a single positional
  take over the manifest rather than repeated per cell.
* Images whose joblib is missing, empty, or malformed are logged and skipped:
  one bad image never aborts the assembly.
* **Colony geometry rides along.** Each image's ``colony_nuclei.parquet`` (from
  the ``colony`` step) is joined on the nucleus label, giving every cell
  ``in_colony``, ``dist_edge_um``, ``r_center_um`` and ``r_norm``; the image's
  ``colony.json`` becomes one row of ``colonies.csv``. An image without colony
  output is skipped and counted: it has no radial coordinate to contribute.
* **Signaling values ride along too.** Each image's ``signaling_nuclei.parquet``
  (from the ``signaling`` step) fills ``nuc_bgsub`` and ``nc_ratio`` for the
  channels it measured; every other row carries NaN there.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from selection import Selection

from . import colony as colony_mod
from .data import CELLS_NAME, CHANNEL, DAPI_NORM, NC_RATIO, NUC_BGSUB, RAW, Data
from .signaling import measure as sig_measure

logger = logging.getLogger(__name__)

JOBLIB_NAME = "per_nuclei_intensities.joblib"

# Per (field × channel) measurement-health scalars recorded by the extractor.
# Written beside the cell tables as `field_channel_qc.csv` and picked up by the
# `qc` step. Unlike the intensity tables this is NOT filtered by
# `exclude_markers`: a clipped DAPI channel is exactly the kind of thing you want
# reported, and DAPI is excluded from analysis everywhere else by design.
CHANNEL_QC_NAME = "field_channel_qc.csv"
CHANNEL_QC_FIELDS = ("saturated_fraction", "nuclei_saturated_fraction",
                     "background", "signal_to_background", "saturation_value")

COLONIES_NAME = "colonies.csv"

# Per-colony scalars copied from colony.json into colonies.csv.
COLONY_FIELDS = ("status", "radius_um", "diameter_um", "area_um2", "fill_ratio",
                 "fit_rms_um", "fit_rms_rel", "border_fraction", "trimmed_fraction", "edge_method",
                 "n_neighbour_colonies", "largest_neighbour_ratio", "n_nuclei_total",
                 "n_nuclei_colony", "n_nuclei_outside", "n_nuclei_neighbours",
                 "outside_fraction", "nuclei_density_per_mm2", "max_dist_edge_um",
                 "pixel_size_um")

# Manifest columns consumed structurally; everything else becomes a condition column.
_MANIFEST_ID = {"dataset", "plate_id", "image_path", "result_dir", "mode",
                "well", "row", "col", "seq_index", "stain_set"}


def _marker_map(row: pd.Series, aliases: dict) -> dict[int, str]:
    """Channel index → canonical marker name from a manifest row's ch{n}_marker columns."""
    out: dict[int, str] = {}
    for col, val in row.items():
        if col.startswith("ch") and col.endswith("_marker") and isinstance(val, str) and val:
            name = aliases.get(val.strip().lower(), val.strip())
            out[int(col[2:-7])] = name
    return out


def _find_manifest(directory: Path, basename: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{basename}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _read_manifest(path: Path) -> pd.DataFrame:
    return (pd.read_parquet(path) if path.suffix == ".parquet"
            else pd.read_csv(path))


def iter_manifests(datasets, cfg: dict):
    """Yield ``(dataset, manifest)`` per dataset, restricted to its image selection."""
    basename = cfg.get("manifest_basename", "manifest")
    for ds in datasets:
        manifest_path = _find_manifest(ds.directory, basename)
        if manifest_path is None:
            logger.warning("[%s] no %s.parquet/.csv in %s: skipping dataset. "
                           "Build it with 'python main.py manifest'.",
                           ds.name, basename, ds.directory)
            continue
        man = _read_manifest(manifest_path)
        sel = Selection.from_config(ds.cfg)
        if sel.active:
            n_all = len(man)
            man = sel.filter_manifest(man)
            logger.info("[%s] selection %s: %d of %d manifest row(s)",
                        ds.name, sel.describe(), len(man), n_all)
        yield ds, man.reset_index(drop=True)


def _colony_columns(res_dir: Path, labels: np.ndarray, n: int):
    """Per-cell colony columns aligned to the joblib's label order, plus colony.json."""
    table = res_dir / colony_mod.TABLE_NAME
    meta_path = res_dir / colony_mod.META_NAME
    if not (table.exists() and meta_path.exists()):
        return None, None
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    tab = pd.read_parquet(table)
    pos = pd.Index(tab["label"].to_numpy()).get_indexer(np.asarray(labels))
    found = pos >= 0
    cols = {}
    for col in ("dist_edge_um", "r_center_um", "r_norm"):
        vals = np.full(n, np.nan, np.float32)
        vals[found] = tab[col].to_numpy(np.float32)[pos[found]]
        cols[col] = vals
    inside = np.zeros(n, bool)
    inside[found] = tab["in_colony"].to_numpy(bool)[pos[found]]
    cols["in_colony"] = inside
    return cols, meta


def _signaling_columns(res_dir: Path, labels: np.ndarray, n: int) -> dict[int, dict]:
    """``channel → {nuc_bgsub, nc_ratio}`` aligned to the joblib's label order."""
    path = res_dir / sig_measure.TABLE_NAME
    if not path.exists():
        return {}
    try:
        tab = pd.read_parquet(path)
    except Exception:                                   # noqa: BLE001
        logger.exception("unreadable %s: signaling values skipped", path)
        return {}
    out: dict[int, dict] = {}
    for ch, part in tab.groupby("channel"):
        pos = pd.Index(part["label"].to_numpy()).get_indexer(np.asarray(labels))
        found = pos >= 0
        cols = {}
        for col in (NUC_BGSUB, NC_RATIO):
            vals = np.full(n, np.nan, np.float32)
            vals[found] = part[col].to_numpy(np.float32)[pos[found]]
            cols[col] = vals
        out[int(ch)] = cols
    return out


def assemble(datasets, cfg: dict, out_dir: Path) -> Data:
    """Build ``cells``/``intensities`` for every dataset and write them to *out_dir*.

    Parameters
    ----------
    datasets
        Resolved :class:`pipeline_config.Dataset` objects.
    cfg
        The ``downstream`` config block.
    out_dir
        Experiment-level analysis directory.
    """
    aliases = {str(k).strip().lower(): str(v)
               for k, v in (cfg.get("marker_aliases") or {}).items()}
    exclude = {str(m).strip().lower() for m in (cfg.get("exclude_markers") or [])} | {"dapi"}
    basename = cfg.get("manifest_basename", "manifest")

    cell_blocks: list[pd.DataFrame] = []
    inten_blocks: list[pd.DataFrame] = []
    channel_qc: list[dict] = []
    summary: list[dict] = []
    field_id = 0
    cell_offset = 0

    colony_rows: list[dict] = []
    for ds, man in iter_manifests(datasets, cfg):
        logger.info("[%s] %d image row(s)", ds.name, len(man))

        meta_cols = [c for c in man.columns
                     if c in _MANIFEST_ID or not (c.startswith("ch") and c.endswith("_marker"))]
        meta_cols = [c for c in meta_cols if c != "image_path"]

        rows_idx: list[int] = []          # manifest row per image kept
        dapi_l, area_l, cen_l = [], [], []
        geo_l: list[dict] = []
        n_cells_l: list[int] = []
        field_l: list[int] = []
        marker_frames: list[pd.DataFrame] = []
        n_missing = n_empty = n_no_colony = 0
        mode_l: list[str] = []
        mismatched: dict[str, int] = {}

        for pos, (_, row) in enumerate(man.iterrows()):
            jpath = ds.directory / str(row["result_dir"]) / JOBLIB_NAME
            if not jpath.exists():
                n_missing += 1
                logger.debug("[%s] missing %s", ds.name, jpath)
                continue
            try:
                d = joblib.load(jpath)
            except Exception:
                logger.exception("[%s] unreadable joblib %s: skipping image", ds.name, jpath)
                continue

            n = int(d.get("n_cells", 0))
            if n == 0:
                n_empty += 1
                continue

            geo, colony_meta = _colony_columns(jpath.parent, d.get("labels", np.arange(1, n + 1)), n)
            if geo is None:
                n_no_colony += 1
                logger.debug("[%s] no colony output for %s", ds.name, row["result_dir"])
                continue

            uid = np.arange(cell_offset, cell_offset + n, dtype=np.int64)
            # The mode the masks were really made in (recorded at extraction), not
            # the one the config asks for: a 2-D result in a folder now set to 3d
            # must not be reported as 3-D.
            from image_processing.utils import segmented_mode
            mode = str(segmented_mode(jpath.parent) or d.get("mode") or ds.seg_mode).lower()
            if mode != ds.seg_mode:
                mismatched[mode] = mismatched.get(mode, 0) + 1

            dapi = _as_float(d.get("dapi_per_cell"), n)
            area = _as_float(d.get("mask_sizes"), n)
            centroids = d.get("centroids")
            cen = (np.asarray(centroids, dtype=np.float32) if centroids is not None
                   else np.full((n, 2), np.nan, dtype=np.float32))
            if cen.shape[1] == 2:            # yx → pad a z column
                cen = np.column_stack([np.full(n, np.nan, np.float32), cen])

            ch2marker = _marker_map(row, aliases)
            raw_i = d.get("raw_intensities") or {}
            norm_i = d.get("dapi_normalized_intensities") or {}
            signal = _signaling_columns(jpath.parent, d.get("labels", np.arange(1, n + 1)), n)
            for ch, marker in ch2marker.items():
                if marker.strip().lower() in exclude or ch not in raw_i:
                    continue
                raw_vals = _as_float(raw_i[ch], n)
                norm_vals = _as_float(norm_i.get(ch), n)
                marker_frames.append(pd.DataFrame({
                    "cell_uid": uid,
                    "marker": marker,
                    CHANNEL: np.int16(ch),
                    RAW: raw_vals,
                    DAPI_NORM: norm_vals,
                    NUC_BGSUB: signal.get(ch, {}).get(NUC_BGSUB, np.full(n, np.nan, np.float32)),
                    NC_RATIO: signal.get(ch, {}).get(NC_RATIO, np.full(n, np.nan, np.float32)),
                }))

            # Measurement health, one row per channel present. `.get` throughout:
            # joblibs written before the extractor recorded these keys still
            # assemble, they just carry NaN here.
            for ch, st in (d.get("global_stats") or {}).items():
                if not isinstance(st, dict):
                    continue
                channel_qc.append({
                    "field_id": field_id, "dataset": ds.name,
                    "result_dir": str(row["result_dir"]),
                    "channel": int(ch), "marker": ch2marker.get(int(ch), f"ch{ch}"),
                    **{k: st.get(k, np.nan) for k in CHANNEL_QC_FIELDS},
                })

            colony_rows.append({
                "field_id": field_id, "dataset": ds.name, "seg_mode": mode,
                **{c: row[c] for c in meta_cols if c in row.index},
                **{k: colony_meta.get(k) for k in COLONY_FIELDS},
            })
            geo_l.append(geo)
            rows_idx.append(pos)
            n_cells_l.append(n)
            mode_l.append(mode)
            field_l.append(field_id)
            dapi_l.append(dapi)
            area_l.append(area)
            cen_l.append(cen)
            cell_offset += n
            field_id += 1

        for mode, k in mismatched.items():
            logger.warning("[%s] %d image(s) were segmented in %s but the config says %s: "
                           "they are analyzed (and labeled) as %s. Re-run the segmentation "
                           "job for this dataset to get %s results.", ds.name, k, mode,
                           ds.seg_mode, mode, ds.seg_mode)
        if n_no_colony:
            logger.warning("[%s] %d image(s) have no colony output and were skipped: "
                           "run the 'colony' step (or --force-step colony).",
                           ds.name, n_no_colony)
        if not rows_idx:
            logger.warning("[%s] no usable images (missing=%d, empty=%d, no colony=%d).",
                           ds.name, n_missing, n_empty, n_no_colony)
            continue

        # Condition/identity columns: one positional take over the manifest,
        # then repeated per cell: never a per-cell Python loop.
        counts = np.asarray(n_cells_l, dtype=np.int64)
        per_cell_row = np.repeat(np.asarray(rows_idx, dtype=np.int64), counts)
        meta = man.iloc[per_cell_row][meta_cols].reset_index(drop=True)

        cen_all = np.concatenate(cen_l, axis=0)
        block = pd.DataFrame({
            "field_id": np.repeat(np.asarray(field_l, dtype=np.int32), counts),
            "dataset": ds.name,
            "seg_mode": np.repeat(np.asarray(mode_l, dtype=object), counts),
            "dapi": np.concatenate(dapi_l),
            "area": np.concatenate(area_l),
            "cz": cen_all[:, 0],
            "cy": cen_all[:, 1],
            "cx": cen_all[:, 2],
        })
        for col in ("in_colony", "dist_edge_um", "r_center_um", "r_norm"):
            block[col] = np.concatenate([g[col] for g in geo_l])
        for col in meta.columns:
            if col not in block.columns:
                block[col] = meta[col].to_numpy()
        cell_blocks.append(block)
        inten_blocks.extend(marker_frames)

        summary.append({
            "dataset": ds.name, "seg_mode": "+".join(sorted(set(mode_l))),
            "images_in_manifest": len(man), "images_used": len(rows_idx),
            "images_missing_joblib": n_missing, "images_empty": n_empty,
            "images_without_colony": n_no_colony,
            "wells": int(man["well"].nunique()) if "well" in man else 0,
            "cells": int(counts.sum()),
        })
        logger.info("[%s] %d image(s) → %d cells (missing=%d, empty=%d)",
                    ds.name, len(rows_idx), int(counts.sum()), n_missing, n_empty)

    if not cell_blocks:
        raise RuntimeError(
            "assemble: no cells collected. Check that 'extract' and the downstream "
            "'colony' step have run and that each dataset directory holds a manifest."
        )

    cells = pd.concat(cell_blocks, ignore_index=True)
    cells.insert(0, "cell_uid", np.arange(len(cells), dtype=np.int64))
    intensities = pd.concat(inten_blocks, ignore_index=True)

    cells = _compact(cells)
    intensities["marker"] = intensities["marker"].astype("category")
    intensities[CHANNEL] = intensities[CHANNEL].astype(np.int16)
    intensities[RAW] = intensities[RAW].astype(np.float32)
    intensities[DAPI_NORM] = intensities[DAPI_NORM].astype(np.float32)
    for col in (NUC_BGSUB, NC_RATIO):
        intensities[col] = intensities[col].astype(np.float32)

    data = Data(cells=cells, intensities=intensities)
    data.validate()
    data.save(out_dir)
    pd.DataFrame(summary).to_csv(Path(out_dir) / "assemble_summary.csv", index=False)
    if channel_qc:
        pd.DataFrame(channel_qc).to_csv(Path(out_dir) / CHANNEL_QC_NAME, index=False)
    colonies = pd.DataFrame(colony_rows)
    colonies.to_csv(Path(out_dir) / COLONIES_NAME, index=False)

    logger.info("assemble: %d cells × %d marker(s) from %d field(s) across %d dataset(s)",
                len(cells), intensities["marker"].nunique(), cells["field_id"].nunique(),
                cells["dataset"].nunique())
    return data


def _as_float(arr, n: int) -> np.ndarray:
    """Coerce a per-cell array to float32 of length *n* (NaN when absent/ragged)."""
    if arr is None:
        return np.full(n, np.nan, dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32).ravel()
    if out.size != n:
        padded = np.full(n, np.nan, dtype=np.float32)
        padded[:min(n, out.size)] = out[:min(n, out.size)]
        return padded
    return out


def _compact(cells: pd.DataFrame) -> pd.DataFrame:
    """Shrink dtypes: low-cardinality object columns → category, floats → float32."""
    for col in cells.columns:
        s = cells[col]
        if col == "cell_uid":
            continue
        if s.dtype == object and s.nunique(dropna=False) <= max(64, len(s) // 100):
            cells[col] = s.astype("category")
        elif s.dtype == np.float64:
            cells[col] = s.astype(np.float32)
    return cells


__all__ = ["assemble", "iter_manifests", "JOBLIB_NAME", "CELLS_NAME", "CHANNEL_QC_NAME",
           "CHANNEL_QC_FIELDS", "COLONIES_NAME", "COLONY_FIELDS"]
