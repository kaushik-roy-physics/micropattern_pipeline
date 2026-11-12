"""
Quality control: run this before believing any profile.

In this pipeline one imaged field is one colony, so every check is a check on a
colony. Three questions:

1. **Is the colony geometry trustworthy?** :func:`colony_qc`. A radial profile is
   only as good as the edge it is measured from. Flags:

   ``failed``      no colony was found (empty field, segmentation failure)
   ``clipped``     more than ``colony.max_border_fraction`` of the rim is on the
                   image border: the edge distance came from the fitted circle
   ``size``        diameter more than ``qc.max_diameter_deviation`` away from the
                   expected pattern diameter (``colony.expected_diameter_um`` or a
                   plate-map column named by ``colony.diameter_column``): a
                   detached, torn or merged colony
   ``shape``       not round: mask fills less than ``qc.min_fill_ratio`` of its
                   fitted circle (unclipped colonies only), or the rim deviates
                   from the circle by more than ``qc.max_fit_rms_rel`` × R
   ``neighbor``   a neighboring colony in the field is at least
                   ``qc.max_neighbour_ratio`` × the main one's area, so "the main
                   colony" may be the wrong one
   ``outside``     more than ``qc.max_outside_fraction`` of the nuclei lie outside
                   the main colony
   ``low_count``   fewer than ``qc.min_nuclei_per_colony`` nuclei in the colony
   ``field``       nucleus count, DAPI or nuclear size is a robust outlier among
                   the colonies of the same condition (:func:`field_qc`)

2. **Did segmentation and staining behave?** :func:`field_qc`: count, nuclear
   size and DAPI per colony, judged against colonies of the same condition.

3. **Did each marker channel behave, colony by colony?** :func:`marker_field_qc`
   compares each colony's marker median against **the other colonies of its own
   well** (one staining reaction on one population, so disagreement there is
   technical) and carries the extractor's saturation and signal-to-background
   flags.

Flagged colonies are **reported, not silently dropped.** ``qc.exclude_flags``
names the flags that remove a colony from the analysis (default ``[failed]``:
a colony that was not found has no profile to contribute); ``[all]`` removes any
flagged colony. Excluded colonies are listed in ``qc_colonies.csv`` and in the
report.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .aggregate import condition_columns, condition_label
from .data import DAPI_NORM, RAW, Data

logger = logging.getLogger(__name__)

# Columns identifying the pool a field is compared against: its own well, in its
# own panel, on its own plate.
_WELL_KEYS = ("dataset", "plate_id", "stain_set", "well")


def mad_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score: (x − median) / (1.4826 · MAD). Zero MAD → all zeros."""
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        return np.zeros_like(values)
    return (values - median) / scale


def field_qc(data: Data, cfg: dict) -> pd.DataFrame:
    """Per-field segmentation/staining metrics with robust outlier flags."""
    qc_cfg = cfg.get("qc") or {}
    z_max = float(qc_cfg.get("outlier_z", 3.5))
    min_cells = int(qc_cfg.get("min_cells_per_field", 20))

    cond_cfg = cfg.get("conditions") or {}
    keys = condition_columns(data.cells, cond_cfg)
    group_cols = [c for c in ("dataset", "seg_mode", "plate_id", "stain_set", "well", "row", "col")
                  if c in data.cells.columns]

    grouped = data.cells.groupby("field_id", observed=True)
    qc = grouped.agg(
        n_cells=("cell_uid", "size"),
        dapi_median=("dapi", "median"),
        area_median=("area", "median"),
        area_cv=("area", lambda s: float(np.nanstd(s) / np.nanmean(s))
                 if np.nanmean(s) else np.nan),
    ).reset_index()

    meta = grouped[group_cols + keys].first().reset_index()
    qc = qc.merge(meta, on="field_id", how="left")
    qc["condition"] = condition_label(qc, keys)

    # Robust z within each condition: comparing a field to its own biological
    # peers, not to the whole plate (a genuinely dense condition is not an outlier).
    # Peers share the dataset and segmentation mode too: a 2-D MIP and a 3-D
    # z-stack of the same colony differ several-fold in nucleus count and DAPI
    # level, and pooling them would inflate the spread and hide real outliers.
    peers = [c for c in ("dataset", "seg_mode") if c in qc.columns] + ["condition"]
    grp = qc[peers].astype(str).agg("|".join, axis=1)
    for col in ("n_cells", "dapi_median", "area_median"):
        qc[f"{col}_z"] = (qc.groupby(grp)[col]
                          .transform(lambda s: pd.Series(mad_z(s.to_numpy()), index=s.index)))

    # A robust z alone is unusable at the 3-6 colonies a condition usually has:
    # the MAD of a few agreeing numbers collapses and a colony 10 % off its peers
    # scores z = 4. Requiring a relative deviation as well keeps the flag on
    # colonies that are actually different.
    min_rel = float(qc_cfg.get("field_min_rel_dev", 0.2))
    outlier = np.zeros(len(qc), bool)
    for col in ("n_cells", "dapi_median", "area_median"):
        med = qc.groupby(grp)[col].transform("median")
        rel = (qc[col] - med).abs() / med.abs().replace(0, np.nan)
        qc[f"{col}_rel_dev"] = rel
        outlier |= ((qc[f"{col}_z"].abs() > z_max) & (rel > min_rel)).to_numpy()
    qc["flag_low_count"] = qc["n_cells"] < min_cells
    qc["flag_outlier"] = outlier | qc["flag_low_count"].to_numpy()

    n_flagged = int(qc["flag_outlier"].sum())
    logger.info("qc: %d field(s), %d flagged (%.1f%%)", len(qc), n_flagged,
                100.0 * n_flagged / max(len(qc), 1))
    if n_flagged:
        worst = qc.loc[qc["flag_outlier"], ["field_id", "well", "condition", "n_cells"]].head(10)
        logger.info("qc: first flagged fields:\n%s", worst.to_string(index=False))
    return qc


def marker_field_qc(data: Data, cfg: dict,
                    channel_qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (field × marker) health, judged against the other fields of the well.

    Three independent flags, none of which changes a single analyzed number:

    ``flag_marker_outlier``
        The field's median for this marker is a robust-z outlier among the other
        fields of the same well × panel **and** deviates from the well's median
        by more than ``qc.marker_min_rel_dev``. Catches localized artifacts
        (patches of non-specific antibody, debris, a flare) that leave DAPI and
        the nucleus count perfectly normal and so pass :func:`field_qc` untouched.

        Both conditions are required because a robust z alone is unusable at the
        3-6 fields per well that most plates carry: the MAD of a handful of
        tightly agreeing numbers collapses toward zero, and then a field sitting
        3 % off its neighbors scores a z of 8. The relative floor is what keeps
        the flag pointed at artifacts rather than at ordinary agreement.
    ``flag_saturated``
        More than ``qc.max_saturated_fraction`` of that channel's pixels sat at
        the detector ceiling.
    ``flag_low_signal``
        Nuclei under ``qc.min_signal_to_background`` × their own background, i.e.
        indistinguishable from what they sit on.

    A note on what this can and cannot see: the robust z needs peers, so a well
    imaged at 2 fields cannot produce an outlier flag (the MAD of two numbers
    gives every point a z of 0) and one imaged at 3 will only catch the gross
    cases. That is a property of the design, not a failure of the check: with
    few fields per well the saturation and signal-to-background flags, which
    judge each field on its own, are what is doing the work.
    """
    qc_cfg = cfg.get("qc") or {}
    z_max = float(qc_cfg.get("marker_outlier_z", qc_cfg.get("outlier_z", 3.5)))
    min_rel = float(qc_cfg.get("marker_min_rel_dev", 0.25))
    max_sat = float(qc_cfg.get("max_saturated_fraction", 0.001))
    min_sb = float(qc_cfg.get("min_signal_to_background", 1.05))

    keys = [c for c in _WELL_KEYS if c in data.cells.columns]
    long = data.attach(keys + ["field_id"])

    # Each marker is judged in the space it is actually analyzed in, so the flag
    # and the figure it would explain are talking about the same numbers.
    spaces = {str(m): data.space_for(str(m), cfg) for m in data.markers}
    long = long.assign(_value=data.values(long, cfg))

    per = (long.groupby(keys + ["field_id", "marker"], observed=True, dropna=False)
           .agg(n_cells=("cell_uid", "size"), median=("_value", "median")).reset_index())
    per["space"] = per["marker"].astype(str).map(spaces)

    per["median_z"] = (per.groupby(keys + ["marker"], observed=True, dropna=False)["median"]
                       .transform(lambda s: pd.Series(mad_z(s.to_numpy()), index=s.index)))
    per["n_fields_in_well"] = (per.groupby(keys + ["marker"], observed=True, dropna=False)["field_id"]
                               .transform("nunique"))
    well_median = (per.groupby(keys + ["marker"], observed=True, dropna=False)["median"]
                   .transform("median"))
    per["rel_dev"] = ((per["median"] - well_median).abs()
                      / well_median.abs().replace(0, np.nan))
    per["flag_marker_outlier"] = (per["median_z"].abs() > z_max) & (per["rel_dev"] > min_rel)

    if channel_qc is not None and len(channel_qc):
        cols = [c for c in ("field_id", "marker", "saturated_fraction",
                            "nuclei_saturated_fraction", "background",
                            "signal_to_background") if c in channel_qc.columns]
        merged = channel_qc[cols].copy()
        merged["marker"] = merged["marker"].astype(str)
        per["marker"] = per["marker"].astype(str)
        per = per.merge(merged, on=["field_id", "marker"], how="left")
        per["flag_saturated"] = per.get("saturated_fraction", np.nan) > max_sat
        per["flag_low_signal"] = per.get("signal_to_background", np.nan) < min_sb
    else:
        per["flag_saturated"] = False
        per["flag_low_signal"] = False

    per["flag_any"] = (per["flag_marker_outlier"] | per["flag_saturated"]
                       | per["flag_low_signal"])

    n_flag = int(per["flag_any"].sum())
    logger.info("qc: %d (field × marker) row(s), %d flagged (%.1f%%)",
                len(per), n_flag, 100.0 * n_flag / max(len(per), 1))
    if n_flag:
        show = [c for c in ("field_id", "well", "marker", "median", "median_z",
                            "saturated_fraction", "signal_to_background") if c in per.columns]
        worst = (per[per["flag_any"]].sort_values("median_z", key=abs, ascending=False)
                 .head(10)[show])
        logger.warning("qc: marker-channel problems, worst first:\n%s",
                       worst.to_string(index=False))
        for marker, grp in per[per["flag_any"]].groupby("marker", observed=True):
            logger.warning("qc: %s; %d suspect field(s); check them before trusting "
                           "its gate, especially any in a control well.",
                           marker, len(grp))
    return per


COLONY_FLAGS = ("failed", "clipped", "size", "shape", "neighbour", "outside",
                "low_count", "field")


def colony_qc(colonies: pd.DataFrame, field: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """One row per colony with every geometry flag, the combined ``flags`` string
    and ``excluded`` (True when a flag listed in ``qc.exclude_flags`` is set)."""
    qc_cfg = cfg.get("qc") or {}
    col_cfg = cfg.get("colony") or {}
    out = colonies.copy()

    max_border = float(col_cfg.get("max_border_fraction", 0.15))
    max_dev = float(qc_cfg.get("max_diameter_deviation", 0.15))
    min_fill = float(qc_cfg.get("min_fill_ratio", 0.85))
    max_rms = float(qc_cfg.get("max_fit_rms_rel", 0.08))
    max_nb = float(qc_cfg.get("max_neighbour_ratio", 0.5))
    max_out = float(qc_cfg.get("max_outside_fraction", 0.2))
    min_nuc = int(qc_cfg.get("min_nuclei_per_colony", 200))

    def num(col):
        return pd.to_numeric(out.get(col), errors="coerce") if col in out else \
            pd.Series(np.nan, index=out.index)

    # Expected diameter: a per-well plate-map column beats the global value.
    expected = pd.Series(np.nan, index=out.index)
    diam_col = col_cfg.get("diameter_column")
    if diam_col and diam_col in out.columns:
        expected = pd.to_numeric(out[diam_col], errors="coerce")
    elif diam_col:
        logger.warning("colony.diameter_column=%r is not a plate-map column: using "
                       "colony.expected_diameter_um instead.", diam_col)
    if col_cfg.get("expected_diameter_um"):
        expected = expected.fillna(float(col_cfg["expected_diameter_um"]))
    out["expected_diameter_um"] = expected

    border = num("border_fraction")
    out["flag_failed"] = out.get("status", "ok").astype(str) != "ok"
    out["flag_clipped"] = border > max_border
    dev = (num("diameter_um") - expected).abs() / expected
    out["diameter_deviation"] = dev
    out["flag_size"] = dev > max_dev
    out["flag_shape"] = (((num("fill_ratio") < min_fill) & ~out["flag_clipped"])
                         | (num("fit_rms_rel") > max_rms))
    out["flag_neighbour"] = num("largest_neighbour_ratio") >= max_nb
    out["flag_outside"] = num("outside_fraction") > max_out
    out["flag_low_count"] = num("n_nuclei_colony") < min_nuc
    out["flag_field"] = False
    if field is not None and len(field) and "flag_outlier" in field:
        bad = set(field.loc[field["flag_outlier"].astype(bool), "field_id"])
        out["flag_field"] = out["field_id"].isin(bad)

    flag_cols = [f"flag_{f}" for f in COLONY_FLAGS]
    out[flag_cols] = out[flag_cols].fillna(False).astype(bool)
    out["flags"] = out[flag_cols].apply(
        lambda r: ";".join(f for f, v in zip(COLONY_FLAGS, r) if v), axis=1)
    out["flag_any"] = out[flag_cols].any(axis=1)

    exclude = qc_cfg.get("exclude_flags", ["failed"])
    exclude = [exclude] if isinstance(exclude, str) else list(exclude or [])
    if "all" in exclude:
        exclude = list(COLONY_FLAGS)
    unknown = [f for f in exclude if f not in COLONY_FLAGS]
    if unknown:
        raise ValueError(f"qc.exclude_flags: unknown flag(s) {unknown}; "
                         f"choose from {list(COLONY_FLAGS)} or 'all'.")
    out["excluded"] = out[[f"flag_{f}" for f in exclude]].any(axis=1) if exclude else False

    n_flag, n_excl = int(out["flag_any"].sum()), int(out["excluded"].sum())
    logger.info("qc: %d colonies, %d flagged, %d excluded (exclude_flags=%s)",
                len(out), n_flag, n_excl, exclude)
    for flag in COLONY_FLAGS:
        n = int(out[f"flag_{flag}"].sum())
        if n:
            logger.warning("qc: %d colony(ies) flagged '%s'", n, flag)
    return out


def saturated_nuclei(data: Data, channel_qc: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """Nuclei whose median intensity sits at the detector ceiling, in any channel.

    A nucleus whose *median* pixel is clipped is not a cell reading high: it is
    bright debris or a flare, and it stretches scales and gates. A nucleus is
    flagged when its raw median is ≥ ``qc.saturation_fraction`` (default 1.0) ×
    the ceiling of that channel in that image: ``qc.saturation_ceiling`` when set
    (e.g. 4095 for a 12-bit camera writing 16-bit files), else the extraction's
    per-channel ``saturation_value`` (the dtype maximum). The nuclear channel
    (the ``dapi`` column) counts too. Float images have no ceiling and are never
    flagged.

    Returns one row per flagged nucleus × channel: ``cell_uid``, ``field_id``,
    ``channel_name``, ``value``, ``ceiling``. Empty when nothing is clipped.
    """
    cols = ["cell_uid", "field_id", "channel_name", "value", "ceiling"]
    qc_cfg = cfg.get("qc") or {}
    frac = float(qc_cfg.get("saturation_fraction", 1.0))
    override = qc_cfg.get("saturation_ceiling")
    cells = data.cells
    field = cells["field_id"].to_numpy()
    markers = {str(m) for m in data.markers}
    have_qc = channel_qc is not None and len(channel_qc) and \
        {"field_id", "marker", "saturation_value"} <= set(channel_qc.columns)
    if override is None and not have_qc:
        return pd.DataFrame(columns=cols)
    if have_qc:
        cq = channel_qc[["field_id", "marker", "saturation_value"]].copy()
        cq["marker"] = cq["marker"].astype(str)
        ceil_of = cq.drop_duplicates(["field_id", "marker"]).set_index(["field_id", "marker"])[
            "saturation_value"]
        # The nuclear channel is the channel-QC row that is not a quantified marker.
        nuc = cq[~cq["marker"].isin(markers)].groupby("field_id")["saturation_value"].min()
    frames = []
    inten = data.intensities
    if RAW in inten:
        uid = inten["cell_uid"].to_numpy()
        names = inten["marker"].astype(str).to_numpy()
        if override is not None:
            ceil = np.full(len(inten), float(override))
        else:
            idx = pd.MultiIndex.from_arrays([field[uid], names])
            ceil = ceil_of.reindex(idx).to_numpy(np.float64)
        value = inten[RAW].to_numpy(np.float64)
        with np.errstate(invalid="ignore"):
            hit = np.isfinite(ceil) & (value >= frac * ceil)
        frames.append(pd.DataFrame({"cell_uid": uid[hit], "field_id": field[uid][hit],
                                    "channel_name": names[hit], "value": value[hit],
                                    "ceiling": ceil[hit]}))
    if "dapi" in cells:
        value = cells["dapi"].to_numpy(np.float64)
        ceil = (np.full(len(cells), float(override)) if override is not None
                else pd.Series(field).map(nuc).to_numpy(np.float64))
        with np.errstate(invalid="ignore"):
            hit = np.isfinite(ceil) & (value >= frac * ceil)
        frames.append(pd.DataFrame({"cell_uid": cells["cell_uid"].to_numpy()[hit],
                                    "field_id": field[hit], "channel_name": "nuclear",
                                    "value": value[hit], "ceiling": ceil[hit]}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return out[cols]


__all__ = ["field_qc", "marker_field_qc", "colony_qc", "saturated_nuclei", "mad_z",
           "COLONY_FLAGS"]
