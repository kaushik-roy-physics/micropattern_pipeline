"""
The built-in micropattern analysis steps.

Each is a small orchestration function: pull what it needs off the context, call
into the computational modules, write tables and figures. The science lives in
:mod:`downstream.colony`, :mod:`downstream.radial`, :mod:`downstream.fda`,
:mod:`downstream.thresholds` and :mod:`downstream.qc`; the drawing in
:mod:`downstream.plots`.

Order (dependencies are declared, not assumed)::

    colony → signaling → assemble → qc → thresholds → profiles → aggregate → compare → report
                                                        │           └→ multimarker ┘
                                                        ├→ coexpression ───────────┤
                                                        └→ signaling_summary ──────┘

``signaling`` (per image) and ``signaling_summary`` do nothing unless a stain set
holds a signaling marker (:mod:`downstream.signaling`).

``qc`` precedes ``thresholds`` and ``profiles``: a colony excluded by QC must not
move a positivity gate or the shared intensity anchors. Every step after
``assemble`` reads the in-colony nuclei of the colonies QC kept
(:attr:`downstream.context.Context.data`).

If the experiment has more than one segmentation mode (MIPs in 2-D *and*
z-stacks in 3-D: two measurements of the same colonies), aggregation and
comparison run once per mode and never pool them; their files then carry a
``_2d`` / ``_3d`` suffix.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import assemble as assemble_mod
from . import boundary as bd_mod
from . import captions as cap_mod
from . import effects as ef_mod
from . import coexpression as coex_mod
from . import colony as colony_mod
from . import fda
from . import plots
from . import qc as qc_mod
from . import signaling as sig_mod
from . import radial
from . import thresholds as thr_mod
from . import trends as trends_mod
from .aggregate import condition_columns, condition_label
from .assemble import assemble as assemble_cells
from .registry import register
from .stats import benjamini_hochberg

logger = logging.getLogger(__name__)

PROFILES_TABLE = "colony_profiles.csv"
BINNED_TABLE = "colony_profiles_binned.csv"
DENSITY_TABLE = "density_colonies.csv"
READOUTS = ("mean", "positive_fraction")


# ---------------------------------------------------------------------------
# 1. Colony geometry (per image)
# ---------------------------------------------------------------------------

def _n_jobs(cfg: dict) -> int:
    """Worker count: config, else the LSF slot count, else up to 8 local cores."""
    want = (cfg.get("colony") or {}).get("n_jobs")
    if want:
        return max(1, int(want))
    lsf = os.environ.get("LSB_DJOB_NUMPROC")
    if lsf and lsf.isdigit():
        return max(1, int(lsf))
    return max(1, min(8, os.cpu_count() or 1))


@register("colony", config_keys=["colony", "manifest_basename", "_selection"], always=True,
          description="Main colony, edge, outliers and distance-from-edge per image (+ QC overlay)")
def step_colony(ctx) -> bool:
    """Returns False when every image was already current (nothing changed)."""
    from joblib import Parallel, delayed

    params = colony_mod.settings(ctx.cfg)
    jobs = []
    for ds, man in assemble_mod.iter_manifests(ctx.datasets, ctx.cfg):
        dapi = int((ds.cfg.get("channels") or {}).get("dapi_index", 0))
        for rel in man["result_dir"].astype(str):
            jobs.append((ds.name, ds.directory / rel, dapi))
    if not jobs:
        raise RuntimeError("colony: no images in any manifest. Run 'python main.py manifest'.")
    # Images without segmentation are skipped with one warning per dataset (a
    # dataset that was never segmented would otherwise log an error per image).
    missing: dict[str, int] = {}
    for name, path, _ in jobs:
        if not (path / colony_mod.MASKS_NAME).exists():
            missing[name] = missing.get(name, 0) + 1
    for name, k in missing.items():
        total = sum(j[0] == name for j in jobs)
        logger.warning("[%s] %d of %d image(s) have no segmentation yet and are skipped. "
                       "Segment them first, or leave this dataset out (--datasets).",
                       name, k, total)
    jobs = [j for j in jobs if (j[1] / colony_mod.MASKS_NAME).exists()]
    if not jobs:
        raise RuntimeError("colony: no image has segmentation results. Run the "
                           "segmentation first.")

    force = ctx.force or getattr(ctx, "step_forced", False)
    digest = colony_mod.settings_hash(params)
    todo = [j for j in jobs if force or not colony_mod.is_current(j[1], digest)]
    # Starting worker processes costs seconds each (imports over a network share),
    # so a handful of images (or a re-run where nearly everything is current)
    # runs in this process.
    n_jobs = min(_n_jobs(ctx.cfg), len(todo))
    if len(todo) < 8:
        n_jobs = 1
    logger.info("colony: %d image(s), %d to (re)compute on %d worker(s)%s", len(jobs),
                len(todo), n_jobs, ", forced" if force else "")
    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(colony_mod.worker)(str(path), params, dapi, force)
            for _, path, dapi in jobs)
    else:
        results = [colony_mod.worker(str(path), params, dapi, force) for _, path, dapi in jobs]

    n_ok = n_skip = 0
    failures = []
    for (name, path, _), (meta, err) in zip(jobs, results):
        if err:
            failures.append((name, path, err))
            continue
        n_skip += bool(meta.get("skipped"))
        n_ok += 1
        if meta.get("status") != "ok":
            logger.warning("[%s] %s: no colony found (%s)", name, path.name, meta.get("status"))
    for name, path, err in failures:
        logger.error("[%s] colony failed for %s: %s", name, path.name, err)
    logger.info("colony: %d ok (%d unchanged, skipped), %d failed", n_ok, n_skip, len(failures))
    if failures and not n_ok:
        raise RuntimeError("colony: every image failed; see the errors above.")
    return bool(todo)


# ---------------------------------------------------------------------------
# 1b. Signaling markers (per image)
# ---------------------------------------------------------------------------

SIG_FIG = "signaling"


def _signaling_plan(ctx):
    """``(jobs, matches)``: per image, the signaling channels and their readouts."""
    aliases = {str(k).strip().lower(): str(v)
               for k, v in (ctx.cfg.get("marker_aliases") or {}).items()}
    exclude = {str(m).strip().lower() for m in (ctx.cfg.get("exclude_markers") or [])} | {"dapi"}
    jobs, names = [], []
    for ds, man in assemble_mod.iter_manifests(ctx.datasets, ctx.cfg):
        dapi = int((ds.cfg.get("channels") or {}).get("dapi_index", 0))
        measured = {int(c) for c in (ds.cfg.get("channels") or {}).get("marker_channels", [])}
        for _, row in man.iterrows():
            ch2marker = assemble_mod._marker_map(row, aliases)
            channels = {}
            for ch, marker in ch2marker.items():
                if ch == dapi or marker.strip().lower() in exclude:
                    continue
                if measured and ch not in measured:
                    continue
                names.append(marker)
                match = sig_mod.classify(marker, ctx.cfg)
                if match is not None:
                    channels[int(ch)] = match.readout
            if channels:
                # Images with the same channel → marker layout (one stain set) are
                # compared with each other in the background check.
                panel = " | ".join(f"{c}:{m}" for c, m in sorted(ch2marker.items()))
                jobs.append((ds.name, ds.directory / str(row["result_dir"]), channels, panel))
    matches = sig_mod.log_resolution(sorted(set(names)), ctx.cfg)
    return jobs, matches


@register("signaling", after=["colony"],
          config_keys=["signaling", "marker_aliases", "exclude_markers", "manifest_basename",
                       "_selection", "colony"], always=True,
          description="Signaling markers: nuclear core, cytoplasmic ring and background per "
                      "nucleus (only when a stain set holds one)")
def step_signaling(ctx) -> bool:
    jobs, matches = _signaling_plan(ctx)
    if not matches:
        logger.info("signaling: no signaling marker in any stain set; nothing to measure.")
        return False
    jobs = [j for j in jobs if (j[1] / colony_mod.MASKS_NAME).exists()]
    panels = {path: panel for _, path, _, panel in jobs}
    jobs = [j[:3] for j in jobs]
    if not jobs:
        logger.warning("signaling: no segmented image holds a signaling marker yet.")
        return False
    params = sig_mod.settings(ctx.cfg)
    cparams = colony_mod.settings(ctx.cfg)
    force = ctx.force or getattr(ctx, "step_forced", False)
    from .signaling import measure as sig_measure

    todo, pixel = [], {}
    for name, path, channels in jobs:
        pixel[path] = colony_mod.pixel_size_for(path, cparams)
        digest = sig_measure.settings_hash(params, channels)
        if force or not sig_measure.is_current(path, digest):
            todo.append((name, path, channels))
    n_jobs = min(_n_jobs(ctx.cfg), len(todo))
    if len(todo) < 8:
        n_jobs = 1
    logger.info("signaling: %d image(s) with signaling channels, %d to (re)measure on %d "
                "worker(s)", len(jobs), len(todo), n_jobs)
    if n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(sig_measure.worker)(str(path), ch, params, pixel[path], force)
            for _, path, ch in todo)
    else:
        results = [sig_measure.worker(str(path), ch, params, pixel[path], force)
                   for _, path, ch in todo]
    failures = [(name, path, err) for (name, path, _), (_, err) in zip(todo, results) if err]
    for name, path, err in failures:
        logger.error("[%s] signaling failed for %s: %s", name, path.name, err)
    if failures and len(failures) == len(todo):
        raise RuntimeError("signaling: every image failed; see the errors above.")

    bg = sig_measure.background_table((name, path, panels[path]) for name, path, _ in jobs)
    ctx.write_table(bg, "signaling_background.csv")
    for r in bg[bg["outlier"]].itertuples(index=False):
        logger.warning("signaling: %s channel %d background %.4g is far from the other images "
                       "of %s with the same stain set; check that image.", r.result_dir, r.channel,
                       r.background, r.dataset)
    fallback = bg[bg["method"] != str(params.get("background", "local"))]
    if len(fallback):
        logger.info("signaling: %d image channel(s) used a fallback background (%s).",
                    len(fallback), ", ".join(sorted(set(fallback["method"].astype(str)))))

    ctx.reset_figures(SIG_FIG)
    entries = []
    for name, path, _ in sorted(jobs, key=lambda j: str(j[1])):
        png = path / sig_measure.QC_NAME
        if png.exists():
            entries.append((png, plots.short_name(path.name), ""))
    per_page = 12
    pages = [entries[i:i + per_page] for i in range(0, len(entries), per_page)]
    for k, page in enumerate(pages, 1):
        plots.overlay_gallery(page, ctx.figure(SIG_FIG, f"ring_qc_{k:02d}.png"), ncols=3,
                              thumb_px=900,
                              title=f"Signaling rings: page {k}/{len(pages)} (orange = nucleus "
                                    f"core, sky blue = cytoplasmic ring inside cell bodies)")
    return bool(todo)


# ---------------------------------------------------------------------------
# 2. Assembly
# ---------------------------------------------------------------------------

def _invalidate(ctx) -> None:
    for key in ("data", "colonies", "gated", "controls", "thresholds", "curves"):
        ctx._cache.pop(key, None)


@register("assemble", after=["colony", "signaling"],
          config_keys=["marker_aliases", "exclude_markers", "manifest_basename", "_selection",
                       "_manifests", "colony", "signaling"],
          description="Join manifests, per-image intensities and colony geometry into cell tables")
def step_assemble(ctx) -> None:
    data = assemble_cells(ctx.datasets, ctx.cfg, ctx.out)
    _invalidate(ctx)
    ctx.set("all_data", data)


# ---------------------------------------------------------------------------
# 3. Quality control
# ---------------------------------------------------------------------------

@register("qc", after=["assemble"],
          config_keys=["qc", "colony", "conditions", "markers", "default_space", "signaling",
                       "nuclear_stain"],
          description="Colony geometry flags, per-colony staining QC, overlay galleries")
def step_qc(ctx) -> None:
    ctx.reset_figures("qc")
    full = ctx.all_data
    field = qc_mod.field_qc(full, ctx.cfg)
    ctx.write_table(field, "qc_fields.csv")
    plots.count_qc(field, ctx.cfg, ctx.figure("qc", "segmentation_yield.png"))

    channel_qc = None
    channel_qc_path = ctx.out / assemble_mod.CHANNEL_QC_NAME
    if channel_qc_path.exists():
        channel_qc = pd.read_csv(channel_qc_path)
    names = _nuclear_names(ctx)
    kind = cap_mod.nuclear_stain_kind(names, ctx.cfg)
    risky = cap_mod.histone_denominator_markers(
        {str(m): ctx.space_for(str(m)) for m in full.markers}, kind)
    if risky:
        logger.warning("qc: %s are divided by the nuclear channel (%s), a histone label, not a "
                       "DNA stain: its level varies from cell to cell, and if it is a reporter "
                       "this divides by the signal. Prefer space: raw (or set nuclear_stain: dna "
                       "if the name misleads).", ", ".join(risky), "/".join(names))
    marker_qc = qc_mod.marker_field_qc(full, ctx.cfg, channel_qc)
    ctx.write_table(marker_qc, "qc_marker_colonies.csv")

    ctx.saturated_mask(full)
    hits = ctx._cache.get("saturated_hits")
    if hits is not None and len(hits):
        ctx.write_table(hits.groupby(["field_id", "channel_name"]).agg(
            n_nuclei=("cell_uid", "nunique"), ceiling=("ceiling", "first")).reset_index(),
            "qc_saturated_nuclei.csv")
    else:
        _remove_tables(ctx, "qc_saturated_nuclei.csv")

    colonies = pd.read_csv(ctx.out / assemble_mod.COLONIES_NAME)
    qc = qc_mod.colony_qc(colonies, field, ctx.cfg)
    ctx.write_table(qc, "qc_colonies.csv")
    _invalidate(ctx)

    plots.colony_qc_scatter(qc, ctx.figure("qc", "colony_geometry.png"))
    _overlay_galleries(ctx, qc)


def _result_path(ctx, dataset: str, result_dir: str) -> Path | None:
    for ds in ctx.datasets:
        if ds.name == str(dataset):
            return ds.directory / str(result_dir)
    return None


def _overlay_galleries(ctx, qc: pd.DataFrame, per_page: int = 24) -> None:
    frame = qc.sort_values([c for c in ("dataset", "well", "field_id") if c in qc.columns])
    entries = []
    for _, r in frame.iterrows():
        path = _result_path(ctx, r["dataset"], r["result_dir"])
        if path is None:
            continue
        diam = r.get("diameter_um")
        cap = f"w{r.get('well', '?')} · {plots.short_name(Path(str(r['result_dir'])).name)}"
        if pd.notna(diam):
            cap += f"\nØ {float(diam):.0f} µm · {int(r.get('n_nuclei_colony') or 0)} nuclei"
        flags = str(r["flags"]) if isinstance(r.get("flags"), str) else ""
        if bool(r.get("excluded")):
            flags = (flags + " (excluded)").strip()
        thumb = path / colony_mod.OVERLAY_THUMB_NAME
        entries.append((thumb if thumb.exists() else path / colony_mod.OVERLAY_NAME, cap, flags))
    pages = [entries[i:i + per_page] for i in range(0, len(entries), per_page)]
    for k, page in enumerate(pages, 1):
        plots.overlay_gallery(page, ctx.figure("qc", f"colony_overlays_{k:02d}.png"),
                              title=f"Colony detection QC: page {k}/{len(pages)} "
                                    f"(orange = edge, dashed = circle fit, x = outside colony)")


# ---------------------------------------------------------------------------
# 4. Thresholds (optional readout: fraction of positive nuclei)
# ---------------------------------------------------------------------------

@register("thresholds", after=["qc"],
          config_keys=["thresholds", "conditions", "markers", "default_space", "qc",
                       "signaling"],
          description="Positivity thresholds per scope × marker (for % positive profiles)")
def step_thresholds(ctx) -> None:
    ctx.reset_figures("thresholds")
    controls = ctx.controls
    table = thr_mod.compute_thresholds(ctx.data, ctx.cfg, controls)
    ctx.write_table(table, "thresholds.csv")
    ctx.set("thresholds", table)
    ctx._cache.pop("gated", None)
    ctx.write_table(controls.summary(ctx.data.markers), "control_assignment.csv")
    plots.threshold_qc(ctx.gated, table, ctx.data.cells, controls, ctx.cfg,
                       ctx.figure("thresholds", "threshold_evidence.png"))
    graded = table[~table["verdict"].isin([thr_mod.OK, thr_mod.NOT_GATED])]
    if len(graded):
        logger.warning("%d (scope × marker) combination(s) have NO gate: their %% positive "
                       "profiles are left empty: %s", len(graded),
                       ", ".join(sorted(set(graded["marker"].astype(str)))))


# ---------------------------------------------------------------------------
# 5. Per-colony profiles
# ---------------------------------------------------------------------------

def colony_info(ctx) -> pd.DataFrame:
    """Kept colonies with condition label, a dataset-unique well key and a name."""
    col = ctx.colonies
    if "excluded" in col:
        col = col[~col["excluded"].astype(bool)]
    col = col.copy()
    keys = condition_columns(col, ctx.cfg.get("conditions") or {})
    col["condition"] = condition_label(col, keys)
    plate = col["plate_id"].astype(str) if "plate_id" in col else ""
    col["well_key"] = col["dataset"].astype(str) + "|" + plate + "|" + col["well"].astype(str)
    col["colony"] = col["result_dir"].astype(str).map(lambda s: Path(s).name)
    return col


def _info_columns(ctx, info: pd.DataFrame) -> list[str]:
    keep = ["field_id", "dataset", "plate_id", "stain_set", "seg_mode", "well", "row", "col",
            "well_key", "condition", "colony", "result_dir", "radius_um", "flags"]
    keep += condition_columns(info, ctx.cfg.get("conditions") or {})
    keep += list(ctx.facet_keys)
    diam = (ctx.cfg.get("colony") or {}).get("diameter_column")
    if diam:
        keep.append(diam)
    exp_col = (ctx.cfg.get("stats") or {}).get("experiment_column")
    if exp_col:
        keep.append(str(exp_col))
    return [c for c in dict.fromkeys(keep) if c in info.columns]


@register("profiles", after=["thresholds", "boundary"],
          config_keys=["profiles", "markers", "default_space", "thresholds", "qc", "conditions",
                       "visualization", "signaling", "boundary", "stats", "nuclear_stain",
                       "skip_steps"],
          description="Shared-scale radial profile per colony on the µm and normalized grids")
def step_profiles(ctx) -> None:
    prm = radial.settings(ctx.cfg)
    gated = ctx.gated if ctx.has_table("thresholds.csv") else None
    if gated is None:
        logger.info("profiles: no thresholds.csv; %% positive profiles will be empty.")
    long = radial.nucleus_table(ctx.data, gated, ctx.cfg)
    pct = tuple(float(p) for p in prm["robust_percentiles"])
    anchors = radial.shared_anchors(long, pct)
    ctx.write_table(anchors, "scale_anchors.csv")
    long["scaled"] = radial.apply_scale(long, anchors)
    long["scaled"] = _signaling_relative(ctx, long, write=True)

    max_dist = float(np.nanmax(long["dist_edge_um"])) if len(long) else 0.0
    blocks = [radial.colony_curves(long, grid, radial.grid_edges(grid, prm, max_dist), prm)
              for grid in radial.GRIDS]
    curves = pd.concat(blocks, ignore_index=True)
    info = colony_info(ctx)
    curves = curves.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
    ctx.write_table(curves, PROFILES_TABLE)
    ctx.set("curves", curves)
    logger.info("profiles: %d colonies × %d marker(s); µm grid %g µm bins to %.0f µm, "
                "normalized grid %d bins", curves["field_id"].nunique(),
                curves["marker"].nunique(), float(prm["bin_um"]),
                float(curves.loc[curves["grid"] == radial.UM, "bin_hi"].max()),
                int(prm["n_norm_bins"]))

    quantiles, areas = _edge_geometry(ctx, info)
    density = radial.density_curves(long, radial.grid_edges(radial.UM, prm, max_dist), areas)
    if len(density):
        density = density.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
        ctx.write_table(density, DENSITY_TABLE)
    binned = radial.colony_binned(long, prm, quantiles, edge_area=areas)
    if binned.empty:
        return
    binned = binned.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
    ctx.write_table(binned, BINNED_TABLE)

    ctx.reset_figures("colony_profiles")
    if not bool((ctx.cfg.get("profiles") or {}).get("colony_figures", True)):
        return
    ylims = _ylims(ctx, binned)
    annotations = _boundary_annotations(ctx)
    for _, part in binned.groupby("field_id"):
        r = part.iloc[0]
        # Only the markers this colony was stained for: a colony of one panel
        # never shows empty panels for the markers of the other panels.
        present = set(part.loc[part["mean"].notna(), "marker"].astype(str))
        markers = [m for m in ctx.markers if m in present]
        plots.colony_profile(part, markers, {"title": _colony_title(r), "ci": prm["ci"],
                                             "caption": _caption(ctx)},
                             ctx.figure("colony_profiles", f"w{r['well']}_{_safe(r['colony'])}.png"),
                             ylims=ylims, annotations=annotations.get(r["field_id"]))


def _bin_check_outputs(ctx, checks: list, table: pd.DataFrame) -> None:
    """``boundary_bin_check.csv`` (per colony) and ``boundary_bin_check_summary.csv``:
    how far each position and width moves under other binnings, in µm and in units
    of its own bootstrap SE (pooled over conditions, then per condition)."""
    names = ("boundary_bin_check.csv", "boundary_bin_check_summary.csv")
    if not checks:
        _remove_tables(ctx, *names)
        return
    check = pd.concat(checks, ignore_index=True)
    ctx.write_table(check, names[0])
    pooled = bd_mod.bin_check_summary(check, table.assign(condition="all"), ["condition"])
    per = bd_mod.bin_check_summary(check, table, ["condition"])
    summary = pd.concat([pooled, per], ignore_index=True)
    ctx.write_table(summary, names[1])
    if len(pooled):
        plots.bin_check(check, table, ctx.figure("boundary", "bin_check.png"))
        worst = pooled.sort_values("median_abs_shift_se", ascending=False).head(3)
        for r in worst.itertuples(index=False):
            logger.info("boundary bin check: %s %s %s (%s) moves a median %.3g (%.2g SE) "
                        "under %s bins", r.kind, r.subject, r.metric, r.readout,
                        r.median_abs_shift, r.median_abs_shift_se, r.scheme)


def _boundary_annotations(ctx) -> dict:
    """``{field_id: {fits, borders, peaks, missing}}`` from the boundary step's tables
    ({} if absent).

    Only accepted values are drawn: a rejected fit would put a confident line on
    a colony that has no transition. ``missing`` says, in words, why a facing pair
    of the colony's stain set has no line.
    """
    if "boundary" in ctx.skipped or not (ctx.has_table(BOUNDARY_COLONY_TABLE)
                                         and ctx.has_table(BOUNDARY_FIT_TABLE)):
        return {}
    table = ctx.read_table(BOUNDARY_COLONY_TABLE)
    out: dict = {}
    for fid, notes in bd_mod.unresolved_pairs(
            table, bd_mod.settings(ctx.cfg)["min_support"]).items():
        out.setdefault(fid, {})["missing"] = notes
    table = table[table["ok"].astype(bool)]
    ok_fit = table[(table["kind"] == "marker") & (table["metric"] == "x50_um")]
    accepted = set(zip(ok_fit["field_id"], ok_fit["subject"].astype(str), ok_fit["readout"]))
    for r in ctx.read_table(BOUNDARY_FIT_TABLE).itertuples(index=False):
        if (r.field_id, str(r.marker), r.readout) in accepted:
            out.setdefault(r.field_id, {}).setdefault("fits", {})[(str(r.marker), r.readout)] = \
                (r.x50_um, r.k, r.lo, r.hi)
    for r in table[table["metric"] == "crossing_um"].itertuples(index=False):
        out.setdefault(r.field_id, {}).setdefault("borders", []).append(
            (float(r.value), float(r.se), str(r.subject)))
    for r in table[table["metric"] == "peak_um"].itertuples(index=False):
        out.setdefault(r.field_id, {}).setdefault("peaks", {})[str(r.subject)] = \
            (float(r.value), float(r.se))
    return out


def _signaling_relative(ctx, long: pd.DataFrame, write: bool = False) -> pd.Series:
    """Signaling markers relative to their reference population; others unchanged.

    The profiles step computes and writes the reference table; later steps read
    it, so every figure and table uses the same reference levels.
    """
    from .signaling import reference as sig_ref
    matches = sig_mod.classify_all([str(m) for m in ctx.data.markers], ctx.cfg)
    if write:
        table = sig_ref.reference_table(long, ctx.data.cells, ctx.cfg, matches) \
            if matches else pd.DataFrame()
        if len(table):
            ctx.write_table(table, sig_ref.TABLE)
        else:
            _remove_tables(ctx, sig_ref.TABLE)
    else:
        table = ctx.read_table(sig_ref.TABLE) if ctx.has_table(sig_ref.TABLE) else pd.DataFrame()
    _set_value_labels(ctx, table)
    return sig_ref.relative_values(long, table, long["scaled"])


def _set_value_labels(ctx, table: pd.DataFrame | None = None) -> None:
    """Tell the figure modules which markers are not on the shared scale."""
    from .signaling import reference as sig_ref
    if table is None:
        table = ctx.read_table(sig_ref.TABLE) if ctx.has_table(sig_ref.TABLE) else None
    plots.set_value_labels(sig_ref.unit_labels(table), sig_ref.reference_markers(table))


def _nuclear_names(ctx) -> list[str]:
    """Names of the nuclear (segmentation) channel, from the extraction channel QC."""
    path = ctx.out / assemble_mod.CHANNEL_QC_NAME
    if not path.exists():
        return []
    qc = pd.read_csv(path)
    names = set()
    for ds in ctx.datasets:
        dapi = int((ds.cfg.get("channels") or {}).get("dapi_index", 0))
        sel = (qc["dataset"].astype(str) == ds.name) & (qc["channel"] == dapi)
        names |= set(qc.loc[sel, "marker"].astype(str))
    return sorted(names)


def _caption(ctx) -> str:
    """What each marker's intensity is normalized to (see :mod:`captions`)."""
    def build():
        from . import scope as scope_mod
        from .signaling import reference as sig_ref
        markers = [str(m) for m in ctx.markers]
        spaces = {m: ctx.space_for(m) for m in markers}
        ref = ctx.read_table(sig_ref.TABLE) if ctx.has_table(sig_ref.TABLE) else None
        signaling = {m: lab for m, (lab, _) in sig_ref.unit_labels(ref).items()}
        signaling.update({m: spaces[m] for m in sig_mod.classify_all(markers, ctx.cfg)
                          if m not in signaling})
        names = _nuclear_names(ctx)
        kind = cap_mod.nuclear_stain_kind(names, ctx.cfg)
        return cap_mod.normalization_caption(
            spaces, {m: scope_mod.level_for(m, ctx.cfg) for m in markers}, signaling, names,
            kind, (ctx.cfg.get("visualization") or {}).get("caption"))
    return ctx._cached("caption", build)


def _ylims(ctx, curves: pd.DataFrame, value: str = "mean") -> dict:
    """Per-marker y-range shared by every figure of a marker (``visualization.ylim``).

    ``shared`` (default): colonies and wells are compared on one axis per marker.
    ``auto``: each panel autoscales, which shows the shape of a weak profile better.
    """
    mode = str((ctx.cfg.get("visualization") or {}).get("ylim", "shared")).lower()
    if mode not in ("shared", "auto"):
        raise ValueError(f"visualization.ylim must be 'shared' or 'auto', not {mode!r}")
    return radial.display_limits(curves, value) if mode == "shared" else {}


def _edge_geometry(ctx, info: pd.DataFrame):
    """``(quantiles, areas)``: ``quantiles(field_id, q)`` → colony pixel-distance-to-edge
    quantiles in µm (equal-area bins); ``areas(field_id, edges)`` → colony area (µm²)
    per distance bin. Both from the colony mask (read once per colony), or from the
    fitted circle when the edge is the circle."""
    by_id = info.set_index("field_id")
    cache: dict = {}

    def distances(field_id):
        if field_id not in cache:
            r = by_id.loc[field_id]
            if str(r.get("edge_method")) == "circle":
                cache[field_id] = None
            else:
                import tifffile
                from scipy import ndimage as ndi
                path = _result_path(ctx, r["dataset"], r["result_dir"]) / colony_mod.MASK_NAME
                mask = tifffile.imread(path).astype(bool)
                cache[field_id] = ndi.distance_transform_edt(mask)[mask] * \
                    float(r["pixel_size_um"])
        return cache[field_id]

    def quantiles(field_id, q):
        q = np.asarray(q, dtype=float)
        dist = distances(field_id)
        if dist is None:                    # equal-area annuli of a disk
            return float(by_id.loc[field_id, "radius_um"]) * (1 - np.sqrt(1 - q))
        return np.quantile(dist, q)

    def areas(field_id, edges):
        dist = distances(field_id)
        r = by_id.loc[field_id]
        if dist is None:
            return radial.ring_areas(None, edges, radius_um=float(r["radius_um"]))
        return radial.ring_areas(dist, edges, float(r["pixel_size_um"]))
    return quantiles, areas


# ---------------------------------------------------------------------------
# 6. Aggregation (wells, conditions)
# ---------------------------------------------------------------------------

def _curves(ctx) -> pd.DataFrame:
    if "curves" in ctx._cache:
        return ctx._cache["curves"]
    return ctx.set("curves", ctx.read_table(PROFILES_TABLE))


def _strata(frame: pd.DataFrame):
    """``(suffix, label, subset)`` per segmentation mode; one pass when there is one."""
    modes = sorted(frame["seg_mode"].astype(str).unique()) if "seg_mode" in frame else []
    if len(modes) <= 1:
        yield "", "", frame
        return
    for mode in modes:
        yield f"_{mode}", f"{mode} segmentation", frame[frame["seg_mode"].astype(str) == mode]


def _readouts(curves: pd.DataFrame) -> list[str]:
    return [r for r in READOUTS if r in curves and curves[r].notna().any()]


@register("aggregate", after=["profiles"],
          config_keys=["profiles", "stats", "conditions", "visualization"],
          description="Well and condition mean profiles (replicate = colony or well) + colony galleries")
def step_aggregate(ctx) -> None:
    ctx.reset_figures("profiles")
    _set_value_labels(ctx)
    prm = radial.settings(ctx.cfg)
    curves = _curves(ctx)
    cond_keys = condition_columns(curves, ctx.cfg.get("conditions") or {})
    facet_keys = [k for k in ctx.facet_keys if k not in cond_keys]
    col = ctx.colonies
    flagged = set(col.loc[col["flag_any"].astype(bool), "field_id"]) if "flag_any" in col else set()

    wells, conds, units = [], [], {}
    for suffix, label, part in _strata(curves):
        info = part.drop_duplicates("field_id")
        unit, note = radial.choose_unit(info, ctx.cfg)
        units[label or "all"] = {"unit": unit, "note": note}
        logger.info("aggregate%s: %s", f" [{label}]" if label else "", note)
        by_well = [c for c in dict.fromkeys(
            ["seg_mode", "well_key", "well", "condition"] + cond_keys + facet_keys) if c in part]
        by_cond = [c for c in dict.fromkeys(
            ["seg_mode", "condition"] + cond_keys + facet_keys) if c in part]
        unit_curves = radial.to_unit(part, unit, radial.experiment_column(part, ctx.cfg))
        for readout in _readouts(part):
            wells.append(radial.summarize(part, by_well, readout, prm))
            s = radial.summarize(unit_curves, by_cond, readout, prm)
            s["unit"] = unit
            conds.append(s)

        # Galleries: every colony, one panel per well, wells ordered by condition.
        cond_order = plots.ordered_conditions(info["condition"].astype(str).unique(), ctx.cfg,
                                              ctx.control_conditions)
        rank = {c: i for i, c in enumerate(cond_order)}
        well_info = info.assign(_r=info["condition"].astype(str).map(rank)).sort_values(["_r", "well"])
        order = list(dict.fromkeys(well_info["well_key"]))
        titles = {k: f"well {w} · {plots.shorten(str(c), 26)}" for k, w, c in
                  zip(well_info["well_key"], well_info["well"], well_info["condition"])}
        for marker in ctx.markers:
            for grid in radial.GRIDS:
                ylim = _ylims(ctx, part[part["grid"] == grid]).get(str(marker))
                for readout in _readouts(part):
                    tag = f"{_safe(marker)}_{grid}_{_short(readout)}{suffix}"
                    plots.colony_gallery(part, marker, grid, readout, order, titles,
                                         ctx.figure("profiles", f"colonies_{tag}.png"),
                                         flagged=flagged, ylim=ylim)

    _density_summary(ctx, prm, units)
    if wells:
        ctx.write_table(pd.concat(wells, ignore_index=True), "well_profiles.csv")
    if conds:
        ctx.write_table(pd.concat(conds, ignore_index=True), "condition_profiles.csv")
    with open(ctx.table("replicate_unit.json"), "w") as f:
        json.dump(units, f, indent=2)


def _density_summary(ctx, prm: dict, units: dict) -> None:
    """Nuclear density per distance bin by condition (same replicate unit as the
    profiles): ``density_profiles.csv`` and ``figures/profiles/density_um*.png``."""
    if not ctx.has_table(DENSITY_TABLE):
        return
    dens = ctx.read_table(DENSITY_TABLE)
    if dens.empty:
        return
    cond_keys = condition_columns(dens, ctx.cfg.get("conditions") or {})
    frames = []
    for suffix, label, part in _strata(dens):
        unit = units.get(label or "all", {}).get("unit", "colony")
        by = [c for c in dict.fromkeys(["seg_mode", "condition"] + cond_keys) if c in part]
        s = radial.summarize(radial.to_unit(part, unit, radial.experiment_column(part, ctx.cfg)),
                             by, "mean", prm)
        s["unit"] = unit
        frames.append(s)
        order = plots.ordered_conditions(part["condition"].astype(str).unique(), ctx.cfg,
                                         ctx.control_conditions)
        plots.condition_profiles(s, [radial.DENSITY], order, radial.UM, "mean", {}, unit,
                                 ctx.figure("profiles", f"density_um{suffix}.png"),
                                 ctx.control_conditions,
                                 title="Nuclear density by distance from the edge"
                                       + (f" · {label}" if label else ""),
                                 caption="density = nuclei in the bin ÷ colony area in the bin "
                                         "(from the colony mask)")
    ctx.write_table(pd.concat(frames, ignore_index=True), "density_profiles.csv")


# ---------------------------------------------------------------------------
# 7. Comparison
# ---------------------------------------------------------------------------

@register("compare", after=["aggregate"],
          config_keys=["compare", "stats", "conditions", "profiles", "colony", "visualization",
                       "nuclear_stain"],
          description="Functional permutation tests, pairwise FDR, features, colony similarity")
def step_compare(ctx) -> None:
    ctx.reset_figures("compare")
    _set_value_labels(ctx)
    cmp_cfg = ctx.cfg.get("compare") or {}
    n_perm = int(cmp_cfg.get("n_permutations", 5000))
    edge_band = float(cmp_cfg.get("edge_band_um", 50.0))
    seed = int((ctx.cfg.get("stats") or {}).get("random_state", 0))
    curves = _curves(ctx)
    summary = ctx.read_table("condition_profiles.csv")
    diam_col = (ctx.cfg.get("colony") or {}).get("diameter_column")

    glob_rows, point_rows, pair_frames, feat_frames, test_frames, sim_frames = [], [], [], [], [], []
    notes: dict = {}
    for suffix, label, stratum in _strata(curves):
        unit, note = radial.choose_unit(stratum.drop_duplicates("field_id"), ctx.cfg)
        sum_part = (summary[summary["seg_mode"].astype(str).isin(set(stratum["seg_mode"].astype(str)))]
                    if suffix else summary)
        for facet, (cur, summ), relabel in ctx.facets(stratum, sum_part):
            ftag = suffix + ctx.facet_suffix(facet)
            ftitle = ", ".join(t for t in (label, ctx.facet_title(facet)) if t)
            controls = {relabel.get(c, c) for c in ctx.control_conditions}
            order = plots.ordered_conditions(cur["condition"].astype(str).unique(), ctx.cfg,
                                             controls)
            grids = list(radial.GRIDS)
            if diam_col and diam_col in cur and cur[diam_col].nunique() > 1:
                logger.warning("compare%s: colonies of different nominal diameters (%s); "
                               "normalized-grid comparison skipped; µm grid only.",
                               f" [{ftitle}]" if ftitle else "", diam_col)
                grids = [radial.UM]
            unit_curves = radial.to_unit(cur, unit, radial.experiment_column(cur, ctx.cfg))
            key = {"stratum": label or "all", "facet": ctx.facet_title(facet) or "all"}
            notes[f"{key['stratum']} / {key['facet']}"] = {"unit": unit, "note": note}
            rng = np.random.default_rng(seed)
            untested: set = set()

            for grid in grids:
                for readout in _readouts(cur):
                    tests, fam = {}, []
                    for marker in ctx.markers:
                        X, groups, centers = _matrix(unit_curves, marker, grid, readout)
                        if X is None:
                            continue
                        # A group with one replicate has no spread, so it cannot enter
                        # an F statistic; it is drawn but left out of the tests.
                        names, sizes = np.unique(groups, return_counts=True)
                        small = set(names[sizes < 2])
                        if small:
                            untested |= small
                            keep = ~np.isin(groups, list(small))
                            X, groups = X[keep], groups[keep]
                        if len(set(groups)) < 2:
                            continue
                        res = fda.functional_anova(X, groups, n_perm, rng)
                        res["centers"] = centers
                        tests[marker] = res
                        glob_rows.append(_global_row(key, grid, readout, marker, unit, res))
                        for c, f_, p_, s_ in zip(centers, res["grid_f"], res["point_p"],
                                                 res["sig_mask"]):
                            point_rows.append({**key, "grid": grid, "readout": readout,
                                               "marker": marker, "bin_center": c, "F": f_,
                                               "point_p": p_, "significant": bool(s_)})
                        names = res["groups"]
                        if len(names) > 2:
                            for i in range(len(names)):
                                for j in range(i + 1, len(names)):
                                    sel = np.isin(groups, [names[i], names[j]])
                                    pr = fda.functional_anova(X[sel], groups[sel], n_perm, rng)
                                    fam.append({**key, "grid": grid, "readout": readout,
                                                "marker": marker, "group_a": names[i],
                                                "group_b": names[j],
                                                "global_F": pr["global_stat"],
                                                "p_value": pr["global_p"], "exact": pr["exact"]})
                    tag = f"{grid}_{_short(readout)}{_safe(ftag)}"
                    if fam:
                        fam = pd.DataFrame(fam)
                        ok = fam["p_value"].notna()
                        fam["q_value"] = np.nan
                        fam.loc[ok, "q_value"] = benjamini_hochberg(fam.loc[ok, "p_value"].to_numpy())
                        fam["significant"] = fam["q_value"] < 0.05
                        pair_frames.append(fam)
                        plots.pairwise_heatmap(fam, ctx.figure("compare", f"pairwise_{tag}.png"),
                                               title=f"Pairwise functional tests: "
                                                     f"{'µm' if grid == radial.UM else 'normalized'} grid, "
                                                     f"{_short(readout)} (BH q, * q < 0.05)"
                                                     + (f"\n{ftitle}" if ftitle else ""))
                    plots.condition_profiles(summ, ctx.markers, order, grid, readout, tests,
                                             unit, ctx.figure("compare", f"profiles_{tag}.png"),
                                             controls, title=ftitle, caption=(
                                                 _caption(ctx) if readout == "mean" else ""))

            if untested:
                logger.warning("compare%s: %s left out of the statistical tests (a single "
                               "%s; at least 2 are needed). Still shown in the figures.",
                               f" [{ftitle}]" if ftitle else "", ", ".join(sorted(untested)),
                               unit)

            feats = _features(cur, edge_band)
            if not feats.empty:
                unit_ids = (cur[["field_id", "well_key"]].drop_duplicates()
                            .rename(columns={"well_key": "unit_id"}) if unit == "well" else
                            cur[["field_id"]].drop_duplicates().assign(
                                unit_id=lambda d: d["field_id"].astype(str)))
                feats = feats.merge(unit_ids, on="field_id", how="left")
                ftests = fda.feature_tests(feats, "condition", "unit_id")
                for k, v in key.items():
                    feats[k] = v
                    ftests[k] = v
                feat_frames.append(feats)
                test_frames.append(ftests)
                plots.feature_strips(feats, ftests, [m for m in ctx.markers if m in set(feats["marker"])],
                                     list(fda.FEATURES), order, "unit_id",
                                     ctx.figure("compare", f"features{_safe(ftag)}.png"), controls)

            sim = _similarity(cur, order, exclude=_absolute_markers(ctx))
            if sim is not None:
                D, labels, groups = sim
                plots.similarity_heatmap(D, labels, groups, order,
                                         ctx.figure("compare", f"similarity{_safe(ftag)}.png"),
                                         title="Colony profile dissimilarity (all markers, "
                                               "normalized grid)" + (f"\n{ftitle}" if ftitle else ""))
                wb = fda.within_between(D, groups)
                for k, v in key.items():
                    wb[k] = v
                sim_frames.append(wb)
                pd.DataFrame(D, index=labels, columns=labels).to_csv(
                    ctx.table(f"similarity_matrix{_safe(ftag)}.csv"))

    glob = pd.DataFrame(glob_rows)
    if glob.empty:
        logger.warning("compare: fewer than two conditions with data; nothing to test. "
                       "Profiles are still drawn; check conditions.condition_key.")
    outputs = (([glob] if len(glob) else [], "compare_global.csv"),
               ([pd.DataFrame(point_rows)] if point_rows else [], "compare_pointwise.csv"),
               (pair_frames, "compare_pairwise.csv"), (feat_frames, "colony_features.csv"),
               (test_frames, "feature_tests.csv"), (sim_frames, "similarity_within_between.csv"))
    for frames, name in outputs:
        if frames:
            ctx.write_table(pd.concat(frames, ignore_index=True), name)
        elif ctx.has_table(name):
            ctx.table(name).unlink()           # never leave a previous run's result behind

    with open(ctx.table("compare_summary.json"), "w") as f:
        json.dump({"replicate_unit": notes, "n_permutations": n_perm,
                   "global_tests": glob.to_dict("records") if len(glob) else []},
                  f, indent=2, default=_json_default)
    for r in glob_rows:
        logger.info("compare %-10s %-4s %-17s global p = %.4g  (%s; %d %s)", r["marker"],
                    r["grid"], r["readout"], r["global_p"], r["groups"], r["n_replicates"],
                    plots.plural(r["unit"], r["n_replicates"]))


def _global_row(key, grid, readout, marker, unit, res) -> dict:
    sig_x = np.asarray(res["centers"])[res["sig_mask"]]
    return {**key, "grid": grid, "readout": readout, "marker": marker, "unit": unit,
            "groups": " / ".join(res["groups"]), "n_replicates": res["n_replicates"],
            "n_points_tested": res["n_points"], "global_F": res["global_stat"],
            "global_p": res["global_p"], "exact": res["exact"],
            "n_permutations": res["n_permutations"],
            "smallest_possible_p": res.get("min_p", np.nan),
            "n_points_significant": int(res["sig_mask"].sum()),
            "significant_from": float(sig_x.min()) if sig_x.size else np.nan,
            "significant_to": float(sig_x.max()) if sig_x.size else np.nan}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _matrix(unit_curves: pd.DataFrame, marker: str, grid: str, readout: str):
    """``(X, groups, centers)``: replicate × bin matrix for one marker/grid/readout."""
    sub = unit_curves[(unit_curves["marker"] == marker) & (unit_curves["grid"] == grid)]
    if sub.empty or sub[readout].notna().sum() == 0:
        return None, None, None
    wide = sub.pivot_table(index="unit_id", columns="bin_center", values=readout,
                           aggfunc="first", dropna=False)
    groups = (sub.drop_duplicates("unit_id").set_index("unit_id")["condition"]
              .astype(str).reindex(wide.index).to_numpy())
    return wide.to_numpy(np.float64), groups, wide.columns.to_numpy(np.float64)


def _features(curves: pd.DataFrame, edge_band: float) -> pd.DataFrame:
    rows = []
    for (field_id, marker), part in curves.groupby(["field_id", "marker"]):
        um = part[part["grid"] == radial.UM].sort_values("bin")
        nm = part[part["grid"] == radial.NORM].sort_values("bin")
        f = fda.colony_features(um["mean"].to_numpy(float), um["bin_center"].to_numpy(float),
                                nm["mean"].to_numpy(float), nm["bin_center"].to_numpy(float),
                                edge_band)
        rows.append({"field_id": field_id, "marker": marker,
                     "condition": str(part["condition"].iloc[0]),
                     "well": part["well"].iloc[0], "colony": part["colony"].iloc[0], **f})
    return pd.DataFrame(rows)


def _absolute_markers(ctx) -> list[str]:
    """Signaling markers shown in their own intensity units (a.u.), not on a 0-1 or ÷ reference scale."""
    from .signaling import reference as sig_ref
    if not ctx.has_table(sig_ref.TABLE):
        return []
    return sig_ref.absolute_markers(ctx.read_table(sig_ref.TABLE))


def _similarity(curves: pd.DataFrame, order: list[str], exclude=()):
    sub = curves[curves["grid"] == radial.NORM]
    if exclude:
        # A marker in raw a.u. (thousands) would outweigh every 0-1 marker in
        # one RMS distance, so it is left out of this all-marker summary.
        logger.info("similarity: %s left out (intensity in a.u., not on a common scale "
                    "with the other markers)", ", ".join(exclude))
        sub = sub[~sub["marker"].astype(str).isin(set(exclude))]
    if sub.empty:
        return None
    wide = sub.pivot_table(index="field_id", columns=["marker", "bin"], values="mean",
                           aggfunc="first")
    if wide.shape[0] < 2:
        return None
    info = sub.drop_duplicates("field_id").set_index("field_id")
    rank = {c: i for i, c in enumerate(order)}
    ids = sorted(wide.index, key=lambda i: (rank.get(str(info.at[i, "condition"]), 99),
                                            str(info.at[i, "well"]), str(info.at[i, "colony"])))
    wide = wide.loc[ids]
    D = fda.pairwise_l2(wide.to_numpy(np.float64))
    labels = [f"w{info.at[i, 'well']} {plots.short_name(info.at[i, 'colony'])}" for i in ids]
    groups = [str(info.at[i, "condition"]) for i in ids]
    return D, labels, groups


# ---------------------------------------------------------------------------
# 8. Several markers at once: stacked profiles and spatial co-expression
# ---------------------------------------------------------------------------

MULTI_FIG = "multimarker_profiles"
COEX_FIG = "coexpression"
COEX_PAGES, COEX_PANELS = "pages", "panels"   # combined pages | one figure per panel
TRENDS_TABLE = "multimarker_trends.csv"
COEX_COLONY_TABLE = "coexpression_colonies.csv"
COEX_WELL_TABLE = "coexpression_wells.csv"
_METHOD_LABEL = {"spearman": ("Spearman", "ρ"), "pearson": ("Pearson", "r")}


def _remove_tables(ctx, *names) -> None:
    """Never leave a previous run's table behind when this run produced none."""
    for name in names:
        if ctx.has_table(name):
            ctx.table(name).unlink()


def _colony_title(r) -> str:
    title = (f"{r['colony']}: well {r['well']} · {r['condition']} · "
             f"Ø {2 * float(r['radius_um']):.0f} µm")
    if isinstance(r.get("flags"), str) and r["flags"]:
        title += f" · flags: {r['flags']}"
    return title


def _well_groups(frame: pd.DataFrame):
    """``(tag, part)`` per well (and segmentation mode), with a filename tag that
    stays unique when several datasets/plates or modes share a well number."""
    keys = [k for k in ("well_key", "seg_mode") if k in frame]
    plates = frame["well_key"].astype(str).str.rsplit("|", n=1).str[0]
    multi_plate = plates.nunique() > 1
    multi_mode = "seg_mode" in frame and frame["seg_mode"].astype(str).nunique() > 1
    for _, part in frame.groupby(keys, observed=True, dropna=False, sort=True):
        r = part.iloc[0]
        tag = f"w{r['well']}"
        if multi_plate:
            tag = _safe(str(r["well_key"]).rsplit("|", 1)[0].replace("|", "_")) + "_" + tag
        if multi_mode:
            tag += f"_{r['seg_mode']}"
        yield tag, part


@register("multimarker", after=["aggregate", "boundary"],
          config_keys=["multimarker", "profiles", "markers_order", "conditions", "visualization",
                       "boundary", "skip_steps"],
          description="All markers of a colony (µm) or well (normalized) stacked on one axis, "
                      "with a local-trend strip")
def step_multimarker(ctx) -> None:
    for sub in ("colonies", "wells"):
        ctx.reset_figures(f"{MULTI_FIG}/{sub}")
    _set_value_labels(ctx)
    prm = trends_mod.settings(ctx.cfg)
    pprm = radial.settings(ctx.cfg)
    curves = _curves(ctx)
    present = set(curves["marker"].astype(str))
    markers = [m for m in ctx.markers if m in present]
    if len(markers) < 2:
        logger.info("multimarker: %d analysis marker(s); stacked multi-marker figures need "
                    "at least 2. Skipped.", len(markers))
        _remove_tables(ctx, TRENDS_TABLE)
        return

    colony = trends_mod.trend_table(curves[curves["grid"] == radial.UM], ["field_id"], prm)
    wells = ctx.read_table("well_profiles.csv")
    well_keys = [k for k in ("seg_mode", "well_key") if k in wells]
    wmean = trends_mod.trend_table(
        wells[(wells["grid"] == radial.NORM) & (wells["readout"] == "mean")], well_keys, prm)
    cols = ["level", "field_id", "colony", "seg_mode", "well_key", "well", "condition",
            "marker", "grid", "bin", "bin_lo", "bin_hi", "bin_center", "mean", "sem",
            "slope", "z", "direction"]
    table = pd.concat([colony.assign(level="colony"), wmean.assign(level="well")],
                      ignore_index=True)
    ctx.write_table(table[[c for c in cols if c in table]], TRENDS_TABLE)

    figure_info = {"z_crit": float(prm["z_crit"]), "z_max": float(prm["z_display_max"])}
    if bool(prm["colony_figures"]):
        um_ylims = _ylims(ctx, curves[curves["grid"] == radial.UM])
        annotations = _boundary_annotations(ctx)
        n_fig = 0
        for _, part in colony.groupby("field_id"):
            tracks, positive = {}, {}
            for marker, mpart in part.groupby("marker"):
                if mpart["mean"].notna().any():
                    tracks[str(marker)] = pd.DataFrame({
                        "x": mpart["bin_center"], "mean": mpart["mean"],
                        "lo": mpart["ci_lo"], "hi": mpart["ci_hi"]})
                    positive[str(marker)] = pd.DataFrame({
                        "x": mpart["bin_center"], "y": mpart["positive_fraction"],
                        "lo": np.nan, "hi": np.nan})
            if len(tracks) < 2:
                continue
            r = part.iloc[0]
            info = dict(figure_info, title=_colony_title(r), x_label=radial.GRID_LABEL[radial.UM],
                        ylims=um_ylims,
                        borders=annotations.get(r["field_id"], {}).get("borders"),
                        missing=annotations.get(r["field_id"], {}).get("missing"),
                        band=f"band = {pprm['ci']:g}% CI over nuclei in each "
                             f"{float(pprm['bin_um']):g} µm bin")
            trend = part.rename(columns={"bin_center": "x"})[["marker", "x", "z"]]
            if plots.multimarker_profile(markers, tracks, positive, trend, info, ctx.figure(
                    f"{MULTI_FIG}/colonies", f"w{r['well']}_{_safe(r['colony'])}.png")):
                n_fig += 1
        logger.info("multimarker: %d colony figure(s) (µm grid)", n_fig)

    if bool(prm["well_figures"]) and len(wmean):
        colony_norm = curves[curves["grid"] == radial.NORM]
        pos_all = wells[(wells["grid"] == radial.NORM)
                        & (wells["readout"] == "positive_fraction")]
        norm_ylims = _ylims(ctx, colony_norm)       # colony traces are drawn too
        n_fig = 0
        for tag, part in _well_groups(wmean):
            r = part.iloc[0]
            sel = colony_norm["well_key"].astype(str) == str(r["well_key"])
            psel = pos_all["well_key"].astype(str) == str(r["well_key"])
            if "seg_mode" in part:
                sel &= colony_norm["seg_mode"].astype(str) == str(r["seg_mode"])
                psel &= pos_all["seg_mode"].astype(str) == str(r["seg_mode"])
            members = colony_norm[sel]
            tracks, positive, traces = {}, {}, {}
            for marker, mpart in part.groupby("marker"):
                marker = str(marker)
                if not mpart["mean"].notna().any():
                    continue
                tracks[marker] = pd.DataFrame({"x": mpart["bin_center"], "mean": mpart["mean"],
                                               "lo": mpart["ci_lo"], "hi": mpart["ci_hi"]})
                traces[marker] = [(c["bin_center"].to_numpy(), c["mean"].to_numpy())
                                  for _, c in members[members["marker"] == marker]
                                  .sort_values("bin").groupby("field_id")]
                p = pos_all[psel & (pos_all["marker"] == marker)]
                if len(p):
                    positive[marker] = pd.DataFrame({"x": p["bin_center"], "y": p["mean"],
                                                     "lo": p["ci_lo"], "hi": p["ci_hi"]})
            if len(tracks) < 2:
                continue
            n_col = int(members["field_id"].nunique())
            info = dict(figure_info, traces=traces, x_label=radial.GRID_LABEL[radial.NORM],
                        ylims=norm_ylims,
                        title=f"well {r['well']} · {r['condition']} · {n_col} "
                              f"{plots.plural('colony', n_col)} · normalized grid",
                        band=f"bold = well mean, band = {pprm['ci']:g}% CI across colonies;\n"
                             f"thin lines = individual colonies")
            trend = part.rename(columns={"bin_center": "x"})[["marker", "x", "z"]]
            if plots.multimarker_profile(markers, tracks, positive, trend, info,
                                         ctx.figure(f"{MULTI_FIG}/wells", f"{tag}.png")):
                n_fig += 1
        logger.info("multimarker: %d well figure(s) (normalized grid)", n_fig)


def _colony_center(ctx, r) -> tuple[float, float] | None:
    """The fitted colony center (y, x) in pixels, from the image's colony.json."""
    path = _result_path(ctx, r["dataset"], r["result_dir"])
    if path is None:
        return None
    try:
        with open(path / colony_mod.META_NAME) as f:
            cy, cx = json.load(f)["center_yx_px"]
        return float(cy), float(cx)
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _fisher_mean(values) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], np.float64)
    if v.size == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(v, -0.999999, 0.999999)))))


@register("coexpression", after=["profiles"],
          config_keys=["coexpression", "profiles", "markers", "default_space", "thresholds",
                       "qc", "conditions", "markers_order", "stats", "visualization",
                       "signaling"],
          description="Pairwise per-nucleus co-expression against distance from the edge "
                      "(colonies in µm, wells on the normalized grid)")
def step_coexpression(ctx) -> None:
    prm = coex_mod.settings(ctx.cfg)
    for sub in ("colonies", "wells"):
        ctx.reset_figures(f"{COEX_FIG}/{sub}")            # layout before pages/ and panels/
        ctx.reset_figures(f"{COEX_FIG}/{COEX_PAGES}/{sub}")
        for kind in coex_mod.PANEL_KINDS:
            ctx.reset_figures(f"{COEX_FIG}/{COEX_PANELS}/{sub}/{kind}")
    pprm = radial.settings(ctx.cfg)
    ci = float(pprm["ci"])
    gated = ctx.gated if ctx.has_table("thresholds.csv") else None
    long = radial.nucleus_table(ctx.data, gated, ctx.cfg)
    anchors = ctx.read_table("scale_anchors.csv")
    long["scaled"] = radial.apply_scale(long, anchors)
    long["scaled"] = _signaling_relative(ctx, long)
    long["threshold_scaled"] = radial.apply_scale(long, anchors, "threshold")
    wide = coex_mod.wide_tables(long)
    cells = ctx.data.cells                    # index == cell_uid (positional)
    pairs = coex_mod.marker_pairs(wide["value"], cells, ctx.cfg, ctx.markers)
    if not pairs:
        logger.info("coexpression: no stain set has two or more analysis markers measured "
                    "in the same nuclei; pairwise co-expression skipped.")
        _remove_tables(ctx, COEX_COLONY_TABLE, COEX_WELL_TABLE)
        return

    info = colony_info(ctx)
    cells = cells[cells["field_id"].isin(set(info["field_id"]))]
    table = coex_mod.colony_table(cells, pairs, wide, prm, pprm, ci)
    if table.empty:
        logger.warning("coexpression: %d pair(s) but no colony has nuclei with both values.",
                       len(pairs))
        _remove_tables(ctx, COEX_COLONY_TABLE, COEX_WELL_TABLE)
        return
    table = table.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
    ctx.write_table(table, COEX_COLONY_TABLE)
    by = [c for c in ("seg_mode", "well_key", "well", "condition") if c in table]
    wells = coex_mod.well_table(table, by, ci, int(pprm["min_replicates"]))
    ctx.write_table(wells, COEX_WELL_TABLE)
    logger.info("coexpression: %d pair(s) (%s), %d colonies; %s correlation, %g µm and "
                "%d normalized bins", len(pairs),
                ", ".join(coex_mod.pair_label(a, b) for a, b in pairs),
                table["field_id"].nunique(), prm["method"], float(prm["bin_um"]),
                int(prm["n_norm_bins"]))

    name, symbol = _METHOD_LABEL[prm["method"]]
    seed = int((ctx.cfg.get("stats") or {}).get("random_state", 0))
    min_n = int(prm["min_nuclei_per_bin"])
    n_bands = int(prm["n_bands"])
    groups = cells.groupby("field_id").groups

    def points(uids, a, b, coord):
        pts = pd.DataFrame({"va": wide["value"][a].reindex(uids).to_numpy(np.float64),
                            "vb": wide["value"][b].reindex(uids).to_numpy(np.float64),
                            "x": cells.loc[uids, coord].to_numpy(np.float64),
                            "field_id": cells.loc[uids, "field_id"].to_numpy()},
                           index=pd.Index(uids, name="cell_uid"))
        return pts[np.isfinite(pts[["va", "vb", "x"]]).all(axis=1)]

    def profile(frame, lo="corr_ci_lo", hi="corr_ci_hi"):
        out = frame.rename(columns={lo: "corr_lo", hi: "corr_hi",
                                    "frac_ab_ci_lo": "frac_ab_lo", "frac_ab_ci_hi": "frac_ab_hi"})
        return out.sort_values("bin")

    base = {"method_name": name, "method_symbol": symbol}

    def emit(level: str, stem: str, rows: list, page: dict, write_page: bool) -> bool:
        """The combined page (pages/) and/or each panel on its own (panels/<kind>/)."""
        if level in prm["panels"]:
            for row in rows:
                for kind in prm["panel_kinds"]:
                    if (kind in ("double_positive", "map") and not row.get("has_gates")) or \
                            (kind == "map" and row.get("spatial") is None):
                        continue                  # nothing to draw: no empty folders
                    plots.coexpression_panel(kind, row, page, ctx.figure(
                        f"{COEX_FIG}/{COEX_PANELS}/{level}/{kind}",
                        f"{stem}__{_safe(row['a'])}_vs_{_safe(row['b'])}.png"))
        return bool(write_page and plots.coexpression_page(
            rows, page, ctx.figure(f"{COEX_FIG}/{COEX_PAGES}/{level}", f"{stem}.png")))

    if bool(prm["colony_figures"]) or "colonies" in prm["panels"]:
        rng = np.random.default_rng(seed)
        by_id = info.set_index("field_id")
        n_fig = 0
        um = table[table["grid"] == radial.UM]
        for field_id, part in um.groupby("field_id"):
            r = by_id.loc[field_id]
            uids = np.asarray(groups.get(field_id, []))
            center = _colony_center(ctx, r)
            px = float(r["pixel_size_um"]) if "pixel_size_um" in r and pd.notna(
                r["pixel_size_um"]) else np.nan
            rows = []
            for a, b in pairs:
                prof = part[(part["marker_a"] == a) & (part["marker_b"] == b)]
                pts = points(uids, a, b, "dist_edge_um")
                if prof.empty or len(pts) < min_n:
                    continue
                top = max(float(pts["x"].max()), 1.0)
                edges = np.linspace(0.0, top, n_bands + 1)
                band_r = []
                for k in range(n_bands):
                    sel = (pts["x"] >= edges[k]) & (pts["x"] <= edges[k + 1])
                    band_r.append(coex_mod.correlation(pts.loc[sel, "va"], pts.loc[sel, "vb"],
                                                       prm["method"]) if sel.sum() >= min_n
                                  else np.nan)
                shown = pts.sample(n=min(len(pts), int(prm["max_points"])),
                                   random_state=rng.integers(1 << 31))
                has_gates = bool(prof["frac_ab"].notna().any())
                spatial = None
                if has_gates and center is not None and np.isfinite(px):
                    pa = wide["positive"][a].reindex(pts.index).to_numpy(np.float64)
                    pb = wide["positive"][b].reindex(pts.index).to_numpy(np.float64)
                    spatial = pd.DataFrame({
                        "x_um": (cells.loc[pts.index, "cx"].to_numpy(np.float64) - center[1]) * px,
                        "y_um": (cells.loc[pts.index, "cy"].to_numpy(np.float64) - center[0]) * px,
                        "cls": coex_mod.classes(pa, pb)})
                    spatial = spatial[spatial["cls"] != ""]
                rows.append(dict(base, a=a, b=b, pts=shown, profile=profile(prof),
                                 band_edges=edges, band_r=band_r, band_unit="µm",
                                 gate_a=coex_mod.colony_gate(wide, a, pts.index),
                                 gate_b=coex_mod.colony_gate(wide, b, pts.index),
                                 has_gates=has_gates, spatial=spatial,
                                 radius_um=float(r["radius_um"]), coord_max=top,
                                 coord_label="µm from edge",
                                 r_all=coex_mod.correlation(pts["va"], pts["vb"], prm["method"]),
                                 scatter_title=f"all nuclei (n = {len(pts):,})",
                                 corr_title=f"Within {float(prm['bin_um']):g} µm bins "
                                            f"({ci:g}% CI)"))
            if not rows:
                continue
            page = {"title": _colony_title(r), "show_map": any(x["has_gates"] for x in rows)}
            if emit("colonies", f"w{r['well']}_{_safe(r['colony'])}", rows, page,
                    bool(prm["colony_figures"])):
                n_fig += 1
        logger.info("coexpression: %d colony page(s) (µm grid)", n_fig)

    if bool(prm["well_figures"]) or "wells" in prm["panels"]:
        rng = np.random.default_rng(seed)
        norm_w = wells[wells["grid"] == radial.NORM]
        norm_c = table[table["grid"] == radial.NORM]
        n_fig = 0
        for tag, part in _well_groups(norm_w):
            r = part.iloc[0]
            sel = norm_c["well_key"].astype(str) == str(r["well_key"])
            if "seg_mode" in part:
                sel &= norm_c["seg_mode"].astype(str) == str(r["seg_mode"])
            members = norm_c[sel]
            fids = members["field_id"].unique()
            uids = np.concatenate([np.asarray(groups[f]) for f in fids if f in groups]) \
                if len(fids) else np.array([], dtype=np.int64)
            rows = []
            for a, b in pairs:
                prof = part[(part["marker_a"] == a) & (part["marker_b"] == b)]
                pts = points(uids, a, b, "r_norm")
                if prof.empty or len(pts) < min_n:
                    continue
                edges = np.linspace(0.0, 1.0, n_bands + 1)
                band_r = []
                for k in range(n_bands):
                    inside = pts[(pts["x"] >= edges[k]) & (pts["x"] <= edges[k + 1])]
                    band_r.append(_fisher_mean(
                        coex_mod.correlation(g["va"], g["vb"], prm["method"])
                        for _, g in inside.groupby("field_id") if len(g) >= min_n))
                per_colony = [coex_mod.correlation(g["va"], g["vb"], prm["method"])
                              for _, g in pts.groupby("field_id") if len(g) >= min_n]
                pooled = coex_mod.equal_subsample(pts, int(prm["points_per_colony"]), rng)
                pooled = pooled.sample(frac=1.0, random_state=rng.integers(1 << 31))
                take = int(pooled.groupby("field_id").size().min()) if len(pooled) else 0
                traces = [(c["bin_center"].to_numpy(), c["correlation"].to_numpy())
                          for _, c in members[(members["marker_a"] == a)
                                              & (members["marker_b"] == b)]
                          .sort_values("bin").groupby("field_id")]
                rows.append(dict(base, a=a, b=b, pts=pooled,
                                 profile=profile(prof, "correlation_ci_lo", "correlation_ci_hi"),
                                 band_edges=edges, band_r=band_r, band_unit="",
                                 gate_a=coex_mod.colony_gate(wide, a, pts.index),
                                 gate_b=coex_mod.colony_gate(wide, b, pts.index),
                                 has_gates=bool(prof["frac_ab"].notna().any()),
                                 coord_max=1.0, coord_label="normalized distance (0 = edge)",
                                 r_all=_fisher_mean(per_colony), band_symbol=f"mean {symbol}",
                                 scatter_title=f"{take:,} nuclei per colony, mean over "
                                               f"{len(per_colony)} colonies",
                                 corr_traces=traces,
                                 corr_title=f"Mean over colonies ({ci:g}% CI); thin = colonies"))
            if not rows:
                continue
            n_col = len(fids)
            page = {"title": f"well {r['well']} · {r['condition']} · {n_col} "
                             f"{plots.plural('colony', n_col)} · normalized grid · every readout "
                             f"computed per colony, then averaged",
                    "show_map": False}
            if emit("wells", tag, rows, page, bool(prm["well_figures"])):
                n_fig += 1
        logger.info("coexpression: %d well page(s) (normalized grid)", n_fig)


# ---------------------------------------------------------------------------
# 8a. Boundaries and peaks
# ---------------------------------------------------------------------------

BOUNDARY_COLONY_TABLE = "boundary_colonies.csv"
BOUNDARY_FIT_TABLE = "boundary_fits.csv"
BOUNDARY_SUMMARY_TABLE = "boundary_summary.csv"
BOUNDARY_SECTOR_TABLE = "boundary_sectors.csv"
BOUNDARY_REGISTERED_TABLE = "boundary_registered.csv"
BOUNDARY_FIG = "boundary"


@register("boundary", after=["thresholds"],
          config_keys=["boundary", "profiles", "markers", "default_space", "thresholds", "qc",
                       "conditions", "signaling", "stats", "markers_order", "visualization"],
          description="Per colony: boundary position and width of each marker, facing-pair "
                      "crossing, overlap and double-positive fraction, interior signaling "
                      "peak and its offset; sector-bootstrap SE; summary by condition")
def step_boundary(ctx) -> None:
    from .signaling import reference as sig_ref
    from . import roughness as rough_mod
    tables = (BOUNDARY_COLONY_TABLE, BOUNDARY_FIT_TABLE, BOUNDARY_SUMMARY_TABLE,
              BOUNDARY_SECTOR_TABLE, BOUNDARY_REGISTERED_TABLE, "boundary_bin_check.csv",
              "boundary_bin_check_summary.csv")
    ctx.reset_figures(BOUNDARY_FIG)
    ctx.reset_figures(f"{BOUNDARY_FIG}/borders")
    prm = bd_mod.settings(ctx.cfg)
    rprm = dict(rough_mod.settings(ctx.cfg), n_boot=prm["n_boot"],
                random_state=prm["random_state"])
    k_max = bd_mod.LN81 / (0.5 * float(prm["bin_um"]))
    if not bool(prm["enabled"]):
        logger.info("boundary: disabled (boundary.enabled: false).")
        _remove_tables(ctx, *tables)
        return
    pprm = radial.settings(ctx.cfg)
    gated = ctx.gated if ctx.has_table("thresholds.csv") else None
    long = radial.nucleus_table(ctx.data, gated, ctx.cfg)
    # Same scale as the profiles: anchors are recomputed (identical), and signaling
    # markers are put relative to their reference in memory, so this step does not
    # have to run after profiles (which draws its lines).
    anchors = radial.shared_anchors(long, tuple(float(p) for p in pprm["robust_percentiles"]))
    long["scaled"] = radial.apply_scale(long, anchors)
    matches = sig_mod.classify_all([str(m) for m in ctx.data.markers], ctx.cfg)
    if matches:
        ref = sig_ref.reference_table(long, ctx.data.cells, ctx.cfg, matches)
        long["scaled"] = sig_ref.relative_values(long, ref, long["scaled"])
    long["threshold_scaled"] = radial.apply_scale(long, anchors, "threshold")
    wide = coex_mod.wide_tables(long)

    present = set(long["marker"].astype(str))
    peaks = [m for m in ctx.markers if m in matches] if prm["peak_markers"] == "auto" \
        else [str(m) for m in (prm["peak_markers"] or []) if str(m) in present]
    gated_markers = set(long.loc[long["positive"].notna(), "marker"].astype(str))
    if prm["pairs"] == "auto":
        pairs_by_set = bd_mod.auto_pairs(long, gated_markers, set(peaks))
    else:
        pairs_by_set = {}
        stain_of = long.groupby(long["stain_set"].astype(str))["marker"].unique()
        for a, b in prm["pairs"]:
            sets = [st for st, ms in stain_of.items() if {a, b} <= set(map(str, ms))]
            if not sets:
                logger.warning("boundary: pair %s / %s is never stained together; skipped.", a, b)
            for st in sets:
                pairs_by_set.setdefault(st, []).append((a, b))
    for st, pairs in sorted(pairs_by_set.items()):
        logger.info("boundary: stain set %s: facing pair(s) %s", st,
                    ", ".join(f"{a} (edge) / {b} (center)" for a, b in pairs))
    if peaks:
        logger.info("boundary: interior peak of %s", ", ".join(peaks))

    info = colony_info(ctx)
    cells = ctx.data.cells
    cells = cells[cells["field_id"].isin(set(info["field_id"]))
                  & cells["in_colony"].astype(bool)]
    groups = cells.groupby("field_id").groups
    rng = np.random.default_rng(int(prm["random_state"]))
    rows, fits, sector_rows, registered, maps, checks = [], [], [], [], {}, []
    for _, r in info.iterrows():
        uids = np.asarray(groups.get(r["field_id"], []))
        if uids.size < int(prm["min_nuclei"]):
            continue
        c = cells.loc[uids]
        px = float(r["pixel_size_um"]) if pd.notna(r.get("pixel_size_um")) else np.nan
        center = _colony_center(ctx, r)
        if center is None:                       # disc: the centroid is a fair stand-in
            center = (float(c["cy"].mean()), float(c["cx"].mean()))
        theta = np.arctan2(c["cy"].to_numpy(np.float64) - center[0],
                           c["cx"].to_numpy(np.float64) - center[1])
        d = c["dist_edge_um"].to_numpy(np.float64)
        markers = [m for m in wide["value"].columns
                   if wide["value"][m].reindex(uids).notna().any()]
        values = {m: wide["value"][m].reindex(uids).to_numpy(np.float64) for m in markers}
        positive = {m: wide["positive"][m].reindex(uids).to_numpy(np.float64) for m in markers}
        spacing = bd_mod.nn_spacing_um(c[["cx", "cy"]].to_numpy(np.float64) * px)
        res = bd_mod.measure_colony(d, theta, values, positive,
                                    pairs_by_set.get(str(r["stain_set"]), []),
                                    [m for m in peaks if m in markers], prm, spacing, rng)
        if res.empty:
            continue
        colony_fits = res.attrs["fits"]            # pd.concat below drops attrs
        if prm["bin_check"]:
            chk = bd_mod.bin_check(d, theta, values, positive,
                                   pairs_by_set.get(str(r["stain_set"]), []),
                                   [m for m in peaks if m in markers], prm, spacing)
            if len(chk):
                checks.append(chk.assign(field_id=r["field_id"]))
        if bool(rprm["enabled"]):
            accepted = res[(res["kind"] == "marker") & (res["readout"] == "positive")
                           & (res["metric"] == "x50_um") & res["ok"].astype(bool)]
            colony_map = {"x_um": (c["cx"].to_numpy(np.float64) - center[1]) * px,
                          "y_um": (c["cy"].to_numpy(np.float64) - center[0]) * px,
                          "radius_um": float(r["radius_um"]), "markers": {}}
            extra = []
            for m in accepted["subject"].astype(str):
                fit = colony_fits.get(f"{m}|positive")
                if fit is None:
                    continue
                met, sec, prof = rough_mod.colony_roughness(
                    d, theta, positive[m], values[m], fit[0], fit[1], rprm, k_max)
                se = met.pop("_se", {})
                enough = met.get("n_sectors_ok", 0) >= int(rprm["min_sectors"])
                for metric, v in met.items():
                    extra.append({"kind": "roughness", "subject": m, "readout": "positive",
                                  "metric": metric, "value": float(v),
                                  "se": se.get(metric, np.nan), "boot_support": np.nan,
                                  "ok": bool(enough and np.isfinite(v))})
                sector_rows.append(sec.assign(field_id=r["field_id"], marker=m))
                if len(prof):
                    registered.append(prof.assign(field_id=r["field_id"], marker=m))
                colony_map["markers"][m] = {"positive": positive[m] > 0.5,
                                            "known": np.isfinite(positive[m]), "sectors": sec,
                                            "metrics": met}
            if extra:
                res = pd.concat([res, pd.DataFrame(extra)], ignore_index=True)
            if colony_map["markers"]:
                maps[r["field_id"]] = colony_map
        res.insert(0, "field_id", r["field_id"])
        res["spacing_um"] = spacing
        rows.append(res)
        for key, f in colony_fits.items():
            m, ro = key.split("|")
            fits.append({"field_id": r["field_id"], "marker": m, "readout": ro,
                         "x50_um": f[0], "k": f[1],
                         "lo": f[2] if len(f) > 2 else 0.0, "hi": f[3] if len(f) > 2 else 1.0})
    if not rows:
        logger.warning("boundary: no colony had a fittable transition or peak.")
        _remove_tables(ctx, *tables)
        return
    table = pd.concat(rows, ignore_index=True)
    table = table.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
    ctx.write_table(table, BOUNDARY_COLONY_TABLE)
    ctx.write_table(pd.DataFrame(fits), BOUNDARY_FIT_TABLE)
    _bin_check_outputs(ctx, checks, table)
    if sector_rows:
        ctx.write_table(pd.concat(sector_rows, ignore_index=True), BOUNDARY_SECTOR_TABLE)
    else:
        _remove_tables(ctx, BOUNDARY_SECTOR_TABLE)
    reg = pd.concat(registered, ignore_index=True) if registered else pd.DataFrame()
    if len(reg):
        reg = reg.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
        ctx.write_table(reg, BOUNDARY_REGISTERED_TABLE)
    else:
        _remove_tables(ctx, BOUNDARY_REGISTERED_TABLE)

    unit, _ = radial.choose_unit(info, ctx.cfg)
    cond_keys = condition_columns(table, ctx.cfg.get("conditions") or {})
    by = [c for c in dict.fromkeys(["seg_mode", "condition"] + cond_keys + list(ctx.facet_keys))
          if c in table]
    # Both levels: wells are the replicate the pipeline chose (``primary``), but a
    # metric defined in one stain panel often has one well per condition, where
    # only the spread across colonies (technical replicates) is available.
    exp_col = radial.experiment_column(table, ctx.cfg)
    levels = ["colony"] + [lvl for lvl in ("well", "experiment")
                           if lvl == unit or (lvl == "well" and unit == "experiment")]
    summary = pd.concat([bd_mod.summarize(table, by, lvl, float(pprm["ci"]), exp_col).assign(
        primary=(lvl == unit)) for lvl in levels], ignore_index=True)
    ctx.write_table(summary, BOUNDARY_SUMMARY_TABLE)
    n_ok = int(table.loc[table["ok"].astype(bool), "field_id"].nunique())
    logger.info("boundary: %d colonies measured (%d with at least one accepted value); "
                "summary over %ss", table["field_id"].nunique(), n_ok, unit)

    order = plots.ordered_conditions(table["condition"].astype(str).unique(), ctx.cfg,
                                     ctx.control_conditions)
    stain_markers = long.groupby(long["stain_set"].astype(str))["marker"].unique()
    stain_labels = {st: " · ".join(m for m in ctx.markers if m in set(map(str, ms)))
                    for st, ms in stain_markers.items()}
    x_label = _condition_axis_label(ctx)
    for st, pairs in pairs_by_set.items():
        for a, b in pairs:
            subject = f"{a} / {b}"
            part = table[(table["subject"].isin([subject, a, b]))
                         & table["stain_set"].astype(str).eq(st)]
            plots.boundary_metrics(part, bd_mod.pair_panels(a, b), order, unit,
                                   ctx.figure(BOUNDARY_FIG, f"pair_{_safe(a)}_{_safe(b)}.png"),
                                   title=f"{a} (edge) / {b} (center) boundary, per colony",
                                   x_label=x_label, controls=ctx.control_conditions,
                                   experiment_col=exp_col)
    for m in peaks:
        part = table[(table["kind"] == "peak") & (table["subject"] == m)]
        if part.empty:
            continue
        plots.boundary_metrics(part, bd_mod.peak_panels(part), order, unit,
                               ctx.figure(BOUNDARY_FIG, f"peak_{_safe(m)}.png"),
                               title=f"{m}: interior peak, per colony", x_label=x_label,
                               controls=ctx.control_conditions, stain_labels=stain_labels,
                               experiment_col=exp_col)
    rough = table[table["kind"] == "roughness"]
    for m in sorted(rough["subject"].astype(str).unique()):
        plots.boundary_metrics(rough[rough["subject"] == m], bd_mod.roughness_panels(m), order,
                               unit, ctx.figure(BOUNDARY_FIG, f"roughness_{_safe(m)}.png"),
                               title=f"{m}: border roughness and angular registration, "
                                     f"per colony", x_label=x_label,
                               controls=ctx.control_conditions, stain_labels=stain_labels,
                               experiment_col=exp_col)
        if len(reg):
            plots.registered_profiles(reg[reg["marker"] == m], m, order,
                                      ctx.figure(BOUNDARY_FIG, f"registered_{_safe(m)}.png"),
                                      rough[rough["subject"] == m], controls=ctx.control_conditions)
    by_id = info.set_index("field_id")
    for fid, cmap in maps.items():
        r = by_id.loc[fid]
        plots.border_map(cmap, _colony_title(r), ctx.figure(
            f"{BOUNDARY_FIG}/borders", f"w{r['well']}_{_safe(r['colony'])}.png"))


# ---------------------------------------------------------------------------
# 8c. Effects, trends, variance components and power
# ---------------------------------------------------------------------------

EFFECT_TABLES = ("effects_variance.csv", "effects_power.csv", "effects_vs_reference.csv",
                 "effects_trends.csv")
EFFECTS_FIG = "effects"
_UNIT_LADDER = ["experiment", "well", "colony"]


def _colony_scalars(ctx, info: pd.DataFrame) -> pd.DataFrame:
    """Every accepted per-colony number, long: field_id, family, subject, readout,
    metric, metric_id, value, plus the colony's condition columns."""
    frames = []
    if "boundary" not in ctx.skipped and ctx.has_table(BOUNDARY_COLONY_TABLE):
        b = ctx.read_table(BOUNDARY_COLONY_TABLE)
        b = b[b["ok"].astype(bool) & ~b["metric"].isin(ef_mod.SKIP_METRICS)]
        frames.append(b.assign(family=b["kind"])[["field_id", "family", "subject", "readout",
                                                  "metric", "value"]])
    if "compare" not in ctx.skipped and ctx.has_table("colony_features.csv"):
        f = ctx.read_table("colony_features.csv").drop_duplicates(["field_id", "marker"])
        feats = [c for c in ("edge_value", "center_value", "mean_value", "peak_um",
                             "transition_um") if c in f]
        f = f.melt(id_vars=["field_id", "marker"], value_vars=feats, var_name="metric")
        frames.append(f.rename(columns={"marker": "subject"}).assign(family="feature",
                                                                     readout="profile"))
    if not frames:
        return pd.DataFrame()
    sc = pd.concat(frames, ignore_index=True)
    sc = sc[np.isfinite(pd.to_numeric(sc["value"], errors="coerce"))]
    sc["value"] = sc["value"].astype(float)
    sc["metric_id"] = (sc["family"].astype(str) + "|" + sc["subject"].astype(str) + "|"
                       + sc["readout"].astype(str) + "|" + sc["metric"].astype(str))
    return sc.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")


def _key_frame(frame: pd.DataFrame, cols: list[str]) -> list[pd.Series]:
    """String group keys ('' where missing): grouping a categorical column that
    holds NaN (a plate-map column set on only some wells) fails in pandas."""
    return [frame[c].astype(object).where(frame[c].notna(), "").astype(str).rename(c)
            for c in cols]


def _numeric_column(frame: pd.DataFrame, col: str) -> pd.Series | None:
    vals = pd.to_numeric(frame[col], errors="coerce")
    present = frame[col].notna()
    return vals if present.any() and vals[present].notna().all() else None


def _units_at(part, by, unit, exp_col, need: int = 2):
    """Replicate values at *unit*, stepping down the ladder until every condition
    in *part* has ≥ *need* replicates. Returns ``(values, unit_used)``."""
    start = _UNIT_LADDER.index(unit)
    for level in _UNIT_LADDER[start:]:
        if level == "experiment" and not exp_col:
            continue
        vals = ef_mod.unit_values(part, by, level, exp_col)
        if len(vals) and vals.groupby("condition").size().min() >= need:
            return vals, level
    return ef_mod.unit_values(part, by, "colony", exp_col), "colony"


@register("effects", after=["boundary", "compare"],
          config_keys=["effects", "stats", "conditions", "boundary", "compare", "profiles",
                       "skip_steps"],
          description="Per-colony numbers → variance components (colony / well / experiment), "
                      "effects vs a reference with CI, trends over numeric condition columns, "
                      "and power / minimum detectable differences")
def step_effects(ctx) -> None:
    ctx.reset_figures(EFFECTS_FIG)
    prm = ef_mod.settings(ctx.cfg)
    if not bool(prm["enabled"]):
        _remove_tables(ctx, *EFFECT_TABLES)
        return
    info = colony_info(ctx)
    sc = _colony_scalars(ctx, info)
    if sc.empty:
        logger.info("effects: no per-colony numbers (boundary / compare features); skipped.")
        _remove_tables(ctx, *EFFECT_TABLES)
        return
    unit, note = radial.choose_unit(info, ctx.cfg)
    exp_col = radial.experiment_column(info, ctx.cfg)
    cond_keys = condition_columns(sc, ctx.cfg.get("conditions") or {})
    rng = np.random.default_rng(int(prm["random_state"]))
    logger.info("effects: %d metrics over %d colonies; %s", sc["metric_id"].nunique(),
                sc["field_id"].nunique(), note)
    ident = ["metric_id", "family", "subject", "readout", "metric"]

    # Variance components and power, per metric.
    var_rows, power_rows = [], []
    targets = prm["power"].get("targets") or []
    for key, part in sc.groupby(ident, observed=True, dropna=False):
        vc = ef_mod.variance_components(part, ["condition"], exp_col)
        per_well = part.groupby("well_key").size()
        wells = part.groupby(["condition"] + ([exp_col] if exp_col else []))["well_key"].nunique()
        row = dict(zip(ident, key), **vc, wells_per_experiment=float(wells.median()),
                   colonies_per_well=float(per_well.median()))
        var_rows.append(row)
        ref_mean = float(part["value"].mean())
        for pr in ef_mod.power_table(vc, prm, row["wells_per_experiment"],
                                     row["colonies_per_well"], ref_mean):
            out = dict(zip(ident, key), **pr)
            for t in ef_mod.match_targets(key[0], key[4], targets):
                delta = float(t["delta"]) if t.get("delta") is not None else \
                    float(t.get("delta_rel", np.nan)) * abs(ref_mean)
                var = ef_mod.group_mean_var(pr["sd_experiment_used"], vc["sd_well"],
                                            vc["sd_colony"], pr["experiments"],
                                            pr["wells_per_experiment"], pr["colonies_per_well"])
                # A permutation test that cannot reach alpha has no power at all.
                out.update(target_delta=delta, power_at_target=(ef_mod.power_for(
                    delta, var, pr["experiments"], float(prm["power"]["alpha"]))
                    if pr["permutation_can_reach_alpha"] else 0.0))
            power_rows.append(out)
    ctx.write_table(pd.DataFrame(var_rows), "effects_variance.csv")
    ctx.write_table(pd.DataFrame(power_rows), "effects_power.csv")

    # Trends over numeric condition columns.
    if prm["trend_columns"] == "auto":
        trend_cols = [c for c in cond_keys if _numeric_column(sc, c) is not None
                      and sc[c].nunique() >= int(prm["min_levels"])]
    else:
        trend_cols = [c for c in (prm["trend_columns"] or []) if c in sc]
    trend_rows = []
    for col in trend_cols:
        others = [k for k in cond_keys if k != col]
        for key, part in sc.groupby(_key_frame(sc, ident + others), observed=True):
            if part[col].nunique() < int(prm["min_levels"]):
                continue
            vals, used = _units_at(part, ["condition", col], unit, exp_col)
            x = pd.to_numeric(vals[col], errors="coerce").to_numpy(float)
            scale = ef_mod.choose_scale(np.unique(x[np.isfinite(x)]), prm["scale"])
            res = ef_mod.trend(x, vals["value"].to_numpy(float), scale, int(prm["n_boot"]),
                               int(prm["n_perm"]), rng,
                               exponential=col in (prm["exponential_columns"] or []))
            group = " | ".join(f"{k}={v}" for k, v in zip(others, key[len(ident):])
                               if v != "") or "all"
            trend_rows.append(dict(zip(ident, key[:len(ident)]), column=col, group=group,
                                   unit=used, pseudo_replicated=(used == "colony"
                                                                 and unit != "colony"), **res))
    trends = pd.DataFrame(trend_rows)
    if len(trends):
        ctx.write_table(trends, "effects_trends.csv")
    else:
        _remove_tables(ctx, "effects_trends.csv")

    # Effects against a reference condition.
    eff_rows = []
    for key, part in sc.groupby(ident, observed=True, dropna=False):
        groups = [("all", part)]
        ref_of = {}
        if prm["reference"] == "auto" and trend_cols:
            col = trend_cols[0]
            others = [k for k in cond_keys if k != col]
            groups = list(part.groupby(_key_frame(part, others))) if others else [("all", part)]
            for gkey, g in groups:
                xs = pd.to_numeric(g[col], errors="coerce")
                ref_of[str(gkey)] = str(g.loc[xs.idxmin(), "condition"]) if xs.notna().any() \
                    else None
        elif prm["reference"] not in (None, "auto"):
            ref_of = {"all": str(prm["reference"])}
        for gkey, g in groups:
            ref = ref_of.get(str(gkey))
            if ref is None or ref not in set(g["condition"].astype(str)):
                continue
            for cond in sorted(set(g["condition"].astype(str)) - {ref}):
                pair = g[g["condition"].astype(str).isin([cond, ref])]
                vals, used = _units_at(pair, ["condition"], unit, exp_col)
                a = vals.loc[vals["condition"].astype(str) == cond, "value"].to_numpy(float)
                r = vals.loc[vals["condition"].astype(str) == ref, "value"].to_numpy(float)
                eff_rows.append(dict(zip(ident, key), condition=cond, reference=ref, unit=used,
                                     pseudo_replicated=(used == "colony" and unit != "colony"),
                                     **ef_mod.effect(a, r, int(prm["n_perm"]), rng)))
    effects = pd.DataFrame(eff_rows)
    if len(effects):
        ctx.write_table(effects, "effects_vs_reference.csv")
    else:
        _remove_tables(ctx, "effects_vs_reference.csv")

    # Figures: boundary families only (features stay in the tables).
    fams = set(prm["figure_families"] or [])
    for (fam, subject), part in sc[sc["family"].isin(fams)].groupby(["family", "subject"]):
        tag = f"{fam}_{_safe(subject)}"
        if len(trends):
            t = trends[(trends["family"] == fam) & (trends["subject"] == subject)]
            if len(t):
                plots.trend_panels(part, t, unit, exp_col,
                                   ctx.figure(EFFECTS_FIG, f"trend_{tag}.png"),
                                   title=f"{subject}: trend with "
                                         f"{', '.join(sorted(set(t['column'])))}")
        if len(effects):
            e = effects[(effects["family"] == fam) & (effects["subject"] == subject)]
            if len(e):
                plots.effect_forest(e, ctx.figure(EFFECTS_FIG, f"vs_reference_{tag}.png"),
                                    title=f"{subject}: difference from the reference condition")
        pw = pd.DataFrame(power_rows)
        pw = pw[(pw["family"] == fam) & (pw["subject"] == subject)]
        if len(pw):
            plots.power_curves(pw, ctx.figure(EFFECTS_FIG, f"power_{tag}.png"),
                               title=f"{subject}: minimum detectable difference "
                                     f"(α = {prm['power']['alpha']}, power = "
                                     f"{prm['power']['power']})")


# ---------------------------------------------------------------------------
# 8b. Signaling summary
# ---------------------------------------------------------------------------

SIG_COLONY_TABLE = "signaling_colonies.csv"
CENTER_R_NORM = 0.75          # 'center' = the inner quarter of the radius


@register("signaling_summary", after=["profiles"],
          config_keys=["signaling", "profiles", "compare", "conditions", "markers_order",
                       "visualization"],
          description="One signaling value per colony (whole, edge band, center) by condition")
def step_signaling_summary(ctx) -> None:
    from .signaling import reference as sig_ref
    matches = sig_mod.classify_all([str(m) for m in ctx.data.markers], ctx.cfg)
    if not matches:
        logger.info("signaling_summary: no signaling marker; skipped.")
        _remove_tables(ctx, SIG_COLONY_TABLE)
        return
    gated = ctx.gated if ctx.has_table("thresholds.csv") else None
    long = radial.nucleus_table(ctx.data, gated, ctx.cfg)
    stain_labels = {}
    if "stain_set" in long:                  # legend: what else each panel stained
        others = long.loc[~long["marker"].astype(str).isin(set(matches)), ["stain_set", "marker"]]
        for stain, m in others.groupby(others["stain_set"].astype(str))["marker"]:
            names = [x for x in ctx.markers if x in set(m.astype(str))]
            stain_labels[stain] = " · ".join(names)
    long = long[long["marker"].astype(str).isin(set(matches))].copy()
    anchors = ctx.read_table("scale_anchors.csv")
    long["scaled"] = radial.apply_scale(long, anchors)
    long["scaled"] = _signaling_relative(ctx, long)
    edge = float((ctx.cfg.get("compare") or {}).get("edge_band_um", 50.0))
    long["_edge"] = long["dist_edge_um"].to_numpy(np.float64) <= edge
    long["_center"] = long["r_norm"].to_numpy(np.float64) >= CENTER_R_NORM

    def med(mask_col=None):
        frame = long if mask_col is None else long[long[mask_col]]
        return frame.groupby(["field_id", "marker"], observed=True)["scaled"].median()

    table = pd.DataFrame({"whole": med(), "edge": med("_edge"), "center": med("_center"),
                          "n_nuclei": long.groupby(["field_id", "marker"],
                                                   observed=True)["scaled"].size()}).reset_index()
    table["marker"] = table["marker"].astype(str)
    info = colony_info(ctx)
    table = table.merge(info[_info_columns(ctx, info)], on="field_id", how="inner")
    ref = ctx.read_table(sig_ref.TABLE) if ctx.has_table(sig_ref.TABLE) else pd.DataFrame()
    if len(ref):
        unit = {m: lab for m, (lab, _) in sig_ref.unit_labels(ref).items()}
        table["unit"] = table["marker"].map(unit)
    table["readout"] = table["marker"].map({m: x.readout for m, x in matches.items()})
    ctx.write_table(table, SIG_COLONY_TABLE)

    ctx.reset_figures("signaling_summary")
    reference = ctx._conditions_for_mask(sig_ref.reference_mask(ctx.data.cells, ctx.cfg))
    order = plots.ordered_conditions(table["condition"].astype(str).unique(), ctx.cfg,
                                     reference)
    markers = [m for m in ctx.markers if m in matches and m in set(table["marker"])]
    for suffix, label, part in _strata(table):
        plots.signaling_summary(part, markers, order,
                                ctx.figure("signaling_summary", f"signaling_by_condition{suffix}.png"),
                                controls=reference,
                                edge_label=f"edge band (0-{edge:g} µm)",
                                center_label=f"center (inner {100 * (1 - CENTER_R_NORM):.0f}% "
                                             f"of radius)",
                                x_label=_condition_axis_label(ctx), stain_labels=stain_labels)


_UNIT_SUFFIXES = {"ngmL": "ng/mL", "ngml": "ng/mL", "ugmL": "µg/mL", "uM": "µM", "nM": "nM",
                  "mM": "mM", "pM": "pM", "h": "h", "hr": "h", "min": "min", "um": "µm",
                  "pct": "%"}


def _condition_axis_label(ctx) -> str:
    """Axis label for the condition axis: ``conditions.axis_label`` or the key columns.

    ``bmp4_ngmL`` reads ``bmp4 (ng/mL)``; several keys are joined with `` | ``
    in the same order as the condition labels.
    """
    cond = ctx.cfg.get("conditions") or {}
    if cond.get("axis_label"):
        return str(cond["axis_label"])
    parts = []
    for key in ctx.condition_keys:
        head, _, tail = str(key).rpartition("_")
        parts.append(f"{head} ({_UNIT_SUFFIXES[tail]})" if head and tail in _UNIT_SUFFIXES
                     else str(key))
    return " | ".join(parts) if parts else "condition"


# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------

@register("report", after=["compare", "multimarker", "coexpression", "signaling_summary",
                          "boundary", "effects"],
          config_keys=["report", "skip_steps"],
          description="Single scrollable HTML contact sheet of every figure and key table")
def step_report(ctx) -> None:
    from .report import build_report
    build_report(ctx)


# ---------------------------------------------------------------------------

def _safe(name) -> str:
    """Filename-safe label ('NKX2.5' → 'NKX2-5')."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(name))


def _short(readout: str) -> str:
    return {"mean": "intensity", "positive_fraction": "positive"}.get(readout, readout)


# ---------------------------------------------------------------------------
# Optional steps: what `downstream.skip_steps` may leave out
# ---------------------------------------------------------------------------
# The core chain (colony → signaling → assemble → qc → thresholds → profiles →
# aggregate → report) always runs: every readout depends on it. These steps add
# readouts on top and may be skipped. A skipped step's earlier outputs are moved
# to <analysis>/_skipped/<step>/ so the report never shows a stale figure, and
# the steps that read them (profiles, multimarker, effects) ignore them.
OPTIONAL_STEPS = {
    # step: (figure folders, table name patterns)
    "boundary": ([BOUNDARY_FIG], ["boundary_*.csv"]),
    "coexpression": ([COEX_FIG], [COEX_COLONY_TABLE, COEX_WELL_TABLE]),
    "multimarker": ([MULTI_FIG], [TRENDS_TABLE]),
    "compare": (["compare"], ["compare_*.csv", "compare_summary.json", "colony_features.csv",
                              "feature_tests.csv", "similarity_*.csv"]),
    "effects": ([EFFECTS_FIG], ["effects_*.csv"]),
    "signaling_summary": (["signaling_summary"], [SIG_COLONY_TABLE]),
}
SKIP_DEPENDANTS = {"effects": {"boundary", "compare"}}   # effects needs at least one of these


def resolve_skips(skip) -> set[str]:
    """Validate ``downstream.skip_steps``; also skip a step left with no input."""
    skip = {str(s).strip() for s in (skip or []) if str(s).strip()}
    bad = sorted(skip - set(OPTIONAL_STEPS))
    if bad:
        raise ValueError(f"downstream.skip_steps: {bad} cannot be skipped. Optional steps: "
                         f"{sorted(OPTIONAL_STEPS)} (the rest is the core of every readout).")
    for step, needs in SKIP_DEPENDANTS.items():
        if step not in skip and needs <= skip:
            logger.info("skip_steps: %s needs %s; skipping it too.", step, " or ".join(sorted(needs)))
            skip.add(step)
    return skip


def set_aside(out_dir: Path, step: str) -> None:
    """Move a skipped step's previous figures and tables to ``_skipped/<step>/``."""
    import shutil
    from .registry import checkpoint_path
    figs, tables = OPTIONAL_STEPS[step]
    dest = Path(out_dir) / "_skipped" / step
    moved = []
    for folder in figs:
        src = Path(out_dir) / "figures" / folder
        if src.exists():
            target = dest / "figures" / folder
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
            moved.append(f"figures/{folder}")
    for pattern in tables:
        for src in sorted((Path(out_dir) / "tables").glob(pattern)):
            (dest / "tables").mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest / "tables" / src.name))
            moved.append(f"tables/{src.name}")
    ck = checkpoint_path(out_dir, step)
    if ck.exists():
        ck.unlink()
    if moved:
        logger.info("[skip] %-14s skipped by downstream.skip_steps; its earlier outputs moved "
                    "to %s (%d files/folders)", step, dest, len(moved))


__all__ = ["OPTIONAL_STEPS", "resolve_skips", "set_aside", "step_colony", "step_assemble", "step_qc", "step_thresholds", "step_profiles",
           "step_aggregate", "step_compare", "step_multimarker", "step_coexpression",
           "step_report", "step_signaling", "step_signaling_summary", "step_boundary", "step_effects",
           "colony_info"]
