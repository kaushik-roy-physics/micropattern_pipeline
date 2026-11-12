#!/usr/bin/env python3
"""
Micropattern radial-profile pipeline: images of micropatterned colonies to
radial marker profiles (µm from the edge, and normalized), compared statistically
across colonies, wells and conditions.

Five subcommands, each independently runnable and independently checkpointed:

  process     Discover images and run the per-image steps:
                load     →  read image, MIP vs z-stack, channel TIFFs + metadata
                segment  →  Cellpose-SAM nuclear segmentation (2-D/3-D) + QC overlays
                extract  →  per-nucleus intensities (raw + DAPI-normalized) + centroids

  init-plate-map
              Scan each dataset folder and write a heavily commented starter
              `plate_map.yaml` into it, pre-filled with the acquisition mode,
              wells and channels actually found. Run this first, edit the file,
              then run `check`.

  manifest    Annotate images with plate well + experimental condition from each
              acquisition folder's `plate_map.yaml`, writing `manifest.parquet`.

  downstream  CPU-only analysis, also step-wise:
                colony     main colony, edge, outliers, distance from the edge
                assemble   one cell table for the whole experiment
                qc         colony geometry + staining flags, overlay galleries
                thresholds positivity gates (for % positive profiles)
                profiles   shared-scale profile per colony, µm + normalized grids
                aggregate  well / condition mean profiles
                compare    functional permutation tests, pairwise FDR, features,
                           colony similarity
                report     one HTML page with every figure

  check       Validate the config + plate maps against the files on disk without
              processing anything: every image maps to a well, every well has a
              condition, every image gets a stain set. Run this first.

One config, many datasets
-------------------------
`config.yaml` carries a `datasets:` list; each entry is deep-merged over
`defaults:`, so a 2-D MIP folder and a 3-D z-stack folder (or one folder per
antibody panel) live in the same file. Every subcommand loops over them and
`--datasets` selects a subset. Configs without a `datasets:` list still work as
a single dataset built from `input.directory`.

Usage examples
--------------
  python main.py                                   # process, all steps, all datasets
  python main.py --steps load                      # just load/convert images
  python main.py process --datasets zmax_stain1    # one dataset
  python main.py --seg-mode 3d                     # override the config's mode
  python main.py --force-step segment              # recompute one step
  python main.py process --seq 0-9 15              # only these Seq indices
  python main.py process --wells 1-3               # only these wells
  python main.py init-plate-map                    # write starter plate maps
  python main.py manifest                          # build every dataset's manifest
  python main.py check                             # validate first (no GPU, no writes)
  python main.py downstream                        # full downstream analysis
  python main.py downstream --steps compare report # redo the statistics + report
  python main.py downstream --force-step colony    # re-detect colonies (new settings)
  python main.py downstream --list-steps
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import provenance  # noqa: E402
from pipeline_config import load_config, resolve_datasets  # noqa: E402


LOG_NAME = "pipeline.log"


def setup_logging(level: str, log_dir=None) -> logging.Logger:
    """Log to stdout and to a file that lives with the experiment.

    *log_dir* is the config file's own directory, so each experiment folder
    accumulates its own ``pipeline.log`` next to the data it describes, rather
    than one landing in whatever directory the command happened to be typed in.

    A directory that cannot be written (a read-only share, a mistyped
    ``--config``) falls back to the working directory and then to stdout only:
    losing the log file is never a reason to abandon a run.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    refused: list[str] = []

    for directory in ([log_dir] if log_dir else []) + [Path.cwd()]:
        path = Path(directory) / LOG_NAME
        try:
            handlers.append(logging.FileHandler(path))
            break
        except OSError as exc:
            refused.append(f"{path} ({exc.strerror or exc})")

    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt,
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers, force=True)
    log = logging.getLogger("pipeline")
    if refused and len(handlers) > 1:
        log.warning("could not write %s: logging to %s instead",
                    refused[0], handlers[-1].baseFilename)
    elif refused:
        log.warning("no log file (%s): this run is logged to stdout only",
                    "; ".join(refused))
    return log


# ---------------------------------------------------------------------------
# Step registry (per-image pipeline)
# ---------------------------------------------------------------------------

STEP_ORDER = ["load", "segment", "extract"]


def build_steps(cfg: dict):
    from image_processing import ImageLoader, IntensityExtractor, Segmentation
    return {
        "load": ImageLoader(cfg),
        "segment": Segmentation(cfg),
        "extract": IntensityExtractor(cfg),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SUBCOMMANDS = {"process", "manifest", "downstream", "check", "init-plate-map"}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subset of dataset names from the config (default: all)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seq", nargs="+", default=None,
                   help="Only images with these Seq indices, e.g. --seq 0-9 15 "
                        "(overrides input.select.seq; default: all images)")
    p.add_argument("--wells", nargs="+", default=None,
                   help="Only images from these wells, e.g. --wells 1-3 or C1-C4 "
                        "(overrides input.select.wells; default: all wells)")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Micropattern radial-profile pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=provenance.summary())
    sub = p.add_subparsers(dest="command")

    proc = sub.add_parser("process", help="Run the per-image load/segment/extract steps")
    _add_common(proc)
    _add_selection(proc)
    proc.add_argument("--steps", nargs="+", choices=STEP_ORDER, default=STEP_ORDER,
                      help="Steps to run, in order (default: all)")
    proc.add_argument("--seg-mode", choices=["2d", "3d"], default=None, dest="seg_mode",
                      help="Override segmentation.mode for every selected dataset")
    proc.add_argument("--force", action="store_true",
                      help="Recompute all selected steps, ignoring checkpoints")
    proc.add_argument("--force-step", nargs="+", choices=STEP_ORDER, default=[],
                      dest="force_step", help="Recompute specific step(s) only")

    man = sub.add_parser("manifest", help="Build the well/condition manifest from plate maps")
    _add_common(man)
    man.add_argument("--data-dir", default=None, dest="data_dir",
                     help="Override the dataset directory (single-dataset use)")
    man.add_argument("--plate-map", default=None, dest="plate_map",
                     help="Use a single explicit plate_map.yaml instead of auto-discovery")
    man.add_argument("--out", default=None, dest="out_basename",
                     help="Output file stem written at the dataset root (default: 'manifest')")

    init = sub.add_parser("init-plate-map",
                          help="Write a commented starter plate_map.yaml into each dataset folder")
    _add_common(init)
    init.add_argument("--data-dir", default=None, dest="data_dir",
                      help="Write into this directory instead of the dataset directories")
    init.add_argument("--plate-id", default=None, dest="plate_id",
                      help="plate_id to write (default: the folder name). Give sibling "
                           "panel folders the SAME value to allow cross-panel pooling")
    init.add_argument("--mode", default=None,
                      choices=["generate", "predefined", "spinning_disk"],
                      help="Force the acquisition mode instead of detecting it")
    init.add_argument("--force", action="store_true",
                      help="Overwrite an existing plate_map.yaml")

    check = sub.add_parser("check", help="Validate config + plate maps against the files on disk")
    _add_common(check)
    _add_selection(check)

    down = sub.add_parser("downstream", help="Run the downstream analysis + figures")
    _add_common(down)
    _add_selection(down)
    down.add_argument("--steps", nargs="+", default=None,
                      help="Downstream steps to run (default: all; see --list-steps)")
    down.add_argument("--force", action="store_true",
                      help="Recompute every step, ignoring checkpoints")
    down.add_argument("--force-step", nargs="+", default=[], dest="force_step",
                      help="Recompute specific step(s) only")
    down.add_argument("--out", default=None, dest="out_dir",
                      help="Analysis output directory (default: downstream.results_dir)")
    down.add_argument("--list-steps", action="store_true", dest="list_steps",
                      help="List the registered downstream steps and exit")
    down.add_argument("--seg-mode", choices=["auto", "2d", "3d", "both"], default=None,
                      dest="seg_mode",
                      help="Which segmentation results to analyze (overrides "
                           "downstream.seg_mode): 2d, 3d, both, or auto = whatever exists")

    # Default to 'process' so legacy invocations (`main.py --steps ...`) still work.
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in {"-h", "--help", "--version"}):
        argv = ["process"] + argv
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "downstream" and args.list_steps:
        from downstream.runner import describe_steps   # no config, no log file needed
        print(describe_steps())
        return 0

    # The config's directory anchors everything: dataset paths, results_dir, and
    # the log itself. Resolve it before logging starts so the first line already
    # lands in the experiment folder.
    config_path = Path(args.config).expanduser()
    log = setup_logging(args.log_level, config_path.resolve().parent)
    log.info("%s", provenance.summary())

    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return 1
    cfg = load_config(config_path)

    try:
        datasets = resolve_datasets(cfg, args.datasets)
    except ValueError:
        log.exception("Could not resolve the dataset list")
        return 1

    if args.command == "init-plate-map":
        return cmd_init_plate_map(args, cfg, datasets, log)
    if args.command == "manifest":
        return cmd_manifest(args, cfg, datasets, log)
    if args.command == "check":
        return cmd_check(args, cfg, datasets, log)
    if args.command == "downstream":
        return cmd_downstream(args, cfg, datasets, log)
    return cmd_process(args, cfg, datasets, log)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init_plate_map(args, cfg: dict, datasets, log: logging.Logger) -> int:
    """Write a starter plate_map.yaml into each dataset folder, pre-filled from disk.

    Deliberately non-destructive: an existing plate map is hand-authored knowledge,
    so it is left alone unless ``--force`` is given.
    """
    from experiment import template

    targets = ([(Path(args.data_dir), cfg)] if args.data_dir
               else [(ds.directory, ds.cfg) for ds in datasets])

    written = skipped = 0
    for directory, ds_cfg in targets:
        path = template.write_template(directory, ds_cfg, plate_id=args.plate_id,
                                       mode=args.mode, force=args.force)
        if path is None:
            skipped += 1
        else:
            written += 1

    log.info("=== init-plate-map: %d written, %d left untouched ===", written, skipped)
    if written:
        log.info("Next: edit the 'wells:' block (look for '<-- EDIT'), then run "
                 "`python main.py check`.")
    return 0


def cmd_manifest(args, cfg: dict, datasets, log: logging.Logger) -> int:
    from experiment import build_manifest

    n_failed = 0
    for ds in datasets:
        try:
            df = build_manifest(
                ds.cfg,
                data_dir=args.data_dir or str(ds.directory),
                out_basename=args.out_basename,
                plate_map_path=args.plate_map,
                dataset=ds.name,
            )
            log.info("[%s] manifest: %d image(s), %d plate(s)",
                     ds.name, len(df), df["plate_id"].nunique())
        except Exception:
            log.exception("[%s] failed to build manifest", ds.name)
            n_failed += 1
    log.info("=== Manifest complete: %d dataset(s), %d failed ===", len(datasets), n_failed)
    return 1 if n_failed else 0


def cmd_check(args, cfg: dict, datasets, log: logging.Logger) -> int:
    """Validate config + plate maps against the files on disk, without processing.

    Answers the questions that otherwise only surface after an hour of GPU time:
    does every image map to a well, does every well have a condition, does every
    image get a stain set with the right number of channels, and, since a gate is
    only as good as its control, does every marker in every panel actually have a
    negative control it can be gated against?
    """
    from experiment.manifest import PLATE_MAP_NAME, _iter_images, _normalize_extensions
    from experiment.plate_map import PlateMap
    from experiment import parser, plate
    from selection import Selection

    down_cfg = cfg.get("downstream") or {}
    problems = 0
    plate_markers: dict[str, dict[str, set]] = {}
    for ds in datasets:
        maps = sorted(ds.directory.rglob(PLATE_MAP_NAME))
        log.info("[%s] %s  (seg_mode=%s)", ds.name, ds.directory, ds.seg_mode)
        if not ds.directory.exists():
            log.error("    directory does not exist")
            problems += 1
            continue
        if not maps:
            log.error("    no %s found: downstream cannot annotate these images",
                      PLATE_MAP_NAME)
            problems += 1
            continue

        extensions = _normalize_extensions(ds.cfg.get("input", {}).get("file_types"))
        exclude_dirs = ds.cfg.get("input", {}).get("exclude_dirs") or []
        for pm_path in maps:
            try:
                pm = PlateMap(pm_path)
            except Exception as exc:
                log.error("    %s: %s", pm_path.name, exc)
                problems += 1
                continue

            images = list(_iter_images(pm_path.parent, extensions, exclude_dirs))
            try:
                sel = Selection.from_config(ds.cfg, args.seq, args.wells)
                images = sel.filter_paths(images, pm_path.parent)
            except ValueError as exc:
                log.error("    selection: %s", exc)
                problems += 1
                continue
            unmapped, per_stain, wells_seen = [], {}, set()
            for img in images:
                info = parser.parse_image(img, pm.mode)
                if pm.is_spinning_disk:
                    label = info["well_label"]
                    well = plate.label_to_well(label, pm.cols) if label else None
                elif pm.mode == "generate":
                    well = info["well"]
                else:
                    well = pm.well_for_seq(info["seq_index"])
                if well is None:
                    unmapped.append(img.name)
                    continue
                wells_seen.add(well)
                stain_set, _ = pm.resolve_stain(well, info["seq_index"])
                per_stain[stain_set] = per_stain.get(stain_set, 0) + 1

            log.info("    %s: %d image(s), %d well(s), plate=%dx%d, mode=%s",
                     pm_path.name, len(images), len(wells_seen), pm.rows, pm.cols, pm.mode)
            for stain_set, n in sorted(per_stain.items()):
                channels = pm.stains.get(stain_set, {})
                marker = "  ".join(f"ch{c}={m}" for c, m in sorted(channels.items()))
                log.info("      stain %-12s %4d image(s)   %s", stain_set, n,
                         marker or "(no channel→marker map!)")
                if not channels:
                    problems += 1
            if pm.sweeps:
                for name, spec in pm.sweeps.items():
                    log.info("      sweep %-12s Seq %d-%d", name, *spec["seq_range"])

            if unmapped:
                log.error("      %d image(s) map to NO well, e.g. %s",
                          len(unmapped), ", ".join(unmapped[:3]))
                problems += 1
            no_condition = sorted(w for w in wells_seen if w not in pm.wells)
            if no_condition:
                log.error("      well(s) %s have images but no entry under 'wells:'",
                          no_condition)
                problems += 1
            unused = sorted(w for w in pm.wells if w not in wells_seen)
            if unused:
                log.warning("      well(s) %s are declared but have no images", unused)

            problems += _check_controls(pm, wells_seen, per_stain, down_cfg, log)
            _collect_plate_markers(pm, wells_seen, per_stain, ds, plate_markers)

    _report_pooling(plate_markers, down_cfg, log)
    problems += _check_micropattern(datasets, down_cfg, log)
    _check_segmentation(args, datasets, down_cfg, log)

    if problems:
        log.error("=== check: %d problem(s) found ===", problems)
    else:
        log.info("=== check: everything resolves; ready to run ===")
    return 1 if problems else 0


def _check_segmentation(args, datasets, down_cfg: dict, log: logging.Logger) -> None:
    """What segmentation is on disk per dataset, and where the analysis will go.

    Advisory only: before the GPU job nothing is segmented, which is fine.
    """
    import seg_status
    statuses = [seg_status.dataset_status(ds, args.seq, args.wells) for ds in datasets]
    seg_status.log_status(statuses, log)
    try:
        choice = seg_status.normalize_choice(down_cfg.get("seg_mode"))
    except ValueError as exc:
        log.error("    %s", exc)
        return
    if any(s.n_ready for s in statuses):
        _, modes = seg_status.select(datasets, statuses, choice)
        out = down_cfg.get("results_dir", "auto")
        name = seg_status.auto_dir_name(modes) if seg_status.is_auto(out) else out
        log.info("    downstream would analyze %s results into %s",
                 " + ".join(m.upper() for m in modes), name)


def _check_micropattern(datasets, down_cfg: dict, log: logging.Logger) -> int:
    """Settings the radial analysis cannot recover from at run time.

    * A pixel size: every µm depends on it. Read from an image already loaded
      (``image_metadata.json``) or from ``colony.pixel_size_um``.
    * The colony / profile settings parse and the diameter column exists.
    """
    import json
    from downstream import colony as colony_mod, radial
    from experiment.manifest import PLATE_MAP_NAME
    from experiment.plate_map import PlateMap

    problems = 0
    try:
        params = colony_mod.settings(down_cfg)
        radial.settings(down_cfg)
    except ValueError as exc:
        log.error("    downstream settings: %s", exc)
        return 1

    override = params.get("pixel_size_um")
    for ds in datasets:
        metas = sorted(ds.directory.rglob("image_metadata.json"))[:3]
        sizes = []
        for m in metas:
            try:
                sizes.append(json.loads(m.read_text()).get("pixel_size_x_um"))
            except (OSError, ValueError):
                pass
        if override:
            log.info("[%s] pixel size: %.4g µm/px (colony.pixel_size_um)", ds.name, float(override))
        elif not metas:
            log.info("[%s] pixel size: read from each image at 'load' (none loaded yet; if the "
                     "files carry no calibration, set downstream.colony.pixel_size_um)", ds.name)
        elif all(sizes):
            log.info("[%s] pixel size: %.4g µm/px (image metadata)", ds.name, float(sizes[0]))
        else:
            log.error("[%s] loaded images have NO pixel size in their metadata: set "
                      "downstream.colony.pixel_size_um, or every colony will fail", ds.name)
            problems += 1

        diam = params.get("diameter_column")
        if diam:
            for pm_path in sorted(ds.directory.rglob(PLATE_MAP_NAME)):
                try:
                    pm = PlateMap(pm_path)
                except Exception:
                    continue
                missing = [w for w, c in pm.wells.items() if diam not in c]
                if missing:
                    log.error("    colony.diameter_column=%r is missing for well(s) %s in %s",
                              diam, missing[:8], pm_path.name)
                    problems += 1
    if not params.get("expected_diameter_um") and not params.get("diameter_column"):
        log.warning("    no expected colony diameter (colony.expected_diameter_um): the "
                    "'size' QC flag cannot catch torn or merged colonies")
    return problems


def _check_controls(pm, wells_seen, per_stain, down_cfg: dict, log: logging.Logger) -> int:
    """Report, per panel, which markers have a usable negative control.

    A marker with no control is not an error: it falls through to the GMM
    fallback, which may legitimately succeed, but it is the single most common
    reason a positive fraction comes out wrong, and finding out here costs seconds
    rather than a full processing run.
    """
    from downstream.controls import split_tokens
    from experiment.plate_map import CONTROL_FOR_KEY, POSITIVE_FOR_KEY

    cond_cfg = down_cfg.get("conditions") or {}
    control_col = cond_cfg.get("control_column", "class")
    control_values = {str(v).lower() for v in
                      (cond_cfg.get("control_values") or ["control"])}  # same default as downstream.controls
    marker_controls = {str(k) for k in (cond_cfg.get("marker_controls") or {})}
    exclude = {str(m).strip().lower() for m in (down_cfg.get("exclude_markers") or [])} | {"dapi"}

    imaged = sorted(w for w in wells_seen if w in pm.wells)
    blanket, named, positive = [], {}, {}
    for well in imaged:
        cond = pm.wells.get(well, {})
        if str(cond.get(control_col, "")).lower() in control_values:
            blanket.append(well)
        for token in split_tokens(cond.get(CONTROL_FOR_KEY)):
            (blanket if token == "all" else named.setdefault(token, [])).append(well)
        for token in split_tokens(cond.get(POSITIVE_FOR_KEY)):
            positive.setdefault(token, []).append(well)

    if not blanket and not named:
        log.info("      no negative control declared (no '%s' in %s, no control_for): "
                 "%% positive profiles will rely on the GMM fallback; intensity profiles "
                 "are unaffected", control_col, sorted(control_values))
        return 0

    log.info("      controls: %d blanket well(s) %s", len(blanket), blanket[:8])
    problems = 0
    for stain_set in sorted(per_stain):
        markers = [m for m in pm.stains.get(stain_set, {}).values()
                   if m.strip().lower() not in exclude]
        orphans = []
        for marker in markers:
            key = marker.strip().lower()
            if key in named or marker in marker_controls:
                continue                                  # explicitly provided for
            demoted = set(positive.get(key, []))
            if demoted and not (set(blanket) - demoted):
                orphans.append(marker)
        for marker, wells in sorted(named.items()):
            log.info("      control_for %-10s wells %s", marker, wells)
        for marker, wells in sorted(positive.items()):
            log.info("      positive_for %-9s wells %s (demoted as its own control)",
                     marker, wells)
        if orphans:
            log.error("      stain %s: %s declared positive_for on every control well "
                      "and have no control_for well: no negative control left. Add "
                      "control_for: [%s] to the well that is their negative control.",
                      stain_set, ", ".join(orphans), orphans[0])
            problems += 1
    return problems


def _collect_plate_markers(pm, wells_seen, per_stain, ds, into: dict) -> None:
    """Accumulate {plate_id: {marker: {(stain_set, channel)}}} for the pooling report."""
    bucket = into.setdefault(f"{pm.plate_id}/{ds.seg_mode}", {})
    chan = ds.cfg.get("channels") or {}
    nuclear = int(chan.get("dapi_index", 0))
    measured = {int(c) for c in chan.get("marker_channels") or []}
    for stain_set in per_stain:
        for channel, marker in pm.stains.get(stain_set, {}).items():
            # The nuclear channel and unmeasured channels are never analyzed.
            if int(channel) == nuclear or (measured and int(channel) not in measured):
                continue
            bucket.setdefault(str(marker), set()).add((stain_set, int(channel)))


def _report_pooling(plate_markers: dict, down_cfg: dict, log: logging.Logger) -> None:
    """Name markers that appear in several panels of one plate: pooling candidates.

    Purely advisory. Whether the antibody and dilution actually matched between
    panels is a fact about the bench that no file records, so the pipeline reports
    the opportunity and leaves the claim to you.
    """
    from downstream.signaling import catalog as sig_catalog

    declared = {m: (spec or {}).get("scope") for m, spec in
                (down_cfg.get("markers") or {}).items()}
    exclude = {str(m).strip().lower() for m in (down_cfg.get("exclude_markers") or [])} | {"dapi"}
    gate_signaling = bool((down_cfg.get("signaling") or {}).get("gate", False))

    names = sorted({m for markers in plate_markers.values() for m in markers
                    if m.strip().lower() not in exclude})
    if names:
        matches = sig_catalog.classify_all(names, down_cfg)
        for match in matches.values():
            log.info("    signaling marker %s", match.describe())
        for m in sig_catalog.near_misses(names, down_cfg):
            log.warning("    %s looks like a signaling protein but is not recognized; it will "
                        "be read as a standard marker. Add it to downstream.signaling.markers "
                        "if it is a signaling readout (e.g. {%s: nc}).", m, m)
        for group in sig_catalog.spelling_groups(names, down_cfg):
            log.warning("    %s are spellings of one marker but will be analyzed as separate "
                        "markers. Use one spelling, or downstream.marker_aliases.",
                        " / ".join(group))
    else:
        matches = {}

    for where, markers in sorted(plate_markers.items()):
        for marker, entries in sorted(markers.items()):
            if marker.strip().lower() in exclude or len(entries) < 2:
                continue
            if marker in matches and not gate_signaling:
                continue                  # not gated: there is no gate to pool
            if declared.get(marker) in {"plate", "experiment"}:
                log.info("    [%s] %s pooled across %d panels (scope: %s)", where, marker,
                         len(entries), declared[marker])
                continue
            channels = {c for _, c in entries}
            detail = ", ".join(f"{s} ch{c}" for s, c in sorted(entries))
            if len(channels) == 1:
                log.warning(
                    "    [%s] %s appears in %d panels on the SAME channel (%s). Gates "
                    "will be estimated separately. If the antibody and dilution matched, "
                    "set downstream.markers.%s.scope: plate to pool them into one gate.",
                    where, marker, len(entries), detail, marker)
            else:
                log.info(
                    "    [%s] %s appears in %d panels on DIFFERENT channels (%s): "
                    "separate gates are almost certainly correct here.",
                    where, marker, len(entries), detail)


def _manifest_digests(datasets, basename: str) -> dict:
    """``dataset → sha256`` of its manifest (CSV preferred: byte-stable), or None."""
    import hashlib
    out = {}
    for ds in datasets:
        digest = None
        for suffix in (".csv", ".parquet"):
            path = Path(ds.directory) / f"{basename}{suffix}"
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                break
        out[ds.name] = digest
    return out


def cmd_downstream(args, cfg: dict, datasets, log: logging.Logger) -> int:
    from downstream.runner import run
    from pipeline_config import experiment_root

    from selection import Selection

    import seg_status

    root = experiment_root(cfg)
    down_cfg = cfg.setdefault("downstream", {}) or {}
    # Which segmentation results to analyze: only datasets that have results, in
    # the mode asked for. The output folder is named after what is used.
    try:
        choice = seg_status.normalize_choice(args.seg_mode or down_cfg.get("seg_mode"))
    except ValueError:
        log.exception("invalid segmentation choice")
        return 1
    statuses = [seg_status.dataset_status(ds, args.seq, args.wells) for ds in datasets]
    seg_status.log_status(statuses, log)
    datasets, modes = seg_status.select(datasets, statuses, choice,
                                        explicit=bool(args.datasets))
    log.info("downstream: analyzing %s segmentation results from %s",
             " + ".join(m.upper() for m in modes), ", ".join(ds.name for ds in datasets))
    # Same image subset as `process`: the CLI overrides input.select per dataset.
    # The resolved selection is part of the assemble step's checkpoint hash, so
    # changing it re-assembles the cell tables.
    selections = {}
    for ds in datasets:
        try:
            sel = Selection.from_config(ds.cfg, args.seq, args.wells)
        except ValueError:
            log.exception("[%s] invalid image selection", ds.name)
            return 1
        ds.cfg.setdefault("input", {})["select"] = {"seq": sel.seq_spec, "wells": sel.wells_spec}
        if sel.active:
            log.info("[%s] downstream restricted to %s", ds.name, sel.describe())
            selections[ds.name] = sel.as_dict()
    # The dataset names belong in the hash too: analyzing a different set of
    # datasets must rebuild the cell tables, not reuse the previous ones.
    down_cfg["_selection"] = {"_datasets": sorted(ds.name for ds in datasets), **selections}
    # The manifests carry the plate map (conditions, stain sets, control_for) into
    # the cell tables. Their content goes into the assemble checkpoint too, so an
    # edited plate map + a rebuilt manifest re-assembles instead of being skipped.
    down_cfg["_manifests"] = _manifest_digests(datasets, down_cfg.get("manifest_basename",
                                                                      "manifest"))
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif seg_status.is_auto(down_cfg.get("results_dir", "auto")):
        out_dir = Path(seg_status.auto_dir_name(modes))
    else:
        out_dir = Path(down_cfg["results_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    n_failed = run(cfg, datasets, out_dir, step_names=args.steps, force=args.force,
                   force_steps=args.force_step, experiment_root=root)
    return 1 if n_failed else 0


def cmd_process(args, cfg: dict, datasets, log: logging.Logger) -> int:
    from image_processing import DirectoryWalker
    from selection import Selection

    steps_to_run = [s for s in STEP_ORDER if s in set(args.steps)]
    force_steps = set(args.force_step)
    total_ok = total_failed = 0

    for ds in datasets:
        ds_cfg = ds.cfg
        if args.seg_mode is not None:
            ds_cfg.setdefault("segmentation", {})["mode"] = args.seg_mode

        try:
            images = DirectoryWalker(ds_cfg).find_images()
            images = Selection.from_config(ds_cfg, args.seq, args.wells).filter_paths(
                images, ds.directory)
        except Exception:
            log.exception("[%s] failed to discover input images", ds.name)
            total_failed += 1
            continue
        if not images:
            log.error("[%s] no images under %s: skipping.", ds.name, ds.directory)
            continue

        steps = build_steps(ds_cfg)
        seg_mode = (ds_cfg.get("segmentation", {}).get("mode") or "2d").lower()
        log.info("=== [%s] %d image(s), steps=%s, seg_mode=%s ===",
                 ds.name, len(images), steps_to_run, seg_mode)

        for i, path in enumerate(images, 1):
            log.info("[%s] [%d/%d] %s", ds.name, i, len(images),
                     _safe_relpath(path, ds.directory))
            try:
                for step in steps_to_run:
                    force = args.force or (step in force_steps)
                    steps[step].process(path, force=force)
                total_ok += 1
            except Exception:
                # Full traceback, then carry on with the remaining images.
                log.exception("[%s] failed processing %s", ds.name, path.name)
                total_failed += 1

    log.info("=== Pipeline finished: %d ok, %d failed ===", total_ok, total_failed)
    return 1 if total_failed else 0


def _safe_relpath(path: Path, root) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
