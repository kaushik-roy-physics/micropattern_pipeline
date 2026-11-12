# Changelog

All notable changes to micropattern_pipeline. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version
numbers follow [semantic versioning](https://semver.org) (see CONTRIBUTING.md).
Each entry says when a change alters numbers in existing results.

## [Unreleased]

## [1.0.1] - 2026-11-12

### Fixed
- Power tables were optimistic for small designs: the minimum detectable
  difference is a t-test calculation, but the comparisons are permutation tests,
  which with E vs E replicates cannot go below p = 2 / C(2E, E) (0.1 for 3 vs 3).
  `effects_power.csv` now carries `smallest_possible_p` and
  `permutation_can_reach_alpha`; the power for a target effect is 0 where alpha
  cannot be reached, and the power figure draws those points hollow.
  **Rerun `effects`** to update existing power tables; no other numbers change.
- `docs/STATISTICAL_METHODS.md` said the size filter drops merged nuclei. It only
  removes masks that are too small; merges stay and must be checked on the overlays.

## [1.0.0] - 2026-11-12

First version under version control. It collects the work of 2026-10 and
2026-11; earlier history is in `docs/SESSION_NOTES_2026-10.md`.

### Pipeline
- Per-image steps `load`, `segment` (Cellpose-SAM, 2-D or 3-D), `extract`, each
  checkpointed; new masks make the extracted intensities and colonies stale.
- Downstream steps `colony`, `assemble`, `qc`, `thresholds`, `profiles`,
  `aggregate`, `compare`, `boundary`, `effects`, `signaling`, `report`, with
  config-aware checkpoints.
- Colony detection with edge-referenced distances, profiles on a shared µm grid
  and a normalized grid, one shared intensity scale, colony QC flags.
- Replicates at colony, well or experiment level; functional permutation tests
  (exact when the design is small) with Benjamini-Hochberg q-values.
- Boundary positions and widths with per-colony SE, facing-pair crossings,
  double-positive fractions, border roughness, angular registration.
- Variance components, dose trends and power (`effects`).
- Signaling readouts (nuclear and N:C) with local background.
- Nuclear density per bin, normalization captions, histone guard, saturated
  nucleus filter.

### Options added 2026-11
- `exclude_markers` leaves markers out by name within a channel (GUI
  "analyze" column); DAPI is always excluded.
- `downstream.skip_steps` switches optional analyses off; their earlier outputs
  move to `_skipped/`.
- `results_dir: auto` names the results folder `analysis_2d` or `analysis_3d`
  from the segmentation found; `--seg-mode` chooses when both exist.
- The GUI saves into `config.yaml` and `plate_map.yaml` keeping comments, and
  asks before overwriting.
- `hpc/run.sh`: one-command HPC submission; the segmentation job submits the
  downstream job; `MP_<NAME>` one-off overrides.
- Plain-language statistics primer in `docs/STATISTICAL_METHODS.md`.

### Version tracking
- `VERSION` file; `python main.py --version`; every run logs the version and git
  commit; downstream writes `pipeline_version.json` and shows the version in the
  report header.

### Fixed
- Report navigation links were printed as literal HTML.
- Signaling background in fields with no cell-free pixels took its noise from
  the whole image, so cytoplasm rings were rejected and N:C came out missing.
  **Rerun `signaling`** for experiments whose fields are confluent.
