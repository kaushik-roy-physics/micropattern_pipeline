# Micropattern pipeline: reference manual

Everything the pipeline does, every setting that changes it, and every file it
writes. The [README](../README.md) is the how-to; [STATISTICAL_METHODS.md](STATISTICAL_METHODS.md)
explains the statistics; [PLATE_MAPS.md](PLATE_MAPS.md) is the plate-map
reference.

---

## Contents

1. [What it does](#what-it-does)
2. [Commands](#commands)
3. [Configuration](#configuration)
4. [The analysis steps](#the-analysis-steps)
5. [Colony detection](#colony-detection)
6. [Radial profiles](#radial-profiles)
7. [Intensity scale and scope](#intensity-scale-and-scope)
8. [% positive profiles and negative controls](#positive-profiles-and-negative-controls)
9. [Several markers at once](#several-markers-at-once)
   - [Signaling markers](#signaling-markers)
   - [Boundaries, peaks and roughness](#boundaries-peaks-and-roughness)
   - [Density, saturation and normalization captions](#density-saturation-and-normalization-captions)
   - [Effects, trends and power](#effects-trends-and-power)
10. [Replication: what n is](#replication-what-n-is)
11. [Guardrails](#guardrails)
12. [Output files](#output-files)
13. [Extending the pipeline](#extending-the-pipeline)
14. [Project structure](#project-structure)
15. [Tests](#tests)

---

## What it does

```
 images ──► process (GPU) ───────────────►  per image: masks + per-nucleus intensities
            load → segment → extract
                                           manifest: image → well → condition
 plate_map.yaml ──► manifest ─────────────►

 downstream (CPU) ─► colony → signaling → assemble → qc → thresholds → boundary → profiles → aggregate → compare
                                                                          │                      │
                                                                          │                      ├─► multimarker
                                                                          │                      └─► coexpression
                                                                          └──────────────────────────► effects → signaling_summary → report
```

* **process** reads any common microscope format (`.nd2`, `.tif`, `.czi`, `.lif`,
  `.oir`), segments nuclei with Cellpose-SAM in 2-D (on the DAPI maximum
  projection) or 3-D (on the z-stack), and measures every marker in every
  nucleus. It knows nothing about colonies or conditions.
* **manifest** maps every image to its well and condition through the plate map.
* **downstream** finds the main colony in each image, gives every nucleus a
  distance from the colony edge, builds one radial profile per colony on two
  coordinates (µm from the edge, and the fraction of the radius), averages them
  per well and condition, and compares conditions with permutation tests on
  whole curves. It then reduces each colony to scalars with a per-colony SE
  (boundary positions and widths, facing-pair crossings, signaling peaks,
  border roughness, density) and compares those across conditions and doses
  with variance components and power. Everything lands in
  `report.html` in the analysis folder (`analysis_2d/` or `analysis_3d/`).

One image is one colony. Micropattern acquisitions usually center one pattern
per imaging position; neighboring patterns that creep into the field are
recognized and left out.

---

## Commands

```bash
python main.py check          --config config.yaml   # validate everything; no images opened
python main.py process        --config config.yaml   # load → segment → extract (GPU)
python main.py manifest       --config config.yaml   # plate maps → manifest.parquet
python main.py downstream     --config config.yaml   # the whole analysis (CPU)
python main.py init-plate-map --config config.yaml   # write a commented starter plate map
```

| Option | Commands | Meaning |
| --- | --- | --- |
| `--datasets a b` | all | only these datasets from the config |
| `--seq 0-9 15` | process, check, downstream | only images with these Seq / Point numbers |
| `--wells 1-3` / `C1-C4` | process, check, downstream | only images from these wells |
| `--steps …` | process, downstream | run only these steps |
| `--force` / `--force-step s` | process, downstream | redo everything / only step *s* |
| `--seg-mode 2d\|3d` | process | override the segmentation mode |
| `--seg-mode auto\|2d\|3d\|both` | downstream | which segmentation results to analyze (overrides `downstream.seg_mode`) |
| `--out DIR` | downstream | write the analysis to DIR instead of `results_dir` |
| `--list-steps` | downstream | print the registered steps and exit |

The HPC job files in `hpc/` run exactly these commands; see the README §4.
`hpc/run.sh <what>` (from the experiment folder) submits them for a re-run:
`all`, `seg`, `analysis`, `redo <step>…`, `redo-all`, `reseg`, `test`, `status`,
with `SEQ=`, `WELLS=`, `DATASETS=`, `SEG=2d|3d` for one submission. It passes
these to the jobs as `MP_<NAME>` variables, which the job files apply over
`<NAME>` from `hpc_job.env`.

**What `check` reports about segmentation.** For each dataset, the images
segmented per mode on disk (from each result folder's
`segmentation_metadata.json`), a warning when that is not the mode the dataset
is set to, and the folder `downstream` would write to.

---

## Configuration

One `config.yaml` per experiment, written by the GUI from
[`templates/config.yaml`](../templates/config.yaml). Every option in that file
carries its explanation as a comment, which the GUI shows as help text.

```yaml
defaults:          # shared by every dataset: input, channels, segmentation, intensity
datasets:          # one entry per acquisition folder; deep-merged over defaults
  - {name: zmax,     directory: ./zmax}
  - {name: stitched, directory: ./stitched, segmentation: {mode: 3d}}
downstream:        # the analysis
  results_dir: auto  # → analysis_2d / analysis_3d / analysis_2d_3d, after the results used
  seg_mode: auto   # which segmentation results: auto | 2d | 3d
  exclude_markers: [DAPI]   # markers left out of the analysis, by name
  skip_steps: []   # optional analyses left out (boundary, coexpression, multimarker, compare, effects, signaling_summary)
  conditions:  …   # what defines a group to compare
  markers:     …   # per-marker intensity space and scale scope
  colony:      …   # colony detection
  qc:          …   # flags and which ones exclude a colony
  thresholds:  …   # positivity gates (% positive readout)
  profiles:    …   # grids, bins, minimum counts
  stats:       …   # replication unit
  compare:     …   # permutation tests
  multimarker: …   # stacked multi-marker profiles and local trends
  coexpression: …  # pairwise co-expression against distance from the edge
  signaling:   …   # signaling readouts, background, reference wells
  boundary:    …   # boundary fits, facing pairs, peaks, roughness, bin check
  effects:     …   # effects vs reference, trends, variance components, power
  visualization: … # dpi, shared y-ranges, caption
  nuclear_stain: auto  # dna | histone | auto: what the nuclear channel is
```

All paths are relative to the folder holding `config.yaml`, so the same file works
on a laptop and on the HPC.

The settings that matter most for micropatterns:

| Setting | Default | What it does |
| --- | --- | --- |
| `results_dir` | `auto` | analysis folder; `auto` names it after the segmentation results used: `analysis_2d`, `analysis_3d`, `analysis_2d_3d` |
| `seg_mode` | `auto` | which results to analyze: `auto` (every dataset with results), `2d`, `3d`; datasets without results are left out |
| `exclude_markers` | `[DAPI]` | markers left out of every table, figure and test, by name (so one of two markers sharing a channel can go); DAPI is always out |
| `skip_steps` | `[]` | optional analyses to leave out; the core (colony → … → report) always runs; a skipped step's old outputs move to `_skipped/<step>/` |
| `conditions.condition_key` | `[treatment]` | plate-map column(s) that define the groups compared, e.g. `[class]` for cell lines |
| `conditions.facet_key` | none | compare within each value of one condition column (crossed designs) |
| `colony.expected_diameter_um` | none | nominal pattern diameter; enables the `size` QC flag |
| `colony.diameter_column` | none | plate-map column with a per-well diameter (several pattern sizes) |
| `colony.pixel_size_um` | metadata | µm per pixel when the files carry no calibration |
| `colony.close_radius_um` | 15 | gap between nuclei bridged when building the colony mask |
| `colony.trim_outside_circle` | 1.1 | cells beyond 1.1 × the fitted pattern radius are outside the colony |
| `profiles.bin_um` | 10 | width of the µm-grid bins (≈ one nucleus) |
| `profiles.n_norm_bins` | 25 | bins of the normalized grid |
| `profiles.min_nuclei_per_bin` | 10 | a colony bin with fewer nuclei is a gap |
| `profiles.min_replicates` | 3 | a mean needs this many colonies (or wells) at a point |
| `qc.exclude_flags` | `[failed]` | which QC flags remove a colony from the analysis |
| `stats.replicate` | `auto` | colony or well as the replicate (see [Replication](#replication-what-n-is)) |
| `compare.n_permutations` | 5000 | permutation-test resolution |
| `multimarker.z_crit` | 1.96 | \|slope / SE\| at which a local trend is marked |
| `coexpression.pairs` | `auto` | every marker pair within a stain set, or an explicit list `[[A, B], ...]` |
| `coexpression.method` | `spearman` | within-bin correlation: `spearman` or `pearson` |
| `coexpression.bin_um`, `n_norm_bins` | 25, 10 | co-expression grids (coarser than the profiles: a correlation needs more nuclei) |
| `nuclear_stain` | `auto` | `dna`, `histone`, or `auto` (an H2B/H3/histone name on the nuclear channel → histone); a histone denominator triggers a warning |
| `qc.drop_saturated_nuclei` | `true` | drop nuclei whose median is at the detector ceiling in any channel |
| `qc.saturation_ceiling` | none | override the ceiling (4095 for 12-bit data in 16-bit files) |
| `signaling.background` | `local` | `local` (band just outside the colony) or `image` (all cell-free pixels) |
| `signaling.reference_values` | from `conditions` | reference wells signaling values are divided by; `[]` = none (absolute a.u.) |
| `boundary.pairs` | `auto` | facing marker pairs (one edge-high, one center-high), or `[[EDGE, CENTER], ...]` |
| `boundary.peak_markers` | `auto` | markers searched for an interior peak (auto = signaling markers) |
| `boundary.bin_um` | 10 | bin width of the boundary fits |
| `boundary.n_sectors`, `n_boot` | 12, 200 | angular blocks and replicates of the per-colony bootstrap |
| `boundary.min_support` | 0.8 | a colony value is used when this fraction of bootstrap replicates find it |
| `boundary.bin_check` | `[fixed:5, fixed:20, quantile:20]` | other binnings every colony is refitted with; `[]` = off |
| `boundary.roughness.n_sectors` | 16 | angular sectors for border radius vs angle |
| `stats.experiment_column` | none | plate-map column naming the independent differentiation |
| `effects.targets` | `[]` | effect sizes to report power for, e.g. `[{metric: crossing_um, delta: 20}]` |
| `effects.power.assumed_experiment_sd` | none | between-experiment SD (× the well or colony SD) when it cannot be estimated |
| `visualization.ylim` | `shared` | one y-range per marker across all figures, or `auto` |
| `visualization.caption` | none | text appended to every normalization caption |

---

## The analysis steps

Steps marked *optional* below can be left out with `downstream.skip_steps`
(GUI step 3, *Analyses to run*); `effects` is skipped too when both `boundary`
and `compare` are. The rest is the core every readout depends on.

Each step is checkpointed in `<analysis>/.checkpoints/<step>.json` with a hash of
the settings it depends on. A step re-runs when (a) one of its settings changed,
(b) a step it depends on ran after it, or (c) you force it. Otherwise it is
skipped, so re-running the whole analysis after changing a bin width redoes only
`profiles` and what follows.

| Step | Reads | Does | Writes |
| --- | --- | --- | --- |
| `colony` | `nuclear_masks.tif`, `image_metadata.json` per image | main colony, edge, circle fit, outliers, edge distance; one QC overlay per image. Per-image checkpoints; parallel on CPU | per image: `colony_mask.tif`, `colony_nuclei.parquet`, `colony.json`, `colony_overlay.png` |
| `signaling` | `nuclear_masks.tif`, `colony_mask.tif`, channel TIFFs, manifests | only for signaling markers: nucleus core, cytoplasmic ring, local background, `nuc_bgsub` and `nc_ratio` per nucleus. Per-image checkpoints (re-measures only images whose settings changed); parallel on CPU | per image: `signaling_nuclei.parquet`, `signaling.json`, `signaling_rings.png`; `tables/signaling_background.csv` (every image's background, method, outlier flag within its stain set); `figures/signaling/` |
| `assemble` | manifests, `per_nuclei_intensities.joblib`, colony and signaling output | one cell table for the experiment with colony columns attached | `cells.parquet`, `intensities.parquet`, `colonies.csv`, `field_channel_qc.csv` |
| `qc` | the cell tables | colony geometry flags, per-colony staining QC, per-marker channel QC; nuclei whose median sits at the detector ceiling (any channel) are dropped from every analysis (`qc.drop_saturated_nuclei`); warns when a marker is divided by a histone nuclear channel (`nuclear_stain`) | `tables/qc_colonies.csv`, `qc_fields.csv`, `qc_marker_colonies.csv`, `qc_saturated_nuclei.csv` (only when any), `figures/qc/` |
| `thresholds` | in-colony nuclei of kept colonies | positivity gate per marker × scope | `tables/thresholds.csv`, `control_assignment.csv`, `figures/thresholds/` |
| `boundary` (optional) | in-colony nuclei, `thresholds.csv` | per colony: logistic boundary position (x50) and width of each marker (gate calls and intensity), crossing / overlap / double-positive fraction of facing marker pairs, interior signaling peak and its offsets, border roughness and angular registration; angular-sector bootstrap SE; summary across replicates | `tables/boundary_colonies.csv`, `boundary_fits.csv`, `boundary_summary.csv`, `boundary_sectors.csv` (border per angular sector), `boundary_registered.csv` (profiles vs distance from the border, angle-averaged and registered), `figures/boundary/` (by condition, `registered_<marker>.png`, `roughness_<marker>.png`), `figures/boundary/borders/` (per-colony border maps); bin check (`boundary.bin_check`): every position and width refitted with other bins, `boundary_bin_check.csv`, `boundary_bin_check_summary.csv` (shift in µm and in units of the bootstrap SE), `figures/boundary/bin_check.png` |
| `effects` (optional) | `boundary_colonies.csv`, `colony_features.csv` | variance components (colony / well / experiment), effects vs a reference condition with CI, trends over numeric condition columns (log/linear, optional decay τ), minimum detectable differences and power for target effects | `tables/effects_variance.csv`, `effects_power.csv`, `effects_vs_reference.csv`, `effects_trends.csv`, `figures/effects/` |
| `profiles` | same | shared intensity scale; per-colony curves on both grids; per-colony binned profiles with each bin's colony area and nuclear density; nuclear density per µm bin; a normalization caption on every profile figure | `tables/scale_anchors.csv`, `colony_profiles.csv`, `colony_profiles_binned.csv` (+ `area_um2`, `density`), `density_colonies.csv`, `figures/colony_profiles/` (bottom strip: one bar per bin, width = bin width, height = density) |
| `aggregate` | `colony_profiles.csv` | well and condition means ± CI; replicate unit chosen; nuclear density by condition | `tables/well_profiles.csv`, `condition_profiles.csv`, `density_profiles.csv`, `replicate_unit.json`, `figures/profiles/` (incl. `density_um.png`) |
| `compare` (optional) | same | functional permutation ANOVA, pairwise tests (BH), features (Kruskal-Wallis, BH), colony similarity | `tables/compare_*.csv`, `colony_features.csv`, `feature_tests.csv`, `similarity_*.csv`, `compare_summary.json`, `figures/compare/` |
| `multimarker` (optional) | `colony_profiles.csv`, `well_profiles.csv`, boundary tables | every marker stacked on one distance axis (own y-range each) + local-trend strip; colony figures carry the accepted boundaries ± SE, and say in the title why a facing pair has none; skipped with one marker | `tables/multimarker_trends.csv`, `figures/multimarker_profiles/{colonies,wells}/` |
| `coexpression` (optional) | in-colony nuclei, `scale_anchors.csv`, `thresholds.csv` | per marker pair: within-bin correlation, double-positive fraction vs independence, bin means, against distance; per colony (µm) and per well (normalized, colonies first); skipped when no stain set has two markers | `tables/coexpression_colonies.csv`, `coexpression_wells.csv`, `figures/coexpression/pages/{colonies,wells}/`, `figures/coexpression/panels/<level>/<kind>/` |
| `signaling_summary` (optional) | in-colony nuclei, `signaling_reference.csv` | one value per colony and signaling marker (whole colony, edge band, center), by condition | `tables/signaling_colonies.csv`, `figures/signaling_summary/` |
| `report` | everything | one HTML page | `report.html` |

With two segmentation modes in one experiment (MIPs in 2-D and z-stacks in 3-D),
`aggregate` and `compare` run once per mode and never pool them: they are two
measurements of the same colonies. Their files then carry `_2d` / `_3d`.

---

## Colony detection

Per image, from the nuclear masks only:

1. **Footprint.** All nuclei are joined into one mask by a morphological closing
   with a disk of `close_radius_um` (bridging the gaps between nuclei) and their
   holes filled. `open_radius_um > 0` additionally severs thin bridges to a
   touching neighbor.
2. **Main colony.** The largest connected piece (`main_colony: largest`), or the
   one nearest the image center (`center`). Other pieces larger than
   `min_colony_area_um2` are **neighboring colonies**; their nuclei, and every
   stray nucleus outside the main mask, have `in_colony = False` and never enter
   a profile.
3. **Circle fit** to the colony's true rim (never to the part cut by the image
   border), iteratively ignoring rim points far off the circle (a tail of
   escaped cells, a bump). The mask is then cut at `trim_outside_circle` × R:
   cells that left the pattern are outside the colony. The fraction cut is
   recorded as `trimmed_fraction`.
4. **Distance from the edge.** The Euclidean distance transform of the mask,
   which follows the colony's real outline. For a colony cut by the image border
   (more than `max_border_fraction` of its rim on the border) the distance
   transform would measure nuclei near the cut to the far side, so the distance
   is taken from the fitted circle instead (`edge_method: circle`) and the colony
   is flagged `clipped`. `edge_distance: mask | circle` forces one method.
5. **Normalized coordinate.** `r_norm = dist_edge_um / R` (0 = edge, 1 = center).

3-D masks are projected to their xy footprint; the radial geometry is lateral.

**The pixel size is never guessed.** It comes from the image metadata or from
`colony.pixel_size_um`. An image with neither fails with a message naming the
setting, and `check` reports it before anything runs.

**Tuning.** Use the *Preview colony detection* panel in GUI step 3: it runs the
detection on one segmented image with the settings on screen and draws the
result, without writing anything. If the edge swallows a neighboring colony,
raise `open_radius_um` or lower `close_radius_um`; if the edge is ragged and
leaves holes at the rim, raise `close_radius_um`.

---

## Radial profiles

Two coordinates, both computed for every colony:

| Grid | Coordinate | Bins | Use |
| --- | --- | --- | --- |
| `um` | distance from the edge in µm | `bin_um` wide, from 0 to the largest radius (or `max_um`) | the absolute length scale fates are organized on; colonies of different sizes |
| `norm` | `r_norm`, 0 = edge, 1 = center | `n_norm_bins` equal bins | comparing colonies of one nominal size |

A colony's value in a bin is the mean shared-scale intensity of its nuclei there
(and the fraction of them that are positive), with a t-based CI over those
nuclei. A bin with fewer than `min_nuclei_per_bin` nuclei is a gap. A colony's µm
curve ends at its own radius.

**Per-colony figures** (`figures/colony_profiles/`) use each colony's own bins,
in µm from the edge: `colony_binning: quantile` (≈ equal nuclei per bin, the
default), `equal_area` (≈ equal colony area per bin, from the mask's own
distance distribution) or `fixed` (`bin_um` wide).

**Well and condition curves** average colony curves bin by bin; the error band is
a t-based 95 % CI across replicates (colonies or wells). A point covered by fewer
than `min_replicates` replicates is left empty.

---

## Intensity scale and scope

Each nucleus's value is its marker intensity in the marker's analysis space
(`dapi_norm` (marker ÷ nuclear stain per nucleus, the default) or `raw`),
rescaled with **one pair of anchors for its whole scope**. Use `raw` for markers
that are not nuclear, and for every marker when the nuclear channel is not a DNA
stain (an H2B fusion varies from cell to cell, so dividing by it adds noise):
`default_space: raw`. Signaling markers have their own readouts
([below](#signaling-markers)).

```
scaled = (value − P1) / (P99 − P1)        P1, P99 over all in-colony nuclei of the scope
```

so 0 and 1 mean the same intensity in every colony of the scope, and a colony
expressing a marker at half the level reads half as high. The scale is not
clipped. The anchors are written to `tables/scale_anchors.csv`.

The scope is the set of colonies that share one scale, declared per marker in the
same way as its positivity gate:

| `scope` | Colonies sharing one scale | When |
| --- | --- | --- |
| `panel` (default) | one dataset × plate × stain set | always safe |
| `plate` | every panel of one plate (per segmentation mode) | the same antibody at the same dilution in several panels |
| `experiment` | everything of one segmentation mode | several plates stained and imaged identically |

```yaml
downstream:
  default_space: dapi_norm
  markers: {ISL1: {scope: plate}, ECAD: {space: raw}}
```

The assumption behind a shared scale is that its colonies were stained and imaged
alike. Colonies on different plates are compared on one scale only if you declare
`scope: experiment`.

---

## % positive profiles and negative controls

The second readout of every profile is the **fraction of nuclei positive** in
each bin. It needs a positivity gate per marker:

* **From a negative control** (`thresholds.method: control_percentile`): the gate
  is the `1 − fpr` quantile of the marker in its negative-control wells: with
  `fpr: 0.01`, at most 1 % of true negatives are called positive. Control wells
  are declared by `conditions.control_column` / `control_values` (default, also
  when the list is empty: a `class` of `control`; name `pluripotent` here to gate off a pluripotent well), or per marker by
  `control_for` / `positive_for` in the plate map (see [PLATE_MAPS.md](PLATE_MAPS.md)).
* **Without a control**: a two-population mixture fit, accepted only if it
  passes explicit separation checks (`thresholds.gmm`).
* **Otherwise no gate.** The marker gets no % positive profile, and the report
  says so at the top. A made-up gate is never used.

`figures/thresholds/threshold_evidence.png` shows every gate on its histogram.
Gates are estimated from in-colony nuclei of kept colonies only.

---

## Several markers at once

Two steps look at the markers of a colony together. Both need at least two
analysis markers and are skipped, with one log line, when there is only one.

**`multimarker`: stacked profiles.** Each colony (µm grid, 10 µm bins) and each
well (normalized grid; bold mean ± CI across colonies over thin colony traces)
gets one figure: one panel per marker on a shared x-axis, each with its **own
y-range**, the % positive curves of every gated marker overlaid on one 0-1 panel,
and a **trend strip** at the bottom. The strip shows, for every marker and
position, the local slope divided by its standard error::

    z_i = (m[i+k] − m[i−k]) / sqrt(se[i+k]² + se[i−k]²)     (k = multimarker.window_bins)

Red rises, blue falls, and ▲ / ▼ mark |z| ≥ `multimarker.z_crit`. Because z
compares each marker's change with its own noise, markers with very different
ranges sit in one strip without rescaling any colony. The z-values are
descriptive (not corrected for the many positions and markers); the SE is over
nuclei for a colony and over colonies for a well. Every value is in
`tables/multimarker_trends.csv` (`level` = colony or well).

**`coexpression`: where in the colony two markers go together.** Profiles are
bin means, one marker at a time, so they cannot say whether the nuclei high for
A are the ones high for B. This step works from the nuclei, on the shared scale.
Pairs are formed only within a stain set (markers measured in the same nuclei).
For each pair, colony and distance bin (`coexpression.bin_um` /
`n_norm_bins`; bins with fewer than `min_nuclei_per_bin` nuclei are NaN):

| Column | Meaning |
| --- | --- |
| `correlation`, `corr_ci_lo/hi` | Spearman (or Pearson) correlation of A and B over the bin's nuclei, Fisher-z CI (Spearman SE `sqrt(1.06/(n-3))`) |
| `mean_a`, `mean_b` | bin means: the colony's path through the (A, B) plane |
| `frac_a`, `frac_b`, `frac_ab` | fraction positive for A, B, both (needs gates) |
| `frac_expected` | `frac_a · frac_b`, the double-positive fraction if A and B were independent within the bin |
| `log2_enrichment` | `log2(frac_ab / frac_expected)` where both are > 0 |

Each page has one row per pair: a per-nucleus scatter colored by distance from
the edge (cividis) with the gates, the same scatter split into `n_bands` radial
bands, the correlation and the double-positive fraction (observed vs
independence) against distance, the bin-mean trajectory from edge to center,
and, on colony pages where both markers are gated, a map of where the
A+B+ / A+ only / B+ only / A−B− nuclei sit.

**Wells: colonies first.** On well pages every readout is computed per colony
and then averaged over colonies (correlations on the Fisher-z scale; n =
colonies, same `min_replicates` rule as the profiles). Pooling the nuclei of
several colonies would turn a level difference between colonies (one colony
brighter in both markers) into an apparent correlation: Simpson's paradox. The
pooled scatter on a well page draws the same number of nuclei from every colony
(`points_per_colony`), so no colony dominates the picture.

---

## Signaling markers

A signaling marker is read by **where** the protein is, not by how much of it
there is, and is never gated into % positive. Module: `downstream.signaling`
(`catalog`, `measure`, `readouts`, `reference`).

**Which markers.** A built-in catalog, matched spelling-insensitively (lower
case, β and "beta" → b, punctuation and spaces removed): β-catenin (`b-CAT`,
`β-catenin`, `beta catenin`, `bCat`, `CTNNB1`, `active b-catenin`) → `nuclear`;
SMAD2/3 (`Smad2-3`, `SMAD 2/3`), SMAD2, SMAD3, SMAD4, SMAD1 → `nc`. A trailing
qualifier is tried off (`b-CAT-r`, `SMAD4 (rb)`, `SMAD2/3-488`) and the log says
so. Phospho forms (`pSMAD1`) are not matched to their total protein. Names that
look like a signaling protein but match nothing (`pSMAD1`, `SMAD1/5/8`), and two
spellings of one marker across panels (`b-CAT` / `B-cat`), are reported by
`check`, the GUI and the run log. `signaling.markers` adds or overrides entries:

```yaml
downstream:
  signaling:
    markers: {pSMAD1: nuclear, YAP1: nc, SMAD4: none}   # none = standard marker
    reference_column: class
    reference_values: [pluripotent]
```

**Readouts** (per nucleus, from the `signaling` step):

| readout | value | used for |
| --- | --- | --- |
| `nuclear` | median of the nucleus core (1-px rim removed) − image background; `nuclear_space: raw` or `dapi_norm` instead | β-catenin |
| `nc` | (nucleus − background) / (ring − background) | SMADs |

The **ring** runs `ring_inner_um` to `ring_outer_um` (1 to 4 µm) outside each
nucleus, grown with `expand_labels` so neighbors split the space between them and
no ring enters another nucleus; for 3-D masks it is grown within each z-plane.
Ring pixels must lie inside **cell bodies** (smoothed channel above background +
`body_k` × MAD of the cell-free pixels): otherwise the empty surface beyond the
colony edge would pull the cytoplasm toward background and inflate N:C exactly
at the rim. The **background** (`background: local`, the default) is the median
of cell-free pixels in a band `local_band_um` wide that starts `free_distance_um`
beyond the cell-covered area (nuclei within `free_distance_um` of each other,
holes filled, plus the colony mask): the surface right around the colony, so
vignetting and dim image corners do not enter, and gaps inside a colony
(unsegmented or sparse cells) are never background. With fewer than
`min_free_px` band pixels it falls back to every cell-free pixel outside the
covered area (`background: image`), then to the camera offset (0.5th
percentile); zero-padding of stitched images is ignored throughout.
`tables/signaling_background.csv` lists every image's level and method and flags
an image more than 5 robust SD (and 10 %) from the others of its dataset and
stain set (images with a different channel → marker layout are never compared). N:C is NaN when
the ring has fewer than `min_ring_px` pixels or does not clear the background.
Check the rings in `figures/signaling/ring_qc_*.png`.

**Relative to a reference, or not.** For each signaling marker and scope (plate × panel
by default, as for the shared scale), the reference is the median over reference
colonies of each colony's median value; every value is divided by it, so 1 =
the reference state. `reference_values: []` turns this off: values stay in
background-subtracted a.u. (`nuc_bgsub`), comparable within one stain panel.
Prefer that when the reference wells hold few, sparse or under-segmented
colonies; check their counts in `qc_colonies.csv` and their density first. A panel without reference colonies borrows the reference of
the same plate (logged). A marker without any reference is shown unnormalized
everywhere (N:C as the plain ratio, a nuclear readout on the shared scale), so
its units never mix. The reference levels are in `tables/signaling_reference.csv`,
and every axis says what 1 means (`nuclear b-CAT (÷ pluripotent)`).

From there signaling markers flow through `profiles`, `aggregate`, `compare`,
`multimarker` and `coexpression` like any marker, with a dashed line at 1.
`signaling_summary` adds one value per colony (whole colony, the
`compare.edge_band_um` edge band, and the center, r_norm ≥ 0.75) by condition.

---

## Boundaries, peaks and roughness

Module `downstream.boundary` (fits, pairs, peaks, bootstrap, bin check) and
`downstream.roughness` (border vs angle, registration); step `boundary`. The
statistics are in [STATISTICAL_METHODS.md §10d](STATISTICAL_METHODS.md#10d-boundaries-and-peaks).

**Per colony** (`tables/boundary_colonies.csv`, one row per colony × kind ×
subject × readout × metric, with `value`, `se`, `boot_support`, `ok`):

| kind | subject | metrics |
| --- | --- | --- |
| `marker` | one marker | `x50_um` (half-on position), `width_um` and `width_cells` (10 %→90 %), `direction` (−1 = edge-high); readout `positive` (gate calls) or `intensity` (shared scale) |
| `pair` | `EDGE / CENTER` | `crossing_um`, `overlap_um` (negative = gap), `dp_fraction`, `dp_expected`, `dp_enrichment` (double positives within `window_um` of the crossing; what independence predicts; their ratio) |
| `peak` | a signaling marker | `peak_um`, `trough_um`, `peak_ratio`, `prominence`, `offset_um vs <boundary>` |
| `roughness` | a gated marker | `roughness_um` (residual SD of the border radius), `raw_sd_um`, `decenter_um`, `mean_border_um`, `width_unregistered_um`, `width_registered_um`, `width_ratio`, `n_sectors_ok` |

**Which pairs.** `pairs: auto` takes, per stain set, every two gated markers
whose positive fraction changes in opposite directions from edge to center by
at least 0.2. Pairs never cross stain sets (the two markers must be measured in
the same nuclei). The run log lists the pairs found.

**Accepted or not.** A fit is valid when x50 lies inside the colony, the width
is below the colony radius and (intensity) the transition spans at least
`min_amplitude` of the shared scale. A value is `ok` when at least
`min_support` (80 %) of the bootstrap replicates also find it. Only `ok` values
enter summaries, effects and figures. When a pair that other colonies of the
same stain set have is missing in a colony, the colony and multimarker figures
state the reason in the title:

| Note | Meaning |
| --- | --- |
| `no boundary (found in 30 % of bootstrap resamples, needs 80 %)` | a crossing exists in the full data but depends on which angular sectors are sampled: the domain is scattered, not coherent |
| `no boundary (no accepted HOXB1 transition)` | one marker has no transition in this colony (e.g. its negative control, or a dose that does not switch it on) |
| `no boundary (the two transitions do not face each other)` | both have transitions, but in the same direction in this colony |

**Across colonies** (`tables/boundary_summary.csv`): mean, SD, SE and t-based
CI per condition at the colony level and at the replicate unit (`primary =
True` marks the unit the pipeline chose); `n_colonies_rejected` counts colonies
without an accepted value.

**Bin check** (`boundary.bin_check`): `boundary_bin_check.csv` (every colony's
refitted value per scheme), `boundary_bin_check_summary.csv` (median shift in
µm and in SE units, fraction within 2 SE, per condition and pooled) and
`figures/boundary/bin_check.png`. The three largest shifts are logged.

**Figures** (`figures/boundary/`): `pair_<A>_<B>.png` (crossing, overlap,
double positives by condition), `peak_<marker>.png`, `roughness_<marker>.png`,
`registered_<marker>.png` (profiles against distance from each sector's own
border), `borders/` (border radius vs angle per colony), `bin_check.png`.

---

## Density, saturation and normalization captions

**Nuclear density.** For each colony and µm bin, unique nuclei ÷ colony area
in the bin × 1000 (nuclei per 1000 µm²). The area comes from a histogram of the
colony mask's pixel distances to the edge, or from annuli of the fitted circle
when no mask is available; bins with less than 5 % of the largest bin's area
are left empty. `density_colonies.csv` per colony, `density_profiles.csv` by
condition over the replicate unit, `figures/profiles/density_um.png`, and a
strip under each colony profile whose bar widths are that figure's bin widths.

**Saturated nuclei.** The `qc` step finds nuclei whose raw median is ≥
`qc.saturation_fraction` × the channel's ceiling in any channel, the nuclear
channel included. The ceiling is `saturation_value` from the extraction QC
(`field_channel_qc.csv`) or `qc.saturation_ceiling`. With
`drop_saturated_nuclei: true` they are removed from every later step;
`tables/qc_saturated_nuclei.csv` lists them per image and channel (written only
when there are any).

**Captions.** Every colony and condition profile figure carries, per marker,
how its value was formed and scaled ("raw nuclear median, P1–P99 → 0–1 per
plate"; a signaling marker's unit), that no colony is rescaled on its own, and
`visualization.caption`. `nuclear_stain` (or its `auto` detection) adds a
warning to the caption and the log when a marker is divided by a histone label.

---

## Effects, trends and power

Module `downstream.effects`, step `effects`. Input: every accepted per-colony
number from `boundary_colonies.csv` and `colony_features.csv`.

| Table | Content |
| --- | --- |
| `effects_variance.csv` | per metric: `sd_colony`, `sd_well`, `sd_experiment` (NaN when a level is not replicated, never 0) |
| `effects_vs_reference.csv` | each condition minus the reference: `diff`, `diff_rel`, Welch 95 % CI, `hedges_g`, `p_perm` (`exact`), `unit`, `pseudo_replicated` |
| `effects_trends.csv` | slope vs each numeric condition column (per decade when log-scaled), bootstrap CI, permutation p, Spearman ρ; optional decay τ |
| `effects_power.csv` | `mdd` (minimum detectable difference) for 2, 3, 4, 6 experiments per condition; `experiment_sd_source` says whether σ_experiment was estimated, assumed or set to 0 (a lower bound); `power_at_target` for each of `effects.targets` |

`effects.reference: auto` uses the lowest level of the trend column within each
group (e.g. the lowest dose); `trend_columns: auto` uses numeric condition
columns with at least `min_levels` levels; `scale: auto` fits on log10 when the
positive levels span ≥ 10× (a zero level is then left out of the fit, and the
table says so).

---

## Replication: what n is

Nuclei are measurements; **colonies are the biological replicates**. Every
error band and test across colonies has n = colonies (or wells), never nuclei.

`stats.replicate: auto` uses **wells** when every condition has at least
`stats.min_wells_for_well_unit` (3) wells (colonies within one well share one
seeding, one dose and one staining reaction, so they are not independent), and
averages colonies within each well first. Otherwise colonies are the replicate,
and the report states the consequence:

* one well per condition: well-to-well variation cannot be separated from the
  condition, so a difference shows the colonies differ, not that it would
  replicate in another well;
* 2 wells per condition: colonies from one well are partly dependent, so
  p-values are optimistic.

`replicate: colony` or `well` forces a choice.

**Independent experiments.** `stats.experiment_column` names the plate-map
column of the differentiation (batch, experiment). With at least
`min_experiments_for_unit` (2) per condition, colonies are averaged per well,
wells per experiment, and n = experiments in the profiles, the boundary summary
and `effects`. Two differentiations are then never pooled into one curve.
Details in [STATISTICAL_METHODS.md](STATISTICAL_METHODS.md).

---

## Guardrails

What stops a wrong number from looking like a right one:

| Where | Guardrail |
| --- | --- |
| `check` | every image maps to a well, every well has a condition, every stain set has channels; markers without a negative control; a pixel size exists; `diameter_column` exists in every well |
| GUI | plate maps validated with the pipeline's own loader before saving; `.bak` of the previous file; colony preview |
| extraction | per-channel saturation and signal-to-background (`field_channel_qc.csv`) |
| `colony` | no pixel size → the image fails; border-cut colonies switch to the circle and are flagged; escaped cells trimmed; neighboring colonies and strays excluded |
| `qc` | eight colony flags (below); per-marker check of each colony against the other colonies of its well; flags reported on every overlay and in the report; saturated nuclei dropped; histone denominator warned |
| `signaling` | background from the bare surface around the colony, never from gaps inside it; per-image background outliers flagged within a stain set; no reference → absolute units, never mixed |
| `boundary` | a fit must be inside the colony, narrower than it and span `min_amplitude`; a value must be found in `min_support` of bootstrap replicates; bootstrap over angular blocks (not nuclei); pairs only within a stain set; a missing boundary is explained, not drawn; bin check |
| `effects` | non-replicated levels reported NaN, not 0; comparisons below two replicates per side flagged `pseudo_replicated`; power from one experiment labeled a lower bound |
| `thresholds` | gate from the marker's own control; leave-one-colony-out gate stability; no gate rather than a guessed one |
| `profiles` | minimum nuclei per bin; one shared scale per scope (no per-colony stretching of a non-expressing colony) |
| `multimarker` / `coexpression` | skipped with fewer than two markers; pairs only within a stain set; correlations per colony, then averaged (never pooled nuclei); minimum nuclei per bin; trend z labeled descriptive |
| `aggregate` / `compare` | minimum replicates per point; replicate unit chosen from the design and stated; exact enumeration for small designs; max-F family-wise control along the radius; BH across pairs and features; 2-D and 3-D never pooled; normalized grid not used across different pattern sizes |
| runner | a step re-runs when its settings or its inputs changed; one failing step never stops the others |

**Colony QC flags** (`tables/qc_colonies.csv`):

| Flag | Meaning | Threshold |
| --- | --- | --- |
| `failed` | no colony found | - |
| `clipped` | rim cut by the image border; edge distance from the circle | `colony.max_border_fraction` |
| `size` | diameter far from the expected pattern diameter | `qc.max_diameter_deviation` |
| `shape` | not round: mask fills too little of its circle (unclipped only), or rim far from the circle | `qc.min_fill_ratio`, `qc.max_fit_rms_rel` |
| `neighbor` | a neighboring colony nearly as large as the main one | `qc.max_neighbour_ratio` |
| `outside` | many nuclei outside the main colony | `qc.max_outside_fraction` |
| `low_count` | few nuclei in the colony | `qc.min_nuclei_per_colony` |
| `field` | nucleus count, DAPI or nuclear size unlike the other colonies of its condition in the same dataset and segmentation mode (robust z **and** relative deviation; a 2-D and a 3-D dataset of the same colonies are never pooled here) | `qc.outlier_z`, `qc.field_min_rel_dev` |

Flags are always reported. Only those listed in `qc.exclude_flags` (default
`[failed]`; `[all]` for every flag) remove a colony from profiles, scales, gates
and tests. Flagged colonies that are kept are drawn dotted in the colony
galleries.

---

## Output files

Per image (in the image's result folder, next to the image):

| File | Content |
| --- | --- |
| `nuclear_masks.tif`, `segmentation_overlay.png` | segmentation and its QC picture |
| `per_nuclei_intensities.joblib` | per-nucleus intensities (raw, ÷ DAPI), centroids, sizes, channel health |
| `colony_mask.tif`, `colony_overlay.png` | the main colony and its QC picture |
| `colony_nuclei.parquet` | per nucleus: `in_colony`, `dist_edge_um`, `r_center_um`, `r_norm` |
| `colony.json` | diameter, fit quality, border fraction, neighbors, counts, settings |
| `signaling_nuclei.parquet` | signaling channels only: per nucleus `nuc`, `ring`, `n_ring_px`, `background`, `nuc_bgsub`, `nc_ratio` |
| `signaling.json`, `signaling_rings.png` | background per channel, settings; the ring QC crop |
| `segmentation_metadata.json` | the mode (2d / 3d) and settings the masks were made with |

A result folder holds one segmentation. Re-segmenting (another mode, or
`--force-step segment`) makes the intensities and the colony geometry of that
image stale, and they are redone on the next run.

Per experiment (the analysis folder: `analysis_2d/`, `analysis_3d/`, … with `results_dir: auto`):

| File | Content |
| --- | --- |
| `report.html` | every figure and the key tables, warnings first |
| `cells.parquet`, `intensities.parquet` | one row per nucleus / per nucleus × marker (`raw`, `dapi_norm`, and for signaling markers `nuc_bgsub`, `nc_ratio`), with colony columns and plate-map conditions |
| `colonies.csv` | one row per colony: geometry + conditions |
| `tables/qc_colonies.csv` | colony flags and `excluded` |
| `tables/colony_profiles.csv` | per colony × marker × grid × bin: `n_nuclei`, `mean`, `sd`, `sem`, `ci_lo/hi`, `positive_fraction` |
| `tables/colony_profiles_binned.csv` | per-colony profile on its own bins (`x` = µm from the edge) |
| `tables/well_profiles.csv`, `condition_profiles.csv` | means ± CI across colonies / replicates, per `readout` (`mean` intensity, `positive_fraction`) |
| `tables/compare_global.csv` | one functional test per marker × grid × readout: global p, exact or not, smallest possible p, where along the radius it differs |
| `tables/compare_pointwise.csv` | F, p and significance per grid point |
| `tables/compare_pairwise.csv` | every pair of conditions, with BH q |
| `tables/colony_features.csv`, `feature_tests.csv` | edge / center / mean level, peak and transition distance per colony; Kruskal-Wallis with BH q |
| `tables/similarity_matrix*.csv`, `similarity_within_between.csv` | colony × colony profile distance; mean within vs between conditions |
| `tables/scale_anchors.csv`, `thresholds.csv` | the shared-scale anchors and the gates |
| `tables/replicate_unit.json`, `compare_summary.json` | the replicate unit and its caveat; the headline tests |
| `tables/multimarker_trends.csv` | local slope, z and direction per colony (µm) and well (normalized) × marker × bin |
| `tables/coexpression_colonies.csv` | per colony × pair × grid × bin: correlation ± CI, bin means, positive / double-positive / expected fractions, log2 enrichment |
| `tables/coexpression_wells.csv` | the same per well: mean over colonies ± CI, `n_<readout>` colonies |
| `tables/signaling_reference.csv` | per signaling marker × scope: readout, reference wells, reference level, how many colonies, where it came from, whether values are relative |
| `tables/signaling_colonies.csv` | per colony × signaling marker: median value of the whole colony, the edge band and the center, with its unit |
| `tables/signaling_background.csv` | per image × signaling channel: background, MAD, method (`local` / `image` / `offset`), cell-free pixels used, outlier flag |
| `tables/qc_saturated_nuclei.csv` | clipped nuclei per image and channel, with value and ceiling (only when any) |
| `tables/boundary_colonies.csv`, `boundary_fits.csv` | per-colony boundary, pair, peak and roughness values with SE and acceptance; the fitted curves |
| `tables/boundary_summary.csv` | the same across colonies / wells / experiments by condition |
| `tables/boundary_sectors.csv`, `boundary_registered.csv` | border position per angular sector; profiles relative to each sector's border |
| `tables/boundary_bin_check.csv`, `boundary_bin_check_summary.csv` | every position refitted with other bins; shift in µm and SE units |
| `tables/density_colonies.csv`, `density_profiles.csv` | nuclei per 1000 µm² per colony bin; by condition |
| `tables/effects_*.csv` | variance components, effects vs reference, trends, power (see [Effects](#effects-trends-and-power)) |

Figures (`analysis/figures/`): `qc/` (overlay galleries, geometry, segmentation
yield), `thresholds/`, `colony_profiles/` (one per colony), `profiles/` (every
colony by well, per marker × grid × readout), `compare/` (condition profiles with
significance, pairwise q, features, similarity), `multimarker_profiles/colonies/`
and `/wells/` (stacked markers + trend strip), `coexpression/pages/colonies/` and
`/wells/` (one page per colony or well, one row per marker pair),
`coexpression/panels/<level>/<kind>/` (each page panel as its own slide-sized
figure; `coexpression.panels` picks the levels, default wells only,
`coexpression.panel_kinds` the panels), `signaling/`
(ring QC galleries), `boundary/` and `boundary/borders/` (per-colony boundary
metrics by condition, registered profiles, border maps, bin check), `effects/`
(trends, effects vs reference, power), `profiles/density_um.png`, and
`signaling_summary/` (one value per colony by
condition; marker shape = stain panel, so a panel brightness offset cannot pass
for a dose effect). Intensity axes share one range per marker across every
colony and well figure (`visualization.ylim: shared`; `auto` lets each panel
autoscale). All use the Okabe-Ito palette
together with line styles and markers, so they also read in grayscale.

---

## Extending the pipeline

A new analysis is a function registered as a step, in your own module next to
`config.yaml`; nothing in `src/` changes:

```yaml
downstream:
  extra_modules: [my_analysis]
```

```python
# my_analysis.py
from downstream import register

@register("domain_widths", after=["profiles"], config_keys=["profiles"])
def domain_widths(ctx):
    curves = ctx.read_table("colony_profiles.csv")
    ...
    ctx.write_table(result, "domain_widths.csv")
```

It is checkpointed, selectable with `--steps` / `--force-step`, and its figures
(in `ctx.figure("<group>", "<name>.png")`) appear in the report. A worked example
is in [`examples/custom_analysis.py`](../examples/custom_analysis.py).

---

## Project structure

```
micropattern_pipeline/
├── main.py                       CLI: process | init-plate-map | manifest | check | downstream
├── launch_gui.bat/.command/.sh   start the GUI
├── gui/                          app.py (Streamlit) + core.py (file writers, no Streamlit)
├── hpc/                          generic LSF jobs, run.sh (re-runs), check_env.py, setup_env.sh
├── templates/                    config.yaml and hpc_job.env, fully commented
├── docs/                         CONFIG_GUIDE.md, REFERENCE.md, STATISTICAL_METHODS.md, PLATE_MAPS.md
├── examples/custom_analysis.py   an experiment-specific step
├── tests/                        synthetic tests, no image data needed
└── src/
    ├── pipeline_config.py        defaults + datasets → resolved dataset configs
    ├── selection.py              --seq / --wells / input.select
    ├── seg_status.py             which segmentation results exist; 2d/3d choice; analysis folder name
    ├── image_processing/         read, segment (Cellpose-SAM), extract: condition-agnostic
    ├── experiment/               filenames, plate geometry, plate maps, manifest
    └── downstream/
        ├── colony.py             colony mask, circle fit, edge distance, overlay
        ├── radial.py             shared scale, grids, per-colony curves, replicate unit
        ├── fda.py                functional permutation ANOVA, features, dissimilarity
        ├── assemble.py           manifests + joblibs + colony output → cell tables
        ├── qc.py                 colony flags, staining and channel QC
        ├── thresholds.py, controls.py, scope.py   positivity gates and their scope
        ├── trends.py, coexpression.py   local trends and co-expression of marker pairs
        ├── boundary.py           logistic boundaries, facing pairs, peaks, block bootstrap, bin check
        ├── roughness.py          border radius vs angle, roughness, angular registration
        ├── effects.py            variance components, effects, trends, power
        ├── captions.py           normalization captions, nuclear-stain kind, histone guard
        ├── signaling/            catalog (names), measure (nucleus / ring / local background),
        │                         readouts, reference (÷ reference wells)
        ├── steps.py              the built-in steps
        ├── registry.py, runner.py, context.py     plugins, checkpoints, execution
        ├── report.py             report.html
        ├── style.py              the only place colors are defined (Okabe-Ito)
        └── plots/                figure builders
```

---

## Tests

```bash
python tests/test_micropattern.py   # colony geometry, shared scale, permutation tests, end to end (~1 min)
python tests/test_downstream.py     # thresholds, controls, scopes, QC, palette rule
python tests/test_experiment.py     # plate maps, Seq → well, stains
python tests/test_selection.py      # --seq / --wells / input.select
python tests/test_gui_core.py       # everything the GUI writes reads back exactly
python tests/test_signaling.py      # signaling names, N:C against a known image, local background, reference
python tests/test_boundary.py       # boundary fits, crossing, peak, bootstrap, bin check, missing-boundary notes
python tests/test_roughness.py      # roughness, decentering, registration
python tests/test_effects.py        # variance components, effects, trends, power, multi-experiment run
python tests/test_density_captions.py  # bin areas, density, captions, histone guard, saturated nuclei
python tests/test_pipeline_options.py  # excluded markers, skipped steps, 2d/3d choice, stale checkpoints, GUI writers, HPC overrides
```

All at once: `python -m pytest -q -p no:cacheprovider tests/`.

`tests/synthetic.py` builds micropattern experiments with known answers (colony
radius, edge distances, profile shapes); the end-to-end test runs `check`,
`manifest` and `downstream` on one and checks the statistics find the difference
that was put in, and only that one.
