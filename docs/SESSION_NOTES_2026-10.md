# Session notes, 2026-10-27 to 2026-10-28: downstream improvements

The improvements answer your critiques of the qualifier proposal (`KR/critiques.docx`), using
`hd_micropatterns_bcat` as the test case. Everything in `src/` is generic. Anything specific
to the HD experiment lives only in `hd_micropatterns_bcat/config.yaml`.

**Where things are**

| What | Path |
|---|---|
| HD results with all the changes below | `hd_micropatterns_bcat/analysis_2026-10-27/` (open `report.html`) |
| Original HD analysis | `hd_micropatterns_bcat/analysis/` (left untouched) |
| HD config backups | `config.yaml.2026-10-27.bak` (before phase 0); `config.yaml.2026-10-28.bak` (before the b-CAT change) |
| HD plate-map backup | `stitched_zmax/plate_map.yaml.2026-10-28.bak` (before panel-3 b-CAT was dropped); manifest backups next to it |
| Methods | `docs/STATISTICAL_METHODS.md` (§5, §10d, §10e) and `docs/REFERENCE.md` |

---

## 1. Status at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Config fixes: pluripotent split from 0 ng/mL (`condition_key: [class, wnt3a_ngmL]`), axis label | done |
| 1 | Plot fixes: ticks, coexpression pages and panels, shared y-limits, signaling legend; groupby NaN bug | done |
| 2 | `boundary` step: boundary position and width, crossing, overlap, double-positive fraction, interior peak, all with SE | done |
| 3 | Roughness and angular registration (inside `boundary`) | done |
| 4 | Experiment > well > colony replicates; `effects` step (variance components, effects, trends, power) | done, validated on synthetic data |
| — | b-CAT: no division by the pluripotent control; robust local background | done (2026-10-28) |
| 5a | Nuclear density per radial bin; bin widths drawn under each colony profile | done |
| 5b | Normalization caption on profile figures; guard against an H2B denominator | done |
| 5c | Check that the bin scheme (fixed vs quantile) doesn't move x50, width or peak | done (session 2) |
| 5d | Drop saturated nuclei | done (session 2); no HD nucleus is clipped, so no change on HD |
| 5e | Curated talk/paper figure set at final physical size | **open** (design in §6) |
| — | Figures say why a facing-pair boundary is missing (`boundary.unresolved_pairs`) | done (session 3) |
| — | HD: panel-3 b-CAT excluded at the plate map (failed stain) | done (session 3) |
| — | README, REFERENCE, STATISTICAL_METHODS, CONFIG_GUIDE, PLATE_MAPS brought up to date | done (session 3) |

Tests: `python -m pytest -q -p no:cacheprovider tests/` (see §7 for the count at the end of
this session).

---

## 2. What each step now produces (new outputs only)

Steps run in this order: `colony → signaling → assemble → qc → thresholds → boundary → profiles →
aggregate → compare → multimarker / coexpression → effects → signaling_summary → report`.

| Step | New tables (`tables/`) | New figures (`figures/`) |
|---|---|---|
| signaling | `signaling_background.csv` (each image's background, method, outlier flag within its stain set) | — |
| qc | — (logs a warning if any marker is divided by a histone nuclear channel) | — |
| boundary | `boundary_colonies.csv`, `boundary_fits.csv`, `boundary_summary.csv` (colony/well/experiment levels), `boundary_sectors.csv`, `boundary_registered.csv` | `boundary/` (pair, peak and roughness metrics; registered profiles), `boundary/borders/` (border radius vs angle) |
| profiles | `density_colonies.csv` | colony profiles gain a density strip whose bar widths are the bin widths, the boundary lines with ± SE, the fitted logistics, peaks, and a normalization caption |
| aggregate | `density_profiles.csv` | `profiles/density_um.png` (nuclei per 1000 µm² vs distance, by condition) |
| compare | — | `compare/profiles_*_mean.png` now carry the normalization caption |
| coexpression | — | `coexpression/pages/{colonies,wells}` (combined) and `coexpression/panels/<level>/<kind>/` (one panel per file) |
| effects | `effects_variance.csv`, `effects_vs_reference.csv`, `effects_trends.csv`, `effects_power.csv` | `effects/` (trend, vs_reference and power figures per metric family) |

---

## 3. New config keys (all generic, with defaults; documented in `templates/config.yaml`)

```yaml
downstream:
  nuclear_stain: auto          # dna | histone | auto (H2B in the nuclear channel's name → histone)
  visualization:
    ylim: shared               # shared | auto
    caption: null              # free text appended to every normalization caption (e.g. the cell line)
  coexpression:
    panels: [wells]            # which levels also get one-panel-per-file figures
    panel_kinds: [...]
  signaling:
    nuclear_space: background_subtracted
    reference_values: null     # [] = no reference: absolute units
    background: local          # local (band just outside the colony) | image (all cell-free pixels)
    local_band_um: 60
  boundary: {...}              # pairs: auto | [[EDGE, CENTER], ...]; peak_markers; bin_um; n_sectors; n_boot; ...
    roughness: {...}           # n_sectors: 16, ...
  stats:
    experiment_column: null    # plate-map column naming the independent differentiation
    min_experiments_for_unit: 2
  effects:
    power: {assumed_experiment_sd: null}
    targets: [{metric: crossing_um, delta: 20}, ...]
```

HD-specific settings (only in `hd_micropatterns_bcat/config.yaml`):
- `condition_key: [class, wnt3a_ngmL]`
- `axis_label: "WNT3A (ng/mL)"`
- `signaling.reference_values: []`
- `effects.targets` (20 µm for the crossing, x50 and peak; 0.03 for peak_ratio)
- `power.assumed_experiment_sd: 1.0`

---

## 4. Main results on HD (one differentiation, so everything is per colony, n = 5 per condition)

- **MHB as the GBX2/OTX2-g crossing:** 87 → 107 → 130 µm from the edge at 100 / 300 / 1000 ng/mL
  WNT3A, about 42 µm per decade.
- **GBX2 and OTX2 don't abut.** There is a 38–82 µm gap between the two domains, and
  double-positive cells are 8–13× rarer than independence predicts.
- **GBX2 border:** roughness 7–12 µm (about one cell). Angular registration narrows the transition
  only 8–33%, so the ~50 µm (6–7 cell) graded transition is real, not an artifact of averaging
  over angle.
- **b-CAT peak vs the GBX2/OTX2-g crossing:** void (panel-3 b-CAT staining failed). Use the
  peak against the HOXB1/OTX2-r crossing and the OTX2-r x50 (panel 1) instead.
- **HOXB1/OTX2-r boundary missing at 0 and 100 ng/mL (wells 1, 2): genuine.** HOXB1+ nuclei are
  0.3% (well 1) and 7.6% (well 2; 20% in the outer 50 µm) against 46–58% at 300/1000. In well 2
  the crossing is found in only 30–76% of bootstrap resamples (< min_support 0.8), so it is
  rejected and not drawn.
- **Power:** these are lower bounds, because between-differentiation variance can't be estimated
  from one differentiation. A 20 µm crossing shift needs about 4 differentiations per condition,
  assuming the between-experiment SD equals the colony SD. The 0.03 peak-ratio target is
  underpowered.

**b-CAT (changed 2026-10-28; final HD numbers in §8):**
- **Why the pluripotent control was dropped.** In each panel the pluripotent "reference" is one
  sparse, ragged colony with ~440 segmented nuclei, against ~2,700 in a treated colony. Inside
  these colonies there are 40–70k cell-free pixels reading about 12,000 a.u. of b-CAT, so most of
  their nuclei were never segmented. b-CAT is now reported as nucleus core − local background, in
  a.u., per panel.
- **WNT wells still read lower than 0 ng/mL.** Even in raw units, WNT-treated wells read lower
  than the 0 ng/mL wells of the same panel (panel 1 medians: 0 ng/mL ≈ 9,800–11,700;
  100 ≈ 7,000–7,900; 300 ≈ 5,100–6,400; 1000 ≈ 6,600–7,100 a.u.). That pattern is in the data,
  not a normalization artifact.
  - Possible causes: membrane/junctional β-catenin entering the nuclear mask in a maximum
    projection; denser WNT colonies; the epitope and antibody.
  - Worth checking in the images, and with N:C (`signaling.markers: {b-CAT: nc}`).
- **Background was never the problem.** The local and whole-image backgrounds differ by only
  15–110 a.u., against 7,000–20,000 of signal.
- **Panel 3 b-CAT is a staining failure** (user, 2026-10-28): about 2× brighter, not biology.
  Panel-3 b-CAT numbers are excluded from interpretation, which includes the b-CAT peak vs the
  GBX2/OTX2-g crossing (GBX2 and OTX2-g share panel 3 only). Panel-3 GBX2 and OTX2-g are fine.
  b-CAT conclusions come from panels 1 and 2.

---

## 5. Running on data in a different folder (same experiment design)

The paths in `config.yaml` are relative to the folder that holds it: `datasets[].directory:
./stitched_zmax` and `downstream.results_dir: ./analysis`. A new data folder therefore needs
its own copy of the config next to it.

1. **Lay out the folder** like `hd_micropatterns_bcat`:
   `NEW/config.yaml`, `NEW/<acquisition folder>/*.nd2`.
2. **Copy the config:** `cp hd_micropatterns_bcat/config.yaml NEW/config.yaml`. This carries over
   every HD-specific setting from §3.
   - Check `datasets:` and fix each `name` / `directory` if the acquisition folder name differs.
   - Check `channels.dapi_index` / `marker_channels`, the pixel size (read from the files), and
     `intensity.saturation_value`. Set the last to 4095 if the camera is 12-bit in a 16-bit file.
   - Keep `results_dir`, or use `OUT=` when running.
3. **Write a `plate_map.yaml` in each acquisition folder.** This is the part that almost always
   changes:
   - `imaged_wells` and `points_per_well`: the Seq → well mapping in acquisition order.
   - `stains:`, the channel → marker mapping per stain set. Keep the marker names identical
     (`OTX2-r`, `OTX2-g`, `GBX2`, `HOXB1`, `b-CAT`, `H2B`) so tables and settings match.
   - `wells:` with `wnt3a_ngmL`, `class`, `stain`, `control_for`.
   - `plate_id`: a new value for a different physical plate.
   - The GUI (`launch_gui.sh`) writes this file for you. Then run
     `python main.py check --config NEW/config.yaml`.
4. **If this is another independent differentiation that should be pooled with HD:**
   - Add a column to every well in both plate maps, e.g. `differentiation: D1` / `D2`, and set
     `stats.experiment_column: differentiation`.
   - List both acquisition folders under `datasets:` in ONE config. A `directory` can be a
     relative path to another folder; confirm with `main.py check`.
   - Use distinct `plate_id`s. Markers declared `scope: plate` get one scale per plate, so plates
     are never put on one intensity scale.
   - With ≥ 2 differentiations per condition, `effects` and `boundary_summary` switch to the
     differentiation as n, and the power estimates stop being lower bounds.
5. **Segment and extract:** `bash run_local_2dseg.sh NEW` (GPU if available; the HPC alternative
   is `hpc/run_pipeline_2dseg.lsf`). Then build the manifest with
   `python main.py manifest --config NEW/config.yaml`; the 2dseg script prints this.
6. **Run the analysis:** `bash run_local_downstream.sh NEW`. Optionally add `OUT=NEW/analysis_<date>`.
7. **Check first:**
   - `report.html`
   - `tables/qc_colonies.csv` and the QC galleries
   - `tables/signaling_background.csv` (outliers)
   - the log, for histone or pooling warnings
   - the colony count of every reference well. A sparse pluripotent colony like HD's is a reason
     to keep `reference_values: []`.

Notes:
- **Checkpoints:** they hash the config, not the code. After any code change, rerun with
  `FORCE_STEPS="<step> ..."`. Never add `signaling` to that list unless you want all images
  remeasured; that step always runs and remeasures only when its settings change.
- **Shared drive:** the SMB share gets very slow while a run is writing to it.

---

## 6. Open items (phase 5 remainder) and designs

- **5c, bin-scheme check** (critique: "check that the fitted peak position and transition width do
  not move when you switch to fixed-width bins").
  - In `boundary.measure_colony`, refit per colony with fixed bins of 5, 10 and 20 µm and with
    quantile bins. `fit_logistic` / `fit_four_pl` already accept arbitrary bin centers.
  - Write `boundary_bin_check.csv` with the per-condition median shift relative to the colony SE,
    plus one figure.
  - Config: `boundary.bin_check: [5, 20, quantile]`.
- **5d, drop saturated nuclei** (critique: "raw intensity at the ceiling … bright debris").
  - No extractor change is needed. `per_nuclei_intensities` holds each nucleus's median, and
    `field_channel_qc.csv` holds each channel's `saturation_value`.
  - In `Context.data`, drop nuclei whose median is ≥ `qc.saturation_fraction` × ceiling in any
    channel, nuclear channel included. Allow an optional `qc.saturation_ceiling` override for
    12-bit data in 16-bit containers.
  - Log the count per field (`qc_saturated_nuclei.csv`).
  - Test on a synthetic image with clipped nuclei.
- **5e, curated figure set** (critiques: one fixed axis direction, readable at print size, 5–6
  panels).
  - New `curated` step and `plots/curated.py`, drawn under rcParams presets
    (`curated.preset: slides | paper`) at a final physical size. Every panel carries an
    "edge (posterior) → center (anterior)" arrow, with the labels taken from
    `visualization.axis_ends`.
  - The panels:
    1. condition profiles of the boundary markers
    2. crossing, width and double-positive fraction with CI
    3. b-CAT peak vs crossing
    4. density per bin
    5. dose trend
- **Not pipeline work, from the critiques:**
  - Proliferation and death readouts (EdU / pHH3 / cleaved caspase-3) will slot in as ordinary
    markers.
  - Stripe patterns need a linear-coordinate mode; that would be a separate project.

---

## 7. Code map (what changed where)

- **Plots base and joint:**
  - `plots/base.py`: `no_data()` no longer wipes shared ticks.
  - `plots/joint.py`: coexpression single-panel figures, scaled text, wrapped titles.
- **`aggregate.py`:** `condition_label` vectorized; warns when one condition pools different
  classes; `dropna=False` in every groupby on condition keys (also in radial, steps, scope,
  coexpression, qc, stats).
- **`radial.py`:**
  - `display_limits` (shared y-limits).
  - `choose_unit` / `to_unit` handle experiments.
  - `ring_areas`, `density_curves`, and `colony_binned(edge_area=)` for density and bin areas.
- **New modules:**
  - `boundary.py`, `roughness.py`, `effects.py`, `captions.py`.
  - Their plots: `plots/boundary.py`, `plots/effects.py`.
- **`plots/radial.py`:** colony profile gains annotations, the density/bin-width strip and the
  caption; condition profiles gain a caption.
- **`plots/signaling.py`:** rewritten (legends, reference marked, n per condition).
- **`signaling/measure.py`:** `covered_area`, `background_masks`, local → image → offset
  background; `background_table`.
- **`steps.py`:**
  - New steps `boundary` and `effects`.
  - Density, captions and the histone guard.
  - The `_edge_geometry` helper, which reads each colony mask once.
- **`report.py`:** the new sections and tables.
- **Tests:**
  - New: `test_boundary.py`, `test_roughness.py`, `test_effects.py` (with a synthetic end-to-end
    run), `test_density_captions.py`.
  - Extended: `test_signaling.py` (local background, outlier table), `test_downstream.py`,
    `test_micropattern.py`.
  - `tests/synthetic.py` generates multi-experiment, multi-dose data with known answers.

---

## 8. Where this session stopped, and how to resume

**State when you left (2026-10-28, about 12:25).**
- A phase 5 HD pass was running (`profiles` onward, run from this session). Its log is
  `/tmp/claude-1001/.../scratchpad/run_p5.log`, which is temporary and may be gone when you return.
  - It finished `qc`, `thresholds`, `boundary` and `profiles` without errors.
  - It was still running `aggregate`; `compare`, `multimarker`, `coexpression`, `effects`,
    `signaling_summary` and `report` were still to come.
- One code change was made after that run started, so its figures don't include it: long y-labels
  are now wrapped on colony profiles, in `plots/radial.py`.
- The full test suite was **not** re-run after the phase 5 edits. Only `test_signaling.py` and
  `test_density_captions.py` were run (13 passed).

**To resume, run these from `micropattern_pipeline/`:**

```bash
# 1. Tests (about 5-10 min on the share)
conda activate cellpose_sam
python -m pytest -q -p no:cacheprovider tests/

# 2. Finish the HD pass, selectively. Checkpoints skip every step whose config is unchanged;
#    FORCE_STEPS redoes the steps whose CODE changed. Do not add `signaling`: it would remeasure
#    all 63 images, and it is already up to date.
OUT=../hd_micropatterns_bcat/analysis_2026-10-27 \
FORCE_STEPS="profiles aggregate compare" \
  bash run_local_downstream.sh ../hd_micropatterns_bcat
#    If the interrupted run did complete, this only redraws the profile figures (y-label fix) and
#    whatever depends on them.
#    To redo only figures quickly:  STEPS="profiles" FORCE_STEPS="profiles" ...

# 3. Check
#    figures/colony_profiles/*.png    density strip, caption, wrapped labels
#    figures/profiles/density_um.png  density by condition
#    tables/signaling_background.csv  no outliers expected
#    report.html
```

Which steps to force after future code changes:

| Code changed in | Force these |
|---|---|
| `signaling/measure.py` (measurement) | `signaling` (remeasures all images, ~15 min) |
| `qc.py` / saturated-nucleus drop | `qc` (everything after it reruns from its outputs) |
| `boundary.py`, `roughness.py`, `plots/boundary.py` | `boundary profiles effects` |
| `radial.py`, `plots/radial.py`, `captions.py` | `profiles aggregate compare` |
| `effects.py`, `plots/effects.py` | `effects` |

Then continue with the open items in §6: 5e (curated figures). 5c and 5d were done in session 2 (§9).

---

## 9. Session 2 (2026-10-28, afternoon)

- **Tests:** the full suite was run before any new work: 113 passed. After the changes below:
  **115 passed**, no warnings.
- **5d, saturated nuclei:**
  - `qc.saturated_nuclei` flags a nucleus whose median is at the ceiling in any channel,
    nuclear channel included.
  - `Context.data` drops these nuclei, and the qc step writes `qc_saturated_nuclei.csv`.
  - Config: `qc.drop_saturated_nuclei` (true), `qc.saturation_ceiling` (null; set 4095 for
    12-bit data in 16-bit files), `qc.saturation_fraction` (1.0).
  - Test: `tests/test_density_captions.py::test_saturated_nuclei_...`.
  - On HD every channel's ceiling is 65535 and no nucleus median reaches it, so nothing is dropped.
- **5c, bin check:**
  - `boundary.bin_check` (default `[fixed:5, fixed:20, quantile:20]`) refits each colony's x50,
    width, crossing, overlap and peak with other bins (point estimates only).
  - Outputs: `boundary_bin_check.csv`, `boundary_bin_check_summary.csv` (shift in µm and in
    bootstrap SE units) and `figures/boundary/bin_check.png`.
  - Synthetic truth (`tests/test_boundary.py::test_bin_check_...`): fixed 5/10/20 µm bins recover
    an interior peak at 175 µm within ~1 µm; 20 quantile bins pull it ~5 µm toward the edge.
    Positions of domain boundaries move < 2 SE under every scheme.
  - So the pipeline's fixed-bin fits are the right default. This answers the critique and is
    worth stating in the Approach.
- Docs updated: `REFERENCE.md` step table; `STATISTICAL_METHODS.md` §10d (bin check, density,
  saturated nuclei).
- **Final HD run: completed.** Every step ran without errors (13:24–13:34). HD outputs are now
  current with all code except one cosmetic fix noted below.
  - **Bin check on HD** (`tables/boundary_bin_check_summary.csv`, `figures/boundary/bin_check.png`):
    - The MHB crossing (GBX2/OTX2-g), the GBX2 x50 and width, and the b-CAT peak move less than
      0.5 SE under 5 µm, 20 µm and quantile bins. Across all metrics and schemes, 95% of median
      shifts are within 2 SE.
    - The OTX2 gate-call widths are bin-sensitive: they widen by 9–14 µm (about 2.5 SE) with
      20 µm bins.
    - With quantile bins, the b-CAT interior peak is found in only 12 of the 25 colonies where
      fixed bins find it. This is the coarse-interior risk raised in the critique.
  - **Density** (`figures/profiles/density_um.png`):
    - WNT-treated colonies: about 7–9 nuclei per 1000 µm², flat with distance.
    - 0 ng/mL colonies: sparser at the rim (about 5.5).
    - Pluripotent colonies: 1–2.5, measured only out to about 90 µm. They are sparse and
      under-segmented, which is another reason not to use them as a reference.
  - **Saturated nuclei:** none on HD, so `qc_saturated_nuclei.csv` is not written.
- **Cosmetic fix after the run:** the bin-check figure's title no longer overlaps the first row
  (`plots/boundary.py`). To redraw it:
  `OUT=../hd_micropatterns_bcat/analysis_2026-10-27 FORCE_STEPS="boundary" bash run_local_downstream.sh ../hd_micropatterns_bcat`.
  This also reruns everything after `boundary`, about 10 min.

---

## 10. Session 3 (2026-10-28, evening)

**Your corrections.**
- Panel-3 b-CAT (wells 11-15) is a staining failure, not biology.
- The missing HOXB1/OTX2-r band in wells 1-2 needed an explanation.

**1. Figures now say why a boundary is missing.**
- `boundary.unresolved_pairs(colonies, min_support)` → `{field_id: [note]}`, for
  every facing pair that other colonies of the same stain set have but this one
  lacks. Three cases:
  - "found in N % of bootstrap resamples, needs 80 %";
  - "no accepted <marker> transition";
  - "the two transitions do not face each other".
- Read by `_boundary_annotations` (`steps.py`) and printed in the titles of
  colony profiles (`plots/radial.py`) and colony multimarker figures
  (`plots/joint.py`; the top margin now grows with the number of title lines).
- Test: `tests/test_boundary.py::test_unresolved_pairs_say_why_a_boundary_is_missing`.
- On HD:
  - well 1: "no accepted HOXB1 transition" (HOXB1 negative control, 0.3 % HOXB1+);
  - well 2: crossings found in 30-76 % of resamples (7.6 % HOXB1+, 20 % in the
    outer 50 µm);
  - wells 3-4: every replicate finds the crossing (98-99.6 % HOXB1+ at the rim).
  HOXB1 switches on between 100 and 300 ng/mL, later than GBX2.
  - Two panel-3 notes worth a look: well 11 (0 ng/mL) colonies show "the two
    transitions do not face each other" for GBX2/OTX2-g.

**2. Panel-3 b-CAT excluded at the plate map.**
- `stitched_zmax/plate_map.yaml`: `panel3: {0: GBX2, 2: H2B, 3: OTX2-g}`, with a
  comment. Backups: `plate_map.yaml.2026-10-28.bak`,
  `manifest.{csv,parquet}.2026-10-28.bak`.
- The rerun (`FORCE_STEPS="assemble"`, everything after re-ran; 18:39-18:52) used
  42 signaling images instead of 63, with no re-measurement. Every step completed
  without errors.
- b-CAT results now come from panels 1-2 only (accepted per-colony values):

  | WNT3A (ng/mL) | b-CAT peak, µm from edge | peak / trough | offset from HOXB1/OTX2-r crossing |
  |---|---|---|---|
  | 100 | ~181 (5 colonies, panels 1+2) | 1.1 | no crossing (above) |
  | 300 | ~221 (1) | 1.1 | ~100 µm (1) |
  | 1000 | ~238-260 (3) | 1.2-1.3 | ~110 µm (2) |

  So the weak interior b-CAT maximum sits on the OTX2 (anterior) side, about
  100 µm inside the posterior boundary, and moves inward with dose like the
  boundaries. With 1-5 colonies per cell this is anecdotal.
- The old "b-CAT peak vs GBX2/OTX2-g crossing" result is void (panel 3 only).

**Tests:** 116 passed (one new), no warnings.

**3. Documentation.**
- `README.md`: what the analyses answer; how to read boundaries, effects and
  density; signaling units; captions, saturated nuclei and experiments;
  troubleshooting rows (missing boundary, failed stain, background outlier);
  re-running after code changes (force table); the same experiment in another
  folder; the full test table.
- `docs/REFERENCE.md`: new sections (Boundaries, peaks and roughness, with every
  table column and the missing-boundary notes; Density, saturation and captions;
  Effects, trends and power), plus the new settings table rows, step diagram,
  guardrails, output files, module map and tests.
- `docs/STATISTICAL_METHODS.md`: nuclear-stain kind and saturation (§1); the
  background check and the no-reference option (§10c); "when a colony has no
  boundary", the bin check on real data, density, saturated nuclei and captions
  as subsections (§10d); four new limitations (§11); quick-reference rows (§12).
- `docs/CONFIG_GUIDE.md`: the H2B guard; whether a signaling reference is good
  enough; new §8 Boundaries and peaks, §9 A stain that failed, §10 Independent
  differentiations.
- `docs/PLATE_MAPS.md`: a channel left out of a stain set is not analyzed.
- `templates/config.yaml`: signaling comment.

**Next:** 5e (curated figure set), designed in §6. Nothing else is pending.

---

## 11. Session 4 (2026-11-03/04): a new experiment, hd_micropatterns_v2

**Experiment.** `../hd_micropatterns_v2`: iPSC β-catenin:GFP, DAPI, GBX2 (561), OTX2 (640;
the nd2 metadata calls it "640 GBX2"). 23 colonies: 0 / 100 / 300 ng/mL WNT3A, CHIR 4 µM
(5 each), 3 pluripotent. `zmax/` (MIPs, 2-D) and `zstacks/` (76 planes at 1 µm, 3-D) in
one config, one plate map per folder (same `plate_id`). Deliverables for a student's
qualifier in `quals_deliverables/` (README there).

**HPC.** `hpc_job.env` there makes `SEQ`, `CONFIG`, `DOWNSTREAM_DATASETS` and
`DOWNSTREAM_OUT` overridable at submission (`SEQ="0-5" bsub < …`), so the 3-D
segmentation ran as four parallel GPU jobs (~22 min per stack on an RTX PRO 6000; 2-D took
2 min for all 23). `config_au.yaml` (b-CAT in a.u., `analysis_au/`) and `config.yaml`
(b-CAT ÷ pluripotent, `analysis/`) ran as chained downstream jobs; they share per-image
signaling output because the reference is not in its hash.

**Pipeline fixes (generic).**
- `qc.field_qc` compared each colony with the other colonies of its condition across
  *all* datasets: with a 2-D and a 3-D dataset of the same colonies (5x the nuclei, other
  DAPI levels) the spread was inflated and a dim colony went unflagged. Peers are now
  dataset × segmentation mode × condition. Test:
  `test_field_qc_compares_fields_within_their_own_segmentation_mode` (fails on the old code).
- `main.py check` counted `class: pluripotent` as a negative control by default, but the
  analysis (`controls.py`) only uses `class: control` when `control_values` is empty.
  `check` now uses the same default; REFERENCE says so.

**Experiment decisions (in its config).**
- Seq 0 excluded with `input.select.seq: "1-22"`: DAPI ~2.4x and OTX2 ~4.6x dimmer than its
  siblings in the raw, unstitched tiles too (same exposure/gain), 488/561 normal: the
  405/640 lasers at the first position. (`qc.exclude_flags` is back to `[failed]`.)
- `default_space: raw`: `intensity.normalize_by_dapi` is off, so there is no ÷DAPI column.
- b-CAT ÷ pluripotent rejected: CHIR falls below pluripotent; pluripotent colonies are
  deformed monolayers (2-D = 3-D nuclei count), the others a ~40 µm multilayer.

**Results.** GBX2/OTX2 border 116 ± 21 / 131 ± 10 µm (100 / 300 ng/mL, 3-D; 105 / 121 in 2-D).
b-CAT minimum at the border and a second peak 64 ± 9 / 76 ± 5 µm on the OTX2 side (3-D;
~100 µm in 2-D), in 9/10 WNT3A colonies, absent at 0 ng/mL and CHIR (3-D). 3-D finds ~4.5x the
nuclei of the MIP. In 3-D the colony center is layered for OTX2 (glass-side cells OTX2-low).
0 ng/mL colonies are ~1.5x brighter in every channel incl. DAPI: compare positions, not levels.

**Tests:** 117 passed (one new) after the QC fix.
