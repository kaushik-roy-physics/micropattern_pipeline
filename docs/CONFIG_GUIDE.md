# Setting up an experiment: the common cases

The plate map says **what is in each well**; `config.yaml` says **how to read
it**. Most mistakes come from a setting that is right for one experiment and
silently wrong for the next. This page goes through the decisions one at a time,
with the setting for each case. The GUI writes all of these; the YAML is shown so
you know what the GUI wrote. Field-by-field details are in
[REFERENCE.md](REFERENCE.md), the statistics in
[STATISTICAL_METHODS.md](STATISTICAL_METHODS.md).

Worked example throughout: `hd_micropatterns_bcat`, three stain panels in one
folder, H2B as the nuclear label, β-catenin in every panel, Wnt3a doses.

---

## 1. Stain panels: which marker is on which channel

| Your experiment                                               | Plate map                                                                                           |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| One panel for every well                                      | one stain set, `default`                                                                            |
| Several panels **in one folder** (wells stained differently)  | one stain set per panel; each well names its set with `stain:`                                      |
| Several panels **in separate folders** (one folder per panel) | one plate map per folder, and the **same `plate_id`** in all of them if they are one physical plate |

```yaml
stains:
  default: {0: OTX2-r, 1: b-CAT, 2: H2B, 3: HOXB1}
  panel2:  {0: GBX2,   1: b-CAT, 2: H2B, 3: HOXB1}
  panel3:  {0: GBX2,   2: H2B, 3: OTX2-g}      # b-CAT (ch 1) left out: that stain failed (§9)
wells:
  6: {wnt3a_ngmL: 0, stain: panel2}
```

What follows from it:

- Every well is measured on the channels in **Channels to measure**; the marker
  name comes from that well's own stain set. A colony's figures show only the
  markers it was stained for, and a marker's comparisons only use the wells that
  carry it.
- **Same marker, same spelling, every time.** `b-CAT` in one set and `bCat` in
  another are two markers to the pipeline (the check and the GUI warn). If old
  plate maps used another spelling, merge them with
  `marker_aliases: {bcat: b-CAT}`.
- **Two antibodies against one protein are two markers on purpose**: give them
  different names (`OTX2-r`, `OTX2-g`). They get separate scales and gates and are
  never mixed.
- Write the **marker**, not the fluorophore.

## 2. The nuclear channel

Set it under **Channels → Nuclear channel**. It is used for segmentation and is
never analyzed as a marker.

| Nuclear label                         | `default_space` (*Standard markers, by default*)                                                                |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| DAPI, Hoechst (DNA stains)            | `dapi_norm`: each marker ÷ that nucleus's DNA stain, which removes how much nucleus sits in the optical section |
| H2B-GFP or any other protein reporter | `raw`: its level varies from cell to cell, so dividing by it adds noise. The GUI warns                          |

Membrane or cytoplasmic markers stay `raw` either way:
`markers: {ECAD: {space: raw}}`.

`nuclear_stain: auto` recognizes a histone label from the channel's name
(`H2B`, `H3`, `histone`) and warns if any marker is still divided by it. Set
`nuclear_stain: dna` or `histone` when the name does not say (e.g. a channel
called `nuc`).

## 3. Which colonies share one intensity scale

A marker's intensities are put on one 0-1 scale per **scope**, and its % positive
gate is fitted per scope. Colonies on one scale must have been stained and imaged
alike.

| The same antibody …                                                              | `one scale across` / `scope`                                  |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| appears in one panel only, or with a different dilution, lot or exposure in each | `plate × panel` (default, always safe)                        |
| appears in several panels of one plate at the same dilution and exposure         | `whole plate (all panels)`: `markers: {GBX2: {scope: plate}}` |
| was stained and imaged identically across several plates                         | `whole experiment`                                            |

The check prints "GBX2 appears in 2 panels on the SAME channel" for every marker
that *could* be pooled. It cannot know whether the dilution matched: you decide.

## 4. Negative controls for % positive (the step that goes wrong most)

The intensity profiles never need a control. **% positive** needs a gate, and a
gate comes from, in this order:

1. **`control_for` in the plate map**: this well is the negative control for
   these markers.
2. **`conditions.marker_controls`** in config.yaml: the same, by well class.
3. **The blanket class**: every well whose `control_column` (usually `class`)
   is in `control_values` is the negative control for **every marker that has no
   `control_for`**, except markers listed in that well's `positive_for`.
4. **Nothing**: a two-population fit (GMM). If the marker does not show two clear
   populations, it gets no gate and no % positive profile, which is the honest
   answer.

| Your controls                                                               | Plate map                              | `conditions.control_values`               |
| --------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------- |
| Wells that are negative for every marker (no primary, secondary only)       | `class: control` on those wells        | `[control]`                               |
| A well that is the negative control for some markers only                   | `control_for: [HOXB1]` on that well    | whatever the rest needs                   |
| A control well that expresses one marker (e.g. pluripotent wells are SOX2+) | `class: control, positive_for: [SOX2]` | `[control]`                               |
| No negative controls at all                                                 | nothing                                | `[]` (every marker falls back to the GMM) |

> **The trap.** Putting `pluripotent` in `control_values` makes the pluripotent
> wells the negative control of *every* marker without a `control_for`, including
> markers pluripotent cells express (OTX2, SOX2, β-catenin, …). Their gate then
> sits above the pluripotent level and most positive cells read negative. This is
> what happened in `hd_micropatterns_bcat` for OTX2-r, OTX2-g and GBX2: the fix was
> `control_values: []`, keeping `control_for: [HOXB1]` on the HOXB1 control wells,
> so HOXB1 is gated from its controls and the other three by the GMM.
> 
> Use `class: pluripotent` to *describe* those wells (and to compare by it); list
> it in `control_values` only if pluripotent cells are truly negative for every
> marker that has no `control_for`.

After a run, `tables/control_assignment.csv` lists, per marker, which wells were
its control and which rule chose them. Read it once per new design.

## 5. What is compared

`condition_key` is the plate-map column(s) whose values define the groups.

| Design                                                     | `condition_key`                          | `facet_key`                                                         |
| ---------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| one factor (dose, or cell line)                            | `[wnt3a_ngmL]`                           | none                                                                |
| two crossed factors (line × dose)                          | `[line, dose]`                           | `[dose]` to compare lines within each dose (or the other way round) |
| reference wells that share a dose value with treated wells | include the column that tells them apart |                                                                     |

The last row matters in the example: wells 1, 6, 11 (Wnt3a 0) and the pluripotent
wells 5, 10, 15 (also `wnt3a_ngmL: 0`) are one condition, "0", when
`condition_key: [wnt3a_ngmL]`. To keep them apart, give every well a value in a
column that differs, e.g. `medium: diff` / `medium: pluripotent`, and use
`condition_key: [medium, wnt3a_ngmL]`.

## 6. Signaling markers (β-catenin, SMADs)

Recognized by name (any spelling); see the **readout** column in the GUI's
Markers table. Never gated.

| You want                                                                                      | `signaling:`                                                   |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| nuclear β-catenin in raw intensity (a.u.)                                                     | `nuclear_space: raw`, `reference_values: []`                   |
| the same with the local background subtracted (recommended when the reference wells are weak) | `nuclear_space: background_subtracted`, `reference_values: []` |
| relative to pluripotent colonies (1 = pluripotent)                                            | `reference_column: class`, `reference_values: [pluripotent]`   |
| SMAD2/3 N:C                                                                                   | automatic (`nc` readout); the same reference choices apply     |
| a reporter the catalog does not know (pSMAD1, YAP1)                                           | `markers: {pSMAD1: nuclear, YAP1: nc}`                         |
| a catalog marker read as an ordinary marker                                                   | `markers: {SMAD4: none}`                                       |

`reference_values` is independent of the negative controls in §4: removing
`pluripotent` from `control_values` does not change the signaling reference, and
the other way round.

**Is the reference good enough?** Dividing by a reference makes every value
inherit the reference's noise and bias. Before using one, open its colonies in
the QC overlays and check `qc_colonies.csv` and `density_profiles.csv`: in
`hd_micropatterns_bcat` each panel's pluripotent "reference" was a single
sparse colony with ~440 segmented nuclei (vs ~2 700 in a treated colony) and
large unsegmented areas, so `reference_values: []` was chosen and β-catenin is
reported in a.u. per panel.

**Background.** `background: local` (default) takes the median of the bare
surface in a `local_band_um` (60 µm) band just outside the colony; `image` uses
every cell-free pixel. Check `tables/signaling_background.csv` after the first
run: any flagged image is worth opening.

## 7. Colony size and the replicate

- **Pattern diameter**: `colony.expected_diameter_um: 700`, or
  `colony.diameter_column: diameter_um` when wells carry different sizes. Used to
  flag torn or merged colonies.
- **Replicate**: leave `stats.replicate: auto`. With ≥ 3 wells per condition,
  wells are n; otherwise colonies are n and the report says what that means.

## 8. Boundaries and peaks

Nothing to set for the common case: `boundary.pairs: auto` finds, per stain
set, every two gated markers that face each other (one high at the edge, one
high in the center), and `peak_markers: auto` searches every signaling marker
for an interior peak. The run log prints what it found:

```
boundary: stain set default: facing pair(s) HOXB1 (edge) / OTX2-r (center)
boundary: stain set panel3: facing pair(s) GBX2 (edge) / OTX2-g (center)
```

| You want                                                                   | Setting                                                                                                                   |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| a specific pair, or a pair auto did not find (e.g. no gate for one marker) | `boundary.pairs: [[GBX2, OTX2-g], [HOXB1, OTX2-r]]` (edge-high first)                                                     |
| no peak search                                                             | `boundary.peak_markers: []`                                                                                               |
| a different fitting bin                                                    | `boundary.bin_um: 10` (≈ one nucleus; the bin check tells you whether it matters)                                         |
| fewer or more bootstrap blocks                                             | `boundary.n_sectors: 12`, `n_boot: 200`                                                                                   |
| accept shakier values                                                      | lower `boundary.min_support` (0.8): not recommended; a value found in < 80 % of resamples is not determined by the colony |
| power for a specific effect                                                | `effects.targets: [{metric: crossing_um, delta: 20}]`                                                                     |

Pairs exist only within a stain set: two markers in different panels were
never measured in the same nuclei.

## 9. A stain that failed

When one antibody failed in one panel (a bad lot, a wrong dilution), take that
channel out of the panel's stain set in `plate_map.yaml` and say why in a
comment:

```yaml
stains:
  # b-CAT (ch 1) left out of panel3: that staining failed (2026-10-28).
  # Original: {0: GBX2, 1: b-CAT, 2: H2B, 3: OTX2-g}
  panel3: {0: GBX2, 2: H2B, 3: OTX2-g}
```

Then re-run downstream. The manifest is rebuilt, the cell tables are
re-assembled without that marker for those wells, and every step after re-runs
by itself. The marker is not measured, scaled, gated, used as a reference or
pooled for those wells; the other panels keep it. Excluding it here, rather than
ignoring it in the figures, is what keeps it out of pooled tables, shared
scales and dose trends. Keep a backup of the plate map
(`plate_map.yaml.<date>.bak`) so the change is recorded.

## 9b. A marker you do not want analyzed at all

Different from a failed stain: the stain is fine, it is just not part of this
question. Example: E-CAD and FOXA2 were both imaged on channel 1, E-CAD in some
wells and FOXA2 in others, and only FOXA2 matters. Channels are chosen by index,
so *Channels to measure* cannot keep one and drop the other; markers are chosen
by name. In GUI step 3, **untick *analyze*** for E-CAD in the *Markers* table:

```yaml
downstream:
  exclude_markers: [DAPI, ECAD]
```

E-CAD is then left out of every table, scale, gate, figure and test, in every
well; FOXA2 on the same channel stays. Use the spelling of the stain sets.
DAPI (the nuclear channel) is always out, listed or not.

## 9c. Analyses your experiment does not need

Everything runs by default. In GUI step 3, *Analyses to run*, untick what does
not apply, e.g. no facing domains and no dose series:

```yaml
downstream:
  skip_steps: [boundary, effects]
```

Optional: `compare`, `boundary`, `effects` (needs `compare` or `boundary`),
`multimarker`, `coexpression`, `signaling_summary`. The core (colony detection,
QC, thresholds, profiles, well and condition means, the report) always runs.
*Figure sets and sub-analyses* turns off single figure sets inside a step:
`profiles.colony_figures`, `colony.overlay`, `boundary.roughness.enabled`,
`boundary.bin_check: []`, `multimarker.colony_figures`, … Results of a skipped
analysis from an earlier run are moved to `<analysis>/_skipped/<step>/`, so the
report only shows what this run made.

## 9d. 2-D or 3-D results, and where the analysis goes

Each image's result folder holds one segmentation. To analyze an experiment both
ways, keep MIPs and z-stacks in two folders (datasets), one set to `2d`, one to
`3d`. Then:

| You want | `downstream.seg_mode` | Analysis folder (`results_dir: auto`) |
| --- | --- | --- |
| whatever has been segmented (default) | `auto` | `analysis_2d`, `analysis_3d`, or `analysis_2d_3d` with both side by side |
| only the 2-D results | `2d` | `analysis_2d` |
| only the 3-D results | `3d` | `analysis_3d` |

The GUI (step 3, *Segmentation results used*) shows what is on disk per folder
before you choose, and `check` prints the same. Datasets with no results yet are
left out with a note. Results are always labeled by the mode they were really
made in: a folder set to `3d` whose masks are still 2-D is analyzed (and named)
as 2-D, with a warning.

## 10. Independent differentiations

Two differentiations of the same design are two replicates of the batch, not
twice as many colonies.

1. List both acquisition folders under `datasets:` in one `config.yaml` (a
   `directory` may point to a sibling folder).
2. Add a column to every well of both plate maps, e.g. `differentiation: D1` /
   `D2`, and give each physical plate its own `plate_id`.
3. Set `stats.experiment_column: differentiation`.

With ≥ 2 differentiations per condition, profiles, the boundary summary and the
`effects` step use the differentiation as n, between-experiment variance becomes
estimable, and the power estimates stop being lower bounds. Markers with
`scope: plate` still get one scale per plate, so plates are never forced onto
one intensity scale.

## 11. Before you submit

1. GUI step 4, **Run check** (or `python main.py check --config config.yaml`):
   images that map to no well, markers spelled two ways, pooling candidates,
   which wells control which marker.
2. After the first run, open `report.html` (in `analysis_2d/` or
   `analysis_3d/`) and read, in order: the colony
   overlays (QC), `tables/control_assignment.csv`, `tables/thresholds.csv`
   (one row per marker × scope, with the gate and how it was found), the
   threshold evidence figures.
3. Changed the plate map? Rebuild the manifest (`python main.py manifest`; the
   HPC downstream job does it by itself) and re-run downstream. A changed
   manifest re-assembles the cell tables automatically, and every step after
   that re-runs.
