# Micropattern radial-profile analysis: with a point-and-click set-up

This pipeline takes images of micropatterned colonies (`.nd2`, `.tif`, `.czi`, …),
finds every nucleus, finds the colony in each image and its edge, and measures
how every marker changes **with distance from the colony edge**. Each colony gets
a radial profile on two coordinates:

- **µm from the edge**: the absolute length scale patterning is organized on;
- **fraction of the radius** (0 = edge, 1 = center): for comparing colonies of
  one size.

Profiles are averaged per well and per condition and compared with permutation
tests on whole curves, which also say *where* along the radius two conditions
differ. Each colony is then reduced to numbers with an uncertainty attached:
where each marker's domain ends and how sharp its border is, where two facing
domains meet, how rough the border is around the colony, where a signaling
peak sits, and how many nuclei there are per µm². Those numbers are compared
across conditions and doses with the replicate (colony, well or independent
experiment) stated, plus a power estimate for the next experiment. Everything
ends up in one `report.html`.

You describe the experiment once in a **GUI on your laptop** (Windows, macOS or
Linux). Segmentation runs on the **HPC GPU**; the analysis runs on an HPC CPU node
or on your laptop. No code or YAML needed.

```
 your laptop (GUI)                      the HPC
 ─────────────────                      ───────
 1 pick the experiment folder
 2 fill in the plate map     ──┐
 3 choose the analysis         ├──►  files saved next to your data  ──►  GPU job: segmentation
 4 check everything            │     (plate_map.yaml, config.yaml,        CPU job: colonies, profiles,
 5 HPC settings              ──┘      hpc_job.env)                                 statistics, report
                                                                                  │
                                                                                  ▼
                                                   analysis_2d/report.html in your experiment folder
```

---

## Contents

1. [Quick start](#1-quick-start)
2. [Installing and starting the GUI](#2-installing-and-starting-the-gui)
3. [The five steps of the GUI](#3-the-five-steps-of-the-gui)
4. [Running on the HPC](#4-running-on-the-hpc)
5. [Running on only some images](#5-running-on-only-some-images)
6. [Where everything lives](#6-where-everything-lives)
7. [Reading the results](#7-reading-the-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Command line](#9-command-line)
10. [For maintainers](#10-for-maintainers)

---

## 1. Quick start

**One-time:** install Python on your laptop ([§2](#2-installing-and-starting-the-gui)).
The HPC environment builds itself on the first job ([§4.1](#41-the-python-environment-on-the-hpc)).

**Setting up a new design** (several stain panels, a non-DNA nuclear label,
controls for some markers only, signaling markers, crossed factors)? Read
[`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md) first: one decision at a time, with
the setting for each case.

**Every experiment:**

1. Put the images on the lab share, one folder per acquisition, e.g.
   `KR/exp_040/zmax/`. **One image per colony**, centered on the pattern.
   Fiji-stitched tiles named `Point0000_….tif` work as they are.
2. Double-click **`launch_gui.bat`** (Windows) or **`launch_gui.command`** (Mac).
   The GUI opens in your browser.
3. Work through the five steps in the left sidebar. Press **💾 Save** on each.
4. Press **🔎 Run check** in step 4 and fix anything it reports.
5. Log in to the HPC and type the two lines shown in step 5: `cd` to the
   experiment folder, then `bsub < …/run_pipeline_2dseg.lsf`. Segmentation
   starts, and the analysis starts by itself when segmentation has finished.
6. Open `exp_040/analysis_2d/report.html` (`analysis_3d` for 3-D results).
   Look at the colony overlays first.

---

## 2. Installing and starting the GUI

### You need

- **Python 3.10 or newer** from <https://www.python.org/downloads/>. On Windows,
  tick **"Add python.exe to PATH"** in the installer.
- The lab share mounted on your laptop (the drive or folder where you see `{your_drive_name}/`).

### Starting it

| Computer    | Do this                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| **Windows** | Double-click `launch_gui.bat` in this folder.                                                           |
| **macOS**   | Double-click `launch_gui.command`. If macOS refuses ("unidentified developer"), right-click → **Open**. |
| **Linux**   | `bash /path/to/micropattern_pipeline/launch_gui.sh`                                                     |

The **first start takes a few minutes**: it builds a private Python environment
in your home folder (`~/.micropattern_pipeline_gui_venv`), so nothing else on
your computer is touched. After that it starts in seconds.

The GUI opens in your web browser (`http://localhost:8501`). It runs only on
your computer. **To stop it, close the black terminal window.**

---

## 3. The five steps of the GUI

Every step saves an ordinary text file next to your data. You can close the GUI
at any time; opening the experiment again reloads everything. The GUI reopens the
last experiment by itself, and every page starts from the `config.yaml` and
`plate_map.yaml` files already on disk, so nothing is typed twice. Moving
between steps keeps what you entered. Saving over an existing file always asks
first (the old version is kept as `.bak`), and warns more loudly if the file was
edited outside the GUI since it was opened.

**Saving keeps your file.** `config.yaml` and `plate_map.yaml` are saved *into*
the file on disk: a setting you did not change keeps its line exactly as it
was, a changed one is replaced on its own line, and every comment you typed by
hand stays. The page lists which settings a save will change; an older config
gets the newer options added with their explanations. Opening an experiment
and saving without changes leaves the files byte for byte as they were.

### Step 1 · Experiment

Choose the **experiment folder**: the folder holding the images, or one
sub-folder per acquisition (for example `zmax/` with maximum projections and
`stitched/` with z-stacks). For each folder:

- **use**: untick folders you don't want analyzed.
- **name**: how the folder is labeled in tables and figures.
- **segmentation**: `2d` segments the DAPI maximum projection (fast; works for
  any image). `3d` segments the z-stack (z-stacks only; more GPU time).

### Step 2 · Plate maps: what is in each well

The one thing the pipeline cannot work out by itself. One plate map per folder.

1. **Plate and acquisition.** The acquisition mode is read from the filenames:
   
   | Filenames look like                  | Mode              | Wells are…                                |
   | ------------------------------------ | ----------------- | ----------------------------------------- |
   | `…_Seq0042.nd2` or `Point0042_….tif` | Predefined points | numbered in the **order you imaged them** |
   | `Well07_….nd2`                       | Generate points   | the well number in the filename           |
   | `WellC10_….nd2`                      | Spinning disk     | the well label in the filename            |

2. **Predefined points: images per well** = how many colonies you imaged per
   well. By default the wells table starts empty: add one row per well in the
   order you imaged them and type its number of images in the *images* column
   (or use *Add wells … with images each*). *Same number of images in every
   well* and *Seq ranges* are the alternatives.

3. **Stain sets.** One row per channel (channel 0 is the first): type the marker
   (`SOX2`), not the fluorophore. Several panels imaged into one folder are
   several stain sets of one plate map; each well names its set in the `stain`
   column.

4. **Wells and conditions.** One row per well; one column per thing that differs
   between wells: the cell line, the BMP4 dose, the pattern diameter, … Keep
   numbers as numbers (`diameter_um: 500`).
   
   - **class** is the column the pipeline reads negative controls from (for the
     *% positive* readout). If your wells differ by cell line, you can put the
     line names in `class` or in a column of your own, e.g. `line`.

5. **The plate picture** colors wells by any column, so mistakes are easy to see.

6. **Check and save.** The GUI loads the plate map with the pipeline's own code
   and maps every image to a well. Red messages must be fixed.

### Step 3 · Analysis settings

The **Essentials** tab covers what most experiments need:

- **Channels:** which channel is the nuclear stain (DAPI, or e.g. H2B), which
  channels to measure. Each channel is labeled with every marker it carries in
  any stain set (`0: OTX2-r / GBX2`).
- **Which images:** empty = all ([§5](#5-running-on-only-some-images)).
- **What is compared:** the plate-map column(s) that define the groups:
  e.g. `class` (cell lines) or `dose`. *Compare separately within each value of*
  splits a crossed design (lines × doses) into one comparison per dose.
- **Markers:** *analyze* leaves a marker out of the whole analysis when
  unticked. It works by name, so two markers imaged on one channel in different
  wells (E-CAD in some wells, FOXA2 in others, both on channel 1) can be kept or
  dropped separately (`downstream.exclude_markers`). *intensity* is *÷ nuclear stain* (nuclear markers with a DNA
  stain such as DAPI) or *raw* (membrane or cytoplasmic markers, and every
  marker when the nuclear channel is not a DNA stain, e.g. H2B: the GUI warns).
  *readout* is *standard*, or *signaling: nuclear* / *signaling: N:C* for
  signaling markers (β-catenin, SMAD2/3, SMAD4, … recognized by name; see
  below). *One scale across* says which colonies share one intensity scale: by
  default each plate × stain panel. See [§7](#7-reading-the-results).
- **Signaling markers** (shown when there are any): the reference wells their
  readouts are divided by (usually the pluripotent wells; none = absolute,
  background-subtracted units), the nuclear readout,
  the N:C ring size, and whether to fit % positive gates (off by default).
- **Colonies and profiles:**
  - *Pattern diameter* (e.g. 500 µm), or the plate-map column holding it when
    wells carry different sizes. Used to flag torn or merged colonies.
  - *Pixel size*: leave 0 when your image files carry their calibration (ND2 and
    Fiji TIFFs do). The check in step 4 tells you.
  - *µm grid bin width* (10 µm ≈ one nucleus) and *normalized grid bins*.
  - *Leave colonies out when flagged:* which QC flags remove a colony
    ([§7](#7-reading-the-results)). Default: only colonies that were not found.
  - *Replicate:* leave on `auto` ([§7](#7-reading-the-results)).
- **Analyses to run:** everything by default. The core always runs (colonies,
  QC, thresholds, radial profiles, the report); untick the optional analyses
  your experiment does not need: *Compare conditions*, *Boundaries and peaks*,
  *Effects, dose trends and power*, *All markers together*, *Co-expression*,
  *Signaling summary* (`downstream.skip_steps`). *Figure sets and
  sub-analyses* switches off single figure sets (one figure per colony, colony
  overlays, border roughness, the bin check, …). A skipped analysis's earlier
  results move to `_skipped/` in the analysis folder, so the report never shows
  a stale figure.
- **Segmentation results used, and the analysis folder:** a table of what is on
  disk for each folder (*22 of 22 images in 3D*), with a warning when it is not
  the mode the folder is set to. *Analyze*: whatever exists, 2-D only, or 3-D
  only (`downstream.seg_mode`). The analysis folder is named after what is used:
  `analysis_2d`, `analysis_3d`, or `analysis_2d_3d` when both are analyzed side
  by side (2-D and 3-D results of the same colonies are never pooled).
- **🔍 Preview colony detection:** once segmentation exists, pick an image and
  see the colony edge, the fitted circle and the excluded cells with the current
  settings: before running the analysis. Nothing is written.

The other tabs hold every remaining option, each with its explanation (hover the
ⓘ). **💾 Save config.yaml** writes the file with every explanation kept as a
comment.

### Step 4 · Check & run

- **🔎 Run check** reads the config, plate maps and filenames (no images opened,
  nothing written) and reports images that map to no well, wells without a
  condition, missing stain sets, a missing pixel size, and markers without a
  negative control. **Run it before every HPC submission.**
- **Run downstream analysis** runs the whole analysis on your laptop once
  segmentation exists (minutes, no GPU).

### Step 5 · HPC job

Writes `hpc_job.env` into the experiment folder and shows the commands to type
on the HPC: `cd` to the experiment, then one `bsub` line per segmentation job,
each with a note saying what it does, and a table of re-run commands
([§4.3](#43-watching-and-re-running)). Fill in once: where the share is on your
laptop and on the HPC (e.g. `Z:\` ↔ `/data/Warmflash`), and the HPC Python
environment ([§4.1](#41-the-python-environment-on-the-hpc)).

---

## 4. Running on the HPC

Three job files and one helper in `hpc/`. **They are generic: you never edit them.**

| File                     | Runs                                                                         | Node |
| ------------------------ | ---------------------------------------------------------------------------- | ---- |
| `run_pipeline_2dseg.lsf` | load → segment (2-D, DAPI projection) → extract, then submits the analysis   | GPU  |
| `run_pipeline_3dseg.lsf` | load → segment (3-D, z-stack) → extract, then submits the analysis           | GPU  |
| `run_downstream.lsf`     | check → manifest → colony detection, profiles, statistics, report            | CPU  |
| `run.sh`                 | re-running parts: analysis only, redo a step, re-segment, dry run, status    | -    |

### How a job finds your experiment

You submit **from your experiment folder**; the job reads `hpc_job.env` from
there. No path is typed into a job file, so a job can't run on the wrong data
because of a stale path, and re-running an experiment months later needs only
`cd` and `bsub`.

```
exp_040/
├── config.yaml        ← what to analyze          (GUI step 3)
├── hpc_job.env        ← how to run it on the HPC  (GUI step 5)
├── zmax/
│   ├── plate_map.yaml ← what is in each well     (GUI step 2)
│   └── *.tif
└── analysis_2d/       ← results appear here (analysis_3d/ for 3-D results)
```

### 4.1 The Python environment on the HPC

Nothing to do by hand: every job checks its Python environment (`VENV_DIR` in
GUI step 5) before it starts, which takes a few seconds, and installs only what
is missing or wrong. The first job builds the environment (several minutes);
later jobs start at once.

The check (`hpc/check_env.py`) reads every package version against
`requirements.txt` and, for PyTorch, which CUDA the wheel was built for and
which GPU architectures it contains. The gpu-b6000 nodes are Blackwell GPUs
(sm_120) with CUDA 12.8: only the **cu128** build fits. An older build (cu126)
imports fine but has no sm_120 kernels; a newer one (the plain PyPI wheel is now
CUDA 13.0) needs a newer GPU driver. When PyTorch is wrong it is uninstalled and
force-reinstalled from the cu128 index; other packages are installed from
`requirements.txt` without touching PyTorch. Each segmentation job then runs a
real computation on the GPU and stops with a clear message if that fails.

If an environment is broken in a way the check does not repair, tick *Rebuild
the environment* in GUI step 5 (`FRESH_VENV="true"`): the next job deletes it and
builds it from scratch. Untick it afterwards. To build or check the
environment ahead of time on the login node:

```bash
bash /data/Warmflash/KR/micropattern_pipeline/hpc/setup_env.sh /scratch/$USER/micropattern_venv
```

> The paths here assume the lab share is `/data/Warmflash` on the HPC and the GPU
> queue is `gpu-b6000`. If yours differ, set the paths in GUI step 5 and add
> `-q <queue>` to `bsub`.

### 4.2 Every experiment

GUI step 5 prints these with your paths filled in:

```bash
cd '/data/Warmflash/KR/exp_040'                                          # the experiment folder
bsub < /data/Warmflash/KR/micropattern_pipeline/hpc/run_pipeline_2dseg.lsf  # segmentation on a GPU
```

That is all. When segmentation has **finished successfully**, the job submits
the analysis job (`run_downstream.lsf`, CPU) by itself. An experiment with a 2-D
and a 3-D folder gets one `bsub` line per job; whichever finishes last starts
the analysis, and a job that fails starts nothing (resubmit it and it will).
`THEN_ANALYSIS="false"` in `hpc_job.env` (a tick box in step 5) turns this off.

### 4.3 Watching and re-running

- `bjobs` lists your jobs; `bpeek <jobid>` shows a running job's output. Logs:
  `lsf_seg2d.<jobid>.log`, `lsf_downstream.<jobid>.log` and `pipeline.log`, all
  in the experiment folder.
- **Resubmitting is safe.** Every image and every analysis step is checkpointed.
  A step re-runs by itself when a setting it depends on, or its input, changed;
  everything else is skipped. New masks (a re-segmentation) make that image's
  intensities and colony geometry stale, so they are redone too.
- **Re-running a part.** `hpc/run.sh`, typed from the experiment folder, submits
  the same job files with a one-off setting; `hpc_job.env` is not changed:

  | When                                                  | Type (after `cd` to the experiment)  |
  | ----------------------------------------------------- | ------------------------------------ |
  | changed an analysis setting or a plate map            | `bsub < $PIPE/hpc/run_downstream.lsf` (or `bash $PIPE/hpc/run.sh analysis`) |
  | redo one analysis step anyway (and all after it)      | `bash $PIPE/hpc/run.sh redo colony`  |
  | redo the whole analysis                               | `bash $PIPE/hpc/run.sh redo-all`     |
  | changed segmentation settings, or masks look wrong    | `bash $PIPE/hpc/run.sh reseg`        |
  | try everything on a few images first                  | `SEQ="0-4" bash $PIPE/hpc/run.sh`    |
  | only the 3-D segmentation                             | `SEG=3d bash $PIPE/hpc/run.sh seg`   |
  | see what a job would do, no GPU, nothing submitted    | `bash $PIPE/hpc/run.sh test`         |
  | where are my jobs                                     | `bash $PIPE/hpc/run.sh status`       |

  `$PIPE` is the pipeline folder on the HPC (`/data/Warmflash/KR/micropattern_pipeline`);
  `bash $PIPE/hpc/run.sh help` lists everything. Under the hood, `run.sh` sets
  `MP_<NAME>` variables (e.g. `MP_SEQ`) that the job files apply over the
  `<NAME>` in `hpc_job.env`, for that job only; you can also type them yourself:
  `MP_SEQ="0-4" bsub < …/run_pipeline_2dseg.lsf`.
- The redo boxes in GUI step 5 (`FORCE` / `DOWNSTREAM_FORCE_STEPS` in
  `hpc_job.env`) do the same permanently, until unticked.
- **More time or memory:** `bsub -W 24:00 -M 96000 < …/run_pipeline_3dseg.lsf`.
  The analysis job uses 4 CPU slots for colony detection; 3-D masks of stitched
  z-stacks need ~3 GB per slot.

### 4.4 Testing without spending GPU time

```bash
cd '/data/Warmflash/KR/exp_040'
bash /data/Warmflash/KR/micropattern_pipeline/hpc/run.sh test
```

prints the exact commands every job would run, and stops (it is
`DRY_RUN=1 bash …/run_pipeline_2dseg.lsf` for each job file).

---

## 5. Running on only some images

To test settings on a few colonies first, select images by:

- **Seq / Point number** as in the filename (`Seq0042` or `Point0042` → `42`):
  `0-9, 15`
- **well**: `1-3` or `C1-C4`.

| Where                                                          | Applies to                         |
| -------------------------------------------------------------- | ---------------------------------- |
| GUI step 3 → *Which images* (`input.select` in `config.yaml`)  | everything                         |
| GUI step 5 → *Which images* (`SEQ` / `WELLS` in `hpc_job.env`) | the HPC jobs; overrides the config |
| `--seq` / `--wells` on the command line                        | that command                       |

The analysis uses the same subset. A subset holding one condition gives profiles
but no comparison (there is nothing to compare).

---

## 6. Where everything lives

```
micropattern_pipeline/             ← the code: one copy for the whole lab
├── launch_gui.bat / .command / .sh   start the GUI
├── gui/                           the GUI
├── hpc/                           generic job files, check_env.py, setup_env.sh
├── templates/                     config.yaml and hpc_job.env, fully commented
├── main.py                        the command-line entry point
├── src/                           the pipeline
├── CHANGELOG.md · CONTRIBUTING.md  what changed in each version; how to change the code
├── docs/                          CONFIG_GUIDE.md · REFERENCE.md · STATISTICAL_METHODS.md · PLATE_MAPS.md · SESSION_NOTES_2026-10.md
├── examples/                      adding your own analysis step
└── tests/                         synthetic tests, no image data needed
```

**Nothing is written into the code folder.** Settings, logs and results go into
the experiment folder; paths in `config.yaml` are relative to it, so the same
files work from the laptop and the HPC.

Each image gets a results folder next to it (`img.tif` → `img/`) holding the
nuclear masks, the per-nucleus measurements, the colony mask and two pictures to
check by eye: `segmentation_overlay.png` and `colony_overlay.png`.

**One segmentation per image folder.** That folder holds either a 2-D or a 3-D
segmentation (`segmentation_metadata.json` says which). To have both for the
same colonies, keep the MIPs and the z-stacks in two acquisition folders (e.g.
`zmax/` set to 2d and `zstacks/` set to 3d), as `hd_micropatterns_v2` does. A
3-D job on a folder segmented in 2-D replaces the 2-D masks (it says so in the
log), and the analysis always labels results by the mode they were really made
in, whatever the config says.

**The analysis folder** is `analysis_2d/`, `analysis_3d/` or `analysis_2d_3d/`
(`downstream.results_dir: auto`), named after the segmentation results it
used, so two analyses of one experiment never overwrite each other. Set
`results_dir` to a fixed name (e.g. `./analysis`) to choose it yourself.

---

## 7. Reading the results

The header of **`report.html`** and `pipeline_version.json` name the pipeline version
and commit that made the results.

Open **`report.html`** in the analysis folder (`analysis_2d/`, `analysis_3d/`, …).
Warnings come first, then:

1. **Quality control: colonies.** Every colony's overlay: orange = the colony
   edge, dashed = the fitted pattern circle, × = cells outside the colony
   (strays, neighboring colonies, escaped cells). Flagged colonies are labeled
   with the reason:
   
   | Flag        | Meaning                                                                                |
   | ----------- | -------------------------------------------------------------------------------------- |
   | `failed`    | no colony found                                                                        |
   | `clipped`   | the colony is cut by the image border (its edge distances come from the fitted circle) |
   | `size`      | diameter far from the pattern diameter: torn, shrunken or merged                       |
   | `shape`     | not round                                                                              |
   | `neighbor`  | a second, large colony in the field                                                    |
   | `outside`   | many cells outside the colony                                                          |
   | `low_count` | few nuclei                                                                             |
   | `field`     | nucleus count / DAPI / nuclear size unlike the other colonies of its condition         |
   
   Flags are always shown; only those you chose in step 3 remove a colony.

2. **Conditions compared.** Mean profile ± 95 % CI per condition, on the µm grid
   and on the normalized grid, for intensity and for % positive nuclei. The
   *global p* in each title tests whether the whole curves differ; the **gray
   bar** under the curves marks where along the radius they do. Then: pairwise
   tests (q-values), profile features (edge level, center level, peak and
   transition distance) and a colony-similarity map.

3. **Every colony, by well.** One thin line per colony, bold = well mean.

4. **Per-colony profiles.** Each colony on its own bins, in µm from the edge.

5. **All markers together** (two or more markers). Every marker of a colony
   (µm) or a well (normalized) stacked on one distance axis, each on its own
   y-range. The colored strip at the bottom shows where each marker rises (red,
   ▲) or falls (blue, ▼).

6. **Co-expression in space** (two or more markers in one stain set). One row
   per marker pair: each nucleus's A vs B values colored by its distance from
   the edge, the same split into radial bands, the correlation of A and B and
   the fraction of double-positive nuclei at each distance, and (per colony,
   when both markers have a gate) a map of where the double positives sit.
   Well pages compute everything per colony first and then average.

7. **Signaling readouts** (β-catenin, SMADs, … when a stain set has one). One
   point per colony for the whole colony, the edge band and the center, by
   condition: relative to the reference wells (dashed line at 1), or in
   background-subtracted a.u. when no reference is used
   (`signaling.reference_values: []`). The **Signaling rings (QC)** pages show
   where the N:C cytoplasm was measured: the rings should hug the nuclei and
   stay off the empty surface past the edge.

8. **Boundaries and peaks** (`figures/boundary/`). One point per colony with
   its bootstrap SE, by condition:
   
   - **x50**: where a marker is half on, in µm from the edge.
   - **width**: the 10 %→90 % distance, in µm and in nuclear spacings.
   - For two markers whose domains face each other (one high at the edge, one
     in the center): their **crossing** (the operational boundary), the
     **overlap** (negative = a gap where neither is on), and the
     **double-positive** fraction near the crossing against what independent
     markers would give.
   - For signaling markers: the **interior peak**, its height over the trough,
     and its offset from every boundary.
   - **Roughness**: how far the border wanders around the colony. The
     **registered** profiles show whether a wide transition is graded at each
     place or just a wandering sharp border.
   - **Bin check**: whether positions move when the bins change (shift in SE
     units).
   
   Colony and multimarker figures draw each accepted boundary as a grey line
   ± SE. When a facing pair has **no** line, the title says why, e.g.
   *"HOXB1 / OTX2-r: no boundary (found in 30 % of bootstrap resamples, needs
   80 %)"* or *"(no accepted HOXB1 transition)"*.

9. **Effects** (`figures/effects/`). Every per-colony number against the
   reference condition (difference with CI), its trend over dose (slope per
   decade with CI and permutation p), how its variance splits between
   colonies, wells and experiments, and how many experiments a target effect
   would need.

10. **Nuclear density** (`figures/profiles/density_um.png`, and the strip under
    each colony profile). Nuclei per 1000 µm² in each distance bin. Look here
    before reading a radial difference as signaling: the rim and center of a
    colony often differ in density too.

Things to know:

- **Intensity is on one shared scale per plate × panel**, 0 ≈ the dimmest 1 %
  and 1 ≈ the brightest 1 % of nuclei in that scale. A line expressing half as
  much reads half as high. This assumes the colonies on one scale were stained
  and imaged alike.
- **n is colonies (or wells), never nuclei.** With fewer than 3 wells per
  condition, colonies are the replicate and the report says what that means:
  with one well per condition, a difference shows the colonies differ, not that
  it would hold in another well. With ≥ 3 wells per condition, wells become the
  replicate automatically.
- **Small designs have a floor on p.** Five colonies per group cannot give p
  below ≈ 0.008; the smallest possible p is listed with every test.
- **% positive** needs a negative control (or two clearly separated populations).
  Without one a marker gets no % positive profile, and the report says so.
- **Signaling markers are not on the shared scale.** Their axes say what 1 means:
  `nuclear b-CAT (÷ pluripotent)` is the background-subtracted nuclear level
  divided by that of the pluripotent colonies; `SMAD2/3 N:C (÷ pluripotent)` the
  same for the nuclear:cytoplasmic ratio. With `reference_values: []` the axis
  is the background-subtracted level in a.u., which is the right choice when the
  reference colonies are few, sparse or badly segmented: a ratio to them
  inherits their noise. Compare a.u. only within one stain panel. The
  background is measured on the bare surface right around each colony
  (`tables/signaling_background.csv` flags any unusual image). Signaling
  markers never get a % positive profile.
  In maximum projections, membrane or cytoplasmic signal above and below a
  nucleus lands on it, which lowers N:C and raises "nuclear" levels of
  membrane-rich proteins; 3-D segmentation measures each nucleus in its own
  planes.
- **Every profile figure says how it was normalized** in its title (e.g.
  *"GBX2: raw nuclear median, P1–P99 → 0–1 per plate"*). Add your own line
  (cell line, day) with `visualization.caption`. If a marker is divided by a
  histone label (H2B), the run warns: that ratio mixes chromatin compaction into
  the marker.
- **Saturated nuclei are dropped.** A nucleus whose median sits at the detector
  ceiling in any channel is clipped debris; it is removed before any scale, gate
  or profile and listed in `tables/qc_saturated_nuclei.csv`. For a 12-bit
  camera writing 16-bit files set `qc.saturation_ceiling: 4095`.
- **Independent experiments.** Name the plate-map column that identifies the
  differentiation in `stats.experiment_column`; with ≥ 2 per condition the
  experiment becomes n everywhere.

The tables behind every figure are in `<analysis folder>/tables/`; per-nucleus
data with each nucleus's distance from the edge are in `cells.parquet` there. The
statistics are explained in [`docs/STATISTICAL_METHODS.md`](docs/STATISTICAL_METHODS.md),
every file in [`docs/REFERENCE.md`](docs/REFERENCE.md).

---

## 8. Troubleshooting

| Problem                                                                    | What to do                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *"Python 3.10 or newer is needed"*                                         | Install Python from python.org (Windows: tick "Add to PATH"), then start the launcher again.                                                                                                                                                         |
| The browser page does not open                                             | Open <http://localhost:8501> yourself (8502 if another GUI is running).                                                                                                                                                                              |
| *"N images map to no well"*                                                | Check *images per well* in step 2; the *Image → well* table shows what went where.                                                                                                                                                                   |
| *"no pixel size … set downstream.colony.pixel_size_um"*                    | Your image files carry no calibration. Enter the pixel size (µm/px) in step 3.                                                                                                                                                                       |
| The colony edge swallows a neighboring colony                              | In the preview, raise `open_radius_um` (Colonies & QC tab), e.g. 20.                                                                                                                                                                                 |
| The edge is ragged / misses the rim                                        | Raise `close_radius_um` (e.g. 20-25).                                                                                                                                                                                                                |
| Many colonies flagged `size`                                               | Check *Pattern diameter* in step 3: it must be the diameter, not the radius.                                                                                                                                                                         |
| A condition shows no comparison                                            | It needs at least two conditions with ≥ 2 colonies each; check *What is compared*.                                                                                                                                                                   |
| No % positive profile for a marker                                         | No negative control and no clear two-population split. Mark control wells (`class`), or rely on the intensity profile.                                                                                                                               |
| A colony or multimarker figure has no boundary line for a marker pair      | Read its title: it states the reason. Usually one marker has no coherent domain in that colony (a negative control, or a dose too low to switch it on): a genuine result, not a failure.                                                             |
| One stain failed in one panel (e.g. a bad antibody lot)                    | Leave that channel out of the panel's stain set in `plate_map.yaml` (`panel3: {0: GBX2, 2: H2B, 3: OTX2-g}`), keep a comment saying why, and re-run downstream. The manifest is rebuilt and that marker is never measured or pooled for those wells. |
| `signaling_background.csv` flags an image                                  | Open the image: debris, a bubble or a focus problem near the colony. The values are still used; drop the colony with QC if the image is bad.                                                                                                         |
| Job log: *"no hpc_job.env in …"*                                           | Submit from the experiment folder, and save step 5 in the GUI.                                                                                                                                                                                       |
| Log: *"segmented in 2d but the config says 3d"*                            | The folder's masks are from the other mode. Run that mode's segmentation job for the folder (`SEG=3d bash $PIPE/hpc/run.sh reseg`), or set the folder back to the mode its results are in.                                                         |
| Two analysis jobs started for one experiment                               | You submitted `run_downstream.lsf` by hand while the segmentation job also starts it. Harmless (the second finds everything up to date); leave the manual line out, or untick *Start the analysis by itself*.                                     |
| Job log: *"NEEDS FIXING: torch …"*                                         | Nothing to do: the job reinstalls PyTorch from the cu128 index itself ([§4.1](#41-the-python-environment-on-the-hpc)).                                                                                                                               |
| Job log: *"still not right after reinstalling"* or *"the GPU test failed"* | Tick *Rebuild the environment* in GUI step 5, save, resubmit ([§4.1](#41-the-python-environment-on-the-hpc)).                                                                                                                                        |

`pipeline.log` in the experiment folder has the full history including error
tracebacks: send it along when asking for help.

---

## 9. Command line

**Running on your own computer.** Two scripts do what the HPC jobs do, without
LSF. Both take the experiment folder (the folder holding `config.yaml`) and
use the `cellpose_sam` conda environment unless you set `CONDA_ENV`:

```bash
# segmentation, 2-D (GPU if PyTorch sees one); test on a few images first
SEQ="0-4" DATASETS="zmax" bash run_local_2dseg.sh /path/to/exp_040
DATASETS="zmax" bash run_local_2dseg.sh /path/to/exp_040

# downstream analysis (CPU): manifest + every analysis step
DATASETS="zmax" bash run_local_downstream.sh /path/to/exp_040
```

Each script lists its options (`SEQ`, `WELLS`, `DATASETS`, `FORCE`, ...) in its
header. `DATASETS` takes the dataset names from `config.yaml`; leave out a
dataset that has not been segmented. GUI step 4 offers the same choice.

The GUI only writes files; the pipeline is a command-line tool:

```bash
PIPE=/data/Warmflash/KR/micropattern_pipeline
EXP=/data/Warmflash/KR/exp_040

python $PIPE/main.py check      --config $EXP/config.yaml
python $PIPE/main.py process    --config $EXP/config.yaml --seq 0-9       # GPU: load + segment + extract
python $PIPE/main.py manifest   --config $EXP/config.yaml
python $PIPE/main.py downstream --config $EXP/config.yaml                 # CPU: everything after
python $PIPE/main.py downstream --config $EXP/config.yaml --force-step profiles
python $PIPE/main.py downstream --config $EXP/config.yaml --seg-mode 3d   # 3-D results only → analysis_3d
python $PIPE/main.py downstream --list-steps
```

**Re-running after a change.** Checkpoints hash the *settings*, not the code.
A changed setting or plate map re-runs what depends on it by itself. After a
change in the code, name the step with `FORCE_STEPS` (local) or
`DOWNSTREAM_FORCE_STEPS` (HPC); every step after it re-runs automatically:

| Changed                                  | Force                                       |
| ---------------------------------------- | ------------------------------------------- |
| signaling measurement code               | `signaling` (re-measures every image: slow) |
| QC code                                  | `qc`                                        |
| boundary / roughness code or their plots | `boundary`                                  |
| profile, density or caption code         | `profiles`                                  |
| effects code                             | `effects`                                   |

`signaling` always runs but only re-measures images whose settings changed, so
do not force it unless the measurement itself changed. `OUT=<folder>` writes
the analysis somewhere other than `results_dir` (e.g. a dated copy).

**The same experiment in another folder.** Copy `config.yaml` next to the new
data, fix `datasets:` (folder names), check the channels and the pixel size,
and write the new folder's `plate_map.yaml` (wells, stains, conditions; keep
marker spellings identical). For an independent differentiation to be pooled
with an earlier one, list both folders in one config, give their wells a
`differentiation` column and set `stats.experiment_column: differentiation`.

---

## 10. For maintainers

**Version control.** The code is a git repository; `main` is what the lab runs.
Changes are made on a branch in a separate clone, tested, written into
[`CHANGELOG.md`](CHANGELOG.md), merged, and released with a version number in
`VERSION` and a `vX.Y.Z` tag. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the
workflow, the release rules, and how to rerun an old result with the code that
made it. `python main.py --version` prints the version and commit in use; every
results folder records it in `pipeline_version.json` and in the report header.

**Tests** (synthetic data only; ~5 min on the share):

```bash
python -m pytest -q -p no:cacheprovider tests/
```

| File                                      | Checks against a known answer                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `test_micropattern.py`                    | colony geometry, shared scale, permutation tests, end to end                                         |
| `test_downstream.py`                      | thresholds, controls, scopes, QC, palette rule                                                       |
| `test_experiment.py`, `test_selection.py` | plate maps, Seq/Point → well, stains; `--seq` / `--wells`                                            |
| `test_gui_core.py`                        | everything the GUI writes reads back exactly                                                         |
| `test_signaling.py`                       | signaling names, N:C on a known image, local background under vignetting, background outlier table   |
| `test_boundary.py`                        | logistic x50 / width, crossing, overlap, peak, block bootstrap, bin check, why a boundary is missing |
| `test_roughness.py`                       | border roughness, decentering, angular registration                                                  |
| `test_effects.py`                         | variance components, effects, dose trends, power; a synthetic multi-experiment run                   |
| `test_density_captions.py`                | bin areas, density gradients, normalization captions, histone guard, saturated nuclei                |
| `test_pipeline_options.py`                | markers left out by name, skipped analyses, 2-D/3-D choice and folder name, stale checkpoints, comment-keeping writers, HPC one-off settings |

Every change to `src/` must be generic (no experiment names in the code) and
come with a synthetic test; experiment-specific choices live in that
experiment's `config.yaml` and `plate_map.yaml`.

**What this pipeline is built from.**

- From `../analysis_pipeline_v2`: the GUI, launchers, generic HPC jobs,
  multi-dataset config, image subsets, `check`, plate maps, the downstream step
  framework (config-aware checkpoints), positivity gates, QC and the report.
- From the BMP4 micropattern pipeline in
  `../20260618_ipsc465-522_ESI_upatterns_bmp48_200k/96_well_multistain`: colony
  detection, edge-referenced distances, per-colony binned profiles, the
  colony-as-replicate design and the functional permutation statistics. They
  are rewritten here to be vectorized and to handle border-cut colonies, escaped
  cells and well-level replication.
- New: per-colony profiles on a µm grid shared by all colonies alongside the
  normalized grid, one shared intensity scale, colony QC flags, the colony
  preview, exact enumeration for small designs, and stale-input detection
  between steps.
- Added 2026-10 (answering a set of qualifier critiques, validated on
  `hd_micropatterns_bcat`): boundary positions and widths with per-colony SE,
  facing-pair crossings and double-positive tests, border roughness and
  angular registration, the experiment > well > colony hierarchy with variance
  components, dose trends and power, local signaling background, nuclear
  density per bin, normalization captions, the histone guard, the
  saturated-nucleus filter and the bin check. The record of that work is
  [`docs/SESSION_NOTES_2026-10.md`](docs/SESSION_NOTES_2026-10.md).

On the validation experiment the colony geometry matches the earlier micropattern
pipeline: edge distances within a median of 0.05 µm, and the same in-colony
call for ≥ 99.5 % of nuclei.

**Cluster defaults** (GPU queue, memory, wall time): the `#BSUB` lines in
`hpc/*.lsf`. Modules and the CUDA build: `templates/hpc_job.env` (`CUDA_MODULE`,
`TORCH_INDEX`); the environment check and repair: `hpc/check_env.py` and
`ensure_env` in `hpc/common.sh`.

**How the GUI stays in step with the pipeline.** Plate maps are validated by the
pipeline's own `PlateMap`; the colony preview calls `downstream.colony`; the
final word is `main.py check`; and `config.yaml` is written by filling
`templates/config.yaml`, so a new option added there appears in the GUI with its
comment as help text.
