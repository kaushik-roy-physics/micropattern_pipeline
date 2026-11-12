# Plate maps: which image is which well, and what is in each well

> The GUI (step 2) writes these files for you and validates them with the code
> described here. This page is the reference for reading or writing one by hand.
> For micropatterns, one image is one colony: `points_per_well` is the number of
> colonies imaged per well.

The image-processing steps are deliberately **condition-agnostic**: they
never need to know which well or treatment an image came from. A separate,
fully decoupled layer (`main.py manifest`) maps each image to its **plate well**
and the **experimental condition** that well represents, and writes a single
`manifest.parquet` / `manifest.csv` you join to the per-image results downstream.

## 1. Author a `plate_map.yaml` per acquisition folder

**Start by generating one**, rather than writing it from scratch:

```bash
python main.py init-plate-map                    # one per dataset folder
python main.py init-plate-map --data-dir ./zmax_stain1 --plate-id exp033
python main.py init-plate-map --mode spinning_disk --force
```

This scans the folder and writes a heavily commented template pre-filled with
what it found: the acquisition mode detected from the filenames, the wells (or
Seq indices) actually present, the images-per-well count, and the channel names
from `image_metadata.json` when `process --steps load` has already run. Lines you
must edit are marked `<-- EDIT`. An existing plate map is never overwritten
without `--force`: it is hand-authored knowledge.

Then edit it. The schema in full:

```yaml
plate: 18well            # 8well | 18well | 96well  (or geometry: {rows, cols})
plate_id: exp033         # optional; defaults to the folder name (see the note below)
mode: predefined         # predefined | generate | spinning_disk

# Predefined/Seq only: map the 0-based global point index to a well:
points_per_well: 5       # constant N, wells filled in order from well 1
# seq_ranges:            #   …or explicit ranges for variable counts (inclusive)
#   1: [0, 4]
#   3: [10, 14]

stains:                  # channel index → marker. 'default' unless a well sets 'stain'.
  default: {0: DAPI, 1: GM130, 2: pS6, 3: LC3}
  setB:    {0: DAPI, 1: Tom20, 2: pS6, 3: LC3}

wells:                   # well → condition; range keys like "2-4" are expanded
  1:   {treatment: Control, dose_uM: 0, day: 1, class: control}
  2-4: {treatment: DrugX,   dose_uM: 10, day: 1, stain: setB}
```

Wells use the **NIKON Elements** row-major numbering you already see in JOBS:
8-well `01-04 / 05-08`, 18-well `01-06 / 07-12 / 13-18`, 96-well `01-12 / …`.

* **`generate` mode**: `Well07_*.nd2` carries the well number directly. Using
  only some wells needs nothing special: you just have no files for the rest.

* **`predefined` mode**: `*_Seq0042.nd2` (or a Fiji-stitched `Point0042_….tif`)
  carries the global point index, which
  is *acquisition order*. The filename says nothing about **where on the plate**
  the point came from, so wells here are numbered **sequentially**: well 1 is the
  first well you imaged, well 2 the second. Using 13 wells of an 18-well plate,
  in any order, is still wells 1-13: you never have to look up physical well
  numbers. Say how the points divide up with one of:
  
  * bare `points_per_well: N`; equal counts per well, in acquisition order. The
    usual case.
  * `points_per_well: {1: 5, 2: 8}` or a parallel list; variable counts.
  * `seq_ranges:`: explicit `well: [start, end]`; overrides everything.
  
  Because there is no plate position, **`row`/`col` are not written to the
  manifest** and the plate-layout heatmaps skip themselves (the step logs why).
  If your predefined keys really *are* plate wells and you want the layout and
  edge-effect figures, opt in with `well_numbering: plate` and key `wells:` (and
  `imaged_wells:`/`seq_ranges:`) by real well numbers.

* **`spinning_disk` mode**: the Spinning-Disk confocal's 'GeneratePoints' writes
  an **alphanumeric** well label, `WellC10_*.nd2` → row C, column 10. Key the
  `wells:` block by label instead of by number, and same-row column ranges work
  the same way:
  
  ```yaml
  mode: spinning_disk
  plate: 96well
  wells:
    C10:   {treatment: Control, class: control}
    C1-C4: {treatment: DrugX, dose_uM: 10}
  ```
  
  `WellE06` and `WellE6` are the same well. Labels resolve to canonical
  row-major well numbers (`C10` on a 96-well plate → 34), so everything
  downstream sees the same numeric `well` regardless of mode. Cross-row ranges
  (`B10-C2`) are rejected rather than guessed at.

* **Per-well stains**: wells default to the `default` stain set; tag the
  exceptions with `stain: <set>`. Each manifest row then carries its own
  `ch{n}_marker` columns, so a `groupby` never mixes markers.
* **A channel left out of a stain set is not analyzed** for the wells using that
  set. This is how to exclude a failed stain from one panel only
  (`panel3: {0: GBX2, 2: H2B, 3: OTX2-g}`, leaving out channel 1); see
  [CONFIG_GUIDE.md §9](CONFIG_GUIDE.md#9-a-stain-that-failed).

> **Filenames carry no marker meaning.** The only things read out of a filename
> are the well number (`Well07`), the point index (`Seq0042`) or the well label
> (`WellC10`) that the acquisition software writes. Which markers an image
> carries is whatever `plate_map.yaml` says it is: never guessed from the name.

> **`plate_id` matters when panels live in sibling folders.** It defaults to the
> folder name, so `zmax_stain1/` and `zmax_stain2/` get *different* plate ids and
> are treated as different batches. If they are two antibody panels of the same
> physical plate, give both maps the same explicit `plate_id:`: that is what
> makes cross-panel gate pooling possible (see
> [REFERENCE.md → one scale across panels](REFERENCE.md#intensity-scale-and-scope)).

### Reserved well keys

Four keys in a `wells:` entry are structural rather than experimental
conditions. Everything else you write becomes a free-form column you can group,
facet and plot by.

| key            | meaning                                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `stain`        | selects a named entry from `stains:` (`stain_id` is accepted as an alias, for older plate maps)                                                                                       |
| `class`        | marks the well as the negative control (values configured in `config.yaml`)                                                |
| `positive_for` | markers this well is the **positive** control for: it stays the negative control for every other marker                   |
| `control_for`  | markers this well **is** the negative control for (`all` accepted); an explicit name beats a blanket declaration elsewhere |

The last two exist for the case where one condition is a negative control for
*some* of the panel and a positive control for the rest: see
[REFERENCE.md → % positive and controls](REFERENCE.md#positive-profiles-and-negative-controls).

## 1b. Several stain sets in one folder

Normally each stain set gets its own folder, and each folder is a `datasets:`
entry in `config.yaml`: that is the simplest arrangement and needs nothing
extra. If instead you acquired several stain sets into a **single** folder, the
per-well `stain:` key can't express it (the same well appears once per stain),
so declare the Seq blocks:

```yaml
stains:
  stain1: {0: DAPI, 1: ISL1, 2: NKX2.5}
  stain2: {0: DAPI, 1: HCN4, 2: MEF2C}

stain_sweeps:                       # non-overlapping, inclusive, 0-based
  stain1: {seq_range: [0, 107],   points_per_well: 6}
  stain2: {seq_range: [108, 215], points_per_well: 6}
```

Each block is that stain's own sweep of the plate, so **its Seq indices are
local to the block**: with the above, `Seq0108` is stain2's *first* point and
maps to well 1, not to well 19. Each block takes the same mapping options as the
plate as a whole (`points_per_well`, `imaged_wells`, `seq_ranges`), and
overlapping blocks are rejected outright.

Stain resolution order is `stain_sweeps` → the well's `stain:` key → `default`.

## 1c. Check before you run

```bash
python main.py check
```

Reads the config, the plate maps and the file listing (no images, no GPU, no
writes) and reports per dataset: how many images map to a well, how many wells
have conditions, which stain set each image gets with its channel→marker map,
and any Seq blocks. It fails loudly on images that map to no well, wells with
images but no condition entry, and stain sets with no channel map; it warns on
wells declared but never imaged. Run it first: these are the mistakes that
otherwise surface after an hour of segmentation.

It also answers the two questions that decide whether your thresholds will mean
anything:

* **does every marker have a negative control?** It lists the control wells, any
  `control_for` / `positive_for` redirection, and errors when a marker has been
  demoted on every candidate control well and has nowhere left to be gated from.
* **is any marker poolable across panels?** When a marker appears in two panels
  of one plate on the *same* channel, it names it as a candidate for
  `scope: plate`. On a different channel it says so too: there, separate gates
  are almost certainly correct.

## 2. Build the manifest

```bash
python main.py manifest                                   # auto-discover all plate maps
python main.py manifest --plate-map data/exp1/plate_map.yaml   # a single plate
python main.py manifest --data-dir /path/to/data --out run42_manifest
```

This writes `manifest.parquet` + `manifest.csv` at each dataset root, one row per
image: `dataset, plate_id, image_path, result_dir, well, seq_index,
<your condition columns…>, stain_set, ch0_marker …`, plus `row, col` when the
well number is a real plate coordinate (see `well_numbering` above).

## 3. Join it to the per-nucleus results downstream

The `downstream` analysis does this join for you (`analysis/cells.parquet`,
with every nucleus's colony and edge distance attached). The snippet shows the
idea for ad-hoc work in a notebook:

```python
import joblib, pandas as pd
from pathlib import Path

root = Path("data")
m = pd.read_parquet(root / "manifest.parquet")

frames = []
for _, r in m.iterrows():
    res = joblib.load(root / r.result_dir / "per_nuclei_intensities.joblib")
    df = pd.DataFrame({"dapi": res["dapi_per_cell"], "ch2": res["raw_intensities"][2]})
    df = df.assign(well=r.well, treatment=r.treatment, ch2_marker=r.ch2_marker)
    frames.append(df)

cells = pd.concat(frames, ignore_index=True)
cells.groupby(["treatment", "ch2_marker"])["ch2"].median()   # per-condition summary
```

`manifest` and `process` never depend on each other: you can build the manifest
before, after, or without ever running segmentation.

---
