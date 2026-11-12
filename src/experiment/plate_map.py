"""
Load, validate, and expand a ``plate_map.yaml`` describing one acquisition.

Schema (all keys but ``plate`` + ``wells`` are optional)::

    plate: 18well            # 8well | 18well | 96well   (or geometry: {rows, cols})
    plate_id: exp42_day1     # optional label; defaults to the folder name
    mode: predefined         # predefined | generate | spinning_disk

    # --- predefined/Seq only: how the global point index maps to a well ---
    # The Seq index is *acquisition order*, not the well number, so for a plate
    # where you skip wells declare the wells you imaged (in the order JOBS
    # visited them): ranges are derived for you:
    points_per_well: 5            # constant count …
    imaged_wells: [1, 5, 9]       # … over exactly these wells, in this order
    # points_per_well: {1: 5, 5: 8, 9: 3}   # or per-well counts (variable)
    #
    # If every well 1..K is imaged contiguously you may omit imaged_wells and
    # give just `points_per_well: 5`. For full control, give explicit ranges
    # (inclusive, 0-based), which override everything else:
    # seq_ranges:
    #   1: [0, 4]
    #   5: [5, 12]

    # --- stain sets: channel index → marker name ---
    stains:
      default: {0: DAPI, 1: GM130, 2: pS6, 3: LC3}   # used when a well has no 'stain'
      setB:    {0: DAPI, 1: Tom20, 2: pS6, 3: LC3}

    # --- optional: several stain sets acquired into ONE folder ---
    # Only needed when you do not separate stains into their own directories.
    # Each block is that stain's own sweep of the plate, so its Seq indices are
    # local to the block and it carries its own well mapping.
    stain_sweeps:
      default: {seq_range: [0, 59],   points_per_well: 6}
      setB:    {seq_range: [60, 119], points_per_well: 6}

    # --- well → condition (range keys like "2-4" are expanded) ---
    wells:
      1:   {treatment: Control, dose_uM: 0, day: 1, class: control}
      2-4: {treatment: DrugX,   dose_uM: 10, day: 1, stain: setB}

Acquisition modes and how wells are keyed
-----------------------------------------
``generate`` and ``predefined`` key ``wells:`` by **number** (``1``, ``"2-4"``).
``spinning_disk``: 'GeneratePoints' on the Spinning-Disk confocal, whose
filenames carry an alphanumeric label like ``WellC10_…nd2``: keys by **label**
instead, and same-row column ranges work the same way::

    mode: spinning_disk
    plate: 96well
    wells:
      C10:     {treatment: Control, class: control}
      C1-C4:   {treatment: DrugX, dose_uM: 10}

Labels resolve to canonical, 1-based, row-major well numbers using the plate's
column count, so everything downstream sees the same numeric ``well`` regardless
of mode. Cross-row ranges (``B10-C2``) are rejected rather than guessed at.

What a well number *means*: ``well_numbering``
----------------------------------------------
``generate`` and ``spinning_disk`` read the well out of the filename, so the
number is a physical plate coordinate: ``row``/``col`` are emitted and the
plate-layout and edge-effect figures describe where the cells actually sat.

``predefined`` filenames carry acquisition order only (``Seq0000``, ``Seq0001``)
and record nothing about plate position. Its wells are therefore numbered
**sequentially** by default (well 1 is the first well imaged, well 2 the second),
so imaging 13 wells of an 18-well plate, in whatever order, is simply wells
1-13. No ``row``/``col`` is emitted, because any value would be invented, and the
layout figures skip themselves rather than draw a geometry nobody imaged.

Set ``well_numbering: plate`` in a predefined map to declare that the keys really
are plate positions (pair it with ``imaged_wells``/``seq_ranges`` naming the real
wells) and the geometry comes back.

Reserved well keys
------------------
Four keys are structural rather than experimental conditions:

``stain``
    Selects a named entry from ``stains``.
``control_for``
    Markers this well is the **negative control** for. ``all`` means every marker
    in its panel. An explicit marker name beats a blanket ``all`` (or a control
    ``class``) declared on other wells, so you annotate only the exception.
``positive_for``
    Markers this well is the **positive control** for. It stays the negative
    control for everything else: this is the one key that fixes the pluripotent
    /SOX2 case::

        wells:
          1-6: {treatment: mTeSR, class: pluripotent, positive_for: [SOX2]}
          7-9: {treatment: BMP50, control_for: [SOX2]}

Every other key becomes a free-form condition column in the manifest. See
:mod:`downstream.controls` for how the three declarations are resolved.

**Nothing here is inferred from filenames.** Beyond the well number or Seq index
that the acquisition software writes (``Well07``, ``Seq0042``), the pipeline
reads no meaning out of a filename: which markers an image carries is whatever
this file says it is. Stain resolution order is ``stain_sweeps`` → the well's
``stain:`` key → ``default``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from . import parser, plate

logger = logging.getLogger(__name__)

# Well-condition keys that are structural rather than experimental conditions.
STAIN_KEY = "stain"
# Older plate maps (the original 96-well pipeline) named the per-well stain set
# 'stain_id'; it is read as 'stain'.
STAIN_KEY_ALIASES = ("stain_id",)
CONTROL_FOR_KEY = "control_for"
POSITIVE_FOR_KEY = "positive_for"
RESERVED_WELL_KEYS = {STAIN_KEY, CONTROL_FOR_KEY, POSITIVE_FOR_KEY}
DEFAULT_STAIN_SET = "default"

# What a well key means: a physical plate coordinate, or acquisition order.
PLATE = "plate"
SEQUENTIAL = "sequential"
_NUMBERING = {PLATE, SEQUENTIAL}


def _canonical_well(cond, source: str, well: int) -> dict:
    """A well's condition dict with alias keys renamed to their canonical names."""
    out = dict(cond or {})
    for alias in STAIN_KEY_ALIASES:
        if alias not in out:
            continue
        value = out.pop(alias)
        if STAIN_KEY in out and str(out[STAIN_KEY]) != str(value):
            raise ValueError(f"{source}: well {well} sets both '{STAIN_KEY}: {out[STAIN_KEY]}' "
                             f"and '{alias}: {value}'. Keep one ('{STAIN_KEY}').")
        out[STAIN_KEY] = value
    return out


def _normalize_marker_list(value) -> str:
    """Normalize ``control_for``/``positive_for`` to a ``;``-joined string.

    Accepts ``SOX2``, ``[SOX2, BRA]`` or ``all``. Stored as a string so the
    manifest stays a flat table that round-trips through parquet and CSV.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "all" if value else ""
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return ";".join(out)


def _expand_well_key(key) -> list[int]:
    """Expand a well key into explicit well numbers: ``2`` → [2]; ``"2-4"`` → [2,3,4]."""
    if isinstance(key, int):
        return [key]
    s = str(key).strip()
    if "-" in s:
        lo, hi = s.split("-", 1)
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise ValueError(f"Invalid well range '{s}' (end < start).")
        return list(range(lo, hi + 1))
    return [int(s)]


def _check_sweep_overlap(sweeps: dict, label: str) -> None:
    """Refuse overlapping Seq blocks: an image must belong to exactly one stain."""
    blocks = sorted(((spec["seq_range"], name) for name, spec in sweeps.items()))
    for (a_range, a_name), (b_range, b_name) in zip(blocks, blocks[1:]):
        if b_range[0] <= a_range[1]:
            raise ValueError(
                f"{label}: stain_sweeps {a_name!r} {list(a_range)} and {b_name!r} "
                f"{list(b_range)} overlap: each Seq index must belong to one stain."
            )


def build_seq_map(points_per_well, imaged_wells, seq_ranges, label: str = "") -> dict[int, int] | None:
    """Build a ``{seq_index: well}`` map from one set of mapping parameters.

    Resolution priority, most explicit first:

    1. ``seq_ranges``    : explicit ``well: [start, end]`` (inclusive, 0-based);
    2. ``imaged_wells``  : wells in acquisition order + per-well counts;
    3. a bare integer ``points_per_well``: handled by the caller as a divisor,
       so ``None`` is returned here.

    Used both for the plate as a whole and, in multi-panel folders, per panel:
    each panel is its own Seq sweep.
    """
    if seq_ranges:
        out: dict[int, int] = {}
        for well_key, rng in seq_ranges.items():
            well = int(well_key)
            start, end = int(rng[0]), int(rng[1])
            for s in range(start, end + 1):
                out[s] = well
        return out

    ppw = points_per_well
    if isinstance(ppw, dict):
        counts = {int(w): int(n) for w, n in ppw.items()}
        order = [int(w) for w in imaged_wells] if imaged_wells else sorted(counts)
    elif imaged_wells is not None:
        order = [int(w) for w in imaged_wells]
        if isinstance(ppw, (list, tuple)):
            if len(ppw) != len(order):
                raise ValueError(
                    f"{label}: points_per_well list length ({len(ppw)}) != "
                    f"imaged_wells length ({len(order)})."
                )
            counts = {w: int(n) for w, n in zip(order, ppw)}
        elif isinstance(ppw, int):
            counts = {w: ppw for w in order}
        else:
            raise ValueError(
                f"{label}: 'imaged_wells' needs 'points_per_well' "
                f"(an int, a per-well dict, or a matching list)."
            )
    else:
        return None

    seq_to_well: dict[int, int] = {}
    offset = 0
    for well in order:
        for s in range(offset, offset + counts[well]):
            seq_to_well[s] = well
        offset += counts[well]
    return seq_to_well


class PlateMap:
    """A validated, queryable view of one ``plate_map.yaml``."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        with open(self.path) as f:
            raw = yaml.safe_load(f) or {}

        self.plate_id: str = str(raw.get("plate_id") or self.path.parent.name)
        self.mode: str = str(raw.get("mode", "generate")).lower()
        self.is_spinning_disk = self.mode in parser.SPINNING_DISK_MODES
        if self.mode not in {"generate", "predefined"} and not self.is_spinning_disk:
            raise ValueError(
                f"{self.path}: mode must be 'generate', 'predefined', or "
                f"'spinning_disk', got {self.mode!r}."
            )

        geom = raw.get("geometry") or {}
        self.rows, self.cols = plate.resolve_geometry(
            raw.get("plate"), geom.get("rows"), geom.get("cols")
        )
        self.n_wells = plate.n_wells(self.rows, self.cols)

        # Does a well key denote a PHYSICAL position on the plate, or merely the
        # order in which wells were visited?
        #
        # 'generate' and 'spinning_disk' read the well out of the filename, so the
        # key is a real plate coordinate and (row, col) describes where the cells
        # actually sat. 'predefined' filenames carry acquisition order only
        # (Seq0000, Seq0001), and nothing on disk records which physical well a
        # point came from. Deriving a row and column there would place wells in
        # cells of the plate-layout figure they were never in, and would make
        # edge-effect QC read a geometry that does not exist. So predefined maps
        # default to *sequential* keys: well 1 is simply the first well imaged,
        # and no (row, col) is emitted.
        #
        # Set `well_numbering: plate` to opt back in when the keys really are
        # plate positions and the layout figures are wanted.
        default_numbering = SEQUENTIAL if self.mode == "predefined" else PLATE
        self.well_numbering = str(raw.get("well_numbering") or default_numbering).lower()
        if self.well_numbering not in _NUMBERING:
            raise ValueError(
                f"{self.path}: well_numbering must be one of {sorted(_NUMBERING)}, "
                f"got {self.well_numbering!r}."
            )
        self.has_plate_positions = self.well_numbering == PLATE

        # Stain sets: {set_name: {channel_index: marker_name}}.
        self.stains: dict[str, dict[int, str]] = {
            str(name): {int(ch): str(marker) for ch, marker in (mapping or {}).items()}
            for name, mapping in (raw.get("stains") or {}).items()
        }

        # Well → condition dict (range keys expanded; later keys win, with a warning).
        # Spinning-Disk plate maps key wells by alphanumeric label ('C10'); every
        # other mode keys by well number. Both resolve to canonical well numbers.
        self.wells: dict[int, dict] = {}
        for key, cond in (raw.get("wells") or {}).items():
            expanded = (plate.expand_label_key(key, self.cols)
                        if self.is_spinning_disk else _expand_well_key(key))
            for w in expanded:
                if w in self.wells:
                    logger.warning("%s: well %d defined more than once.", self.path.name, w)
                self.wells[w] = _canonical_well(cond, self.path.name, w)
        plate.validate_wells(self.wells.keys(), self.rows, self.cols)

        # Sequence → well mapping (predefined mode only). Resolution priority:
        #   1. explicit seq_ranges        (most explicit; handles anything)
        #   2. imaged_wells + counts      (derive ranges from acquisition order)
        #   3. bare integer points_per_well (assumes wells 1..K, contiguous)
        self.points_per_well = raw.get("points_per_well")
        self.imaged_wells = (
            [int(w) for w in raw["imaged_wells"]] if raw.get("imaged_wells") else None
        )
        self._seq_to_well = build_seq_map(
            self.points_per_well, self.imaged_wells, raw.get("seq_ranges"), self.path.name
        )
        if self._seq_to_well is not None:
            plate.validate_wells(set(self._seq_to_well.values()), self.rows, self.cols)

        # Optional: several stain sets acquired into ONE folder, declared as
        # non-overlapping Seq blocks. Each block is its own sweep of the plate,
        # so it carries its own well mapping and its Seq indices are local to it.
        self.sweeps: dict[str, dict] = {}
        self._sweep_maps: dict[str, dict[int, int] | None] = {}
        for name, spec in (raw.get("stain_sweeps") or {}).items():
            spec = dict(spec or {})
            name = str(name)
            if name not in self.stains:
                raise ValueError(
                    f"{self.path.name}: stain_sweeps declares {name!r}, which is not "
                    f"one of the 'stains' sets ({sorted(self.stains)})."
                )
            rng = spec.get("seq_range")
            if not rng or len(rng) != 2:
                raise ValueError(
                    f"{self.path.name}: stain_sweeps.{name} needs "
                    f"'seq_range: [first, last]' (inclusive, 0-based)."
                )
            spec["seq_range"] = (int(rng[0]), int(rng[1]))
            self.sweeps[name] = spec
            mapping = build_seq_map(spec.get("points_per_well"), spec.get("imaged_wells"),
                                    spec.get("seq_ranges"), f"{self.path.name}:{name}")
            if mapping is not None:
                plate.validate_wells(set(mapping.values()), self.rows, self.cols)
            self._sweep_maps[name] = mapping
        _check_sweep_overlap(self.sweeps, self.path.name)

        if (self.mode == "predefined" and self._seq_to_well is None
                and not isinstance(self.points_per_well, int) and not self.sweeps):
            logger.warning(
                "%s: predefined mode but no usable Seq→well mapping "
                "('seq_ranges', 'imaged_wells', or integer 'points_per_well'): "
                "sequence indices cannot be mapped to wells.",
                self.path.name,
            )

    # -- queries -----------------------------------------------------------

    def sweep_for_seq(self, seq_index: int | None) -> str | None:
        """Stain set whose declared Seq block contains this point index."""
        if seq_index is None or not self.sweeps:
            return None
        seq_index = int(seq_index)
        for name, spec in self.sweeps.items():
            first, last = spec["seq_range"]
            if first <= seq_index <= last:
                return name
        return None

    def well_for_seq(self, seq_index: int | None) -> int | None:
        """Map a 0-based point index to a well number, or ``None``.

        When the index falls inside a declared stain sweep, that sweep's own
        mapping applies and the index is taken relative to the sweep's start:
        each sweep re-visits the plate from its first well.
        """
        if seq_index is None:
            return None
        seq_index = int(seq_index)

        sweep = self.sweep_for_seq(seq_index)
        if sweep is not None:
            spec = self.sweeps[sweep]
            local = seq_index - spec["seq_range"][0]
            mapping = self._sweep_maps.get(sweep)
            if mapping is not None:
                return mapping.get(local)
            ppw = spec.get("points_per_well")
            if isinstance(ppw, int) and ppw > 0:
                return local // ppw + 1
            return None

        if self._seq_to_well is not None:
            return self._seq_to_well.get(seq_index)
        if isinstance(self.points_per_well, int) and self.points_per_well > 0:
            return seq_index // self.points_per_well + 1
        return None

    def resolve_stain(self, well: int | None,
                      seq_index: int | None = None) -> tuple[str, dict[int, str]]:
        """Stain set name + channel→marker map for one image.

        Resolution order, all of it declared by you in ``plate_map.yaml``:
        the Seq block it falls in (``stain_sweeps``) → the well's ``stain:`` key
        → ``default``.
        """
        name = self.sweep_for_seq(seq_index) or self.stain_set_name(well)
        return name, self.stains.get(name, self.stains.get(DEFAULT_STAIN_SET, {}))

    def conditions_for_well(self, well: int | None) -> dict:
        """Free-form condition columns for a well (excludes the reserved keys)."""
        cond = self.wells.get(well, {})
        return {k: v for k, v in cond.items() if k not in RESERVED_WELL_KEYS}

    def control_for_well(self, well: int | None) -> str:
        """Markers this well is the negative control for, as a ``;``-joined string."""
        return _normalize_marker_list(self.wells.get(well, {}).get(CONTROL_FOR_KEY))

    def positive_for_well(self, well: int | None) -> str:
        """Markers this well is the positive control for, as a ``;``-joined string."""
        return _normalize_marker_list(self.wells.get(well, {}).get(POSITIVE_FOR_KEY))

    @property
    def declares_marker_controls(self) -> bool:
        """True when any well carries ``control_for``/``positive_for``."""
        return any(k in cond for cond in self.wells.values() for k in
                   (CONTROL_FOR_KEY, POSITIVE_FOR_KEY))

    def declared_control_markers(self) -> set[str]:
        """Every marker name mentioned in a ``control_for``/``positive_for`` key."""
        out: set[str] = set()
        for cond in self.wells.values():
            for key in (CONTROL_FOR_KEY, POSITIVE_FOR_KEY):
                tokens = _normalize_marker_list(cond.get(key))
                out.update(t for t in tokens.split(";") if t and t.lower() != "all")
        return out

    def stain_set_name(self, well: int | None) -> str:
        return str(self.wells.get(well, {}).get(STAIN_KEY, DEFAULT_STAIN_SET))

    def stains_for_well(self, well: int | None) -> dict[int, str]:
        """Channel-index → marker mapping in effect for this well."""
        name = self.stain_set_name(well)
        if name in self.stains:
            return self.stains[name]
        return self.stains.get(DEFAULT_STAIN_SET, {})


__all__ = ["PlateMap", "STAIN_KEY", "STAIN_KEY_ALIASES", "CONTROL_FOR_KEY", "POSITIVE_FOR_KEY",
           "RESERVED_WELL_KEYS", "DEFAULT_STAIN_SET"]
