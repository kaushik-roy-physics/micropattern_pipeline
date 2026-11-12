"""
Run the pipeline on a subset of the images in a dataset folder.

The default is every image. A subset is declared by **Seq index** (the
``Seq0042`` token the microscope writes, i.e. acquisition order) and/or by
**well**, either in the config::

    defaults:                       # or inside one `datasets:` entry
      input:
        select:
          seq: "0-9, 15, 20-24"     # Seq indices, as written in the filename
          wells: "1-3"              # well numbers, or labels like "C1-C4, D5"

or on the command line, where it overrides the config for every selected
dataset::

    python main.py process    --seq 0-9 15
    python main.py downstream --wells 1-3

When both ``seq`` and ``wells`` are given an image must match both.

Seq indices are matched against the number in the filename (``Seq0042`` → 42),
so they are the same numbers you see in the file browser. Wells are matched
against the well the image *resolves to*: read from the filename in
``generate``/``spinning_disk`` mode, and through the plate map's Seq → well
mapping in ``predefined`` mode (which therefore needs a ``plate_map.yaml``).

The same selection is applied by ``process`` (which images are loaded,
segmented and extracted), by ``check`` (which images are validated) and by
``downstream`` (which manifest rows are assembled), so a test run on five
fields gives an analysis of exactly those five fields.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_ALL = {"", "all", "*", "none", "null"}
_LABEL_RE = re.compile(r"^([A-Za-z])0*(\d+)$")


def _tokens(spec) -> list[str]:
    """Split ``"0-9, 15"`` / ``[0, "3-5"]`` / ``7`` into stripped string tokens."""
    if spec is None:
        return []
    items = spec if isinstance(spec, (list, tuple, set)) else [spec]
    out: list[str] = []
    for item in items:
        for tok in re.split(r"[,\s;]+", str(item).strip()):
            if tok:
                out.append(tok)
    return out


def is_all(spec) -> bool:
    """True when *spec* means "no restriction"."""
    toks = _tokens(spec)
    return not toks or (len(toks) == 1 and toks[0].lower() in _ALL)


def parse_index_spec(spec) -> set[int] | None:
    """``"0-9, 15"`` → ``{0, …, 9, 15}``. ``None``/``""``/``"all"`` → ``None`` (everything).

    Raises ``ValueError`` on anything that is not an integer or an ``a-b`` range,
    so a typo fails loudly instead of silently selecting nothing.
    """
    if is_all(spec):
        return None
    out: set[int] = set()
    for tok in _tokens(spec):
        if "-" in tok.strip("-"):
            lo, hi = tok.split("-", 1)
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError:
                raise ValueError(f"selection: {tok!r} is not a range like '3-7'.") from None
            if hi_i < lo_i:
                raise ValueError(f"selection: range {tok!r} ends before it starts.")
            out.update(range(lo_i, hi_i + 1))
        else:
            try:
                out.add(int(tok))
            except ValueError:
                raise ValueError(f"selection: {tok!r} is not a number.") from None
    return out


def parse_well_spec(spec) -> tuple[set[int], set[str]] | None:
    """Well selection → ``(numbers, labels)``.

    Numbers and ranges (``"1-3, 7"``) select by well number; alphanumeric labels
    and same-row label ranges (``"C1-C4, D5"``) select Spinning-Disk wells by the
    label in the filename. Labels are normalized (``C04`` → ``C4``).
    """
    if is_all(spec):
        return None
    numbers: set[int] = set()
    labels: set[str] = set()
    for tok in _tokens(spec):
        if tok[0].isalpha():
            if "-" in tok:
                lo, hi = tok.split("-", 1)
                m_lo, m_hi = _LABEL_RE.match(lo), _LABEL_RE.match(hi)
                if not (m_lo and m_hi):
                    raise ValueError(f"selection: {tok!r} is not a label range like 'C1-C4'.")
                if m_lo.group(1).upper() != m_hi.group(1).upper():
                    raise ValueError(f"selection: label range {tok!r} must stay in one row.")
                row = m_lo.group(1).upper()
                for c in range(int(m_lo.group(2)), int(m_hi.group(2)) + 1):
                    labels.add(f"{row}{c}")
            else:
                m = _LABEL_RE.match(tok)
                if not m:
                    raise ValueError(f"selection: {tok!r} is not a well label like 'C10'.")
                labels.add(f"{m.group(1).upper()}{int(m.group(2))}")
        else:
            numbers.update(parse_index_spec(tok) or set())
    return numbers, labels


class Selection:
    """A parsed ``seq``/``wells`` selection. Empty means "every image"."""

    def __init__(self, seq=None, wells=None):
        self.seq_spec, self.wells_spec = seq, wells
        self.seq = parse_index_spec(seq)
        self.wells = parse_well_spec(wells)

    @classmethod
    def from_config(cls, ds_cfg: dict, seq=None, wells=None) -> "Selection":
        """Config ``input.select`` with command-line values taking precedence."""
        sel = (ds_cfg.get("input") or {}).get("select") or {}
        return cls(seq if seq is not None else sel.get("seq"),
                   wells if wells is not None else sel.get("wells"))

    @property
    def active(self) -> bool:
        return self.seq is not None or self.wells is not None

    def as_dict(self) -> dict:
        """Canonical form, used for checkpoint hashing and logging."""
        out = {}
        if self.seq is not None:
            out["seq"] = sorted(self.seq)
        if self.wells is not None:
            out["wells"] = sorted(self.wells[0]) + sorted(self.wells[1])
        return out

    def describe(self) -> str:
        parts = []
        if self.seq is not None:
            parts.append(f"seq={self.seq_spec}")
        if self.wells is not None:
            parts.append(f"wells={self.wells_spec}")
        return ", ".join(parts) or "all images"

    # -- matching ------------------------------------------------------------

    def matches(self, seq_index: int | None, well: int | None,
                well_label: str | None = None) -> bool:
        """Does one image (described by its Seq index / well / label) pass?"""
        if self.seq is not None and (seq_index is None or int(seq_index) not in self.seq):
            return False
        if self.wells is not None:
            numbers, labels = self.wells
            hit = ((well is not None and int(well) in numbers)
                   or (well_label is not None and well_label in labels))
            if not hit:
                return False
        return True

    def filter_paths(self, paths, directory: Path | None = None) -> list[Path]:
        """Filter image paths found on disk.

        Wells are resolved from the dataset's ``plate_map.yaml`` when one exists
        (needed for ``predefined`` mode, where the filename carries no well);
        otherwise from the filename, with the mode detected from the names.
        """
        paths = list(paths)
        if not self.active:
            return paths
        from experiment import parser

        resolve_well = _well_resolver(paths, directory, need_wells=self.wells is not None)
        kept = []
        for p in paths:
            seq = parser.parse_seq(p.name)
            label = parser.parse_well_label(p.name)
            if self.matches(seq, resolve_well(p, seq), label):
                kept.append(p)
        logger.info("selection (%s): %d of %d image(s)", self.describe(), len(kept), len(paths))
        if not kept:
            logger.warning("selection (%s) matched no images: check the Seq numbers "
                           "against the filenames.", self.describe())
        return kept

    def filter_manifest(self, man):
        """Filter manifest rows (which already carry ``seq_index`` and ``well``)."""
        if not self.active or man.empty:
            return man
        from experiment import parser

        def ok(row) -> bool:
            seq = row.get("seq_index")
            seq = None if seq is None or seq != seq else int(seq)       # NaN → None
            well = row.get("well")
            well = None if well is None or well != well else int(well)
            label = parser.parse_well_label(Path(str(row.get("image_path", ""))).name)
            return self.matches(seq, well, label)

        mask = man.apply(ok, axis=1)
        return man[mask.astype(bool)]


def _well_resolver(paths, directory: Path | None, need_wells: bool):
    """Return ``f(path, seq) -> well number | None`` for this folder."""
    from experiment import parser, plate

    if not need_wells:
        return lambda p, seq: None

    pm = None
    if directory is not None:
        pm_path = Path(directory) / "plate_map.yaml"
        if pm_path.exists():
            from experiment.plate_map import PlateMap
            pm = PlateMap(pm_path)

    if pm is not None:
        if pm.is_spinning_disk:
            def f(p, seq):
                lab = parser.parse_well_label(p.name)
                return plate.label_to_well(lab, pm.cols) if lab else None
            return f
        if pm.mode == "generate":
            return lambda p, seq: parser.parse_well(p.name)
        return lambda p, seq: pm.well_for_seq(seq)

    # No plate map: fall back on the filename alone.
    from experiment.template import detect_mode
    mode = detect_mode(paths)
    if mode == "predefined":
        raise ValueError(
            "selection by well in 'predefined' (Seq) mode needs a plate_map.yaml to "
            "know which Seq indices belong to which well: select by --seq instead, "
            "or write the plate map first.")
    if mode == "generate":
        return lambda p, seq: parser.parse_well(p.name)
    return lambda p, seq: None          # spinning disk: labels are matched directly


__all__ = ["Selection", "parse_index_spec", "parse_well_spec", "is_all"]
