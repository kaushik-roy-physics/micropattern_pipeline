"""
The downstream data model: two tables, and the helpers that join them cheaply.

``assemble`` writes a star schema rather than one wide table:

``cells.parquet``: one row per nucleus::

    cell_uid  field_id  dataset  plate_id  well  row  col  seq_index
    stain_set  seg_mode  dapi  area  cy  cx  cz  <condition columns…>

``intensities.parquet``: one row per (nucleus × marker)::

    cell_uid  marker  channel  raw  dapi_norm  nuc_bgsub  nc_ratio

``nuc_bgsub`` and ``nc_ratio`` are filled only for signaling markers (see
:mod:`downstream.signaling`): the nuclear core minus the image background, and
the nuclear:cytoplasmic ratio. Every other row holds NaN there.

``channel`` is the acquisition channel index the marker was measured on. It is
not an analysis axis; it is the evidence behind the "these two panels could share
a gate" advisory in :mod:`downstream.scope`: same channel index means the same
filter and detector settings, a different one means a different fluorophore.

Why long-in-marker: pooling one marker over every panel it appears in (and
faceting by any plate-map column) becomes a ``groupby``, with no code that knows
marker names. Analyses that need per-cell co-expression pivot back to wide with
:meth:`Data.wide`, which is cheap because a panel has a handful of markers.

**The join is a take, not a merge.** ``cells`` is written in ``cell_uid`` order,
so ``cell_uid`` *is* the row position: attaching a cell column to the long table
is ``cells[col].to_numpy()[intens.cell_uid]``: a vectorized gather instead of a
hash join. :meth:`Data.attach` relies on that invariant and
:meth:`Data.validate` checks it after loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CELLS_NAME = "cells.parquet"
INTENSITIES_NAME = "intensities.parquet"

RAW = "raw"
DAPI_NORM = "dapi_norm"
NUC_BGSUB = "nuc_bgsub"
NC_RATIO = "nc_ratio"
SPACES = (RAW, DAPI_NORM)                       # what markers.<m>.space may declare
ALL_SPACES = (RAW, DAPI_NORM, NUC_BGSUB, NC_RATIO)
CHANNEL = "channel"

# Columns of ``cells`` that describe identity/geometry rather than an experimental
# condition. Everything else came from the plate map and is a usable analysis axis.
# ``control_for``/``positive_for`` are declarations *about* the experiment design,
# not axes of it: leaving them out here keeps them from being faceted on or
# auto-detected as two-level experimental factors.
ID_COLUMNS = {
    "cell_uid", "field_id", "dataset", "plate_id", "image", "result_dir", "mode",
    "well", "row", "col", "seq_index", "stain_set", "seg_mode",
    "dapi", "area", "cy", "cx", "cz",
    "control_for", "positive_for",
    "in_colony", "dist_edge_um", "r_center_um", "r_norm",
}


@dataclass
class Data:
    """The assembled per-cell tables plus the config-derived semantics."""

    cells: pd.DataFrame
    intensities: pd.DataFrame

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, out_dir: Path) -> "Data":
        out_dir = Path(out_dir)
        cells_path = out_dir / CELLS_NAME
        intens_path = out_dir / INTENSITIES_NAME
        if not cells_path.exists():
            raise FileNotFoundError(
                f"{cells_path} not found: run the 'assemble' step first "
                f"(python main.py downstream --steps assemble)."
            )
        data = cls(pd.read_parquet(cells_path), pd.read_parquet(intens_path))
        data.validate()
        return data

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.cells.to_parquet(out_dir / CELLS_NAME, index=False)
        self.intensities.to_parquet(out_dir / INTENSITIES_NAME, index=False)
        logger.info("Wrote %s (%d cells) and %s (%d rows)",
                    CELLS_NAME, len(self.cells), INTENSITIES_NAME, len(self.intensities))

    def validate(self) -> None:
        """Check the positional-``cell_uid`` invariant :meth:`attach` depends on."""
        uid = self.cells["cell_uid"].to_numpy()
        if len(uid) and not np.array_equal(uid, np.arange(len(uid), dtype=uid.dtype)):
            raise ValueError(
                "cells.parquet is not in cell_uid order: re-run 'assemble --force-step'."
            )

    def subset(self, keep: np.ndarray) -> "Data":
        """Keep the cells where *keep* is True, re-indexing ``cell_uid`` positionally."""
        keep = np.asarray(keep, dtype=bool)
        if keep.all():
            return self
        cells = self.cells[keep].reset_index(drop=True)
        new_uid = np.full(len(self.cells), -1, dtype=np.int64)
        new_uid[keep] = np.arange(int(keep.sum()), dtype=np.int64)
        old = self.intensities["cell_uid"].to_numpy()
        intens = self.intensities[keep[old]].reset_index(drop=True)
        intens["cell_uid"] = new_uid[intens["cell_uid"].to_numpy()]
        cells["cell_uid"] = np.arange(len(cells), dtype=np.int64)
        return Data(cells=cells, intensities=intens)

    # -- accessors ---------------------------------------------------------

    @property
    def condition_columns(self) -> list[str]:
        """Plate-map columns available as grouping / faceting axes."""
        return [c for c in self.cells.columns if c not in ID_COLUMNS]

    @property
    def markers(self) -> list[str]:
        cats = self.intensities["marker"]
        vals = cats.cat.categories if isinstance(cats.dtype, pd.CategoricalDtype) else cats.unique()
        return list(vals)

    def attach(self, columns: list[str], intensities: pd.DataFrame | None = None) -> pd.DataFrame:
        """Long intensity table with *columns* from ``cells`` gathered onto it.

        Vectorized positional take (see the module docstring): no hash join.
        """
        src = self.intensities if intensities is None else intensities
        idx = src["cell_uid"].to_numpy()
        out = src.copy()
        for col in columns:
            if col in out.columns:
                continue
            out[col] = self.cells[col].to_numpy()[idx]
        return out

    def cell_column(self, column: str, cell_uid: np.ndarray) -> np.ndarray:
        """Gather one ``cells`` column for an arbitrary array of cell_uids."""
        return self.cells[column].to_numpy()[np.asarray(cell_uid)]

    def wide(self, markers: list[str] | None = None, space: str = DAPI_NORM,
             subset: pd.Series | None = None) -> pd.DataFrame:
        """Per-cell wide table: one column per marker, indexed by ``cell_uid``.

        Only cells measured for *every* requested marker are kept, so the columns
        stay mutually aligned per cell, which is what co-expression, quadrant and
        scatter analyses require.
        """
        df = self.intensities if subset is None else self.intensities[subset]
        if markers:
            df = df[df["marker"].isin(markers)]
        wide = df.pivot_table(index="cell_uid", columns="marker", values=space,
                              observed=True, aggfunc="first")
        if markers:
            present = [m for m in markers if m in wide.columns]
            wide = wide[present].dropna(how="any")
        return wide

    def space_for(self, marker: str, marker_cfg: dict) -> str:
        """Intensity space a marker is analyzed in (config-declared, not hard-coded).

        A signaling marker is analyzed in its readout's space (nuclear,
        background-subtracted, or N:C); if that was never measured (the
        ``signaling`` step has not run) it falls back to raw, with a warning.
        """
        from .signaling.catalog import settings as sig_settings
        from .signaling.readouts import readout_for
        readout = readout_for(marker, marker_cfg)
        if readout is not None:
            space = readout.space(sig_settings(marker_cfg))
            if self._measured(marker, space):
                return space
            warned = self.__dict__.setdefault("_space_warned", set())
            if marker not in warned:
                warned.add(marker)
                logger.warning("%s is a signaling marker (%s) but has no '%s' values: run "
                               "the 'signaling' and 'assemble' steps. Using raw intensity.",
                               marker, readout.name, space)
            return RAW
        default = marker_cfg.get("default_space", DAPI_NORM)
        spec = (marker_cfg.get("markers") or {}).get(marker) or {}
        space = str(spec.get("space", default))
        if space not in SPACES:
            raise ValueError(f"marker {marker!r}: space must be one of {SPACES}, got {space!r}.")
        # Fall back to raw when DAPI normalization was never computed.
        if space == DAPI_NORM and not self.intensities[DAPI_NORM].notna().any():
            return RAW
        return space

    def _measured(self, marker: str, space: str) -> bool:
        cache = self.__dict__.setdefault("_measured_cache", {})
        key = (marker, space)
        if key not in cache:
            if space not in self.intensities.columns:
                cache[key] = False
            else:
                rows = self.intensities["marker"].astype(str) == str(marker)
                cache[key] = bool(self.intensities.loc[rows, space].notna().any())
        return cache[key]

    def values(self, frame: pd.DataFrame, cfg: dict, marker_col: str = "marker") -> np.ndarray:
        """Each row's value in its own marker's analysis space (:meth:`space_for`)."""
        spaces = frame[marker_col].astype(str).map(
            {str(m): self.space_for(str(m), cfg) for m in frame[marker_col].astype(str).unique()}
        ).to_numpy()
        return select_values(frame, spaces)


def select_values(frame: pd.DataFrame, spaces) -> np.ndarray:
    """Per-row value from the column named in *spaces* (raw where the space is unknown)."""
    spaces = np.asarray(spaces, dtype=object)
    out = frame[RAW].to_numpy(dtype=np.float64).copy()
    for space in pd.unique(spaces):
        if space in (RAW, None) or not isinstance(space, str) or space not in frame.columns:
            continue
        rows = spaces == space
        out[rows] = frame[space].to_numpy(dtype=np.float64)[rows]
    return out


__all__ = ["Data", "CELLS_NAME", "INTENSITIES_NAME", "RAW", "DAPI_NORM", "NUC_BGSUB",
           "NC_RATIO", "SPACES", "ALL_SPACES", "CHANNEL", "ID_COLUMNS", "select_values"]
