"""
The analysis context handed to every step.

Its job is to make each step **independently runnable**. A step never assumes the
step before it ran in the same process: it asks the context for what it needs,
and the context either returns the cached in-memory object or reloads it from the
table the earlier step wrote. That is what lets you re-render one figure set
without re-assembling 300 images::

    python main.py downstream --steps figures              # reads tables from disk
    python main.py downstream --force-step thresholds      # redo one step

Two views of the cell table
---------------------------
:attr:`Context.all_data` is every nucleus ``assemble`` wrote, including those
outside the main colony: QC needs them (how many strays, how many in a
neighboring colony). :attr:`Context.data` is what every analysis uses: nuclei
**inside** the main colony of a colony that QC did not exclude. Because the
thresholds, the controls and the shared intensity scale are all built from
``data``, a stray cell or a torn colony can never move a gate or an anchor.

Derived objects that are cheap to recompute (the gated long table) are rebuilt
on demand rather than stored, so there is no intermediate file to keep in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import qc as qc_mod, scope as scope_mod, thresholds as thr_mod
from .aggregate import condition_columns, condition_label
from .controls import Controls
from .data import Data

logger = logging.getLogger(__name__)

TABLES = "tables"
FIGURES = "figures"


class Context:
    """Paths, config, and lazily-materialized analysis products."""

    def __init__(self, cfg: dict, datasets, out_dir: Path, force: bool = False):
        self.cfg = cfg                     # the 'downstream' config block
        self.datasets = datasets
        self.out = Path(out_dir)
        self.force = force
        self.skipped: set[str] = set()     # optional steps left out (downstream.skip_steps)
        self._cache: dict = {}

    # -- paths -------------------------------------------------------------

    def table(self, name: str) -> Path:
        path = self.out / TABLES / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def figure(self, group: str, name: str) -> Path:
        path = self.out / FIGURES / group / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def reset_figures(self, group: str) -> None:
        """Delete a figure group's PNGs before the step rewrites them.

        Without this, renaming a figure (say, when a marker gains a second panel
        and its files acquire a scope suffix) silently leaves the old file behind
        and the contact sheet shows both: the stale one indistinguishable from
        the current. Each figure step owns exactly one group directory.
        """
        directory = self.out / FIGURES / group
        if not directory.exists():
            return
        for png in directory.glob("*.png"):
            try:
                png.unlink()
            except OSError:
                logger.warning("could not remove stale figure %s", png.name)

    def write_table(self, df: pd.DataFrame, name: str) -> Path:
        path = self.table(name)
        df.to_csv(path, index=False)
        logger.info("table → %s (%d rows)", name, len(df))
        return path

    def read_table(self, name: str) -> pd.DataFrame:
        path = self.table(name)
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found: run the step that produces it first "
                f"(see 'python main.py downstream --list-steps')."
            )
        return pd.read_csv(path)

    def has_table(self, name: str) -> bool:
        return self.table(name).exists()

    # -- lazily materialized products --------------------------------------

    def _cached(self, key: str, builder):
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def set(self, key: str, value):
        self._cache[key] = value
        return value

    @property
    def all_data(self) -> Data:
        """Every assembled nucleus, in or out of the main colony."""
        return self._cached("all_data", lambda: Data.load(self.out))

    @property
    def data(self) -> Data:
        """In-colony nuclei of the colonies QC kept: the analysis population."""
        def build():
            full = self.all_data
            keep = full.cells["in_colony"].to_numpy(bool) if "in_colony" in full.cells \
                else np.ones(len(full.cells), bool)
            if self.has_table("qc_colonies.csv"):
                qc = self.read_table("qc_colonies.csv")
                dropped = qc.loc[qc["excluded"].astype(bool), "field_id"]
                keep &= ~full.cells["field_id"].isin(set(dropped)).to_numpy()
            keep &= ~self.saturated_mask(full)
            return full.subset(keep)
        return self._cached("data", build)

    def saturated_mask(self, full: Data | None = None) -> np.ndarray:
        """Nuclei clipped at the detector ceiling (``qc.drop_saturated_nuclei``, default
        on): left out of every analysis, listed by the qc step."""
        full = full if full is not None else self.all_data
        if not bool((self.cfg.get("qc") or {}).get("drop_saturated_nuclei", True)):
            return np.zeros(len(full.cells), bool)

        def build():
            from .assemble import CHANNEL_QC_NAME
            path = self.out / CHANNEL_QC_NAME
            channel_qc = pd.read_csv(path) if path.exists() else None
            hits = qc_mod.saturated_nuclei(full, channel_qc, self.cfg)
            mask = np.zeros(len(full.cells), bool)
            mask[hits["cell_uid"].to_numpy(np.int64)] = True
            if mask.any():
                logger.info("qc: %d nucle%s at the detector ceiling left out (bright debris).",
                            int(mask.sum()), "us" if mask.sum() == 1 else "i")
            self._cache["saturated_hits"] = hits
            return mask
        return self._cached("saturated_mask", build)

    @property
    def colonies(self) -> pd.DataFrame:
        """One row per colony with geometry and (after ``qc``) its flags."""
        def build():
            if self.has_table("qc_colonies.csv"):
                return self.read_table("qc_colonies.csv")
            from .assemble import COLONIES_NAME
            path = self.out / COLONIES_NAME
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run the 'assemble' step first.")
            return pd.read_csv(path)
        return self._cached("colonies", build)

    @property
    def condition_keys(self) -> list[str]:
        return condition_columns(self.data.cells, self.cfg.get("conditions") or {})

    @property
    def facet_keys(self) -> list[str]:
        """Condition columns that split a figure into separate files.

        A crossed design, three treatments × four progenitor origins, puts every
        combination on one crowded axis, which is the wrong picture when the
        question is "how does origin behave *within* a treatment". Naming
        ``treatment`` here gives one figure per treatment with origin on the x-axis.

        Purely presentational: the facet columns must already be part of
        ``condition_key``, so each condition still belongs to exactly one facet and
        no statistic changes: only which panel it is drawn in.
        """
        cond = self.cfg.get("conditions") or {}
        keys = cond.get("facet_key") or []
        keys = [keys] if isinstance(keys, str) else list(keys)
        if not keys:
            return []
        declared = self.condition_keys
        unknown = [k for k in keys if k not in declared]
        if unknown:
            logger.warning(
                "conditions.facet_key names %s, which is not in condition_key=%s. "
                "Faceting on a column that does not define the condition would split "
                "a condition across panels: ignoring it.", unknown, declared)
        keys = [k for k in keys if k in declared]
        if keys and len(keys) == len(declared):
            logger.warning(
                "conditions.facet_key %s uses every condition_key column, leaving "
                "nothing on the x-axis: ignoring it.", keys)
            return []
        return keys

    @staticmethod
    def facet_suffix(facet: dict) -> str:
        """Filename fragment for a facet: ``{'treatment': 'DrugX'}`` → ``__DrugX``."""
        return "__" + "_".join(str(v) for v in facet.values()) if facet else ""

    @staticmethod
    def facet_title(facet: dict) -> str:
        """Human label naming what is held fixed, e.g. ``treatment = DrugX``."""
        return ", ".join(f"{k} = {v}" for k, v in facet.items())

    def facets(self, *frames: pd.DataFrame):
        """Yield ``(facet, subsets, relabel)``: one facet per value combination.

        *facet* is a ``{column: value}`` dict (empty when nothing is faceted);
        :meth:`facet_suffix` and :meth:`facet_title` turn it into a filename and a
        panel title.

        Each subset's ``condition`` column is rebuilt from the *remaining* keys, so
        an axis inside the Activin panel reads ``DEPLM`` rather than
        ``Activin | DEPLM``. ``relabel`` maps the full labels to the short ones, so
        callers can translate their control-condition sets to match.

        With no ``facet_key`` configured this yields exactly one pass-through
        facet, which keeps the call sites free of branching.
        """
        keys = self.facet_keys
        base = next((f for f in frames
                     if all(k in f.columns for k in keys)), None) if keys else None
        if base is None or base.empty:
            yield {}, list(frames), {}
            return

        rest = [k for k in self.condition_keys if k not in keys]
        values = base[keys].astype(str).drop_duplicates().sort_values(keys)
        for _, key_row in values.iterrows():
            vals = [str(key_row[k]) for k in keys]
            relabel: dict[str, str] = {}
            subsets = []
            for frame in frames:
                if not all(k in frame.columns for k in keys):
                    subsets.append(frame)
                    continue
                mask = np.ones(len(frame), dtype=bool)
                for key, val in zip(keys, vals):
                    mask &= frame[key].astype(str).to_numpy() == val
                sub = frame.loc[mask].copy()
                if rest and all(k in sub.columns for k in rest) and "condition" in sub:
                    short = condition_label(sub, rest).astype(str)
                    relabel.update(dict(zip(sub["condition"].astype(str), short)))
                    sub["condition"] = short
                subsets.append(sub)
            yield dict(zip(keys, vals)), subsets, relabel

    @property
    def controls(self) -> Controls:
        """Resolver answering "is this cell *this marker's* negative control?"."""
        return self._cached("controls", lambda: Controls(self.data.cells, self.cfg))

    @property
    def is_control(self) -> np.ndarray:
        """Marker-agnostic control mask: for whole-plate figures and palettes."""
        return self._cached("is_control", lambda: self.controls.any_mask())

    def control_mask_for(self, marker: str) -> np.ndarray:
        """Control mask for one marker (the SOX2 case: different well from the rest)."""
        return self.controls.mask_for(marker)

    @property
    def control_conditions(self) -> set[str]:
        """Conditions that are a control for *something*: figure-level shading."""
        def build():
            return self._conditions_for_mask(self.is_control)
        return self._cached("control_conditions", build)

    def control_conditions_for(self, marker: str) -> set[str]:
        """Conditions that are *this marker's* negative control.

        Per-marker figures use this so a pluripotent well is grayed out as the
        control on the BRA panel and drawn as an ordinary condition on the SOX2
        one, where it is the positive control.
        """
        key = f"control_conditions:{marker}"
        if key not in self._cache:
            self._cache[key] = self._conditions_for_mask(self.control_mask_for(marker))
        return self._cache[key]

    def _conditions_for_mask(self, mask: np.ndarray) -> set[str]:
        if not mask.any():
            return set()
        labels = condition_label(self.data.cells.loc[mask], self.condition_keys)
        return set(labels.astype(str).unique())

    @property
    def thresholds(self) -> pd.DataFrame:
        return self._cached("thresholds", lambda: self.read_table("thresholds.csv"))

    @property
    def gated(self) -> pd.DataFrame:
        """Long intensity table with value / threshold / positive / condition."""
        def build():
            data = self.data
            long = thr_mod.apply_thresholds(data, self.thresholds, self.cfg)
            uid = long["cell_uid"].to_numpy()
            keys = self.condition_keys
            for k in keys:
                long[k] = data.cell_column(k, uid)
            long["condition"] = condition_label(long, keys)
            long["field_id"] = data.cell_column("field_id", uid)
            return long
        return self._cached("gated", build)

    # -- convenience -------------------------------------------------------

    @property
    def markers(self) -> list[str]:
        explicit = self.cfg.get("markers_order")
        markers = self.data.markers
        if explicit:
            ordered = [m for m in explicit if m in markers]
            return ordered + [m for m in markers if m not in ordered]
        return sorted(markers)

    # -- scopes ------------------------------------------------------------
    #
    # A marker's gate is estimated over one scope, and figures must not pool
    # across scope boundaries. Everything user-facing therefore iterates over
    # (marker × scope), where the scope is exactly the key its threshold was
    # estimated for, which is per marker, since a marker may be declared
    # ``scope: plate`` and pool the panels its neighbors keep separate.

    @property
    def scope_columns(self) -> list[str]:
        """Union of every scope column in play: the table/figure faceting axes."""
        cells = self.data.cells
        return scope_mod.union_scope_columns(self.markers, self.cfg, cells.columns)

    def scope_columns_for(self, marker: str) -> list[str]:
        """Scope columns for one marker (a subset of :attr:`scope_columns`)."""
        return scope_mod.scope_columns_for(marker, self.cfg, self.data.cells.columns)

    def marker_scopes(self) -> list[tuple[str, dict, str]]:
        """``(marker, scope, suffix)`` for every measured marker × scope.

        The scope dict holds only that marker's own columns, so a plate-scoped
        marker yields ONE entry naming ``plate_id``/``seg_mode`` and
        :meth:`scoped` keeps both of its panels, which is the whole point of
        declaring the wider scope.

        ``suffix`` is empty when the marker exists in a single scope, and carries
        the scope name when it does not, so filenames stay clean in the common
        case and stay unambiguous in the multi-panel case.
        """
        table = self.thresholds
        out: list[tuple[str, dict, str]] = []
        for marker, block in table.groupby("marker", observed=True):
            marker = str(marker)
            cols = [c for c in self.scope_columns_for(marker) if c in block.columns]
            keys = (block[cols].drop_duplicates() if cols
                    else pd.DataFrame(index=[0]))
            multi = len(keys) > 1
            for _, row in keys.iterrows():
                scope = {c: row[c] for c in cols}
                # Scope columns often repeat the same word (dataset 'zmax' inside
                # plate 'zmax'); de-duplicate so the suffix stays readable.
                parts: list[str] = []
                for col in cols:
                    value = str(row[col])
                    if value != scope_mod.POOLED and value not in parts:
                        parts.append(value)
                suffix = "_" + "-".join(parts) if multi and parts else ""
                out.append((marker, scope, suffix))
        order = {m: i for i, m in enumerate(self.markers)}
        out.sort(key=lambda t: (order.get(t[0], 999), t[2]))
        return out

    def scoped(self, frame: pd.DataFrame, scope: dict) -> pd.DataFrame:
        """Restrict a frame to one scope (columns it lacks are ignored).

        A ``*`` value means the marker was pooled over that column, so it is not a
        filter: skipping it is what lets a plate-scoped marker's figure draw both
        of its panels.
        """
        for key, value in (scope or {}).items():
            if key in frame.columns and str(value) != scope_mod.POOLED:
                frame = frame[frame[key].astype(str) == str(value)]
        return frame

    def threshold_for(self, marker: str, scope: dict | None = None) -> float | None:
        """The gate for a marker within a scope (None when it has no accepted gate)."""
        sub = self.thresholds[(self.thresholds["marker"] == marker)
                              & (self.thresholds["verdict"] == thr_mod.OK)]
        sub = self.scoped(sub, scope or {})
        if sub.empty:
            return None
        values = sub["threshold"].astype(float)
        return float(values.iloc[0]) if values.nunique() == 1 else float(values.median())

    def space_for(self, marker: str) -> str:
        return self.data.space_for(marker, self.cfg)


__all__ = ["Context", "TABLES", "FIGURES"]
