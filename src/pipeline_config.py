"""
Config resolution: one ``config.yaml`` for every dataset in an experiment.

An experiment is usually several acquisition folders that differ in only a
couple of parameters: a MIP folder segmented in 2-D and a z-stack folder
segmented in 3-D, or one folder per antibody panel. Historically that meant one
config file per folder (``config_2d.yaml`` / ``config_3d.yaml`` …) and a wrapper
script to loop over them.

Instead, this module resolves a single config into **N fully-formed dataset
configs**::

    defaults:                       # everything shared
      channels: {dapi_index: 0, marker_channels: [1, 2, 3]}
      segmentation: {mode: 2d, min_mask_size: 15}

    datasets:
      - {name: zmax_stain1,    directory: ./zmax_stain1}
      - {name: zstacks_stain1, directory: ./zstacks_stain1,
         segmentation: {mode: 3d}}          # deep-merged over defaults

Each entry is deep-merged over ``defaults`` (dicts merge key-by-key, scalars and
lists replace), so a dataset states only what makes it different. The resolved
name lands in the manifest as a ``dataset`` column, which is what lets the
downstream layer pool or facet across 2-D/3-D and across panels.

Backwards compatible: a config with no ``datasets:`` key is treated as a single
dataset built from the legacy top-level ``input.directory``, so every existing
config and command line keeps working unchanged.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Top-level keys that describe the pipeline itself rather than one dataset.
_NON_DEFAULT_KEYS = {"datasets", "defaults", "_root", "_name"}


def load_config(path: str | Path) -> dict:
    """Read a YAML config and stamp it with the directory it came from."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = str(path.parent.resolve())
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts merge key-by-key; every other type (scalars, lists) replaces
    wholesale: a dataset that sets ``marker_channels: [1, 2]`` means exactly
    those channels, not those appended to the default.
    """
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


class Dataset:
    """One resolved acquisition folder: a name, a directory, and a full config."""

    def __init__(self, name: str, directory: Path, cfg: dict):
        self.name = name
        self.directory = directory
        self.cfg = cfg

    @property
    def seg_mode(self) -> str:
        return (self.cfg.get("segmentation", {}).get("mode") or "2d").lower()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Dataset({self.name!r}, {self.seg_mode}, {self.directory})"


def _defaults_from(cfg: dict) -> dict:
    """The shared parameter block: an explicit ``defaults:`` plus legacy top-level keys.

    Legacy configs put ``channels`` / ``segmentation`` / … at the top level. Those
    are still honored as defaults so old config files resolve unchanged; an
    explicit ``defaults:`` block wins where the two overlap.
    """
    legacy = {k: v for k, v in cfg.items() if k not in _NON_DEFAULT_KEYS}
    return deep_merge(legacy, cfg.get("defaults") or {})


def resolve_datasets(cfg: dict, names: list[str] | None = None) -> list[Dataset]:
    """Expand a config into its dataset list, deep-merging each over the defaults.

    Parameters
    ----------
    cfg
        The loaded config (must carry ``_root``, as :func:`load_config` sets).
    names
        Optional subset of dataset names to return, e.g. from ``--datasets``.
        Unknown names raise, so a typo fails loudly instead of silently running
        nothing.
    """
    root = Path(cfg.get("_root", ".")).resolve()
    defaults = _defaults_from(cfg)
    entries = cfg.get("datasets")

    if not entries:
        # Legacy single-dataset config: derive one dataset from input.directory.
        directory = defaults.get("input", {}).get("directory")
        if not directory:
            raise ValueError(
                "config: define either a 'datasets:' list or 'input.directory'."
            )
        directory = _abspath(directory, root)
        entries = [{"name": directory.name, "directory": str(directory)}]

    resolved: list[Dataset] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"config: datasets[{i}] must be a mapping, got {type(entry).__name__}.")
        override = copy.deepcopy(entry)
        name = str(override.pop("name", None) or f"dataset{i}")
        directory = override.pop("directory", None)
        if not directory:
            raise ValueError(f"config: dataset {name!r} has no 'directory'.")
        if name in seen:
            raise ValueError(f"config: duplicate dataset name {name!r}.")
        seen.add(name)

        directory = _abspath(directory, root)
        merged = deep_merge(defaults, override)
        merged.setdefault("input", {})["directory"] = str(directory)
        merged["_root"] = str(root)
        merged["_name"] = name
        resolved.append(Dataset(name, directory, merged))

    if names:
        by_name = {d.name: d for d in resolved}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise ValueError(
                f"Unknown dataset(s) {unknown}. Available: {sorted(by_name)}."
            )
        resolved = [by_name[n] for n in names]

    logger.info("Resolved %d dataset(s): %s", len(resolved),
                ", ".join(f"{d.name}[{d.seg_mode}]" for d in resolved))
    return resolved


def _abspath(directory: str | Path, root: Path) -> Path:
    p = Path(directory).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def experiment_root(cfg: dict) -> Path:
    """Directory holding the config: the anchor for experiment-level outputs."""
    return Path(cfg.get("_root", ".")).resolve()


__all__ = ["load_config", "deep_merge", "resolve_datasets", "Dataset",
           "experiment_root"]
