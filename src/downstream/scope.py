"""
Gating scope: the pool a marker's threshold and band edges are estimated from.

A threshold is only meaningful over cells whose intensities are on a **common
scale**. What "common scale" means depends on the antibody, not on the code, so
the scope is declared per marker rather than hardcoded:

============  =========================================  ===========================
level         grouping columns                           pools
============  =========================================  ===========================
``panel``     ``dataset`` × ``plate_id`` × ``stain_set``  nothing (the default)
``plate``     ``plate_id`` × ``seg_mode``                 panels **and** datasets
``experiment`` ``seg_mode``                               plates as well
============  =========================================  ===========================

``panel`` is the conservative default and the historical behavior: a marker
measured in two antibody panels gets two thresholds, because a different
secondary, filter or exposure makes the raw scales incomparable.

``plate`` is for the common real case where the *same* antibody at the *same*
dilution appears in several panels across the wells of one plate: say
``[NKX2.5, ISL1, HAND1]`` in some wells and ``[MEF2C, ISL1, HAND2]`` in others.
There, splitting ISL1 into two gates is actively wrong: it makes ISL1-positive
fractions incomparable between wells of the same plate for no reason. Declaring
``markers: {ISL1: {scope: plate}}`` pools them into one gate estimated from more
cells.

Note what ``plate`` keeps apart. It drops ``dataset`` (panels are often stored
as sibling folders (``zmax_stain1/``, ``zmax_stain2/``), which become separate
datasets) but retains ``seg_mode``, so the same plate imaged as MIPs and as
z-stacks never shares a gate. Those are two different measurements of the same
cells, not two panels.

Because ``plate_id`` defaults to the *folder name*, sibling panel folders must
declare the same explicit ``plate_id:`` in their plate maps or ``plate`` scope
silently pools nothing. ``python main.py check`` reports this.

Mixed scopes in one table
-------------------------
Different markers may sit at different levels, so ``thresholds.csv`` carries the
**union** of the grouping columns and writes :data:`POOLED` (``*``) in the ones a
marker was pooled over. Joins never use those columns positionally; they go
through :func:`scope_keys`, which builds each row's key from that row's marker's
own columns.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PANEL = "panel"
PLATE = "plate"
EXPERIMENT = "experiment"

SCOPE_LEVELS: dict[str, list[str]] = {
    PANEL: ["dataset", "plate_id", "stain_set"],
    PLATE: ["plate_id", "seg_mode"],
    EXPERIMENT: ["seg_mode"],
}

# Stable column order for the union written into tables.
UNION_ORDER = ["dataset", "plate_id", "stain_set", "seg_mode"]

# Written into a scope column the marker was pooled over. A literal rather than
# NaN so it survives a CSV round-trip and can be used as a join key.
POOLED = "*"

DEFAULT_LEVEL = PANEL


def level_for(marker: str, cfg: dict) -> str:
    """Scope level declared for *marker* (``markers.<name>.scope``, else the default)."""
    default = str((cfg.get("thresholds") or {}).get("scope", DEFAULT_LEVEL))
    spec = (cfg.get("markers") or {}).get(marker) or {}
    level = str(spec.get("scope", default))
    if level not in SCOPE_LEVELS:
        raise ValueError(
            f"marker {marker!r}: scope must be one of {sorted(SCOPE_LEVELS)}, got {level!r}."
        )
    return level


def scope_columns_for(marker: str, cfg: dict, available) -> list[str]:
    """Grouping columns for *marker*, restricted to those the frame actually has."""
    available = set(available)
    return [c for c in SCOPE_LEVELS[level_for(marker, cfg)] if c in available]


def union_scope_columns(markers, cfg: dict, available) -> list[str]:
    """Union of the grouping columns used by *markers*, in a stable order."""
    available = set(available)
    used: set[str] = set()
    for marker in markers:
        used.update(SCOPE_LEVELS[level_for(marker, cfg)])
    return [c for c in UNION_ORDER if c in used and c in available]


def scope_keys(frame: pd.DataFrame, cfg: dict, available=None,
               marker_col: str = "marker") -> pd.Series:
    """Per-row scope key, each row keyed by *its own marker's* scope columns.

    The key is what every threshold/edge join goes through, so a marker at
    ``plate`` scope matches rows from both its panels while a marker at ``panel``
    scope stays split: in one pass over a frame holding both.
    """
    available = frame.columns if available is None else available
    out = pd.Series("", index=frame.index, dtype=object)
    for marker, idx in frame.groupby(marker_col, observed=True).groups.items():
        cols = scope_columns_for(str(marker), cfg, available)
        if not cols:
            continue
        block = frame.loc[idx, cols].astype(str)
        out.loc[idx] = block.agg("|".join, axis=1) if len(cols) > 1 else block[cols[0]]
    return out


def fill_scope_row(scope: dict, union_cols: list[str]) -> dict:
    """Expand one marker's scope dict to the union columns, marking pooled ones."""
    return {col: scope.get(col, POOLED) for col in union_cols}


def mark_pooled(frame: pd.DataFrame, cfg: dict,
                marker_col: str = "marker") -> pd.DataFrame:
    """Blank out, with :data:`POOLED`, the scope columns each row's marker pooled.

    Rollups that group by the *union* of the scope columns silently un-pool a
    pooled marker: a ``plate``-scoped marker's rows still carry their own
    ``stain_set``, so grouping on it splits one condition back into one row per
    panel: three rows claiming to be the same condition, whose fields are never
    combined into the single number the shared gate was estimated for. Marking
    the pooled columns first makes the grouping follow each marker's own scope,
    and leaves the table honest about what was pooled: exactly the ``*``
    convention ``thresholds.csv`` already uses.

    Columns a marker was *not* pooled over keep their values, so markers at
    different levels aggregate correctly side by side in one frame.
    """
    if marker_col not in frame.columns:
        return frame
    cols = [c for c in UNION_ORDER if c in frame.columns]
    if not cols:
        return frame

    out = frame.copy()
    groups = out.groupby(marker_col, observed=True).groups
    pooled_by_marker = {
        marker: [c for c in cols
                 if c not in set(scope_columns_for(str(marker), cfg, out.columns))]
        for marker in groups
    }
    # Categoricals reject a value outside their categories, and these columns are
    # written to CSV as text regardless.
    for col in {c for pooled in pooled_by_marker.values() for c in pooled}:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    for marker, idx in groups.items():
        if pooled_by_marker[marker]:
            out.loc[idx, pooled_by_marker[marker]] = POOLED
    return out


def pooling_candidates(long: pd.DataFrame, cfg: dict) -> list[dict]:
    """Markers that *could* be pooled across panels but are still gated per panel.

    A candidate is a marker measured in two or more ``stain_set``s within one
    ``plate_id`` × ``seg_mode`` while sitting at ``panel`` scope. When the long
    table carries a ``channel`` column the check is sharpened: sharing a channel
    index across those panels means the same filter/detector settings, which is
    the situation where pooling is usually right; a differing channel index means
    a different fluorophore and pooling would merge two genuinely different
    scales.

    Reported, never acted on: whether the antibody and dilution actually matched
    is a fact about the bench that no table records.
    """
    need = {"marker", "stain_set", "plate_id"}
    if not need.issubset(long.columns):
        return []

    group_cols = [c for c in ("plate_id", "seg_mode") if c in long.columns]
    has_channel = "channel" in long.columns
    out: list[dict] = []

    for keys, grp in long.groupby(group_cols, observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        where = dict(zip(group_cols, keys))
        for marker, sub in grp.groupby("marker", observed=True):
            marker = str(marker)
            panels = sorted(str(s) for s in sub["stain_set"].unique())
            if len(panels) < 2 or level_for(marker, cfg) != PANEL:
                continue
            channels: dict[str, int] = {}
            if has_channel:
                for panel, part in sub.groupby("stain_set", observed=True):
                    values = pd.unique(part["channel"].dropna())
                    if len(values) == 1:
                        channels[str(panel)] = int(values[0])
            same_channel = (len(channels) == len(panels)
                            and len(set(channels.values())) == 1) if channels else None
            out.append({**where, "marker": marker, "panels": panels,
                        "channels": channels, "same_channel": same_channel})
    return out


def log_pooling_candidates(long: pd.DataFrame, cfg: dict) -> list[dict]:
    """Emit the 'you could pool this marker' advisory for each candidate."""
    candidates = pooling_candidates(long, cfg)
    for cand in candidates:
        where = "/".join(str(v) for k, v in cand.items()
                         if k in ("plate_id", "seg_mode"))
        if cand["channels"]:
            detail = ", ".join(f"{p} ch{c}" for p, c in sorted(cand["channels"].items()))
        else:
            detail = ", ".join(cand["panels"])
        if cand["same_channel"] is False:
            logger.warning(
                "thresholds: %s appears in %d panels on %s (%s) on DIFFERENT channels; "
                "gates estimated separately, which is almost certainly right here. "
                "Do not set markers.%s.scope: plate unless the fluorophore really matched.",
                cand["marker"], len(cand["panels"]), where, detail, cand["marker"],
            )
        else:
            same = ", same channel" if cand["same_channel"] else ""
            logger.warning(
                "thresholds: %s appears in %d panels on %s (%s%s). Gates estimated "
                "separately. Set markers.%s.scope: plate to pool them into one gate "
                "if the antibody and dilution matched.",
                cand["marker"], len(cand["panels"]), where, detail, same, cand["marker"],
            )
    return candidates


def describe(marker: str, cfg: dict) -> str:
    """Human-readable scope description for logs and table notes."""
    level = level_for(marker, cfg)
    return f"{level} ({' × '.join(SCOPE_LEVELS[level])})"


__all__ = ["SCOPE_LEVELS", "UNION_ORDER", "POOLED", "PANEL", "PLATE", "EXPERIMENT",
           "DEFAULT_LEVEL", "level_for", "scope_columns_for", "union_scope_columns",
           "scope_keys", "fill_scope_row", "mark_pooled", "pooling_candidates",
           "log_pooling_candidates", "describe"]
