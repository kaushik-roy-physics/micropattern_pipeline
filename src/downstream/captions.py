"""
What every intensity figure is normalized to, in one line, and a guard on the
nuclear denominator.

A profile is only readable when its legend says what the y-axis is. Per marker
the value is one of: the raw nuclear median, the nuclear median divided by the
nuclear stain (``dapi_norm``), or, for a signaling marker, its own readout
(background-subtracted, N:C, or relative to reference wells). Standard markers
are then rescaled 0-1 between their P1 and P99 over every nucleus of their
scope, never per colony. :func:`normalization_caption` writes that down,
grouping markers that share a description, and appends
``visualization.caption`` (free text, e.g. the cell line).

**Histone denominators.** Dividing by the nuclear channel is a correction for
DNA content and focus only when that channel is a DNA stain. A histone fusion
(H2B-GFP, H2B-RFP) varies with chromatin state and from cell to cell; if it is
also the reporter of interest, dividing by it divides by the signal itself.
:func:`nuclear_stain_kind` classifies the nuclear channel from
``nuclear_stain`` (``dna`` | ``histone`` | ``auto``: from its channel
name), and :func:`histone_denominator_markers` lists the markers that divide by
a histone channel, which the qc step logs as a warning and the caption states.
"""

from __future__ import annotations

import re
import textwrap

from .data import DAPI_NORM, NC_RATIO, NUC_BGSUB, RAW

HISTONE = "histone"
DNA = "dna"
_HISTONE_NAME = re.compile(r"(^|[^a-z0-9])(h2a|h2b|h3|h4|hist|histone)", re.I)

_SPACE_TEXT = {
    RAW: "raw nuclear median",
    DAPI_NORM: "nuclear median ÷ nuclear stain{nuc}",
    NUC_BGSUB: "nucleus core − local background (a.u.)",
    NC_RATIO: "nucleus:cytoplasm ratio, background-subtracted",
}


def nuclear_stain_kind(names, cfg: dict) -> str:
    """``dna`` or ``histone`` for the nuclear channel(s) named *names*."""
    declared = str(cfg.get("nuclear_stain") or "auto").lower()
    if declared in (DNA, HISTONE):
        return declared
    if declared != "auto":
        raise ValueError("nuclear_stain must be dna, histone or auto")
    return HISTONE if any(_HISTONE_NAME.search(str(n)) for n in names or ()) else DNA


def histone_denominator_markers(spaces: dict, kind: str) -> list[str]:
    """Markers analyzed ÷ the nuclear channel when that channel is a histone label."""
    if kind != HISTONE:
        return []
    return sorted(m for m, s in spaces.items() if s == DAPI_NORM)


def normalization_caption(spaces: dict, scopes: dict, signaling: dict | None = None,
                          nuclear_names=(), kind: str = DNA, note: str | None = None,
                          width: int = 150) -> str:
    """One wrapped caption naming each marker's normalization.

    *spaces*: ``marker → space`` (data column) for standard markers; *scopes*:
    ``marker → scope level`` (panel, plate, experiment); *signaling*: ``marker →
    unit label`` for signaling markers, which keep their own unit.
    """
    nuc = ""
    if nuclear_names:
        nuc = f" ({'/'.join(sorted(set(map(str, nuclear_names))))}"
        nuc += ", a histone label, not a DNA stain)" if kind == HISTONE else ")"
    groups: dict[str, list[str]] = {}
    for marker, space in spaces.items():
        if signaling and marker in signaling:
            continue
        text = _SPACE_TEXT.get(space, space).format(nuc=nuc)
        text += f", P1–P99 → 0–1 per {scopes.get(marker, 'panel')}"
        groups.setdefault(text, []).append(str(marker))
    parts = [f"{', '.join(ms)}: {t}" for t, ms in groups.items()]
    for marker, label in (signaling or {}).items():
        parts.append(f"{marker}: {label}")
    text = "Normalization: " + " · ".join(parts) + " · no per-colony rescaling"
    if note:
        text += " · " + str(note).strip()
    return "\n".join(textwrap.wrap(text, width))


__all__ = ["DNA", "HISTONE", "nuclear_stain_kind", "histone_denominator_markers",
           "normalization_caption"]
