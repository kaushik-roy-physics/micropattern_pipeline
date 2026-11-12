"""
Which markers are signaling reporters, and how each one is read.

Most markers are fate markers: an intensity on the shared scale, optionally a
positivity gate. A **signaling** marker is read differently, because what
carries the information is *where* the protein is, not how much there is:

``nuclear``
    Nuclear intensity with the image background subtracted, no division by the
    nuclear stain. Used for β-catenin, whose signaling pool is the nuclear one
    and which is never gated into positive / negative.
``nc``
    Nuclear-to-cytoplasmic ratio, the cytoplasm being a thin ring grown outward
    from each nucleus (:mod:`downstream.signaling.measure`). Used for the SMADs,
    which shuttle into the nucleus when their pathway is on.

Either readout is then expressed relative to a reference population (pluripotent
colonies, by default the negative-control wells) when the plate has one
(:mod:`downstream.signaling.reference`).

Spelling guardrails
-------------------
People write ``b-CAT``, ``β-catenin``, ``beta catenin``, ``bCat`` or ``CTNNB1`` for
the same antibody, and ``SMAD2/3``, ``Smad2-3`` or ``SMAD 2/3`` for another. Names
are matched on a **key**: lower case, Greek β and the word "beta" turned into
``b``, everything that is not a letter or digit removed. When the key alone does
not match, a trailing qualifier is tried off: a fluorophore (``-488``,
``AF647``), a species (``rb``, ``mouse``), a one-letter variant tag (``-r``,
``-g``) or anything in parentheses. Such a match is reported, never silent.

Phospho forms (``pSMAD1``, ``pSMAD2``) are deliberately **not** matched to their
total protein: a phospho-specific antibody is read as nuclear intensity by
convention and belongs in ``signaling.markers`` if it should be a signaling
readout. Names that look like a signaling protein but match nothing are listed
as near misses, so a typo shows up in the log instead of quietly turning a
SMAD into a fate marker.

Extension
---------
``signaling.markers`` in config.yaml adds or overrides entries by marker name::

    signaling:
      markers: {pSMAD1: nuclear, YAP1: nc, SMAD4: none}

``none`` turns a catalog match off (read the marker as a standard marker).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NUCLEAR = "nuclear"
NC = "nc"
NONE = "none"
READOUT_NAMES = (NUCLEAR, NC)


@dataclass(frozen=True)
class Entry:
    """One signaling protein: display name, default readout, accepted name keys."""
    name: str
    readout: str
    keys: tuple[str, ...]


# The built-in catalog. Keys are already in :func:`name_key` form.
CATALOG: tuple[Entry, ...] = (
    Entry("beta-catenin", NUCLEAR, ("bcat", "bcatenin", "ctnnb1", "activebcatenin",
                                    "activebcat", "nonphosphobcatenin")),
    Entry("SMAD2/3", NC, ("smad23",)),
    Entry("SMAD2", NC, ("smad2",)),
    Entry("SMAD3", NC, ("smad3",)),
    Entry("SMAD4", NC, ("smad4",)),
    Entry("SMAD1", NC, ("smad1",)),
)

# Fragments that make a name *look* like a signaling protein (near-miss check).
_LOOKALIKE = ("smad", "catenin", "bcat", "ctnnb", "yap", "taz", "wwtr")

_SPECIES = {"rb", "rabbit", "ms", "mouse", "gt", "goat", "rt", "rat", "ck", "chicken",
            "hu", "human", "dk", "donkey", "sh", "sheep", "gp", "guineapig"}
_FLUOR = re.compile(r"^(af|alexa|alexafluor|cy|atto|dy|df)?\d{3}$|^cy[357]$|^(fitc|tritc|pe|apc)$")


def name_key(name) -> str:
    """Spelling-insensitive key: ``'β-Catenin'`` → ``'bcatenin'``, ``'SMAD 2/3'`` → ``'smad23'``."""
    s = unicodedata.normalize("NFKC", str(name)).lower()
    s = s.replace("β", "b").replace("ß", "b")
    s = re.sub(r"beta", "b", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _strip_qualifiers(name: str) -> tuple[str, list[str]]:
    """Drop parentheses and trailing fluorophore / species / one-letter tokens."""
    dropped: list[str] = []
    base = str(name)
    for part in re.findall(r"\(([^)]*)\)", base):
        dropped.append(part.strip())
    base = re.sub(r"\([^)]*\)", " ", base)
    tokens = [t for t in re.split(r"[\s_\-]+", base.strip()) if t]
    while len(tokens) > 1:
        t = tokens[-1]
        k = name_key(t)
        if len(k) <= 1 or k in _SPECIES or _FLUOR.match(k):
            dropped.insert(0, t)
            tokens.pop()
            continue
        break
    return " ".join(tokens), dropped


def _catalog_index(cfg: dict) -> dict[str, tuple[str, str]]:
    """Name key → (entry name, readout) from the built-in catalog."""
    index: dict[str, tuple[str, str]] = {}
    for entry in CATALOG:
        index[name_key(entry.name)] = (entry.name, entry.readout)
        for k in entry.keys:
            index[k] = (entry.name, entry.readout)
    return index


def settings(cfg: dict) -> dict:
    """The ``signaling`` block with defaults filled in (see :data:`DEFAULTS`)."""
    from .measure import DEFAULTS as MEASURE_DEFAULTS
    out = dict(DEFAULTS)
    out.update(MEASURE_DEFAULTS)
    out.update({k: v for k, v in (cfg.get("signaling") or {}).items() if v is not None})
    if str(out["nuclear_space"]) not in NUCLEAR_SPACES:
        raise ValueError(f"signaling.nuclear_space must be one of {NUCLEAR_SPACES}")
    return out


DEFAULTS = {
    "enabled": True,               # recognize signaling markers at all
    "markers": {},                 # {name: nuclear | nc | none}: adds to / overrides the catalog
    "nuclear_space": "background_subtracted",   # nuclear readout: background_subtracted | raw | dapi_norm
    "reference_column": None,      # None → conditions.control_column
    "reference_values": None,      # None → conditions.control_values
    "gate": False,                 # also fit % positive gates for signaling markers
}
NUCLEAR_SPACES = ("background_subtracted", "raw", "dapi_norm")


@dataclass(frozen=True)
class Match:
    """How one marker name was resolved."""
    marker: str
    readout: str                   # nuclear | nc
    entry: str                     # catalog name, or 'config'
    how: str                       # 'config' | 'exact' | 'qualifier'
    dropped: tuple[str, ...] = ()

    def describe(self) -> str:
        what = {NUCLEAR: "nuclear level (signaling.nuclear_space)", NC: "nuclear:cytoplasmic ratio"}
        text = f"{self.marker}: {what.get(self.readout, self.readout)}"
        if self.how == "config":
            return text + " (signaling.markers)"
        text += f" (recognized as {self.entry}"
        if self.dropped:
            text += f", ignoring '{' '.join(self.dropped)}'"
        return text + ")"


def classify(marker: str, cfg: dict) -> Match | None:
    """The signaling readout of *marker*, or None for a standard marker.

    *cfg* is the ``downstream`` block. Order: an explicit ``signaling.markers``
    entry (exact name or same key) wins, then the catalog by key, then the
    catalog after stripping a trailing qualifier.
    """
    sig = cfg.get("signaling") or {}
    if sig.get("enabled", True) is False:
        return None
    explicit = {str(k): str(v).strip().lower() for k, v in (sig.get("markers") or {}).items()}
    key = name_key(marker)
    hit = explicit.get(str(marker))
    if hit is None:
        hit = next((v for k, v in explicit.items() if name_key(k) == key), None)
    if hit is not None:
        if hit in (NONE, "standard", "false", "off", ""):
            return None
        if hit not in READOUT_NAMES:
            raise ValueError(f"signaling.markers.{marker}: readout must be one of "
                             f"{READOUT_NAMES + (NONE,)}, got {hit!r}")
        return Match(str(marker), hit, "config", "config")
    index = _catalog_index(cfg)
    if key in index:
        entry, readout = index[key]
        return Match(str(marker), readout, entry, "exact")
    base, dropped = _strip_qualifiers(str(marker))
    if dropped and name_key(base) in index:
        entry, readout = index[name_key(base)]
        return Match(str(marker), readout, entry, "qualifier", tuple(dropped))
    return None


def classify_all(markers, cfg: dict) -> dict[str, Match]:
    out = {}
    for m in markers:
        match = classify(str(m), cfg)
        if match is not None:
            out[str(m)] = match
    return out


def near_misses(markers, cfg: dict) -> list[str]:
    """Markers that look like a signaling protein but were not recognized."""
    sig = cfg.get("signaling") or {}
    explicit = {name_key(k) for k in (sig.get("markers") or {})}
    out = []
    for m in markers:
        key = name_key(m)
        if key in explicit or classify(str(m), cfg) is not None:
            continue
        if any(f in key for f in _LOOKALIKE):
            out.append(str(m))
    return out


def spelling_groups(markers, cfg: dict) -> list[list[str]]:
    """Signaling marker names that differ only in spelling (same full key).

    Two such names are two markers to the pipeline, each with its own scale and
    reference. Names that differ by a qualifier (``b-CAT-r`` / ``b-CAT-g``) are
    left alone: those are usually two antibodies on purpose.
    """
    groups: dict[str, list[str]] = {}
    for m in markers:
        if classify(str(m), cfg) is None:
            continue
        groups.setdefault(name_key(m), []).append(str(m))
    return [sorted(set(g)) for g in groups.values() if len(set(g)) > 1]


def log_resolution(markers, cfg: dict) -> dict[str, Match]:
    """Resolve, and say in the log what was recognized, what nearly was, and why."""
    matches = classify_all(markers, cfg)
    for match in matches.values():
        logger.info("signaling: %s", match.describe())
    for m in near_misses(markers, cfg):
        logger.warning("signaling: '%s' looks like a signaling protein but is not in the "
                       "catalog, so it is read as a standard marker. If it is a signaling "
                       "readout, add it to signaling.markers (e.g. {%s: nc}).", m, m)
    for group in spelling_groups(markers, cfg):
        logger.warning("signaling: %s are spellings of one marker but are analyzed as "
                       "separate markers. Add marker_aliases to merge them, e.g. {%s: %s}.",
                       " / ".join(group), group[1], group[0])
    return matches


__all__ = ["CATALOG", "Entry", "Match", "NUCLEAR", "NC", "NONE", "READOUT_NAMES",
           "DEFAULTS", "NUCLEAR_SPACES", "name_key", "settings", "classify", "classify_all",
           "near_misses", "spelling_groups", "log_resolution"]
