"""
Signaling readouts: which per-nucleus value a signaling marker is analyzed in.

A readout is a small object naming the column of ``intensities.parquet`` it
reads (its *space*) and whether it needs the cytoplasmic ring. The rest of the
pipeline (scales, profiles, aggregation, tests, figures) only ever sees that
column, so a readout plugs in without any step knowing it exists. Add one with
:func:`register_readout`, e.g. a total-cell readout for a cytoplasmic reporter.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data import DAPI_NORM, NC_RATIO, NUC_BGSUB, RAW
from . import catalog


@dataclass(frozen=True)
class Readout:
    name: str
    needs_ring: bool
    description: str

    def space(self, sig: dict) -> str:          # noqa: ARG002
        raise NotImplementedError

    def label(self, marker: str) -> str:
        return marker


@dataclass(frozen=True)
class NuclearReadout(Readout):
    """Nuclear intensity; background-subtracted by default, never ÷ nuclear stain
    unless ``signaling.nuclear_space: dapi_norm`` says so."""

    def space(self, sig: dict) -> str:
        return {"background_subtracted": NUC_BGSUB, "raw": RAW,
                "dapi_norm": DAPI_NORM}[str(sig.get("nuclear_space", "background_subtracted"))]

    def label(self, marker: str) -> str:
        return f"nuclear {marker}"


@dataclass(frozen=True)
class NCReadout(Readout):
    """(nucleus − background) / (cytoplasmic ring − background)."""

    def space(self, sig: dict) -> str:          # noqa: ARG002
        return NC_RATIO

    def label(self, marker: str) -> str:
        return f"{marker} N:C"


READOUTS: dict[str, Readout] = {
    catalog.NUCLEAR: NuclearReadout(catalog.NUCLEAR, False,
                                    "nuclear intensity, background-subtracted"),
    catalog.NC: NCReadout(catalog.NC, True, "nuclear:cytoplasmic ratio (annular ring)"),
}


def register_readout(readout: Readout) -> None:
    """Make a new readout available to ``signaling.markers`` by its name."""
    READOUTS[readout.name] = readout
    if readout.name not in catalog.READOUT_NAMES:
        catalog.READOUT_NAMES = catalog.READOUT_NAMES + (readout.name,)


def readout_for(marker: str, cfg: dict) -> Readout | None:
    """The readout of *marker* (``downstream`` config block), or None for a standard marker."""
    match = catalog.classify(str(marker), cfg)
    return READOUTS.get(match.readout) if match is not None else None


__all__ = ["Readout", "NuclearReadout", "NCReadout", "READOUTS", "register_readout",
           "readout_for"]
