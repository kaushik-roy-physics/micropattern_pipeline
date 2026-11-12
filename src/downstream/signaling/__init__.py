"""
Signaling markers: read by where the protein is, not by how much of it there is.

The pipeline stays general: a marker is a standard (fate) marker unless the
catalog or ``signaling.markers`` names it a signaling reporter. Those get

* a per-image measurement step (``signaling``) that measures the nucleus core,
  a cytoplasmic ring and the image background (:mod:`.measure`);
* a readout choosing which of those values the marker is analyzed in
  (:mod:`.readouts`): ``nuclear`` (background-subtracted, not ÷ nuclear stain)
  or ``nc`` (nuclear:cytoplasmic ratio);
* expression relative to a reference population, pluripotent colonies by
  default (:mod:`.reference`);
* no positivity gate unless ``signaling.gate: true``.

From there on they flow through the same profiles, aggregation, tests and
multi-marker figures as every other marker. Modules:

:mod:`.catalog`    names, spellings, guardrails (no heavy imports; the GUI uses it)
:mod:`.measure`    generic nucleus / ring / background measurement
:mod:`.readouts`   readout registry (extensible)
:mod:`.reference`  reference-population normalization and axis labels
"""

from .catalog import (CATALOG, NC, NUCLEAR, classify, classify_all, log_resolution,
                      name_key, near_misses, settings, spelling_groups)

__all__ = ["CATALOG", "NC", "NUCLEAR", "classify", "classify_all", "log_resolution",
           "name_key", "near_misses", "settings", "spelling_groups"]
