"""
Positivity thresholds: one cut per (scope × marker).

Scope
-----
A threshold is estimated once per **scope × marker** and applied to every well in
that scope. Never per well, never per treatment. The scope is declared per marker
(see :mod:`downstream.scope`) and defaults to ``panel`` (``dataset × plate_id ×
stain_set``), which encodes two claims:

* The plate is the batch. Staining, exposure and background differ between
  plates, so a cut is never shared across ``plate_id`` by default.
* A marker measured in two different panels gets **two** thresholds, because a
  different secondary, filter or exposure makes the raw scales incomparable.

That second claim is a default, not a law. When the *same* antibody at the *same*
dilution appears in several panels across the wells of one plate, splitting it
into two gates makes its positive fractions incomparable between wells for no
reason. ``markers: {ISL1: {scope: plate}}`` pools those panels into one gate;
``seg_mode`` stays in the key either way, so MIPs and z-stacks of the same cells
never share a cut. Markers left at ``panel`` scope that *look* poolable are
reported as advisories: see :func:`downstream.scope.log_pooling_candidates`.

Controls
--------
The negative control is resolved **per marker** by :class:`downstream.controls.Controls`,
so a pluripotent well can be the negative control for BRA and CDX2 while being the
positive control for SOX2. ``control_source`` in the output table records which
rule supplied each marker's control.

Methods
-------
``control_percentile`` (default): the threshold is the ``1 − fpr`` quantile of
the negative control's own values, i.e. an empirical false-positive-rate cut:
``fpr = 0.01`` calls at most 1 % of true negatives positive. Estimated from the
**control wells only**; the pooled all-well distribution is what it is *applied*
to. This is Eq. (2) of ``Cell_fate_classification.pdf`` with ``fpr`` in place of
the paper's P95.

``mad``: ``median + sigma · 1.4826 · MAD`` of the control, raised to the
``1 − fpr`` quantile if that is higher (a hybrid guard against a control that
itself contains positives).

``fpr`` may be overridden per marker, ``markers: {NKX2-5: {fpr: 0.001}}``, for
an antibody whose negative control is a much tighter spike than the rest of the
panel's, where the shared P99 sits barely above background.

**GMM fallback**, used only when a marker has no usable control on that plate:
a 2-component Gaussian mixture is fitted to the pooled distribution of *all*
wells with that panel, and the cut is the weighted intersection of the two
components. This is gated behind a bimodality test: the mixture is accepted only
if all of:

* BIC improves over a 1-component fit by ``min_bic_gain``;
* standardized separation ``|μ₂−μ₁| / √((σ₁²+σ₂²)/2) ≥ min_separation``;
* the minor component holds at least ``min_weight`` of the mass.

If any check fails **no threshold is emitted**. The marker is marked
``graded_only``: its distributions are still plotted, but nothing downstream will
report a positive fraction for it. Refusing to answer is the correct behavior:
a fabricated cut propagates silently into every composition figure.

Is the gate trustworthy?
------------------------
A control-percentile gate is a quantile of one population, so it inherits
whatever is in that population. If a patch of non-specific staining, a bright
z-slice or a bubble contaminates part of a **control** well, the quantile climbs
and the marker reports *fewer* positives: it looks biologically negative rather
than obviously broken, which is the hardest kind of error to notice.

:func:`control_field_leverage` recomputes each gate with one control field held
out at a time and records the spread in ``thresholds.csv``
(``gate_loo_min/max``, ``gate_loo_spread_pct``, ``gate_loo_worst_field``,
``n_control_fields``). Fields of one control well are replicates of one
population, so a healthy gate barely moves; one that jumps when a single field
leaves is describing that field rather than the control. Exceeding
``thresholds.loo_warn_pct`` (default 25 %) logs a warning naming the field, and a
gate resting on a single control field always warns. The emitted threshold is
never changed by this: it is a statement about how much to trust the number,
not a correction to it.

Every decision, and the evidence for it, lands in ``thresholds.csv``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from . import scope as scope_mod
from .controls import Controls, control_mask
from .data import Data, select_values

logger = logging.getLogger(__name__)

# Verdicts recorded per (scope, marker).
OK = "ok"
GRADED_ONLY = "graded_only"
NOT_GATED = "signaling"     # signaling marker: read relative to a reference, never gated

# Every column that can take part in a scope key, in table order. A given marker
# uses a subset (see :mod:`downstream.scope`); the rest are written as ``*``.
SCOPE_COLUMNS = scope_mod.UNION_ORDER


def _mad_threshold(values: np.ndarray, sigma: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + sigma * 1.4826 * mad


def _gate_from_control(ctrl: np.ndarray, method: str, fpr: float, sigma: float) -> float:
    """The rule that turns a control population into a cut, in one place.

    Factored out so the leave-one-field-out check re-applies the *same* rule
    rather than an approximation of it.
    """
    quantile = float(np.quantile(ctrl, 1.0 - fpr))
    if method == "mad":
        return max(_mad_threshold(ctrl, sigma), quantile)
    return quantile


def control_field_leverage(ctrl: np.ndarray, fields: np.ndarray, gate: float,
                           method: str, fpr: float, sigma: float) -> dict:
    """How much of the gate rests on any single control field.

    Recomputes the threshold with each control field held out in turn. A gate
    estimated from a healthy control barely moves: the fields are replicates of
    one population, so removing one shifts a quantile of the pooled cells only
    slightly. A gate that jumps when one field leaves is not a property of the
    control population at all; it is a property of that field, which is precisely
    what a patch of non-specific staining, a bright z-slice or a bubble produces.

    This is the cheap detector for the failure mode that is otherwise invisible:
    a contaminated **control** raises the cut, so the marker reports *fewer*
    positives and looks biologically negative rather than obviously broken.

    Cost is one quantile per control field, microseconds, and nothing here
    changes the emitted threshold. It only records how much to trust it.
    """
    out = {"n_control_fields": int(np.unique(fields).size),
           "gate_loo_min": np.nan, "gate_loo_max": np.nan,
           "gate_loo_spread_pct": np.nan, "gate_loo_worst_field": np.nan}
    uniq = np.unique(fields)
    if uniq.size < 2:
        return out

    gates = np.array([_gate_from_control(ctrl[fields != f], method, fpr, sigma)
                      for f in uniq], dtype=float)
    out["gate_loo_min"] = float(np.nanmin(gates))
    out["gate_loo_max"] = float(np.nanmax(gates))
    if np.isfinite(gate) and gate > 0:
        out["gate_loo_spread_pct"] = float(100.0 * (gates.max() - gates.min()) / gate)
    out["gate_loo_worst_field"] = float(uniq[int(np.argmax(np.abs(gates - gate)))])
    return out


def fit_bimodal(values: np.ndarray, gmm_cfg: dict) -> dict:
    """Fit a 2-component mixture and judge whether the data are really bimodal.

    Returns the diagnostics plus ``threshold`` (the weighted intersection of the
    two components, back-transformed) and ``accepted``.
    """
    out = {
        "gmm_n": int(values.size), "gmm_bic_gain": np.nan, "gmm_separation": np.nan,
        "gmm_min_weight": np.nan, "threshold": np.nan, "accepted": False,
        "reason": "",
    }
    min_cells = int(gmm_cfg.get("min_cells", 200))
    if values.size < min_cells:
        out["reason"] = f"only {values.size} cells (<{min_cells})"
        return out

    log_transform = bool(gmm_cfg.get("log_transform", True))
    x = np.log1p(np.clip(values, 0, None)) if log_transform else values
    x = x[np.isfinite(x)].reshape(-1, 1)
    if x.size < min_cells:
        out["reason"] = "too few finite values"
        return out

    seed = int(gmm_cfg.get("random_state", 0))
    one = GaussianMixture(n_components=1, random_state=seed, max_iter=300).fit(x)
    two = GaussianMixture(n_components=2, random_state=seed, max_iter=300,
                          n_init=int(gmm_cfg.get("n_init", 3))).fit(x)

    means = two.means_.ravel()
    stds = np.sqrt(two.covariances_.ravel())
    weights = two.weights_.ravel()
    lo, hi = int(np.argmin(means)), int(np.argmax(means))

    bic_gain = float(one.bic(x) - two.bic(x))
    pooled_sd = float(np.sqrt((stds[lo] ** 2 + stds[hi] ** 2) / 2.0))
    separation = float(abs(means[hi] - means[lo]) / pooled_sd) if pooled_sd > 0 else np.inf
    min_weight = float(weights.min())

    out.update(gmm_bic_gain=bic_gain, gmm_separation=separation, gmm_min_weight=min_weight)

    failures = []
    if bic_gain < float(gmm_cfg.get("min_bic_gain", 10.0)):
        failures.append(f"BIC gain {bic_gain:.1f} < {gmm_cfg.get('min_bic_gain', 10.0)}")
    if separation < float(gmm_cfg.get("min_separation", 2.0)):
        failures.append(f"separation {separation:.2f} < {gmm_cfg.get('min_separation', 2.0)}")
    if min_weight < float(gmm_cfg.get("min_weight", 0.05)):
        failures.append(f"minor weight {min_weight:.3f} < {gmm_cfg.get('min_weight', 0.05)}")
    if failures:
        out["reason"] = "not bimodal: " + "; ".join(failures)
        return out

    # Weighted intersection of the two components, back on the original scale.
    cut = (means[lo] * stds[hi] + means[hi] * stds[lo]) / (stds[lo] + stds[hi])
    out["threshold"] = float(np.expm1(cut) if log_transform else cut)
    out["accepted"] = True
    return out


def compute_thresholds(data: Data, cfg: dict, controls: Controls | None = None) -> pd.DataFrame:
    """Estimate every (scope × marker) threshold and record the evidence.

    Each marker is grouped by *its own* scope columns and gated against *its own*
    negative control, so one pass handles a panel-scoped marker, a plate-scoped
    one pooled over two stain sets, and a marker whose control lives in a
    different well from everybody else's.
    """
    thr_cfg = cfg.get("thresholds") or {}
    gmm_cfg = thr_cfg.get("gmm") or {}

    method = str(thr_cfg.get("method", "control_percentile"))
    fpr = float(thr_cfg.get("fpr", 0.01))
    sigma = float(thr_cfg.get("sigma", 4.5))
    min_control = int(thr_cfg.get("min_control_cells", 200))
    gmm_enabled = bool(gmm_cfg.get("enabled", True))
    loo_warn = float(thr_cfg.get("loo_warn_pct", 25.0))

    cells = data.cells
    controls = controls if controls is not None else Controls(cells, cfg)

    available = [c for c in SCOPE_COLUMNS if c in cells.columns]
    carry = [c for c in (available + ["well", "field_id"]) if c in cells.columns]
    long = data.attach(carry)

    markers = list(pd.unique(long["marker"]))
    union_cols = scope_mod.union_scope_columns(markers, cfg, available)
    from .signaling.catalog import classify, settings as sig_settings
    gate_signaling = bool(sig_settings(cfg).get("gate", False))
    gated = [m for m in markers if gate_signaling or classify(str(m), cfg) is None]
    scope_mod.log_pooling_candidates(long[long["marker"].isin(gated)], cfg)

    if not controls.any_mask().any():
        logger.warning(
            "No negative-control cells found (conditions.control_column=%r, values=%r, "
            "and no plate-map control_for declarations). Every marker will use the GMM "
            "fallback.",
            controls.control_column, sorted(controls.control_values),
        )

    records: list[dict] = []
    for marker in markers:
        marker = str(marker)
        space = data.space_for(marker, cfg)
        match = classify(marker, cfg)
        if match is not None and not gate_signaling:
            logger.info("thresholds: %s is a signaling marker (%s); no positivity gate "
                        "(set signaling.gate: true to fit one).", marker, match.readout)
            continue
        scope_cols = scope_mod.scope_columns_for(marker, cfg, available)
        level = scope_mod.level_for(marker, cfg)

        # Per-marker false-positive rate. One plate-wide `fpr` assumes every
        # antibody has the same background behavior; a marker whose negative
        # control is a very tight spike puts its P99 barely above background and
        # calls nearly everything positive, which makes its binary fraction
        # useless for separating conditions even when it is literally correct.
        # `markers: {NKX2-5: {fpr: 0.001}}` tightens that one gate and leaves the
        # rest of the panel alone.
        marker_fpr = float(((cfg.get("markers") or {}).get(marker) or {})
                           .get("fpr", fpr))
        if marker_fpr != fpr:
            logger.info("thresholds: %s uses a per-marker fpr of %g (panel default %g)",
                        marker, marker_fpr, fpr)

        block = long[long["marker"] == marker].copy()
        block["_is_control"] = controls.mask_for(marker)[block["cell_uid"].to_numpy()]
        source = controls.source_for(marker)

        groups = (block.groupby(scope_cols, observed=True, sort=True)
                  if scope_cols else [((), block)])
        for keys, grp in groups:
            keys = keys if isinstance(keys, tuple) else (keys,)
            scope = dict(zip(scope_cols, keys))
            in_group = grp["_is_control"].to_numpy()

            values = grp[space].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            pooled = values[finite]
            ctrl = values[finite & in_group]

            rec = {
                **scope_mod.fill_scope_row(scope, union_cols),
                "marker": marker,
                "scope_level": level,
                "scope_key": "|".join(str(scope[c]) for c in scope_cols) if scope_cols else "",
                "space": space,
                "n_cells_pooled": int(pooled.size),
                "n_wells_pooled": int(grp["well"].nunique()) if "well" in grp else np.nan,
                "n_panels_pooled": (int(grp["stain_set"].nunique())
                                    if "stain_set" in grp else np.nan),
                "n_control_cells": int(ctrl.size),
                "n_control_wells": (int(grp.loc[in_group, "well"].nunique())
                                    if "well" in grp else np.nan),
                "control_source": source,
                "fpr": marker_fpr,
                "threshold": np.nan, "method": "", "verdict": GRADED_ONLY, "note": "",
                "threshold_mad": np.nan, "threshold_control_percentile": np.nan,
                "gmm_bic_gain": np.nan, "gmm_separation": np.nan, "gmm_min_weight": np.nan,
                "n_control_fields": np.nan, "gate_loo_min": np.nan, "gate_loo_max": np.nan,
                "gate_loo_spread_pct": np.nan, "gate_loo_worst_field": np.nan,
            }

            if ctrl.size >= min_control:
                quantile = float(np.quantile(ctrl, 1.0 - marker_fpr))
                mad = _mad_threshold(ctrl, sigma)
                rec["threshold_control_percentile"] = quantile
                rec["threshold_mad"] = mad
                if method == "mad":
                    chosen = max(mad, quantile)
                    rec["note"] = ("MAD raised to the control quantile"
                                   if quantile > mad else f"median + {sigma}·1.4826·MAD")
                else:
                    chosen = quantile
                    rec["note"] = f"negative-control P{100 * (1 - marker_fpr):.4g}"
                rec.update(threshold=float(chosen), method=method, verdict=OK)

                if "field_id" in grp.columns:
                    ctrl_fields = grp["field_id"].to_numpy()[finite & in_group]
                    lev = control_field_leverage(ctrl, ctrl_fields, float(chosen),
                                                 method, marker_fpr, sigma)
                    rec.update(lev)
                    _warn_on_leverage(marker, _scope_label(scope) or level, lev,
                                      float(chosen), loo_warn)
            elif gmm_enabled:
                fit = fit_bimodal(pooled, gmm_cfg)
                rec.update(gmm_bic_gain=fit["gmm_bic_gain"],
                           gmm_separation=fit["gmm_separation"],
                           gmm_min_weight=fit["gmm_min_weight"], method="gmm")
                short_control = (f"{ctrl.size} control cells (<{min_control}); "
                                 if ctrl.size else "no control wells; ")
                if fit["accepted"]:
                    rec.update(threshold=float(fit["threshold"]), verdict=OK,
                               note=short_control + "validated bimodal GMM")
                else:
                    rec["note"] = short_control + fit["reason"]
            else:
                rec.update(method="none", note="no control and GMM fallback disabled")

            label = _scope_label(scope) or level
            if rec["verdict"] != OK:
                logger.warning("No threshold for %s [%s]: %s", marker, label, rec["note"])
            else:
                pooled_note = ""
                if level != scope_mod.PANEL and rec.get("n_panels_pooled", 1) > 1:
                    pooled_note = f", pooled over {int(rec['n_panels_pooled'])} panels"
                logger.info("Threshold %-10s [%s] %s = %.5g  (%s; control %s%s)",
                            marker, label, space, rec["threshold"], rec["note"],
                            source, pooled_note)
            records.append(rec)

    table = pd.DataFrame(records)
    if table.empty:              # every marker is an ungated signaling marker
        table = pd.DataFrame(columns=union_cols + ["marker", "scope_level", "scope_key",
                                                   "space", "threshold", "method", "verdict",
                                                   "note"])
    n_ok = int((table["verdict"] == OK).sum())
    logger.info("thresholds: %d/%d (scope × marker) gated; %d graded-only",
                n_ok, len(table), len(table) - n_ok)
    return table


def threshold_lookup(table: pd.DataFrame, cfg: dict | None = None) -> dict:
    """``{(marker, scope_key): threshold}`` for accepted gates only.

    ``scope_key`` is the marker's own scope columns joined with ``|``: the same
    key :func:`downstream.scope.scope_keys` builds on the cell side, so a
    plate-scoped marker's single gate matches rows from every panel it appears in.
    """
    ok = table[table["verdict"] == OK]
    if not len(ok):
        return {}
    keys = _table_scope_keys(ok, cfg)
    return dict(zip(zip(ok["marker"].astype(str), keys), ok["threshold"].astype(float)))


def _table_scope_keys(table: pd.DataFrame, cfg: dict | None) -> pd.Series:
    """Scope key per threshold row, preferring the stored column when present."""
    if "scope_key" in table.columns:
        return table["scope_key"].fillna("").astype(str)
    available = [c for c in SCOPE_COLUMNS if c in table.columns]
    return scope_mod.scope_keys(table, cfg or {}, available)


def apply_thresholds(data: Data, table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a numeric ``positive`` column (1.0 / 0.0 / NaN) to the long table.

    Cells of a marker with no accepted threshold get **NaN**, "no call", so the
    absence of a gate propagates as NaN through every mean and fraction instead of
    silently counting as negative and reporting 0 %.

    Float rather than a boolean dtype on purpose: ``mean()`` of an all-NaN float
    column is NaN and every numpy predicate keeps working, whereas pandas'
    nullable boolean returns ``pd.NA`` and raises inside ``np.isfinite``.

    The join is by ``(marker, scope_key)`` rather than by a fixed column tuple,
    which is what lets markers at different scope levels coexist in one table.
    """
    available = [c for c in SCOPE_COLUMNS if c in data.cells.columns]
    long = data.attach(available)
    long["scope_key"] = scope_mod.scope_keys(long, cfg, available)

    table_keys = _table_scope_keys(table, cfg)
    index = pd.MultiIndex.from_arrays([table["marker"].astype(str), table_keys])
    thr_map = pd.Series(table["threshold"].to_numpy(), index=index)
    space_map = pd.Series(table["space"].to_numpy(), index=index)

    keys = pd.MultiIndex.from_arrays([long["marker"].astype(str), long["scope_key"]])
    thr = thr_map.reindex(keys).to_numpy(dtype=float)
    spaces = space_map.reindex(keys).to_numpy()

    # A marker without a row in the table (a signaling marker that is not
    # gated) is still read in its own analysis space, never silently in raw.
    spaces = np.where(pd.isna(spaces), long["marker"].astype(str).map(
        {str(m): data.space_for(str(m), cfg) for m in long["marker"].astype(str).unique()}
    ).to_numpy(), spaces)
    values = select_values(long, spaces)

    long["value"] = values.astype(np.float32)
    long["threshold"] = thr
    callable_ = np.isfinite(thr) & np.isfinite(values)
    positive = np.where(callable_, (values >= thr).astype(float), np.nan)
    long["positive"] = positive.astype(np.float32)
    return long


def _warn_on_leverage(marker: str, label: str, lev: dict, gate: float,
                      warn_pct: float) -> None:
    """Say it in the log, where it is seen during the run rather than after it."""
    n = lev.get("n_control_fields")
    if n == 1:
        logger.warning(
            "thresholds: %s [%s]; the whole gate rests on ONE control field. "
            "Nothing can check it; a single artifact in that field moves every "
            "positive fraction this marker reports.", marker, label)
        return
    spread = lev.get("gate_loo_spread_pct")
    if spread is not None and np.isfinite(spread) and spread > warn_pct:
        logger.warning(
            "thresholds: %s [%s]; gate %.4g moves by %.0f%% (%.4g-%.4g) when a "
            "single control field is dropped; field %g is the most influential. "
            "Check that field before trusting this marker's positive fractions.",
            marker, label, gate, spread, lev.get("gate_loo_min"),
            lev.get("gate_loo_max"), lev.get("gate_loo_worst_field"))


def _scope_label(scope: dict) -> str:
    return "/".join(str(v) for k, v in scope.items() if k != "marker")


__all__ = ["compute_thresholds", "apply_thresholds", "threshold_lookup",
           "control_mask", "Controls", "fit_bimodal", "control_field_leverage",
           "OK", "GRADED_ONLY", "NOT_GATED", "SCOPE_COLUMNS"]
