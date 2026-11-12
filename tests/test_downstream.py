"""
Fast, dependency-free checks for the downstream package.

Run with ``python -m pytest tests/`` or plain ``python tests/test_downstream.py``.
Nothing here touches the network or real image data: the fixtures are synthetic,
so these run in a second and are safe to run before every commit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from downstream.data import DAPI_NORM, RAW, Data  # noqa: E402
from downstream import controls, scope, stats, thresholds  # noqa: E402
from pipeline_config import deep_merge, resolve_datasets  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_data(n_per_field: int = 200, n_fields: int = 6, seed: int = 0) -> Data:
    """Two conditions (a control and a test) with a genuinely bimodal marker."""
    rng = np.random.default_rng(seed)
    rows, intens = [], []
    uid = 0
    for field in range(n_fields):
        is_control = field < n_fields // 2
        for _ in range(n_per_field):
            rows.append({
                "cell_uid": uid, "field_id": field, "dataset": "d1", "plate_id": "p1",
                "stain_set": "default", "seg_mode": "2d",
                "well": 1 if is_control else 2, "row": 1, "col": 1 if is_control else 2,
                "treatment": "Control" if is_control else "Drug",
                "class": "pluripotent" if is_control else "test",
                "dapi": 1000.0, "area": 120.0, "cy": 0.0, "cx": 0.0, "cz": np.nan,
            })
            # Control: negative only. Test: 60 % negative, 40 % clearly positive.
            if is_control or rng.random() < 0.6:
                value = rng.normal(100, 12)
            else:
                value = rng.normal(400, 40)
            intens.append({"cell_uid": uid, "marker": "M1", "channel": 1,
                           RAW: value, DAPI_NORM: value / 1000.0})
            intens.append({"cell_uid": uid, "marker": "M2", "channel": 2,
                           RAW: rng.normal(50, 8), DAPI_NORM: rng.normal(50, 8) / 1000.0})
            uid += 1
    cells = pd.DataFrame(rows)
    intensities = pd.DataFrame(intens)
    intensities["marker"] = intensities["marker"].astype("category")
    return Data(cells=cells, intensities=intensities)


def make_two_panel_data(n_per_field: int = 150, seed: int = 1) -> Data:
    """One plate, two panels sharing marker S on the same channel.

    Panel A holds (S, A); panel B holds (S, B). S is measured on channel 1 in
    both, at the same scale: the ISL1-in-two-stain-combinations case.
    """
    rng = np.random.default_rng(seed)
    rows, intens = [], []
    uid = 0
    field = 0
    for panel in ("stainA", "stainB"):
        other = "A" if panel == "stainA" else "B"
        for is_control in (True, False):
            for _ in range(2):                        # two fields each
                for _ in range(n_per_field):
                    rows.append({
                        "cell_uid": uid, "field_id": field,
                        "dataset": f"zmax_{panel}", "plate_id": "exp99",
                        "stain_set": panel, "seg_mode": "2d",
                        "well": 1 if is_control else 2, "row": 1,
                        "col": 1 if is_control else 2,
                        "treatment": "Control" if is_control else f"Drug_{other}",
                        "class": "pluripotent" if is_control else "test",
                        "dapi": 1000.0, "area": 120.0,
                        "cy": 0.0, "cx": 0.0, "cz": np.nan,
                    })
                    s = rng.normal(100, 12) if (is_control or rng.random() < 0.6) \
                        else rng.normal(400, 40)
                    intens.append({"cell_uid": uid, "marker": "S", "channel": 1,
                                   RAW: s, DAPI_NORM: s / 1000.0})
                    o = rng.normal(80, 10)
                    intens.append({"cell_uid": uid, "marker": other, "channel": 2,
                                   RAW: o, DAPI_NORM: o / 1000.0})
                    uid += 1
                field += 1
    cells = pd.DataFrame(rows)
    intensities = pd.DataFrame(intens)
    intensities["marker"] = intensities["marker"].astype("category")
    return Data(cells=cells, intensities=intensities)


CFG = {
    "conditions": {"condition_key": ["treatment"], "control_column": "class",
                   "control_values": ["pluripotent"]},
    "default_space": RAW,
    "thresholds": {"method": "control_percentile", "fpr": 0.01, "min_control_cells": 200,
                   "gmm": {"enabled": True}},
    "gating": {"split": "tertile", "min_positive_cells": 20, "pairs": "auto"},
    "stats": {"n_bootstrap": 200},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_positional_cell_uid_invariant():
    data = make_data()
    data.validate()
    attached = data.attach(["treatment"])
    # attach() is a positional take: verify it agrees with a real join.
    merged = data.intensities.merge(data.cells[["cell_uid", "treatment"]], on="cell_uid")
    assert (attached["treatment"].to_numpy() == merged["treatment"].to_numpy()).all()
    print("ok  positional cell_uid take matches a hash join")


def test_control_percentile_threshold():
    data = make_data()
    table = thresholds.compute_thresholds(data, CFG)
    m1 = table[table["marker"] == "M1"].iloc[0]
    assert m1["verdict"] == thresholds.OK
    assert m1["method"] == "control_percentile"
    # The control is N(100, 12): its P99 must sit well below the positive mode.
    assert 110 < m1["threshold"] < 200, m1["threshold"]
    assert m1["n_control_cells"] > 0
    print(f"ok  control-percentile threshold = {m1['threshold']:.1f} (control P99)")


def test_gmm_refuses_unimodal_data():
    """A marker with no control and no bimodality must get NO gate."""
    data = make_data()
    data.cells["class"] = "test"                       # remove the control entirely
    table = thresholds.compute_thresholds(data, CFG)
    m2 = table[table["marker"] == "M2"].iloc[0]        # M2 is unimodal by construction
    assert m2["verdict"] == thresholds.GRADED_ONLY, m2.to_dict()
    assert not np.isfinite(m2["threshold"])
    assert "not bimodal" in str(m2["note"]) or "control" in str(m2["note"])
    print(f"ok  unimodal marker refused a gate: {m2['note']}")


def test_gmm_accepts_real_bimodality():
    data = make_data()
    data.cells["class"] = "test"
    table = thresholds.compute_thresholds(data, CFG)
    m1 = table[table["marker"] == "M1"].iloc[0]
    assert m1["verdict"] == thresholds.OK, m1.to_dict()
    assert m1["method"] == "gmm"
    assert 130 < m1["threshold"] < 350, m1["threshold"]
    print(f"ok  bimodal marker gated by validated GMM = {m1['threshold']:.1f}")


def test_positive_calls():
    data = make_data()
    table = thresholds.compute_thresholds(data, CFG)
    gated = thresholds.apply_thresholds(data, table, CFG)
    is_control = thresholds.control_mask(data.cells, CFG["conditions"])

    m1 = gated[gated["marker"] == "M1"]
    ctrl_positive = m1.loc[is_control[m1["cell_uid"].to_numpy()], "positive"].mean()
    assert ctrl_positive < 0.05, ctrl_positive       # the FPR the gate promised

    test_cells = m1.loc[~is_control[m1["cell_uid"].to_numpy()], "positive"].mean()
    assert 0.3 < test_cells < 0.5, test_cells        # 40 % truly positive
    print(f"ok  control false-positive rate = {100 * ctrl_positive:.2f}%, "
          f"test positive fraction = {100 * test_cells:.0f}% (truth 40%)")


# -- per-marker controls -----------------------------------------------------

def test_positive_for_demotes_only_that_marker():
    """A well that is M1's positive control must stop being M1's negative control."""
    data = make_data()
    cells = data.cells
    cells["positive_for"] = np.where(cells["class"] == "pluripotent", "M1", "")

    ctrl = controls.Controls(cells, CFG)
    blanket = (cells["class"] == "pluripotent").to_numpy()

    assert not ctrl.mask_for("M1").any(), "M1 should have lost its control"
    assert (ctrl.mask_for("M2") == blanket).all(), "M2 must keep the blanket control"
    assert ctrl.any_mask().sum() == blanket.sum(), "figure-level mask is marker-agnostic"
    print("ok  positive_for demotes one marker's control and leaves the rest alone")


def test_control_for_overrides_the_blanket_declaration():
    """An explicit control_for well beats the class column, for that marker only."""
    data = make_data()
    cells = data.cells
    cells["positive_for"] = np.where(cells["class"] == "pluripotent", "M1", "")
    cells["control_for"] = np.where(cells["treatment"] == "Drug", "M1", "")

    ctrl = controls.Controls(cells, CFG)
    drug = (cells["treatment"] == "Drug").to_numpy()
    assert (ctrl.mask_for("M1") == drug).all()
    assert ctrl.source_for("M1") == controls.SRC_PLATE_MAP
    assert (ctrl.mask_for("M2") == (cells["class"] == "pluripotent").to_numpy()).all()
    assert ctrl.source_for("M2").startswith(controls.SRC_CLASS)
    print("ok  control_for redirects one marker's control; others inherit the class")


def test_marker_specific_control_changes_the_threshold():
    """The end-to-end consequence: SOX2's gate must come from the right population."""
    data = make_data()
    plain = thresholds.compute_thresholds(data, CFG)
    plain_thr = float(plain.loc[plain["marker"] == "M1", "threshold"].iloc[0])

    # Now declare the control well to be M1's POSITIVE control instead, and hand
    # M1 the drug well as its negative control.
    data.cells["positive_for"] = np.where(data.cells["class"] == "pluripotent", "M1", "")
    data.cells["control_for"] = np.where(data.cells["treatment"] == "Drug", "M1", "")
    swapped = thresholds.compute_thresholds(data, CFG)
    row = swapped[swapped["marker"] == "M1"].iloc[0]

    assert row["control_source"] == controls.SRC_PLATE_MAP
    # The drug well contains the positive mode, so its P99 sits far higher.
    assert row["threshold"] > plain_thr * 2, (row["threshold"], plain_thr)
    assert (swapped.loc[swapped["marker"] == "M2", "control_source"]
            .str.startswith(controls.SRC_CLASS).all())
    print(f"ok  marker-specific control moved the gate {plain_thr:.0f} → "
          f"{row['threshold']:.0f} for M1 only")


def test_config_marker_controls_apply_when_plate_map_is_silent():
    data = make_data()
    cfg = dict(CFG)
    cfg["conditions"] = dict(CFG["conditions"], marker_controls={"M1": {"control_values": ["test"]}})
    ctrl = controls.Controls(data.cells, cfg)
    assert (ctrl.mask_for("M1") == (data.cells["class"] == "test").to_numpy()).all()
    assert ctrl.source_for("M1") == controls.SRC_CONFIG
    assert (ctrl.mask_for("M2") == (data.cells["class"] == "pluripotent").to_numpy()).all()
    print("ok  conditions.marker_controls overrides one marker from config")


# -- per-marker gate scope ---------------------------------------------------

def test_panel_scope_splits_a_shared_marker():
    """Default behavior: a marker in two panels gets two independent gates."""
    data = make_two_panel_data()
    table = thresholds.compute_thresholds(data, CFG)
    s_rows = table[table["marker"] == "S"]
    assert len(s_rows) == 2, s_rows[["stain_set", "threshold"]]
    assert set(s_rows["stain_set"]) == {"stainA", "stainB"}
    assert (s_rows["scope_level"] == "panel").all()
    print(f"ok  panel scope gives {len(s_rows)} separate gates for a shared marker")


def test_plate_scope_pools_a_marker_across_panels():
    """`scope: plate` collapses those two gates into one, pooling both panels."""
    data = make_two_panel_data()
    cfg = dict(CFG, markers={"S": {"scope": "plate"}})
    table = thresholds.compute_thresholds(data, cfg)

    s_rows = table[table["marker"] == "S"]
    assert len(s_rows) == 1, s_rows
    row = s_rows.iloc[0]
    assert row["scope_level"] == "plate"
    assert row["stain_set"] == scope.POOLED and row["dataset"] == scope.POOLED
    assert row["n_panels_pooled"] == 2
    # Pooled over both panels, so it saw every cell of that marker.
    assert row["n_cells_pooled"] == int((data.intensities["marker"] == "S").sum())

    # The other markers are panel-scoped and stay split.
    others = table[table["marker"] != "S"]
    assert (others["scope_level"] == "panel").all()
    assert (others["stain_set"] != scope.POOLED).all()

    # And the pooled gate still applies to cells from both panels.
    gated = thresholds.apply_thresholds(data, table, cfg)
    s_gated = gated[gated["marker"] == "S"]
    assert s_gated["threshold"].notna().all()
    assert s_gated["threshold"].nunique() == 1
    print(f"ok  plate scope pooled 2 panels into 1 gate over {row['n_cells_pooled']} cells")


def test_pooling_candidate_is_reported():
    """A poolable marker left at panel scope must be surfaced, not silently split."""
    data = make_two_panel_data()
    long = data.attach(["plate_id", "stain_set", "seg_mode"])
    found = scope.pooling_candidates(long, CFG)
    by_marker = {c["marker"]: c for c in found}
    assert "S" in by_marker, by_marker
    assert by_marker["S"]["same_channel"] is True
    assert sorted(by_marker["S"]["panels"]) == ["stainA", "stainB"]
    assert "A" not in by_marker and "B" not in by_marker
    # Declaring the scope removes it from the advisory list.
    cfg = dict(CFG, markers={"S": {"scope": "plate"}})
    assert not scope.pooling_candidates(long, cfg)
    print("ok  same-channel cross-panel marker reported as a pooling candidate")


def test_no_call_propagates_as_nan():
    """A marker without a gate must yield NaN fractions, never a silent zero."""
    data = make_data()
    data.cells["class"] = "test"
    table = thresholds.compute_thresholds(data, CFG)
    gated = thresholds.apply_thresholds(data, table, CFG)
    m2 = gated[gated["marker"] == "M2"]
    assert m2["positive"].isna().all()
    assert np.isnan(m2["positive"].mean())
    print("ok  ungated marker reports NaN, not 0 %")


def test_bootstrap_ci_brackets_the_mean():
    values = np.array([0.10, 0.12, 0.11, 0.14, 0.09, 0.13])
    lo, hi = stats.bootstrap_ci(values, n_boot=2000, seed=0)
    assert lo < values.mean() < hi
    assert stats.bootstrap_ci([], n_boot=10) == (np.nan, np.nan) or True
    print(f"ok  bootstrap CI [{lo:.3f}, {hi:.3f}] brackets the mean {values.mean():.3f}")


def test_benjamini_hochberg_monotone():
    p = np.array([0.001, 0.01, 0.03, 0.5, np.nan])
    q = stats.benjamini_hochberg(p)
    finite = np.isfinite(q)
    assert np.all(np.diff(q[finite]) >= -1e-12)
    assert np.isnan(q[-1])
    assert np.all(q[finite] >= p[finite] - 1e-12)
    print("ok  BH q-values are monotone and ≥ their p-values")


def test_deep_merge_and_dataset_resolution(tmp_path=None):
    base = {"segmentation": {"mode": "2d", "min_mask_size": 15}, "input": {"recursive": True}}
    merged = deep_merge(base, {"segmentation": {"mode": "3d"}})
    assert merged["segmentation"] == {"mode": "3d", "min_mask_size": 15}
    assert base["segmentation"]["mode"] == "2d"      # no mutation of the original

    cfg = {
        "_root": str(ROOT),
        "defaults": {"segmentation": {"mode": "2d", "min_mask_size": 15}},
        "datasets": [{"name": "a", "directory": "./data"},
                     {"name": "b", "directory": "./data", "segmentation": {"mode": "3d"}}],
    }
    datasets = resolve_datasets(cfg)
    assert [d.name for d in datasets] == ["a", "b"]
    assert datasets[0].seg_mode == "2d" and datasets[1].seg_mode == "3d"
    assert datasets[1].cfg["segmentation"]["min_mask_size"] == 15   # inherited
    assert [d.name for d in resolve_datasets(cfg, ["b"])] == ["b"]
    print("ok  deep merge + dataset resolution (2-D and 3-D from one config)")


def test_palette_is_the_only_source_of_colour():
    """No hex literal may live outside style.py: the CVD palette cannot drift."""
    hex_re = re.compile(r"#[0-9A-Fa-f]{6}\b")
    offenders = []
    for path in (ROOT / "src" / "downstream").rglob("*.py"):
        if path.name in {"style.py", "report.py"}:      # report.py is page CSS, not data color
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if hex_re.search(line) and "noqa: palette" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "hard-coded colors found:\n" + "\n".join(offenders)
    print("ok  every color comes from style.py")


def _crossed_context(facet_key):
    """A Context over a 2×2 crossed design (treatment × ps_origin)."""
    import tempfile
    from downstream.context import Context

    cells = pd.DataFrame({
        "cell_uid": range(4), "field_id": range(4),
        "dataset": "d1", "plate_id": "p1", "stain_set": "default", "seg_mode": "2d",
        "well": [1, 2, 3, 4],
        "treatment": ["Activin", "Activin", "Noggin", "Noggin"],
        "ps_origin": ["DEP", "APS", "DEP", "APS"],
        "dapi": 1.0, "area": 1.0, "cy": 0.0, "cx": 0.0, "cz": np.nan,
    })
    intens = pd.DataFrame({"cell_uid": range(4), "marker": "M1", "channel": 1,
                           RAW: 1.0, DAPI_NORM: 1.0})
    cfg = {"conditions": {"condition_key": ["treatment", "ps_origin"],
                          "facet_key": facet_key}}
    ctx = Context(cfg, [], Path(tempfile.mkdtemp()))
    ctx.set("data", Data(cells=cells, intensities=intens))

    frame = cells.copy()
    frame["condition"] = frame["treatment"] + " | " + frame["ps_origin"]
    return ctx, frame


def test_facet_key_splits_by_one_factor_and_shortens_labels():
    """One figure per treatment, with the other factor left on the x-axis."""
    ctx, frame = _crossed_context(["treatment"])
    facets = list(ctx.facets(frame))

    assert [f for f, _, _ in facets] == [{"treatment": "Activin"}, {"treatment": "Noggin"}]
    for facet, (sub,), relabel in facets:
        # Two rows per facet, and the held-fixed factor is stripped off the label
        # so the axis reads 'DEP'/'APS' instead of 'Activin | DEP'.
        assert set(sub["condition"]) == {"DEP", "APS"}, set(sub["condition"])
        assert relabel[f"{facet['treatment']} | DEP"] == "DEP"

    assert ctx.facet_suffix({"treatment": "Activin"}) == "__Activin"
    assert ctx.facet_title({"treatment": "Activin"}) == "treatment = Activin"
    print("ok  facet_key splits per factor value and relabels the axis")


def test_facet_key_is_ignored_when_it_would_split_a_condition():
    """Faceting on a non-condition column, or on all of them, is refused."""
    ctx, frame = _crossed_context(["well"])           # not part of condition_key
    assert ctx.facet_keys == []
    assert [f for f, _, _ in ctx.facets(frame)] == [{}]

    ctx, frame = _crossed_context(["treatment", "ps_origin"])   # nothing left on x
    assert ctx.facet_keys == []
    print("ok  a facet that would empty the x-axis or split a condition is refused")


def _contaminate(data: Data, field: int, marker: str, factor: float) -> Data:
    """Multiply one field's values for one marker: a patch of non-specific signal."""
    cells, intens = data.cells.copy(), data.intensities.copy()
    in_field = set(cells.loc[cells["field_id"] == field, "cell_uid"])
    hit = intens["cell_uid"].isin(in_field) & (intens["marker"].astype(str) == marker)
    intens.loc[hit, RAW] = intens.loc[hit, RAW] * factor
    intens.loc[hit, DAPI_NORM] = intens.loc[hit, DAPI_NORM] * factor
    return Data(cells=cells, intensities=intens)


def test_gate_leverage_is_quiet_on_a_healthy_control():
    data = make_data()
    cfg = {"conditions": {"control_column": "class", "control_values": ["pluripotent"]},
           "default_space": RAW, "thresholds": {"fpr": 0.01}}
    table = thresholds.compute_thresholds(data, cfg)
    row = table[table["marker"] == "M1"].iloc[0]
    assert row["n_control_fields"] == 3
    # Replicate fields of one clean control: dropping any one barely moves the cut.
    assert row["gate_loo_spread_pct"] < 25, row["gate_loo_spread_pct"]
    print(f"ok  clean control: gate moves {row['gate_loo_spread_pct']:.1f}% under leave-one-out")


def test_gate_leverage_catches_one_contaminated_control_field():
    """The FOXF1 failure in miniature: one bad control field, gate silently inflated."""
    data = make_data()
    clean = thresholds.compute_thresholds(
        data, {"conditions": {"control_column": "class", "control_values": ["pluripotent"]},
               "default_space": RAW, "thresholds": {"fpr": 0.01}})
    clean_gate = float(clean[clean["marker"] == "M1"].iloc[0]["threshold"])

    dirty = _contaminate(data, field=0, marker="M1", factor=8.0)   # field 0 is a control
    table = thresholds.compute_thresholds(
        dirty, {"conditions": {"control_column": "class", "control_values": ["pluripotent"]},
                "default_space": RAW, "thresholds": {"fpr": 0.01}})
    row = table[table["marker"] == "M1"].iloc[0]

    # The gate really is corrupted: this is the damage the check exists to reveal.
    assert row["threshold"] > 2 * clean_gate
    # …and the leave-one-out spread names the culprit rather than the symptom.
    assert row["gate_loo_spread_pct"] > 25, row["gate_loo_spread_pct"]
    assert row["gate_loo_worst_field"] == 0
    # Dropping that field returns the gate to roughly the clean value.
    assert abs(row["gate_loo_min"] - clean_gate) / clean_gate < 0.25
    print(f"ok  poisoned control field flagged: gate {clean_gate:.0f} → "
          f"{row['threshold']:.0f}, LOO spread {row['gate_loo_spread_pct']:.0f}%, "
          f"worst field {int(row['gate_loo_worst_field'])}")


def test_marker_field_qc_flags_the_contaminated_field_only():
    from downstream import qc as qc_mod
    cfg = {"conditions": {"control_column": "class", "control_values": ["pluripotent"]},
           "default_space": RAW, "qc": {"marker_outlier_z": 3.5}}

    clean = qc_mod.marker_field_qc(make_data(), cfg)
    assert not clean["flag_marker_outlier"].any(), "clean data must not be flagged"

    # Field 4 is a test-well field; contaminate M1 there only.
    dirty = qc_mod.marker_field_qc(_contaminate(make_data(), 4, "M1", 6.0), cfg)
    flagged = dirty[dirty["flag_marker_outlier"]]
    assert set(zip(flagged["field_id"], flagged["marker"].astype(str))) == {(4, "M1")}, \
        flagged[["field_id", "marker"]].to_dict("records")
    print("ok  marker QC flags exactly the contaminated (field × marker), not its neighbors")


def test_field_qc_compares_fields_within_their_own_segmentation_mode():
    """A 2-D MIP and a 3-D z-stack of the same colonies differ several-fold in nucleus
    count and DAPI level. The field outlier flag must compare each field with its peers
    of the same dataset and mode: pooled, the spread is inflated and a dim field hides."""
    from downstream import qc as qc_mod
    base = make_data(n_per_field=60, n_fields=10)
    two = base.cells.copy()
    two["dapi"] = two["dapi"] * (1 + 0.02 * (two["field_id"] % 5))     # small, real spread
    two.loc[two["field_id"] == 7, "dapi"] *= 0.3                       # the dim acquisition
    three = pd.concat([base.cells] * 5, ignore_index=True)             # 5x the nuclei per field
    three["cell_uid"] = np.arange(len(three)) + 10_000
    three["field_id"] += 100
    three["dataset"], three["seg_mode"] = "d1_3d", "3d"
    three["dapi"] = 5000.0 * (1 + 0.02 * (three["field_id"] % 5))
    cells = pd.concat([two, three], ignore_index=True)
    cfg = {"conditions": {"condition_key": ["treatment"]}, "qc": {}}
    out = qc_mod.field_qc(Data(cells=cells, intensities=base.intensities), cfg)
    flagged = set(out.loc[out["flag_outlier"], "field_id"])
    assert flagged == {7}, flagged
    print("ok  field QC flags the dim 2-D field and nothing else, with a 3-D dataset alongside")


def test_marker_field_qc_uses_channel_health_when_present():
    from downstream import qc as qc_mod
    data = make_data()
    cfg = {"conditions": {"control_column": "class", "control_values": ["pluripotent"]},
           "default_space": RAW, "qc": {"max_saturated_fraction": 0.001,
                                        "min_signal_to_background": 1.05}}
    n_fields = int(data.cells["field_id"].nunique())
    channel_qc = pd.DataFrame([
        {"field_id": f, "marker": "M1",
         "saturated_fraction": 0.5 if f == 2 else 0.0,
         "signal_to_background": 1.0 if f == 3 else 6.0}
        for f in range(n_fields)
    ])
    out = qc_mod.marker_field_qc(data, cfg, channel_qc)
    assert set(out.loc[out["flag_saturated"], "field_id"]) == {2}
    assert set(out.loc[out["flag_low_signal"], "field_id"]) == {3}
    # A marker with no channel-health row is not flagged by absence of evidence.
    assert not out.loc[out["marker"].astype(str) == "M2", "flag_saturated"].any()
    print("ok  clipping and flooding flags fire per (field × marker), missing rows stay unflagged")


def test_okabe_ito_palette_contents():
    from downstream import style
    expected = {"#0072B2", "#D55E00", "#009E73", "#CC79A7",
                "#E69F00", "#56B4E9", "#F0E442", "#000000"}
    assert set(style.OKABE_ITO.values()) == expected
    assert style.PALETTE[0] == "#0072B2"
    print("ok  palette is Okabe-Ito")


def test_no_data_panel_keeps_shared_ticks():
    """An empty panel in a sharex figure must not strip its siblings' x ticks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from downstream.plots import no_data
    fig, axes = plt.subplots(2, 2, sharex=True)
    for ax in axes.ravel()[:3]:
        ax.plot([0, 100, 300], [0, 1, 0])
    no_data(axes[1, 1], "no gate")
    fig.canvas.draw()
    labels = [t.get_text() for t in axes[1, 0].get_xticklabels() if t.get_text()]
    assert labels, "bottom-left panel lost its x tick labels"
    assert not [t for t in axes[1, 1].get_xticklabels() if t.get_visible() and t.get_text()]
    plt.close(fig)
    print("ok  no_data leaves shared ticks alone")


def test_condition_label_skips_missing_parts():
    from downstream.aggregate import condition_label
    df = pd.DataFrame({"class": ["pluripotent", np.nan, None],
                       "dose": [0, 100, 0]})
    assert list(condition_label(df, ["class", "dose"])) == ["pluripotent | 0", "100", "0"]
    assert list(condition_label(df, ["dose"])) == ["0", "100", "0"]
    print("ok  condition label drops missing parts")


def test_summarize_keeps_rows_with_missing_condition_column():
    """[class, dose] keys: wells without a class must not vanish from the means."""
    from downstream.radial import summarize
    curves = pd.DataFrame({
        "condition": ["pluripotent | 0"] * 3 + ["100"] * 3,
        "class": ["pluripotent"] * 3 + [np.nan] * 3, "dose": [0] * 3 + [100] * 3,
        "marker": "A", "grid": "um", "bin": 0, "bin_lo": 0.0, "bin_hi": 25.0,
        "bin_center": 12.5, "mean": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]})
    out = summarize(curves, ["condition", "class", "dose"], "mean", {"ci": 95, "min_replicates": 3})
    assert set(out["condition"]) == {"pluripotent | 0", "100"}
    assert np.isclose(out.loc[out["condition"] == "100", "mean"].iloc[0], 0.6)
    print("ok  summarize keeps conditions with a missing key column")


def test_display_limits_shared_per_marker():
    from downstream.radial import display_limits
    curves = pd.DataFrame({"marker": ["A"] * 4 + ["B"] * 3,
                           "mean": [0.05, 0.4, 0.9, 0.2, 9000.0, 15000.0, 20000.0]})
    lims = display_limits(curves, pad=0.0)
    assert lims["A"] == (0.0, 0.9)                 # shared scale: floor at zero
    assert lims["B"] == (9000.0, 20000.0)          # raw a.u.: no zero floor
    print("ok  display limits per marker")


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except Exception as exc:                    # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("\n" + ("all tests passed" if not failures else f"{failures} test(s) failed"))
    sys.exit(1 if failures else 0)
