"""
Boundary and peak measurements recover known truth on synthetic colonies.

Run with ``python -m pytest tests/`` or plain ``python tests/test_boundary.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from downstream import boundary as bd  # noqa: E402

LN81 = 2 * np.log(9)


def _colony(seed=0, n=3000, radius=320.0, x_a=80.0, w_a=40.0, x_b=150.0, w_b=50.0,
            peak=175.0, wobble=0.0):
    """A disc of nuclei: A edge-high (x50 x_a), B center-high (x50 x_b), mutually
    exclusive, a signaling marker with an edge maximum and an interior peak.
    *wobble*: border radius varying with angle (µm amplitude)."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(-np.pi, np.pi, n)
    d = radius - r
    shift = wobble * np.sin(3 * theta)
    p_a = 1 / (1 + np.exp(LN81 / w_a * (d - x_a - shift)))
    p_b = 1 / (1 + np.exp(-LN81 / w_b * (d - x_b - shift)))
    a = rng.uniform(size=n) < p_a
    b = (rng.uniform(size=n) < p_b) & ~a                 # exclusive: no double positives
    va = np.where(a, 0.8, 0.05) + rng.normal(0, 0.05, n)
    vb = np.where(b, 0.7, 0.05) + rng.normal(0, 0.05, n)
    sig = 1.0 + 0.6 * np.exp(-d / 30) + 0.4 * np.exp(-0.5 * ((d - peak) / 25) ** 2) \
        + rng.normal(0, 0.1, n)
    values = {"A": va, "B": vb, "S": sig}
    positive = {"A": a.astype(float), "B": b.astype(float), "S": np.full(n, np.nan)}
    return d, theta, values, positive


def _metric(out, kind, subject, metric, readout=None):
    sel = (out["kind"] == kind) & (out["subject"] == subject) & (out["metric"] == metric)
    if readout:
        sel &= out["readout"] == readout
    row = out[sel]
    assert len(row) == 1, (kind, subject, metric, readout, len(row))
    return row.iloc[0]


def test_logistic_fit_recovers_position_and_width():
    rng = np.random.default_rng(1)
    x = np.arange(5, 300, 10.0)
    n = np.full(x.size, 80.0)
    p = 1 / (1 + np.exp(-LN81 / 40.0 * (x - 120.0)))
    k = rng.binomial(n.astype(int), p).astype(float)
    x50, c = bd.fit_logistic(x, n, k, k_max=10.0)
    assert abs(x50 - 120) < 6 and abs(LN81 / c - 40) < 10, (x50, LN81 / c)
    print(f"ok  logistic: x50 {x50:.1f} (120), width {LN81 / c:.1f} (40)")


def test_four_pl_recovers_a_falling_domain():
    rng = np.random.default_rng(2)
    x = np.arange(5, 300, 10.0)
    y = 0.1 + 0.6 / (1 + np.exp(LN81 / 30.0 * (x - 90.0))) + rng.normal(0, 0.01, x.size)
    x50, c, lo, hi = bd.fit_four_pl(x, np.full(x.size, 50.0), y, k_max=10.0)
    assert abs(x50 - 90) < 5 and c < 0 and abs(LN81 / abs(c) - 30) < 8, (x50, c)
    assert abs(lo - 0.1) < 0.05 and abs(hi - 0.7) < 0.05, (lo, hi)
    print("ok  4PL: falling domain recovered")


def test_bin_check_true_boundaries_do_not_move_with_binning():
    import pandas as pd
    d, theta, values, positive = _colony(seed=7)
    prm = dict(bd.DEFAULTS, n_boot=40)
    out = bd.measure_colony(d, theta, values, positive, [("A", "B")], ["S"], prm, 12.0,
                            np.random.default_rng(0))
    check = bd.bin_check(d, theta, values, positive, [("A", "B")], ["S"], prm, 12.0)
    assert set(check["scheme"]) == {"fixed:5", "fixed:20", "quantile:20"}, check["scheme"]
    colonies = out.assign(field_id=0, condition="c")
    summary = bd.bin_check_summary(check.assign(field_id=0), colonies, ["condition"])
    # Positions (x50 of gate calls, the crossing, the peak) stay put under every binning:
    # the truth is the same, only the bins change.
    is_pos = summary["metric"].isin(["x50_um", "crossing_um", "peak_um"])
    gate_or_peak = (summary["readout"] == "positive") | (summary["kind"] == "peak")
    pos = summary[is_pos & gate_or_peak]
    fixed = pos[pos["scheme"].str.startswith("fixed")]
    assert len(fixed) and (fixed["median_abs_shift"] < 3).all(), fixed
    # Equal-count bins are wide in the interior: an interior peak (truth 175 µm) moves a
    # few µm with them, which is what the check exists to show; it stays bounded.
    quant = pos[pos["scheme"].str.startswith("quantile")]
    assert (quant["median_abs_shift"] < 12).all(), quant
    peak = check[(check["metric"] == "peak_um") & check["scheme"].str.startswith("fixed")]
    assert (np.abs(peak["value"] - 175) < 4).all(), peak
    cross = check[(check["metric"] == "crossing_um")]
    truth = (LN81 / 40 * 80 + LN81 / 50 * 150) / (LN81 / 40 + LN81 / 50)
    assert (np.abs(cross["value"] - truth) < 12).all(), (cross, truth)
    with pytest_raises(ValueError):
        bd.parse_scheme("adaptive:3")
    print("ok  bin check: positions agree across fixed 5/20 µm and quantile bins")


def pytest_raises(exc):
    import contextlib

    @contextlib.contextmanager
    def ctx():
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{exc.__name__} not raised")
    return ctx()


def test_crossing_is_closed_form_and_needs_facing_markers():
    assert np.isclose(bd.crossing(80, -0.1, 150, 0.1), 115.0)
    assert np.isnan(bd.crossing(80, -0.1, 150, -0.1))
    print("ok  crossing")


def test_measure_colony_recovers_boundary_overlap_and_peak():
    d, theta, values, positive = _colony()
    prm = bd.settings({"boundary": {"n_boot": 60}})
    out = bd.measure_colony(d, theta, values, positive, [("A", "B")], ["S"], prm, 8.0,
                            np.random.default_rng(0))
    xa = _metric(out, "marker", "A", "x50_um", "positive")
    xb = _metric(out, "marker", "B", "x50_um", "positive")
    assert abs(xa.value - 80) < 8 and abs(xb.value - 150) < 10, (xa.value, xb.value)
    assert np.isfinite(xa.se) and 0 < xa.se < 15 and bool(xa.ok)
    wa = _metric(out, "marker", "A", "width_um", "positive").value
    assert abs(wa - 40) < 15, wa
    cross = _metric(out, "pair", "A / B", "crossing_um").value
    assert 80 < cross < 150, cross
    overlap = _metric(out, "pair", "A / B", "overlap_um").value
    assert abs(overlap - (80 - 150)) < 15, overlap         # a gap of ~70 µm
    dp = _metric(out, "pair", "A / B", "dp_fraction")
    assert dp.value == 0.0                                 # exclusive by construction
    assert _metric(out, "pair", "A / B", "dp_expected").value > 0
    peak = _metric(out, "peak", "S", "peak_um")
    assert abs(peak.value - 175) < 15, peak.value          # interior, not the edge maximum
    off = _metric(out, "peak", "S", "offset_um vs A / B").value
    assert np.isclose(off, peak.value - cross)
    print(f"ok  colony: A {xa.value:.0f}±{xa.se:.1f}, B {xb.value:.0f}, cross {cross:.0f}, "
          f"gap {-overlap:.0f}, peak {peak.value:.0f}")


def test_flat_colony_yields_no_accepted_transition():
    rng = np.random.default_rng(3)
    n = 2500
    r = 320 * np.sqrt(rng.uniform(0, 1, n))
    d, theta = 320 - r, rng.uniform(-np.pi, np.pi, n)
    values = {"A": 0.3 + rng.normal(0, 0.05, n)}
    positive = {"A": (rng.uniform(size=n) < 0.02).astype(float)}
    out = bd.measure_colony(d, theta, values, positive, [], [], bd.settings({"boundary": {
        "n_boot": 40}}), 8.0, np.random.default_rng(0))
    accepted = out[(out["metric"] == "x50_um") & out["ok"].astype(bool)]
    assert accepted.empty, accepted
    print("ok  flat colony: no transition reported")


def test_ragged_border_inflates_colony_se():
    """The sector bootstrap sees a border that wobbles with angle; nuclei would not."""
    prm = bd.settings({"boundary": {"n_boot": 80}})
    se = []
    for wobble in (0.0, 40.0):
        d, theta, values, positive = _colony(seed=4, wobble=wobble)
        out = bd.measure_colony(d, theta, values, positive, [("A", "B")], [], prm, 8.0,
                                np.random.default_rng(0))
        se.append(_metric(out, "marker", "A", "x50_um", "positive").se)
    assert se[1] > 1.5 * se[0], se
    print(f"ok  sector bootstrap: SE {se[0]:.1f} → {se[1]:.1f} µm with a ragged border")


def test_summary_uses_accepted_values_and_keeps_missing_keys():
    import pandas as pd
    t = pd.DataFrame({"condition": ["a", "a", "a", "b"], "class": [np.nan] * 4, "kind": "pair",
                      "subject": "A / B", "readout": "positive", "metric": "crossing_um",
                      "value": [100.0, 110.0, 500.0, 90.0], "ok": [True, True, False, True],
                      "well_key": ["w1", "w2", "w2", "w3"]})
    s = bd.summarize(t, ["condition", "class"], "colony").set_index("condition")
    assert s.loc["a", "mean"] == 105.0 and s.loc["a", "n"] == 2      # 500 was rejected
    assert s.loc["a", "n_colonies_rejected"] == 1 and s.loc["b", "n"] == 1
    w = bd.summarize(t, ["condition", "class"], "well").set_index("condition")
    assert w.loc["a", "n"] == 2 and w.loc["a", "mean"] == 105.0
    print("ok  summary: accepted values only, NaN keys kept")



def test_unresolved_pairs_say_why_a_boundary_is_missing():
    import pandas as pd
    rows = []

    def add(fid, kind, subject, metric, ok, support=1.0, stain="s1"):
        rows.append({"field_id": fid, "stain_set": stain, "kind": kind, "subject": subject,
                     "readout": "positive", "metric": metric, "value": 100.0, "ok": ok,
                     "boot_support": support})
    for m in ("A", "B"):                                  # colony 1: a clean boundary
        add(1, "marker", m, "x50_um", True)
    add(1, "pair", "A / B", "crossing_um", True)
    add(2, "marker", "B", "x50_um", True)                 # colony 2: A never turns on
    add(3, "marker", "A", "x50_um", True)                 # colony 3: a shaky crossing
    add(3, "marker", "B", "x50_um", True)
    add(3, "pair", "A / B", "crossing_um", False, support=0.45)
    add(4, "marker", "C", "x50_um", True, stain="s2")     # other stain set: no A / B
    notes = bd.unresolved_pairs(pd.DataFrame(rows), 0.8)
    assert set(notes) == {2, 3}, notes
    assert "no accepted A transition" in notes[2][0]
    assert "45% of bootstrap resamples, needs 80%" in notes[3][0]
    print("ok  unresolved pairs: each missing boundary carries its reason")

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
