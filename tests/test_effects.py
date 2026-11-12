"""
Effects, trends, variance components and power against simulated designs with a
known answer (colonies nested in wells nested in experiments).

Run with ``python -m pytest tests/`` or plain ``python tests/test_effects.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from downstream import effects as ef  # noqa: E402


def simulate(rng, conditions=None, E=3, W=2, C=4, sd_e=6.0, sd_w=4.0, sd_c=8.0):
    """Per-colony values: condition mean + experiment + well + colony noise.

    The experiment effect is shared by every condition of that differentiation
    (a batch), as it is in a real plate.
    """
    conditions = conditions or {"ctrl": 100.0, "treat": 120.0}
    rows, fid = [], 0
    for e in range(E):
        b_e = rng.normal(0, sd_e)
        for cond, mu in conditions.items():
            for w in range(W):
                b_w = rng.normal(0, sd_w)
                for _ in range(C):
                    rows.append({"field_id": fid, "condition": cond, "experiment": f"E{e}",
                                 "well_key": f"E{e}|{cond}|w{w}",
                                 "value": mu + b_e + b_w + rng.normal(0, sd_c)})
                    fid += 1
    return pd.DataFrame(rows)


def test_unit_values_average_colonies_then_wells():
    df = pd.DataFrame({"field_id": range(6), "condition": "a", "experiment": ["E1"] * 6,
                       "well_key": ["w1", "w1", "w1", "w2", "w2", "w2"],
                       "value": [1.0, 2.0, 3.0, 10.0, 10.0, 10.0]})
    # Wells 2 and 10 → experiment 6 (each well weighs the same, not each colony).
    exp = ef.unit_values(df, ["condition"], "experiment", "experiment")
    assert len(exp) == 1 and exp["value"].iloc[0] == 6.0 and exp["n_colonies"].iloc[0] == 6
    assert len(ef.unit_values(df, ["condition"], "well")) == 2
    assert len(ef.unit_values(df, ["condition"], "colony")) == 6
    print("ok  unit values: colony → well → experiment")


def test_variance_components_recover_the_truth():
    rng = np.random.default_rng(0)
    est = []
    for _ in range(40):
        df = simulate(rng, E=6, W=3, C=5)
        vc = ef.variance_components(df, ["condition"], "experiment")
        est.append((vc["sd_colony"], vc["sd_well"], vc["sd_experiment"]))
    c, w, e = np.nanmean(np.asarray(est), axis=0)
    assert abs(c - 8) < 0.8 and abs(w - 4) < 1.2, (c, w)
    # The experiment SD is estimated within condition; the shared batch shifts
    # every condition alike, so it is identifiable here.
    assert abs(e - 6) < 1.8, e
    print(f"ok  variance components: colony {c:.1f} (8), well {w:.1f} (4), experiment {e:.1f} (6)")


def test_unestimable_levels_are_nan_not_zero():
    rng = np.random.default_rng(1)
    df = simulate(rng, E=1, W=1, C=5)
    vc = ef.variance_components(df, ["condition"], "experiment")
    assert np.isfinite(vc["sd_colony"])
    assert np.isnan(vc["sd_well"]) and np.isnan(vc["sd_experiment"]), vc
    vc2 = ef.variance_components(df, ["condition"], None)
    assert np.isnan(vc2["sd_experiment"])
    print("ok  one well, one experiment: well and experiment SDs are NaN")


def test_mdd_has_the_promised_power_in_simulation():
    """At the computed MDD, simulated experiments detect the effect ≈ 80 % of the time."""
    sd_e, sd_w, sd_c, E, W, C = 6.0, 4.0, 8.0, 4, 2, 4
    var = ef.group_mean_var(sd_e, sd_w, sd_c, E, W, C)
    delta = ef.mdd(var, E, 0.05, 0.8)
    assert abs(ef.power_for(delta, var, E, 0.05) - 0.8) < 0.02
    rng = np.random.default_rng(2)
    hits, n = 0, 600
    from scipy import stats as sps
    for _ in range(n):
        # Independent differentiations per condition (the design the MDD assumes).
        a = simulate(rng, {"x": 0.0}, E=E, W=W, C=C, sd_e=sd_e, sd_w=sd_w, sd_c=sd_c)
        b = simulate(rng, {"x": delta}, E=E, W=W, C=C, sd_e=sd_e, sd_w=sd_w, sd_c=sd_c)
        ua = ef.unit_values(a, ["condition"], "experiment", "experiment")["value"]
        ub = ef.unit_values(b, ["condition"], "experiment", "experiment")["value"]
        hits += sps.ttest_ind(ua, ub).pvalue < 0.05
    rate = hits / n
    assert 0.72 < rate < 0.88, rate
    print(f"ok  MDD {delta:.1f}: simulated power {rate:.2f} (target 0.80)")


def test_effect_ci_covers_and_permutation_is_exact_for_small_n():
    rng = np.random.default_rng(3)
    cover, n = 0, 300
    for _ in range(n):
        a, r = rng.normal(5, 2, 6), rng.normal(3, 2, 6)
        e = ef.effect(a, r, 2000, rng)
        cover += e["ci_lo"] <= 2 <= e["ci_hi"]
    assert 0.9 < cover / n < 0.99, cover / n
    p, exact = ef.permutation_p(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), 5000, rng)
    assert exact and np.isclose(p, 2 / 20)         # 2 of the 20 labelings are as extreme
    print(f"ok  effect CI coverage {cover / n:.2f}; exact permutation p = {p:.2f}")


def test_trend_recovers_a_log_dose_slope_and_drops_zero():
    rng = np.random.default_rng(4)
    doses = np.repeat([0, 10, 100, 1000], 4).astype(float)
    y = 50 + 20 * np.log10(np.where(doses > 0, doses, 1)) + rng.normal(0, 3, doses.size)
    scale = ef.choose_scale(np.unique(doses), "auto")
    out = ef.trend(doses, y, scale, 1000, 1000, rng)
    assert scale == "log" and "left out" in out["note"] and out["n_levels"] == 3
    assert out["slope_ci_lo"] < 20 < out["slope_ci_hi"], out
    assert out["p_perm"] < 0.01
    flat = ef.trend(doses, rng.normal(0, 3, doses.size), "linear", 500, 1000, rng)
    assert flat["p_perm"] > 0.01
    print(f"ok  log-dose slope {out['slope']:.1f} per decade (20), CI "
          f"[{out['slope_ci_lo']:.1f}, {out['slope_ci_hi']:.1f}]")


def test_exponential_time_constant():
    rng = np.random.default_rng(5)
    t = np.repeat([0, 12, 24, 48, 96], 4).astype(float)
    y = 10 + 40 * np.exp(-t / 30) + rng.normal(0, 1.5, t.size)
    out = ef.trend(t, y, "linear", 300, 200, rng, exponential=True)
    assert abs(out["tau"] - 30) < 6 and out["tau_ci_lo"] < 30 < out["tau_ci_hi"], out
    print(f"ok  decay τ {out['tau']:.1f} h (30), CI [{out['tau_ci_lo']:.1f}, {out['tau_ci_hi']:.1f}]")


def test_power_is_finite_for_huge_effects():
    assert ef.power_for(100.0, 1.0, 2, 0.05) == 1.0
    assert 0.0 <= ef.power_for(0.0, 1.0, 2, 0.05) <= 0.06
    print("ok  power: finite in the far tail")


def test_power_table_labels_an_unestimated_experiment_sd():
    vc = {"sd_colony": 8.0, "sd_well": 4.0, "sd_experiment": np.nan}
    prm = ef.settings({"effects": {"power": {"assumed_experiment_sd": 1.0}}})
    rows = pd.DataFrame(ef.power_table(vc, prm, 2, 5, 100.0))
    assert set(rows["experiment_sd_source"].str.split(":").str[0]) >= {"not estimable"}
    lower = rows[rows["sd_experiment_used"] == 0].set_index("experiments")["mdd"]
    assumed = rows[rows["sd_experiment_used"] == 4.0].set_index("experiments")["mdd"]
    assert (assumed > lower).all() and lower.is_monotonic_decreasing
    print("ok  power table: lower bound without experiment variance, larger MDD when assumed")


def test_end_to_end_experiments_doses_and_power():
    """Synthetic images: 3 differentiations × 3 doses × 2 wells × 2 colonies. The
    CENTER boundary moves 15 µm per decade of dose, with experiment offsets of
    ±6 µm and well offsets of ±3 µm. The pipeline must pick experiments as the
    replicate, recover the slope, and report variance components and power."""
    import tempfile
    import main as pipeline_main
    sys.path.insert(0, str(ROOT / "tests"))
    import synthetic as syn
    exp_off = {"E1": -6.0, "E2": 0.0, "E3": 6.0}
    wells, w = {}, 1
    for e, off in exp_off.items():
        for dose in (1, 10, 100):
            for k, well_off in enumerate((-3.0, 3.0)):
                wells[w] = {"experiment": e, "dose_ngmL": dose, "mid_sd": 2.0,
                            "mid": 100 + 15 * np.log10(dose) + off + well_off}
                w += 1
    root = Path(tempfile.mkdtemp())
    cfg = syn.make_experiment(root, wells, colonies_per_well=2, seed=7, artifacts=False,
                              downstream={
        "conditions": {"condition_key": ["dose_ngmL"]},
        "stats": {"experiment_column": "experiment"},
        "profiles": {"colony_figures": False},
        "multimarker": {"colony_figures": False, "well_figures": False},
        "coexpression": {"colony_figures": False, "well_figures": False, "panels": []},
        "boundary": {"n_boot": 40, "roughness": {"enabled": False}},
        "effects": {"n_boot": 400, "n_perm": 400,
                    "power": {"targets": [{"metric": "x50_um", "delta": 20}]}},
    })
    args = ["--config", str(cfg), "--log-level", "WARNING"]
    assert pipeline_main.main(["manifest"] + args) == 0
    assert pipeline_main.main(["downstream"] + args) == 0
    t = root / "analysis" / "tables"
    import json
    units = json.loads((t / "replicate_unit.json").read_text())
    assert next(iter(units.values()))["unit"] == "experiment", units

    key = "marker|CENTER|intensity|x50_um"
    tr = pd.read_csv(t / "effects_trends.csv")
    row = tr[tr["metric_id"] == key].iloc[0]
    assert row["unit"] == "experiment" and row["scale"] == "log" and row["n_units"] == 9, row
    assert row["slope_ci_lo"] < 15 < row["slope_ci_hi"] and abs(row["slope"] - 15) < 4, row

    vc = pd.read_csv(t / "effects_variance.csv").set_index("metric_id").loc[key]
    assert vc["n_experiments"] == 3 and np.isfinite(vc["sd_experiment"]), vc
    assert 0.5 < vc["sd_colony"] < 5, vc                  # truth 2 µm (+ fitting noise)

    eff = pd.read_csv(t / "effects_vs_reference.csv")
    e = eff[(eff["metric_id"] == key) & (eff["condition"].astype(str) == "100")].iloc[0]
    assert str(e["reference"]) == "1" and e["unit"] == "experiment" and e["n"] == 3
    assert e["ci_lo"] < 30 < e["ci_hi"], e                # 2 decades × 15 µm

    pw = pd.read_csv(t / "effects_power.csv")
    p = pw[pw["metric_id"] == key]
    assert set(p["experiment_sd_source"]) == {"estimated"}
    assert p.set_index("experiments")["mdd"].is_monotonic_decreasing
    assert p["power_at_target"].notna().all()
    figs = root / "analysis" / "figures" / "effects"
    assert any(figs.glob("trend_marker_CENTER.png")) and any(figs.glob("power_*.png"))
    print(f"ok  end to end: slope {row['slope']:.1f} µm/decade (15) "
          f"[{row['slope_ci_lo']:.1f}, {row['slope_ci_hi']:.1f}], n = 9 experiment-doses; "
          f"dose 100 − 1 = {e['diff']:.1f} µm (30)")


def test_power_rows_flag_the_permutation_floor():
    """MDD rows say when a permutation test cannot reach alpha (E vs E: floor 2 / C(2E, E))."""
    from downstream import effects as ef_mod
    assert abs(ef_mod.permutation_floor(3) - 0.1) < 1e-12
    assert abs(ef_mod.permutation_floor(4) - 2 / 70) < 1e-12
    prm = ef_mod.settings({})
    prm["power"]["experiments"] = [3, 4, 6]
    vc = {"sd_experiment": 8.0, "sd_well": 5.0, "sd_colony": 10.0}
    rows = {r["experiments"]: r for r in ef_mod.power_table(vc, prm, 3, 5, 50.0)}
    assert not rows[3]["permutation_can_reach_alpha"]
    assert rows[4]["permutation_can_reach_alpha"] and rows[6]["permutation_can_reach_alpha"]
    assert abs(rows[3]["mdd"] - 27.0) < 0.1 and abs(rows[6]["mdd"] - 15.9) < 0.1
    print("ok  power: 3 vs 3 flagged (p floor 0.1); MDD 27.0 / 15.9 as in the compendium")


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
