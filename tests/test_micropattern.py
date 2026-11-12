"""
Checks for the micropattern analysis: every one against a known answer.

Run with ``python tests/test_micropattern.py`` (or pytest). Synthetic colonies
come from :mod:`synthetic`; nothing touches real data or a GPU. The end-to-end
test builds a 12-colony experiment and runs ``manifest`` + ``downstream`` on it
(~1 minute).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import synthetic as syn  # noqa: E402
from downstream import coexpression, colony, fda, radial, trends  # noqa: E402
from experiment import parser  # noqa: E402

PARAMS = colony.settings({"colony": {"close_radius_um": 15}})


def test_point_index_is_read_from_stitched_names():
    assert parser.parse_seq("Point0042_Channel405,488,561,640_zmax.tif") == 42
    assert parser.parse_seq("POINT0001_CHANNEL405_Seq0007.nd2") == 7     # Seq wins
    assert parser.parse_image("Point0003_stitched.tif", "predefined")["seq_index"] == 3
    print("ok  Point#### (Fiji-stitched) is read as the Seq index; Seq#### wins when both")


def test_circle_fit_is_robust_to_protrusions():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 400)
    pts = np.column_stack([200 + 150 * np.sin(t), 300 + 150 * np.cos(t)])
    pts[:30] += rng.normal(0, 40, (30, 2))                    # cells sticking out of the rim
    fit = colony.fit_circle(pts)
    assert abs(fit["r"] - 150) < 1.5 and abs(fit["cy"] - 200) < 1.5 and abs(fit["cx"] - 300) < 1.5
    arc = pts[30:][(t[30:] > 0) & (t[30:] < np.pi)]           # only half the rim visible
    fit = colony.fit_circle(arc)
    assert abs(fit["r"] - 150) < 1.5
    print("ok  circle fit: exact through outliers and from a half-visible rim")


def test_morphology_matches_disk_closing():
    from scipy import ndimage as ndi
    rng = np.random.default_rng(2)
    mask = rng.random((80, 80)) > 0.93
    yy, xx = np.mgrid[-4:5, -4:5]
    ref = ndi.binary_closing(np.pad(mask, 10), structure=(yy ** 2 + xx ** 2) <= 16)[10:-10, 10:-10]
    ours = colony.disk_close(mask, 4)
    assert (ours[10:-10, 10:-10] == ref[10:-10, 10:-10]).mean() > 0.99
    print("ok  distance-transform closing equals a disk closing")


def _geometry(**kw):
    masks, _, true_dist, _, labels, _, kind = syn.make_colony_image(**kw)
    geo = colony.detect_colony(masks, syn.PIXEL_UM, PARAMS)
    nuc = geo["nuclei"].set_index("label").loc[labels]
    return geo, nuc, true_dist, kind


def test_whole_colony_geometry_and_outliers():
    geo, nuc, true_dist, kind = _geometry(neighbor=(40, 560, 90), seed=3)
    meta = geo["meta"]
    assert meta["edge_method"] == "mask" and meta["border_fraction"] == 0
    assert abs(meta["diameter_um"] - 500) < 15, meta["diameter_um"]
    assert meta["n_neighbour_colonies"] == 1
    inside = nuc["in_colony"].to_numpy()
    assert inside[kind == 0].all() and not inside[kind != 0].any()
    err = nuc["dist_edge_um"].to_numpy()[kind == 0] - true_dist[kind == 0]
    assert abs(np.median(err)) < 8 and np.percentile(np.abs(err), 95) < 15, np.median(err)
    print(f"ok  whole colony: Ø {meta['diameter_um']:.0f} µm (truth 500), strays and the "
          f"neighbor excluded, edge distance median error {np.median(err):+.1f} µm")


def test_clipped_colony_switches_to_the_circle():
    geo, nuc, true_dist, kind = _geometry(center=(300, 120), seed=4)
    meta = geo["meta"]
    assert meta["border_fraction"] > PARAMS["max_border_fraction"]
    assert meta["edge_method"] == "circle"
    assert abs(meta["diameter_um"] - 500) < 15
    sel = kind == 0
    err = nuc["dist_edge_um"].to_numpy()[sel] - true_dist[sel]
    assert np.percentile(np.abs(err), 95) < 15
    # The mask-only answer is what the old pipeline did, and it is wrong near the cut.
    forced = colony.detect_colony(syn.make_colony_image(center=(300, 120), seed=4)[0],
                                  syn.PIXEL_UM, {**PARAMS, "edge_distance": "mask"})
    bad = forced["nuclei"].set_index("label").loc[nuc.index, "dist_edge_um"].to_numpy()[sel]
    assert np.percentile(np.abs(bad - true_dist[sel]), 95) > 30
    print(f"ok  clipped colony ({100 * meta['border_fraction']:.0f}% of rim on the border): "
          f"distances from the circle fit; the mask alone would be off by "
          f"{np.percentile(np.abs(bad - true_dist[sel]), 95):.0f} µm")


def test_tail_of_escaped_cells_is_trimmed():
    masks, _, true_dist, _, labels, _, kind = syn.make_colony_image(n_strays=0, seed=6)
    # A 60-px-wide tongue of cells trailing off the bottom of the colony.
    lab = int(masks.max())
    for y in range(495, 590, 10):
        for x in range(275, 330, 10):
            lab += 1
            masks[y - 4:y + 5, x - 4:x + 5][masks[y - 4:y + 5, x - 4:x + 5] == 0] = lab
    geo = colony.detect_colony(masks, syn.PIXEL_UM, PARAMS)
    meta = geo["meta"]
    assert abs(meta["diameter_um"] - 500) < 15, meta["diameter_um"]
    assert meta["trimmed_fraction"] > 0
    tail = geo["nuclei"].set_index("label").loc[labels.max() + 1:]
    assert (~tail["in_colony"]).mean() > 0.7
    nuc = geo["nuclei"].set_index("label").loc[labels]
    err = nuc["dist_edge_um"].to_numpy()[kind == 0] - true_dist[kind == 0]
    assert abs(np.median(err)) < 8
    print(f"ok  tail of escaped cells: circle stays Ø {meta['diameter_um']:.0f} µm, "
          f"{100 * (~tail['in_colony']).mean():.0f}% of tail cells trimmed out of the colony")


def test_missing_pixel_size_is_an_error():
    d = Path(tempfile.mkdtemp())
    masks, dapi, _, values, labels, cents, _ = syn.make_colony_image(seed=5)
    syn.write_result_dir(d, masks, dapi, values, labels, cents, pixel_um=None)
    try:
        colony.process_image(d, PARAMS)
    except ValueError as exc:
        assert "pixel_size_um" in str(exc)
    else:
        raise AssertionError("missing pixel size must not be guessed")
    meta = colony.process_image(d, {**PARAMS, "pixel_size_um": syn.PIXEL_UM, "overlay": False})
    assert meta["status"] == "ok" and not meta["skipped"]
    again = colony.process_image(d, {**PARAMS, "pixel_size_um": syn.PIXEL_UM, "overlay": False})
    assert again["skipped"], "same settings → checkpoint hit"
    changed = colony.process_image(d, {**PARAMS, "pixel_size_um": syn.PIXEL_UM, "overlay": False,
                                       "close_radius_um": 20})
    assert not changed["skipped"], "changed settings → recomputed"
    print("ok  no pixel size → error (never 1.0 µm/px); per-image checkpoint follows settings")


def test_shared_scale_keeps_level_differences():
    rows = []
    for field, amp in ((0, 1.0), (1, 0.5)):
        d = np.linspace(0, 200, 400)
        rows.append(pd.DataFrame({"field_id": field, "marker": "M", "scope_key": "p",
                                  "value": syn.edge_profile(d, amp), "dist_edge_um": d}))
    long = pd.concat(rows, ignore_index=True)
    anchors = radial.shared_anchors(long, (1, 99))
    long["scaled"] = radial.apply_scale(long, anchors)
    edge = long[long["dist_edge_um"] < 10].groupby("field_id")["scaled"].mean()
    assert 0.45 < edge[1] / edge[0] < 0.55, edge.to_dict()
    print(f"ok  shared scale: a half-amplitude colony reads {edge[1] / edge[0]:.2f}× as high "
          f"(per-colony rescaling would have made it 1.0×)")


def test_permutation_test_is_calibrated_and_powered():
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 20)
    groups = np.repeat(["a", "b", "c"], 5)
    hits = 0
    n_sim = 200
    for _ in range(n_sim):
        curves = np.exp(-3 * x)[None, :] + rng.normal(0, 0.2, (15, 20))
        hits += fda.functional_anova(curves, groups, 199, rng)["global_p"] < 0.05
    assert 0.015 <= hits / n_sim <= 0.10, hits / n_sim
    curves = np.exp(-3 * x)[None, :] + rng.normal(0, 0.2, (15, 20))
    curves[groups == "c", :5] += 0.6                       # differs only near the edge
    res = fda.functional_anova(curves, groups, 999, rng)
    assert res["global_p"] < 0.01
    assert res["sig_mask"][:5].any() and not res["sig_mask"][10:].any()
    print(f"ok  permutation ANOVA: false-positive rate {hits / n_sim:.3f} at α = 0.05; a real "
          f"edge difference found (p = {res['global_p']:.3g}) and localized to the edge")


def test_small_designs_are_enumerated_exactly():
    rng = np.random.default_rng(8)
    curves = rng.normal(0, 1, (6, 10))
    curves[:3] += 5
    res = fda.functional_anova(curves, ["a"] * 3 + ["b"] * 3, 5000, rng)
    assert res["exact"] and res["n_permutations"] == 20
    assert abs(res["global_p"] - 0.1) < 1e-9          # 2 of 20 labelings (a↔b swap) as extreme
    print("ok  2 × 3 design: all 20 labelings enumerated, p = 0.1 is the honest minimum")


def test_replicate_unit_choice():
    one_well = pd.DataFrame({"condition": list("aaabbb"), "well_key": list("111222")})
    unit, note = radial.choose_unit(one_well, {})
    assert unit == "colony" and "single well" in note
    many = pd.DataFrame({"condition": ["a"] * 3 + ["b"] * 3, "well_key": list("123456")})
    assert radial.choose_unit(many, {})[0] == "well"
    assert radial.choose_unit(many, {"stats": {"replicate": "colony"}})[0] == "colony"
    print("ok  replicate = well when every condition has ≥ 3 wells, else colony with a caveat")


def test_local_trend_marks_rises_falls_and_keeps_gaps():
    x = np.arange(10) * 10.0 + 5
    se = np.full(10, 0.01)
    _, z_up = trends.local_trend(x, x / 100.0, se)
    _, z_down = trends.local_trend(x, 1 - x / 100.0, se)
    assert (z_up > 1.96).all() and (z_down < -1.96).all()
    flat = 0.5 + np.random.default_rng(0).normal(0, 0.01, 10)
    _, z_flat = trends.local_trend(x, flat, se)
    assert np.mean(np.abs(z_flat) >= 1.96) <= 0.3
    gap = x / 100.0
    gap[5] = np.nan
    _, z_gap = trends.local_trend(x, gap, se)
    assert np.isnan(z_gap[4]) and np.isnan(z_gap[6]) and np.isfinite(z_gap[2])
    print("ok  local trend: rises > 0, falls < 0, flat mostly unmarked, never bridged over a gap")


def _coexpression_cells(seed=0):
    """Three colonies: A and B share noise within 50 µm of the edge only; each
    colony has its own level offset in both markers (the Simpson trap)."""
    rng = np.random.default_rng(seed)
    cells, values, positive = [], {"A": [], "B": []}, {"A": [], "B": []}
    uid = 0
    for field, offset in enumerate((0.0, 0.4, 0.8)):
        n = 3000
        d = rng.uniform(0, 200, n)
        shared = rng.normal(0, 1, n)
        rim = d < 50
        a = offset + np.where(rim, shared, rng.normal(0, 1, n)) * 0.1
        b = offset + np.where(rim, shared, rng.normal(0, 1, n)) * 0.1
        for k in range(n):
            cells.append({"cell_uid": uid + k, "field_id": field, "dist_edge_um": d[k],
                          "r_norm": d[k] / 200.0, "stain_set": "s1"})
        values["A"] += list(a)
        values["B"] += list(b)
        positive["A"] += list((a > offset + 0.05).astype(float))
        positive["B"] += list((b > offset + 0.05).astype(float))
        uid += n
    cells = pd.DataFrame(cells).set_index("cell_uid", drop=False)
    wide = {"value": pd.DataFrame(values, index=cells.index),
            "positive": pd.DataFrame(positive, index=cells.index)}
    wide["gate"] = wide["value"] * np.nan
    return cells, wide


def test_coexpression_finds_rim_only_coexpression_without_pooling():
    cells, wide = _coexpression_cells()
    prm = coexpression.settings({"coexpression": {"bin_um": 25}})
    pprm = radial.settings({})
    table = coexpression.colony_table(cells, [("A", "B")], wide, prm, pprm)
    um = table[table["grid"] == radial.UM]
    rim = um[um["bin_hi"] <= 50]["correlation"]
    inner = um[um["bin_lo"] >= 50]["correlation"]
    assert (rim > 0.9).all() and (inner.abs() < 0.15).all(), (rim.tolist(), inner.tolist())
    ratio = um[um["bin_hi"] <= 50]["frac_ab"] / um[um["bin_hi"] <= 50]["frac_expected"]
    assert (ratio > 1.5).all()
    # Colonies first: the well mean stays ~0 inside, where pooled nuclei would
    # correlate strongly only because the colonies differ in level.
    table["well_key"] = "w1"
    wells = coexpression.well_table(table, ["well_key"])
    w_inner = wells[(wells["grid"] == radial.UM) & (wells["bin_lo"] >= 50)]["correlation"]
    sel = cells["dist_edge_um"] >= 50
    pooled = coexpression.correlation(wide["value"]["A"][sel], wide["value"]["B"][sel])
    assert (w_inner.abs() < 0.15).all() and pooled > 0.5, (w_inner.tolist(), pooled)
    print(f"ok  co-expression: rim ρ ≥ {rim.min():.2f}, interior |ρ| ≤ {inner.abs().max():.2f}; "
          f"well mean over colonies stays ~0 inside (pooled nuclei would give {pooled:.2f})")


def test_marker_pairs_stay_within_a_panel():
    idx = pd.RangeIndex(6)
    cells = pd.DataFrame({"stain_set": ["p1"] * 3 + ["p2"] * 3}, index=idx)
    nan = np.nan
    values = pd.DataFrame({"S": [1.0] * 6, "A": [1.0, 1, 1, nan, nan, nan],
                           "B": [nan, nan, nan, 1.0, 1, 1]}, index=idx)
    order = ["S", "A", "B"]
    assert coexpression.marker_pairs(values, cells, {}, order) == [("S", "A"), ("S", "B")]
    assert coexpression.marker_pairs(values[["S"]], cells, {}, order) == []
    cfg = {"coexpression": {"pairs": [["A", "B"], ["A", "S"]]}}
    assert coexpression.marker_pairs(values, cells, cfg, order) == [("A", "S")]
    print("ok  pairs: only markers of one panel, none for a single marker, bad explicit pairs dropped")


def test_end_to_end_synthetic_experiment():
    import main as pipeline_main
    root = Path(tempfile.mkdtemp())
    cfg = syn.make_experiment(root)
    args = ["--config", str(cfg), "--log-level", "WARNING"]
    assert pipeline_main.main(["check"] + args) == 0
    assert pipeline_main.main(["manifest"] + args) == 0
    assert pipeline_main.main(["downstream"] + args) == 0
    t = root / "analysis" / "tables"
    qc = pd.read_csv(t / "qc_colonies.csv")
    assert qc.loc[qc["field_id"] == 0, "flags"].iloc[0] == "clipped"
    glob = pd.read_csv(t / "compare_global.csv")
    p = glob[glob["readout"] == "mean"].set_index(["grid", "marker"])["global_p"]
    assert p[("um", "EDGE")] < 0.05 and p[("norm", "EDGE")] < 0.05
    assert p[("um", "FLAT")] > 0.05 and p[("um", "CENTER")] > 0.05
    cond = pd.read_csv(t / "condition_profiles.csv")
    edge = cond[(cond["marker"] == "EDGE") & (cond["grid"] == "um") & (cond["readout"] == "mean")
                & (cond["bin"] == 0)].set_index("condition")["mean"]
    assert 0.4 < edge["lineC"] / edge["lineA"] < 0.6
    assert (root / "analysis" / "report.html").exists()
    assert any((root / "analysis" / "figures" / "colony_profiles").glob("*.png"))
    # Re-running is a no-op; forcing one step re-runs exactly what depends on it.
    import logging
    ck = root / "analysis" / ".checkpoints"
    before = {f.stem: f.stat().st_mtime for f in ck.glob("*.json")}
    assert pipeline_main.main(["downstream"] + args) == 0
    assert {f.stem: f.stat().st_mtime for f in ck.glob("*.json")} == before
    assert pipeline_main.main(["downstream", "--force-step", "profiles"] + args) == 0
    after = {f.stem: f.stat().st_mtime for f in ck.glob("*.json")}
    rerun = {k for k in after if after[k] != before.get(k)}
    assert rerun == {"profiles", "aggregate", "compare", "multimarker", "coexpression",
                     "effects", "signaling_summary", "report"}, rerun
    figs = root / "analysis" / "figures"
    for group in ("multimarker_profiles/colonies", "multimarker_profiles/wells",
                  "coexpression/pages/colonies", "coexpression/pages/wells",
                  "coexpression/panels/wells/scatter", "coexpression/panels/wells/correlation"):
        assert len(list((figs / group).glob("*.png"))) > 0, group
    # Panels are opt-in per level: colonies are off by default.
    assert not list((figs / "coexpression/panels/colonies/scatter").glob("*.png"))
    coex = pd.read_csv(t / "coexpression_colonies.csv")
    assert set(coex["pair"]) == {"CENTER vs EDGE", "CENTER vs FLAT", "EDGE vs FLAT"}
    med = coex[coex["grid"] == "um"].groupby("pair")["correlation"].median()
    # Within a bin, opposite gradients still anti-correlate; FLAT is pure noise.
    assert med["CENTER vs EDGE"] < -0.15 and abs(med["EDGE vs FLAT"]) < 0.05, med.to_dict()
    assert "coexpression-pages-wells" in (root / "analysis" / "report.html").read_text()

    # One analysis marker: no multi-marker figures, no pairs, no stale tables.
    import yaml
    cfg_dict = yaml.safe_load(cfg.read_text())
    cfg_dict["downstream"]["exclude_markers"] = ["DAPI", "CENTER", "FLAT"]
    cfg.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    assert pipeline_main.main(["downstream"] + args) == 0
    for group in ("multimarker_profiles/colonies", "multimarker_profiles/wells",
                  "coexpression/pages/colonies", "coexpression/pages/wells",
                  "coexpression/panels/wells/scatter"):
        assert not list((figs / group).glob("*.png")), group
    for name in ("coexpression_colonies.csv", "coexpression_wells.csv",
                 "multimarker_trends.csv"):
        assert not (t / name).exists(), name
    print("ok  end to end: clipped colony flagged, EDGE differs (µm + normalized), CENTER/FLAT "
          f"do not; lineC/lineA edge level {edge['lineC'] / edge['lineA']:.2f} (truth 0.5)")
    logging.getLogger().handlers.clear()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    failures = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except Exception as exc:                          # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print("\n" + ("all tests passed" if not failures else f"{failures} test(s) failed"))
    sys.exit(1 if failures else 0)
