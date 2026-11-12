"""
Checks for the signaling module, each against a known answer.

Run with ``python tests/test_signaling.py`` (or pytest). Synthetic images only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from downstream.data import NC_RATIO, NUC_BGSUB, RAW, Data  # noqa: E402
from downstream.signaling import catalog, measure, reference  # noqa: E402

CFG = {"signaling": {}}


def test_spellings_are_recognized_and_reported():
    for name in ("b-CAT", "β-catenin", "beta catenin", "bCat", "CTNNB1", "B-Catenin",
                 "Active b-catenin"):
        m = catalog.classify(name, CFG)
        assert m is not None and m.entry == "beta-catenin" and m.readout == "nuclear", name
    for name in ("SMAD2/3", "Smad2-3", "SMAD 2/3", "smad2+3"):
        assert catalog.classify(name, CFG).entry == "SMAD2/3", name
    for name in ("SMAD4", "Smad4", "SMAD2", "SMAD1"):
        assert catalog.classify(name, CFG).readout == "nc", name
    # Qualifiers are dropped, and the match says so.
    m = catalog.classify("b-CAT-r", CFG)
    assert m.entry == "beta-catenin" and m.dropped == ("r",)
    assert catalog.classify("SMAD4 (rb)", CFG).entry == "SMAD4"
    assert catalog.classify("SMAD2/3-488", CFG).entry == "SMAD2/3"
    # Fate markers, and phospho forms, are not signaling markers.
    for name in ("OTX2-r", "SOX2", "TBXT", "H2B", "pSMAD1", "pSMAD2"):
        assert catalog.classify(name, CFG) is None, name
    assert catalog.near_misses(["pSMAD1", "SOX2", "SMAD158", "b-CAT"], CFG) == ["pSMAD1",
                                                                              "SMAD158"]
    assert catalog.spelling_groups(["b-CAT", "B-cat", "b-CAT-r", "SOX2"], CFG) == \
        [["B-cat", "b-CAT"]]
    print("ok  signaling names match whatever the spelling; phospho forms do not")


def test_config_adds_and_overrides():
    cfg = {"signaling": {"markers": {"pSMAD1": "nuclear", "SMAD4": "none", "YAP1": "nc"}}}
    assert catalog.classify("pSMAD1", cfg).readout == "nuclear"
    assert catalog.classify("psmad-1", cfg).how == "config"       # same key as the entry
    assert catalog.classify("SMAD4", cfg) is None
    assert catalog.classify("YAP1", cfg).readout == "nc"
    assert catalog.classify("b-CAT", {"signaling": {"enabled": False}}) is None
    assert catalog.near_misses(["pSMAD1"], cfg) == []
    print("ok  signaling.markers adds, overrides and switches off entries")


def _cells_image(shape=(160, 160), nuc=500.0, cyto=300.0, bg=100.0):
    """A patch of square cells (nucleus + cytoplasm) on a cell-free surface."""
    labels = np.zeros(shape, np.int32)
    img = np.full(shape, bg, np.float32)
    k = 0
    for y0 in range(40, 112, 18):
        for x0 in range(40, 112, 18):
            k += 1
            img[y0 - 3:y0 + 15, x0 - 3:x0 + 15] = cyto        # cell body around the nucleus
    k = 0
    for y0 in range(40, 112, 18):
        for x0 in range(40, 112, 18):
            k += 1
            labels[y0:y0 + 12, x0:x0 + 12] = k
            img[y0:y0 + 12, x0:x0 + 12] = nuc
    img += np.random.default_rng(0).normal(0, 4, shape).astype(np.float32)   # camera noise
    return labels, img


def test_nc_ratio_matches_the_construction():
    labels, img = _cells_image()
    prm = dict(measure.DEFAULTS, ring_inner_um=1.0, ring_outer_um=3.0, free_distance_um=12.0,
               min_free_px=100)
    table, bg, _ = measure.measure(labels, {1: img}, {1}, 1.0, prm)
    assert bg[1]["method"] == "local" and abs(bg[1]["background"] - 100) < 1
    expected = (500 - 100) / (300 - 100)
    assert np.allclose(table["nc_ratio"], expected, rtol=0.03), table["nc_ratio"].describe()
    assert np.allclose(table["nuc_bgsub"], 400, rtol=0.03)
    # A ring reaching past the cell bodies (5 px; bodies end at 3 px) touches the empty
    # surface around the outer cells. The cell-body mask drops those pixels, so N:C
    # stays exact; without it the outer cells' rings hold background pixels.
    wide = dict(prm, ring_outer_um=5.0)
    strict = measure.measure(labels, {1: img}, {1}, 1.0, wide)[0]
    loose = measure.measure(labels, {1: img}, {1}, 1.0, dict(wide, body_k=0))[0]
    assert np.allclose(strict["nc_ratio"], expected, rtol=0.03)
    assert (loose["n_ring_px"] > strict["n_ring_px"]).any()
    assert (loose["n_ring_px"] >= strict["n_ring_px"]).all()
    # A nuclear-only channel gets no ring.
    t2, _, _ = measure.measure(labels, {1: img}, set(), 1.0, prm)
    assert t2["nc_ratio"].isna().all() and np.allclose(t2["nuc_bgsub"], 400, rtol=0.03)
    print("ok  N:C = (nucleus - background) / (ring - background); empty surface left out")


def test_ring_grows_in_plane_for_3d_masks():
    labels, img = _cells_image()
    lab3 = np.stack([np.zeros_like(labels), labels, np.zeros_like(labels)])
    img3 = np.stack([np.full_like(img, 100), img, np.full_like(img, 100)])
    ring = measure.ring_labels(lab3, 1, 3)
    assert ring[0].max() == 0 and ring[2].max() == 0 and ring[1].max() > 0
    prm = dict(measure.DEFAULTS, ring_inner_um=1.0, ring_outer_um=3.0, free_distance_um=12.0,
               min_free_px=100)
    table, _, _ = measure.measure(lab3, {1: img3}, {1}, 1.0, prm)
    assert np.allclose(table["nc_ratio"], 2.0, rtol=0.03)
    print("ok  3-D: the ring stays in the nucleus' own planes")


def _vignetted_colony(shape=(400, 400), r_um=80.0):
    """A disk colony of nuclei on a surface whose background falls from 200 at the
    center to ~100 in the corners; a patch inside the colony has bright signal but
    no segmented nuclei (as in a colony that segmented poorly)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    cy, cx = shape[0] / 2, shape[1] / 2
    rr = np.hypot(yy - cy, xx - cx)
    img = (100 + 100 * np.exp(-(rr / 160) ** 2)).astype(np.float32)
    labels = np.zeros(shape, np.int32)
    k = 0
    for y0 in range(int(cy - r_um), int(cy + r_um), 10):
        for x0 in range(int(cx - r_um), int(cx + r_um), 10):
            if np.hypot(y0 + 3 - cy, x0 + 3 - cx) > r_um - 4:
                continue
            if abs(y0 - cy + 30) < 22 and abs(x0 - cx) < 22:    # unsegmented patch
                img[y0:y0 + 6, x0:x0 + 6] = 5000
                continue
            k += 1
            labels[y0:y0 + 6, x0:x0 + 6] = k
            img[y0:y0 + 6, x0:x0 + 6] += 1000
    img[(rr < r_um) & (labels == 0)] += 300               # cytoplasm between nuclei
    colony = rr < r_um
    return labels, img, colony, rr


def test_local_background_follows_the_colony_surroundings():
    labels, img, colony, rr = _vignetted_colony()
    prm = dict(measure.DEFAULTS, free_distance_um=12.0, local_band_um=40.0, min_free_px=500)
    table, bg, _ = measure.measure(labels, {1: img}, set(), 1.0, prm, colony)
    # Truth: the surface level in the band 12-52 µm beyond the colony edge.
    band = (rr > 80 + 12) & (rr <= 80 + 12 + 40)
    truth = float(np.median(img[band]))
    err = abs(bg[1]["background"] - truth)
    assert bg[1]["method"] == "local" and err < 5, (bg, truth)
    # The image-wide median (of the measured crop) is pulled down by the dimmer surface
    # farther out.
    img_bg = measure.measure(labels, {1: img}, set(), 1.0, dict(prm, background="image"),
                             colony)[1][1]
    assert img_bg["method"] == "image" and img_bg["background"] < truth - max(2 * err, 5)
    # The bright, unsegmented patch inside the colony is never background (filled hole),
    # even without a colony mask.
    no_mask = measure.measure(labels, {1: img}, set(), 1.0, prm)[1][1]
    assert abs(no_mask["background"] - truth) < 10, no_mask
    regions = measure.background_masks(labels, img > 0, 12.0, 40.0, None)
    assert not (regions["image"] & (rr < 60)).any()
    # Too few band pixels → the image-wide level, then the camera offset.
    few = measure.measure(labels, {1: img}, set(), 1.0, dict(prm, min_free_px=10 ** 5),
                          colony)[1][1]
    assert few["method"] in ("image", "offset")
    assert np.allclose(table["nuc_bgsub"], table["nuc"] - bg[1]["background"])
    print("ok  local background: the band around the colony, not the corners or the gaps")


def test_background_table_flags_an_outlier_image(tmp_path=None):
    import json
    import tempfile
    root = Path(tmp_path or tempfile.mkdtemp())
    levels = [700, 710, 690, 705, 1400, 698]
    entries = []
    for i, lv in enumerate(levels):
        d = root / f"img{i}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / measure.META_NAME, "w") as f:
            json.dump({"background": {"1": {"background": lv, "mad": 40, "method": "local",
                                            "n_free_px": 5000}}}, f)
        entries.append(("ds", d))
    entries.append(("ds", root / "missing"))
    table = measure.background_table(entries)
    assert len(table) == len(levels)
    assert table.loc[table["outlier"], "background"].tolist() == [1400]
    # A second stain set that is uniformly brighter is not an outlier of the first.
    for i, lv in enumerate([1300, 1320, 1290]):
        d = root / f"other{i}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / measure.META_NAME, "w") as f:
            json.dump({"background": {"1": {"background": lv, "mad": 40, "method": "local",
                                            "n_free_px": 5000}}}, f)
    grouped = [(ds, p, "A") for ds, p in entries] + \
        [("ds", root / f"other{i}", "B") for i in range(3)]
    table = measure.background_table(grouped)
    assert table.loc[table["outlier"], "background"].tolist() == [1400]
    print("ok  background table: one image far from the rest is flagged")


def _data(values_by_field, reference_fields):
    rows, cells = [], []
    uid = 0
    for field, vals in values_by_field.items():
        for v in vals:
            cells.append({"cell_uid": uid, "field_id": field, "class":
                          "pluripotent" if field in reference_fields else "diff"})
            rows.append({"cell_uid": uid, "marker": "SMAD2/3", "channel": 1, RAW: v,
                         "dapi_norm": np.nan, NUC_BGSUB: np.nan, NC_RATIO: v})
            uid += 1
    return Data(cells=pd.DataFrame(cells), intensities=pd.DataFrame(rows))


def test_reference_normalization_uses_colony_medians():
    # Reference colonies with medians 1.0 and 3.0 (one is huge): reference = 2.0.
    data = _data({0: [1.0] * 500, 1: [3.0] * 5, 2: [4.0] * 50}, reference_fields={0, 1})
    cfg = {"conditions": {"control_column": "class", "control_values": ["pluripotent"]}}
    assert data.space_for("SMAD2/3", cfg) == NC_RATIO
    long = data.intensities[["cell_uid", "marker"]].copy()
    long["value"] = data.values(data.intensities, cfg)
    long["field_id"] = data.cells["field_id"].to_numpy()[long["cell_uid"]]
    long["scope_key"] = "p1"
    matches = catalog.classify_all(["SMAD2/3"], cfg)
    table = reference.reference_table(long, data.cells, cfg, matches)
    assert abs(table["reference"].iloc[0] - 2.0) < 1e-9 and bool(table["relative"].iloc[0])
    rel = reference.relative_values(long, table, pd.Series(np.zeros(len(long)),
                                                           index=long.index))
    assert np.allclose(rel[long["field_id"] == 2], 2.0)
    labels = reference.unit_labels(table)
    assert labels["SMAD2/3"][0] == "SMAD2/3 N:C (÷ pluripotent)"
    # No reference wells: N:C stays the plain ratio, never a mix of units.
    cfg2 = {"conditions": {"control_column": "class", "control_values": ["nope"]}}
    t2 = reference.reference_table(long, data.cells, cfg2, matches)
    assert not bool(t2["relative"].iloc[0])
    rel2 = reference.relative_values(long, t2, pd.Series(np.zeros(len(long)), index=long.index))
    assert np.allclose(rel2, long["value"])
    print("ok  reference = median of reference colonies' medians; no reference, no mixing")


def test_nuclear_readout_without_reference_stays_in_its_own_units():
    rows, cells = [], []
    for uid, (field, v) in enumerate([(0, 1000.0), (0, 1200.0), (1, 3000.0)]):
        cells.append({"cell_uid": uid, "field_id": field, "class": "pluripotent"})
        rows.append({"cell_uid": uid, "marker": "b-CAT", "channel": 1, RAW: v,
                     "dapi_norm": np.nan, NUC_BGSUB: v - 100, NC_RATIO: np.nan})
    data = Data(cells=pd.DataFrame(cells), intensities=pd.DataFrame(rows))
    # reference_values: [] = no normalization wanted, even with pluripotent wells present
    cfg = {"conditions": {"control_column": "class", "control_values": ["pluripotent"]},
           "signaling": {"reference_values": [], "nuclear_space": "raw"}}
    assert data.space_for("b-CAT", cfg) == RAW
    long = data.intensities[["cell_uid", "marker"]].copy()
    long["value"] = data.values(data.intensities, cfg)
    long["field_id"] = data.cells["field_id"].to_numpy()[long["cell_uid"]]
    long["scope_key"] = "p1"
    table = reference.reference_table(long, data.cells, cfg,
                                      catalog.classify_all(["b-CAT"], cfg))
    assert not bool(table["relative"].iloc[0])
    rel = reference.relative_values(long, table, pd.Series(np.zeros(len(long)),
                                                           index=long.index))
    assert np.allclose(rel, [1000.0, 1200.0, 3000.0])          # raw a.u., not 0-1
    assert reference.unit_labels(table)["b-CAT"][0] == "nuclear b-CAT (raw, a.u.)"
    assert reference.absolute_markers(table) == ["b-CAT"]
    print("ok  reference_values: [] keeps a nuclear readout in a.u., labeled as such")


def test_signaling_marker_without_measurement_falls_back_to_raw():
    data = _data({0: [1.0, 2.0]}, reference_fields=set())
    data.intensities[NC_RATIO] = np.nan
    assert data.space_for("SMAD2/3", {}) == RAW
    print("ok  a signaling marker whose step has not run is read raw, with a warning")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
