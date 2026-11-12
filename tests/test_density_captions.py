"""
Checks for nuclear density per radial bin, bin areas, and normalization captions,
each against a known answer. Synthetic data only.

Run with ``python tests/test_density_captions.py`` (or pytest).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from downstream import captions, radial  # noqa: E402


def test_ring_areas_disk_and_mask_agree():
    R, px = 200.0, 0.5
    edges = np.array([0, 20, 50, 100, 200, 250], float)
    disk = radial.ring_areas(None, edges, radius_um=R)
    yy, xx = np.mgrid[-R:R:px, -R:R:px]
    rr = np.hypot(yy, xx)
    dist = (R - rr)[rr < R]                       # pixel distance to the edge
    mask = radial.ring_areas(dist, edges, px)
    assert np.allclose(mask, disk, rtol=0.01, atol=50), (mask, disk)
    assert disk[-1] == 0 and np.isclose(disk.sum(), np.pi * R ** 2)
    print("ok  ring areas: mask histogram = analytic annuli of a disk")


def test_density_recovers_a_radial_gradient():
    # Nuclei at density 4 per 1000 µm² in the outer 100 µm, 2 inside, R = 300 µm.
    rng = np.random.default_rng(1)
    R = 300.0
    rows = []
    uid = 0
    for field in (0, 1):
        for lo, hi, dens in ((0, 100, 4.0), (100, 300, 2.0)):
            area = np.pi * ((R - lo) ** 2 - (R - hi) ** 2)
            n = rng.poisson(dens * area / 1000)
            # uniform over the annulus: r² uniform between the inner and outer radii
            r = np.sqrt(rng.uniform((R - hi) ** 2, (R - lo) ** 2, n))
            for d in R - r:
                for marker in ("A", "B"):             # each nucleus once per marker
                    rows.append({"cell_uid": uid, "field_id": field, "marker": marker,
                                 "dist_edge_um": d})
                uid += 1
    long = pd.DataFrame(rows)
    edges = np.arange(0, 301, 50.0)
    out = radial.density_curves(long, edges, lambda f, e: radial.ring_areas(None, e, radius_um=R))
    assert set(out["marker"]) == {radial.DENSITY}
    mean = out.groupby("bin")["mean"].mean()
    assert np.allclose(mean.iloc[:2], 4.0, rtol=0.1), mean
    assert np.allclose(mean.iloc[2:], 2.0, rtol=0.15), mean
    # each nucleus counted once, not once per marker
    assert out["n_nuclei"].sum() == uid
    print("ok  density per bin: 4 → 2 nuclei per 1000 µm² recovered")


def test_quantile_bins_report_area_and_density():
    rng = np.random.default_rng(2)
    R = 250.0
    n = 4000
    r = np.sqrt(rng.uniform(0, R ** 2, n))        # uniform density 1000·n/(πR²)
    long = pd.DataFrame({"cell_uid": np.arange(n), "field_id": 0, "marker": "A",
                         "dist_edge_um": R - r, "scaled": rng.normal(size=n),
                         "positive": np.nan})
    prm = dict(radial.DEFAULTS, colony_binning="quantile", colony_n_bins=10)
    binned = radial.colony_binned(long, prm,
                                  edge_area=lambda f, e: radial.ring_areas(None, e, radius_um=R))
    truth = 1000.0 * n / (np.pi * R ** 2)
    widths = binned["bin_hi"] - binned["bin_lo"]
    assert widths.iloc[-1] > 3 * widths.iloc[0]       # equal-count bins widen inward
    assert np.allclose(binned["density"], truth, rtol=0.15), binned["density"].describe()
    print("ok  quantile bins: widths grow toward the center, density stays flat")


def test_captions_and_histone_guard():
    assert captions.nuclear_stain_kind(["H2B-RFP"], {}) == captions.HISTONE
    assert captions.nuclear_stain_kind(["DAPI"], {}) == captions.DNA
    assert captions.nuclear_stain_kind(["H2B"], {"nuclear_stain": "dna"}) == captions.DNA
    assert captions.nuclear_stain_kind(["Hoechst"], {}) == captions.DNA
    spaces = {"X": "raw", "Y": "dapi_norm", "S": "nuc_bgsub"}
    assert captions.histone_denominator_markers(spaces, captions.HISTONE) == ["Y"]
    assert captions.histone_denominator_markers(spaces, captions.DNA) == []
    text = captions.normalization_caption(spaces, {"X": "plate"}, {"S": "S (a.u.)"},
                                          ["H2B"], captions.HISTONE, "line Z").replace("\n", " ")
    assert "X: raw nuclear median, P1–P99 → 0–1 per plate" in text
    assert "histone label" in text and "S: S (a.u.)" in text and text.endswith("line Z")
    print("ok  captions name each normalization; histone denominators are caught")


def test_saturated_nuclei_are_found_per_channel_and_image():
    from downstream import qc
    from downstream.data import RAW, Data
    # 3 images × 4 nuclei; markers A, B; nuclear channel "H2B". Ceilings: 65535 in images
    # 0-1, 4095 in image 2 (a 12-bit camera declared through the extraction QC).
    cells = pd.DataFrame({"cell_uid": np.arange(12), "field_id": np.repeat([0, 1, 2], 4),
                          "dapi": [500.0] * 12})
    cells.loc[5, "dapi"] = 65535.0                      # clipped nucleus in the nuclear channel
    rows = []
    for uid in range(12):
        for m in ("A", "B"):
            rows.append({"cell_uid": uid, "marker": m, RAW: 1000.0})
    inten = pd.DataFrame(rows)
    inten.loc[(inten.cell_uid == 1) & (inten.marker == "A"), RAW] = 65535.0
    inten.loc[(inten.cell_uid == 9) & (inten.marker == "B"), RAW] = 4095.0
    inten.loc[(inten.cell_uid == 2) & (inten.marker == "B"), RAW] = 4095.0   # fine at 16 bit
    qc_rows = [{"field_id": f, "marker": m, "saturation_value": 4095.0 if f == 2 else 65535.0}
               for f in (0, 1, 2) for m in ("A", "B", "H2B")]
    data = Data(cells=cells, intensities=inten)
    hits = qc.saturated_nuclei(data, pd.DataFrame(qc_rows), {})
    assert sorted(hits["cell_uid"]) == [1, 5, 9], hits
    assert set(hits.loc[hits.cell_uid == 5, "channel_name"]) == {"nuclear"}
    # An explicit 12-bit ceiling flags nucleus 2 as well.
    hits12 = qc.saturated_nuclei(data, pd.DataFrame(qc_rows),
                                 {"qc": {"saturation_ceiling": 4095}})
    assert 2 in set(hits12["cell_uid"])
    # No channel QC and no override: nothing can be judged.
    assert qc.saturated_nuclei(data, None, {}).empty
    print("ok  saturated nuclei: clipped medians found per channel and per image ceiling")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
