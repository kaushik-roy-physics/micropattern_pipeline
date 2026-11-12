"""
Border roughness and angular registration separate a ragged sharp border, an
off-center pattern and a truly graded transition (synthetic colonies).

Run with ``python -m pytest tests/`` or plain ``python tests/test_roughness.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from downstream import boundary as bd  # noqa: E402
from downstream import roughness as rg  # noqa: E402

K_MAX = bd.LN81 / 5.0


def _colony(wobble=0.0, width=20.0, dx=0.0, seed=0, n=3000, radius=320.0):
    """Edge-high marker whose border is 100 µm in from the pattern's edge.
    *wobble*: border radius ± wobble·sin 4θ; *dx*: pattern center offset (µm)."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(-np.pi, np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    rp, thp = np.hypot(x - dx, y), np.arctan2(y, x - dx)
    border = radius - 100 - wobble * np.sin(4 * thp)
    p = 1 / (1 + np.exp(-bd.LN81 / width * (rp - border)))
    pos = (rng.uniform(size=n) < p).astype(float)
    return radius - r, th, pos, np.where(pos > 0, 0.8, 0.1)


def _measure(**kw):
    d, th, pos, val = _colony(**kw)
    x = np.arange(5, 320, 10.0)
    idx = np.clip((d // 10).astype(int), 0, x.size - 1)
    n = np.bincount(idx, minlength=x.size).astype(float)
    k = np.bincount(idx, weights=pos, minlength=x.size)
    x50, c = bd.fit_logistic(x, n, k, K_MAX)
    m, sectors, prof = rg.colony_roughness(d, th, pos, val, x50, c,
                                           dict(rg.settings({}), n_boot=50), K_MAX)
    return m, sectors, prof


def test_round_sharp_border_is_smooth_and_unchanged_by_registration():
    m, sectors, prof = _measure()
    assert m["n_sectors_ok"] == 16 and m["roughness_um"] < 5, m
    assert 0.75 < m["width_ratio"] < 1.25, m
    assert set(prof["frame"]) == {"registered", "unregistered"}
    print(f"ok  round border: roughness {m['roughness_um']:.1f} µm, ratio {m['width_ratio']:.2f}")


def test_wandering_border_reads_as_roughness_and_registration_sharpens_it():
    m, _, _ = _measure(wobble=30.0)
    truth = 30.0 / np.sqrt(2)                       # SD of 30·sin 4θ
    assert abs(m["roughness_um"] - truth) < 6, m
    assert m["width_ratio"] < 0.65, m               # the 'gradient' was the wandering
    assert m["_se"]["roughness_um"] > 0
    print(f"ok  wandering border: roughness {m['roughness_um']:.1f} (truth {truth:.1f}), "
          f"width ratio {m['width_ratio']:.2f}")


def test_off_center_pattern_is_decentering_not_roughness():
    m, _, _ = _measure(dx=25.0)
    assert abs(m["decenter_um"] - 25) < 6 and m["roughness_um"] < 5, m
    print(f"ok  off-center: decenter {m['decenter_um']:.1f} (25), roughness {m['roughness_um']:.1f}")


def test_truly_graded_transition_survives_registration():
    m, _, _ = _measure(width=60.0)
    assert m["width_ratio"] > 0.85 and m["roughness_um"] < 8, m
    print(f"ok  graded: width ratio {m['width_ratio']:.2f}")


def test_harmonic_roughness_subtracts_fitting_noise():
    th = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    rng = np.random.default_rng(0)
    x = 100 + rng.normal(0, 5, th.size)             # pure fitting noise, se = 5
    mean, dec, rough, raw = rg.harmonic_roughness(th, x, np.full(th.size, 5.0))
    assert abs(mean - 100) < 4 and rough < 3 and raw > 3, (mean, rough, raw)
    print("ok  noise-only sectors: roughness ≈ 0")


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
