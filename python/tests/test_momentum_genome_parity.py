"""M18.3 parity: momentum_from_summary conditional-Gaussian path from a Tier-M genome.

Asserts against packages/usir-core/tests/fixtures/momentum_genome_parity.json, the same file the TS
vitest (packages/usir-bridges/tests/momentum-genome-parity.test.ts) asserts against.
Regenerate (deterministic, seeded): cd python && uv run python tests/test_momentum_genome_parity.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from usir_py.nishiki_unified_field import EQ_FLOOR, momentum_from_summary
from usir_py.usir_forge import OSWEStreamingCompiler, load_tier_m_binary

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "packages" / "usir-core" / "tests" / "fixtures" / "momentum_genome_parity.json"
)
RTOL = 1e-5


def _build_fixture() -> dict:
    rng = np.random.default_rng(1803)
    comp = OSWEStreamingCompiler()
    mode_q = [np.full(12, 0.25), np.full(12, 0.5), np.full(12, 0.8)]
    mode_p = [np.full(12, 0.1), np.full(12, 0.3), np.full(12, 0.6)]
    mix = rng.normal(0.0, 0.15, size=(12, 12))
    for ci in range(5):
        for k in range(3):
            for _ in range(40):
                q = mode_q[k] + rng.normal(0.0, 0.08, 12)
                p = np.abs(mode_p[k] + 0.4 * (mix @ (q - mode_q[k])) + rng.normal(0.0, 0.03, 12))
                comp.add_sample(np.concatenate([q, p]), mode=k, cluster_id=f"c{ci}")
    blob = comp.compile_tier_m_binary()
    qs = [mode_q[0] + 0.03, mode_q[1] - 0.04, 0.5 * (mode_q[1] + mode_q[2]) + rng.normal(0.0, 0.05, 12)]
    out = {
        "tier_m_b64": base64.b64encode(blob).decode("ascii"),
        "cases": [],
    }
    g = load_tier_m_binary(blob)
    for q in qs:
        out["cases"].append({"q": [float(v) for v in q], "p": [float(v) for v in momentum_from_summary(q, genome=g)]})
    return out


FX = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else None
needs_fx = pytest.mark.skipif(FX is None, reason="fixture not generated")


@needs_fx
@pytest.mark.parametrize("i", range(3))
def test_genome_matches_fixture_bytes_and_decoded(i):
    blob = base64.b64decode(FX["tier_m_b64"])
    case = FX["cases"][i]
    q = np.array(case["q"])
    exp = np.array(case["p"])
    assert np.allclose(momentum_from_summary(q, genome=blob), exp, rtol=RTOL, atol=1e-9)
    assert np.allclose(momentum_from_summary(q, genome=load_tier_m_binary(blob)), exp, rtol=RTOL, atol=1e-9)
    assert np.all(exp >= EQ_FLOOR)


@needs_fx
def test_responsibility_selects_nearer_mode():
    g = load_tier_m_binary(base64.b64decode(FX["tier_m_b64"]))
    q_lo = np.array(g["centroids"][0][:12])
    q_hi = np.array(g["centroids"][2][:12])
    p_lo = momentum_from_summary(q_lo, genome=g)
    p_hi = momentum_from_summary(q_hi, genome=g)
    assert np.allclose(p_lo, np.maximum(g["centroids"][0][12:], EQ_FLOOR), atol=1e-3)
    assert np.allclose(p_hi, np.maximum(g["centroids"][2][12:], EQ_FLOOR), atol=1e-3)
    assert not np.allclose(p_lo, p_hi, atol=0.1)


@needs_fx
def test_invalid_modes_skipped_and_all_invalid_falls_back():
    g = load_tier_m_binary(base64.b64decode(FX["tier_m_b64"]))
    q = np.array(FX["cases"][0]["q"])
    only0 = dict(g, weights=[g["weights"][0], 0.0, float("nan")])
    near = momentum_from_summary(q, genome=only0)
    assert np.all(np.isfinite(near))
    none = dict(g, weights=[0.0, 0.0, 0.0])
    assert np.array_equal(momentum_from_summary(q, genome=none), momentum_from_summary(q))


def test_genome_none_unchanged():
    q = np.linspace(0.1, 0.9, 12)
    from usir_py.nishiki_unified_field import EQ_SCALE, _EQ_LAMBDA

    g = _EQ_LAMBDA @ (q - 0.5)
    legacy = EQ_SCALE * np.sqrt(g * g + EQ_FLOOR * EQ_FLOOR)
    assert np.array_equal(momentum_from_summary(q), legacy)
    assert np.array_equal(momentum_from_summary(q, genome=None), legacy)
    proxy = np.zeros(12)
    proxy[0] = 0.4
    expect = legacy.copy()
    expect[0] = np.hypot(0.4, legacy[0])
    assert np.array_equal(momentum_from_summary(q, proxy), expect)


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(_build_fixture(), indent=1) + "\n", encoding="utf-8")
    print("wrote", FIXTURE)
