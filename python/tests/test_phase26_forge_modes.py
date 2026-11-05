"""Phase 26.1: quantile 3-mode assignment for the Tier-M forge.

Python <-> TS parity fixture: packages/usir-core/tests/fixtures/quantile_modes_parity.json
Regenerate (deterministic): cd python && uv run python tests/test_phase26_forge_modes.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from usir_py import style_atlas as sa
from usir_py.usir_forge import (
    QUANTILE_MIN_WINDOWS,
    assign_modes_quantile,
    load_tier_m_binary,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "packages" / "usir-core" / "tests" / "fixtures" / "quantile_modes_parity.json"
)


def _corpus(n: int = 200, seed: int = 2606):
    rng = np.random.default_rng(seed)
    Z = np.abs(rng.normal(0.5, 0.15, size=(n, 24)))
    d = rng.random(n) * 0.05
    return Z, d


def test_fractions_all_modes_at_least_15_percent():
    for n in (20, 21, 37, 100, 200, 1001):
        Z, d = _corpus(n, seed=n)
        m = assign_modes_quantile(Z, d)
        assert m is not None and len(m) == n
        for k in range(3):
            assert (m == k).sum() / n >= 0.15, (n, k)
    Z, d = _corpus(200)
    m = assign_modes_quantile(Z, d)
    assert (m == 1).sum() == 50 and (m == 2).sum() == 38 and (m == 0).sum() == 112


def test_mode_semantics():
    Z, d = _corpus(200)
    m = assign_modes_quantile(Z, d)
    assert d[m == 1].min() >= d[m != 1].max()  # Dialogue = top by the dialogue signal
    assert m[np.argmax(d)] == 1

    def shock(Z_):
        return (Z_[:, 3] + Z_[:, 7] + Z_[:, 11]) / 3.0

    assert shock(Z[m == 2]).mean() > shock(Z[m == 0]).mean()


def test_determinism_and_ties_by_index():
    Z, d = _corpus(60)
    assert np.array_equal(assign_modes_quantile(Z, d), assign_modes_quantile(Z.copy(), d.copy()))
    # all-equal dialogue signal: ties resolved by smaller index -> first 25% are Dialogue
    m = assign_modes_quantile(Z, np.zeros(60))
    assert set(np.where(m == 1)[0]) == set(range(15))


def test_fallback_below_min_windows():
    Z, d = _corpus(QUANTILE_MIN_WINDOWS - 1)
    assert assign_modes_quantile(Z, d) is None
    # distill_genome falls back to the rule: identical bytes to mode_assign="rule"
    works = [("w", Z)]
    a = sa.distill_genome(works, mode_assign="rule")
    b = sa.distill_genome(works, mode_assign="quantile", dialogue={"w": d})
    assert a == b
    with pytest.raises(ValueError):
        sa.distill_genome(works, mode_assign="bogus")


def test_rule_default_unchanged():
    Z, _ = _corpus(80)
    assert sa.distill_genome([("w", Z)]) == sa.distill_genome([("w", Z)], mode_assign="rule")


def test_binary_decodes_with_three_distinct_centroids():
    # two works so that work boundaries are exercised
    Z1, d1 = _corpus(90, seed=1)
    Z2, d2 = _corpus(110, seed=2)
    for shrink in ("scaled", "legacy"):
        blob = sa.distill_genome([("a", Z1), ("b", Z2)], shrinkage=shrink, mode_assign="quantile",
                                 dialogue={"a": d1, "b": d2})
        assert len(blob) == 4096
        dec = load_tier_m_binary(blob)
        w = np.array(dec["weights"])
        assert abs(w.sum() - 1.0) < 1e-5 and (w >= 0.15).all()
        assert w[1] == pytest.approx(0.25, abs=0.01)
        C = np.array(dec["centroids"])
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.linalg.norm(C[i] - C[j]) > 1e-3
        gm = sa.Gmm.from_binary(blob)
        assert gm.active == [0, 1, 2]
        assert sa.audit_cleanroom_binary(blob, "これは検査用の本文であり、バイナリには含まれない。" * 3)["passed"]


def test_quote_ratios_windows():
    text = ("「こんにちは」と彼は言った。" * 30) + ("静かな夜であった。" * 200)
    r = sa.quote_ratios(text)
    assert len(r) == len(sa.make_windows(text))
    assert r[0] > 0.0 and r[-1] == 0.0


def _build_fixture() -> dict:
    cases = []
    for n, seed in ((20, 1), (57, 2), (200, 3)):
        Z, d = _corpus(n, seed)
        d = np.round(d, 2)  # introduce ties
        m = assign_modes_quantile(Z, d)
        cases.append({"Z": Z.tolist(), "dialogue": d.tolist(), "modes": m.tolist()})
    return {"cases": cases}


def test_parity_fixture_matches_python():
    assert FIXTURE.exists(), "regenerate: cd python && uv run python tests/test_phase26_forge_modes.py"
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for c in fx["cases"]:
        m = assign_modes_quantile(np.array(c["Z"]), c["dialogue"])
        assert m.tolist() == c["modes"]


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(_build_fixture()) + "\n", encoding="utf-8")
    print("wrote", FIXTURE)
