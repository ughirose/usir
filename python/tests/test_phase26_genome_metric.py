"""Phase 26.2: genome mode precision blocks as G_style (scaled vs legacy must steer differently)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from usir_py.genome_fit import genome_modes
from usir_py.hsmm_longform_director import HSMMLongformDirector
from usir_py.realtime_decoder import ClauseTwistedSMCDecoder

ATLAS = Path(__file__).resolve().parents[2] / "artifacts" / "style_atlas" / "quantile"
SCALED = sorted(ATLAS.glob("*.usir"))
pytestmark = pytest.mark.skipif(not SCALED, reason="quantile atlas genomes not present")


def _pair():
    p = SCALED[0]
    return genome_modes(p), genome_modes(ATLAS / "legacy" / p.name)


def test_three_modes_with_trace_normalised_metric():
    scaled, legacy = _pair()
    assert [m["mode"] for m in scaled] == [0, 1, 2]
    for m in scaled + legacy:
        assert m["G"].shape == (12, 12)
        assert np.isclose(np.trace(m["G"]), 12.0)
        assert np.all(np.linalg.eigvalsh(m["G"]) > 0)          # positive definite


def test_scaled_and_legacy_metrics_differ():
    scaled, legacy = _pair()
    off = lambda G: float(np.abs(G - np.diag(np.diag(G))).max())  # noqa: E731
    assert off(legacy[0]["G"]) < 0.05                      # legacy shrinkage is close to the identity
    assert off(scaled[0]["G"]) > 0.1                       # scaled keeps the author's correlations
    assert not np.allclose(scaled[0]["G"], legacy[0]["G"], atol=1e-3)


def test_director_sets_well_targets_and_metrics():
    scaled, _ = _pair()
    d = HSMMLongformDirector(dim=12, seed=0)
    d.set_genome_modes(scaled)
    for m in scaled:
        assert np.allclose(d.wells[m["mode"]].target_q, m["centroid"][:12])
        assert np.allclose(d.well_G[m["mode"]], m["G"])


def test_smc_style_term_selects_differently_under_scaled_vs_legacy():
    scaled, legacy = _pair()
    rng = np.random.default_rng(0)
    target = scaled[0]["centroid"][:12]
    qs = {c: target + rng.normal(0, 0.3, 12) for c in ("甲。", "乙。", "丙。", "丁。")}
    terms = []
    for modes in (scaled, legacy):
        dec = ClauseTwistedSMCDecoder(dim=12, seed=0, G_style=modes[0]["G"], feature_extractor=lambda t: qs[t[-2:]])
        res = dec.compute_particle_weights("", list(qs), target_q=target)
        terms.append(np.asarray(res["term3_style"]))
    assert not np.allclose(terms[0], terms[1], atol=1e-6)
