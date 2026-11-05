"""M17 parity: general-constraint WorldCraftArbitrator (SVD) + director momentum accounting.

Asserts against packages/usir-core/tests/fixtures/worldcraft_arbitrator_parity.json, which is the
same file the TS vitest (worldcraft-parity.test.ts) asserts against.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from usir_py.hsmm_longform_director import HSMMLongformDirector
from usir_py.nishiki_unified_field import WorldCraftArbitrator

FIXTURE = Path(__file__).resolve().parents[2] / "packages" / "usir-core" / "tests" / "fixtures" / "worldcraft_arbitrator_parity.json"
FX = json.loads(FIXTURE.read_text(encoding="utf-8"))
TOL = 1e-6
I12 = np.eye(12)


@pytest.mark.parametrize("name", list(FX["constraint_sets"]))
def test_projector_matches_fixture_and_properties(name):
    cs = FX["constraint_sets"][name]
    j = WorldCraftArbitrator.build_general_jacobian(cs["rows"])
    p = WorldCraftArbitrator.compute_null_space_projector_svd(j)
    assert np.allclose(p, np.array(cs["p_null_svd"]), atol=TOL)
    assert np.allclose(WorldCraftArbitrator.compute_null_space_projector(j), np.array(cs["p_null_tikhonov"]), atol=TOL)
    assert np.allclose(p, p.T, atol=1e-12)
    assert np.allclose(p @ p, p, atol=1e-12)
    if j.shape[0]:
        assert np.allclose(j @ p, 0.0, atol=1e-12)
    assert int(round(12 - np.trace(p))) == cs["rank"] == (np.linalg.matrix_rank(j) if j.shape[0] else 0)


@pytest.mark.parametrize("name", list(FX["constraint_sets"]))
def test_arbitration_matches_fixture(name):
    cs = FX["constraint_sets"][name]
    j = WorldCraftArbitrator.build_general_jacobian(cs["rows"])
    svd = WorldCraftArbitrator(projector_method="svd")
    for m, exp in zip(FX["momenta"], cs["arbitration"]):
        safe, cut, _ = svd.arbitrate_momentum(np.array(m), j)
        assert np.allclose(safe, exp["svd_safe"], atol=TOL)
        assert np.allclose(cut, exp["svd_cut"], atol=TOL)
        assert np.allclose(safe + cut, m, atol=1e-12)


def test_unit_rows_legacy_default_unchanged():
    """Explicit tikhonov path on unit rows is the closed form I - (1/(1+eps)) sum e_i e_i^T; default is svd."""
    j = np.array(FX["constraint_sets"]["unit_evidence"]["rows"])
    p = WorldCraftArbitrator.compute_null_space_projector(j)
    expected = I12.copy()
    for i in (7, 11):
        expected[i, i] = 1.0 - 1.0 / (1.0 + 1e-6)
    assert np.allclose(p, expected, atol=1e-12)
    assert WorldCraftArbitrator().projector_method == "svd"
    assert WorldCraftArbitrator(projector_method="tikhonov").projector_method == "tikhonov"
    dq = np.arange(1.0, 13.0)
    safe_def, _, _ = WorldCraftArbitrator().arbitrate_momentum(dq, j)
    assert np.max(np.abs(j @ safe_def)) < 1e-12
    # SVD on unit rows is the exact coordinate projector
    ps = WorldCraftArbitrator.compute_null_space_projector_svd(j)
    exact = I12.copy()
    exact[7, 7] = exact[11, 11] = 0.0
    assert np.allclose(ps, exact, atol=1e-12)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        WorldCraftArbitrator.compute_null_space_projector_svd(np.ones((2, 5)))
    with pytest.raises(ValueError):
        WorldCraftArbitrator.compute_null_space_projector_svd(np.ones((13, 12)))
    with pytest.raises(ValueError):
        WorldCraftArbitrator(projector_method="bogus")


def test_director_step_sequence_matches_fixture():
    cfg = FX["director"]["config"]
    d = HSMMLongformDirector(
        transition_matrix=np.array(cfg["transition_matrix"]),
        refractory_cooldown=cfg["refractory_cooldown"],
        seed=0,
        arbitrator=WorldCraftArbitrator(projector_method=cfg["projector_method"]),
    )
    for rec in FX["director"]["steps"]:
        op, out = rec["op"], rec["out"]
        if op == "arbitrate_rows":
            safe, cut = d.arbitrate_with_jacobian(np.array(rec["target_p"]), WorldCraftArbitrator.build_general_jacobian(rec["rows"]))
            assert np.allclose(safe, out["safe_p"], atol=TOL) and np.allclose(cut, out["cut_p"], atol=TOL)
        elif op == "accumulate":
            delta = d.accumulate_unreleased_momentum(np.array(rec["target_p"]), np.array(rec["measured_p"]))
            assert np.allclose(delta, out["delta"], atol=TOL)
        elif op == "step":
            assert d.step_transition(rec["current_well"]) == out["next_well"]
        elif op == "discharge":
            boosted, discharged = d.discharge_momentum(np.array(rec["base_climax_p"]))
            assert np.allclose(boosted, out["boosted_p"], atol=TOL) and np.allclose(discharged, out["discharged_p"], atol=TOL)
        st = rec["state"]
        assert np.allclose(d.p_banked, st["p_banked"], atol=TOL)
        assert d.refractory_counter == st["refractory_counter"]
        assert d.get_banked_energy() == pytest.approx(st["banked_energy"], abs=TOL)
    # p_banked never negative (max(0, cut) accounting)
    assert np.all(d.p_banked >= 0.0)
