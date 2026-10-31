r"""Workstream 3 (Gap 5): Style Chemistry & Unexplored Style Discovery Operator (usir-genesis).

Tests:
1. evaluate_style_molecules:
   - Strict activation score bounding in [0, 1]
   - Emergence of the 4 canonical style molecules:
     * Mol_Yugen: Sg-03 Pr x Sg-08 Lp x Sg-11 Ma
     * Mol_Wabi: Sg-04 Jt x Sg-07 Cs x Sg-11 Ma
     * Mol_QiYun: Sg-01 Dn/Gr x Sg-06 Mp/Sy x Sg-10 At gated by momentum tanh(||p||_{G_style})
     * Mol_Iki: Sg-02 Cr x Sg-07 Cs x Sg-12 Ru
   - Momentum circulation gating proof: ||p|| = 0 strongly attenuates Mol_QiYun to 0.0
   - Synesthetic off-diagonal SO(8) metric coupling amplification
   - Synesthetic coupling energy diagnostics
2. discover_voronoi_genesis / usir_genesis:
   - Centroid and regularized shrinkage covariance \hat{\Sigma}_{24}
   - Strict 95% Mahalanobis confidence ellipsoid trust region (d_M^2 <= 36.4150)
   - Box constraints q in [0, 1]^12, p >= 0
   - Voronoi void Riemannian distance maximization (min_dist > initial_centroid_dist)
   - Evaluated molecules on discovered genesis coordinate
"""

import math
import numpy as np
import pytest

from usir_py.v2.style_chemistry import (
    CHI2_24_095,
    discover_voronoi_genesis,
    evaluate_style_molecules,
    usir_genesis,
)
from usir_py.v2.thermodynamics_so8 import SO8PullbackMetric


def test_evaluate_style_molecules_bounds_and_zero_state():
    """Verify bounds in [0, 1] and zero state response."""
    z_zero = np.zeros(24)
    res_zero = evaluate_style_molecules(z_zero)
    assert res_zero.Mol_Yugen == 0.0
    assert res_zero.Mol_Wabi == 0.0
    assert res_zero.Mol_QiYun == 0.0
    assert res_zero.Mol_Iki == 0.0
    assert res_zero.synestheticCouplingEnergy == 0.0
    assert res_zero.dominantMolecule in ["Mol_Yugen", "Mol_Wabi", "Mol_QiYun", "Mol_Iki"]

    # 12D input handling
    q_ones = np.ones(12)
    res_12d = evaluate_style_molecules(q_ones)
    for k in ["Mol_Yugen", "Mol_Wabi", "Mol_Iki"]:
        assert 0.0 <= res_12d[k] <= 1.0
    # Because 12D input has p = 0, Mol_QiYun must be 0.0
    assert res_12d.Mol_QiYun == 0.0


def test_yugen_emergence_and_constituent_pruning():
    """Mol_Yugen emerges from Pr(2) x Lp(7) x Ma(10)."""
    # Active Yugen state
    q = np.zeros(12)
    q[2] = 0.85  # Sg-03 Pr
    q[7] = 0.80  # Sg-08 Lp
    q[10] = 0.90 # Sg-11 Ma
    res = evaluate_style_molecules(q)

    assert res.Mol_Yugen > 0.75
    assert res.dominantMolecule == "Mol_Yugen"
    assert res.Mol_Wabi == 0.0  # Jt(3) and Cs(6) are 0
    assert res.Mol_QiYun == 0.0 # p is 0, Gr(0) is 0
    assert res.Mol_Iki == 0.0   # Cr(1) and Cs(6) are 0

    # Prune one constituent: Lp(7) -> 0
    q_pruned = q.copy()
    q_pruned[7] = 0.0
    res_pruned = evaluate_style_molecules(q_pruned)
    assert res_pruned.Mol_Yugen == 0.0


def test_wabi_emergence():
    """Mol_Wabi emerges from Jt(3) x Cs(6) x Ma(10)."""
    q = np.zeros(12)
    q[3] = 0.85  # Sg-04 Jt
    q[6] = 0.90  # Sg-07 Cs
    q[10] = 0.80 # Sg-11 Ma
    res = evaluate_style_molecules(q)

    assert res.Mol_Wabi > 0.75
    assert res.dominantMolecule == "Mol_Wabi"
    assert res.Mol_Yugen == 0.0 # Pr(2) and Lp(7) are 0
    assert res.Mol_Iki == 0.0   # Cr(1) is 0


def test_iki_emergence():
    """Mol_Iki emerges from Cr(1) x Cs(6) x Ru(11)."""
    q = np.zeros(12)
    q[1] = 0.85  # Sg-02 Cr
    q[6] = 0.80  # Sg-07 Cs
    q[11] = 0.90 # Sg-12 Ru
    res = evaluate_style_molecules(q)

    assert res.Mol_Iki > 0.75
    assert res.dominantMolecule == "Mol_Iki"
    assert res.Mol_Wabi == 0.0 # Jt(3) and Ma(10) are 0


def test_qiyun_momentum_circulation_gating():
    """Mol_QiYun is strongly gated by symplectic momentum circulation tanh(||p||_{G_style})."""
    # High static coordinates for QiYun: Dn/Gr(0), Mp/Sy(5), At(9)
    q = np.zeros(12)
    q[0] = 0.85
    q[5] = 0.85
    q[9] = 0.85

    # Case A: Overdamped static text (p = 0)
    z_static = np.concatenate([q, np.zeros(12)])
    res_static = evaluate_style_molecules(z_static)
    assert res_static.Mol_QiYun == 0.0, "Overdamped static text must have Mol_QiYun == 0"

    # Case B: Active dynamic symplectic circulation (p > 0 along active style flow)
    p_active = np.zeros(12)
    p_active[0] = 1.0
    p_active[5] = 1.0
    p_active[9] = 1.0
    z_dynamic = np.concatenate([q, p_active])
    res_dynamic = evaluate_style_molecules(z_dynamic)

    assert res_dynamic.Mol_QiYun > 0.65
    assert res_dynamic.dominantMolecule == "Mol_QiYun"
    # Living resonance increases monotonically with momentum circulation
    p_high = p_active * 1.5
    z_high = np.concatenate([q, p_high])
    res_high = evaluate_style_molecules(z_high)
    assert res_high.Mol_QiYun >= res_dynamic.Mol_QiYun


def test_so8_metric_coupling_amplification_and_energy():
    """SO(8) off-diagonal coupling amplifies molecule score relative to diagonal baseline."""
    g_full = SO8PullbackMetric().metric_matrix()
    g_diag = np.diag(np.diag(g_full))

    # Test state with active Yugen elements
    q = np.zeros(12)
    q[2] = 0.7
    q[7] = 0.7
    q[10] = 0.7

    res_coupled = evaluate_style_molecules(q, g_style=g_full)
    res_uncoupled = evaluate_style_molecules(q, g_style=g_diag)

    # Coupling amplification: off-diagonal coupling amplifies score
    assert res_coupled.Mol_Yugen >= res_uncoupled.Mol_Yugen
    assert res_coupled.synestheticCouplingEnergy > 0.0
    assert res_uncoupled.synestheticCouplingEnergy == 0.0


def test_voronoi_genesis_trust_region_and_distance_maximization():
    """Operator 4 (usir_genesis) finds Voronoi void inside 95% trust region."""
    np.random.seed(42)
    # 4 known reference particles in 24D
    k_particles = [
        np.array([0.2]*12 + [0.1]*12),
        np.array([0.8]*12 + [0.3]*12),
        np.array([0.1, 0.9]*6 + [0.2]*12),
        np.array([0.7, 0.2]*6 + [0.4]*12),
    ]

    res = usir_genesis(
        k_particles,
        options={
            "n_restarts": 6,
            "n_steps": 40,
            "learning_rate": 0.08,
            "lambda_shrinkage": 0.15,
        }
    )

    # 1. Coordinate dimensions and box bounds
    assert res.z_genesis.shape == (24,)
    assert np.all(res.z_genesis[:12] >= 0.0)
    assert np.all(res.z_genesis[:12] <= 1.0)
    assert np.all(res.z_genesis[12:] >= 0.0)

    # 2. Hard In-Distribution Trust Region (d_M^2 <= 36.4150)
    assert res.inside_trust_region is True
    assert res.mahalanobis_d2 <= CHI2_24_095 + 1e-4

    # 3. Voronoi void isolation: min distance strictly exceeds initial centroid distance
    assert res.min_riemannian_distance > res.initial_centroid_min_dist
    assert res.min_riemannian_distance > 0.5

    # 4. Evaluated molecules attached to genesis coordinate
    assert hasattr(res.molecules, "dominantMolecule")
    assert res.molecules.dominantMolecule in ["Mol_Yugen", "Mol_Wabi", "Mol_QiYun", "Mol_Iki"]
    for mol in ["Mol_Yugen", "Mol_Wabi", "Mol_QiYun", "Mol_Iki"]:
        assert 0.0 <= res.molecules[mol] <= 1.0


def test_voronoi_genesis_small_k_and_alias():
    """Operator 4 works for small K=3 and discover_voronoi_genesis == usir_genesis."""
    p1 = np.ones(24) * 0.3
    p2 = np.ones(24) * 0.7
    p3 = np.array([0.2, 0.8] * 12)

    res1 = discover_voronoi_genesis([p1, p2, p3])
    assert res1.inside_trust_region is True
    assert res1.min_riemannian_distance > res1.initial_centroid_min_dist

    # Verify alias
    res2 = usir_genesis([p1, p2, p3])
    assert res2.inside_trust_region is True
