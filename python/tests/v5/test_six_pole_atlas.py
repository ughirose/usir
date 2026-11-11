"""Unit tests for USIR v5 Six-Pole Integrated Atlas (Stage B-4).

Validates:
1. NarrativeRGNormalizer: extracts 15 scale-invariant deep features without NaN/Inf.
2. normalized_hsic: symmetric, unit diagonal, near zero for independent random vectors.
3. svd_lowdin_orthogonalize: Frobenius orthogonality error < 1e-6.
4. GoldilocksVoidFinder: Wundt hedonic function and empty sphere discovery logic.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "python"))

from usir_v5_six_pole_atlas import (
    NarrativeRGNormalizer,
    normalized_hsic,
    svd_lowdin_orthogonalize,
    GoldilocksVoidFinder,
    POLE_NAMES,
)

PROFILE_PATH = ROOT / "docs" / "handoff" / "v10" / "nishiki_profile_106works.json"


@pytest.fixture(scope="module")
def profile_data():
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    pytest.skip("Profile JSON not available")


def test_narrative_rg_normalizer_features(profile_data):
    """1. Test NarrativeRGNormalizer feature extraction and scale invariance."""
    normalizer = NarrativeRGNormalizer(profile_data)
    assert len(normalizer.DEEP_FEATURE_NAMES) == 15

    # Test single work extraction
    sample_wid = profile_data["works"][0]["work_id"]
    feats = normalizer.extract_work_features(sample_wid)
    assert feats.shape == (15,)
    assert np.all(np.isfinite(feats))

    # Test all 106 works
    wids = [w["work_id"] for w in profile_data["works"]]
    X_all = normalizer.extract_all(wids)
    assert X_all.shape == (106, 15)
    assert np.all(np.isfinite(X_all))

    # Check that densities and rates are within plausible bounds
    # hook_survival in [0, 1]
    assert np.all(X_all[:, 0] >= 0.0) and np.all(X_all[:, 0] <= 1.0)
    # phi_zeami in [0, 1]
    assert np.all(X_all[:, 7] >= 0.0) and np.all(X_all[:, 7] <= 1.0)


def test_normalized_hsic_properties():
    """2. Test mathematical invariants of normalized HSIC."""
    np.random.seed(42)
    N = 100
    X = np.random.normal(0, 1, size=(N, 4))
    Y = np.random.normal(0, 1, size=(N, 4))

    # Self-nHSIC must be 1.0
    nhsic_self = normalized_hsic(X, X)
    assert abs(nhsic_self - 1.0) < 1e-5, f"Self-nHSIC should be 1.0, got {nhsic_self}"

    # Independent random Gaussian vectors must have near-zero nHSIC (< 0.15)
    nhsic_indep = normalized_hsic(X, Y)
    assert nhsic_indep < 0.15, f"Independent nHSIC should be < 0.15, got {nhsic_indep}"

    # Symmetry
    nhsic_XY = normalized_hsic(X, Y)
    nhsic_YX = normalized_hsic(Y, X)
    assert abs(nhsic_XY - nhsic_YX) < 1e-6, "nHSIC must be symmetric"


def test_svd_lowdin_orthogonalization():
    """3. Test SVD-Lowdin symmetric orthogonalization error < 1e-6."""
    np.random.seed(2026)
    N, D = 106, 6
    # Create correlated data
    A = np.random.normal(0, 1, size=(D, D))
    cov = A @ A.T + 0.5 * np.eye(D)
    Z_raw = np.random.multivariate_normal(mean=np.zeros(D), cov=cov, size=N)

    # Standardize
    Z_std = (Z_raw - np.mean(Z_raw, axis=0)) / np.std(Z_raw, axis=0)
    Z_orth, W_lowdin = svd_lowdin_orthogonalize(Z_std)

    # Check dimensions
    assert Z_orth.shape == (N, D)
    assert W_lowdin.shape == (D, D)

    # Check Frobenius orthogonality error || (1/N) Z_orth^T Z_orth - I_D ||_F < 1e-6
    sample_cov = (Z_orth.T @ Z_orth) / N
    error = np.linalg.norm(sample_cov - np.eye(D), "fro")
    assert error < 1e-6, f"SVD-Lowdin orthogonality error {error:.2e} exceeds 1e-6"


def test_goldilocks_void_finder_logic():
    """4. Test GoldilocksVoidFinder Wundt potential and void candidate scoring."""
    # Synthetic 20 works, 3 authors, 6D canonical space
    np.random.seed(123)
    N = 30
    D = 6
    Z_orth = np.random.normal(0, 1, size=(N, D))
    # Make orthogonal
    Z_orth, _ = svd_lowdin_orthogonalize(Z_orth)

    aids = ["A1", "A2", "A3"]
    labels = {"A1": "作家1", "A2": "作家2", "A3": "作家3"}
    works = []
    for i in range(N):
        aid = aids[i % 3]
        works.append({"id": f"w_{i}", "author_id": aid, "title": f"作品{i}"})

    ppm_surprisals = {w["id"]: 4.65 + np.random.normal(0, 0.2) for w in works}
    breath_intervals = {w["id"]: 22.0 + np.random.normal(0, 3.0) for w in works}

    finder = GoldilocksVoidFinder(
        Z_orth=Z_orth,
        works=works,
        aids=aids,
        ppm_surprisals=ppm_surprisals,
        breath_intervals=breath_intervals,
        labels=labels,
    )

    # Test Wundt potential
    # Optimal at S = 4.65 -> Wundt should be high (> 0.8)
    w_opt = finder.wundt_hedonic_potential(4.65)
    assert 0.8 <= w_opt <= 1.0, f"Expected high Wundt at 4.65, got {w_opt}"

    # Extreme low / high S should be low
    w_low = finder.wundt_hedonic_potential(3.20)
    w_high = finder.wundt_hedonic_potential(6.00)
    assert w_low < 0.2, f"Expected low Wundt at 3.20, got {w_low}"
    assert w_high < 0.2, f"Expected low Wundt at 6.00, got {w_high}"

    # Test void candidate search
    voids = finder.find_top_voids(n_candidates=300, top_k=3)
    assert isinstance(voids, list)
    for v in voids:
        assert len(v["center_orth"]) == D
        assert v["empty_radius"] >= 0.85
        assert v["boundary_author_count"] >= 3
        assert 0.0 <= v["goldilocks_score"] <= 1.0


def test_order0_and_order1_calibration_artifacts():
    """5. Test Order 0 (D0) Calibration & Order 1 (U6) artifacts integrity."""
    cal_path = ROOT / "docs" / "handoff" / "v10" / "usir_v5_instrument_calibration.json"
    if not cal_path.exists():
        pytest.skip("Calibration JSON not yet generated")
    data = json.loads(cal_path.read_text(encoding="utf-8"))

    # Metamorphic tests
    m1 = data["metamorphic_calibration"]["M1_comma_deletion"]
    assert m1["max_deep_pole_abs_error"] < 1e-6
    assert m1["kasane_surface_delta"] > 0.0

    m2 = data["metamorphic_calibration"]["M2_entity_substitution"]
    assert m2["max_abs_error"] < 1e-6

    m3 = data["metamorphic_calibration"]["M3_scene_order_shuffle"]
    assert m3["sadame_e_inter_mean_max_error"] < 1e-6
    assert m3["awase_mean_max_error"] < 1e-6

    # Noise floors
    floors = data["noise_floors"]
    for pole in ["Kasane", "Sasae", "Kamae", "Sadame", "Nagare", "Awase"]:
        assert pole in floors
        assert floors[pole]["sigma_noise"] > 0.0
        assert floors[pole]["level3_gate_2sigma"] > floors[pole]["sigma_noise"]

    # U6 Filter
    u6 = data["order1_u6_polish_separation"]
    assert u6["f1"] >= 0.95
    assert u6["precision"] >= 0.95


def test_order2_and_order3_diachronic_artifacts():
    """6. Test Order 2 (Gate 3a) & Order 3 (D10 / Section 6) artifacts integrity."""
    dia_path = ROOT / "docs" / "handoff" / "v10" / "usir_v5_gate3a_and_diachronic.json"
    if not dia_path.exists():
        pytest.skip("Gate 3a / Diachronic JSON not yet generated")
    data = json.loads(dia_path.read_text(encoding="utf-8"))

    # Gate 3a
    g3a = data["order2_gate3a_distribution_shift"]
    assert "meiji_in_domain" in g3a
    assert "modern_colloquial" in g3a
    assert g3a["modern_colloquial"]["closed_class_coverage_pct"] >= 50.0

    # D10 ANOVA
    d10 = data["order3_d10_multi_translator_anova"]
    assert d10["hypothesis_supported"] is True
    assert d10["spearman_rho"] >= 0.70

    # Section 6 Change Point Detection
    s6 = data["order3_section6_diachronic_trajectories"]
    errors = [v["absolute_year_error"] for v in s6.values()]
    assert max(errors) <= 5, f"Expected max change point year error <= 5, got {max(errors)}"
    assert np.mean(errors) <= 3.5, f"Expected mean error <= 3.5, got {np.mean(errors)}"

