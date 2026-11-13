"""Tests for USIR v5 Stage B-10: Gate 3a Scene-Level Denominator Redesign & Scale Separation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

from gate3a_register_centering_eval import (
    GATE3A_STATUS,
    GATE3B_STATUS,
    BENCHMARK_64_PATH,
    OUTPUT_B10_JSON_PATH,
    CONT24_DIMS,
    load_benchmark_records,
    extract_aozora_106_atlas,
    compute_eta_squared,
    diagonal_mahalanobis,
    regularized_mahalanobis,
)


def test_gate_status_constants():
    """Verify that Gate 3a and Gate 3b statuses strictly adhere to repository rules."""
    assert GATE3A_STATUS == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert GATE3B_STATUS == "GATE3B_FROZEN"


def test_scene_variance_calculations_and_rank_deficiency():
    """Test scene-level within-register variance calculations and rank-deficiency diagnostic."""
    rng = np.random.default_rng(42)
    n_r = 16
    d = 24
    X = rng.normal(loc=2.0, scale=1.5, size=(n_r, d))

    # 1. In-sample variance with 1e-6 floor
    var_in = np.maximum(np.var(X, axis=0, ddof=1), 1e-6)
    assert var_in.shape == (d,)
    assert np.all(var_in >= 1e-6)

    # 2. LOO univariate variance and shrinkage variance
    var_pool = np.maximum(np.var(X, axis=0, ddof=1) * 1.2, 1e-6)
    for i in range(n_r):
        others = np.delete(X, i, axis=0)
        assert len(others) == 15
        var_loo = np.maximum(np.var(others, axis=0, ddof=1), 1e-6)
        var_shrink = np.maximum(0.8 * var_loo + 0.2 * var_pool, 1e-6)
        assert var_loo.shape == (d,)
        assert var_shrink.shape == (d,)
        assert np.all(var_loo >= 1e-6)
        assert np.all(var_shrink >= 1e-6)

    # 3. Covariance rank deficiency: d=24 > nu=14
    m = n_r - 1  # 15
    nu = m - 1   # 14 degrees of freedom
    assert d > nu, f"Dimension {d} must exceed degrees of freedom {nu}"

    # Verify that regularized covariance LOO on rank-deficient sample inflates distances
    records = load_benchmark_records(BENCHMARK_64_PATH)
    r4_records = [r for r in records if r["register"] == "R4"]
    X24_r4 = np.array([r["profile"]["ortho_v2_31d"] for r in r4_records])[:, CONT24_DIMS]
    thresh_c2_95 = float(np.sqrt(stats.chi2.ppf(0.95, df=d)))

    d_cov_r4 = []
    for i in range(len(X24_r4)):
        others = np.delete(X24_r4, i, axis=0)
        mu_loo = np.mean(others, axis=0)
        cov_loo = np.cov(others, rowvar=False)
        d_val = regularized_mahalanobis(X24_r4[i], mu_loo, cov_loo, shrinkage=0.15)
        d_cov_r4.append(d_val)

    cov_outlier_pct = float(np.mean(np.array(d_cov_r4) > thresh_c2_95) * 100.0)
    assert cov_outlier_pct >= 75.0, (
        f"Rank-deficient sample covariance should yield >=75% false outliers, got {cov_outlier_pct}%"
    )
    assert cov_outlier_pct == 81.25


def test_condition_a_and_b_behavior():
    """Test Condition (a) and Condition (b) behavior on tier_c_modern_benchmark_expanded.jsonl."""
    records = load_benchmark_records(BENCHMARK_64_PATH)
    assert len(records) == 64
    labels = [r["register"] for r in records]
    registers = ["R1", "R2", "R3", "R4"]
    reg_indices = {r: [i for i, lab in enumerate(labels) if lab == r] for r in registers}

    X24 = np.array([r["profile"]["ortho_v2_31d"] for r in records])[:, CONT24_DIMS]
    aozora_atlas = extract_aozora_106_atlas()
    thresh_3s_atlas_24 = aozora_atlas["thresh_3sigma_24"]
    mu_aozora_24 = aozora_atlas["mean_24"]

    # 1. Condition (a) R4 outlier rate under scene scaling drops vs B-9 work-level Atlas (81.25%)
    r4_idx = reg_indices["R4"]
    X24_r4 = X24[r4_idx]
    var_scene_r4 = np.maximum(np.var(X24_r4, axis=0, ddof=1), 1e-6)

    d_cond_a_in = [diagonal_mahalanobis(X24_r4[i], mu_aozora_24, var_scene_r4) for i in range(len(X24_r4))]
    out_a_in_3s = float(np.mean(np.array(d_cond_a_in) > thresh_3s_atlas_24) * 100.0)
    # Under in-sample scene variance, R4 3-sigma outlier rate drops to 0.0% (vs 81.25% in B-9)
    assert out_a_in_3s == 0.0, f"Expected 0.0% 3sigma outlier rate under scene scale, got {out_a_in_3s}%"

    # 2. Condition (b) eta^2_register elimination
    eta2_raw = compute_eta_squared(X24, labels)
    assert abs(eta2_raw - 0.4965) < 0.001

    X24_within_c = np.zeros_like(X24)
    for r in registers:
        idx = reg_indices[r]
        sub = X24[idx]
        for i in range(len(idx)):
            others = [j for j in range(len(idx)) if j != i]
            X24_within_c[idx[i]] = sub[i] - np.mean(sub[others], axis=0)

    eta2_within_c = compute_eta_squared(X24_within_c, labels)
    assert eta2_within_c < 1e-6, f"Expected eta^2 -> 0.000000, got {eta2_within_c}"

    # 3. Distance structure preservation (Spearman rho = 1.0000 for LOO centering)
    for r in registers:
        idx = reg_indices[r]
        sub = X24[idx]
        sub_c = X24_within_c[idx]
        n_r = len(idx)
        pdist_raw = []
        pdist_c = []
        for i in range(n_r):
            for j in range(i + 1, n_r):
                pdist_raw.append(float(np.linalg.norm(sub[i] - sub[j])))
                pdist_c.append(float(np.linalg.norm(sub_c[i] - sub_c[j])))
        sp_corr, _ = stats.spearmanr(pdist_raw, pdist_c)
        assert abs(sp_corr - 1.0) < 1e-4, f"Register {r} Spearman rho expected 1.0, got {sp_corr}"


def test_b10_json_artifact_schema_and_values():
    """Verify schema, fixed H1/H2 ratings, and quantification in the B-10 JSON artifact."""
    assert OUTPUT_B10_JSON_PATH.exists(), f"Missing {OUTPUT_B10_JSON_PATH}"
    data = json.loads(OUTPUT_B10_JSON_PATH.read_text(encoding="utf-8"))

    # Metadata & Gate Status
    assert data["metadata"]["gate_3a_status"] == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert data["metadata"]["gate_3b_status"] == "GATE3B_FROZEN"
    assert data["gate_3a_official_decision"]["status"] == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert data["gate_3b_official_decision"]["status"] == "GATE3B_FROZEN"

    # Task 2 Fixed H1/H2 Ratings
    fixed_ratings = data["task2_fixed_h1_h2_ratings"]
    h1 = fixed_ratings["h1_micro_scale"]
    h2 = fixed_ratings["h2_macro_scale"]

    # All 6 spaces must be significant vs other authors after Bonferroni
    for space_name, space_dict in {**h1, **h2}.items():
        assert space_dict["vs_other"]["significant_after_bonferroni"] is True
        assert space_dict["vs_other"]["bonferroni_p_two"] < 0.05
        # None of the spaces survive two-sided Bonferroni vs random
        assert space_dict["vs_random"]["significant_after_bonferroni"] is False

    # In H2 5D continuous core, vs-random effect completely disappears
    core_5d = h2["4pole_cont_plus_hw25"]
    assert core_5d["vs_random"]["p_two_sided"] == pytest.approx(0.4060, abs=0.001)
    assert core_5d["vs_random"]["bonferroni_p_two"] == 1.0

    # Reference-only Scheme 2 disclaimer
    assert "同義反復的補正" in data["reference_only_scheme_2"]["disclaimer"]

    # R4 quantification
    r4_quant = data["b10_scene_scale_redesign_evaluation"]["r4_nominal_level_exceedance_quantification"]
    assert r4_quant["1_scale_ratio_scene_vs_work_23_active_dims"]["active_dims_count"] == 23
    assert r4_quant["1_scale_ratio_scene_vs_work_23_active_dims"]["mean_scale_ratio"] == pytest.approx(1.4233, abs=0.005)
    assert r4_quant["2_finite_sample_loo_student_t_inflation"]["theoretical_d2_inflation_factor"] == pytest.approx(1.2444, abs=0.001)
    assert r4_quant["4_single_nonzero_loo_variance_collapse_31d"]["affected_sample"] == "R4_16"
    assert r4_quant["4_single_nonzero_loo_variance_collapse_31d"]["unregularized_loo_distance"] == pytest.approx(52.7863, abs=0.01)
