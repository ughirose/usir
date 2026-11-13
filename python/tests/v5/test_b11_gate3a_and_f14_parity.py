"""Tests for USIR v5 Stage B-11: Gate 3a Finite-Sample Calibration & F-14 Hybrid Parity."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

from gate3a_register_centering_eval import (
    BENCHMARK_64_PATH,
    CONT24_DIMS,
    GATE3A_STATUS,
    GATE3B_STATUS,
    OUTPUT_B10_JSON_PATH,
    OUTPUT_B11_JSON_PATH,
    build_b11_threshold_calibration_comparison,
    compute_b10_mc_quantiles,
    evaluate_b10_scene_scale_redesign,
    extract_aozora_106_atlas,
    get_or_compute_b10_mc_quantiles,
    load_benchmark_records,
    TASK1_STAGE_JSON_PATH,
)
from scale_separated_level3 import DEFAULT_NISHIKI_EXE, evaluate_f14_hybrid_alignment_and_parity
from usir_py.v5.f14_hybrid_noise_floor import (
    CANONICAL_F14_NOISE_FLOOR_PATH,
    F14_SIGMA_PROVENANCE,
    KASANE8_ACTIVE_DIM_NAMES,
    KASANE11_DIM_NAMES,
    SHORT_DEAD_DIMS,
    evaluate_f14_hybrid_pair,
    is_eight_dim,
    load_f14_noise_floor,
    select_length_band,
    split_sentences_kasane,
    window_count_kasane,
)


def test_b11_gate_status_and_non_conflation_invariants():
    """Governance: Gate 3a remains SHIFT_QUANTIFIED_NOT_PASSED and Gate 3b remains GATE3B_FROZEN."""
    assert GATE3A_STATUS == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert GATE3B_STATUS == "GATE3B_FROZEN"
    assert "adjacent_paragraph_diff_sd" == F14_SIGMA_PROVENANCE["sigma_kind"]
    assert "sigma_scene_r" in F14_SIGMA_PROVENANCE["non_conflation_guard"]


def test_b10_mc_quantiles_exact_reproducibility():
    """Verify exact B-10 §4 finite-sample MC 95%/99% quantiles (seed=42, 5000 trials)."""
    records = load_benchmark_records(BENCHMARK_64_PATH)
    r4_records = [r for r in records if r["register"] == "R4"]
    X24_r4 = np.array([r["profile"]["ortho_v2_31d"] for r in r4_records])[:, CONT24_DIMS]

    mc = compute_b10_mc_quantiles(X24_r4, seed=42, n_trials=5000)
    indep = mc["independent_gaussian_24d"]
    r4corr = mc["r4_correlated_gaussian_24d"]

    assert mc["theoretical_d2_inflation_factor"] == pytest.approx(56.0 / 45.0, abs=1e-5)
    assert indep["mean_d2"] == pytest.approx(29.8831, abs=0.01)
    assert indep["exceedance_rate_chi2_p95_pct"] == pytest.approx(22.29, abs=0.05)
    assert indep["mc_p95_d"] == pytest.approx(6.9267, abs=0.005)
    assert indep["mc_p99_d"] == pytest.approx(7.7032, abs=0.005)

    assert r4corr["mean_d2"] == pytest.approx(28.6274, abs=0.01)
    assert r4corr["exceedance_rate_chi2_p95_pct"] == pytest.approx(24.51, abs=0.05)
    assert r4corr["mc_p95_d"] == pytest.approx(7.6574, abs=0.005)
    assert r4corr["mc_p99_d"] == pytest.approx(9.0381, abs=0.005)


def test_b11_old_vs_new_thresholds_on_64_benchmark():
    """Verify calibrated MC thresholds vs old chi2 thresholds under canonical sigma2_scene_r (24D)."""
    records_64 = load_benchmark_records(BENCHMARK_64_PATH)
    atlas = extract_aozora_106_atlas()
    b10_eval = evaluate_b10_scene_scale_redesign(records_64, atlas, TASK1_STAGE_JSON_PATH)
    comp = build_b11_threshold_calibration_comparison(b10_eval)

    assert comp["canonical_space"] == "ortho24_cont"
    assert comp["canonical_denominator"] == "diagonal_sigma2_scene_r"
    assert comp["gate_3a_status"] == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert comp["gate_3b_status"] == "GATE3B_FROZEN"
    assert "同義反復的補正" in comp["scheme2_governance_note"]
    assert "NOT counted as Gate 3a pass evidence" in comp["scheme3_governance_note"]

    # Condition (a) uncentered: R4 false outlier rate drops from 25.0% -> 6.25% / 0.0% while R1-R3 stay outliers
    ca_shrink = comp["condition_a_uncentered_scene_scaled"]["shrinkage_loo"]
    assert ca_shrink["R4"]["old_chi2_p95_pct"] == 25.0
    assert ca_shrink["R4"]["new_mc_indep_p95_pct"] == 6.25
    assert ca_shrink["R4"]["new_mc_indep_p99_pct"] == 0.0
    assert ca_shrink["R4"]["new_mc_r4corr_p95_pct"] == 0.0
    assert ca_shrink["R4"]["new_mc_r4corr_p99_pct"] == 0.0
    assert ca_shrink["R1_to_R3_pooled"]["new_mc_indep_p95_pct"] == 100.0
    assert ca_shrink["R1_to_R3_pooled"]["new_mc_indep_p99_pct"] == 95.83
    assert ca_shrink["R2"]["new_mc_r4corr_p99_pct"] == 100.0
    assert ca_shrink["R3"]["new_mc_r4corr_p99_pct"] == 100.0

    ca_in = comp["condition_a_uncentered_scene_scaled"]["in_sample"]
    assert ca_in["R4"]["old_chi2_p95_pct"] == 6.25
    assert ca_in["R4"]["new_mc_indep_p95_pct"] == 0.0
    assert ca_in["R4"]["new_mc_r4corr_p95_pct"] == 0.0
    assert ca_in["R1_to_R3_pooled"]["new_mc_indep_p99_pct"] == 100.0

    # Condition (b) Scheme 3 LOO centered (diagnostic only): false exceedances vanish under MC quantiles
    cb_shrink = comp["condition_b_scheme3_loo_centered_diagnostic_only"]["shrinkage_loo"]
    assert cb_shrink["R4"]["old_chi2_p95_pct"] == 31.25
    assert cb_shrink["R4"]["new_mc_indep_p95_pct"] == 6.25
    assert cb_shrink["R4"]["new_mc_r4corr_p95_pct"] == 0.0
    assert cb_shrink["R1_to_R3_pooled"]["new_mc_indep_p95_pct"] == 0.0


def test_f14_window_count_and_hybrid_rule_mechanics():
    """Verify Rust-parity sentence splitting, window counting, Band M 8D/11D split, and ac_multi sigma."""
    nf_data, prov = load_f14_noise_floor()
    assert Path(CANONICAL_F14_NOISE_FLOOR_PATH).exists()
    assert len(prov["sha256"]) == 64
    assert prov["n_works_in_file"] == 216  # F-22: default switched from F-17 (46) to 216 canonical
    assert prov["n_pairs_in_file"] == 52812  # F-17 was 10859

    assert len(KASANE11_DIM_NAMES) == 11
    assert len(SHORT_DEAD_DIMS) == 3
    assert len(KASANE8_ACTIVE_DIM_NAMES) == 8
    assert "ac_mean_crest" in KASANE8_ACTIVE_DIM_NAMES
    assert "ac_mean_velocity" not in KASANE8_ACTIVE_DIM_NAMES

    # Exact sentence closer behavior (。\n stays attached to slice)
    s_multi = split_sentences_kasane("第一文。\n第二文。「会話。」末尾")
    assert s_multi == ["第一文。\n", "第二文。", "「会話。」", "末尾"]

    # Hybrid rule: Band M (260 chars) single-window vs single-window -> 8D!
    single_260 = "あ" * 260 + "。"  # 261 chars, 1 sentence => 1 window, Band M (251..=500)
    assert window_count_kasane(single_260) == 1
    assert is_eight_dim((1, 1)) is True
    ev_m8 = evaluate_f14_hybrid_pair(single_260, single_260, nf_data, provenance=prov)
    assert ev_m8["band"] == "M"
    assert ev_m8["n_evaluated"] == 8
    assert ev_m8["n_eval_rule"] == "both_single_window:8"
    assert ev_m8["ac_sigma_source"] == "not_applicable"
    assert set(ev_m8["excluded_structurally_dead_dims"]) == set(SHORT_DEAD_DIMS)

    # Rule v2 (B-14): mixed pair (one side >=2 windows) -> 8D; ac_* excluded; 11D only when both >=2 windows.
    multi_390 = ("あ" * 129 + "。") * 3  # three 130-char sentences = 390 chars (Band M) -> 2 windows
    assert window_count_kasane(multi_390) == 2
    assert is_eight_dim((1, 2)) is True
    assert is_eight_dim((2, 2)) is False
    ev_mixed = evaluate_f14_hybrid_pair(single_260, multi_390, nf_data, provenance=prov)
    assert ev_mixed["band"] == "M"
    assert ev_mixed["n_evaluated"] == 8
    assert ev_mixed["n_eval_rule"] == "mixed_window:8"
    assert ev_mixed["eval_rule"] == "v2"
    assert ev_mixed["mixed_window_pair"] is True
    assert ev_mixed["ac_sigma_source"] == "not_applicable"
    assert set(ev_mixed["excluded_structurally_dead_dims"]) == set(SHORT_DEAD_DIMS)
    ev_m11 = evaluate_f14_hybrid_pair(multi_390, multi_390, nf_data, provenance=prov)
    assert ev_m11["band"] == "M"
    assert ev_m11["n_evaluated"] == 11
    assert ev_m11["n_eval_rule"] == "both_multi_window:11"
    assert ev_m11["mixed_window_pair"] is False
    assert ev_m11["ac_sigma_source"] == "multi_window_pairs"
    assert ev_m11["excluded_structurally_dead_dims"] == []
    m_band = select_length_band(300, nf_data["length_bands"]["bands"])
    assert ev_m11["sigma_used"]["ac_mean_velocity"] == pytest.approx(
        m_band["kasane11"]["ac_multi_window_per_dim_std"]["ac_mean_velocity"], rel=1e-12
    )
    assert ev_m11["sigma_used"]["ac_max_sis"] == pytest.approx(
        m_band["kasane11"]["ac_multi_window_per_dim_std"]["ac_max_sis"], rel=1e-12
    )
    assert ev_m11["sigma_used"]["ac_coupling"] == pytest.approx(
        m_band["kasane11"]["ac_multi_window_per_dim_std"]["ac_coupling"], rel=1e-12
    )
    assert ev_m11["sigma_used"]["ac_mean_crest"] == pytest.approx(
        m_band["kasane11"]["per_dim_std"]["ac_mean_crest"], rel=1e-12
    )


def test_b11_f14_parity_artifact_and_zero_discrepancies():
    """Verify B-11 JSON artifact and 0 discrepancies between Python and Rust on all 3 pair sets."""
    assert OUTPUT_B10_JSON_PATH.exists(), f"Missing {OUTPUT_B10_JSON_PATH}"
    assert OUTPUT_B11_JSON_PATH.exists(), f"Missing {OUTPUT_B11_JSON_PATH}"

    mc_loaded = get_or_compute_b10_mc_quantiles(OUTPUT_B10_JSON_PATH)
    assert mc_loaded["source"] == "loaded_from_b10_json"

    b11 = json.loads(OUTPUT_B11_JSON_PATH.read_text(encoding="utf-8"))
    f14 = b11["task2_f14_hybrid_judgment_and_sigma_parity"]

    # Set 1: 10,859 Aozora adjacent paragraph pairs across 46 works
    s1 = f14["set1_aozora_10859_adjacent_pairs"]
    assert s1["n_works"] == 46
    assert s1["n_pairs_total"] == 10859
    assert s1["discrepancy_count"] == 0
    assert s1["by_band_python"]["S"]["n_eval_8"] == 8258
    assert s1["by_band_python"]["M"]["n_eval_8"] == 621
    assert s1["by_band_python"]["M"]["n_eval_11"] == 1331
    assert s1["by_band_python"]["L"]["n_eval_11"] == 649

    # Set 2: 86 revision pairs
    s2 = f14["set2_u6_86_revision_pairs"]
    assert s2["n_pairs_total"] == 86
    assert s2["by_class"]["Class_A_micro_polish"]["n_eval_8"] == 42
    assert s2["by_class"]["Class_B_macro_revision"]["n_eval_11"] == 22
    assert s2["by_class"]["Class_C_content_rewrite"]["n_eval_8"] == 22
    if s2["rust_cli_verification"]["rust_binary_used"] is not None:
        assert s2["rust_cli_verification"]["cli_aligned_pairs"] == 77
        assert s2["rust_cli_verification"]["n_evaluated_discrepancy_on_aligned"] == 0

    # Set 3: 28 STYLE_POLISH pairs x 3 granularities
    s3 = f14["set3_style_polish_28_pairs_3_granularities"]
    assert s3["span_only"]["n_eval_8"] == 22 and s3["span_only"]["n_eval_11"] == 6
    assert s3["local_context"]["n_eval_8"] == 7 and s3["local_context"]["n_eval_11"] == 21
    assert s3["whole_scene"]["n_eval_8"] == 0 and s3["whole_scene"]["n_eval_11"] == 28
    assert s3["span_only"]["rust_mismatches"] == 0
    assert s3["local_context"]["rust_mismatches"] == 0
    assert s3["whole_scene"]["rust_mismatches"] == 0
