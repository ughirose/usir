"""Tests for USIR v5 Stage B-12: Dice Gate Unification, Alarm Threshold Spec, KASANE8 H1/H2 Recalc, & v1 Dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

from gate3a_register_centering_eval import (
    AUXILIARY_ALARM_THRESHOLD_D,
    AUXILIARY_ALARM_THRESHOLD_KEY,
    GATE3A_STATUS,
    GATE3B_STATUS,
    OUTPUT_B12_JSON_PATH,
    OUTPUT_B12_SUMMARY_PATH,
    PRIMARY_ALARM_THRESHOLD_D,
    PRIMARY_ALARM_THRESHOLD_KEY,
    build_b12_gate3a_summary,
    get_alarm_threshold_specification,
)
from scale_separated_level3 import (
    DEFAULT_NISHIKI_EXE,
    KASANE8_LEGACY_SLICE_INDICES,
    evaluate_f14_hybrid_alignment_and_parity,
    extract_kasane8_active,
    extract_kasane8_active_legacy_slice,
    verify_kasane8_index_unification_and_recalc_h1_h2,
)
from usir_py.v5.f14_hybrid_noise_floor import (
    ALIGN_MIN_SIMILARITY,
    KASANE8_ACTIVE_DIM_NAMES,
    KASANE8_ACTIVE_INDICES,
    KASANE11_DIM_NAMES,
    char_bigram_dice,
    char_bigrams,
    compute_sha256,
    evaluate_f14_hybrid_pair,
    load_f14_noise_floor,
)

EXPECTED_LOW_DICE_U6_IDS = ["A_29", "A_35", "B_19", "B_20", "B_22", "C_19", "C_20", "C_21", "C_22"]
F15_NOISE_FLOOR_PATH = ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_f15_all_novel.json"
EXPECTED_F15_SHA256 = "5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910"
KASANE8_RECALC_JSON_PATH = ROOT / "docs" / "handoff" / "v10" / "b12_kasane8_unification_h1_h2_recalc.json"
V1_DATASET_PATH = ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1.jsonl"
V1_SUMMARY_PATH = ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_f15_harvest_summary.json"

FROZEN_ARTIFACTS_SHA256 = {
    "data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json": "975765a13db7fdf89f1b32a21a88bf02f470817a4308a750b0be1bd521671de1",
    "data/usir_v5_eval/usir_v5_b9_register_centering_and_level3.json": "4a555708dbcba73f4b08ad6e13c1aca4828231b698421f7903b3e64dd09032a0",
    "data/usir_v5_eval/usir_v5_b10_gate3a_scene_scale_and_level3.json": "2e0833d5c300273f73222c9ea3b039ec7358006849e9ff4957e4d9105c991c60",
    "data/usir_v5_eval/usir_v5_b11_gate3a_calibrated_and_f14_parity.json": "929bbfe9e7648145403e9c9f82ef9d2c78e7101e09ffe7c417ca2d84",
}


def test_task1_dice_formula_and_u6_unaligned_gate():
    """Verify char_bigram_dice Rust parity and Dice < 0.25 unaligned gate behavior."""
    assert ALIGN_MIN_SIMILARITY == 0.25
    assert char_bigram_dice("", "") == 1.0
    assert char_bigram_dice("あ", "") == 0.0
    assert char_bigram_dice("あ", "あ") == 1.0
    assert len(char_bigrams("あいう")) == 2
    assert char_bigram_dice("あ い う", "あいう") == 1.0

    nf_data, prov = load_f14_noise_floor()
    disjoint_a = "あ" * 100 + "。"
    disjoint_b = "か" * 100 + "。"
    ev_unaligned = evaluate_f14_hybrid_pair(disjoint_a, disjoint_b, nf_data, provenance=prov, apply_dice_gate=True)
    assert ev_unaligned["kind"] == "unaligned"
    assert ev_unaligned["aligned"] is False
    assert ev_unaligned["n_evaluated"] is None
    assert ev_unaligned["dice_similarity"] == 0.0

    ev_bypassed = evaluate_f14_hybrid_pair(disjoint_a, disjoint_b, nf_data, provenance=prov, apply_dice_gate=False)
    assert ev_bypassed["kind"] == "pair"
    assert ev_bypassed["n_evaluated"] == 8

    # Verify full 3-set parity artifact (or recompute)
    if OUTPUT_B12_JSON_PATH.exists():
        b12 = json.loads(OUTPUT_B12_JSON_PATH.read_text(encoding="utf-8"))
        f14 = b12["task1_dice_gate_canonical_parity"]
    else:
        f14 = evaluate_f14_hybrid_alignment_and_parity()

    s1 = f14["set1_aozora_10859_adjacent_pairs"]
    assert s1["n_pairs_total"] == 10859
    assert s1["discrepancy_count"] == 0

    s2 = f14["set2_u6_86_revision_pairs"]
    assert s2["n_pairs_total"] == 86
    assert s2["python_aligned_pairs"] == 77
    assert s2["python_unaligned_low_dice_pairs"] == EXPECTED_LOW_DICE_U6_IDS
    if s2["rust_cli_verification"]["rust_binary_used"] is not None:
        assert s2["rust_cli_verification"]["cli_aligned_pairs"] == 77
        assert s2["rust_cli_verification"]["cli_unaligned_low_dice_pairs"] == EXPECTED_LOW_DICE_U6_IDS
        assert s2["rust_cli_verification"]["aligned_pair_ids_match_python"] is True
        assert s2["rust_cli_verification"]["total_discrepancies_including_unaligned"] == 0

    s3 = f14["set3_style_polish_28_pairs_3_granularities"]
    for gran in ("span_only", "local_context", "whole_scene"):
        assert s3[gran]["n_aligned"] == 28
        assert s3[gran]["n_eval_unaligned"] == 0
        assert s3[gran]["rust_mismatches"] == 0


def test_task2_gate3a_explicit_primary_auxiliary_alarm_thresholds():
    """Verify Gate 3a primary/auxiliary alarm role documentation without changing thresholds or gates."""
    assert GATE3A_STATUS == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert GATE3B_STATUS == "GATE3B_FROZEN"
    assert PRIMARY_ALARM_THRESHOLD_KEY == "mc_indep_p95"
    assert PRIMARY_ALARM_THRESHOLD_D == 6.9267
    assert AUXILIARY_ALARM_THRESHOLD_KEY == "mc_r4corr_p95"
    assert AUXILIARY_ALARM_THRESHOLD_D == 7.6574

    spec = get_alarm_threshold_specification()
    assert spec["primary_alarm"]["label_ja"] == "一次警報"
    assert spec["primary_alarm"]["threshold_key"] == "mc_indep_p95"
    assert spec["primary_alarm"]["threshold_d"] == 6.9267
    assert spec["auxiliary_alarm"]["label_ja"] == "補助警報"
    assert spec["auxiliary_alarm"]["threshold_key"] == "mc_r4corr_p95"
    assert spec["auxiliary_alarm"]["threshold_d"] == 7.6574

    assert OUTPUT_B12_SUMMARY_PATH.exists(), f"Missing {OUTPUT_B12_SUMMARY_PATH}"
    summary = json.loads(OUTPUT_B12_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["gate_3a_status"] == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert summary["gate_3b_status"] == "GATE3B_FROZEN"
    assert summary["primary_alarm_threshold_key"] == "mc_indep_p95"
    assert summary["primary_alarm_threshold_d"] == 6.9267
    assert summary["auxiliary_alarm_threshold_key"] == "mc_r4corr_p95"
    assert summary["auxiliary_alarm_threshold_d"] == 7.6574
    assert summary["gate3a_calibrated_evaluation"]["thresholds"]["mc_indep_p95"] == 6.9267
    assert summary["gate3a_calibrated_evaluation"]["thresholds"]["mc_r4corr_p95"] == 7.6574


def test_task3_kasane8_index_unification_and_h1_h2_recalc():
    """Verify KASANE8_ACTIVE_INDICES unification vs legacy [:8] slice and H1/H2 supplementary diff."""
    assert KASANE8_LEGACY_SLICE_INDICES == (0, 1, 2, 3, 4, 5, 6, 7)
    assert KASANE8_ACTIVE_INDICES == (0, 1, 2, 3, 4, 5, 6, 10)
    assert tuple(KASANE11_DIM_NAMES[i] for i in KASANE8_ACTIVE_INDICES) == KASANE8_ACTIVE_DIM_NAMES

    sample = "雨が静かに降る、だが街は静まり返っていた。しかし誰かが歩いた。"
    v_leg = extract_kasane8_active_legacy_slice(sample)
    v_uni = extract_kasane8_active(sample)
    assert v_leg.shape == (8,)
    assert v_uni.shape == (8,)
    assert list(v_leg[:7]) == list(v_uni[:7])

    assert KASANE8_RECALC_JSON_PATH.exists(), f"Missing {KASANE8_RECALC_JSON_PATH}"
    recalc = json.loads(KASANE8_RECALC_JSON_PATH.read_text(encoding="utf-8"))
    assert recalc["index_verification"]["same_index_set"] is False
    assert recalc["index_verification"]["only_in_legacy_indices"] == [7]
    assert recalc["index_verification"]["only_in_unified_indices"] == [10]
    assert recalc["h1_32d_core"]["diff_unified_vs_frozen"]["legacy_repro_matches_frozen_exact"] is True
    assert recalc["h2_5d_continuous_core"]["diff_vs_frozen"]["exact_match_with_frozen"] is True
    assert recalc["rating_conclusions_assessment"]["rating_conclusion_changed"] is False


def test_frozen_b8_to_b11_artifacts_unmodified():
    """Ensure Stage B-12 never overwrote frozen B-8/B-9/B-10/B-11 JSON files."""
    expected_prefix_b11 = "929bbfe9e7340bcf4f648145403e9c9f82ef9d2c78e7101e09ffe7c417ca2d84"
    checks = dict(FROZEN_ARTIFACTS_SHA256)
    checks["data/usir_v5_eval/usir_v5_b11_gate3a_calibrated_and_f14_parity.json"] = expected_prefix_b11
    for rel, expected_sha in checks.items():
        p = ROOT / rel
        assert p.exists(), f"Missing frozen artifact: {p}"
        assert compute_sha256(p) == expected_sha, f"Frozen artifact modified: {rel}"


def test_task4_real_revision_pairs_v1_dataset_and_f15_harvest_diff():
    """Verify real_revision_pairs_v1.jsonl provenance, F-15 noise floor SHA-256, and zero parity discrepancy."""
    assert F15_NOISE_FLOOR_PATH.exists(), f"Missing F-15 noise floor: {F15_NOISE_FLOOR_PATH}"
    assert compute_sha256(F15_NOISE_FLOOR_PATH) == EXPECTED_F15_SHA256

    assert V1_DATASET_PATH.exists(), f"Missing v1 dataset: {V1_DATASET_PATH}"
    assert V1_SUMMARY_PATH.exists(), f"Missing v1 harvest summary: {V1_SUMMARY_PATH}"

    rows = [json.loads(ln) for ln in V1_DATASET_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) > 0
    seen_pairs = set()
    for r in rows:
        assert r["text_before"] != r["text_after"]
        assert (r["text_before"], r["text_after"]) not in seen_pairs
        seen_pairs.add((r["text_before"], r["text_after"]))
        assert r["dice_similarity"] >= ALIGN_MIN_SIMILARITY
        assert r["provenance"]["source"].startswith("Aozora Bunko")
        assert r["provenance"]["edition_a"]["work_id"]
        assert r["provenance"]["edition_b"]["work_id"]

    summary = json.loads(V1_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["f15_noise_floor_sha256"] == EXPECTED_F15_SHA256
    assert summary["f15_sha256_matches_expected"] is True
    assert summary["python_rust_parity_discrepancy_count"] == 0
    assert summary["distributions"]["overall"]["n_pairs"] == len(rows)
