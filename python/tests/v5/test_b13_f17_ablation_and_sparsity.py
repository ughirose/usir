"""Tests for USIR v5 Stage B-13: F-17 Re-measurement, 11D->8D->7D Ablation, Orthography Sensitivity, and 106-work Sparsity Audit."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

from gate3a_register_centering_eval import GATE3A_STATUS, GATE3B_STATUS
from usir_py.v5.f14_hybrid_noise_floor import compute_sha256

F17_NOISE_FLOOR_PATH = ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_all_novel.json"
EXPECTED_F17_SHA256 = "0575b64a75a8af8dd1ad0db81bdf4d8898c473ce8ac1876c6e7e2f8b61a6f604"

V1_DATASET_PATH = ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1.jsonl"
EXPECTED_DATASET_SHA256 = "62da75271f453bd3a8f161d916fdf7b7b87138b773b564b737f90aee85b6b8e4"

B13_SUMMARY_PATH = ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_f17_harvest_summary.json"
SPARSITY_106_PATH = ROOT / "docs" / "handoff" / "v10" / "sparsity_audit_106works_canonical.json"
EXPECTED_SPARSITY_LF_SHA256 = "d16d95b54c89ad7010cadea6da4c1a0d553c2188c24b30cbea6b0e52c8c6e64a"
EXPECTED_SPARSITY_CRLF_SHA256 = "efee120d503c929fd363f81a305cf8845a3c661c271faeea5867dcbb5f9c93a7"


def test_b13_f17_noise_floor_sha256_and_per_band_ac_values():
    """Verify F-17 paragraph_noise_floor_all_novel.json SHA-256 and per-band ac_* values."""
    assert F17_NOISE_FLOOR_PATH.exists(), f"Missing {F17_NOISE_FLOOR_PATH}"
    assert compute_sha256(F17_NOISE_FLOOR_PATH) == EXPECTED_F17_SHA256

    nf = json.loads(F17_NOISE_FLOOR_PATH.read_text(encoding="utf-8"))
    bands = {b["name"]: b["kasane11"]["ac_multi_window_per_dim_std"] for b in nf["length_bands"]["bands"]}
    assert round(bands["M"]["ac_mean_velocity"], 2) == 134.30
    assert round(bands["M"]["ac_max_sis"], 4) == 0.1266
    assert round(bands["M"]["ac_coupling"], 4) == 1.4079
    assert round(bands["L"]["ac_mean_velocity"], 2) == 102.57
    assert round(bands["L"]["ac_max_sis"], 4) == 0.1494
    assert round(bands["L"]["ac_coupling"], 4) == 1.0668


def test_b13_task1_f17_summary_parity_and_verification_gate():
    """Verify Task 1 F-17 harvest summary, 0/415 parity, n_eval=8 bit-invariance, and Band L split."""
    assert compute_sha256(V1_DATASET_PATH) == EXPECTED_DATASET_SHA256
    assert B13_SUMMARY_PATH.exists(), f"Missing {B13_SUMMARY_PATH}"

    s = json.loads(B13_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert s["stage"] == "B-13"
    assert s["dataset_sha256"] == EXPECTED_DATASET_SHA256
    assert s["f17_noise_floor_sha256"] == EXPECTED_F17_SHA256
    assert s["python_rust_parity_discrepancy_count"] == 0

    vg = s["verification_gate"]
    assert vg["n_evaluated_8_pair_count"] == 196
    assert vg["n_evaluated_8_bit_level_diff_count"] == 0
    assert vg["bit_identical_to_f15"] is True

    d17 = s["distributions_f17"]
    assert d17["overall"]["n_pairs"] == 415
    assert d17["band_L_both_single_window"]["n_pairs"] == 50
    assert d17["band_L_both_single_window"]["n_evaluated_distribution"]["8"] == 50
    assert d17["band_L_both_multi_window"]["n_pairs"] == 61
    assert d17["band_L_both_multi_window"]["n_evaluated_distribution"]["11"] == 61

    cross = s["mathematical_predictions_verification"]["threshold_1sd_crossings"]
    assert cross["exact_1_0_boundary_hits"] == 0
    assert cross["ac_mean_velocity_total"] == 8
    assert cross["ac_max_sis_total"] == 2


def test_b13_task2_ablation_and_stratification():
    """Verify Task 2 empirical 1-window ac_* zeros, 11D->8D->7D ablation, and minor-band stratification."""
    s = json.loads(B13_SUMMARY_PATH.read_text(encoding="utf-8"))
    t2 = s["task2_ablation_and_stratification"]
    ow = t2["one_window_ac_empirical_check"]
    assert ow["one_window_sides_total"] == 476
    assert ow["nonzero_counts"]["ac_mean_velocity"] == 0
    assert ow["nonzero_counts"]["ac_max_sis"] == 0
    assert ow["nonzero_counts"]["ac_coupling"] == 0

    mb = t2["stratification_2_minor_revision_band"]
    assert mb["mixed"]["n_pairs"] == 9
    assert mb["mixed"]["11D"]["abs_max"]["median"] > 2.0
    assert mb["mixed"]["8D"]["abs_max"]["median"] < 0.4
    assert mb["mixed"]["7D"]["abs_max"]["median"] < 0.2


def test_b13_task3_orthography_and_core_cohorts():
    """Verify Task 3 cohort counts and core set consistency."""
    s = json.loads(B13_SUMMARY_PATH.read_text(encoding="utf-8"))
    t3 = s["task3_orthography_and_core_sensitivity"]
    assert t3["all_415"]["n_pairs"] == 415
    assert t3["same_orthography_true_381"]["n_pairs"] == 381
    assert t3["same_orthography_false_34"]["n_pairs"] == 34
    assert t3["same_orthography_true_core_dl_le_30pct"]["n_pairs"] == 267


def test_b13_task4_sparsity_audit_106works_canonical():
    """Verify Task 4 canonical 106-work sparsity audit JSON, LF/CRLF SHA-256, and dimension classes."""
    assert SPARSITY_106_PATH.exists(), f"Missing {SPARSITY_106_PATH}"
    raw = SPARSITY_106_PATH.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(lf).hexdigest() == EXPECTED_SPARSITY_LF_SHA256
    assert hashlib.sha256(crlf).hexdigest() == EXPECTED_SPARSITY_CRLF_SHA256

    sp = json.loads(SPARSITY_106_PATH.read_text(encoding="utf-8"))
    poles = sp["usir_metrics_poles"]["aozora_106works_scenes.jsonl"]
    assert poles["n_scenes_read"] == 4576
    assert poles["n_works"] == 106
    assert poles["profile_had_errors"] is False

    ps = poles["per_scene"]
    assert (ps["n_dims"], ps["n_continuous"], ps["n_mode_heavy"], ps["n_zero_heavy"], ps["n_discrete"]) == (8, 5, 1, 2, 0)
    pw = poles["per_work"]
    assert (pw["n_dims"], pw["n_continuous"], pw["n_discrete"], pw["n_mode_heavy"], pw["n_zero_heavy"]) == (16, 13, 3, 0, 0)


def test_b13_governance_gates_unchanged():
    """Verify Gate 3a and Gate 3b remain frozen/unchanged."""
    assert GATE3A_STATUS == "SHIFT_QUANTIFIED_NOT_PASSED"
    assert GATE3B_STATUS == "GATE3B_FROZEN"
