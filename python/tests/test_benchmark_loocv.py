"""Regression & LOOCV Verification Test Suite for USIR."""

import numpy as np
import pytest

from usir_py.corpus_40_benchmark import CORPUS
from usir_py.extractor_24d import extract_usir_24d, run_loocv
from usir_py.grpo_reward import compute_grpo_reward


@pytest.fixture(scope="module")
def processed_corpus():
    return [{**item, **extract_usir_24d(item["text"])} for item in CORPUS]


def test_corpus_count_and_phases(processed_corpus):
    assert len(processed_corpus) == 40
    phase_counts = {}
    for item in processed_corpus:
        phase_counts[item["phase"]] = phase_counts.get(item["phase"], 0) + 1
    assert phase_counts["I_Crystalline"] == 6
    assert phase_counts["II_Critical"] == 10
    assert phase_counts["III_Void"] == 10
    assert phase_counts["IV_Turbulent"] == 6
    assert phase_counts["V_Deadzone"] == 8


def test_5phase_loocv_golden_accuracy(processed_corpus):
    """Verifies that all 5 phases achieve exact 97.5% LOOCV (39/40)."""
    res = run_loocv(processed_corpus, feature_fn=lambda r: r["z_24d"])
    assert res["total"] == 40
    assert res["correct"] == 39
    assert pytest.approx(res["accuracy"], abs=1e-4) == 0.975
    assert res["accuracy"] >= 0.950


def test_core3_phase_loocv_golden_accuracy(processed_corpus):
    """Verifies that Core 3-Phase (Void vs Critical vs Deadzone) achieves exact 96.4% (27/28)."""
    core3 = ["II_Critical", "III_Void", "V_Deadzone"]
    res = run_loocv(processed_corpus, feature_fn=lambda r: r["z_24d"], target_filter=core3)
    assert res["total"] == 28
    assert res["correct"] == 27
    assert pytest.approx(res["accuracy"], abs=1e-4) == (27 / 28)
    assert res["accuracy"] >= 0.950


def test_ablation_inequalities(processed_corpus):
    """Verifies that Full 24D strictly outperforms all ablation subsets."""
    core3 = ["II_Critical", "III_Void", "V_Deadzone"]

    def select_without_indices(r, drop_indices):
        keep_q = [r["q"][i] for i in range(12) if i not in drop_indices]
        keep_p = [r["p"][i] for i in range(12) if i not in drop_indices]
        return np.array(keep_q + keep_p)

    # Full 24D
    acc_full_5phase = run_loocv(processed_corpus, feature_fn=lambda r: r["z_24d"])["accuracy"]
    acc_pos12_5phase = run_loocv(processed_corpus, feature_fn=lambda r: r["q"])["accuracy"]

    acc_core3_full = run_loocv(processed_corpus, feature_fn=lambda r: r["z_24d"], target_filter=core3)["accuracy"]
    acc_core3_no_void = run_loocv(processed_corpus, feature_fn=lambda r: select_without_indices(r, [2, 6, 10]), target_filter=core3)["accuracy"]
    acc_core3_no_shock = run_loocv(processed_corpus, feature_fn=lambda r: select_without_indices(r, [3, 7, 11]), target_filter=core3)["accuracy"]
    acc_core3_no_mom = run_loocv(processed_corpus, feature_fn=lambda r: r["q"], target_filter=core3)["accuracy"]

    # 1. Full 24D strictly superior to 12D position on 5-phase
    assert acc_full_5phase > acc_pos12_5phase
    assert pytest.approx(acc_pos12_5phase, abs=1e-3) == 0.925

    # 2. Core 3-phase ablation inequalities
    assert acc_core3_full > acc_core3_no_void
    assert acc_core3_full > acc_core3_no_shock
    assert acc_core3_full > acc_core3_no_mom

    assert pytest.approx(acc_core3_no_void, abs=1e-3) == (26 / 28)
    assert pytest.approx(acc_core3_no_shock, abs=1e-3) == (26 / 28)
    assert pytest.approx(acc_core3_no_mom, abs=1e-3) == (25 / 28)


def test_grpo_reward_ordering(processed_corpus):
    """Verifies that authentic texts achieve significantly higher rewards than AI Deadzone text."""
    # Target OBSIDIAN_VOID (Phase III)
    void_text = next(r["text"] for r in processed_corpus if r["id"] == "VOID-01")
    dead_text = next(r["text"] for r in processed_corpus if r["id"] == "DEAD-01")

    r_void = compute_grpo_reward(void_text, target="OBSIDIAN_VOID")
    r_dead = compute_grpo_reward(dead_text, target="OBSIDIAN_VOID")

    assert r_void["reward"] > r_dead["reward"]
    assert r_void["reward"] > 0.40
    assert r_dead["reward"] < 0.10
    assert r_void["cadence_alive"] > 0.80
    assert r_dead["cadence_alive"] < 0.50

    # Target AMBER_CRITICAL (Phase II)
    crit_text = next(r["text"] for r in processed_corpus if r["id"] == "CRIT-01")
    r_crit = compute_grpo_reward(crit_text, target="AMBER_CRITICAL")
    r_dead_crit = compute_grpo_reward(dead_text, target="AMBER_CRITICAL")

    assert r_crit["reward"] > r_dead_crit["reward"]
    assert r_crit["reward"] > 0.35
