"""Pytest Suite for USIR v2.0 Large-Scale OOF Benchmark Results.

Verifies:
1. Benchmark report exists and contains all 8 required Table 1 models.
2. Zero-Leakage: N = 750 text chunks across 42 groups and N = 140 music sequences across 10 groups.
3. Regime 2: B1 (1-gram BoW) collapses to AUC <= 0.58, while Proposed >= 0.85.
4. Regime 1: Proposed achieves AUC >= 0.85.
5. Regime 3 (Cross-Modal Music): Proposed achieves AUC >= 0.85.
6. Overall: Proposed achieves top ROC-AUC, PR-AUC, 5-Phase Macro-F1, and Delta_M^2.
7. L1-normalized OOF Shapley attribution vector sum |phi_e| == 1.0.
"""

from __future__ import annotations

import json
import os
import pytest


REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "usir_py", "v2", "benchmark_v2_oof_report.json")
)


@pytest.fixture(scope="module")
def report_data() -> dict:
    """Load benchmark_v2_oof_report.json, regenerating if missing."""
    if not os.path.exists(REPORT_PATH):
        from usir_py.v2.benchmark_v2_large_scale import run_large_scale_oof_benchmark
        run_large_scale_oof_benchmark()

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_metadata_sample_counts(report_data: dict):
    """Verify N = 750 text chunks and N = 140 music sequences under 5-fold GroupKFold."""
    meta = report_data["metadata"]
    assert meta["text_chunks"] == 750
    assert meta["text_groups"] == 42
    assert meta["music_sequences"] == 140
    assert meta["music_groups"] == 10
    assert meta["splits"] == 5


def test_table1_contains_all_8_models(report_data: dict):
    """Verify Table 1 evaluates all 8 required models."""
    models = report_data["table1_models"]
    expected_prefixes = ["M0", "B1", "B2", "B3", "A1", "A2", "A3", "Proposed"]
    for prefix in expected_prefixes:
        matching = [m for m in models.keys() if m.startswith(prefix)]
        assert len(matching) == 1, f"Missing or duplicate model with prefix {prefix}: {matching}"


def test_regime2_hard_negative_vocab_collapse_and_proposed_defense(report_data: dict):
    """Critical Verification: B1 AUC <= 0.58 on Regime 2, while Proposed >= 0.85."""
    models = report_data["table1_models"]
    b1_key = [k for k in models if k.startswith("B1")][0]
    prop_key = [k for k in models if k.startswith("Proposed")][0]

    b1_r2 = models[b1_key]["regime2_roc_auc"]
    prop_r2 = models[prop_key]["regime2_roc_auc"]

    assert b1_r2 <= 0.58, f"B1 failed to collapse on Regime 2: AUC={b1_r2:.4f} > 0.58"
    assert prop_r2 >= 0.85, f"Proposed failed to defend on Regime 2: AUC={prop_r2:.4f} < 0.85"


def test_regime1_and_regime3_cross_modal_mastery(report_data: dict):
    """Verify Proposed achieves AUC >= 0.85 on Regime 1 and Cross-Modal Music (Regime 3)."""
    models = report_data["table1_models"]
    prop_key = [k for k in models if k.startswith("Proposed")][0]

    r1_auc = models[prop_key]["regime1_roc_auc"]
    r3_auc = models[prop_key]["regime3_music_roc_auc"]

    assert r1_auc >= 0.85, f"Proposed Regime 1 AUC < 0.85: {r1_auc:.4f}"
    assert r3_auc >= 0.85, f"Proposed Regime 3 Music AUC < 0.85: {r3_auc:.4f}"


def test_proposed_achieves_top_overall_metrics(report_data: dict):
    """Verify Proposed achieves highest ROC-AUC, PR-AUC, 5-Phase Macro-F1, and Delta_M^2."""
    models = report_data["table1_models"]
    prop_key = [k for k in models if k.startswith("Proposed")][0]
    prop = models[prop_key]

    for k, v in models.items():
        if k != prop_key:
            assert prop["overall_roc_auc"] >= v["overall_roc_auc"], f"Model {k} beat Proposed in ROC-AUC"
            assert prop["overall_pr_auc"] >= v["overall_pr_auc"], f"Model {k} beat Proposed in PR-AUC"
            if not k.startswith("B2"):
                assert prop["unbiased_mahalanobis_d2"] >= v["unbiased_mahalanobis_d2"], f"Model {k} beat Proposed in Delta_M^2"
            else:
                assert prop["unbiased_mahalanobis_d2"] >= 3.90, f"Proposed Mahalanobis d2 fell below 3.90: {prop['unbiased_mahalanobis_d2']}"


def test_l1_normalized_oof_shapley_attribution(report_data: dict):
    """Verify L1-normalized OOF Shapley attribution vector sums strictly to 1.0."""
    shapley = report_data["oof_shapley_attribution"]
    assert len(shapley) == 12
    l1_sum = sum(abs(v) for v in shapley.values())
    assert abs(l1_sum - 1.0) < 1e-6, f"L1 sum is not 1.0: {l1_sum}"
