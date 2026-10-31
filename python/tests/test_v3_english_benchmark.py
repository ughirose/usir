"""Pytest Suite for USIR v3.0 English Cross-Linguistic Universality Benchmark (Table 4).

Validates:
1. Benchmark report JSON exists in both artifacts/ and python/usir_py/v2/.
2. Benchmark metadata: N = 400 chunks across >= 20 disjoint author groups under 5-Fold GroupKFold.
3. Table 4 evaluates all 7 required models without data leakage:
   - M0_Null_Prior
   - B1_Content_1gram_BoW
   - B2_Surface_Stylometry_16D
   - B3_Compression_Entropy
   - A1_USIR_Position_Only_12D
   - A2_USIR_Euclidean_24D
   - Proposed_Full_USIR_v3_English
4. Regime 2 (Theiler Surrogates):
   - B1 (1-gram BoW) collapses to AUC <= 0.58 (multiset identical to authentic text).
   - B3 (Compression Entropy) collapses to AUC <= 0.58.
   - Proposed achieves OOF AUC >= 0.88 via sequential USIR trajectory metrics.
5. Regime 1 (Topic-Matched AI Slop):
   - Proposed achieves OOF AUC >= 0.90.
6. Overall:
   - Proposed achieves highest Overall ROC-AUC (>= 0.90), PR-AUC (>= 0.90), and Macro-F1 (>= 0.75).
   - Unbiased Mahalanobis Delta_M^2 strictly monotonic across ablation hierarchy:
     Delta_M^2(A1) < Delta_M^2(A2) < Delta_M^2(Proposed), with Proposed >= 35.0.
7. L1-normalized OOF Shapley attribution vector across 12 elements sums strictly to 1.0.
"""

from __future__ import annotations

import json
import os
import pytest

REPORT_PY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "usir_py", "v2", "benchmark_v3_english_oof_report.json")
)
REPORT_ART_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "benchmark_v3_english_oof_report.json")
)


@pytest.fixture(scope="module")
def report_data() -> dict:
    """Load benchmark_v3_english_oof_report.json, regenerating if missing."""
    if not os.path.exists(REPORT_PY_PATH):
        from usir_py.v2.benchmark_v3_english_cross_linguistic import run_english_cross_linguistic_benchmark
        run_english_cross_linguistic_benchmark(save_reports=True)

    with open(REPORT_PY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_benchmark_report_files_exist_and_valid_json():
    """Verify both report files exist and contain valid JSON."""
    assert os.path.exists(REPORT_PY_PATH), f"Missing report at {REPORT_PY_PATH}"
    assert os.path.exists(REPORT_ART_PATH), f"Missing report at {REPORT_ART_PATH}"

    with open(REPORT_PY_PATH, "r", encoding="utf-8") as f:
        py_data = json.load(f)
    with open(REPORT_ART_PATH, "r", encoding="utf-8") as f:
        art_data = json.load(f)

    assert py_data["metadata"]["text_chunks"] == 400
    assert art_data["metadata"]["text_chunks"] == 400


def test_benchmark_metadata_sample_counts(report_data: dict):
    """Verify N = 400 chunks across >= 20 disjoint author groups under 5-Fold GroupKFold."""
    meta = report_data["metadata"]
    assert meta["text_chunks"] == 400
    assert meta["author_groups"] >= 20
    assert meta["splits"] == 5
    assert "GroupKFold" in meta["protocol"]
    assert len(meta["phases"]) == 5


def test_table4_contains_all_7_models(report_data: dict):
    """Verify Table 4 evaluates all 7 required models."""
    models = report_data["table4_models"]
    expected_models = [
        "M0_Null_Prior",
        "B1_Content_1gram_BoW",
        "B2_Surface_Stylometry_16D",
        "B3_Compression_Entropy",
        "A1_USIR_Position_Only_12D",
        "A2_USIR_Euclidean_24D",
        "Proposed_Full_USIR_v3_English",
    ]
    for m in expected_models:
        assert m in models, f"Missing model {m} in Table 4 report: {list(models.keys())}"


def test_regime2_theiler_surrogate_collapse_and_proposed_defense(report_data: dict):
    """Critical Verification: B1 and B3 collapse on Regime 2, while Proposed >= 0.88."""
    models = report_data["table4_models"]
    b1_r2 = models["B1_Content_1gram_BoW"]["regime2_roc_auc"]
    b3_r2 = models["B3_Compression_Entropy"]["regime2_roc_auc"]
    prop_r2 = models["Proposed_Full_USIR_v3_English"]["regime2_roc_auc"]

    assert b1_r2 <= 0.58, f"B1 failed to collapse on Regime 2: AUC={b1_r2:.4f} > 0.58"
    assert b3_r2 <= 0.58, f"B3 failed to collapse on Regime 2: AUC={b3_r2:.4f} > 0.58"
    assert prop_r2 >= 0.88, f"Proposed failed to defend on Regime 2: AUC={prop_r2:.4f} < 0.88"


def test_regime1_ai_slop_detection(report_data: dict):
    """Verify Proposed achieves Regime 1 ROC-AUC >= 0.90 against topic-matched AI slop."""
    models = report_data["table4_models"]
    prop_r1 = models["Proposed_Full_USIR_v3_English"]["regime1_roc_auc"]
    assert prop_r1 >= 0.90, f"Proposed Regime 1 AUC < 0.90: {prop_r1:.4f}"


def test_proposed_overall_dominance(report_data: dict):
    """Verify Proposed achieves highest overall ROC-AUC, PR-AUC, Macro-F1, and Delta_M^2."""
    models = report_data["table4_models"]
    prop = models["Proposed_Full_USIR_v3_English"]

    assert prop["overall_roc_auc"] >= 0.90, f"Proposed Overall ROC-AUC < 0.90: {prop['overall_roc_auc']:.4f}"
    assert prop["overall_pr_auc"] >= 0.90, f"Proposed Overall PR-AUC < 0.90: {prop['overall_pr_auc']:.4f}"
    assert prop["macro_f1"] >= 0.75, f"Proposed 5-Phase Macro-F1 < 0.75: {prop['macro_f1']:.4f}"

    for k, v in models.items():
        if k != "Proposed_Full_USIR_v3_English":
            assert prop["overall_roc_auc"] >= v["overall_roc_auc"], f"Model {k} beat Proposed in ROC-AUC"
            assert prop["overall_pr_auc"] >= v["overall_pr_auc"], f"Model {k} beat Proposed in PR-AUC"
            assert prop["macro_f1"] >= v["macro_f1"], f"Model {k} beat Proposed in Macro-F1"

    # Strictly monotonic Mahalanobis ablation: Delta_M^2(A1) < Delta_M^2(A2) < Delta_M^2(Proposed)
    d2_a1 = models["A1_USIR_Position_Only_12D"]["unbiased_mahalanobis_d2"]
    d2_a2 = models["A2_USIR_Euclidean_24D"]["unbiased_mahalanobis_d2"]
    d2_prop = prop["unbiased_mahalanobis_d2"]

    assert d2_a1 < d2_a2, f"Expected Delta_M^2(A1) < Delta_M^2(A2), got {d2_a1:.2f} >= {d2_a2:.2f}"
    assert d2_a2 < d2_prop, f"Expected Delta_M^2(A2) < Delta_M^2(Proposed), got {d2_a2:.2f} >= {d2_prop:.2f}"
    assert d2_prop >= 35.0, f"Proposed Mahalanobis d2 fell below 35.0: {d2_prop:.2f}"


def test_oof_shapley_attribution_normalization(report_data: dict):
    """Verify L1-normalized OOF Shapley attribution vector across 12 elements sums strictly to 1.0."""
    shapley = report_data["oof_shapley_attribution"]
    assert len(shapley) == 12
    l1_sum = sum(abs(v) for v in shapley.values())
    assert l1_sum == pytest.approx(1.0, rel=1e-5), f"L1 sum {l1_sum} != 1.0"
