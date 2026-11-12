"""USIR v5 Stage B-8 Task 3: Scale-Separated Level 3b Residual Specificity & U6 ROC Calibration.

Components:
1. U6 Guards ROC Calibration:
   - Stratified 2-fold split (Selection vs Evaluation) on 86 pairs (Class A, B, C).
   - Grid search across min_jaccard in [0.35..0.75] (step 0.05) x max_char_change in [0.15..0.45] (step 0.05).
   - Metrics: Precision, Recall, F1, FPR on Selection vs Evaluation sets.
   - Benchmark against uncalibrated initial heuristics (0.55, 0.25).
   - Optimism bias and calibration scope analysis.
2. Two-Scale Separated Level 3b Residual Specificity Test:
   - H1 (Micro-scale on Class A, n=42): ortho28 and ortho28+kasane11 vs random and vs other authors.
   - H2 (Macro-scale on Class B, n=22): 6pole and 6pole+HW-25 vs random and vs other styles/authors.
   - Gram-Schmidt orthogonalization: rho_perp, theta_deg, quasi-parallel filter (rho_perp < 0.10).
   - Two-sided Wilcoxon signed-rank tests with W statistic, p-value, and effect size (mean cos_spec).
   - Both full set and quasi-parallel filtered subsets reported.
3. Kamae eta^2=1.00 Full Structural Code Audit:
   - Re-calculation on expanded dataset and structural pathway explanation of constitutive artifacts.

Outputs:
  data/usir_v5_eval/usir_v5_stage_b8_scale_separation_and_gate3a.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from fable_v5_grand_shootout import load_works
from usir_py.orthographic_prior import (
    ortho_matrix_ext_v2,
    ortho_matrix_narr_v2,
)
from usir_py.v5 import (
    AozoraCorpusSplitter,
    aggregate_work_profile_from_scenes,
    build_scene_annotation,
    split_work_into_scenes,
)
from usir_py.v5.narrative_pacing_and_metaphor import extract_hw25_uid_ratio
from usir_py.v5.u6_guards import u6_classify
from usir_v5_gate3a_and_diachronic import build_crossed_dataset, compute_two_way_anova_eta_squared

DATASET_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "u6_calibration_and_level3_dataset.jsonl"
OUTPUT_REPORT = REPO_ROOT / "data" / "usir_v5_eval" / "usir_v5_stage_b8_scale_separation_and_gate3a.json"

COMMA_DIMS = [2, 3, 29]
KEEP28 = [i for i in range(31) if i not in COMMA_DIMS]


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_ortho28(text: str) -> np.ndarray:
    """Extract 28D non-comma ortho_v2 features."""
    ext = ortho_matrix_ext_v2([text])[0]
    narr = ortho_matrix_narr_v2([text])[0]
    full31 = np.concatenate([ext, narr])
    return full31[KEEP28].astype(float)


def extract_kasane11(text: str) -> np.ndarray:
    """Extract 11D Kasane local rhythm/connector features."""
    n = max(len(text), 1)
    # 1. Comma density
    c_comma = text.count("、") / n
    # 2. Period density
    c_period = text.count("。") / n
    # 3. Particle te
    c_te = text.count("て") / n
    # 4. Particle de
    c_de = text.count("で") / n
    # 5. Particle ni
    c_ni = text.count("に") / n
    # 6. Particle wa
    c_wa = text.count("は") / n
    # 7. Particle ga
    c_ga = text.count("が") / n
    # 8. Particle no
    c_no = text.count("の") / n
    # 9. Past / assertion endings ta / da
    c_tada = (text.count("た") + text.count("だ")) / n
    # 10. Conjunction shift markers
    conj_matches = sum(text.count(w) for w in ["だが", "しかし", "ところが", "それゆえ", "けれども", "だから"])
    c_conj = conj_matches / n
    # 11. Mean clause length
    clauses = [cl for cl in text.replace("。", "、").split("、") if cl.strip()]
    mean_cl = float(np.mean([len(cl) for cl in clauses])) if clauses else float(n)

    return np.array([
        c_comma, c_period, c_te, c_de, c_ni, c_wa, c_ga, c_no, c_tada, c_conj, mean_cl / 100.0
    ], dtype=float)


def extract_ortho28_kasane11(text: str) -> np.ndarray:
    """Combine ortho28 (28D) and kasane11 (11D) into 39D micro feature vector."""
    o28 = extract_ortho28(text)
    k11 = extract_kasane11(text)
    return np.concatenate([o28, k11])


def extract_6pole(text: str, author_id: str, work_id: str, hook_idf: Any) -> np.ndarray:
    """Extract 6-pole coordinates."""
    sc = build_scene_annotation(text, author_id, work_id, 0, 1)
    prof = aggregate_work_profile_from_scenes([sc], hook_idf=hook_idf)
    l_text = max(len(text), 1)

    kasane = (text.count("、") / l_text) * 100.0
    sasae = (text.count("。") / l_text) * 100.0
    kamae = float(prof["kamae"]["hook_survival"])
    sadame = float(prof["sadame"]["total"])
    nagare = float(prof["nagare"]["phi_zeami"])
    awase = float(prof["awase"]["irony"]["mean"])

    return np.array([kasane, sasae, kamae, sadame, nagare, awase], dtype=float)


def extract_6pole_hw25(text: str, author_id: str, work_id: str, hook_idf: Any) -> np.ndarray:
    """Extract 6-pole + HW-25 (7D) macro feature vector."""
    p6 = extract_6pole(text, author_id, work_id, hook_idf)
    hw25 = extract_hw25_uid_ratio(text)
    return np.concatenate([p6, [hw25]])


# ---------------------------------------------------------------------------
# Gram-Schmidt Residual Orthogonalization
# ---------------------------------------------------------------------------

def gram_schmidt_residual(v_target: np.ndarray, u_corp: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray, bool]:
    """Orthogonalize target vector v_target against corpus direction u_corp.

    Returns:
      v_perp: orthogonal residual
      rho_perp: ||v_perp|| / ||v_target||
      theta_deg: angle between v_target and u_corp
      u_spec: unit residual vector
      quasi_parallel: True if rho_perp < 0.10 (theta < 5.74 deg)
    """
    norm_vt = float(np.linalg.norm(v_target))
    norm_uc = float(np.linalg.norm(u_corp))
    if norm_vt < 1e-12 or norm_uc < 1e-12:
        zero = np.zeros_like(v_target)
        return zero, 0.0, 0.0, zero, True

    u_corp_unit = u_corp / norm_uc
    proj = float(np.dot(v_target, u_corp_unit))
    v_perp = v_target - proj * u_corp_unit
    norm_v_perp = float(np.linalg.norm(v_perp))

    rho_perp = float(norm_v_perp / norm_vt)
    cos_theta = float(np.clip(proj / norm_vt, -1.0, 1.0))
    theta_deg = float(math.degrees(math.acos(cos_theta)))

    if norm_v_perp < 1e-12:
        u_spec = np.zeros_like(v_perp)
    else:
        u_spec = v_perp / norm_v_perp

    quasi_parallel = bool(rho_perp < 0.10)
    return v_perp, rho_perp, theta_deg, u_spec, quasi_parallel


def run_wilcoxon_two_sided(diffs: Sequence[float]) -> Tuple[float, float, float]:
    """Run two-sided Wilcoxon signed-rank test against 0, return (W, p, effect_size_mean)."""
    arr = np.array(diffs, dtype=float)
    nonzero = arr[np.abs(arr) > 1e-12]
    if len(nonzero) < 5:
        return 0.0, 1.0, float(np.mean(arr))
    try:
        res = wilcoxon(nonzero, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), float(np.mean(arr))
    except Exception:
        return 0.0, 1.0, float(np.mean(arr))


# ---------------------------------------------------------------------------
# 1. U6 ROC Calibration
# ---------------------------------------------------------------------------

def run_u6_roc_calibration(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n--- 1. U6 Guards ROC Calibration ---")
    # Stratified 50/50 split
    rng = np.random.default_rng(42)

    by_class = {"Class_A_micro_polish": [], "Class_B_macro_revision": [], "Class_C_content_rewrite": []}
    for p in pairs:
        by_class[p["class"]].append(p)

    sel_pairs = []
    eval_pairs = []
    for cname, plist in by_class.items():
        indices = rng.permutation(len(plist))
        split_point = len(plist) // 2
        for idx in indices[:split_point]:
            sel_pairs.append(plist[idx])
        for idx in indices[split_point:]:
            eval_pairs.append(plist[idx])

    print(f"Selection set: {len(sel_pairs)} pairs, Evaluation set: {len(eval_pairs)} pairs")

    def eval_thresholds(pair_subset: List[Dict[str, Any]], min_j: float, max_c: float, scope: str) -> Dict[str, float]:
        tp, fp, tn, fn = 0, 0, 0, 0
        for p in pair_subset:
            if scope == "micro_only" and p["class"] == "Class_B_macro_revision":
                continue

            # Evaluate with custom thresholds
            res = u6_classify(p["text_before"], p["text_after"], min_jaccard=min_j, max_char_change=max_c)
            pred = res["label"]  # STYLE_POLISH or CONTENT_REWRITE
            gt = p["ground_truth_label"]

            if gt == "STYLE_POLISH":
                if pred == "STYLE_POLISH":
                    tp += 1
                else:
                    fn += 1
            else:  # CONTENT_REWRITE
                if pred == "STYLE_POLISH":
                    fp += 1
                else:
                    tn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    # Grid search ranges
    jaccard_grid = [round(v, 2) for v in np.arange(0.35, 0.76, 0.05)]
    change_grid = [round(v, 2) for v in np.arange(0.15, 0.46, 0.05)]

    scopes = ["Scope1_micro_polish_boundary", "Scope2_full_revision_boundary"]
    calibration_results = {}

    for scope in scopes:
        scope_key = "micro_only" if "Scope1" in scope else "all"
        best_f1 = -1.0
        best_params = (0.55, 0.25)
        best_metrics = {}

        grid_records = []
        for j in jaccard_grid:
            for c in change_grid:
                m_sel = eval_thresholds(sel_pairs, j, c, scope_key)
                grid_records.append({
                    "min_jaccard": j,
                    "max_char_change": c,
                    "f1": round(m_sel["f1"], 4),
                    "precision": round(m_sel["precision"], 4),
                    "recall": round(m_sel["recall"], 4),
                    "fpr": round(m_sel["fpr"], 4),
                })
                if m_sel["f1"] > best_f1:
                    best_f1 = m_sel["f1"]
                    best_params = (j, c)
                    best_metrics = m_sel

        # Evaluate best params on evaluation set
        m_eval_best = eval_thresholds(eval_pairs, best_params[0], best_params[1], scope_key)
        optimism_bias = round(best_f1 - m_eval_best["f1"], 4)

        # Evaluate initial baseline (0.55, 0.25)
        m_sel_init = eval_thresholds(sel_pairs, 0.55, 0.25, scope_key)
        m_eval_init = eval_thresholds(eval_pairs, 0.55, 0.25, scope_key)

        calibration_results[scope] = {
            "scope_description": (
                "Class A (micro-polish) vs Class C (content rewrite)" if scope_key == "micro_only"
                else "Class A + Class B (micro + macro revision) vs Class C (content rewrite)"
            ),
            "uncalibrated_initial_heuristic": {
                "min_jaccard": 0.55,
                "max_char_change": 0.25,
                "selection_set": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_sel_init.items()},
                "evaluation_set": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_eval_init.items()},
            },
            "calibrated_optimal": {
                "min_jaccard": best_params[0],
                "max_char_change": best_params[1],
                "selection_set": {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()},
                "evaluation_set": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_eval_best.items()},
                "optimism_bias": optimism_bias,
            },
            "grid_search_summary": {
                "n_grid_points": len(grid_records),
                "top_3_candidates": sorted(grid_records, key=lambda x: x["f1"], reverse=True)[:3],
            },
        }

    return calibration_results


# ---------------------------------------------------------------------------
# 2. Level 3b Residual Specificity Test (H1 & H2)
# ---------------------------------------------------------------------------

def run_level3b_scale_separation_test(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n--- 2. Scale-Separated Level 3b Residual Specificity Test ---")

    works = load_works()
    splitter = AozoraCorpusSplitter()
    profile_json = REPO_ROOT / "docs" / "handoff" / "v10" / "nishiki_profile_106works.json"
    hook_idf = json.loads(profile_json.read_text(encoding="utf-8")).get("hook_idf") if profile_json.exists() else None

    # Pre-extract author centroids across 10 authors for author-specificity
    all_aids = sorted({w["author_id"] for w in works})
    scenes_by_aid: Dict[str, List[str]] = {a: [] for a in all_aids}
    for w in works[:40]:  # Sample works
        txt = splitter.get_cached_text(w)
        scs = split_work_into_scenes(txt)
        scenes_by_aid[w["author_id"]].extend(scs[:2])

    print("Computing baseline corpus vectors and author prototypes...")
    proto_ortho28 = {a: np.mean([extract_ortho28(s) for s in scs], axis=0) for a, scs in scenes_by_aid.items() if scs}
    proto_o28_k11 = {a: np.mean([extract_ortho28_kasane11(s) for s in scs], axis=0) for a, scs in scenes_by_aid.items() if scs}
    proto_6pole = {a: np.mean([extract_6pole(s, a, "base", hook_idf) for s in scs], axis=0) for a, scs in scenes_by_aid.items() if scs}
    proto_6pole_hw25 = {a: np.mean([extract_6pole_hw25(s, a, "base", hook_idf) for s in scs], axis=0) for a, scs in scenes_by_aid.items() if scs}

    corp_ortho28 = np.mean(list(proto_ortho28.values()), axis=0)
    corp_o28_k11 = np.mean(list(proto_o28_k11.values()), axis=0)
    corp_6pole = np.mean(list(proto_6pole.values()), axis=0)
    corp_6pole_hw25 = np.mean(list(proto_6pole_hw25.values()), axis=0)

    # -----------------------------------------------------------------------
    # H1: Class A (Micro-polish, n=42)
    # -----------------------------------------------------------------------
    print("\n[H1: Micro-Scale Specificity Test on Class A (n=42)]")
    class_a_pairs = [p for p in pairs if p["class"] == "Class_A_micro_polish"]
    assert len(class_a_pairs) == 42

    h1_results = {}
    for space_name, extract_fn, corp_vec, proto_map in [
        ("ortho28", extract_ortho28, corp_ortho28, proto_ortho28),
        ("ortho28_plus_kasane11", extract_ortho28_kasane11, corp_o28_k11, proto_o28_k11),
    ]:
        v_targets = []
        rhos = []
        thetas = []
        quasis = []
        u_specs = []
        cos_rands = []
        cos_others = []

        dim = len(corp_vec)
        rng = np.random.default_rng(123)

        for p in class_a_pairs:
            z_b = extract_fn(p["text_before"])
            z_a = extract_fn(p["text_after"])
            v_t = z_a - z_b
            v_perp, rho, th, u_s, q = gram_schmidt_residual(v_t, corp_vec)

            v_targets.append(v_t)
            rhos.append(rho)
            thetas.append(th)
            quasis.append(q)
            u_specs.append(u_s)

            # Random isotropic comparison
            rand_v = rng.standard_normal(dim)
            rand_v /= np.linalg.norm(rand_v)
            cos_rands.append(float(np.dot(u_s, rand_v)))

            # Other authors residual comparison
            aid = p["author_id"]
            other_aids = [a for a in proto_map.keys() if a != aid]
            if other_aids:
                other_proto_mean = np.mean([proto_map[a] for a in other_aids], axis=0)
                v_other = other_proto_mean - corp_vec
                norm_vo = np.linalg.norm(v_other)
                if norm_vo > 1e-12:
                    cos_others.append(float(np.dot(u_s, v_other / norm_vo)))
                else:
                    cos_others.append(0.0)
            else:
                cos_others.append(0.0)

        # Wilcoxon tests: Full set vs Filtered subset
        n_total = len(class_a_pairs)
        n_quasi = sum(quasis)
        non_quasi_idx = [i for i, q in enumerate(quasis) if not q]
        n_filtered = len(non_quasi_idx)

        # Test vs random
        w_rand_all, p_rand_all, eff_rand_all = run_wilcoxon_two_sided(cos_rands)
        w_rand_filt, p_rand_filt, eff_rand_filt = run_wilcoxon_two_sided([cos_rands[i] for i in non_quasi_idx])

        # Test vs other authors
        w_oth_all, p_oth_all, eff_oth_all = run_wilcoxon_two_sided(cos_others)
        w_oth_filt, p_oth_filt, eff_oth_filt = run_wilcoxon_two_sided([cos_others[i] for i in non_quasi_idx])

        h1_results[space_name] = {
            "n_total": n_total,
            "n_quasi_parallel": n_quasi,
            "n_filtered_non_quasi": n_filtered,
            "rho_perp_stats": {
                "median": round(float(np.median(rhos)), 4),
                "iqr": round(float(stats.iqr(rhos)), 4),
                "min": round(float(np.min(rhos)), 4),
                "max": round(float(np.max(rhos)), 4),
            },
            "theta_deg_stats": {
                "median": round(float(np.median(thetas)), 2),
                "iqr": round(float(stats.iqr(thetas)), 2),
                "min": round(float(np.min(thetas)), 2),
                "max": round(float(np.max(thetas)), 2),
            },
            "test_vs_isotropic_random": {
                "full_cohort": {
                    "n": n_total,
                    "w_statistic": round(w_rand_all, 2),
                    "p_value_two_sided": float(p_rand_all),
                    "mean_cos_spec": round(eff_rand_all, 4),
                    "significant_005": bool(p_rand_all < 0.05),
                },
                "quasi_parallel_excluded": {
                    "n": n_filtered,
                    "w_statistic": round(w_rand_filt, 2),
                    "p_value_two_sided": float(p_rand_filt),
                    "mean_cos_spec": round(eff_rand_filt, 4),
                    "significant_005": bool(p_rand_filt < 0.05),
                },
            },
            "test_vs_other_authors": {
                "full_cohort": {
                    "n": n_total,
                    "w_statistic": round(w_oth_all, 2),
                    "p_value_two_sided": float(p_oth_all),
                    "mean_cos_spec": round(eff_oth_all, 4),
                    "significant_005": bool(p_oth_all < 0.05),
                },
                "quasi_parallel_excluded": {
                    "n": n_filtered,
                    "w_statistic": round(w_oth_filt, 2),
                    "p_value_two_sided": float(p_oth_filt),
                    "mean_cos_spec": round(eff_oth_filt, 4),
                    "significant_005": bool(p_oth_filt < 0.05),
                },
            },
        }

    # -----------------------------------------------------------------------
    # H2: Class B (Macro-revision, n=22)
    # -----------------------------------------------------------------------
    print("\n[H2: Macro-Scale Specificity Test on Class B (n=22)]")
    class_b_pairs = [p for p in pairs if p["class"] == "Class_B_macro_revision"]
    assert len(class_b_pairs) == 22

    h2_results = {}
    for space_name, extract_fn, corp_vec, proto_map in [
        ("6pole", lambda t: extract_6pole(t, "trans", "macro", hook_idf), corp_6pole, proto_6pole),
        ("6pole_plus_hw25", lambda t: extract_6pole_hw25(t, "trans", "macro", hook_idf), corp_6pole_hw25, proto_6pole_hw25),
    ]:
        v_targets = []
        rhos = []
        thetas = []
        quasis = []
        u_specs = []
        cos_rands = []
        cos_others = []

        dim = len(corp_vec)
        rng = np.random.default_rng(456)

        for p in class_b_pairs:
            z_b = extract_fn(p["text_before"])
            z_a = extract_fn(p["text_after"])
            v_t = z_a - z_b
            v_perp, rho, th, u_s, q = gram_schmidt_residual(v_t, corp_vec)

            v_targets.append(v_t)
            rhos.append(rho)
            thetas.append(th)
            quasis.append(q)
            u_specs.append(u_s)

            # Random isotropic comparison
            rand_v = rng.standard_normal(dim)
            rand_v /= np.linalg.norm(rand_v)
            cos_rands.append(float(np.dot(u_s, rand_v)))

            # Other style/author comparison
            aid = p["author_id"]
            other_aids = [a for a in proto_map.keys() if a != aid]
            if other_aids:
                other_proto_mean = np.mean([proto_map[a] for a in other_aids], axis=0)
                v_other = other_proto_mean - corp_vec
                norm_vo = np.linalg.norm(v_other)
                if norm_vo > 1e-12:
                    cos_others.append(float(np.dot(u_s, v_other / norm_vo)))
                else:
                    cos_others.append(0.0)
            else:
                cos_others.append(0.0)

        n_total = len(class_b_pairs)
        n_quasi = sum(quasis)
        non_quasi_idx = [i for i, q in enumerate(quasis) if not q]
        n_filtered = len(non_quasi_idx)

        w_rand_all, p_rand_all, eff_rand_all = run_wilcoxon_two_sided(cos_rands)
        w_rand_filt, p_rand_filt, eff_rand_filt = run_wilcoxon_two_sided([cos_rands[i] for i in non_quasi_idx])

        w_oth_all, p_oth_all, eff_oth_all = run_wilcoxon_two_sided(cos_others)
        w_oth_filt, p_oth_filt, eff_oth_filt = run_wilcoxon_two_sided([cos_others[i] for i in non_quasi_idx])

        h2_results[space_name] = {
            "n_total": n_total,
            "n_quasi_parallel": n_quasi,
            "n_filtered_non_quasi": n_filtered,
            "rho_perp_stats": {
                "median": round(float(np.median(rhos)), 4),
                "iqr": round(float(stats.iqr(rhos)), 4),
                "min": round(float(np.min(rhos)), 4),
                "max": round(float(np.max(rhos)), 4),
            },
            "theta_deg_stats": {
                "median": round(float(np.median(thetas)), 2),
                "iqr": round(float(stats.iqr(thetas)), 2),
                "min": round(float(np.min(thetas)), 2),
                "max": round(float(np.max(thetas)), 2),
            },
            "test_vs_isotropic_random": {
                "full_cohort": {
                    "n": n_total,
                    "w_statistic": round(w_rand_all, 2),
                    "p_value_two_sided": float(p_rand_all),
                    "mean_cos_spec": round(eff_rand_all, 4),
                    "significant_005": bool(p_rand_all < 0.05),
                },
                "quasi_parallel_excluded": {
                    "n": n_filtered,
                    "w_statistic": round(w_rand_filt, 2),
                    "p_value_two_sided": float(p_rand_filt),
                    "mean_cos_spec": round(eff_rand_filt, 4),
                    "significant_005": bool(p_rand_filt < 0.05),
                },
            },
            "test_vs_other_styles_authors": {
                "full_cohort": {
                    "n": n_total,
                    "w_statistic": round(w_oth_all, 2),
                    "p_value_two_sided": float(p_oth_all),
                    "mean_cos_spec": round(eff_oth_all, 4),
                    "significant_005": bool(p_oth_all < 0.05),
                },
                "quasi_parallel_excluded": {
                    "n": n_filtered,
                    "w_statistic": round(w_oth_filt, 2),
                    "p_value_two_sided": float(p_oth_filt),
                    "mean_cos_spec": round(eff_oth_filt, 4),
                    "significant_005": bool(p_oth_filt < 0.05),
                },
            },
        }

    return {"h1_micro_scale": h1_results, "h2_macro_scale": h2_results}


# ---------------------------------------------------------------------------
# 3. Kamae eta^2=1.00 Audit
# ---------------------------------------------------------------------------

def run_kamae_eta_squared_audit(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n--- 3. Kamae eta^2=1.00 Structural Code Audit ---")
    cells = build_crossed_dataset()
    splitter = AozoraCorpusSplitter()

    # Re-calculate on D10 16 crossed cells
    kamae_vals = []
    srcs = []
    transs = []
    for c in cells:
        ann = splitter.process_work_scenes({"author_id": c["trans_id"], "work_id": c["cell_id"]}, c["scene_texts"], include_audit_meta=False)
        prof = aggregate_work_profile_from_scenes(ann)
        kamae_vals.append(prof["kamae"]["hook_survival"])
        srcs.append(c["source_id"])
        transs.append(c["trans_id"])

    res_d10 = compute_two_way_anova_eta_squared(kamae_vals, srcs, transs)

    # Re-calculate across expanded Class B pairs
    class_b = [p for p in pairs if p["class"] == "Class_B_macro_revision"]
    kamae_diffs = []
    for p in class_b:
        kb = extract_6pole(p["text_before"], "a", "w", None)[2]
        ka = extract_6pole(p["text_after"], "a", "w", None)[2]
        kamae_diffs.append(abs(ka - kb))

    kamae_mean_delta = float(np.mean(kamae_diffs))
    kamae_identical_ratio = float(np.mean([1.0 if d < 1e-6 else 0.0 for d in kamae_diffs]))

    audit_finding = {
        "claimed_metric_in_b7": "Kamae eta^2_source = 1.0000",
        "recalculated_d10_anova": res_d10,
        "empirical_kamae_values_in_d10": [round(float(v), 4) for v in kamae_vals],
        "kamae_delta_in_expanded_class_b": {
            "mean_abs_delta": round(kamae_mean_delta, 4),
            "ratio_exact_zero_delta": round(kamae_identical_ratio, 4),
        },
        "pathway_and_mechanism_analysis": {
            "root_cause": (
                "Constitutive artifact of discrete hook survival calculation on short texts. "
                "In six_pole_atlas.py (L375), hook_survival = unresolved_unweighted / (open_unweighted + eps). "
                "In the 4 D10 source texts, hook detection produces either 0 open hooks (SRC_MELOS, SRC_SANGETSU -> 0.0) "
                "or 1 unresolved hook that never closes (SRC_GINGA, SRC_PRINCE -> 1.0). "
                "The 4 translator styles add only prefix/suffix framing sentences which do NOT alter scene-internal "
                "question/suspense hooks. Consequently, SS_trans == 0.0 and SS_interaction == 0.0, "
                "producing an artificial mathematical extreme eta^2_source = 1.0000."
            ),
            "scientific_status": "CONSTITUTIVE_ARTIFACT_RETRACTED_AS_INDEPENDENT_RESULT",
            "citation_rule": "MUST NOT be cited as empirical achievement or proof of perfect semantic preservation until continuous hook model is calibrated on full-length works.",
        },
    }
    return audit_finding


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=== USIR v5 Stage B-8: Task 3 Level 3b Scale Separation & U6 ROC Calibration ===")

    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    pairs = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs from {DATASET_PATH}")

    # 1. U6 ROC Calibration
    u6_calib = run_u6_roc_calibration(pairs)

    # 2. Scale-Separated Level 3b Residual Specificity
    level3_res = run_level3b_scale_separation_test(pairs)

    # 3. Kamae eta^2=1.00 Audit
    kamae_audit = run_kamae_eta_squared_audit(pairs)

    report = {
        "metadata": {
            "title": "USIR v5 Stage B-8: Scale Separation Level 3b & U6 ROC Calibration Report",
            "version": "1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_file": str(DATASET_PATH.relative_to(REPO_ROOT)),
            "n_pairs": len(pairs),
            "class_distribution": {
                "Class_A_micro_polish": len([p for p in pairs if p["class"] == "Class_A_micro_polish"]),
                "Class_B_macro_revision": len([p for p in pairs if p["class"] == "Class_B_macro_revision"]),
                "Class_C_content_rewrite": len([p for p in pairs if p["class"] == "Class_C_content_rewrite"]),
            },
        },
        "u6_roc_calibration": u6_calib,
        "level3b_scale_separated_specificity": level3_res,
        "kamae_eta2_structural_audit": kamae_audit,
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    elapsed = round(time.time() - t0, 2)
    print(f"\nSuccessfully wrote comprehensive Stage B-8 report to {OUTPUT_REPORT} ({elapsed}s)")


if __name__ == "__main__":
    main()
