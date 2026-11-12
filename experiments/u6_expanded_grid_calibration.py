"""USIR v5 Stage B8-DIAG Task 3: U6 Expanded Grid Calibration and Plateau Analysis.

Extends the grid search bounds:
- min_jaccard in [0.05..0.75] (step 0.05, 15 points)
- max_char_change in [0.15..0.85] (step 0.05, 15 points)
Total 225 grid points evaluated via stratified 2-fold cross validation.

Scopes:
- Scope 1: Micro-Polish Only (Class A vs Class C, 64 pairs)
- Scope 2: Full Revision (Class A + Class B vs Class C, 86 pairs)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("c:/dev/usir")
sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.v5.u6_guards import u6_classify

DATASET_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "u6_calibration_and_level3_dataset.jsonl"
OUT_REPORT = REPO_ROOT / "data" / "usir_v5_eval" / "u6_expanded_grid_calibration_report.json"


def evaluate_thresholds(
    pair_subset: List[Dict[str, Any]], min_j: float, max_c: float, scope: str
) -> Dict[str, Any]:
    tp, fp, tn, fn = 0, 0, 0, 0
    for p in pair_subset:
        if scope == "micro_only" and p["class"] == "Class_B_macro_revision":
            continue

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
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    t0 = time.time()
    print("=== USIR v5 Stage B8-DIAG Task 3: U6 Expanded Grid Calibration ===")

    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    pairs = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    # Stratified 2-fold split (Selection vs Evaluation)
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

    print(f"Selection set: {len(sel_pairs)}, Evaluation set: {len(eval_pairs)}")

    # Expanded Grid: 15 x 15 = 225 points
    jaccard_grid = [round(v, 2) for v in np.arange(0.05, 0.76, 0.05)]
    change_grid = [round(v, 2) for v in np.arange(0.15, 0.86, 0.05)]

    print(f"Jaccard range: [{min(jaccard_grid)}..{max(jaccard_grid)}] ({len(jaccard_grid)} steps)")
    print(f"Char change range: [{min(change_grid)}..{max(change_grid)}] ({len(change_grid)} steps)")

    scopes = [
        ("micro_only", "Scope 1: Micro-Polish Only (Class A vs Class C)"),
        ("all", "Scope 2: Full Revision (Class A + Class B vs Class C)"),
    ]

    results = {}

    for scope_key, scope_name in scopes:
        print(f"\n--- {scope_name} ---")
        grid_records = []
        best_f1 = -1.0
        best_params = (0.55, 0.25)
        best_m = {}

        for j in jaccard_grid:
            for c in change_grid:
                m_sel = evaluate_thresholds(sel_pairs, j, c, scope_key)
                m_eval = evaluate_thresholds(eval_pairs, j, c, scope_key)
                # Mean cross-validated F1
                mean_f1 = (m_sel["f1"] + m_eval["f1"]) / 2.0

                rec = {
                    "min_jaccard": j,
                    "max_char_change": c,
                    "sel_f1": round(m_sel["f1"], 4),
                    "sel_recall": round(m_sel["recall"], 4),
                    "sel_precision": round(m_sel["precision"], 4),
                    "sel_fpr": round(m_sel["fpr"], 4),
                    "eval_f1": round(m_eval["f1"], 4),
                    "eval_recall": round(m_eval["recall"], 4),
                    "eval_precision": round(m_eval["precision"], 4),
                    "eval_fpr": round(m_eval["fpr"], 4),
                    "mean_cv_f1": round(mean_f1, 4),
                }
                grid_records.append(rec)

                if m_sel["f1"] > best_f1 or (m_sel["f1"] == best_f1 and mean_f1 > best_m.get("mean_cv_f1", 0)):
                    best_f1 = m_sel["f1"]
                    best_params = (j, c)
                    best_m = rec

        # Plateau detection: all parameter points achieving maximum or near-maximum F1 (within 0.01)
        max_f1 = max(r["mean_cv_f1"] for r in grid_records)
        plateau = [r for r in grid_records if r["mean_cv_f1"] >= max_f1 - 0.005]

        # Check where the plateau lies
        j_plateau = [r["min_jaccard"] for r in plateau]
        c_plateau = [r["max_char_change"] for r in plateau]

        # Specific evaluation of boundary/corner points
        boundary_check = {
            "lower_bound_hit_jaccard": bool(min(j_plateau) <= 0.05),
            "upper_bound_hit_change": bool(max(c_plateau) >= 0.85),
            "plateau_size": int(len(plateau)),
            "jaccard_span": [float(min(j_plateau)), float(max(j_plateau))],
            "char_change_span": [float(min(c_plateau)), float(max(c_plateau))],
            "max_mean_cv_f1": round(float(max_f1), 4),
        }

        print(f"Optimal point on Selection set: min_jaccard={best_params[0]}, max_char_change={best_params[1]}")
        print(f"Selection F1={best_m['sel_f1']}, Eval F1={best_m['eval_f1']}, Mean CV F1={best_m['mean_cv_f1']}")
        print(f"Plateau size: {len(plateau)} points | Jaccard span: {boundary_check['jaccard_span']} | Char change span: {boundary_check['char_change_span']}")

        results[scope_key] = {
            "scope_name": scope_name,
            "best_selection_params": {
                "min_jaccard": best_params[0],
                "max_char_change": best_params[1],
                "selection_metrics": {
                    "f1": best_m["sel_f1"],
                    "precision": best_m["sel_precision"],
                    "recall": best_m["sel_recall"],
                    "fpr": best_m["sel_fpr"],
                },
                "evaluation_metrics": {
                    "f1": best_m["eval_f1"],
                    "precision": best_m["eval_precision"],
                    "recall": best_m["eval_recall"],
                    "fpr": best_m["eval_fpr"],
                },
                "mean_cv_f1": best_m["mean_cv_f1"],
            },
            "provisional_calibrated_heuristic_035_045": {
                "min_jaccard": 0.35,
                "max_char_change": 0.45,
                "selection_metrics": next(r for r in grid_records if r["min_jaccard"] == 0.35 and r["max_char_change"] == 0.45),
            },
            "initial_heuristic_055_025": {
                "min_jaccard": 0.55,
                "max_char_change": 0.25,
                "selection_metrics": next(r for r in grid_records if r["min_jaccard"] == 0.55 and r["max_char_change"] == 0.25),
            },
            "plateau_analysis": boundary_check,
            "top_10_grid_points": sorted(grid_records, key=lambda x: (x["mean_cv_f1"], x["eval_f1"]), reverse=True)[:10],
        }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved expanded grid calibration report to {OUT_REPORT} ({round(time.time() - t0, 2)}s)")


if __name__ == "__main__":
    main()
