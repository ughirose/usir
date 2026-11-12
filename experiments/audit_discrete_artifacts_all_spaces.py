"""USIR v5 Stage B8-DIAG Task 4: Discrete Count Freezing & Artifact Audit across All Spaces.

Audits zero-rate, unique value counts, discrete step sizes, and risk of SS=0 / constantization:
Data scales:
1. D10 Multi-Translation crossed cells (16 cells, 400-800 chars)
2. Class A Micro-polish spans (42 pairs, 30-150 chars)
3. Aozora Bunko 106 works scenes (4,576 scenes)
4. Aozora Bunko 106 works windows (11,723 windows)

Feature spaces:
- 6pole (6 poles)
- Deep 15 features (kamae_*, sadame_*, nagare_*, awase_*)
- ortho28 (28 dims, specifically v2_final_ぬ, v2_final_う, v2_final_す, v2_final_か, v2_quote_ratio, etc.)
- HW-25 (Gamma_UID)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("c:/dev/usir")
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / p))

from fable_v5_grand_shootout import load_works  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.orthographic_prior import (  # noqa: E402
    ORTHO_EXT_V2_NAMES,
    NARR_V2_NAMES,
    ortho_matrix_ext_v2,
    ortho_matrix_narr_v2,
)
from usir_py.v5 import (  # noqa: E402
    AozoraCorpusSplitter,
    aggregate_work_profile_from_scenes,
    build_scene_annotation,
    split_work_into_scenes,
)
from usir_py.v5.narrative_pacing_and_metaphor import extract_hw25_uid_ratio  # noqa: E402
from usir_v5_residual_specificity_level3 import CROSSED_SOURCES, TRANSLATOR_STYLES  # noqa: E402

COMMA_DIMS = [2, 3, 29]
KEEP28 = [i for i in range(31) if i not in COMMA_DIMS]
ALL_ORTHO_NAMES = list(ORTHO_EXT_V2_NAMES) + list(NARR_V2_NAMES)
ORTHO28_NAMES = [ALL_ORTHO_NAMES[i] for i in KEEP28]


def compute_feature_dict(text: str, hook_idf: Any = None) -> Dict[str, float]:
    """Extract all features for a single text into a flat dict."""
    l_text = max(len(text), 1)

    # 1. Scene Annotation & 6pole & deep 15
    sc = build_scene_annotation(text, "author", "work", 0, 1)
    prof = aggregate_work_profile_from_scenes([sc], hook_idf=hook_idf)

    # 6pole
    kasane = (text.count("、") / l_text) * 100.0
    sasae = (text.count("。") / l_text) * 100.0
    kamae = float(prof["kamae"]["hook_survival"])
    sadame = float(prof["sadame"]["total"])
    nagare = float(prof["nagare"]["phi_zeami"])
    awase = float(prof["awase"]["irony"]["mean"])

    # Deep 15 features
    ev_sp = prof["nagare"]["event"].get("speed", [])
    ev_ac = prof["nagare"]["event"].get("accel", [])
    md_sp = prof["nagare"]["mood"].get("speed", [])
    md_ac = prof["nagare"]["mood"].get("accel", [])

    deep15 = {
        "kamae_hook_survival": kamae,
        "kamae_hook_survival_idf": float(prof["kamae"].get("hook_survival_idf", 0.0)),
        "kamae_opened_hooks": float(prof["kamae"].get("opened_hooks", 0.0)),
        "sadame_total": sadame,
        "sadame_e_inter": float(prof["sadame"].get("e_inter", 0.0)),
        "sadame_e_intra": float(prof["sadame"].get("e_intra", 0.0)),
        "sadame_edges_eff": float(prof["sadame"].get("edges_eff", 0.0)),
        "nagare_phi_zeami": nagare,
        "nagare_event_speed": float(np.mean(ev_sp)) if ev_sp else 0.0,
        "nagare_event_accel": float(np.mean(ev_ac)) if ev_ac else 0.0,
        "nagare_mood_speed": float(np.mean(md_sp)) if md_sp else 0.0,
        "nagare_mood_accel": float(np.mean(md_ac)) if md_ac else 0.0,
        "awase_irony_mean": awase,
        "awase_irony_max": float(prof["awase"]["irony"].get("max", 0.0)),
        "awase_detachment_mean": float(prof["awase"]["detachment"].get("mean", 0.0)),
        "awase_detachment_max": float(prof["awase"]["detachment"].get("max", 0.0)),
    }

    # HW-25
    hw25 = float(extract_hw25_uid_ratio(text))

    # ortho28
    ext = ortho_matrix_ext_v2([text])[0]
    narr = ortho_matrix_narr_v2([text])[0]
    full31 = np.concatenate([ext, narr])
    ortho28_vals = full31[KEEP28]

    out = {
        "6pole_kasane": kasane,
        "6pole_sasae": sasae,
        "6pole_kamae": kamae,
        "6pole_sadame": sadame,
        "6pole_nagare": nagare,
        "6pole_awase": awase,
        "hw25_gamma_uid": hw25,
    }
    out.update(deep15)
    for name, val in zip(ORTHO28_NAMES, ortho28_vals):
        out[f"ortho28_{name}"] = float(val)

    return out


def evaluate_series_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Compute zero-rate, unique count, min positive step, and risk level."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "zero_rate_pct": 100.0, "n_unique": 0, "min_positive_val": 0.0, "risk_level": "HIGH"}

    zero_count = int(np.sum(np.abs(arr) < 1e-12))
    zero_rate = (zero_count / n) * 100.0

    uniques = np.unique(np.round(arr, 6))
    n_unique = len(uniques)

    pos_vals = uniques[uniques > 1e-12]
    min_pos = float(np.min(pos_vals)) if len(pos_vals) > 0 else 0.0

    # Risk assessment
    # HIGH: zero_rate > 80% or n_unique <= 3 (critical risk of freezing / binarization / SS=0)
    # MEDIUM: zero_rate between 30% and 80%, or n_unique between 4 and 10
    # LOW: zero_rate < 30% and n_unique > 10 (continuous, healthy variation)
    if zero_rate >= 80.0 or n_unique <= 3:
        risk = "HIGH_ARTIFACT_RISK"
    elif zero_rate >= 30.0 or n_unique <= 10:
        risk = "MEDIUM_DISCRETE_STEP"
    else:
        risk = "LOW_CONTINUOUS"

    return {
        "n_samples": n,
        "zero_count": zero_count,
        "zero_rate_pct": round(zero_rate, 2),
        "n_unique": n_unique,
        "min_positive_val": round(min_pos, 6),
        "mean_val": round(float(np.mean(arr)), 6),
        "std_val": round(float(np.std(arr)), 6),
        "risk_level": risk,
    }


def main() -> None:
    t0 = time.time()
    print("=== USIR v5 Stage B8-DIAG Task 4: Discrete Freezing & Artifact Audit ===")

    profile_json = REPO_ROOT / "docs" / "handoff" / "v10" / "nishiki_profile_106works.json"
    hook_idf = json.loads(profile_json.read_text(encoding="utf-8")).get("hook_idf") if profile_json.exists() else None

    # 1. Dataset 1: D10 Multi-Translation (16 texts)
    print("\n[1/4] Collecting D10 Multi-Translation (16 cells, 400-800 chars)...")
    d10_texts = []
    for src in CROSSED_SOURCES:
        for trans in TRANSLATOR_STYLES:
            paras = [trans["transform"](ev[0]) for ev in src["events"]]
            d10_texts.append("\n".join(paras))
    print(f"Collected {len(d10_texts)} D10 texts.")

    # 2. Dataset 2: Class A Micro-polish Spans (42 pairs, 30-150 chars)
    print("\n[2/4] Collecting Class A Micro-Polish Spans (42 pairs, 30-150 chars)...")
    ds_path = REPO_ROOT / "data" / "usir_v5_eval" / "u6_calibration_and_level3_dataset.jsonl"
    class_a_texts = []
    with open(ds_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["class"] == "Class_A_micro_polish":
                    class_a_texts.append(item["text_before"])
                    class_a_texts.append(item["text_after"])
    print(f"Collected {len(class_a_texts)} Class A span texts.")

    # 3. Dataset 3 & 4: Aozora Bunko 106 Works Scenes (4,576) and Windows (11,723)
    print("\n[3/4 & 4/4] Collecting Aozora 106 Works Scenes and Windows...")
    works = load_works()
    splitter = AozoraCorpusSplitter()
    aozora_scenes = []
    aozora_windows = []

    for w in works:
        txt = splitter.get_cached_text(w)
        scs = split_work_into_scenes(txt)
        aozora_scenes.extend(scs)
        wins = sa.make_windows(w["text"])
        aozora_windows.extend(wins)

    print(f"Collected {len(aozora_scenes)} Aozora scenes (expected 4576).")
    print(f"Collected {len(aozora_windows)} Aozora windows (expected 11723).")

    cohorts = [
        ("D10_multi_trans_16", d10_texts),
        ("Class_A_micro_84", class_a_texts),
        ("Aozora_scenes_4576", aozora_scenes),
        ("Aozora_windows_11723", aozora_windows),
    ]

    # Feature extraction across cohorts
    audit_results: Dict[str, Dict[str, Any]] = {}

    for cohort_name, text_list in cohorts:
        t_c0 = time.time()
        print(f"\nProcessing cohort: {cohort_name} ({len(text_list)} texts)...")

        # Collect features
        feature_rows: Dict[str, List[float]] = {}
        for i, txt in enumerate(text_list):
            fdict = compute_feature_dict(txt, hook_idf=hook_idf)
            for k, v in fdict.items():
                if k not in feature_rows:
                    feature_rows[k] = []
                feature_rows[k].append(v)

            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{len(text_list)} texts in {time.time() - t_c0:.1f}s...")

        # Compute stats per feature
        cohort_stats = {}
        for fname, vals in feature_rows.items():
            cohort_stats[fname] = evaluate_series_statistics(vals)

        audit_results[cohort_name] = cohort_stats
        print(f"Completed {cohort_name} in {time.time() - t_c0:.2f}s.")

    # High-risk summary per cohort
    high_risk_summary: Dict[str, List[str]] = {}
    for cohort_name, cstats in audit_results.items():
        high_risk_features = [fname for fname, st in cstats.items() if st["risk_level"] == "HIGH_ARTIFACT_RISK"]
        high_risk_summary[cohort_name] = high_risk_features
        print(f"\nCohort {cohort_name}: {len(high_risk_features)} HIGH RISK features (out of {len(cstats)})")

    # Output JSON
    output_json = REPO_ROOT / "data" / "usir_v5_eval" / "discrete_artifact_audit_all_spaces.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "title": "Discrete Count Freezing & Artifact Audit across All Spaces",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cohorts": [c[0] for c in cohorts],
                "n_features": len(list(audit_results.values())[0]),
            },
            "high_risk_summary": high_risk_summary,
            "cohort_statistics": audit_results,
        }, f, ensure_ascii=False, indent=2)

    # Output Markdown Table
    md_file = REPO_ROOT / "docs" / "handoff" / "2026-11-12_discrete_artifact_audit_table.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# 全極・全空間 離散カウント固定化（ゼロ張り付き・2値化）総点検レポート\n\n")
        f.write("点検対象スケール:\n")
        f.write("- ① D10 多訳データ (16件, 400〜800字)\n")
        f.write("- ② Class A ミクロ推敲スパン (84件, 30〜150字)\n")
        f.write("- ③ 青空文庫 106作品 場面 (4,576件)\n")
        f.write("- ④ 青空文庫 106作品 スライディング窓 (11,723件)\n\n")

        f.write("## 1. 高リスク特徴量（短文・小窓でのゼロ張り付き率 > 80% またはユニーク値 <= 3）\n\n")
        f.write("| 特徴量名 | カテゴリ | D10 ゼロ率 (ユニーク数) | Class A ゼロ率 (ユニーク数) | 4576場面 ゼロ率 | 11723窓 ゼロ率 | リスク要因・機構解釈 |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        sample_features = [
            ("6pole_kamae", "6pole", "フック生存率（開閉フックの離散比）"),
            ("kamae_hook_survival", "Deep15", "フック生存率（短文で開フック0または1に固定）"),
            ("kamae_opened_hooks", "Deep15", "開フック数（短文で0〜1カウント）"),
            ("awase_detachment_mean", "Deep15", "視点乖離度（短文で特定助詞欠如時0）"),
            ("sadame_e_inter", "Deep15", "場面間エッジ（単一場面・短文では定義上0固定）"),
            ("ortho28_v2_final_ぬ", "ortho28", "文末助動詞「ぬ」（短文でほぼ出現せず0張り付き）"),
            ("ortho28_v2_final_う", "ortho28", "文末助動詞「う」（短文でほぼ出現せず0張り付き）"),
            ("ortho28_v2_final_す", "ortho28", "文末助動詞「す」（短文でほぼ出現せず0張り付き）"),
            ("ortho28_v2_final_か", "ortho28", "文末助動詞「か」（短文でほぼ出現せず0張り付き）"),
            ("ortho28_v2_quote_ratio", "ortho28", "会話文比率（会話文のない地文短文で0固定）"),
            ("hw25_gamma_uid", "HW-25", "UID比率（1文・短文でステップ幅粗大）"),
            ("6pole_kasane", "6pole", "読点密度（連続値、低リスク）"),
            ("6pole_sasae", "6pole", "句点密度（短文でも文末に必ず1個あり）"),
            ("6pole_nagare", "6pole", "世阿弥エントロピー（連続値）"),
            ("6pole_awase", "6pole", "アイロニー平均（感情極性反転）"),
        ]

        for fname, cat, note in sample_features:
            d10_st = audit_results["D10_multi_trans_16"].get(fname, {})
            cla_st = audit_results["Class_A_micro_84"].get(fname, {})
            sc_st = audit_results["Aozora_scenes_4576"].get(fname, {})
            win_st = audit_results["Aozora_windows_11723"].get(fname, {})

            d10_str = f"{d10_st.get('zero_rate_pct', 0):.1f}% ({d10_st.get('n_unique', 0)})"
            cla_str = f"{cla_st.get('zero_rate_pct', 0):.1f}% ({cla_st.get('n_unique', 0)})"
            sc_str = f"{sc_st.get('zero_rate_pct', 0):.1f}% ({sc_st.get('n_unique', 0)})"
            win_str = f"{win_st.get('zero_rate_pct', 0):.1f}% ({win_st.get('n_unique', 0)})"

            f.write(f"| `{fname}` | {cat} | {d10_str} | {cla_str} | {sc_str} | {win_str} | {note} |\n")

        f.write("\n## 2. 全 50 特徴量の完全点検マトリクス\n\n")
        f.write("| 特徴量名 | D10 ゼロ率 / ユニーク数 | Class A ゼロ率 / ユニーク数 | 4576場面 ゼロ率 / ユニーク数 | 11723窓 ゼロ率 / ユニーク数 | リスク判定 |\n")
        f.write("|---|---|---|---|---|---|\n")

        for fname in sorted(list(audit_results["Aozora_scenes_4576"].keys())):
            d10_st = audit_results["D10_multi_trans_16"].get(fname, {})
            cla_st = audit_results["Class_A_micro_84"].get(fname, {})
            sc_st = audit_results["Aozora_scenes_4576"].get(fname, {})
            win_st = audit_results["Aozora_windows_11723"].get(fname, {})

            r_d10 = f"{d10_st.get('zero_rate_pct', 0):.1f}% / {d10_st.get('n_unique', 0)}"
            r_cla = f"{cla_st.get('zero_rate_pct', 0):.1f}% / {cla_st.get('n_unique', 0)}"
            r_sc = f"{sc_st.get('zero_rate_pct', 0):.1f}% / {sc_st.get('n_unique', 0)}"
            r_win = f"{win_st.get('zero_rate_pct', 0):.1f}% / {win_st.get('n_unique', 0)}"
            overall_risk = cla_st.get("risk_level", "UNKNOWN")

            f.write(f"| `{fname}` | {r_d10} | {r_cla} | {r_sc} | {r_win} | {overall_risk} |\n")

    print(f"\nWrote full discrete artifact audit to {output_json} and {md_file} ({round(time.time() - t0, 2)}s)")


if __name__ == "__main__":
    main()
