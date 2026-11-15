#!/usr/bin/env python
"""USIR v5 Stage B-22 Task 1: IDF-Weighted Hazard Survival (S_haz_idf) vs Unweighted S_haz.

Read-only analysis script (does NOT modify production Python, Rust, or frozen artifacts).
Outputs deterministic verification results to `b22_task1_idf_hazard.json`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "python" / "scripts" / "b20_kamae"))

import run_b20_kamae_analysis as b20  # noqa: E402
from usir_py.v5.aozora_scene_splitter import (  # noqa: E402
    AozoraCorpusSplitter,
    split_work_into_scenes,
)

OUT_JSON = Path(__file__).resolve().parent / "b22_task1_idf_hazard.json"


def calc_hazard_survival_detail(
    sc_list: List[Dict[str, Any]],
    horizon: Optional[int],
    h_ref: int = 5,
    prior_tau: float = 1.0,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute S_haz and intermediate C_H, E_H (unweighted or 4-quartile IDF-weighted)."""
    n, op, cl = b20.extract_hook_events(sc_list)
    if not op:
        return {"s_haz": 0.0, "c_h": 0.0, "e_h": 0.0, "opened": 0, "closed_total": 0}
    e_sum = 0.0
    c_sum = 0.0
    for hid, o_idx in op.items():
        b = min(3, (o_idx * 4) // max(n, 1))
        w = weights[min(b, len(weights) - 1)] if weights is not None else 1.0
        c_idx = cl.get(hid)
        max_rem = max(0, n - 1 - o_idx)
        if horizon is not None:
            max_rem = min(max_rem, horizon)
        if c_idx is not None and (horizon is None or (c_idx - o_idx) <= horizon):
            e_sum += w * float(c_idx - o_idx)
            c_sum += w
        else:
            e_sum += w * float(max_rem)
    denom = e_sum + prior_tau
    lam = c_sum / denom if denom > 0.0 else 0.0
    s_step = max(0.0, min(1.0, 1.0 - lam))
    return {
        "s_haz": float(s_step ** h_ref),
        "c_h": float(c_sum),
        "e_h": float(e_sum),
        "opened": int(len(op)),
        "closed_total": int(len(cl)),
    }


def evaluate_vector(
    vec: np.ndarray,
    log_chars_py: np.ndarray,
    log_chars_nonws: np.ndarray,
    log_tw: np.ndarray,
    log_sents: np.ndarray,
    authors: List[str],
) -> Dict[str, Any]:
    return {
        "mean": float(vec.mean()),
        "std": float(vec.std()),
        "min": float(vec.min()),
        "max": float(vec.max()),
        "n_unique": int(len(np.unique(np.round(vec, 6)))),
        "n_at_1": int(np.sum(vec >= 1.0 - 1e-9)),
        "vs_log_chars_py": b20.corr_pair(vec, log_chars_py),
        "vs_log_chars_nonws": b20.corr_pair(vec, log_chars_nonws),
        "vs_log_scene_count": b20.corr_pair(vec, log_tw),
        "vs_log_sentence_count": b20.corr_pair(vec, log_sents),
        "nhsic_vs_log_chars_py": b20.nhsic_vs_log_chars(vec, log_chars_py),
        "nhsic_vs_log_scene_count": b20.nhsic_vs_log_chars(vec, log_tw),
        "nhsic_vs_log_sentence_count": b20.nhsic_vs_log_chars(vec, log_sents),
        "author_discrim": b20.author_discrimination(vec, authors),
    }


def main() -> None:
    works = b20.load_works()
    w_by_wid = {w["id"].split(":")[-1]: w for w in works}
    wids = [w["id"].split(":")[-1] for w in works]
    authors = [w["author_id"] for w in works]

    prof_rust = json.loads(b20.NISHIKI_PROFILE_JSON.read_text(encoding="utf-8"))
    followup = json.loads(b20.FOLLOWUP_JSON.read_text(encoding="utf-8"))
    labels = followup["labels"]
    rust_works_by_wid = {rw["work_id"]: rw for rw in prof_rust["works"]}

    scenes_by_work: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with b20.SCENES_JSONL.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            scenes_by_work[s["work_id"]].append(s)
    for wid in scenes_by_work:
        scenes_by_work[wid].sort(key=lambda s: s["scene_index"])

    hook_idf = b20.compute_rust_hook_idf(scenes_by_work)
    assert max(abs(a - b) for a, b in zip(hook_idf["weight"], prof_rust["hook_idf"]["weight"])) < 1e-15
    idf_weights = list(prof_rust["hook_idf"]["weight"])

    log_chars_py = np.array([math.log(len(w_by_wid[wid]["text"])) for wid in wids], dtype=float)
    log_chars_nonws = np.array(
        [math.log(b20.count_non_whitespace_chars(w_by_wid[wid]["text"])) for wid in wids], dtype=float
    )
    log_tw = np.array([math.log(len(scenes_by_work[wid])) for wid in wids], dtype=float)
    log_sents = np.array(
        [math.log(b20.count_rust_sentences(w_by_wid[wid]["text"])) for wid in wids], dtype=float
    )

    # Full 106-work numerical table
    per_work_table: List[Dict[str, Any]] = []
    vecs: Dict[str, List[float]] = {
        "H10_unweighted": [],
        "H10_idf_weighted": [],
        "H15_unweighted": [],
        "H15_idf_weighted": [],
    }
    for wid in wids:
        scs = scenes_by_work[wid]
        rw = rust_works_by_wid[wid]
        w_meta = w_by_wid[wid]
        d10_u = calc_hazard_survival_detail(scs, 10, h_ref=5, prior_tau=1.0, weights=None)
        d10_w = calc_hazard_survival_detail(scs, 10, h_ref=5, prior_tau=1.0, weights=idf_weights)
        d15_u = calc_hazard_survival_detail(scs, 15, h_ref=5, prior_tau=1.0, weights=None)
        d15_w = calc_hazard_survival_detail(scs, 15, h_ref=5, prior_tau=1.0, weights=idf_weights)

        vecs["H10_unweighted"].append(d10_u["s_haz"])
        vecs["H10_idf_weighted"].append(d10_w["s_haz"])
        vecs["H15_unweighted"].append(d15_u["s_haz"])
        vecs["H15_idf_weighted"].append(d15_w["s_haz"])

        per_work_table.append(
            {
                "work_id": wid,
                "author_id": w_meta["author_id"],
                "author_ja": labels.get(w_meta["author_id"], ""),
                "title": w_meta.get("title", ""),
                "T_W": len(scs),
                "char_len_py": len(w_meta["text"]),
                "char_len_nonws": b20.count_non_whitespace_chars(w_meta["text"]),
                "rust_sentences": b20.count_rust_sentences(w_meta["text"]),
                "opened_hooks": d10_u["opened"],
                "closed_hooks": d10_u["closed_total"],
                "raw_hook_survival": float(rw["kamae"]["hook_survival"]),
                "raw_hook_survival_idf": float(rw["kamae"]["hook_survival_idf"]),
                "S_haz_H10": d10_u["s_haz"],
                "S_haz_idf_H10": d10_w["s_haz"],
                "S_haz_H15": d15_u["s_haz"],
                "S_haz_idf_H15": d15_w["s_haz"],
                "C_H10": d10_u["c_h"],
                "E_H10": d10_u["e_h"],
                "C_H10_idf": d10_w["c_h"],
                "E_H10_idf": d10_w["e_h"],
                "C_H15": d15_u["c_h"],
                "E_H15": d15_u["e_h"],
                "C_H15_idf": d15_w["c_h"],
                "E_H15_idf": d15_w["e_h"],
            }
        )

    summary = {}
    for k, vals in vecs.items():
        summary[k] = evaluate_vector(
            np.array(vals, dtype=float), log_chars_py, log_chars_nonws, log_tw, log_sents, authors
        )

    # Pairwise parity between unweighted and IDF-weighted
    pairwise_diff = {}
    for h_tag in ["H10", "H15"]:
        u = np.array(vecs[f"{h_tag}_unweighted"], dtype=float)
        w = np.array(vecs[f"{h_tag}_idf_weighted"], dtype=float)
        diff = np.abs(u - w)
        pairwise_diff[h_tag] = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(np.median(diff)),
            "spearman_rho": float(spearmanr(u, w).statistic),
            "pearson_r": float(pearsonr(u, w).statistic),
        }

    # Prefix truncation experiment (15 works >= 100 scenes, 7 works >= 200 scenes)
    splitter = AozoraCorpusSplitter(data_dir=ROOT / "data")
    manifests = splitter.load_manifests()
    manifests_by_wid = {m["work_id"]: m for m in manifests}
    long_wids_100 = sorted(
        [wid for wid in wids if len(scenes_by_work[wid]) >= 100],
        key=lambda x: len(scenes_by_work[x]),
        reverse=True,
    )
    long_wids_200 = [wid for wid in long_wids_100 if len(scenes_by_work[wid]) >= 200]
    long_sc_texts = {
        wid: split_work_into_scenes(splitter.get_cached_text(manifests_by_wid[wid]))
        for wid in long_wids_100
    }
    ks = [5, 10, 20, 40, 80, 100, 150, 200]
    trunc_summary: List[Dict[str, Any]] = []
    per_work_200_traces: Dict[str, Any] = {
        wid: {
            "title": w_by_wid[wid].get("title", ""),
            "author_ja": labels.get(w_by_wid[wid]["author_id"], ""),
            "full_T_W": len(scenes_by_work[wid]),
            "steps": [],
        }
        for wid in long_wids_200
    }

    for k in ks:
        wids_for_k = [wid for wid in long_wids_100 if len(scenes_by_work[wid]) >= k]
        acc: Dict[str, List[float]] = defaultdict(list)
        for wid in wids_for_k:
            sub_slice = b20.slice_jsonl_prefix(scenes_by_work[wid], k)
            recs = splitter.process_work_scenes(manifests_by_wid[wid], long_sc_texts[wid][:k])

            row_vals: Dict[str, float] = {}
            for h_val in [10, 15]:
                for mode_name, wts in [("unw", None), ("idf", idf_weights)]:
                    s_val = calc_hazard_survival_detail(sub_slice, h_val, 5, 1.0, wts)["s_haz"]
                    r_val = calc_hazard_survival_detail(recs, h_val, 5, 1.0, wts)["s_haz"]
                    acc[f"slice_H{h_val}_{mode_name}"].append(s_val)
                    acc[f"reannot_H{h_val}_{mode_name}"].append(r_val)
                    row_vals[f"slice_H{h_val}_{mode_name}"] = round(s_val, 4)
                    row_vals[f"reannot_H{h_val}_{mode_name}"] = round(r_val, 4)
            if wid in per_work_200_traces:
                per_work_200_traces[wid]["steps"].append({"K": k, **row_vals})

        step_row: Dict[str, Any] = {"K": k, "n_works": len(wids_for_k)}
        for key_name, arr in acc.items():
            step_row[f"{key_name}_mean"] = float(np.mean(arr))
            step_row[f"{key_name}_median"] = float(np.median(arr))
        trunc_summary.append(step_row)

    drift_7works: Dict[str, float] = {}
    for metric_key in [
        "slice_H10_unw",
        "slice_H10_idf",
        "reannot_H10_unw",
        "reannot_H10_idf",
        "slice_H15_unw",
        "slice_H15_idf",
        "reannot_H15_unw",
        "reannot_H15_idf",
    ]:
        diffs = [
            t["steps"][2][metric_key] - t["steps"][-1][metric_key]
            for t in per_work_200_traces.values()
        ]
        drift_7works[f"{metric_key}_20_minus_200_mean"] = float(np.mean(diffs))
        drift_7works[f"{metric_key}_20_minus_200_abs_mean"] = float(np.mean(np.abs(diffs)))

    results = {
        "meta": {
            "script": "python/scripts/b22_kamae_haz/task1_idf_hazard.py",
            "scenes_jsonl_sha256": hashlib.sha256(b20.SCENES_JSONL.read_bytes()).hexdigest().upper(),
            "n_works": len(wids),
            "h_ref": 5,
            "prior_tau": 1.0,
            "hook_idf": prof_rust["hook_idf"],
        },
        "summary_comparison": summary,
        "pairwise_unweighted_vs_idf": pairwise_diff,
        "truncation_summary_15works": trunc_summary,
        "truncation_drift_7works_20_to_200": drift_7works,
        "truncation_200_traces_7works": per_work_200_traces,
        "per_work_table": per_work_table,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    for k, v in summary.items():
        print(
            f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}, uniq={v['n_unique']}, at1={v['n_at_1']}, "
            f"rho_chars={v['vs_log_chars_py']['spearman_rho']:.4f}, r_chars={v['vs_log_chars_py']['pearson_r']:.4f}, "
            f"rho_tw={v['vs_log_scene_count']['spearman_rho']:.4f}, rho_sents={v['vs_log_sentence_count']['spearman_rho']:.4f}, "
            f"nHSIC={v['nhsic_vs_log_chars_py']:.4f}, eta2={v['author_discrim']['eta2_author']:.4f}"
        )
    print("Pairwise diffs:", json.dumps(pairwise_diff, indent=2))
    print("Drift 7 works:", json.dumps(drift_7works, indent=2))


if __name__ == "__main__":
    main()
