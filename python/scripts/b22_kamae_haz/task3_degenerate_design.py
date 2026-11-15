#!/usr/bin/env python
"""USIR v5 Stage B-22 Task 3: Degenerate Case (E_H=0 / T_W=1) & Sentence-Exposure Design.

Read-only analysis script (does NOT modify production Python, Rust, or frozen artifacts).
Outputs deterministic verification results to `b22_task3_degenerate_and_sent_exposure.json`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import f_oneway, kruskal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "python" / "scripts" / "b20_kamae"))
sys.path.insert(0, str(ROOT / "python" / "scripts" / "b22_kamae_haz"))

import run_b20_kamae_analysis as b20  # noqa: E402
from task1_idf_hazard import calc_hazard_survival_detail, evaluate_vector  # noqa: E402
from usir_py.v5.aozora_scene_splitter import (  # noqa: E402
    COMMON_HOOK_STOPWORDS,
    RE_HOOK_TRIG,
    RE_RESOLV_MARKER,
    SENT_SPLIT,
    AozoraCorpusSplitter,
    compute_work_scene_frequencies,
    extract_kanji_tokens,
    filter_hook_tokens_by_isf,
    split_work_into_scenes,
    strip_commas,
)
from usir_v5_residual_specificity_level3 import (  # noqa: E402
    CROSSED_SOURCES,
    TRANSLATOR_STYLES,
    build_d10_cells,
)

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "b22_task3_degenerate_and_sent_exposure.json"
REANNOT_A_JSONL = HERE / "reannot_isf_a_current.jsonl"
REANNOT_C_JSONL = HERE / "reannot_isf_c_corpus.jsonl"


def load_scenes_jsonl(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    scenes_by_work: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            scenes_by_work[rec["work_id"]].append(rec)
    for wid in scenes_by_work:
        scenes_by_work[wid].sort(key=lambda s: s["scene_index"])
    return scenes_by_work


def check_event_collection_parity(scenes_by_work: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Check whether any work opens or closes a hook_id multiple times and compare Rust/Py collect logic."""
    dup_opens = 0
    dup_closes = 0
    max_abs_diff_h10 = 0.0
    max_abs_diff_h15 = 0.0
    for wid, sc_list in scenes_by_work.items():
        n = len(sc_list)
        op_py: Dict[int, int] = {}
        cl_py: Dict[int, int] = {}
        # Mirror Rust BTreeMap or_insert semantics exactly
        op_rust: Dict[int, Tuple[int, Optional[int]]] = {}
        for s in sc_list:
            idx = int(s["scene_index"])
            km = s.get("kamae", {})
            for h in km.get("opened_hooks", []):
                hid = int(h["id"])
                if hid in op_py:
                    dup_opens += 1
                else:
                    op_py[hid] = idx
                    op_rust[hid] = (idx, None)
            for cid in km.get("closed_hook_ids", []):
                hid = int(cid)
                if hid in cl_py:
                    dup_closes += 1
                cl_py[hid] = idx
                if hid in op_rust and op_rust[hid][1] is None:
                    op_rust[hid] = (op_rust[hid][0], idx)

        for horizon in [10, 15]:
            # Compare against task1 calc_hazard_survival_detail
            d_py = calc_hazard_survival_detail(sc_list, horizon, 5, 1.0, None)["s_haz"]
            if not op_rust:
                d_ru = 0.0
            else:
                e_sum = 0.0
                c_sum = 0.0
                for hid, (o_idx, c_idx) in op_rust.items():
                    max_rem = min(max(0, n - 1 - o_idx), horizon)
                    if c_idx is not None and (c_idx - o_idx) <= horizon:
                        e_sum += float(c_idx - o_idx)
                        c_sum += 1.0
                    else:
                        e_sum += float(max_rem)
                lam = c_sum / (e_sum + 1.0)
                d_ru = (1.0 - lam) ** 5
            diff = abs(d_py - d_ru)
            if horizon == 10:
                max_abs_diff_h10 = max(max_abs_diff_h10, diff)
            else:
                max_abs_diff_h15 = max(max_abs_diff_h15, diff)

    return {
        "duplicate_opened_hook_ids": dup_opens,
        "duplicate_closed_hook_ids": dup_closes,
        "max_abs_diff_rust_vs_py_collect_H10": max_abs_diff_h10,
        "max_abs_diff_rust_vs_py_collect_H15": max_abs_diff_h15,
    }


def hook_exposure_census(
    dataset_name: str,
    scenes_by_work: Dict[str, List[Dict[str, Any]]],
    wids: List[str],
    w_by_wid: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    tw_list: List[int] = []
    opened_list: List[int] = []
    e10_list: List[float] = []
    e15_list: List[float] = []
    c10_list: List[float] = []
    c15_list: List[float] = []

    zero_exp_hooks = 0
    total_opened_hooks = 0
    works_with_zero_exp_hook = 0
    zero_exp_works_h10 = 0
    zero_exp_works_h15 = 0
    zero_open_works = 0
    tw1_works = 0
    ch10_zero_works: List[Dict[str, Any]] = []
    ch15_zero_works: List[Dict[str, Any]] = []
    work_rows: List[Dict[str, Any]] = []

    for wid in wids:
        sc_list = scenes_by_work[wid]
        n = len(sc_list)
        tw_list.append(n)
        if n == 1:
            tw1_works += 1
        _, op, cl = b20.extract_hook_events(sc_list)
        d10 = calc_hazard_survival_detail(sc_list, 10, 5, 1.0, None)
        d15 = calc_hazard_survival_detail(sc_list, 15, 5, 1.0, None)

        opened_list.append(len(op))
        e10_list.append(d10["e_h"])
        e15_list.append(d15["e_h"])
        c10_list.append(d10["c_h"])
        c15_list.append(d15["c_h"])

        if len(op) == 0:
            zero_open_works += 1
        if d10["e_h"] == 0.0:
            zero_exp_works_h10 += 1
        if d15["e_h"] == 0.0:
            zero_exp_works_h15 += 1

        w_zero_hooks = 0
        for hid, o_idx in op.items():
            total_opened_hooks += 1
            if o_idx >= n - 1 and hid not in cl:
                w_zero_hooks += 1
                zero_exp_hooks += 1
        if w_zero_hooks > 0:
            works_with_zero_exp_hook += 1

        row = {
            "work_id": wid,
            "title": w_by_wid[wid].get("title", ""),
            "author_id": w_by_wid[wid]["author_id"],
            "T_W": n,
            "opened": len(op),
            "closed_total": len(cl),
            "C_H10": d10["c_h"],
            "E_H10": d10["e_h"],
            "S_haz_H10": d10["s_haz"],
            "C_H15": d15["c_h"],
            "E_H15": d15["e_h"],
            "S_haz_H15": d15["s_haz"],
            "zero_exp_hooks": w_zero_hooks,
        }
        work_rows.append(row)
        if d10["c_h"] == 0.0:
            ch10_zero_works.append(row)
        if d15["c_h"] == 0.0:
            ch15_zero_works.append(row)

    def qstats(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=float)
        return {
            "min": float(np.min(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p25": float(np.quantile(arr, 0.25)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p75": float(np.quantile(arr, 0.75)),
            "max": float(np.max(arr)),
        }

    smallest_e10 = sorted(work_rows, key=lambda r: (r["E_H10"], r["T_W"], r["work_id"]))[:12]

    return {
        "dataset": dataset_name,
        "n_works": len(wids),
        "n_works_T_W_eq_1": tw1_works,
        "n_works_N_open_eq_0": zero_open_works,
        "n_works_E_H10_eq_0": zero_exp_works_h10,
        "n_works_E_H15_eq_0": zero_exp_works_h15,
        "n_works_C_H10_eq_0_at_1": len(ch10_zero_works),
        "n_works_C_H15_eq_0_at_1": len(ch15_zero_works),
        "total_opened_hooks": total_opened_hooks,
        "zero_exposure_hooks_e_eq_0": zero_exp_hooks,
        "zero_exposure_hook_rate": zero_exp_hooks / max(total_opened_hooks, 1),
        "n_works_with_at_least_one_zero_exp_hook": works_with_zero_exp_hook,
        "quantiles": {
            "T_W": qstats([float(x) for x in tw_list]),
            "N_open": qstats([float(x) for x in opened_list]),
            "E_H10": qstats(e10_list),
            "E_H15": qstats(e15_list),
        },
        "smallest_E_H10_works": smallest_e10,
        "zero_closure_H10_work_ids": [r["work_id"] for r in ch10_zero_works],
        "zero_closure_H15_work_ids": [r["work_id"] for r in ch15_zero_works],
    }


def evaluate_filtered_subset(
    vals: np.ndarray,
    mask: np.ndarray,
    log_chars_py: np.ndarray,
    log_chars_nonws: np.ndarray,
    log_tw: np.ndarray,
    log_sents: np.ndarray,
    authors: List[str],
    wids: List[str],
) -> Dict[str, Any]:
    idx = np.where(mask)[0]
    sub_vals = vals[idx]
    sub_auth = [authors[i] for i in idx]
    excluded_wids = [wids[i] for i in range(len(wids)) if not mask[i]]

    if len(sub_vals) < 3 or np.std(sub_vals) < 1e-12:
        return {
            "n_retained": int(len(sub_vals)),
            "n_excluded": int(len(wids) - len(sub_vals)),
            "excluded_wids": excluded_wids,
            "degenerate": True,
        }

    sub_lc = log_chars_py[idx]
    sub_lcn = log_chars_nonws[idx]
    sub_ltw = log_tw[idx]
    sub_ls = log_sents[idx]

    by_auth: Dict[str, List[float]] = defaultdict(list)
    for v, a in zip(sub_vals, sub_auth):
        by_auth[a].append(float(v))
    groups = [np.array(vs, dtype=float) for vs in by_auth.values() if len(vs) > 0]
    n_nonempty_authors = len(groups)
    # Guard eta2/anova when an author has < 2 samples or < 2 non-singleton groups
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) >= 2:
        f_stat, anova_p = f_oneway(*groups)
        k_stat, kruskal_p = kruskal(*groups)
        gm = float(np.mean(sub_vals))
        ss_b = sum(len(g) * (float(np.mean(g)) - gm) ** 2 for g in groups)
        ss_t = float(np.sum((sub_vals - gm) ** 2))
        eta2 = float(ss_b / ss_t) if ss_t > 0.0 else 0.0
        auth_disc = {
            "n_authors_represented": n_nonempty_authors,
            "eta2_author": eta2,
            "anova_f": float(f_stat),
            "anova_p": float(anova_p),
            "kruskal_h": float(k_stat),
            "kruskal_p": float(kruskal_p),
        }
    else:
        auth_disc = {"n_authors_represented": n_nonempty_authors, "eta2_author": None}

    return {
        "n_retained": int(len(sub_vals)),
        "n_excluded": int(len(wids) - len(sub_vals)),
        "excluded_wids": excluded_wids,
        "mean": float(sub_vals.mean()),
        "std": float(sub_vals.std()),
        "min": float(sub_vals.min()),
        "max": float(sub_vals.max()),
        "n_unique": int(len(np.unique(np.round(sub_vals, 6)))),
        "n_at_1": int(np.sum(sub_vals >= 1.0 - 1e-9)),
        "vs_log_chars_py": b20.corr_pair(sub_vals, sub_lc),
        "vs_log_chars_nonws": b20.corr_pair(sub_vals, sub_lcn),
        "vs_log_scene_count": b20.corr_pair(sub_vals, sub_ltw),
        "vs_log_sentence_count": b20.corr_pair(sub_vals, sub_ls),
        "nhsic_vs_log_chars_py": b20.nhsic_vs_log_chars(sub_vals, sub_lc),
        "nhsic_vs_log_scene_count": b20.nhsic_vs_log_chars(sub_vals, sub_ltw),
        "nhsic_vs_log_sentence_count": b20.nhsic_vs_log_chars(sub_vals, sub_ls),
        "author_discrim": auth_disc,
    }


def extract_and_score_sentence_exposure(
    sc_texts: List[str],
    isf_mode: str,
    sub_open_sc: bool,
    horizon_sent: int,
    h_ref_sent: int,
    prior_tau: float,
    corpus_sf: Optional[Dict[str, int]] = None,
    total_corpus_scenes: Optional[int] = None,
) -> Dict[str, Any]:
    """Candidate (iii): Sentence-index distance exposure S_haz_sent on scene list sc_texts."""
    tw = len(sc_texts)
    wsf = compute_work_scene_frequencies(sc_texts)
    sent_records: List[Tuple[int, int, str]] = []  # (global_sent_idx, sc_idx, clean_sent)
    opened_hooks: List[Dict[str, Any]] = []
    total_trigs = 0
    total_resolv_sents = 0

    g_idx = 0
    for sc_idx, raw_sc in enumerate(sc_texts):
        clean_sc = strip_commas(raw_sc)
        sents = [s.strip() for s in SENT_SPLIT.split(clean_sc) if s.strip()]
        sc_opened = 0
        for cs in sents:
            sent_records.append((g_idx, sc_idx, cs))
            if RE_RESOLV_MARKER.search(cs):
                total_resolv_sents += 1
            if RE_HOOK_TRIG.search(cs):
                total_trigs += 1
                if sc_opened < 2:
                    label = cs[:30]
                    ktoks = extract_kanji_tokens(label)
                    if isf_mode == "c_corpus_fixed" and corpus_sf is not None and total_corpus_scenes is not None:
                        spec = filter_hook_tokens_by_isf(
                            ktoks, corpus_sf, total_scenes=total_corpus_scenes, isf_threshold=1.2
                        )
                    else:
                        spec = filter_hook_tokens_by_isf(ktoks, wsf, total_scenes=tw, isf_threshold=1.2)
                    if not spec:
                        fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                        final_spec = fb if fb else {cs[:4]}
                    else:
                        final_spec = spec
                    opened_hooks.append(
                        {
                            "open_sent_idx": g_idx,
                            "open_sc_idx": sc_idx,
                            "tokens": final_spec,
                            "close_sent_idx": None,
                        }
                    )
                    sc_opened += 1
            g_idx += 1

    s_w = len(sent_records)
    if not opened_hooks:
        return {
            "s_haz_sent": None,
            "s_haz_sent_fallback": 0.0,
            "S_W": s_w,
            "T_W": tw,
            "total_trigs": total_trigs,
            "total_resolv_sents": total_resolv_sents,
            "opened": 0,
            "closed": 0,
            "c_sent": 0.0,
            "e_sent": 0.0,
            "n_zero_exp_hooks": 0,
        }

    # Resolve hooks sequentially across subsequent sentences
    active = list(opened_hooks)
    for g_s, sc_idx, cs in sent_records:
        if not RE_RESOLV_MARKER.search(cs):
            continue
        for h in list(active):
            if g_s <= h["open_sent_idx"]:
                continue
            if sub_open_sc and sc_idx <= h["open_sc_idx"]:
                continue
            if any(tok in cs for tok in h["tokens"]):
                h["close_sent_idx"] = g_s
                active.remove(h)

    e_sum = 0.0
    c_sum = 0.0
    n_zero_exp_hooks = 0
    closed_total = 0
    for h in opened_hooks:
        o_s = h["open_sent_idx"]
        c_s = h["close_sent_idx"]
        max_rem = min(max(0, s_w - 1 - o_s), horizon_sent)
        if c_s is not None and (c_s - o_s) <= horizon_sent:
            e_i = float(c_s - o_s)
            c_sum += 1.0
            closed_total += 1
        else:
            e_i = float(max_rem)
        if e_i <= 0.0:
            n_zero_exp_hooks += 1
        e_sum += e_i

    if e_sum <= 0.0:
        s_val: Optional[float] = None
        s_fallback = 1.0
    else:
        lam = c_sum / (e_sum + prior_tau)
        s_val = float(max(0.0, min(1.0, 1.0 - lam)) ** h_ref_sent)
        s_fallback = s_val

    return {
        "s_haz_sent": s_val,
        "s_haz_sent_fallback": float(s_fallback),
        "S_W": s_w,
        "T_W": tw,
        "total_trigs": total_trigs,
        "total_resolv_sents": total_resolv_sents,
        "opened": len(opened_hooks),
        "closed": closed_total,
        "c_sent": float(c_sum),
        "e_sent": float(e_sum),
        "n_zero_exp_hooks": n_zero_exp_hooks,
    }


def main() -> None:
    works = b20.load_works()
    w_by_wid = {w["id"].split(":")[-1]: w for w in works}
    wids = [w["id"].split(":")[-1] for w in works]
    authors = [w["author_id"] for w in works]

    scenes_s2_frozen = load_scenes_jsonl(b20.SCENES_JSONL)
    scenes_s1_a = load_scenes_jsonl(REANNOT_A_JSONL)
    scenes_s1_c = load_scenes_jsonl(REANNOT_C_JSONL)

    log_chars_py = np.array([math.log(len(w_by_wid[wid]["text"])) for wid in wids], dtype=float)
    log_chars_nonws = np.array(
        [math.log(b20.count_non_whitespace_chars(w_by_wid[wid]["text"])) for wid in wids], dtype=float
    )
    log_tw = np.array([math.log(len(scenes_s2_frozen[wid])) for wid in wids], dtype=float)
    log_sents = np.array(
        [math.log(b20.count_rust_sentences(w_by_wid[wid]["text"])) for wid in wids], dtype=float
    )

    # ------------------------------------------------------------------
    # Section 1: Zero-Exposure & Degenerate Census + Parity Check
    # ------------------------------------------------------------------
    parity_frozen = check_event_collection_parity(scenes_s2_frozen)
    census = {
        "rust_vs_py_collect_parity_frozen": parity_frozen,
        "series2_frozen_a_current": hook_exposure_census("series2_frozen_a_current", scenes_s2_frozen, wids, w_by_wid),
        "series1_reannot_a_current": hook_exposure_census("series1_reannot_a_current", scenes_s1_a, wids, w_by_wid),
        "series1_reannot_c_corpus_fixed": hook_exposure_census(
            "series1_reannot_c_corpus_fixed", scenes_s1_c, wids, w_by_wid
        ),
    }

    # ------------------------------------------------------------------
    # Section 2: Candidate (i) vs Candidate (ii) Exclusion Rules
    # ------------------------------------------------------------------
    cand_i_vs_ii: Dict[str, Any] = {}
    for ds_name, sc_map in [
        ("series2_frozen_a_current", scenes_s2_frozen),
        ("series1_reannot_a_current", scenes_s1_a),
        ("series1_reannot_c_corpus_fixed", scenes_s1_c),
    ]:
        ds_res: Dict[str, Any] = {}
        for h_val in [10, 15]:
            details = [calc_hazard_survival_detail(sc_map[wid], h_val, 5, 1.0, None) for wid in wids]
            # Candidate (i): E_H==0, N_open>0 -> 1.0; N_open==0 -> 0.0 (already returned by calc_hazard_survival_detail)
            vals_i = np.array([d["s_haz"] for d in details], dtype=float)
            e_arr = np.array([d["e_h"] for d in details], dtype=float)
            c_arr = np.array([d["c_h"] for d in details], dtype=float)
            op_arr = np.array([d["opened"] for d in details], dtype=int)
            tw_arr = np.array([len(sc_map[wid]) for wid in wids], dtype=int)

            eval_i = evaluate_vector(vals_i, log_chars_py, log_chars_nonws, log_tw, log_sents, authors)
            rules = {
                "ii_a_Nopen_gt_0_and_E_gt_0": (op_arr > 0) & (e_arr > 0.0),
                "ii_b_E_ge_5": (op_arr > 0) & (e_arr >= 5.0),
                "ii_c_E_ge_10": (op_arr > 0) & (e_arr >= 10.0),
                "ii_d_E_ge_15": (op_arr > 0) & (e_arr >= 15.0),
                "ii_e_TW_ge_5": (op_arr > 0) & (tw_arr >= 5),
                "ii_f_TW_ge_10": (op_arr > 0) & (tw_arr >= 10),
                "ii_g_C_gt_0": (op_arr > 0) & (c_arr > 0.0),
            }
            eval_ii = {
                rname: evaluate_filtered_subset(
                    vals_i, mask, log_chars_py, log_chars_nonws, log_tw, log_sents, authors, wids
                )
                for rname, mask in rules.items()
            }
            ds_res[f"H{h_val}"] = {
                "candidate_i_full_106": eval_i,
                "candidate_ii_exclusions": eval_ii,
            }
        cand_i_vs_ii[ds_name] = ds_res

    # ------------------------------------------------------------------
    # Section 3: Candidate (iii) Sentence-Index Distance Exposure Prototype
    # ------------------------------------------------------------------
    splitter = AozoraCorpusSplitter(data_dir=ROOT / "data")
    manifests = splitter.load_manifests()
    manifests_by_wid = {m["work_id"]: m for m in manifests}

    sc_texts_by_wid: Dict[str, List[str]] = {}
    corpus_sf_s1: Dict[str, int] = {}
    total_corpus_scenes = 0
    for wid in wids:
        scs = split_work_into_scenes(splitter.get_cached_text(manifests_by_wid[wid]))
        sc_texts_by_wid[wid] = scs
        total_corpus_scenes += len(scs)
        wsf = compute_work_scene_frequencies(scs)
        for tok, cnt in wsf.items():
            corpus_sf_s1[tok] = corpus_sf_s1.get(tok, 0) + cnt

    long_wids_200 = sorted(
        [wid for wid in wids if len(scenes_s2_frozen[wid]) >= 200],
        key=lambda x: len(scenes_s2_frozen[x]),
        reverse=True,
    )

    sent_configs = [
        (200, 100, 20.0),
        (250, 125, 25.0),
        (300, 150, 30.0),
    ]
    cand_iii_results: Dict[str, Any] = {}
    for isf_mode in ["a_current", "c_corpus_fixed"]:
        cand_iii_results[isf_mode] = {}
        for sub_open_sc in [False, True]:
            sub_tag = "sub_open_sc_true" if sub_open_sc else "sub_open_sc_false"
            cand_iii_results[isf_mode][sub_tag] = {}
            for h_s, href_s, tau_s in sent_configs:
                cfg_key = f"Hsent{h_s}_Href{href_s}_tau{int(tau_s)}"
                vals: List[float] = []
                n_zero_hooks_total = 0
                n_zero_works = 0
                n_ch_zero_works = 0
                n_opened_total = 0
                n_closed_total = 0
                for wid in wids:
                    res = extract_and_score_sentence_exposure(
                        sc_texts_by_wid[wid],
                        isf_mode=isf_mode,
                        sub_open_sc=sub_open_sc,
                        horizon_sent=h_s,
                        h_ref_sent=href_s,
                        prior_tau=tau_s,
                        corpus_sf=corpus_sf_s1,
                        total_corpus_scenes=total_corpus_scenes,
                    )
                    vals.append(float(res["s_haz_sent_fallback"]))
                    n_zero_hooks_total += res["n_zero_exp_hooks"]
                    n_opened_total += res["opened"]
                    n_closed_total += res["closed"]
                    if res["e_sent"] <= 0.0:
                        n_zero_works += 1
                    if res["c_sent"] == 0.0:
                        n_ch_zero_works += 1

                ev = evaluate_vector(
                    np.array(vals, dtype=float),
                    log_chars_py,
                    log_chars_nonws,
                    log_tw,
                    log_sents,
                    authors,
                )

                # 7-work truncation drift (K=20 -> K=200)
                drifts: List[float] = []
                k20_vals: List[float] = []
                k200_vals: List[float] = []
                for wid in long_wids_200:
                    r20 = extract_and_score_sentence_exposure(
                        sc_texts_by_wid[wid][:20],
                        isf_mode=isf_mode,
                        sub_open_sc=sub_open_sc,
                        horizon_sent=h_s,
                        h_ref_sent=href_s,
                        prior_tau=tau_s,
                        corpus_sf=corpus_sf_s1,
                        total_corpus_scenes=total_corpus_scenes,
                    )
                    r200 = extract_and_score_sentence_exposure(
                        sc_texts_by_wid[wid][:200],
                        isf_mode=isf_mode,
                        sub_open_sc=sub_open_sc,
                        horizon_sent=h_s,
                        h_ref_sent=href_s,
                        prior_tau=tau_s,
                        corpus_sf=corpus_sf_s1,
                        total_corpus_scenes=total_corpus_scenes,
                    )
                    v20 = float(r20["s_haz_sent_fallback"])
                    v200 = float(r200["s_haz_sent_fallback"])
                    k20_vals.append(v20)
                    k200_vals.append(v200)
                    drifts.append(v20 - v200)

                cand_iii_results[isf_mode][sub_tag][cfg_key] = {
                    "n_opened_hooks_total": n_opened_total,
                    "n_closed_within_horizon_total": n_closed_total,
                    "n_zero_exp_hooks": n_zero_hooks_total,
                    "n_zero_exp_works": n_zero_works,
                    "n_ch_zero_works": n_ch_zero_works,
                    "metrics_106": ev,
                    "truncation_7works_K20_to_K200": {
                        "mean_K20": float(np.mean(k20_vals)),
                        "mean_K200": float(np.mean(k200_vals)),
                        "drift_20_minus_200_mean": float(np.mean(drifts)),
                        "drift_20_minus_200_abs_mean": float(np.mean(np.abs(drifts))),
                    },
                }

    # ------------------------------------------------------------------
    # Section 4: P-3 D10 (4 Sources x 4 Translator Styles) Evaluation
    # ------------------------------------------------------------------
    d10_cells = build_d10_cells()
    # Map src/trans joined single-scene strings as well
    single_texts: Dict[str, str] = {}
    for src in CROSSED_SOURCES:
        for trans in TRANSLATOR_STYLES:
            cid = f"{src['source_id']}_{trans['trans_id']}"
            single_texts[cid] = "\n".join(trans["transform"](ev[0]) for ev in src["events"])

    d10_eval_modes: Dict[str, List[Dict[str, Any]]] = {
        "d10_single_scene_joined_16": [],
        "d10_full_text_auto_split_16": [],
        "d10_explicit_3scenes_16": [],
    }
    for cell in d10_cells:
        cid = cell["cell_id"]
        modes = [
            ("d10_single_scene_joined_16", [single_texts[cid]]),
            ("d10_full_text_auto_split_16", split_work_into_scenes(cell["full_text"])),
            ("d10_explicit_3scenes_16", cell["scene_texts"]),
        ]
        for mname, sc_list_txt in modes:
            # Scene-level annotation (via splitter._extract... equivalent using task2 c_corpus_fixed & a_current)
            tw = len(sc_list_txt)
            # Annotate scenes cleanly in memory
            wsf = compute_work_scene_frequencies(sc_list_txt)
            active_sc: List[Dict[str, Any]] = []
            h_seq = 1
            sc_records: List[Dict[str, Any]] = []
            for sc_i, raw_sc in enumerate(sc_list_txt):
                proc_txt = strip_commas(raw_sc)
                sents = [s.strip() for s in SENT_SPLIT.split(proc_txt) if s.strip()]
                op_sc = []
                for cs in sents:
                    if RE_HOOK_TRIG.search(cs):
                        ktoks = extract_kanji_tokens(cs[:30])
                        spec = filter_hook_tokens_by_isf(ktoks, wsf, total_scenes=tw, isf_threshold=1.2)
                        if not spec:
                            fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                            final_spec = fb if fb else {cs[:4]}
                        else:
                            final_spec = spec
                        op_sc.append({"id": h_seq, "label": cs[:30]})
                        active_sc.append({"id": h_seq, "tokens": final_spec, "open_idx": sc_i})
                        h_seq += 1
                        if len(op_sc) >= 2:
                            break
                cl_sc = []
                if RE_RESOLV_MARKER.search(proc_txt):
                    for h in list(active_sc):
                        if h["open_idx"] < sc_i and any(tok in proc_txt for tok in h["tokens"]):
                            cl_sc.append(h["id"])
                            active_sc.remove(h)
                sc_records.append(
                    {
                        "work_id": cid,
                        "scene_index": sc_i,
                        "scene_count": tw,
                        "kamae": {"opened_hooks": op_sc, "closed_hook_ids": cl_sc},
                    }
                )

            scene_haz = calc_hazard_survival_detail(sc_records, 10, 5, 1.0, None)
            sent_haz = extract_and_score_sentence_exposure(
                sc_list_txt,
                isf_mode="a_current",
                sub_open_sc=False,
                horizon_sent=250,
                h_ref_sent=125,
                prior_tau=25.0,
            )
            full_str = "\n".join(sc_list_txt)
            d10_eval_modes[mname].append(
                {
                    "cell_id": cid,
                    "source_id": cell["source_id"],
                    "trans_id": cell["trans_id"],
                    "chars": len(full_str),
                    "T_W": tw,
                    "S_W": sent_haz["S_W"],
                    "total_trigs": sent_haz["total_trigs"],
                    "total_resolv_sents": sent_haz["total_resolv_sents"],
                    "scene_opened": scene_haz["opened"],
                    "scene_closed": scene_haz["closed_total"],
                    "scene_E_H10": scene_haz["e_h"],
                    "scene_C_H10": scene_haz["c_h"],
                    "scene_S_haz_cand_i": scene_haz["s_haz"],
                    "scene_S_haz_cand_ii_a": (
                        None if (scene_haz["opened"] == 0 or scene_haz["e_h"] == 0.0) else scene_haz["s_haz"]
                    ),
                    "sent_opened": sent_haz["opened"],
                    "sent_closed": sent_haz["closed"],
                    "sent_E": sent_haz["e_sent"],
                    "sent_C": sent_haz["c_sent"],
                    "sent_S_haz_cand_ii_a": sent_haz["s_haz_sent"],
                    "sent_S_haz_cand_iii_fallback": sent_haz["s_haz_sent_fallback"],
                }
            )

    d10_summary: Dict[str, Any] = {}
    for mname, rows in d10_eval_modes.items():
        d10_summary[mname] = {
            "n_cells": len(rows),
            "chars_min": int(min(r["chars"] for r in rows)),
            "chars_max": int(max(r["chars"] for r in rows)),
            "T_W_unique": sorted({r["T_W"] for r in rows}),
            "S_W_min": int(min(r["S_W"] for r in rows)),
            "S_W_max": int(max(r["S_W"] for r in rows)),
            "total_trigs_across_16": int(sum(r["total_trigs"] for r in rows)),
            "total_resolv_sents_across_16": int(sum(r["total_resolv_sents"] for r in rows)),
            "n_cells_scene_opened_gt_0": int(sum(1 for r in rows if r["scene_opened"] > 0)),
            "n_cells_scene_E_gt_0": int(sum(1 for r in rows if r["scene_E_H10"] > 0.0)),
            "n_cells_scene_C_gt_0": int(sum(1 for r in rows if r["scene_C_H10"] > 0.0)),
            "scene_cand_i_unique_vals": sorted({round(float(r["scene_S_haz_cand_i"]), 6) for r in rows}),
            "n_cells_scene_cand_ii_a_non_null": int(sum(1 for r in rows if r["scene_S_haz_cand_ii_a"] is not None)),
            "n_cells_sent_opened_gt_0": int(sum(1 for r in rows if r["sent_opened"] > 0)),
            "n_cells_sent_E_gt_0": int(sum(1 for r in rows if r["sent_E"] > 0.0)),
            "n_cells_sent_C_gt_0": int(sum(1 for r in rows if r["sent_C"] > 0.0)),
            "sent_cand_iii_unique_vals": sorted({round(float(r["sent_S_haz_cand_iii_fallback"]), 6) for r in rows}),
            "cells": rows,
        }

    output = {
        "meta": {
            "script": "python/scripts/b22_kamae_haz/task3_degenerate_design.py",
            "frozen_scenes_sha256": hashlib.sha256(b20.SCENES_JSONL.read_bytes()).hexdigest().upper(),
            "reannot_a_sha256": hashlib.sha256(REANNOT_A_JSONL.read_bytes()).hexdigest().upper(),
            "reannot_c_sha256": hashlib.sha256(REANNOT_C_JSONL.read_bytes()).hexdigest().upper(),
            "n_works": len(wids),
        },
        "zero_exposure_census": census,
        "candidate_i_vs_ii_comparison": cand_i_vs_ii,
        "candidate_iii_sentence_exposure": cand_iii_results,
        "d10_short_text_evaluation": d10_summary,
    }

    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    # Concise stdout verification summary
    for ds in ["series2_frozen_a_current", "series1_reannot_a_current", "series1_reannot_c_corpus_fixed"]:
        c = census[ds]
        print(
            f"[Census:{ds}] tw1={c['n_works_T_W_eq_1']}, op0={c['n_works_N_open_eq_0']}, "
            f"eh10_0={c['n_works_E_H10_eq_0']}, ch10_0={c['n_works_C_H10_eq_0_at_1']}, "
            f"zero_hooks={c['zero_exposure_hooks_e_eq_0']}/{c['total_opened_hooks']} "
            f"({c['zero_exposure_hook_rate']:.4%}) in {c['n_works_with_at_least_one_zero_exp_hook']} works"
        )
    print(f"[Parity] dup_opens={parity_frozen['duplicate_opened_hook_ids']}, "
          f"dup_closes={parity_frozen['duplicate_closed_hook_ids']}, "
          f"max_diff_H10={parity_frozen['max_abs_diff_rust_vs_py_collect_H10']:.2e}")
    for isf_m in ["a_current", "c_corpus_fixed"]:
        for sub_t in ["sub_open_sc_false", "sub_open_sc_true"]:
            r250 = cand_iii_results[isf_m][sub_t]["Hsent250_Href125_tau25"]
            ev = r250["metrics_106"]
            dr = r250["truncation_7works_K20_to_K200"]
            print(
                f"[Sent:{isf_m}:{sub_t}:H250] zero_hooks={r250['n_zero_exp_hooks']}/{r250['n_opened_hooks_total']}, "
                f"zero_works={r250['n_zero_exp_works']}, ch0={r250['n_ch_zero_works']}, "
                f"rho_chars={ev['vs_log_chars_py']['spearman_rho']:.4f}, r_chars={ev['vs_log_chars_py']['pearson_r']:.4f}, "
                f"nHSIC={ev['nhsic_vs_log_chars_py']:.4f}, eta2={ev['author_discrim']['eta2_author']:.4f}, "
                f"drift20_200={dr['drift_20_minus_200_mean']:+.4f} (abs {dr['drift_20_minus_200_abs_mean']:.4f})"
            )
    for mname, s in d10_summary.items():
        print(
            f"[D10:{mname}] chars={s['chars_min']}..{s['chars_max']}, TW={s['T_W_unique']}, SW={s['S_W_min']}..{s['S_W_max']}, "
            f"trigs={s['total_trigs_across_16']}, resolvs={s['total_resolv_sents_across_16']}, "
            f"scene_op>0={s['n_cells_scene_opened_gt_0']}, scene_E>0={s['n_cells_scene_E_gt_0']}, "
            f"scene_C>0={s['n_cells_scene_C_gt_0']}, sent_C>0={s['n_cells_sent_C_gt_0']}, "
            f"scene_vals={s['scene_cand_i_unique_vals']}"
        )


if __name__ == "__main__":
    main()
