#!/usr/bin/env python
"""USIR v5 Stage B-20: Investigation of Kamae hook_survival length dependence (P-3 Task 0).

Read-only analysis script (does NOT modify production Python or Rust code).
Outputs deterministic verification results to `b20_kamae_results.json`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

from fable_v5_grand_shootout import load_works  # noqa: E402
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
from usir_py.v5.six_pole_atlas import (  # noqa: E402
    NarrativeRGNormalizer,
    compute_kamae_profile,
    normalized_hsic,
)

SCENES_JSONL = ROOT / "data" / "v5" / "aozora_106works_scenes.jsonl"
NISHIKI_PROFILE_JSON = ROOT / "docs" / "handoff" / "v10" / "nishiki_profile_106works.json"
ATLAS_JSON = ROOT / "docs" / "handoff" / "v10" / "usir_v5_six_pole_atlas.json"
F6_AUDIT_JSON = Path("C:/dev/nishiki/docs/handoff/evidence/2026-11-15_f6/f6_real_corpus_audit.json")
FOLLOWUP_JSON = ROOT / "docs" / "handoff" / "v10" / "fable_v5_calibrated_shootout.json"
OUT_JSON = Path(__file__).resolve().parent / "b20_kamae_results.json"


def count_rust_sentences(text: str) -> int:
    """Exact mirror of `fable_support::sentences` in `C:/dev/nishiki/crates/nishiki-cli/tests/fable_support/mod.rs`."""
    n_sents = 0
    n_chars = 0
    pending = False
    for c in text:
        if pending and c not in "」』）。！？!?":
            if n_chars > 0:
                n_sents += 1
            n_chars = 0
            pending = False
        if c == "\n":
            if n_chars > 0:
                n_sents += 1
            n_chars = 0
            pending = False
            continue
        if c.isspace():
            continue
        n_chars += 1
        if c in "。！？!?":
            pending = True
    if n_chars > 0:
        n_sents += 1
    return n_sents


def count_non_whitespace_chars(text: str) -> int:
    """Exact mirror of Rust `text.chars().filter(|c| !c.is_whitespace()).count()`."""
    return sum(1 for c in text if not c.isspace())


def corr_pair(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    sp = spearmanr(x, y)
    pe = pearsonr(x, y)
    return {
        "n": int(len(x)),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pe.statistic),
        "pearson_p": float(pe.pvalue),
    }


def ols_residualize(x: np.ndarray, log_chars: np.ndarray, threshold: float = 0.20) -> Dict[str, Any]:
    """Exact mirror of `NarrativeRGNormalizer.fit_orthogonalizer` + `orthogonalize`."""
    r_raw = float(spearmanr(x, log_chars).statistic)
    slope, intercept = np.polyfit(log_chars, x, 1)
    res = x - (slope * log_chars + intercept)
    r_res = float(spearmanr(res, log_chars).statistic)
    applied = abs(r_raw) >= threshold and abs(r_res) < abs(r_raw)
    out = res if applied else x.copy()
    return {
        "applied": bool(applied),
        "r_raw": r_raw,
        "r_res": r_res,
        "slope": float(slope),
        "intercept": float(intercept),
        "values": out,
    }


def compute_rust_hook_idf(scenes_by_work: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Exact mirror of `usir_metrics::kamae::HookIdf::from_corpus`."""
    n_works = 0
    df = [0, 0, 0, 0]
    for wid in sorted(scenes_by_work.keys()):
        sc_list = sorted(scenes_by_work[wid], key=lambda s: s["scene_index"])
        n = len(sc_list)
        if n == 0:
            continue
        valid = True
        for i, s in enumerate(sc_list):
            if s["scene_index"] != i or s["scene_count"] != n or s["work_id"] != sc_list[0]["work_id"]:
                valid = False
                break
        if not valid:
            continue
        n_works += 1
        seen = [False, False, False, False]
        hooks: Dict[int, int] = {}
        for s in sc_list:
            b = min(3, (s["scene_index"] * 4) // n)
            for h in s["kamae"]["opened_hooks"]:
                hid = h["id"]
                if hid not in hooks:
                    hooks[hid] = b
                    seen[b] = True
        for b in range(4):
            if seen[b]:
                df[b] += 1
    weights = [math.log((n_works + 1) / (d + 1)) + 1.0 for d in df]
    return {"n_works": n_works, "doc_freq": df, "weight": weights}


def extract_hook_events(sc_list: List[Dict[str, Any]]) -> Tuple[int, Dict[int, int], Dict[int, int]]:
    """Return (T_W, open_scene_by_hid, close_scene_by_hid) using Rust/Python collection rules."""
    n = len(sc_list)
    open_scene: Dict[int, int] = {}
    close_scene: Dict[int, int] = {}
    for s in sc_list:
        idx = s["scene_index"]
        for h in s["kamae"]["opened_hooks"]:
            hid = h["id"]
            if hid not in open_scene:
                open_scene[hid] = idx
        for cid in s["kamae"]["closed_hook_ids"]:
            if cid in open_scene and cid not in close_scene:
                close_scene[cid] = idx
    return n, open_scene, close_scene


def calc_raw_survival(sc_list: List[Dict[str, Any]]) -> float:
    _, op, cl = extract_hook_events(sc_list)
    if not op:
        return 0.0
    unres = sum(1 for hid in op if hid not in cl)
    return float(unres / (len(op) + 1e-12))


def calc_fixed_horizon(sc_list: List[Dict[str, Any]], horizon: Optional[int], uncensored_only: bool = False) -> Optional[float]:
    n, op, cl = extract_hook_events(sc_list)
    open_cnt = 0
    unres_cnt = 0
    for hid, o_idx in op.items():
        if uncensored_only and horizon is not None and o_idx + horizon >= n:
            continue
        open_cnt += 1
        c_idx = cl.get(hid)
        if c_idx is None:
            unres_cnt += 1
        elif horizon is not None and (c_idx - o_idx) > horizon:
            unres_cnt += 1
    if open_cnt == 0:
        return None if uncensored_only else 0.0
    return float(unres_cnt / (open_cnt + 1e-12))


def calc_hazard_survival(
    sc_list: List[Dict[str, Any]],
    horizon: Optional[int],
    h_ref: int = 5,
    prior_tau: float = 1.0,
) -> float:
    """Right-censored exposure-normalized discrete hazard survival: (1 - C / (E + tau_0))^h_ref."""
    n, op, cl = extract_hook_events(sc_list)
    if not op:
        return 0.0
    exposure = 0.0
    closed = 0.0
    for hid, o_idx in op.items():
        c_idx = cl.get(hid)
        max_rem = max(0, n - 1 - o_idx)
        if horizon is not None:
            max_rem = min(max_rem, horizon)
        if c_idx is not None and (horizon is None or (c_idx - o_idx) <= horizon):
            exposure += float(c_idx - o_idx)
            closed += 1.0
        else:
            exposure += float(max_rem)
    lam = closed / (exposure + prior_tau) if (exposure + prior_tau) > 0 else 0.0
    s_step = max(0.0, min(1.0, 1.0 - lam))
    return float(s_step ** h_ref)


def calc_scene_block_median(sc_list: List[Dict[str, Any]], block_len: int) -> float:
    """Tier-2 style local scene-window bootstrap median for hook_survival."""
    n = len(sc_list)
    if n == 0:
        return 0.0
    if n <= block_len:
        blocks = [sc_list]
    else:
        stride = max(1, block_len // 2)
        blocks = [sc_list[i : i + block_len] for i in range(0, n - block_len + 1, stride)]
    vals = []
    for blk in blocks:
        op_ids = {h["id"] for s in blk for h in s["kamae"]["opened_hooks"]}
        if not op_ids:
            continue
        cl_ids = {cid for s in blk for cid in s["kamae"]["closed_hook_ids"] if cid in op_ids}
        vals.append((len(op_ids) - len(cl_ids)) / len(op_ids))
    return float(np.median(vals)) if vals else 0.0


def calc_rel_horizon(sc_list: List[Dict[str, Any]], tau: float) -> float:
    n, op, cl = extract_hook_events(sc_list)
    if not op:
        return 0.0
    unres = 0
    for hid, o_idx in op.items():
        c_idx = cl.get(hid)
        if c_idx is None or ((c_idx - o_idx) / max(n, 1)) > tau:
            unres += 1
    return float(unres / (len(op) + 1e-12))


def calc_rel_hazard_rate(sc_list: List[Dict[str, Any]], prior_u: float = 0.1) -> float:
    """Relative-time decay rate k_rel = C / (sum delta_u + prior_u)."""
    n, op, cl = extract_hook_events(sc_list)
    if not op:
        return 0.0
    exp_u = 0.0
    closed = 0.0
    for hid, o_idx in op.items():
        c_idx = cl.get(hid)
        if c_idx is not None:
            exp_u += (c_idx - o_idx) / max(n, 1)
            closed += 1.0
        else:
            exp_u += max(0, n - 1 - o_idx) / max(n, 1)
    return float(closed / (exp_u + prior_u))


def calc_act_survivals(sc_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """5-act breakdown (by opening act m = min(4, 5*o_idx // n)) and Act5 - Act1 gradient."""
    n, op, cl = extract_hook_events(sc_list)
    act_open = [0] * 5
    act_unres = [0] * 5
    for hid, o_idx in op.items():
        m = min(4, (o_idx * 5) // max(n, 1))
        act_open[m] += 1
        if hid not in cl:
            act_unres[m] += 1
    rates = [act_unres[m] / act_open[m] if act_open[m] > 0 else None for m in range(5)]
    valid_rates = [r for r in rates if r is not None]
    if rates[0] is not None and rates[4] is not None:
        grad = rates[4] - rates[0]
    elif len(valid_rates) >= 2:
        grad = valid_rates[-1] - valid_rates[0]
    else:
        grad = 0.0
    return {"act_rates": rates, "act5_minus_act1": float(grad)}


def slice_jsonl_prefix(sc_list: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Slice first k scenes and retain only hooks opened and closed within [0, k)."""
    sub = sc_list[:k]
    out = []
    for s in sub:
        s_copy = {
            "scene_index": s["scene_index"],
            "scene_count": len(sub),
            "kamae": {
                "opened_hooks": s["kamae"]["opened_hooks"],
                "closed_hook_ids": s["kamae"]["closed_hook_ids"],
            },
        }
        out.append(s_copy)
    return out


def nhsic_vs_log_chars(feature_vec: np.ndarray, log_chars: np.ndarray) -> float:
    x = ((feature_vec - np.mean(feature_vec)) / (np.std(feature_vec) + 1e-12)).reshape(-1, 1)
    y = ((log_chars - np.mean(log_chars)) / (np.std(log_chars) + 1e-12)).reshape(-1, 1)
    return float(normalized_hsic(x, y))


def author_discrimination(vals: np.ndarray, authors: List[str]) -> Dict[str, Any]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for v, a in zip(vals, authors):
        groups[a].append(float(v))
    arrs = [np.array(groups[a]) for a in sorted(groups.keys())]
    f_stat, f_p = f_oneway(*arrs)
    kw_stat, kw_p = kruskal(*arrs)
    grand_mean = float(np.mean(vals))
    ss_tot = float(np.sum((vals - grand_mean) ** 2))
    ss_bet = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in arrs))
    eta2 = ss_bet / ss_tot if ss_tot > 0 else 0.0
    per_author_mean = {a: round(float(np.mean(groups[a])), 4) for a in sorted(groups.keys())}
    return {
        "eta2_author": float(eta2),
        "anova_f": float(f_stat),
        "anova_p": float(f_p),
        "kruskal_h": float(kw_stat),
        "kruskal_p": float(kw_p),
        "per_author_mean": per_author_mean,
    }


def analyze_isf_boundary_mechanics(splitter: AozoraCorpusSplitter, manifests: List[Dict[str, Any]], scenes_by_work: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Quantify within-work ISF threshold boundary effects in aozora_scene_splitter.py."""
    by_tw_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_hooks = 0
    total_fallback = 0
    total_sf1_only_passed = 0

    for m in manifests:
        wid = m["work_id"]
        txt = splitter.get_cached_text(m)
        sc_texts = split_work_into_scenes(txt)
        tw = len(sc_texts)
        freqs = compute_work_scene_frequencies(sc_texts)

        # Max sf passable under ISF >= 1.2
        if tw < 4:
            max_sf_pass = 1
        else:
            # ln(tw / (sf + 1)) >= 1.2 => sf <= tw / exp(1.2) - 1
            max_sf_pass = math.floor(tw / math.exp(1.2) - 1.0 + 1e-12)

        hooks_w = 0
        fb_w = 0
        sf1_only_w = 0
        for sc_txt in sc_texts:
            clean_sc = strip_commas(sc_txt)
            sents = [s for s in SENT_SPLIT.split(clean_sc) if s.strip()]
            opened_in_sc = 0
            for sent in sents:
                cs = sent.strip()
                if RE_HOOK_TRIG.search(cs):
                    label = cs[:30]
                    ktoks = extract_kanji_tokens(label)
                    spec = filter_hook_tokens_by_isf(ktoks, freqs, total_scenes=tw, isf_threshold=1.2)
                    hooks_w += 1
                    if not spec:
                        fb_w += 1
                        fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                        spec = fb if fb else {cs[:4]}
                    else:
                        if all(freqs.get(t, 0) <= 1 for t in spec):
                            sf1_only_w += 1
                    opened_in_sc += 1
                    if opened_in_sc >= 2:
                        break

        total_hooks += hooks_w
        total_fallback += fb_w
        total_sf1_only_passed += sf1_only_w

        raw_surv = calc_raw_survival(scenes_by_work[wid])
        if tw < 10:
            bkt = "T_W < 10 (max_sf <= 1)"
        elif tw < 20:
            bkt = "10 <= T_W < 20"
        elif tw < 50:
            bkt = "20 <= T_W < 50"
        elif tw < 100:
            bkt = "50 <= T_W < 100"
        else:
            bkt = "T_W >= 100"

        by_tw_bucket[bkt].append({
            "work_id": wid,
            "T_W": tw,
            "max_sf_pass": max_sf_pass,
            "hooks": hooks_w,
            "fallback": fb_w,
            "sf1_only_passed": sf1_only_w,
            "raw_survival": raw_surv,
        })

    bucket_summary = {}
    for bkt in ["T_W < 10 (max_sf <= 1)", "10 <= T_W < 20", "20 <= T_W < 50", "50 <= T_W < 100", "T_W >= 100"]:
        rows = by_tw_bucket[bkt]
        h_sum = sum(r["hooks"] for r in rows)
        fb_sum = sum(r["fallback"] for r in rows)
        sf1_sum = sum(r["sf1_only_passed"] for r in rows)
        bucket_summary[bkt] = {
            "n_works": len(rows),
            "mean_T_W": float(np.mean([r["T_W"] for r in rows])),
            "total_hooks": h_sum,
            "fallback_rate": float(fb_sum / h_sum) if h_sum > 0 else 0.0,
            "sf1_only_pass_rate": float(sf1_sum / h_sum) if h_sum > 0 else 0.0,
            "mean_raw_survival": float(np.mean([r["raw_survival"] for r in rows])),
        }

    # Counterfactual: what if ISF filtering were disabled (keep all non-stopword kanji tokens, or global threshold)?
    cf_no_isf_surv = []
    cf_corpus_isf_surv = []
    # Build corpus-level scene frequencies across all 4576 scenes
    corpus_sf: Dict[str, int] = {}
    all_scene_texts_by_wid: Dict[str, List[str]] = {}
    total_corpus_scenes = 0
    for m in manifests:
        wid = m["work_id"]
        sc_texts = split_work_into_scenes(splitter.get_cached_text(m))
        all_scene_texts_by_wid[wid] = sc_texts
        total_corpus_scenes += len(sc_texts)
        for sc in sc_texts:
            for tok in extract_kanji_tokens(strip_commas(sc)):
                corpus_sf[tok] = corpus_sf.get(tok, 0) + 1

    for mode_name, target_list in [("no_isf", cf_no_isf_surv), ("corpus_isf_1.2", cf_corpus_isf_surv)]:
        for m in manifests:
            wid = m["work_id"]
            sc_texts = all_scene_texts_by_wid[wid]
            active_hooks: List[Dict[str, Any]] = []
            h_seq = 1
            opened = 0
            closed = 0
            for idx, sc_txt in enumerate(sc_texts):
                clean_sc = strip_commas(sc_txt)
                sents = [s for s in SENT_SPLIT.split(clean_sc) if s.strip()]
                op_here = 0
                for sent in sents:
                    cs = sent.strip()
                    if RE_HOOK_TRIG.search(cs):
                        label = cs[:30]
                        ktoks = extract_kanji_tokens(label)
                        if mode_name == "no_isf":
                            spec = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                        else:
                            spec = filter_hook_tokens_by_isf(ktoks, corpus_sf, total_scenes=total_corpus_scenes, isf_threshold=1.2)
                        if not spec:
                            fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                            spec = fb if fb else {cs[:4]}
                        active_hooks.append({"id": h_seq, "tokens": spec, "open_idx": idx})
                        h_seq += 1
                        opened += 1
                        op_here += 1
                        if op_here >= 2:
                            break
                has_resolv = bool(RE_RESOLV_MARKER.search(clean_sc))
                for h in list(active_hooks):
                    if h["open_idx"] < idx and has_resolv:
                        if any(tok in clean_sc for tok in h["tokens"]):
                            closed += 1
                            active_hooks.remove(h)
            target_list.append((opened - closed) / (opened + 1e-12) if opened > 0 else 0.0)

    return {
        "isf_threshold_theory": {
            "formula": "ISF(w) = ln(T_W / (sf(w) + 1)) >= 1.2",
            "min_T_W_for_sf2": float(3.0 * math.exp(1.2)),
            "min_T_W_for_sf2_int": 10,
            "total_hooks": total_hooks,
            "total_fallback": total_fallback,
            "total_sf1_only_passed": total_sf1_only_passed,
        },
        "bucket_summary": bucket_summary,
        "counterfactual_no_isf_surv": cf_no_isf_surv,
        "counterfactual_corpus_isf_surv": cf_corpus_isf_surv,
    }


def main() -> None:
    works = load_works()
    w_by_wid = {w["id"].split(":")[-1]: w for w in works}
    all_wids_py_order = [w["id"].split(":")[-1] for w in works]

    prof_rust = json.loads(NISHIKI_PROFILE_JSON.read_text(encoding="utf-8"))
    atlas_py = json.loads(ATLAS_JSON.read_text(encoding="utf-8"))
    f6_audit = json.loads(F6_AUDIT_JSON.read_text(encoding="utf-8")) if F6_AUDIT_JSON.exists() else {}
    followup = json.loads(FOLLOWUP_JSON.read_text(encoding="utf-8"))
    labels = followup["labels"]

    scenes_by_work: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with SCENES_JSONL.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            scenes_by_work[s["work_id"]].append(s)
    for wid in scenes_by_work:
        scenes_by_work[wid].sort(key=lambda s: s["scene_index"])

    # ---------------------------------------------------------------------------------------------
    # Task 1: Parity & definition audit (Rust profile vs Python compute_kamae_profile)
    # ---------------------------------------------------------------------------------------------
    rust_idf_recomputed = compute_rust_hook_idf(scenes_by_work)
    max_idf_weight_diff = max(
        abs(a - b) for a, b in zip(rust_idf_recomputed["weight"], prof_rust["hook_idf"]["weight"])
    )

    rust_works_by_wid = {rw["work_id"]: rw for rw in prof_rust["works"]}
    diffs_hs = []
    diffs_hsidf_with_idf = []
    diffs_hsidf_no_idf = []
    diffs_opened = []
    for wid in all_wids_py_order:
        scs = scenes_by_work[wid]
        rw = rust_works_by_wid[wid]
        py_with_idf = compute_kamae_profile(scs, hook_idf=prof_rust["hook_idf"])
        py_no_idf = compute_kamae_profile(scs, hook_idf=None)
        diffs_hs.append(abs(rw["kamae"]["hook_survival"] - py_with_idf["hook_survival"]))
        diffs_hsidf_with_idf.append(abs(rw["kamae"]["hook_survival_idf"] - py_with_idf["hook_survival_idf"]))
        diffs_hsidf_no_idf.append(abs(rw["kamae"]["hook_survival_idf"] - py_no_idf["hook_survival_idf"]))
        diffs_opened.append(abs(rw["kamae"]["opened_hooks"] - py_with_idf["opened_hooks"]))

    task1_parity = {
        "n_works": len(all_wids_py_order),
        "hook_idf_match": {
            "rust_n_works": prof_rust["hook_idf"]["n_works"],
            "rust_doc_freq": prof_rust["hook_idf"]["doc_freq"],
            "recomputed_doc_freq": rust_idf_recomputed["doc_freq"],
            "rust_weight": prof_rust["hook_idf"]["weight"],
            "recomputed_weight": rust_idf_recomputed["weight"],
            "max_weight_abs_diff": float(max_idf_weight_diff),
        },
        "max_abs_diff_hook_survival": float(max(diffs_hs)),
        "max_abs_diff_hook_survival_idf_with_corpus_idf": float(max(diffs_hsidf_with_idf)),
        "max_abs_diff_hook_survival_idf_without_idf_arg": float(max(diffs_hsidf_no_idf)),
        "max_abs_diff_opened_hooks": int(max(diffs_opened)),
        "note": "experiments/usir_v5_six_pole_atlas.py loads nishiki_profile_106works.json (Rust profile output) directly.",
    }

    # ---------------------------------------------------------------------------------------------
    # Task 2: Controlled recalculation & factor decomposition
    # ---------------------------------------------------------------------------------------------
    # Identify F6 excluded works (< 64 sentences)
    sent_counts = {wid: count_rust_sentences(w_by_wid[wid]["text"]) for wid in all_wids_py_order}
    excluded_f6 = [
        {
            "work_id": wid,
            "author_id": w_by_wid[wid]["author_id"],
            "author_ja": labels.get(w_by_wid[wid]["author_id"], ""),
            "title": w_by_wid[wid].get("title", ""),
            "rust_sentences": sent_counts[wid],
            "scene_count": len(scenes_by_work[wid]),
            "char_len_py": len(w_by_wid[wid]["text"]),
            "char_len_nonws": count_non_whitespace_chars(w_by_wid[wid]["text"]),
            "hook_survival": rust_works_by_wid[wid]["kamae"]["hook_survival"],
            "hook_survival_idf": rust_works_by_wid[wid]["kamae"]["hook_survival_idf"],
        }
        for wid in all_wids_py_order
        if sent_counts[wid] < 64
    ]
    wids_105 = [rw["work_id"] for rw in prof_rust["works"] if sent_counts[rw["work_id"]] >= 64]
    wids_106_rust_order = [rw["work_id"] for rw in prof_rust["works"]]

    def build_vectors(wids: List[str]) -> Dict[str, np.ndarray]:
        return {
            "hs": np.array([rust_works_by_wid[wid]["kamae"]["hook_survival"] for wid in wids], dtype=float),
            "hd": np.array(
                [rust_works_by_wid[wid]["kamae"]["opened_hooks"] / max(rust_works_by_wid[wid]["scene_count"], 1) for wid in wids],
                dtype=float,
            ),
            "hsidf": np.array([rust_works_by_wid[wid]["kamae"]["hook_survival_idf"] for wid in wids], dtype=float),
            "log_c_nonws": np.array([math.log(count_non_whitespace_chars(w_by_wid[wid]["text"])) for wid in wids], dtype=float),
            "log_c_py": np.array([math.log(len(w_by_wid[wid]["text"])) for wid in wids], dtype=float),
            "log_c_scene_sum": np.array([math.log(rust_works_by_wid[wid]["char_len"]) for wid in wids], dtype=float),
            "log_tw": np.array([math.log(rust_works_by_wid[wid]["scene_count"]) for wid in wids], dtype=float),
        }

    v105 = build_vectors(wids_105)
    v106 = build_vectors(all_wids_py_order)

    def eval_condition(vecs: Dict[str, np.ndarray], len_key: str, residualize: bool) -> Dict[str, Any]:
        lc = vecs[len_key]
        out: Dict[str, Any] = {}
        for feat_key, feat_name in [("hs", "kamae_hook_survival"), ("hd", "kamae_hook_density"), ("hsidf", "kamae_survival_idf")]:
            x = vecs[feat_key]
            if residualize:
                ols = ols_residualize(x, lc, threshold=0.20)
                c = corr_pair(ols["values"], lc)
                c["ols_applied"] = ols["applied"]
                c["ols_slope"] = ols["slope"]
                c["ols_intercept"] = ols["intercept"]
            else:
                c = corr_pair(x, lc)
                c["ols_applied"] = False
            out[feat_name] = c
        return out

    conditions = {
        "a_rust_f6_exact_n105_nonws_raw": eval_condition(v105, "log_c_nonws", False),
        "b1_n105_nonws_ols_resid": eval_condition(v105, "log_c_nonws", True),
        "b2_python_b4_exact_n106_pylen_ols_resid": eval_condition(v106, "log_c_py", True),
        "c1_n106_pylen_raw": eval_condition(v106, "log_c_py", False),
        "c2_n106_nonws_raw": eval_condition(v106, "log_c_nonws", False),
        "c3_n105_pylen_raw": eval_condition(v105, "log_c_py", False),
        "c4_n106_scene_char_sum_raw": eval_condition(v106, "log_c_scene_sum", False),
        "c5_n106_log_scene_count_raw": eval_condition(v106, "log_tw", False),
        "c6_n106_log_scene_count_ols_resid": eval_condition(v106, "log_tw", True),
    }

    # Verify exact match with reported F6 and Python B-4 numbers
    f6_reported = f6_audit.get("tier2_spearman_vs_log_chars", {}).get("deep_kamae_sadame_nagare_awase", [[]])[0]
    py_reported = atlas_py.get("rg_length_audit", {}).get("per_feature_rho", {})

    rho_a = conditions["a_rust_f6_exact_n105_nonws_raw"]["kamae_hook_survival"]["spearman_rho"]
    rho_c2 = conditions["c2_n106_nonws_raw"]["kamae_hook_survival"]["spearman_rho"]
    rho_c1 = conditions["c1_n106_pylen_raw"]["kamae_hook_survival"]["spearman_rho"]
    rho_b2 = conditions["b2_python_b4_exact_n106_pylen_ols_resid"]["kamae_hook_survival"]["spearman_rho"]

    decomposition = {
        "total_gap_b2_minus_a": float(rho_b2 - rho_a),
        "step1_sample_n105_to_n106": float(rho_c2 - rho_a),
        "step2_char_def_nonws_to_pylen": float(rho_c1 - rho_c2),
        "step3_ols_residualization": float(rho_b2 - rho_c1),
        "f6_reported": f6_reported,
        "python_b4_reported": {
            "kamae_hook_survival": py_reported.get("kamae_hook_survival"),
            "kamae_hook_density": py_reported.get("kamae_hook_density"),
            "kamae_survival_idf": py_reported.get("kamae_survival_idf"),
        },
    }

    # ---------------------------------------------------------------------------------------------
    # Task 3: Horizon Bias Verification
    # ---------------------------------------------------------------------------------------------
    # 3.1 Corpus hook lag & T_W distribution
    all_lags: List[int] = []
    all_rel_lags: List[float] = []
    total_opened = 0
    total_closed = 0
    tw_arr = np.array([len(scenes_by_work[wid]) for wid in all_wids_py_order], dtype=int)
    for wid in all_wids_py_order:
        n, op, cl = extract_hook_events(scenes_by_work[wid])
        total_opened += len(op)
        total_closed += len(cl)
        for hid, c_idx in cl.items():
            lag = c_idx - op[hid]
            all_lags.append(lag)
            all_rel_lags.append(lag / max(n, 1))
    lags_np = np.array(all_lags, dtype=float)
    rel_lags_np = np.array(all_rel_lags, dtype=float)

    corpus_hook_stats = {
        "n_works": len(all_wids_py_order),
        "total_scenes": int(np.sum(tw_arr)),
        "scene_count_quantiles": {
            q: float(np.percentile(tw_arr, p)) for q, p in [("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100)]
        },
        "total_opened_hooks": int(total_opened),
        "total_closed_hooks": int(total_closed),
        "global_closure_rate": float(total_closed / total_opened),
        "closure_lag_scenes_quantiles": {
            q: float(np.percentile(lags_np, p)) for q, p in [("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100)]
        },
        "closure_lag_rel_time_quantiles": {
            q: float(np.percentile(rel_lags_np, p)) for q, p in [("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100)]
        },
    }

    # 3.2 ISF boundary mechanics
    splitter = AozoraCorpusSplitter(data_dir=ROOT / "data")
    manifests = splitter.load_manifests()
    isf_mechanics = analyze_isf_boundary_mechanics(splitter, manifests, scenes_by_work)
    cf_no_isf_arr = np.array(isf_mechanics.pop("counterfactual_no_isf_surv"), dtype=float)
    cf_corp_isf_arr = np.array(isf_mechanics.pop("counterfactual_corpus_isf_surv"), dtype=float)
    # align manifests order with log_c_py (manifests is loaded in exact same order as load_works())
    assert [m["work_id"] for m in manifests] == all_wids_py_order
    isf_mechanics["counterfactual_correlations_vs_log_chars"] = {
        "original_within_work_isf_1.2": corr_pair(v106["hs"], v106["log_c_py"]),
        "no_isf_filter": {
            "mean": float(cf_no_isf_arr.mean()),
            **corr_pair(cf_no_isf_arr, v106["log_c_py"]),
        },
        "corpus_wide_isf_1.2": {
            "mean": float(cf_corp_isf_arr.mean()),
            **corr_pair(cf_corp_isf_arr, v106["log_c_py"]),
        },
    }

    # 3.3 Fixed scene horizons (H) across all 4 definitions
    horizons: List[Optional[int]] = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, None]
    fixed_horizon_results = []
    for H in horizons:
        v_raw_h = np.array([calc_fixed_horizon(scenes_by_work[wid], H, False) for wid in all_wids_py_order], dtype=float)
        v_unc_list = [calc_fixed_horizon(scenes_by_work[wid], H, True) for wid in all_wids_py_order]
        valid_mask = [i for i, v in enumerate(v_unc_list) if v is not None]
        v_unc_arr = np.array([v_unc_list[i] for i in valid_mask], dtype=float)
        lc_unc = v106["log_c_py"][valid_mask]

        h_ref = H if H is not None else 5
        v_haz_href = np.array([calc_hazard_survival(scenes_by_work[wid], H, h_ref=h_ref, prior_tau=1.0) for wid in all_wids_py_order], dtype=float)
        v_haz_h5 = np.array([calc_hazard_survival(scenes_by_work[wid], H, h_ref=5, prior_tau=1.0) for wid in all_wids_py_order], dtype=float)

        fixed_horizon_results.append({
            "H": H if H is not None else "inf",
            "simple_fixed_H": {
                "mean": float(v_raw_h.mean()),
                "std": float(v_raw_h.std()),
                "vs_log_chars": corr_pair(v_raw_h, v106["log_c_py"]),
                "vs_log_tw": corr_pair(v_raw_h, v106["log_tw"]),
            },
            "uncensored_only_H": {
                "n_valid": len(valid_mask),
                "mean": float(v_unc_arr.mean()),
                "std": float(v_unc_arr.std()),
                "vs_log_chars": corr_pair(v_unc_arr, lc_unc),
            },
            "hazard_norm_href_H": {
                "h_ref": h_ref,
                "mean": float(v_haz_href.mean()),
                "std": float(v_haz_href.std()),
                "vs_log_chars": corr_pair(v_haz_href, v106["log_c_py"]),
                "vs_log_tw": corr_pair(v_haz_href, v106["log_tw"]),
            },
            "hazard_norm_h5": {
                "h_ref": 5,
                "mean": float(v_haz_h5.mean()),
                "std": float(v_haz_h5.std()),
                "vs_log_chars": corr_pair(v_haz_h5, v106["log_c_py"]),
                "vs_log_tw": corr_pair(v_haz_h5, v106["log_tw"]),
            },
        })

    # 3.4 Scene-block bootstrap median
    block_median_results = []
    for W in [4, 6, 8, 10, 12, 16]:
        v_blk = np.array([calc_scene_block_median(scenes_by_work[wid], W) for wid in all_wids_py_order], dtype=float)
        block_median_results.append({
            "W": W,
            "stride": max(1, W // 2),
            "mean": float(v_blk.mean()),
            "std": float(v_blk.std()),
            "vs_log_chars": corr_pair(v_blk, v106["log_c_py"]),
            "vs_log_tw": corr_pair(v_blk, v106["log_tw"]),
        })

    # 3.5 Five-act relative time normalization
    rel_horizon_results = []
    for tau in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        v_rel = np.array([calc_rel_horizon(scenes_by_work[wid], tau) for wid in all_wids_py_order], dtype=float)
        rel_horizon_results.append({
            "tau": tau,
            "mean": float(v_rel.mean()),
            "std": float(v_rel.std()),
            "vs_log_chars": corr_pair(v_rel, v106["log_c_py"]),
            "vs_log_tw": corr_pair(v_rel, v106["log_tw"]),
        })

    v_rel_k = np.array([calc_rel_hazard_rate(scenes_by_work[wid], prior_u=0.1) for wid in all_wids_py_order], dtype=float)
    act_grad_vals = np.array([calc_act_survivals(scenes_by_work[wid])["act5_minus_act1"] for wid in all_wids_py_order], dtype=float)
    act_rates_by_m = {m: [] for m in range(5)}
    act_lcs_by_m = {m: [] for m in range(5)}
    for idx_w, wid in enumerate(all_wids_py_order):
        res_act = calc_act_survivals(scenes_by_work[wid])
        for m in range(5):
            if res_act["act_rates"][m] is not None:
                act_rates_by_m[m].append(res_act["act_rates"][m])
                act_lcs_by_m[m].append(float(v106["log_c_py"][idx_w]))

    rel_time_other = {
        "rel_time_decay_rate_k_rel": {
            "mean": float(v_rel_k.mean()),
            "std": float(v_rel_k.std()),
            "vs_log_chars": corr_pair(v_rel_k, v106["log_c_py"]),
            "vs_log_tw": corr_pair(v_rel_k, v106["log_tw"]),
        },
        "act5_minus_act1_survival_grad": {
            "mean": float(act_grad_vals.mean()),
            "std": float(act_grad_vals.std()),
            "vs_log_chars": corr_pair(act_grad_vals, v106["log_c_py"]),
        },
        "per_act_survival_vs_log_chars": {
            f"act_{m+1}": {
                "n_valid": len(act_rates_by_m[m]),
                "mean": float(np.mean(act_rates_by_m[m])),
                **corr_pair(np.array(act_rates_by_m[m]), np.array(act_lcs_by_m[m])),
            }
            for m in range(5)
        },
    }

    # 3.6 Prefix truncation experiment on long works (T_W >= 100 and T_W >= 200)
    manifests_by_wid = {m["work_id"]: m for m in manifests}
    long_wids_100 = sorted([wid for wid in all_wids_py_order if len(scenes_by_work[wid]) >= 100], key=lambda w: len(scenes_by_work[w]), reverse=True)
    long_wids_200 = [wid for wid in long_wids_100 if len(scenes_by_work[wid]) >= 200]
    ks = [5, 10, 20, 40, 80, 100, 150, 200]

    trunc_summary = []
    per_work_200_traces: Dict[str, Any] = {}
    for wid in long_wids_200:
        per_work_200_traces[wid] = {
            "title": w_by_wid[wid].get("title", ""),
            "author_ja": labels.get(w_by_wid[wid]["author_id"], ""),
            "full_T_W": len(scenes_by_work[wid]),
            "full_raw_survival": calc_raw_survival(scenes_by_work[wid]),
            "steps": [],
        }

    # Pre-split long works once for speed
    long_sc_texts = {wid: split_work_into_scenes(splitter.get_cached_text(manifests_by_wid[wid])) for wid in long_wids_100}

    for k in ks:
        wids_for_k = [wid for wid in long_wids_100 if len(scenes_by_work[wid]) >= k]
        vals_slice_raw = []
        vals_reannot_raw = []
        vals_slice_haz10 = []
        vals_reannot_haz10 = []
        for wid in wids_for_k:
            sub_slice = slice_jsonl_prefix(scenes_by_work[wid], k)
            s_raw = calc_raw_survival(sub_slice)
            s_haz = calc_hazard_survival(sub_slice, horizon=10, h_ref=5, prior_tau=1.0)

            recs = splitter.process_work_scenes(manifests_by_wid[wid], long_sc_texts[wid][:k])
            r_raw = calc_raw_survival(recs)
            r_haz = calc_hazard_survival(recs, horizon=10, h_ref=5, prior_tau=1.0)

            vals_slice_raw.append(s_raw)
            vals_reannot_raw.append(r_raw)
            vals_slice_haz10.append(s_haz)
            vals_reannot_haz10.append(r_haz)

            if wid in per_work_200_traces:
                per_work_200_traces[wid]["steps"].append({
                    "K": k,
                    "slice_raw": round(s_raw, 4),
                    "reannot_raw": round(r_raw, 4),
                    "slice_hazard_H10_h5": round(s_haz, 4),
                    "reannot_hazard_H10_h5": round(r_haz, 4),
                })

        trunc_summary.append({
            "K": k,
            "n_works": len(wids_for_k),
            "slice_raw_mean": float(np.mean(vals_slice_raw)),
            "slice_raw_median": float(np.median(vals_slice_raw)),
            "reannot_raw_mean": float(np.mean(vals_reannot_raw)),
            "reannot_raw_median": float(np.median(vals_reannot_raw)),
            "slice_hazard_H10_h5_mean": float(np.mean(vals_slice_haz10)),
            "reannot_hazard_H10_h5_mean": float(np.mean(vals_reannot_haz10)),
        })

    # Also compute exact 20 vs 200 (and 20 vs full) drift on the 7 >=200 works
    drift_7works = {
        "slice_raw_20_minus_200_mean": float(np.mean([t["steps"][2]["slice_raw"] - t["steps"][-1]["slice_raw"] for t in per_work_200_traces.values()])),
        "reannot_raw_20_minus_200_mean": float(np.mean([t["steps"][2]["reannot_raw"] - t["steps"][-1]["reannot_raw"] for t in per_work_200_traces.values()])),
        "slice_hazard_H10_h5_20_minus_200_mean": float(np.mean([t["steps"][2]["slice_hazard_H10_h5"] - t["steps"][-1]["slice_hazard_H10_h5"] for t in per_work_200_traces.values()])),
        "reannot_hazard_H10_h5_20_minus_200_mean": float(np.mean([t["steps"][2]["reannot_hazard_H10_h5"] - t["steps"][-1]["reannot_hazard_H10_h5"] for t in per_work_200_traces.values()])),
    }

    # ---------------------------------------------------------------------------------------------
    # Task 4: P-3 Candidate Evaluation Matrix (correlation, nHSIC, author separation, sensitivity)
    # ---------------------------------------------------------------------------------------------
    authors_106 = [w["author_id"] for w in works]
    ols_raw = ols_residualize(v106["hs"], v106["log_c_py"], threshold=0.20)["values"]
    cand_defs: List[Tuple[str, np.ndarray]] = [
        ("baseline_raw_hook_survival", v106["hs"]),
        ("baseline_python_ols_residualized", ols_raw),
        ("simple_fixed_H5", np.array([calc_fixed_horizon(scenes_by_work[wid], 5, False) for wid in all_wids_py_order])),
        ("rel_time_tau_0.2", np.array([calc_rel_horizon(scenes_by_work[wid], 0.2) for wid in all_wids_py_order])),
        ("opt_A_hazard_H10_h5_tau1", np.array([calc_hazard_survival(scenes_by_work[wid], 10, h_ref=5, prior_tau=1.0) for wid in all_wids_py_order])),
        ("opt_A_hazard_H15_h5_tau1", np.array([calc_hazard_survival(scenes_by_work[wid], 15, h_ref=5, prior_tau=1.0) for wid in all_wids_py_order])),
        ("opt_A_hazard_H10_h5_tau2", np.array([calc_hazard_survival(scenes_by_work[wid], 10, h_ref=5, prior_tau=2.0) for wid in all_wids_py_order])),
        ("opt_A_hazard_H15_h15_tau1", np.array([calc_hazard_survival(scenes_by_work[wid], 15, h_ref=15, prior_tau=1.0) for wid in all_wids_py_order])),
        ("opt_B_scene_block_median_W6", np.array([calc_scene_block_median(scenes_by_work[wid], 6) for wid in all_wids_py_order])),
        ("opt_B_scene_block_median_W8", np.array([calc_scene_block_median(scenes_by_work[wid], 8) for wid in all_wids_py_order])),
    ]

    candidate_eval = {}
    for cname, cvec in cand_defs:
        n_exact_0 = int(np.sum(cvec <= 1e-9))
        n_exact_1 = int(np.sum(cvec >= 1.0 - 1e-9))
        n_unique = int(len(np.unique(np.round(cvec, 6))))
        candidate_eval[cname] = {
            "mean": float(cvec.mean()),
            "std": float(cvec.std()),
            "min": float(cvec.min()),
            "max": float(cvec.max()),
            "n_unique_values": n_unique,
            "n_at_0": n_exact_0,
            "n_at_1": n_exact_1,
            "vs_log_chars": corr_pair(cvec, v106["log_c_py"]),
            "vs_log_tw": corr_pair(cvec, v106["log_tw"]),
            "nhsic_vs_log_chars": nhsic_vs_log_chars(cvec, v106["log_c_py"]),
            "author_discrim": author_discrimination(cvec, authors_106),
        }

    # Check prior_tau sensitivity for Hazard H=10, h_ref=5
    tau_sensitivity = []
    for ptau in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]:
        vec = np.array([calc_hazard_survival(scenes_by_work[wid], 10, h_ref=5, prior_tau=ptau) for wid in all_wids_py_order])
        tau_sensitivity.append({
            "prior_tau": ptau,
            "mean": float(vec.mean()),
            "std": float(vec.std()),
            "rho_vs_log_chars": float(spearmanr(vec, v106["log_c_py"]).statistic),
            "nhsic_vs_log_chars": nhsic_vs_log_chars(vec, v106["log_c_py"]),
            "eta2_author": author_discrimination(vec, authors_106)["eta2_author"],
        })

    results = {
        "meta": {
            "script": "python/scripts/b20_kamae/run_b20_kamae_analysis.py",
            "scenes_jsonl_sha256": hashlib.sha256(SCENES_JSONL.read_bytes()).hexdigest().upper(),
            "n_works_total": len(all_wids_py_order),
        },
        "task1_parity": task1_parity,
        "task2_recalculation": {
            "excluded_in_f6": excluded_f6,
            "conditions": conditions,
            "decomposition": decomposition,
        },
        "task3_horizon_bias": {
            "corpus_hook_stats": corpus_hook_stats,
            "isf_boundary_mechanics": isf_mechanics,
            "fixed_horizon": fixed_horizon_results,
            "scene_block_median": block_median_results,
            "relative_time_horizon": rel_horizon_results,
            "relative_time_other": rel_time_other,
            "prefix_truncation_summary": trunc_summary,
            "prefix_truncation_200_traces": per_work_200_traces,
            "prefix_truncation_drift_7works": drift_7works,
        },
        "task4_p3_candidates": {
            "author_labels": labels,
            "candidates": candidate_eval,
            "hazard_H10_h5_prior_tau_sensitivity": tau_sensitivity,
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print("Task1 max diff hook_survival:", task1_parity["max_abs_diff_hook_survival"])
    print("Task1 max diff hook_survival_idf (with idf):", task1_parity["max_abs_diff_hook_survival_idf_with_corpus_idf"])
    print("Task2 (a) F6 exact:", conditions["a_rust_f6_exact_n105_nonws_raw"]["kamae_hook_survival"]["spearman_rho"])
    print("Task2 (b2) Py B-4 exact:", conditions["b2_python_b4_exact_n106_pylen_ols_resid"]["kamae_hook_survival"]["spearman_rho"])
    print("Task2 (c1) N=106 PyLen raw:", conditions["c1_n106_pylen_raw"]["kamae_hook_survival"]["spearman_rho"])
    print("Task3 Truncation 20->200 drift:", drift_7works)


if __name__ == "__main__":
    main()
