#!/usr/bin/env python
"""USIR v5 Stage B-22 Task 2: Quantitative Comparison of 3 ISF Modification Variants.

Variants:
  (a) a_current: within-work ISF >= 1.2 with T_W < 4 => sf <= 1 guard
  (b) b_t_eff_10: within-work ISF with effective denominator T_eff = max(T_W, 10)
  (c) c_corpus_fixed: corpus-wide ISF across all 4,576 scenes of the 106 works

Evaluated on BOTH:
  - Series I (B-20 re-annotation basis / commit acce484: strip_commas + extract_kanji_tokens)
  - Series II (Frozen aozora_106works_scenes.jsonl basis / commit 913b0b5: raw regex [\u4e00-\u9fff]{2,})

Outputs re-annotated Kamae JSONL files (Series I) and `b22_task2_isf_comparison.json`
strictly under `python/scripts/b22_kamae_haz/`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

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

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "b22_task2_isf_comparison.json"

TW_BUCKETS = [
    ("T_W < 10", lambda tw: tw < 10),
    ("10 <= T_W < 20", lambda tw: 10 <= tw < 20),
    ("20 <= T_W < 50", lambda tw: 20 <= tw < 50),
    ("50 <= T_W < 100", lambda tw: 50 <= tw < 100),
    ("T_W >= 100", lambda tw: tw >= 100),
]


def extract_kanji_913(text: str) -> Set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]{2,}", text))


def run_series(
    basis_name: str,
    wids: List[str],
    w_by_wid: Dict[str, Dict[str, Any]],
    sc_texts_by_wid: Dict[str, List[str]],
    log_chars_py: np.ndarray,
    log_chars_nonws: np.ndarray,
    log_tw: np.ndarray,
    log_sents: np.ndarray,
    authors: List[str],
    write_jsonl: bool = False,
) -> Dict[str, Any]:
    use_strip_commas = basis_name == "series1_b20_reannot_acce484"

    # Pass 1: within-work and corpus-wide scene frequencies
    work_sf_by_wid: Dict[str, Dict[str, int]] = {}
    corpus_sf: Dict[str, int] = {}
    total_corpus_scenes = 0
    for wid in wids:
        scs = sc_texts_by_wid[wid]
        total_corpus_scenes += len(scs)
        if use_strip_commas:
            wsf = compute_work_scene_frequencies(scs)
        else:
            wsf = {}
            for sc in scs:
                for tok in extract_kanji_913(sc):
                    wsf[tok] = wsf.get(tok, 0) + 1
        work_sf_by_wid[wid] = wsf
        for tok, cnt in wsf.items():
            corpus_sf[tok] = corpus_sf.get(tok, 0) + cnt

    variants = [
        ("a_current", "reannot_isf_a_current.jsonl"),
        ("b_t_eff_10", "reannot_isf_b_teff10.jsonl"),
        ("c_corpus_fixed", "reannot_isf_c_corpus.jsonl"),
    ]
    series_out: Dict[str, Any] = {
        "total_corpus_scenes": total_corpus_scenes,
        "corpus_vocab_size": len(corpus_sf),
        "variants": {},
    }

    for vname, jsonl_name in variants:
        scenes_by_wid: Dict[str, List[Dict[str, Any]]] = {}
        hook_records: List[Dict[str, Any]] = []
        all_jsonl_lines: List[str] = []

        for wid in wids:
            scs = sc_texts_by_wid[wid]
            tw = len(scs)
            wsf = work_sf_by_wid[wid]
            author_id = w_by_wid[wid]["author_id"]

            active_hooks: List[Dict[str, Any]] = []
            work_hooks_map: Dict[int, Dict[str, Any]] = {}
            h_seq = 1
            w_scenes: List[Dict[str, Any]] = []

            for idx, sc_txt in enumerate(scs):
                proc_txt = strip_commas(sc_txt) if use_strip_commas else sc_txt
                sents = [s for s in SENT_SPLIT.split(proc_txt) if s.strip()]
                opened_hooks: List[Dict[str, Any]] = []

                for sent in sents:
                    cs = sent.strip() if use_strip_commas else sent.strip()
                    if RE_HOOK_TRIG.search(cs):
                        label = cs[:30]
                        ktoks = extract_kanji_tokens(label) if use_strip_commas else extract_kanji_913(label)
                        if vname == "a_current":
                            spec = filter_hook_tokens_by_isf(ktoks, wsf, total_scenes=tw, isf_threshold=1.2)
                        elif vname == "b_t_eff_10":
                            spec = filter_hook_tokens_by_isf(ktoks, wsf, total_scenes=max(tw, 10), isf_threshold=1.2)
                        else:
                            spec = filter_hook_tokens_by_isf(
                                ktoks, corpus_sf, total_scenes=total_corpus_scenes, isf_threshold=1.2
                            )

                        is_fallback = not bool(spec)
                        if is_fallback:
                            fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                            final_spec = fb if fb else {cs[:4]}
                        else:
                            final_spec = spec

                        max_wsf_after_fb = max((wsf.get(t, 0) for t in final_spec), default=0)
                        all_sf_le1_after_fb = all(wsf.get(t, 0) <= 1 for t in final_spec)
                        # If all tokens appear in <= 1 scene within the work, matching any later scene is impossible
                        # (for fallback 4-char non-kanji strings not in wsf, check direct substring hits in later scenes).
                        can_match_later_scene = any(
                            any(tok in (strip_commas(later_sc) if use_strip_commas else later_sc) for tok in final_spec)
                            for later_sc in scs[idx + 1 :]
                        )

                        h_meta = {
                            "work_id": wid,
                            "T_W": tw,
                            "open_idx": idx,
                            "is_fallback": is_fallback,
                            "isf_passed_sf1_only": (not is_fallback) and all(wsf.get(t, 0) <= 1 for t in spec),
                            "isf_passed_has_sf2_plus": (not is_fallback) and any(wsf.get(t, 0) >= 2 for t in spec),
                            "final_spec_sf1_only": all_sf_le1_after_fb,
                            "structurally_unrecoverable_later": not can_match_later_scene,
                            "closed": False,
                            "max_wsf_after_fb": max_wsf_after_fb,
                        }
                        work_hooks_map[h_seq] = h_meta
                        hook_records.append(h_meta)

                        opened_hooks.append({"id": h_seq, "label": label})
                        active_hooks.append({"id": h_seq, "tokens": final_spec, "open_idx": idx})
                        h_seq += 1
                        if len(opened_hooks) >= 2:
                            break

                closed_ids: List[int] = []
                has_resolv = bool(RE_RESOLV_MARKER.search(proc_txt))
                for h in list(active_hooks):
                    if h["open_idx"] < idx and has_resolv:
                        if any(tok in proc_txt for tok in h["tokens"]):
                            closed_ids.append(h["id"])
                            work_hooks_map[h["id"]]["closed"] = True
                            active_hooks.remove(h)

                rec = {
                    "work_id": wid,
                    "author_id": author_id,
                    "scene_index": idx,
                    "scene_count": tw,
                    "char_len": max(len(sc_txt), 1),
                    "kamae": {
                        "opened_hooks": opened_hooks,
                        "closed_hook_ids": closed_ids,
                    },
                }
                w_scenes.append(rec)
                if write_jsonl:
                    all_jsonl_lines.append(json.dumps(rec, ensure_ascii=False))

            scenes_by_wid[wid] = w_scenes

        if write_jsonl:
            out_path = OUT_DIR / jsonl_name
            out_path.write_text("\n".join(all_jsonl_lines) + "\n", encoding="utf-8")

        # Compute variant-specific HookIdf
        var_idf = b20.compute_rust_hook_idf(scenes_by_wid)
        var_weights = var_idf["weight"]

        # Summary across hooks & by T_W bucket
        def summarize_hooks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            n_h = len(rows)
            if n_h == 0:
                return {"n_hooks": 0}
            n_fb = sum(1 for r in rows if r["is_fallback"])
            n_isf_pass = n_h - n_fb
            n_sf1_pass = sum(1 for r in rows if r["isf_passed_sf1_only"])
            n_sf2_pass = sum(1 for r in rows if r["isf_passed_has_sf2_plus"])
            n_final_sf1 = sum(1 for r in rows if r["final_spec_sf1_only"])
            n_unrecov_later = sum(1 for r in rows if r["structurally_unrecoverable_later"])
            n_closed = sum(1 for r in rows if r["closed"])
            return {
                "n_hooks": n_h,
                "n_closed": n_closed,
                "closure_rate": n_closed / n_h,
                "unclosed_rate": (n_h - n_closed) / n_h,
                "fallback_count": n_fb,
                "fallback_rate": n_fb / n_h,
                "isf_passed_count": n_isf_pass,
                "sf2_plus_isf_pass_count": n_sf2_pass,
                "sf2_plus_isf_pass_rate": n_sf2_pass / n_h,
                "sf1_only_pass_count": n_sf1_pass,
                "sf1_only_pass_rate": n_sf1_pass / n_h,
                "sf1_only_among_isf_passed_rate": (n_sf1_pass / n_isf_pass) if n_isf_pass > 0 else 0.0,
                "final_spec_sf1_only_count": n_final_sf1,
                "final_spec_sf1_only_rate": n_final_sf1 / n_h,
                "structurally_unrecoverable_count": n_unrecov_later,
                "structurally_unrecoverable_rate": n_unrecov_later / n_h,
            }

        bucket_stats = {}
        for bname, pred in TW_BUCKETS:
            b_rows = [r for r in hook_records if pred(r["T_W"])]
            b_wids = [wid for wid in wids if pred(len(scenes_by_wid[wid]))]
            raw_b = [b20.calc_raw_survival(scenes_by_wid[wid]) for wid in b_wids]
            haz10_b = [calc_hazard_survival_detail(scenes_by_wid[wid], 10, 5, 1.0, None)["s_haz"] for wid in b_wids]
            bucket_stats[bname] = {
                "n_works": len(b_wids),
                "mean_raw_survival": float(np.mean(raw_b)),
                "mean_S_haz_H10": float(np.mean(haz10_b)),
                **summarize_hooks(b_rows),
            }

        # Metric vectors across 106 works
        raw_vec = np.array([b20.calc_raw_survival(scenes_by_wid[wid]) for wid in wids], dtype=float)
        haz10_unw = np.array(
            [calc_hazard_survival_detail(scenes_by_wid[wid], 10, 5, 1.0, None)["s_haz"] for wid in wids],
            dtype=float,
        )
        haz10_idf = np.array(
            [calc_hazard_survival_detail(scenes_by_wid[wid], 10, 5, 1.0, var_weights)["s_haz"] for wid in wids],
            dtype=float,
        )
        haz15_unw = np.array(
            [calc_hazard_survival_detail(scenes_by_wid[wid], 15, 5, 1.0, None)["s_haz"] for wid in wids],
            dtype=float,
        )
        haz15_idf = np.array(
            [calc_hazard_survival_detail(scenes_by_wid[wid], 15, 5, 1.0, var_weights)["s_haz"] for wid in wids],
            dtype=float,
        )

        metrics_eval = {
            "raw_hook_survival": evaluate_vector(
                raw_vec, log_chars_py, log_chars_nonws, log_tw, log_sents, authors
            ),
            "S_haz_H10_unweighted": evaluate_vector(
                haz10_unw, log_chars_py, log_chars_nonws, log_tw, log_sents, authors
            ),
            "S_haz_H10_idf_weighted": evaluate_vector(
                haz10_idf, log_chars_py, log_chars_nonws, log_tw, log_sents, authors
            ),
            "S_haz_H15_unweighted": evaluate_vector(
                haz15_unw, log_chars_py, log_chars_nonws, log_tw, log_sents, authors
            ),
            "S_haz_H15_idf_weighted": evaluate_vector(
                haz15_idf, log_chars_py, log_chars_nonws, log_tw, log_sents, authors
            ),
        }

        series_out["variants"][vname] = {
            "jsonl_file": jsonl_name if write_jsonl else None,
            "hook_idf": var_idf,
            "overall_hooks": summarize_hooks(hook_records),
            "tw_bucket_breakdown": bucket_stats,
            "metrics_evaluation": metrics_eval,
        }

    return series_out


def main() -> None:
    works = b20.load_works()
    w_by_wid = {w["id"].split(":")[-1]: w for w in works}
    wids = [w["id"].split(":")[-1] for w in works]
    authors = [w["author_id"] for w in works]

    splitter = AozoraCorpusSplitter(data_dir=ROOT / "data")
    manifests = splitter.load_manifests()
    manifests_by_wid = {m["work_id"]: m for m in manifests}

    sc_texts_by_wid = {
        wid: split_work_into_scenes(splitter.get_cached_text(manifests_by_wid[wid])) for wid in wids
    }
    log_chars_py = np.array([math.log(len(w_by_wid[wid]["text"])) for wid in wids], dtype=float)
    log_chars_nonws = np.array(
        [math.log(b20.count_non_whitespace_chars(w_by_wid[wid]["text"])) for wid in wids], dtype=float
    )
    log_tw = np.array([math.log(len(sc_texts_by_wid[wid])) for wid in wids], dtype=float)
    log_sents = np.array(
        [math.log(b20.count_rust_sentences(w_by_wid[wid]["text"])) for wid in wids], dtype=float
    )

    # Verify frozen JSONL parity on Series II (913b0b5)
    frozen_scenes_by_wid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with b20.SCENES_JSONL.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            frozen_scenes_by_wid[s["work_id"]].append(s)
    for wid in frozen_scenes_by_wid:
        frozen_scenes_by_wid[wid].sort(key=lambda s: s["scene_index"])

    s1 = run_series(
        "series1_b20_reannot_acce484",
        wids,
        w_by_wid,
        sc_texts_by_wid,
        log_chars_py,
        log_chars_nonws,
        log_tw,
        log_sents,
        authors,
        write_jsonl=True,
    )
    s2 = run_series(
        "series2_frozen_913b0b5",
        wids,
        w_by_wid,
        sc_texts_by_wid,
        log_chars_py,
        log_chars_nonws,
        log_tw,
        log_sents,
        authors,
        write_jsonl=False,
    )

    # Exact verification: Series II (a_current) vs frozen data/v5/aozora_106works_scenes.jsonl
    # Re-run deterministic comparison against frozen JSONL directly
    mismatches_vs_frozen = 0
    for wid in wids:
        scs = sc_texts_by_wid[wid]
        tw = len(scs)
        wsf: Dict[str, int] = {}
        for sc in scs:
            for tok in extract_kanji_913(sc):
                wsf[tok] = wsf.get(tok, 0) + 1
        active_hooks: List[Dict[str, Any]] = []
        h_seq = 1
        for idx, sc_txt in enumerate(scs):
            sents = [s for s in SENT_SPLIT.split(sc_txt) if s.strip()]
            op_hooks = []
            for sent in sents:
                if RE_HOOK_TRIG.search(sent):
                    label = sent.strip()[:30]
                    ktoks = extract_kanji_913(label)
                    spec = filter_hook_tokens_by_isf(ktoks, wsf, total_scenes=tw, isf_threshold=1.2)
                    if not spec:
                        fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                        spec = fb if fb else {label[:4]}
                    op_hooks.append({"id": h_seq, "label": label})
                    active_hooks.append({"id": h_seq, "tokens": spec, "open_idx": idx})
                    h_seq += 1
                    if len(op_hooks) >= 2:
                        break
            cl_ids = []
            has_resolv = bool(RE_RESOLV_MARKER.search(sc_txt))
            for h in list(active_hooks):
                if h["open_idx"] < idx and has_resolv:
                    if any(tok in sc_txt for tok in h["tokens"]):
                        cl_ids.append(h["id"])
                        active_hooks.remove(h)
            orig = frozen_scenes_by_wid[wid][idx]["kamae"]
            if op_hooks != orig["opened_hooks"] or cl_ids != orig["closed_hook_ids"]:
                mismatches_vs_frozen += 1
    assert mismatches_vs_frozen == 0, f"Expected 0 mismatches vs frozen JSONL, got {mismatches_vs_frozen}"

    # Check frozen JSONL parity against Series II (a_current) metrics
    frozen_raw = np.array([b20.calc_raw_survival(frozen_scenes_by_wid[wid]) for wid in wids], dtype=float)
    frozen_haz10 = np.array(
        [calc_hazard_survival_detail(frozen_scenes_by_wid[wid], 10, 5, 1.0, None)["s_haz"] for wid in wids],
        dtype=float,
    )

    # Corpus-fixed ISF threshold diagnostic (Series I & II): which tokens are actually filtered?
    corp_sf_s1: Dict[str, int] = {}
    total_sc = 0
    for wid in wids:
        scs = sc_texts_by_wid[wid]
        total_sc += len(scs)
        for tok, cnt in compute_work_scene_frequencies(scs).items():
            corp_sf_s1[tok] = corp_sf_s1.get(tok, 0) + cnt
    max_allowed_corp_sf = total_sc / math.exp(1.2) - 1.0
    excluded_corp_tokens = [
        {"token": tok, "sf": cnt, "in_common_stopwords": tok in COMMON_HOOK_STOPWORDS}
        for tok, cnt in sorted(corp_sf_s1.items(), key=lambda x: (-x[1], x[0]))
        if cnt > max_allowed_corp_sf
    ]
    top20_corp_tokens = [
        {"token": tok, "sf": cnt, "in_common_stopwords": tok in COMMON_HOOK_STOPWORDS}
        for tok, cnt in sorted(corp_sf_s1.items(), key=lambda x: (-x[1], x[0]))[:20]
    ]

    # Prefix truncation K=20 -> 200 on the 7 long works (T_W >= 200) across ISF variants (Series I)
    long_wids_200 = [wid for wid in wids if len(sc_texts_by_wid[wid]) >= 200]
    trunc_drift_7works: Dict[str, Any] = {}
    for vname in ["a_current", "b_t_eff_10", "c_corpus_fixed"]:
        diffs_h10 = []
        diffs_h15 = []
        for wid in long_wids_200:
            pair_vals: Dict[int, Tuple[float, float]] = {}
            for k in [20, 200]:
                scs_k = sc_texts_by_wid[wid][:k]
                tw_k = len(scs_k)
                wsf_k = compute_work_scene_frequencies(scs_k)
                active_k: List[Dict[str, Any]] = []
                h_seq = 1
                recs_k: List[Dict[str, Any]] = []
                for idx_k, sc_txt in enumerate(scs_k):
                    clean_sc = strip_commas(sc_txt)
                    sents = [s for s in SENT_SPLIT.split(clean_sc) if s.strip()]
                    op_k = []
                    for sent in sents:
                        cs = sent.strip()
                        if RE_HOOK_TRIG.search(cs):
                            label = cs[:30]
                            ktoks = extract_kanji_tokens(label)
                            if vname == "a_current":
                                spec = filter_hook_tokens_by_isf(ktoks, wsf_k, tw_k, 1.2)
                            elif vname == "b_t_eff_10":
                                spec = filter_hook_tokens_by_isf(ktoks, wsf_k, max(tw_k, 10), 1.2)
                            else:
                                spec = filter_hook_tokens_by_isf(ktoks, corp_sf_s1, total_sc, 1.2)
                            if not spec:
                                fb = {t for t in ktoks if t not in COMMON_HOOK_STOPWORDS}
                                spec = fb if fb else {cs[:4]}
                            op_k.append({"id": h_seq, "label": label})
                            active_k.append({"id": h_seq, "tokens": spec, "open_idx": idx_k})
                            h_seq += 1
                            if len(op_k) >= 2:
                                break
                    cl_k = []
                    has_res = bool(RE_RESOLV_MARKER.search(clean_sc))
                    for h in list(active_k):
                        if h["open_idx"] < idx_k and has_res and any(t in clean_sc for t in h["tokens"]):
                            cl_k.append(h["id"])
                            active_k.remove(h)
                    recs_k.append(
                        {
                            "work_id": wid,
                            "scene_index": idx_k,
                            "scene_count": tw_k,
                            "kamae": {"opened_hooks": op_k, "closed_hook_ids": cl_k},
                        }
                    )
                s10 = calc_hazard_survival_detail(recs_k, 10, 5, 1.0, None)["s_haz"]
                s15 = calc_hazard_survival_detail(recs_k, 15, 5, 1.0, None)["s_haz"]
                pair_vals[k] = (s10, s15)
            diffs_h10.append(pair_vals[20][0] - pair_vals[200][0])
            diffs_h15.append(pair_vals[20][1] - pair_vals[200][1])
        trunc_drift_7works[vname] = {
            "H10_20_minus_200_mean": float(np.mean(diffs_h10)),
            "H10_20_minus_200_abs_mean": float(np.mean(np.abs(diffs_h10))),
            "H15_20_minus_200_mean": float(np.mean(diffs_h15)),
            "H15_20_minus_200_abs_mean": float(np.mean(np.abs(diffs_h15))),
        }

    out = {
        "meta": {
            "script": "python/scripts/b22_kamae_haz/task2_isf_variants.py",
            "frozen_scenes_jsonl_sha256": hashlib.sha256(b20.SCENES_JSONL.read_bytes()).hexdigest().upper(),
            "series2_a_current_mismatches_vs_frozen_jsonl_across_4576_scenes": mismatches_vs_frozen,
            "frozen_raw_mean": float(frozen_raw.mean()),
            "frozen_haz10_mean": float(frozen_haz10.mean()),
        },
        "corpus_isf_diagnostics_series1": {
            "total_scenes": total_sc,
            "vocab_size": len(corp_sf_s1),
            "max_allowed_sf_at_1_2": max_allowed_corp_sf,
            "excluded_tokens": excluded_corp_tokens,
            "top20_corpus_tokens": top20_corp_tokens,
        },
        "reannot_truncation_drift_7works_series1": trunc_drift_7works,
        "series1_b20_reannot_acce484": s1,
        "series2_frozen_913b0b5": s2,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print("Series II a_current mismatches vs frozen JSONL:", mismatches_vs_frozen)
    for sname, sdata in [("Series I (acce484)", s1), ("Series II (913b0b5)", s2)]:
        print(f"\n--- {sname} ---")
        for vname, vdata in sdata["variants"].items():
            ov = vdata["overall_hooks"]
            lt10 = vdata["tw_bucket_breakdown"]["T_W < 10"]
            me = vdata["metrics_evaluation"]
            print(
                f"  [{vname}] hooks={ov['n_hooks']}, closed={ov['n_closed']} ({ov['closure_rate']*100:.2f}%), "
                f"T_W<10: fb={lt10['fallback_rate']*100:.2f}%, sf2+={lt10['sf2_plus_isf_pass_rate']*100:.2f}%, "
                f"sf1_pass={lt10['sf1_only_pass_rate']*100:.2f}%, final_sf1={lt10['final_spec_sf1_only_rate']*100:.2f}%, "
                f"unrecov_later={lt10['structurally_unrecoverable_rate']*100:.2f}%, unclosed={lt10['unclosed_rate']*100:.2f}%"
            )
            for mk in ["raw_hook_survival", "S_haz_H10_unweighted", "S_haz_H15_unweighted"]:
                m = me[mk]
                print(
                    f"    {mk}: mean={m['mean']:.4f}, rho_chars={m['vs_log_chars_py']['spearman_rho']:.4f}, "
                    f"r_chars={m['vs_log_chars_py']['pearson_r']:.4f}, rho_tw={m['vs_log_scene_count']['spearman_rho']:.4f}, "
                    f"rho_sents={m['vs_log_sentence_count']['spearman_rho']:.4f}, nHSIC={m['nhsic_vs_log_chars_py']:.4f}, "
                    f"eta2={m['author_discrim']['eta2_author']:.4f}"
                )


if __name__ == "__main__":
    main()
