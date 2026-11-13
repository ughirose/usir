#!/usr/bin/env python
"""USIR v5 Stage B-16 Task 3: Build real_revision_pairs_v1_1.jsonl and verify Python/Rust v2 parity.

- Keeps real_revision_pairs_v1.jsonl (415 pairs) completely untouched.
- Evaluates the 11 new same-orthography Aozora Bunko candidates identified in Stage B-15
  (prioritizing Mori Ogai 'Takasebune' and Higuchi Ichiyo 'Wakaremichi') using the exact
  B-12/B-13 extraction pipeline (Dice gate >= ALIGN_MIN_SIMILARITY, l_min >= 50, header exclusion).
- Records reasons when candidates yield 0 pairs (identical text, <50 chars, low Dice / different article).
- Verifies 100% Python/Rust v2 parity across all v1.1 pairs (including all 415 v1 pairs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_real_revision_pairs_v1 as b12  # noqa: E402
from usir_py.v5.f14_hybrid_noise_floor import (  # noqa: E402
    ALIGN_MIN_SIMILARITY,
    char_bigram_dice,
    compute_sha256,
    evaluate_f14_hybrid_pair,
    load_f14_noise_floor,
    window_count_kasane,
)

V1_DATASET_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1.jsonl"
V1_1_DATASET_OUT_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1_1.jsonl"
F17_NOISE_FLOOR_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_all_novel.json"
NISHIKI_EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")
SUMMARY_OUT_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_1_eval_rule_v2_summary.json"

# 11 new same-orthography candidates from Stage B-15 Task 2 (prioritizing novels first)
NEW_CANDIDATE_PAIRS: List[Tuple[str, str, str, str, str, bool]] = [
    ("ogai_takasebune_ed", "000692", "000691", "first_publication_vs_book", "森鴎外『高瀬舟』(中央公論初出 vs 日本現代文學全集7)", True),
    ("ichiyo_wakaremichi_ed", "000391", "056008", "cross_edition_variant", "樋口一葉『わかれ道』(現代日本文學全集10 vs 樋口一葉全集2)", True),
    ("makino_seibugeki_ed", "045218", "052796", "cross_edition_variant", "牧野信一『西部劇通信』(全集3 vs 全集4)", True),
    ("kishida_tonari_ed", "053193", "044891", "cross_edition_variant", "岸田国士『隣の花』(全集3 vs 全集28, 戯曲)", False),
    ("nakahara_samui_ed", "051317", "051892", "cross_edition_variant", "中原中也『寒い夜の自我像』(詩集 vs 新編全集2, 詩)", False),
    ("miyamoto_kotoshi_ed", "003276", "004044", "cross_edition_variant", "宮本百合子『今年こそは』(全集別巻比較)", False),
    ("miyamoto_bungei_ed", "002781", "002784", "cross_edition_variant", "宮本百合子『文芸時評』(別巻比較)", False),
    ("miyamoto_mudai1_ed", "004158", "007913", "cross_edition_variant", "宮本百合子『無題（一）』(別巻比較)", False),
    ("miyamoto_mudai2_ed", "004161", "007917", "cross_edition_variant", "宮本百合子『無題（二）』(別巻比較)", False),
    ("miyamoto_mudai3_ed", "004162", "007929", "cross_edition_variant", "宮本百合子『無題（三）』(別巻比較)", False),
    ("miyamoto_mudai4_ed", "004195", "007936", "cross_edition_variant", "宮本百合子『無題（四）』(別巻比較)", False),
]


def _diagnose_candidate_rejection(
    harvest_recs: List[Dict[str, Any]],
    paras_a: List[str],
    paras_b: List[str],
) -> Dict[str, int]:
    diag = {
        "n_paras_a": len(paras_a),
        "n_paras_b": len(paras_b),
        "n_aligned_pairs": 0,
        "identical_text": 0,
        "dice_below_min": 0,
        "non_content_or_lt50": 0,
        "l_min_lt50": 0,
        "duplicate_text_pair": 0,
        "accepted": 0,
    }
    for hrec in harvest_recs:
        if hrec.get("kind") != "pair":
            continue
        diag["n_aligned_pairs"] += 1
        pa = hrec["text_before"].strip()
        pb = hrec["text_after"].strip()
        if pa == pb:
            diag["identical_text"] += 1
            continue
        dice_sim = char_bigram_dice(pa, pb)
        if dice_sim < ALIGN_MIN_SIMILARITY or dice_sim >= 1.0:
            diag["dice_below_min"] += 1
            continue
        if b12._is_non_content_header_or_metadata(pa) or b12._is_non_content_header_or_metadata(pb):
            diag["non_content_or_lt50"] += 1
            continue
        if min(len(pa), len(pb)) < 50:
            diag["l_min_lt50"] += 1
            continue
        diag["accepted"] += 1
    return diag


def verify_parity_v2(
    rows: List[Dict[str, Any]],
    nf_path: Path,
    nishiki_exe: Path,
) -> Dict[str, Any]:
    nf_data, prov = load_f14_noise_floor(nf_path)
    discrepancies: List[Dict[str, Any]] = []
    max_per_dim_diff = 0.0
    for idx, row in enumerate(rows):
        tb, ta = row["text_before"], row["text_after"]
        rust_rec = b12._run_rust_harvest_pair(nishiki_exe, nf_path, tb, ta)
        if rust_rec is None or rust_rec.get("kind") != "pair":
            discrepancies.append({"pair_id": row["pair_id"], "reason": "rust_unaligned"})
            continue
        py_ev = evaluate_f14_hybrid_pair(
            tb, ta, nf_data, delta_kasane11=rust_rec["delta_kasane11"], provenance=prov, apply_dice_gate=True
        )
        rn = rust_rec["noise_floor_ratio_kasane11"]
        if (
            rust_rec["window_counts"] != py_ev["window_counts"]
            or rn["n_evaluated"] != py_ev["n_evaluated"]
            or rn["band"] != py_ev["band"]
            or rn.get("ac_sigma_source") != py_ev["ac_sigma_source"]
        ):
            discrepancies.append({"pair_id": row["pair_id"], "reason": "metadata_mismatch"})
            continue
        rust_pd = rn.get("per_dim") or {}
        py_ratios = py_ev.get("ratios") or {}
        py_pd = py_ratios.get("per_dim") or {}
        if set(rust_pd.keys()) != set(py_pd.keys()):
            discrepancies.append({"pair_id": row["pair_id"], "reason": "per_dim_keys_mismatch"})
            continue
        for k in ("max", "mean", "median"):
            d = abs(float(py_ratios[k]) - float(rn[k]))
            if d > 1e-12:
                discrepancies.append({"pair_id": row["pair_id"], "reason": f"{k}_diff_{d}"})
                break
        else:
            for k in py_pd:
                d = abs(float(rust_pd[k]) - float(py_pd[k]))
                if d > max_per_dim_diff:
                    max_per_dim_diff = d
                if d > 1e-12:
                    discrepancies.append({"pair_id": row["pair_id"], "reason": f"dim_{k}_diff_{d}"})
                    break
        if (idx + 1) % 100 == 0:
            print(f"  parity verified {idx + 1}/{len(rows)} pairs (max_diff={max_per_dim_diff:.3e})...")
    return {
        "total_verified": len(rows),
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies[:20],
        "max_per_dim_abs_diff": max_per_dim_diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    v1_rows = [json.loads(ln) for ln in V1_DATASET_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"Loaded v1 dataset: {len(v1_rows)} pairs")
    seen_text_pairs = {(r["text_before"], r["text_after"]) for r in v1_rows}

    index_map = b12.load_aozora_index_map()
    new_rows: List[Dict[str, Any]] = []
    candidate_logs: List[Dict[str, Any]] = []

    for group_id, wid_a, wid_b, rel, desc, is_novel in NEW_CANDIDATE_PAIRS:
        row_a = index_map.get(wid_a)
        row_b = index_map.get(wid_b)
        if row_a is None or row_b is None:
            candidate_logs.append({"group_id": group_id, "desc": desc, "status": "missing_in_index", "extracted": 0})
            continue
        txt_a = b12.fetch_or_load_edition_text(row_a, offline=args.offline)
        txt_b = b12.fetch_or_load_edition_text(row_b, offline=args.offline)
        if txt_a is None or txt_b is None:
            candidate_logs.append({"group_id": group_id, "desc": desc, "status": "download_failed", "extracted": 0})
            continue

        paras_a = b12.split_aozora_true_paragraphs(txt_a)
        paras_b = b12.split_aozora_true_paragraphs(txt_b)
        harvest_recs = b12._run_rust_harvest_work_pair(NISHIKI_EXE, F17_NOISE_FLOOR_PATH, paras_a, paras_b)
        diag = _diagnose_candidate_rejection(harvest_recs, paras_a, paras_b)

        extracted_for_work: List[Dict[str, Any]] = []
        for hrec in harvest_recs:
            if hrec.get("kind") != "pair":
                continue
            pa = hrec["text_before"].strip()
            pb = hrec["text_after"].strip()
            if pa == pb:
                continue
            dice_sim = char_bigram_dice(pa, pb)
            if dice_sim < ALIGN_MIN_SIMILARITY or dice_sim >= 1.0:
                continue
            if b12._is_non_content_header_or_metadata(pa) or b12._is_non_content_header_or_metadata(pb):
                continue
            l_a, l_b = len(pa), len(pb)
            l_min, l_max = min(l_a, l_b), max(l_a, l_b)
            if l_min < 50:
                continue
            if (pa, pb) in seen_text_pairs:
                diag["duplicate_text_pair"] += 1
                continue
            seen_text_pairs.add((pa, pb))

            w_a = window_count_kasane(pa)
            w_b = window_count_kasane(pb)
            mixed = (w_a >= 2) != (w_b >= 2)
            band = "S" if l_max <= 250 else ("M" if l_max <= 500 else "L")
            bi = int(hrec["before_index"])
            aj = int(hrec["after_index"])
            extracted_for_work.append({
                "pair_id": f"{group_id}_p{bi}_p{aj}",
                "work_group_id": group_id,
                "author": f"{row_a['姓']}{row_a['名']}",
                "work_description": desc,
                "variant_relationship": rel,
                "same_orthography": row_a.get("文字遣い種別") == row_b.get("文字遣い種別"),
                "provenance": {
                    "source": "Aozora Bunko (public domain, 作品著作権フラグ=なし)",
                    "edition_a": {
                        "work_id": wid_a,
                        "title": row_a["作品名"],
                        "subtitle": row_a.get("副題", ""),
                        "char_type": row_a.get("文字遣い種別", ""),
                        "base_book": row_a.get("底本名1", ""),
                        "source_url": row_a.get("テキストファイルURL", ""),
                        "card_url": row_a.get("図書カードURL", ""),
                        "paragraph_index": bi,
                    },
                    "edition_b": {
                        "work_id": wid_b,
                        "title": row_b["作品名"],
                        "subtitle": row_b.get("副題", ""),
                        "char_type": row_b.get("文字遣い種別", ""),
                        "base_book": row_b.get("底本名1", ""),
                        "source_url": row_b.get("テキストファイルURL", ""),
                        "card_url": row_b.get("図書カードURL", ""),
                        "paragraph_index": aj,
                    },
                },
                "text_before": pa,
                "text_after": pb,
                "len_before": l_a,
                "len_after": l_b,
                "l_min": l_min,
                "l_max": l_max,
                "band": band,
                "window_counts": [w_a, w_b],
                "mixed_window_pair": mixed,
                "dice_similarity": round(dice_sim, 6),
            })

        new_rows.extend(extracted_for_work)
        s_c = sum(1 for r in extracted_for_work if r["band"] == "S")
        m_c = sum(1 for r in extracted_for_work if r["band"] == "M")
        l_c = sum(1 for r in extracted_for_work if r["band"] == "L")
        mx_c = sum(1 for r in extracted_for_work if r["mixed_window_pair"])
        candidate_logs.append({
            "group_id": group_id,
            "desc": desc,
            "is_novel": is_novel,
            "extracted": len(extracted_for_work),
            "S": s_c,
            "M": m_c,
            "L": l_c,
            "mixed": mx_c,
            "diag": diag,
        })
        print(f"[cand] {group_id}: extracted={len(extracted_for_work)} (S={s_c}, M={m_c}, L={l_c}, mixed={mx_c}) diag={diag}")

    v1_1_rows = v1_rows + new_rows
    V1_1_DATASET_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with V1_1_DATASET_OUT_PATH.open("w", encoding="utf-8") as fh:
        for r in v1_1_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved v1.1 dataset: {len(v1_1_rows)} pairs (v1={len(v1_rows)} + new={len(new_rows)}) -> {V1_1_DATASET_OUT_PATH}")

    parity = None
    if not args.skip_parity:
        print("Verifying Python/Rust v2 parity on all v1.1 pairs...")
        parity = verify_parity_v2(v1_1_rows, F17_NOISE_FLOOR_PATH, NISHIKI_EXE)
        print(f"Parity result: {parity}")

    summary = {
        "v1_count": len(v1_rows),
        "new_count": len(new_rows),
        "v1_1_count": len(v1_1_rows),
        "v1_1_sha256": compute_sha256(V1_1_DATASET_OUT_PATH),
        "band_counts_v1_1": {
            "S": sum(1 for r in v1_1_rows if r["band"] == "S"),
            "M": sum(1 for r in v1_1_rows if r["band"] == "M"),
            "L": sum(1 for r in v1_1_rows if r["band"] == "L"),
            "mixed": sum(1 for r in v1_1_rows if r["mixed_window_pair"]),
        },
        "candidate_logs": candidate_logs,
        "parity_v2_f17": parity,
    }
    SUMMARY_OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary -> {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
