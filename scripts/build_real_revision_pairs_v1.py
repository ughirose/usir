#!/usr/bin/env python
"""USIR v5 Stage B-12 Task 4: Build Real Revision Pairs Expanded Dataset v1 & Measure under F-15 Noise Floor.

Constructs authentic single-paragraph revision pairs exclusively from public-domain Aozora Bunko
edition/draft variants (first publication vs book, initial draft vs final, single volume vs
complete works, magazine vs collected works). Zero Tier B-1 private manuscripts, zero paid APIs,
and zero artificial padding or multi-paragraph merging.

Outputs:
  - data/usir_v5_eval/real_revision_pairs_v1.jsonl
  - docs/handoff/v10/real_revision_pairs_v1_f15_harvest_summary.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.request
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from fetch_aozora import clean_aozora_text
from usir_py.v5.f14_hybrid_noise_floor import (
    ALIGN_MIN_SIMILARITY,
    char_bigram_dice,
    compute_sha256,
    evaluate_f14_hybrid_pair,
    load_f14_noise_floor,
    window_count_kasane,
)

INDEX_CSV_PATH = REPO_ROOT / "data" / "aozora_cache" / "_index.csv"
VARIANT_CACHE_DIR = REPO_ROOT / "data" / "aozora_cache" / "edition_variants"
DATASET_OUT_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1.jsonl"
F15_NOISE_FLOOR_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_f15_all_novel.json"
EXPECTED_F15_SHA256 = "5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910"
SUMMARY_OUT_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_f15_harvest_summary.json"
NISHIKI_EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")

USER_AGENT = "usir-research-stage-b12"
DELAY_SEC = 0.4
_LAST_REQ = [0.0]


# Curated public-domain Aozora Bunko edition / draft pairs (作品著作権フラグ == "なし").
# Format: (pair_group_id, edition_a_work_id, edition_b_work_id, variant_relationship, notes)
CANDIDATE_WORK_PAIRS: List[Tuple[str, str, str, str, str]] = [
    # 1. Explicit draft / initial publication vs final edition
    ("ginga_initial_vs_final", "060681", "000456", "initial_draft_vs_final", "宮沢賢治『〔「銀河鉄道の夜」初期形一〕』 vs 『銀河鉄道の夜』(新潮文庫)"),
    ("horoki_first_vs_new", "045649", "001813", "first_publication_vs_book", "林芙美子『放浪記（初出）』 vs 『新版 放浪記』"),
    ("oda_ame_first_vs_rev", "053804", "042689", "first_publication_vs_book", "織田作之助『雨』(俗臭 初出) vs 『雨』(日本文学全集72)"),
    ("oda_horo_first_vs_rev", "053806", "000380", "first_publication_vs_book", "織田作之助『放浪』(俗臭 初出) vs 『放浪』(定本織田作之助全集1)"),
    ("oda_wagamachi_first_vs_rev", "053808", "001326", "first_publication_vs_book", "織田作之助『わが町』(俗臭 初出) vs 『わが町』(名作選集9)"),
    ("takeuchi_donso_draft_vs_rev", "054816", "054776", "draft_vs_revision", "竹内浩三『鈍走記（草稿）』 vs 『鈍走記』"),
    ("kishida_kotoba_draft1_vs_final", "044505", "044506", "initial_draft_vs_final", "岸田国士『言葉の魅力［第一稿］』 vs 『言葉の魅力』"),
    ("kishida_doin_draft1_vs_final", "052310", "052309", "initial_draft_vs_final", "岸田国士『動員挿話［第一稿］』 vs 『動員挿話（二幕）』"),
    ("kishida_fujin_alt_vs_final", "044751", "044750", "draft_vs_revision", "岸田国士『Ｓ夫人への手紙［別稿］』 vs 『Ｓ夫人への手紙』"),
    ("roka_muhon_draft_vs_final", "001708", "000371", "draft_vs_revision", "徳冨蘆花『謀叛論（草稿）』 vs 『謀叛論』"),
    ("origuchi_kokubungaku_1_vs_3", "043764", "042613", "draft_vs_revision", "折口信夫『国文学の発生（第一稿）』 vs 『国文学の発生（第三稿）』"),
    ("origuchi_kokubungaku_2_vs_4", "047022", "047023", "draft_vs_revision", "折口信夫『国文学の発生（第二稿）』 vs 『国文学の発生（第四稿）』"),
    ("origuchi_shisha_ed", "024380", "024379", "single_volume_vs_complete_works", "折口信夫『死者の書』(全集第廿四巻 vs 単行本)"),
    # 2. Magazine initial / first book vs complete works (same orthography)
    ("tanizaki_neko_ed", "059300", "059232", "single_volume_vs_complete_works", "谷崎潤一郎『猫と庄造と二人のをんな』(単行本 vs 全集18)"),
    ("ichiyo_takekurabe_ed", "055671", "000389", "first_publication_vs_book", "樋口一葉『たけくらべ』(文藝倶樂部初出 vs 現代日本文學全集10)"),
    ("ichiyo_nigorie_ed", "055672", "000387", "first_publication_vs_book", "樋口一葉『にごりえ』(文藝倶樂部初出 vs 現代日本文學全集10)"),
    ("ichiyo_jusanya_ed", "055670", "000386", "first_publication_vs_book", "樋口一葉『十三夜』(文藝倶樂部初出 vs 現代日本文學全集10)"),
    ("ichiyo_otsugomori_ed", "055668", "000388", "first_publication_vs_book", "樋口一葉『大つごもり』(太陽 vs 現代日本文學全集10)"),
    ("ichiyo_yukukumo_ed", "055674", "000392", "first_publication_vs_book", "樋口一葉『ゆく雲』(太陽初出 vs 現代日本文學全集10)"),
    ("ichiyo_utsusemi_ed", "000390", "056007", "cross_edition_variant", "樋口一葉『うつせみ』(現代日本文學全集10 vs 一葉全集2)"),
    ("kishida_hogen_ed", "043581", "044507", "cross_edition_variant", "岸田国士『方言について』(名随筆別巻66 vs 全集22)"),
    ("nakahara_ware_ed", "046579", "055785", "cross_edition_variant", "中原中也『我が生活』(名随筆別巻65 vs 新編全集4)"),
    # 3. Cross-edition variants in 新字新仮名 (different 底本 / editions of same work)
    ("yamamoto_aobeka_ed", "057495", "057574", "single_volume_vs_complete_works", "山本周五郎『青べか物語』(単行本 vs 全集14)"),
    ("miyazawa_ginga_ed", "043737", "000456", "cross_edition_variant", "宮沢賢治『銀河鉄道の夜』(岩波文庫 vs 新潮文庫)"),
    ("okamoto_rogisho_ed", "000447", "058431", "single_volume_vs_complete_works", "岡本かの子『老妓抄』(文庫 vs 全集5)"),
    ("okamoto_paris_ed", "004549", "058435", "single_volume_vs_complete_works", "岡本かの子『巴里のむす子へ』(単行本 vs 全集12)"),
    ("okamoto_michinoku_ed", "000444", "058428", "single_volume_vs_complete_works", "岡本かの子『みちのく』(ちくま vs 全集5)"),
    ("okamoto_yuki_ed", "058429", "043089", "cross_edition_variant", "岡本かの子『雪の日』(全集11 vs 世界紀行文学全集7)"),
    ("okamoto_satsuki_ed", "000436", "058421", "cross_edition_variant", "岡本かの子『五月の朝の花』(随筆1 vs 全集12)"),
    ("hisao_mugetsu_ed", "052224", "046086", "single_volume_vs_complete_works", "久生十蘭『無月物語』(短篇選 vs 全集Ⅱ)"),
    ("sakaguchi_tantei_ed", "045752", "061107", "selected_vs_complete_works", "坂口安吾『私の探偵小説』(選集8 vs 全集06)"),
    ("kikuchi_irefuda_ed", "047858", "000486", "cross_edition_variant", "菊池寛『入れ札』(新潮文庫 vs 短篇と戯曲)"),
    ("kikuchi_tojuro_ed", "000503", "047857", "cross_edition_variant", "菊池寛『藤十郎の恋』(短篇と戯曲 vs 新潮文庫)"),
    ("akutagawa_sennin_ed", "000143", "000144", "cross_edition_variant", "芥川竜之介『仙人』(全集1 vs 全集5)"),
    ("kusaka_saigo_ed", "059900", "004970", "cross_edition_variant", "久坂葉子『幾度目かの最期』(単行本 vs 作品集)"),
    ("ishikawa_yama_ed", "057354", "049593", "cross_edition_variant", "石川欣一『可愛い山』(単行本 vs 山の旅)"),
    ("nakaya_alaska_ed", "053491", "056731", "cross_edition_variant", "中谷宇吉郎『アラスカの氷河』(名随筆33 vs 紀行集)"),
    ("toyoshima_shii_ed", "042458", "045066", "cross_edition_variant", "豊島与志雄『椎の木』(著作集3 vs 児童文学大系16)"),
    ("toyoshima_bungaku_ed", "042536", "042565", "cross_edition_variant", "豊島与志雄『文学以前』二種底本比較"),
]


def _http_get(url: str) -> bytes:
    wait = DELAY_SEC - (time.time() - _LAST_REQ[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    finally:
        _LAST_REQ[0] = time.time()


def load_aozora_index_map() -> Dict[str, Dict[str, str]]:
    with open(INDEX_CSV_PATH, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_id: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if r.get("作品著作権フラグ") != "なし":
            continue
        wid = r["作品ID"]
        if wid not in by_id or r.get("役割フラグ") == "著者":
            by_id[wid] = r
    return by_id


def fetch_or_load_edition_text(row: Dict[str, str], offline: bool = False) -> Optional[str]:
    VARIANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wid = row["作品ID"]
    txt_path = VARIANT_CACHE_DIR / f"{wid}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    if offline:
        return None
    url = row.get("テキストファイルURL", "")
    if not url:
        return None
    try:
        data = _http_get(url)
    except Exception as exc:
        print(f"[warn] failed to download {wid} ({url}): {exc}", file=sys.stderr)
        return None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            return None
        raw = zf.read(txt_names[0]).decode("cp932", errors="replace")
    cleaned = clean_aozora_text(raw)
    txt_path.write_text(cleaned, encoding="utf-8")
    meta_path = VARIANT_CACHE_DIR / f"{wid}.json"
    meta_path.write_text(
        json.dumps(
            {
                "work_id": wid,
                "author": f"{row['姓']}{row['名']}",
                "title": row["作品名"],
                "subtitle": row.get("副題", ""),
                "char_type": row.get("文字遣い種別", ""),
                "base_book": row.get("底本名1", ""),
                "zip_url": url,
                "card_url": row.get("図書カードURL", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return cleaned


def split_aozora_true_paragraphs(text: str) -> List[str]:
    """Split cleaned Aozora text strictly into single-line paragraphs (1 non-empty line = 1 paragraph)."""
    return [ln.strip() for ln in text.replace("\r\n", "\n").splitlines() if ln.strip()]


def _is_non_content_header_or_metadata(text: str) -> bool:
    s = text.strip()
    if len(s) < 50:
        return True
    if s.startswith(("底本：", "入力：", "校正：", "【", "※", "ここまで原稿なし")):
        return True
    return False


def _run_rust_harvest_work_pair(
    nishiki_exe: Path,
    noise_floor_path: Path,
    paras_a: Sequence[str],
    paras_b: Sequence[str],
) -> List[Dict[str, Any]]:
    """Run Rust ``nishiki harvest-diff`` on blank-line-separated true paragraphs (1 line = 1 paragraph)."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        va = td_path / "va.txt"
        vb = td_path / "vb.txt"
        out = td_path / "out.jsonl"
        va.write_text("\n\n".join(paras_a), encoding="utf-8")
        vb.write_text("\n\n".join(paras_b), encoding="utf-8")
        subprocess.run(
            [
                str(nishiki_exe),
                "harvest-diff",
                str(va),
                str(vb),
                "--noise-floor",
                str(noise_floor_path),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _run_rust_harvest_pair(
    nishiki_exe: Path,
    noise_floor_path: Path,
    text_before: str,
    text_after: str,
) -> Optional[Dict[str, Any]]:
    tb_norm = "\n".join(ln for ln in text_before.replace("\r\n", "\n").split("\n") if ln.strip())
    ta_norm = "\n".join(ln for ln in text_after.replace("\r\n", "\n").split("\n") if ln.strip())
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        v1 = td_path / "v1.txt"
        v2 = td_path / "v2.txt"
        out = td_path / "out.jsonl"
        v1.write_text(tb_norm + "\n\nSENTINEL_UNCHANGED_PARAGRAPH", encoding="utf-8")
        v2.write_text(ta_norm + "\n\nSENTINEL_UNCHANGED_PARAGRAPH", encoding="utf-8")
        subprocess.run(
            [
                str(nishiki_exe),
                "harvest-diff",
                str(v1),
                str(v2),
                "--noise-floor",
                str(noise_floor_path),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        return json.loads(lines[0])


def _quantiles(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "iqr": None,
            "max": None,
            "mean": None,
        }
    arr = np.array(vals, dtype=float)
    q25 = float(np.percentile(arr, 25))
    med = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    return {
        "count": int(len(arr)),
        "min": round(float(np.min(arr)), 4),
        "q25": round(q25, 4),
        "median": round(med, 4),
        "q75": round(q75, 4),
        "iqr": round(q75 - q25, 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def _evaluate_and_summarize(
    dataset_rows: List[Dict[str, Any]],
    f15_nf_path: Path,
    nishiki_exe: Path,
) -> Dict[str, Any]:
    nf_data, prov = load_f14_noise_floor(f15_nf_path)
    nf_sha = compute_sha256(f15_nf_path)
    rust_available = nishiki_exe.is_file()

    discrepancies: List[Dict[str, Any]] = []
    per_pair_metrics: List[Dict[str, Any]] = []

    for idx, row in enumerate(dataset_rows):
        tb, ta = row["text_before"], row["text_after"]
        py_ev = evaluate_f14_hybrid_pair(tb, ta, nf_data, provenance=prov, apply_dice_gate=True)

        rust_rec = _run_rust_harvest_pair(nishiki_exe, f15_nf_path, tb, ta) if rust_available else None
        rust_ok = rust_rec is not None and rust_rec.get("kind") == "pair"
        if rust_available:
            if not rust_ok:
                discrepancies.append({"pair_id": row["pair_id"], "reason": "rust_unaligned"})
                continue
            rn = rust_rec["noise_floor_ratio_kasane11"]
            if (
                rust_rec["window_counts"] != py_ev["window_counts"]
                or rn["n_evaluated"] != py_ev["n_evaluated"]
                or rn["band"] != py_ev["band"]
                or rn.get("ac_sigma_source") != py_ev["ac_sigma_source"]
            ):
                discrepancies.append(
                    {
                        "pair_id": row["pair_id"],
                        "reason": "metadata_mismatch",
                        "python": {
                            "window_counts": py_ev["window_counts"],
                            "n_evaluated": py_ev["n_evaluated"],
                            "band": py_ev["band"],
                            "ac_sigma_source": py_ev["ac_sigma_source"],
                        },
                        "rust": {
                            "window_counts": rust_rec["window_counts"],
                            "n_evaluated": rn["n_evaluated"],
                            "band": rn["band"],
                            "ac_sigma_source": rn.get("ac_sigma_source"),
                        },
                    }
                )
            ratios_obj = rn
            per_dim_map = ratios_obj.get("per_dim") or ratios_obj.get("ratios") or {}
            abs_ratios = [abs(float(v)) for v in per_dim_map.values() if v is not None]
            abs_median = float(ratios_obj["median"])
            abs_mean = float(ratios_obj["mean"])
            n_above_1sd = int(ratios_obj["n_above_1"]) if "n_above_1" in ratios_obj else int(sum(1 for r in abs_ratios if r >= 1.0))
            n_above_2sd = int(sum(1 for r in abs_ratios if r >= 2.0))
            n_eval = int(ratios_obj["n_evaluated"])
            per_pair_metrics.append(
                {
                    "pair_id": row["pair_id"],
                    "band": ratios_obj["band"],
                    "l_max": row["l_max"],
                    "l_min": row["l_min"],
                    "window_counts": rust_rec["window_counts"],
                    "mixed_window_pair": bool(row["mixed_window_pair"]),
                    "variant_relationship": row["variant_relationship"],
                    "same_orthography": row["same_orthography"],
                    "n_evaluated": n_eval,
                    "ac_sigma_source": ratios_obj.get("ac_sigma_source"),
                    "abs_ratio_median": abs_median,
                    "abs_ratio_mean": abs_mean,
                    "abs_ratio_max": float(ratios_obj["max"]),
                    "argmax_dim": ratios_obj.get("argmax"),
                    "n_above_1sd": n_above_1sd,
                    "n_above_2sd": n_above_2sd,
                    "frac_above_1sd": n_above_1sd / n_eval,
                    "frac_above_2sd": n_above_2sd / n_eval,
                }
            )
        if (idx + 1) % 50 == 0:
            print(f"  harvest-diff verified {idx + 1}/{len(dataset_rows)} pairs...")

    def _summarize_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_eval_dist: Dict[str, int] = {"8": 0, "10": 0, "11": 0}
        for it in items:
            k = str(it["n_evaluated"])
            n_eval_dist[k] = n_eval_dist.get(k, 0) + 1
        return {
            "n_pairs": len(items),
            "n_evaluated_distribution": n_eval_dist,
            "mixed_window_count": sum(1 for it in items if it["mixed_window_pair"]),
            "ratio_distributions": {
                "per_pair_abs_ratio_median": _quantiles([it["abs_ratio_median"] for it in items]),
                "per_pair_abs_ratio_mean": _quantiles([it["abs_ratio_mean"] for it in items]),
                "per_pair_abs_ratio_max": _quantiles([it["abs_ratio_max"] for it in items]),
                "per_pair_frac_above_1sd": _quantiles([it["frac_above_1sd"] for it in items]),
                "per_pair_frac_above_2sd": _quantiles([it["frac_above_2sd"] for it in items]),
            },
        }

    groups = {
        "overall": per_pair_metrics,
        "band_S": [m for m in per_pair_metrics if m["band"] == "S"],
        "band_M": [m for m in per_pair_metrics if m["band"] == "M"],
        "band_L": [m for m in per_pair_metrics if m["band"] == "L"],
        "mixed_window_pairs": [m for m in per_pair_metrics if m["mixed_window_pair"]],
        "non_mixed_both_single_window": [m for m in per_pair_metrics if max(m["window_counts"]) <= 1],
        "non_mixed_both_multi_window": [m for m in per_pair_metrics if min(m["window_counts"]) >= 2],
        "band_M_n_eval_8": [m for m in per_pair_metrics if m["band"] == "M" and m["n_evaluated"] == 8],
        "band_M_n_eval_11": [m for m in per_pair_metrics if m["band"] == "M" and m["n_evaluated"] == 11],
        "band_L_mixed_window": [m for m in per_pair_metrics if m["band"] == "L" and m["mixed_window_pair"]],
        "band_L_both_multi_window": [m for m in per_pair_metrics if m["band"] == "L" and not m["mixed_window_pair"]],
    }

    provenance_works: Dict[str, int] = {}
    rel_counts: Dict[str, int] = {}
    for row in dataset_rows:
        provenance_works[row["work_group_id"]] = provenance_works.get(row["work_group_id"], 0) + 1
        rel_counts[row["variant_relationship"]] = rel_counts.get(row["variant_relationship"], 0) + 1

    return {
        "dataset_path": str(DATASET_OUT_PATH.relative_to(REPO_ROOT)),
        "f15_noise_floor_path": str(f15_nf_path.relative_to(REPO_ROOT)),
        "f15_noise_floor_sha256": nf_sha,
        "f15_sha256_matches_expected": nf_sha == EXPECTED_F15_SHA256,
        "rust_binary_used": str(nishiki_exe) if rust_available else None,
        "python_rust_parity_discrepancy_count": len(discrepancies),
        "python_rust_parity_discrepancies": discrepancies,
        "target_achievement": {
            "total_pairs": len(per_pair_metrics),
            "band_M_count": len(groups["band_M"]),
            "band_M_target": 30,
            "band_M_target_met": len(groups["band_M"]) >= 30,
            "band_L_count": len(groups["band_L"]),
            "band_L_target": 20,
            "band_L_target_met": len(groups["band_L"]) >= 20,
            "mixed_window_count": len(groups["mixed_window_pairs"]),
            "mixed_window_target": 20,
            "mixed_window_target_met": len(groups["mixed_window_pairs"]) >= 20,
        },
        "provenance_breakdown": {
            "by_work_group": provenance_works,
            "by_variant_relationship": rel_counts,
        },
        "distributions": {k: _summarize_group(v) for k, v in groups.items()},
        "governance_note": (
            "Descriptive distributional measurements only under F-15 noise floor on strictly 1-line-per-paragraph "
            "authentic Aozora Bunko variants. No classification or rating (STYLE_POLISH etc.) claims are made."
        ),
    }


def main(offline: bool = False) -> None:
    assert INDEX_CSV_PATH.exists(), f"Missing Aozora index: {INDEX_CSV_PATH}"
    assert F15_NOISE_FLOOR_PATH.exists(), f"Missing F-15 noise floor: {F15_NOISE_FLOOR_PATH}"
    assert NISHIKI_EXE.is_file(), f"Missing nishiki binary: {NISHIKI_EXE}"
    nf_sha = compute_sha256(F15_NOISE_FLOOR_PATH)
    assert nf_sha == EXPECTED_F15_SHA256, f"F-15 SHA-256 mismatch: {nf_sha} != {EXPECTED_F15_SHA256}"

    index_map = load_aozora_index_map()
    all_extracted: List[Dict[str, Any]] = []
    seen_text_pairs: set[Tuple[str, str]] = set()
    work_pair_yields: List[Dict[str, Any]] = []

    for group_id, wid_a, wid_b, rel, desc in CANDIDATE_WORK_PAIRS:
        row_a = index_map.get(wid_a)
        row_b = index_map.get(wid_b)
        if row_a is None or row_b is None:
            print(f"[skip] {group_id}: missing {wid_a} or {wid_b} in public-domain index")
            continue

        txt_a = fetch_or_load_edition_text(row_a, offline=offline)
        txt_b = fetch_or_load_edition_text(row_b, offline=offline)
        if txt_a is None or txt_b is None:
            print(f"[skip] {group_id}: failed to obtain text")
            continue

        paras_a = split_aozora_true_paragraphs(txt_a)
        paras_b = split_aozora_true_paragraphs(txt_b)
        harvest_recs = _run_rust_harvest_work_pair(NISHIKI_EXE, F15_NOISE_FLOOR_PATH, paras_a, paras_b)

        extracted_for_work = []
        for hrec in harvest_recs:
            if hrec.get("kind") != "pair":
                continue
            pa = hrec["text_before"].strip()
            pb = hrec["text_after"].strip()
            assert "\n" not in pa and "\n" not in pb, "Expected single-line true paragraph"
            if pa == pb:
                continue
            dice_sim = char_bigram_dice(pa, pb)
            if dice_sim < ALIGN_MIN_SIMILARITY or dice_sim >= 1.0:
                continue
            if _is_non_content_header_or_metadata(pa) or _is_non_content_header_or_metadata(pb):
                continue
            l_a = len(pa)
            l_b = len(pb)
            l_min = min(l_a, l_b)
            l_max = max(l_a, l_b)
            if l_min < 50:
                continue
            if (pa, pb) in seen_text_pairs:
                continue
            seen_text_pairs.add((pa, pb))

            w_a = window_count_kasane(pa)
            w_b = window_count_kasane(pb)
            mixed = (w_a >= 2) != (w_b >= 2)
            band = "S" if l_max <= 250 else ("M" if l_max <= 500 else "L")
            bi = int(hrec["before_index"])
            aj = int(hrec["after_index"])

            rec = {
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
            }
            extracted_for_work.append(rec)

        all_extracted.extend(extracted_for_work)
        s_cnt = sum(1 for r in extracted_for_work if r["band"] == "S")
        m_cnt = sum(1 for r in extracted_for_work if r["band"] == "M")
        l_cnt = sum(1 for r in extracted_for_work if r["band"] == "L")
        mix_cnt = sum(1 for r in extracted_for_work if r["mixed_window_pair"])
        work_pair_yields.append(
            {
                "group_id": group_id,
                "desc": desc,
                "rel": rel,
                "same_ortho": row_a.get("文字遣い種別") == row_b.get("文字遣い種別"),
                "total": len(extracted_for_work),
                "S": s_cnt,
                "M": m_cnt,
                "L": l_cnt,
                "mixed": mix_cnt,
            }
        )
        print(f"[work] {group_id}: total={len(extracted_for_work)} (S={s_cnt}, M={m_cnt}, L={l_cnt}, mixed={mix_cnt})")

    # Keep ALL authentic Band M, Band L, and mixed-window single paragraphs without fabrication,
    # plus up to 40 Band S pairs (prioritizing explicit draft/first-publication pairs).
    selected_ml_mix = [r for r in all_extracted if r["band"] in ("M", "L") or r["mixed_window_pair"]]
    band_s_only = [r for r in all_extracted if r["band"] == "S" and not r["mixed_window_pair"]]
    band_s_priority = [
        r for r in band_s_only if r["variant_relationship"] in ("initial_draft_vs_final", "first_publication_vs_book", "draft_vs_revision")
    ]
    band_s_other = [r for r in band_s_only if r not in band_s_priority]
    selected_s = (band_s_priority + band_s_other)[:40]

    final_dataset = selected_ml_mix + selected_s
    DATASET_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_OUT_PATH, "w", encoding="utf-8") as fh:
        for row in final_dataset:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"\nSaved {len(final_dataset)} authentic single-paragraph pairs -> {DATASET_OUT_PATH}\n"
        f"  Band S={sum(1 for r in final_dataset if r['band']=='S')}, "
        f"M={sum(1 for r in final_dataset if r['band']=='M')}, "
        f"L={sum(1 for r in final_dataset if r['band']=='L')}, "
        f"mixed_window={sum(1 for r in final_dataset if r['mixed_window_pair'])}"
    )

    print("\nRunning harvest-diff under F-15 noise floor and verifying Python/Rust parity...")
    summary = _evaluate_and_summarize(final_dataset, F15_NOISE_FLOOR_PATH, NISHIKI_EXE)
    summary["work_pair_yields_all_extracted"] = work_pair_yields
    summary["candidate_pool_totals_before_s_cap"] = {
        "total_extracted": len(all_extracted),
        "S": sum(1 for r in all_extracted if r["band"] == "S"),
        "M": sum(1 for r in all_extracted if r["band"] == "M"),
        "L": sum(1 for r in all_extracted if r["band"] == "L"),
        "mixed_window": sum(1 for r in all_extracted if r["mixed_window_pair"]),
    }

    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Task 4 F-15 harvest summary -> {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build real_revision_pairs_v1 and run F-15 harvest-diff")
    parser.add_argument("--offline", action="store_true", help="Use cached texts only")
    args = parser.parse_args()
    main(offline=args.offline)
