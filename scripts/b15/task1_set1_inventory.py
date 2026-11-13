#!/usr/bin/env python3
"""Stage B-15 Task 1: Set 1 拡張の在庫棚卸しスクリプト（読み取り専用分析）.

106作品全場面ファイル（data/v5/aozora_106works_scenes.jsonl）と
Set 1（aozora_manifest.json の novel 46作品）を突き合わせ、
Set 1 外の60作品（および aozora_cache 内のその他キャッシュ作品）について、
段落分割・ウィンドウ数・帯別隣接段落ペア数を作品単位で集計する。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.v5.f14_hybrid_noise_floor import window_count_kasane  # noqa: E402

SCENES_106_PATH = REPO_ROOT / "data" / "v5" / "aozora_106works_scenes.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "aozora_cache"
INDEX_CSV_PATH = CACHE_DIR / "_index.csv"
MANIFEST_PATHS = [
    REPO_ROOT / "data" / "aozora_manifest.json",
    REPO_ROOT / "data" / "aozora_heldout_manifest.json",
    REPO_ROOT / "data" / "aozora_heldout2_manifest.json",
]
OUT_JSON_PATH = REPO_ROOT / "scripts" / "b15" / "b15_task1_inventory.json"

MIN_PARAGRAPH_CHARS = 30
BAND_S_MAX = 250
BAND_M_MAX = 500


def split_paragraphs_py(text: str) -> List[str]:
    """Rust `paragraph_noise::split_paragraphs_py` と同一の段落抽出."""
    return [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= MIN_PARAGRAPH_CHARS]


def classify_band(l_max: int) -> str:
    if l_max <= BAND_S_MAX:
        return "S"
    if l_max <= BAND_M_MAX:
        return "M"
    return "L"


def classify_era(birth_str: str, death_str: str) -> str:
    """生没年から簡易時代区分を返す."""
    try:
        birth_y = int(birth_str[:4]) if birth_str else None
        death_y = int(death_str[:4]) if death_str else None
    except ValueError:
        return "不明"
    if death_y is not None:
        if death_y <= 1912:
            return "明治期没(〜1912)"
        if death_y <= 1926:
            return "大正期没(1913-1926)"
        if death_y <= 1945:
            return "昭和戦前期没(1927-1945)"
        return "昭和戦後期没(1946〜)"
    if birth_y is not None:
        return f"生年{birth_y}"
    return "不明"


def load_index_csv() -> Dict[str, Dict[str, str]]:
    by_wid: Dict[str, Dict[str, str]] = {}
    with INDEX_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            wid = row["作品ID"]
            if wid not in by_wid or row.get("役割フラグ") == "著者":
                by_wid[wid] = row
    return by_wid


def analyze_text_file(txt_path: Path) -> Dict[str, Any]:
    text = txt_path.read_text(encoding="utf-8")
    paras = split_paragraphs_py(text)
    lens = [len(p) for p in paras]
    wins = [window_count_kasane(p) for p in paras]

    counts = {
        "n_paragraphs": len(paras),
        "n_pairs_total": max(0, len(paras) - 1),
        "S_all": 0,
        "M_all": 0,
        "L_all": 0,
        "S_both_single": 0,
        "S_mixed": 0,
        "S_both_multi": 0,
        "M_both_single": 0,
        "M_mixed": 0,
        "M_both_multi": 0,
        "L_both_single": 0,
        "L_mixed": 0,
        "L_both_multi": 0,
    }
    for i in range(len(paras) - 1):
        l_max = max(lens[i], lens[i + 1])
        band = classify_band(l_max)
        w0, w1 = wins[i], wins[i + 1]
        if w0 >= 2 and w1 >= 2:
            wstate = "both_multi"
        elif w0 >= 2 or w1 >= 2:
            wstate = "mixed"
        else:
            wstate = "both_single"
        counts[f"{band}_all"] += 1
        counts[f"{band}_{wstate}"] += 1
    return counts


def main() -> None:
    index_map = load_index_csv()

    # Load manifests
    manifest_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for mp in MANIFEST_PATHS:
        for item in json.loads(mp.read_text(encoding="utf-8")):
            key = (item["author_id"], item["work_id"])
            manifest_info[key] = {**item, "manifest_file": mp.name}

    # Load 106 works from aozora_106works_scenes.jsonl
    works_106_keys: List[Tuple[str, str]] = []
    seen = set()
    with SCENES_106_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = (r["author_id"], r["work_id"])
            if k not in seen:
                seen.add(k)
                works_106_keys.append(k)

    # Map all cached .txt files
    cache_by_key: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for p in sorted(CACHE_DIR.glob("*.txt")):
        parts = p.stem.split("_")
        if len(parts) >= 3:
            key = (parts[1], parts[2])
            cache_by_key[key].append(p)

    set1_records: List[Dict[str, Any]] = []
    non_set1_106_records: List[Dict[str, Any]] = []
    extra_cache_records: List[Dict[str, Any]] = []

    processed_keys = set()

    for aid, wid in works_106_keys:
        processed_keys.add((aid, wid))
        m = manifest_info.get((aid, wid), {})
        idx_r = index_map.get(wid, {})
        paths = cache_by_key.get((aid, wid), [])
        # Prefer novel_ > heldout_ > article_
        txt_path = None
        for prefix in ("novel_", "heldout_", "article_"):
            for p in paths:
                if p.name.startswith(prefix):
                    txt_path = p
                    break
            if txt_path is not None:
                break
        if txt_path is None and paths:
            txt_path = paths[0]

        stats = analyze_text_file(txt_path) if txt_path else {}
        is_set1 = bool(txt_path and txt_path.name.startswith("novel_") and m.get("genre") == "novel")

        rec = {
            "author_id": aid,
            "work_id": wid,
            "author_ja": m.get("author_ja") or f"{idx_r.get('姓', '')}{idx_r.get('名', '')}",
            "title": m.get("title") or idx_r.get("作品名", ""),
            "genre": m.get("genre", ""),
            "manifest_file": m.get("manifest_file", ""),
            "cache_file": txt_path.name if txt_path else "",
            "char_type": m.get("char_type") or idx_r.get("文字遣い種別", ""),
            "birth": idx_r.get("生年月日", ""),
            "death": idx_r.get("没年月日", ""),
            "era": classify_era(idx_r.get("生年月日", ""), idx_r.get("没年月日", "")),
            "in_set1": is_set1,
            **stats,
        }
        if is_set1:
            set1_records.append(rec)
        else:
            non_set1_106_records.append(rec)

    # Extra cache works not in 106works
    for (aid, wid), paths in sorted(cache_by_key.items()):
        if (aid, wid) in processed_keys:
            continue
        txt_path = paths[0]
        m = manifest_info.get((aid, wid), {})
        idx_r = index_map.get(wid, {})
        stats = analyze_text_file(txt_path)
        extra_cache_records.append(
            {
                "author_id": aid,
                "work_id": wid,
                "author_ja": m.get("author_ja") or f"{idx_r.get('姓', '')}{idx_r.get('名', '')}",
                "title": m.get("title") or idx_r.get("作品名", ""),
                "genre": m.get("genre", ""),
                "manifest_file": m.get("manifest_file", ""),
                "cache_file": txt_path.name,
                "char_type": m.get("char_type") or idx_r.get("文字遣い種別", ""),
                "birth": idx_r.get("生年月日", ""),
                "death": idx_r.get("没年月日", ""),
                "era": classify_era(idx_r.get("生年月日", ""), idx_r.get("没年月日", "")),
                "in_set1": False,
                **stats,
            }
        )

    def summarize(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n_works": len(recs),
            "n_paragraphs": sum(r["n_paragraphs"] for r in recs),
            "n_pairs_total": sum(r["n_pairs_total"] for r in recs),
            "M_all": sum(r["M_all"] for r in recs),
            "L_all": sum(r["L_all"] for r in recs),
            "S_all": sum(r["S_all"] for r in recs),
            "M_both_multi_pairs": sum(r["M_both_multi"] for r in recs),
            "M_both_multi_works": sum(1 for r in recs if r["M_both_multi"] > 0),
            "L_both_multi_pairs": sum(r["L_both_multi"] for r in recs),
            "L_both_multi_works": sum(1 for r in recs if r["L_both_multi"] > 0),
            "M_mixed_pairs": sum(r["M_mixed"] for r in recs),
            "L_mixed_pairs": sum(r["L_mixed"] for r in recs),
            "authors": dict(Counter(r["author_ja"] for r in recs)),
            "eras": dict(Counter(r["era"] for r in recs)),
            "char_types": dict(Counter(r["char_type"] for r in recs)),
            "genres": dict(Counter(r["genre"] for r in recs)),
        }

    s1_sum = summarize(set1_records)
    non_s1_sum = summarize(non_set1_106_records)
    combined_106_sum = summarize(set1_records + non_set1_106_records)
    extra_sum = summarize(extra_cache_records)
    all_cached_sum = summarize(set1_records + non_set1_106_records + extra_cache_records)

    out = {
        "verification_set1": s1_sum,
        "non_set1_106_summary": non_s1_sum,
        "combined_106_summary": combined_106_sum,
        "extra_cache_summary": extra_sum,
        "all_cached_summary": all_cached_sum,
        "set1_works": set1_records,
        "non_set1_106_works": non_set1_106_records,
        "extra_cache_works": extra_cache_records,
    }
    OUT_JSON_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Set 1 Verification (expected: 46 works, 10859 pairs, M_multi=165/22, L_multi=225/24) ===")
    print(json.dumps(s1_sum, ensure_ascii=False, indent=2))
    print("=== Non-Set1 in 106works (60 works) ===")
    print(json.dumps(non_s1_sum, ensure_ascii=False, indent=2))
    print("=== Combined 106works ===")
    print(json.dumps(combined_106_sum, ensure_ascii=False, indent=2))
    print("=== Extra cache (not in 106works) ===")
    print(json.dumps(extra_sum, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
