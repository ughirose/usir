#!/usr/bin/env python3
"""Stage B-17 Task 1: 青空文庫 NDC 913 新字新仮名から新規著者中心に 110 作品を選定・取得し、
data/aozora_heldout3_manifest.json および選定リストを構築する.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fetch_aozora import _death_year, fetch_clean_text  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "aozora_cache"
INDEX_CSV = CACHE_DIR / "_index.csv"
OUT_MANIFEST = REPO_ROOT / "data" / "aozora_heldout3_manifest.json"
OUT_SELECTION = REPO_ROOT / "scripts" / "b17" / "b17_task1_selection_list.json"

TARGET_COUNT = 110  # 106 + 110 = 216
SEED = 20261114
MIN_CHAR_COUNT = 6000
MAX_CHAR_COUNT = 200000
MIN_PARAGRAPHS = 30
MIN_PAIRS_TOTAL = 25
MIN_PARAGRAPH_CHARS = 30
MAX_PER_AUTHOR = 2

_NDC_913_RE = re.compile(r"(?:^|\s)(?:NDC\s*)?913(?:\.\d+)?(?:\s|$)")
_SPLIT_SUFFIX_RE = re.compile(r"[（(](?:上|中|下|前|後|続|[一二三四五六七八九十0-9]+)[）)]$")


def split_paragraphs_py(text: str) -> List[str]:
    return [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= MIN_PARAGRAPH_CHARS]


def load_existing() -> tuple[set[str], set[str]]:
    wids, aids = set(), set()
    for m in ["aozora_manifest.json", "aozora_heldout_manifest.json", "aozora_heldout2_manifest.json"]:
        arr = json.loads((REPO_ROOT / "data" / m).read_text(encoding="utf-8"))
        for e in arr:
            if m == "aozora_manifest.json" and e.get("genre") != "novel":
                continue
            wids.add(e["work_id"].zfill(6))
            aids.add(e["author_id"].zfill(6))
    return wids, aids


def build_candidate_pool(existing_wids: set[str], existing_aids: set[str]) -> Dict[str, List[Dict[str, str]]]:
    by_author: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_wids = set()
    with INDEX_CSV.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ") != "なし":
                continue
            if r.get("文字遣い種別") != "新字新仮名":
                continue
            if r.get("役割フラグ") != "著者":
                continue
            ndc = r.get("分類番号", "")
            # K913 (児童文学) を除外し、913 (日本小説) のみを対象とする
            if "K913" in ndc or not _NDC_913_RE.search(ndc):
                continue
            dy = _death_year(r)
            if dy is None or dy < 1868 or dy > 1967:
                continue
            txt_url = r.get("テキストファイルURL", "").strip()
            if not txt_url.startswith("https://www.aozora.gr.jp/cards/") or not txt_url.endswith(".zip"):
                continue
            wid = r["作品ID"].zfill(6)
            aid = r["人物ID"].zfill(6)
            if wid in existing_wids or wid in seen_wids:
                continue
            if aid in existing_aids:
                continue
            seen_wids.add(wid)
            by_author[aid].append(r)
    return by_author


def main() -> None:
    existing_wids, existing_aids = load_existing()
    pool = build_candidate_pool(existing_wids, existing_aids)
    rng = random.Random(SEED)

    authors = sorted(pool.keys())
    rng.shuffle(authors)
    for a in authors:
        pool[a].sort(key=lambda r: r["作品ID"])
        rng.shuffle(pool[a])

    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    author_selected_count: Dict[str, int] = defaultdict(int)
    author_base_titles: Dict[str, set[str]] = defaultdict(set)

    # ラウンドロビンで1著者あたり最大 MAX_PER_AUTHOR 作品を選定（著者多様性を最大化）
    for round_idx in range(MAX_PER_AUTHOR):
        if len(selected) >= TARGET_COUNT:
            break
        for aid in authors:
            if len(selected) >= TARGET_COUNT:
                break
            if author_selected_count[aid] > round_idx:
                continue
            # この著者から1作品決まるまで候補を試す
            while pool[aid]:
                row = pool[aid].pop(0)
                wid = row["作品ID"].zfill(6)
                title = row["作品名"]
                base_title = _SPLIT_SUFFIX_RE.sub("", title).strip()
                if base_title in author_base_titles[aid]:
                    skipped.append({"work_id": wid, "author_id": aid, "title": title, "reason": "split_or_series_duplicate"})
                    continue

                cache_path = CACHE_DIR / f"heldout3_{aid}_{wid}.txt"
                try:
                    if cache_path.is_file():
                        text = cache_path.read_text(encoding="utf-8")
                        zip_url = row["テキストファイルURL"]
                    else:
                        res = fetch_clean_text(row)
                        text = res["text"]
                        zip_url = res["zip_url"]
                        cache_path.write_text(text, encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    skipped.append({"work_id": wid, "author_id": aid, "title": title, "reason": f"fetch_error: {exc}"})
                    continue

                char_count = len(text)
                paras = split_paragraphs_py(text)
                n_paras = len(paras)
                n_pairs = max(0, n_paras - 1)
                if char_count < MIN_CHAR_COUNT:
                    cache_path.unlink(missing_ok=True)
                    skipped.append({"work_id": wid, "author_id": aid, "title": title, "char_count": char_count, "reason": f"too_short_chars<{MIN_CHAR_COUNT}"})
                    continue
                if char_count > MAX_CHAR_COUNT:
                    cache_path.unlink(missing_ok=True)
                    skipped.append({"work_id": wid, "author_id": aid, "title": title, "char_count": char_count, "reason": f"too_long_chars>{MAX_CHAR_COUNT}"})
                    continue
                if n_paras < MIN_PARAGRAPHS or n_pairs < MIN_PAIRS_TOTAL:
                    cache_path.unlink(missing_ok=True)
                    skipped.append({"work_id": wid, "author_id": aid, "title": title, "n_paragraphs": n_paras, "n_pairs": n_pairs, "reason": "insufficient_paragraphs"})
                    continue

                sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                n_windows = max(1, (char_count - 125) // 125)
                author_ja = f"{row['姓']}{row['名']}"
                entry = {
                    "author_id": aid,
                    "author_ja": author_ja,
                    "title": title,
                    "work_id": wid,
                    "char_type": row["文字遣い種別"],
                    "ndc": row["分類番号"],
                    "card_url": row["図書カードURL"],
                    "zip_url": zip_url,
                    "sha256": sha,
                    "char_count": char_count,
                    "n_windows": n_windows,
                    "genre": "heldout3",
                }
                selected.append(entry)
                author_selected_count[aid] += 1
                author_base_titles[aid].add(base_title)
                print(
                    f"[{len(selected):3d}/{TARGET_COUNT}] {aid} {author_ja} 『{title}』 "
                    f"(id={wid}, chars={char_count}, paras={n_paras}, pairs={n_pairs})",
                    flush=True,
                )
                break

    if len(selected) < TARGET_COUNT:
        raise RuntimeError(f"Only selected {len(selected)} < {TARGET_COUNT} works!")

    # author_id, work_id 順にソートしてマニフェスト保存
    selected_sorted = sorted(selected, key=lambda e: (e["author_id"], e["work_id"]))
    OUT_MANIFEST.write_text(json.dumps(selected_sorted, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "target_count": TARGET_COUNT,
        "selected_count": len(selected_sorted),
        "unique_new_authors": sum(1 for v in author_selected_count.values() if v > 0),
        "seed": SEED,
        "criteria": {
            "ndc": "913 (exclude K913)",
            "char_type": "新字新仮名",
            "copyright": "なし",
            "death_year": "1868..=1967",
            "char_count": [MIN_CHAR_COUNT, MAX_CHAR_COUNT],
            "min_paragraphs": MIN_PARAGRAPHS,
            "min_pairs_total": MIN_PAIRS_TOTAL,
            "max_per_author": MAX_PER_AUTHOR,
            "exclude_existing_10_authors": True,
        },
        "selected": [
            {
                "work_id": e["work_id"],
                "author_id": e["author_id"],
                "author_ja": e["author_ja"],
                "title": e["title"],
                "ndc": e["ndc"],
                "char_count": e["char_count"],
                "reason": "新規著者優先・NDC913新字新仮名・長さ基準(chars>=6000, pairs>=25)充足",
            }
            for e in selected_sorted
        ],
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
    }
    OUT_SELECTION.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Done! Selected {len(selected_sorted)} works from {len(author_selected_count)} new authors. Skipped {len(skipped)}.")


if __name__ == "__main__":
    main()
