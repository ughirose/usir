#!/usr/bin/env python3
"""Stage B-15 Task 2: 実推敲ペアの拡張候補棚卸しスクリプト（改良版・ローカル _index.csv スキャン）."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV_PATH = REPO_ROOT / "data" / "aozora_cache" / "_index.csv"
V1_JSONL_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1.jsonl"
BUILD_V1_SCRIPT = REPO_ROOT / "scripts" / "build_real_revision_pairs_v1.py"
OUT_JSON_PATH = REPO_ROOT / "scripts" / "b15" / "b15_task2_candidates.json"

REVISION_KEYWORDS = (
    "初出",
    "草稿",
    "第一稿",
    "第二稿",
    "第三稿",
    "第四稿",
    "別稿",
    "初期形",
    "初期",
    "新版",
    "改訂",
    "改稿",
    "定本",
    "異本",
    "異稿",
    "未定稿",
)


def normalize_title(title: str) -> str:
    t = title.strip()
    t = re.sub(r"^[〔\[（(]|[\〕\]）)]$", "", t)
    t = re.sub(r"[〔\[（(][^〕\]）)]*[〕\]）)]", "", t)
    t = re.sub(r"^(新版|定本|改訂版|改訂|新編)\s*", "", t)
    t = re.sub(r"[\s　・：:「」『』]+", "", t)
    return t


def normalize_subtitle(sub: str) -> str:
    s = sub.strip()
    s = re.sub(r"[〔\[（(][^〕\]）)]*[〕\]）)]", "", s)
    s = re.sub(r"[\s　・：:]+", "", s)
    return s


def has_revision_marker(row: Dict[str, str]) -> bool:
    text = f"{row.get('作品名', '')} {row.get('副題', '')}"
    return any(kw in text for kw in REVISION_KEYWORDS)


def extract_v1_candidate_pairs() -> List[Tuple[str, str, str, str, str]]:
    """build_real_revision_pairs_v1.py の CANDIDATE_WORK_PAIRS を読み取る."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import build_real_revision_pairs_v1 as b12
    return list(b12.CANDIDATE_WORK_PAIRS)


def main() -> None:
    v1_used_wids: Set[str] = set()
    v1_groups: Set[str] = set()
    with V1_JSONL_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v1_groups.add(r["work_group_id"])
            prov = r.get("provenance", {})
            if prov.get("edition_a_work_id"):
                v1_used_wids.add(prov["edition_a_work_id"])
            if prov.get("edition_b_work_id"):
                v1_used_wids.add(prov["edition_b_work_id"])

    v1_candidates_all = extract_v1_candidate_pairs()
    v1_unused_candidates = [c for c in v1_candidates_all if c[0] not in v1_groups]

    by_wid: Dict[str, Dict[str, str]] = {}
    with INDEX_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ") != "なし":
                continue
            if not r.get("テキストファイルURL"):
                continue
            wid = r["作品ID"]
            if wid not in by_wid or r.get("役割フラグ") == "著者":
                by_wid[wid] = r

    ndc913_all = sum(1 for r in by_wid.values() if r.get("分類番号", "").startswith("NDC 913"))
    ndc913_shin = sum(
        1 for r in by_wid.values()
        if r.get("分類番号", "").startswith("NDC 913") and r.get("文字遣い種別") == "新字新仮名"
    )
    ndc913_authors = len({
        r["人物ID"] for r in by_wid.values()
        if r.get("分類番号", "").startswith("NDC 913") and r.get("文字遣い種別") == "新字新仮名"
    })

    # Group by (author_id, normalized_title, normalized_subtitle) to avoid matching episodes of series
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for wid, r in by_wid.items():
        norm_t = normalize_title(r.get("作品名", ""))
        norm_s = normalize_subtitle(r.get("副題", ""))
        if len(norm_t) < 2:
            continue
        grouped[(r.get("人物ID", ""), norm_t, norm_s)].append(r)

    explicit_rev_same_ortho: List[Dict[str, Any]] = []
    explicit_rev_diff_ortho: List[Dict[str, Any]] = []
    cross_edition_same_ortho: List[Dict[str, Any]] = []

    for (aid, norm_t, norm_s), items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        wids = {it["作品ID"] for it in items}
        if wids & v1_used_wids:
            continue

        author = f"{items[0]['姓']}{items[0]['名']}"
        ndc = items[0].get("分類番号", "")
        if not ndc.startswith("NDC 91"):
            continue

        any_rev = any(has_revision_marker(it) for it in items)
        ortho_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for it in items:
            ortho_groups[it.get("文字遣い種別", "")].append(it)

        editions = [
            {
                "work_id": it["作品ID"],
                "title": it["作品名"],
                "subtitle": it.get("副題", ""),
                "char_type": it.get("文字遣い種別", ""),
                "ndc": it.get("分類番号", ""),
                "base_book": it.get("底本名1", ""),
                "first_pub": it.get("初出", ""),
                "card_url": it.get("図書カードURL", ""),
                "zip_url": it.get("テキストファイルURL", ""),
            }
            for it in sorted(items, key=lambda x: x["作品ID"])
        ]
        entry = {
            "author_id": aid,
            "author": author,
            "norm_title": norm_t,
            "norm_subtitle": norm_s,
            "ndc": ndc,
            "n_editions": len(items),
            "char_types": sorted(ortho_groups.keys()),
            "has_revision_marker": any_rev,
            "editions": editions,
        }

        if any_rev:
            if any(len(v) >= 2 for v in ortho_groups.values()):
                explicit_rev_same_ortho.append(entry)
            else:
                explicit_rev_diff_ortho.append(entry)
        else:
            # Exclude obvious volume splits like 上巻/下巻 or 前編/後編 in base_book
            for ct, ct_items in ortho_groups.items():
                if len(ct_items) >= 2:
                    bbs = [x.get("底本名1", "") for x in ct_items]
                    if len(set(bbs)) >= 2 and not any(
                        any(w in bb for w in ("上巻", "中巻", "下巻", "前編", "後編", "上", "下"))
                        and any(w2 in bb2 for bb2 in bbs for w2 in ("上巻", "中巻", "下巻", "前編", "後編") if bb2 != bb)
                        for bb in bbs
                    ):
                        cross_edition_same_ortho.append(entry)
                        break

    out = {
        "aozora_index_stats": {
            "total_copyright_free_works": len(by_wid),
            "ndc913_all": ndc913_all,
            "ndc913_shinjishinkana": ndc913_shin,
            "ndc913_shinjishinkana_authors": ndc913_authors,
        },
        "v1_built_groups": len(v1_groups),
        "v1_unused_from_b12_list": [
            {"work_group_id": c[0], "ed_a": c[1], "ed_b": c[2], "rel": c[3], "notes": c[4]}
            for c in v1_unused_candidates
        ],
        "explicit_rev_same_ortho": explicit_rev_same_ortho,
        "explicit_rev_diff_ortho": explicit_rev_diff_ortho,
        "cross_edition_same_ortho_count": len(cross_edition_same_ortho),
        "cross_edition_same_ortho": cross_edition_same_ortho,
    }
    OUT_JSON_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Aozora NDC913 新字新仮名: {ndc913_shin} works / {ndc913_authors} authors")
    print(f"v1 unused from B-12 list: {len(v1_unused_candidates)}")
    for c in v1_unused_candidates:
        print(f"  [unused] {c[0]}: {c[4]} ({c[1]} vs {c[2]})")
    print(f"New explicit_rev_same_ortho: {len(explicit_rev_same_ortho)}")
    for e in explicit_rev_same_ortho:
        print(f"  [rev-same] {e['author']}『{e['norm_title']}』: " + " vs ".join(f"{ed['work_id']}:{ed['title']}({ed['char_type']})" for ed in e['editions']))
    print(f"New explicit_rev_diff_ortho: {len(explicit_rev_diff_ortho)}")
    for e in explicit_rev_diff_ortho:
        print(f"  [rev-diff] {e['author']}『{e['norm_title']}』: " + " vs ".join(f"{ed['work_id']}:{ed['title']}({ed['char_type']})" for ed in e['editions']))
    print(f"New cross_edition_same_ortho (NDC 91x): {len(cross_edition_same_ortho)}")
    for e in cross_edition_same_ortho[:40]:
        print(f"  [cross] {e['author']}『{e['norm_title']}』[{e['ndc']}]: " + " vs ".join(f"{ed['work_id']}:{ed['base_book']}({ed['char_type']})" for ed in e['editions']))


if __name__ == "__main__":
    main()
