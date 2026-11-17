#!/usr/bin/env python3
"""P-2b Task A' step 1: 青空文庫の戦後作品（没年 1946-1955, NDC913, 新字新仮名）を決定論的に選定・取得する。

- 原文は data/aozora_cache/ (gitignore 対象) にのみ保存し、リポジトリには出所情報のみ
  (data/aozora_p2b_manifest.json: 作品ID・著者・題名・URL・取得テキストの SHA-256・文字数) を書く。
- B-17 (scripts/b17/task1_select_heldout3.py) の方式に倣う: seed 固定のシャッフル + 著者単位ラウンド。
- 既存 15 作家（4 マニフェスト）の著者・作品は除外する（Layer C を atlas から独立させるため）。
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))
from fetch_aozora import _death_year, fetch_clean_text  # noqa: E402
from usir_py.v5 import split_work_into_scenes  # noqa: E402

CACHE = REPO / "data" / "aozora_cache"
INDEX = CACHE / "_index.csv"
OUT = REPO / "data" / "aozora_p2b_manifest.json"
SEED = 20261117
TARGET_AUTHORS = 12
MAX_PER_AUTHOR = 3
MIN_WORKS_PER_AUTHOR = 1
MIN_SCENES_PER_AUTHOR = 12
MIN_CHARS, MAX_CHARS, MIN_PARAS, MIN_PARA_CHARS = 6000, 200000, 30, 30
DEATH_LO, DEATH_HI = 1946, 1955
_NDC = re.compile(r"(?:^|\s)(?:NDC\s*)?913(?:\.\d+)?(?:\s|$)")
_SPLIT_SUFFIX = re.compile(r"[（(](?:上|中|下|前|後|続|[一二三四五六七八九十0-9]+)[）)]$")


def existing_ids():
    wids, aids = set(), set()
    for m in ["aozora_manifest", "aozora_heldout_manifest", "aozora_heldout2_manifest", "aozora_heldout3_manifest"]:
        for e in json.loads((REPO / "data" / f"{m}.json").read_text(encoding="utf-8")):
            wids.add(e["work_id"].zfill(6)); aids.add(e["author_id"].zfill(6))
    return wids, aids


def candidate_pool(ex_w, ex_a):
    by = defaultdict(list)
    with INDEX.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ") != "なし" or r.get("文字遣い種別") != "新字新仮名" or r.get("役割フラグ") != "著者":
                continue
            ndc = r.get("分類番号", "")
            if "K913" in ndc or not _NDC.search(ndc):
                continue
            dy = _death_year(r)
            if dy is None or not (DEATH_LO <= dy <= DEATH_HI):
                continue
            u = r.get("テキストファイルURL", "").strip()
            if not u.startswith("https://www.aozora.gr.jp/cards/") or not u.endswith(".zip"):
                continue
            wid, aid = r["作品ID"].zfill(6), r["人物ID"].zfill(6)
            if wid in ex_w or aid in ex_a:
                continue
            by[aid].append(r)
    return by


def main():
    ex_w, ex_a = existing_ids()
    pool = candidate_pool(ex_w, ex_a)
    pool = {a: v for a, v in pool.items() if len(v) >= MIN_WORKS_PER_AUTHOR}
    rng = random.Random(SEED)
    authors = sorted(pool)
    rng.shuffle(authors)
    for a in authors:
        pool[a].sort(key=lambda r: r["作品ID"])
        rng.shuffle(pool[a])
    print(f"candidate authors (>= {MIN_WORKS_PER_AUTHOR} works): {len(authors)}; excluded existing authors: {len(ex_a)}", flush=True)

    accepted: List[Dict[str, Any]] = []
    rejected_authors: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    n_auth = 0
    for aid in authors:
        if n_auth >= TARGET_AUTHORS:
            break
        got, titles, scenes_total = [], set(), 0
        for row in pool[aid]:
            if len(got) >= MAX_PER_AUTHOR:
                break
            wid, title = row["作品ID"].zfill(6), row["作品名"]
            base = _SPLIT_SUFFIX.sub("", title).strip()
            if base in titles:
                skipped.append({"work_id": wid, "author_id": aid, "reason": "split_or_series_duplicate"}); continue
            cp = CACHE / f"p2b_{aid}_{wid}.txt"
            try:
                if cp.is_file():
                    text = cp.read_text(encoding="utf-8"); zurl = row["テキストファイルURL"]
                else:
                    res = fetch_clean_text(row); text, zurl = res["text"], res["zip_url"]
                    cp.write_text(text, encoding="utf-8"); time.sleep(0.4)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"work_id": wid, "author_id": aid, "reason": f"fetch_error: {type(exc).__name__}"}); continue
            n = len(text)
            paras = [ln for ln in (x.strip() for x in text.split("\n")) if len(ln) >= MIN_PARA_CHARS]
            why = None
            if n < MIN_CHARS: why = "too_short"
            elif n > MAX_CHARS: why = "too_long"
            elif len(paras) < MIN_PARAS: why = "insufficient_paragraphs"
            if why:
                skipped.append({"work_id": wid, "author_id": aid, "reason": why, "char_count": n}); continue
            ns = len(split_work_into_scenes(text))
            got.append({
                "author_id": aid, "author_ja": f"{row['姓']}{row['名']}", "death_year": _death_year(row),
                "work_id": wid, "title": title, "ndc": row["分類番号"], "char_type": row["文字遣い種別"],
                "card_url": row["図書カードURL"], "zip_url": zurl,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "char_count": n, "n_scenes": ns,
            })
            titles.add(base); scenes_total += ns
        if len(got) >= MIN_WORKS_PER_AUTHOR and scenes_total >= MIN_SCENES_PER_AUTHOR:
            accepted.extend(got); n_auth += 1
            print(f"[author {n_auth:2d}/{TARGET_AUTHORS}] {aid} {got[0]['author_ja']} works={len(got)} scenes={scenes_total}", flush=True)
        else:
            rejected_authors.append({"author_id": aid, "works_ok": len(got), "scenes": scenes_total})
            for g in got:  # 不採用著者のキャッシュは残すが manifest には載せない
                pass
    if n_auth < TARGET_AUTHORS:
        print(f"WARNING: only {n_auth} authors accepted (target {TARGET_AUTHORS})")
    accepted.sort(key=lambda e: (e["author_id"], e["work_id"]))
    out = {
        "stage": "P-2b", "seed": SEED, "selection_script": "scripts/p2b/p2b_select.py",
        "criteria": {"death_year": [DEATH_LO, DEATH_HI], "ndc": "913 (exclude K913)", "char_type": "新字新仮名", "copyright": "なし",
                     "chars": [MIN_CHARS, MAX_CHARS], "min_paragraphs": MIN_PARAS, "max_per_author": MAX_PER_AUTHOR,
                     "min_works_per_author": MIN_WORKS_PER_AUTHOR, "min_scenes_per_author": MIN_SCENES_PER_AUTHOR,
                     "target_authors": TARGET_AUTHORS, "excluded_existing_authors": len(ex_a)},
        "n_authors": n_auth, "n_works": len(accepted), "works": accepted,
        "rejected_authors": rejected_authors, "skipped_count": len(skipped), "skipped": skipped[:80],
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"done: {n_auth} authors, {len(accepted)} works, skipped {len(skipped)}")


if __name__ == "__main__":
    main()
