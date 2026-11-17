#!/usr/bin/env python3
"""P-2c Task A1: add authors with >=2 works for work-split R_xx (pre-registered in docs/handoff/STATUS_p2c.md, 登録 A1).

Differences from scripts/p2b/p2b_select.py (Orion decision): existing-manifest authors/works are NOT excluded; only the P-2b adopted
authors (same-person key) and works are excluded. Same-person rule (生年月日+没年月日) is applied to the whole set. Each author's
provenance (which existing manifests list the same person) is recorded. Raw text only in data/aozora_cache/ (gitignored).
"""
from __future__ import annotations
import csv, hashlib, json, random, sys, time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "p2b")); sys.path.insert(0, str(REPO / "python"))
import p2b_select as S  # noqa: E402
from fetch_aozora import _death_year, fetch_clean_text  # noqa: E402
from usir_py.v5 import split_work_into_scenes  # noqa: E402

CACHE = S.CACHE
OUT = REPO / "data" / "aozora_p2c_manifest.json"
SEED = 20261317
TARGET_AUTHORS = 12
MAX_PER_AUTHOR = 3
MIN_WORKS_PER_AUTHOR = 2
MIN_SCENES_PER_AUTHOR = 12
MANIFESTS = ["aozora_manifest", "aozora_heldout_manifest", "aozora_heldout2_manifest", "aozora_heldout3_manifest"]


def main():
    km = S.person_key_map()
    key = lambda a: km.get(a, a)  # noqa: E731
    p2b = json.loads((REPO / "data" / "aozora_p2b_manifest.json").read_text(encoding="utf-8"))
    p2b_aids = {w["author_id"] for w in p2b["works"]}; p2b_wids = {w["work_id"] for w in p2b["works"]}
    p2b_keys = {key(a) for a in p2b_aids}
    src = defaultdict(set)  # person key -> manifests
    for m in MANIFESTS:
        for e in json.loads((REPO / "data" / f"{m}.json").read_text(encoding="utf-8")):
            src[key(e["author_id"].zfill(6))].add(m)
    src_work = defaultdict(set)
    for m in MANIFESTS:
        for e in json.loads((REPO / "data" / f"{m}.json").read_text(encoding="utf-8")):
            src_work[e["work_id"].zfill(6)].add(m)

    by = defaultdict(list)
    with S.INDEX.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("作品著作権フラグ") != "なし" or r.get("文字遣い種別") != "新字新仮名" or r.get("役割フラグ") != "著者": continue
            ndc = r.get("分類番号", "")
            if "K913" in ndc or not S._NDC.search(ndc): continue
            dy = _death_year(r)
            if dy is None or not (S.DEATH_LO <= dy <= S.DEATH_HI): continue
            u = r.get("テキストファイルURL", "").strip()
            if not u.startswith("https://www.aozora.gr.jp/cards/") or not u.endswith(".zip"): continue
            wid, aid = r["作品ID"].zfill(6), r["人物ID"].zfill(6)
            if wid in p2b_wids or key(aid) in p2b_keys: continue
            by[aid].append(r)
    groups = defaultdict(list)
    for aid in by: groups[key(aid)].append(aid)
    pool = {min(ids): [r for i in sorted(ids) for r in by[i]] for ids in groups.values()}
    pool = {a: v for a, v in pool.items() if len(v) >= MIN_WORKS_PER_AUTHOR}
    rng = random.Random(SEED)
    authors = sorted(pool); rng.shuffle(authors)
    for a in authors:
        pool[a].sort(key=lambda r: r["作品ID"]); rng.shuffle(pool[a])
    print(f"candidate authors (>= {MIN_WORKS_PER_AUTHOR} works, P-2b excluded): {len(authors)}", flush=True)

    accepted, rejected, skipped, n_auth = [], [], [], 0
    for aid in authors:
        if n_auth >= TARGET_AUTHORS: break
        got, titles, scenes_total = [], set(), 0
        for row in pool[aid]:
            if len(got) >= MAX_PER_AUTHOR: break
            wid, title = row["作品ID"].zfill(6), row["作品名"]
            base = S._SPLIT_SUFFIX.sub("", title).strip()
            if base in titles:
                skipped.append({"work_id": wid, "author_id": aid, "reason": "split_or_series_duplicate"}); continue
            cp = CACHE / f"p2c_{aid}_{wid}.txt"
            try:
                if cp.is_file(): text = cp.read_text(encoding="utf-8"); zurl = row["テキストファイルURL"]
                else:
                    res = fetch_clean_text(row); text, zurl = res["text"], res["zip_url"]
                    cp.write_text(text, encoding="utf-8"); time.sleep(0.4)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"work_id": wid, "author_id": aid, "reason": f"fetch_error: {type(exc).__name__}"}); continue
            n = len(text)
            paras = [ln for ln in (x.strip() for x in text.split("\n")) if len(ln) >= S.MIN_PARA_CHARS]
            why = "too_short" if n < S.MIN_CHARS else "too_long" if n > S.MAX_CHARS else "insufficient_paragraphs" if len(paras) < S.MIN_PARAS else None
            if why:
                skipped.append({"work_id": wid, "author_id": aid, "reason": why, "char_count": n}); continue
            ns = len(split_work_into_scenes(text))
            got.append({"author_id": aid, "author_ja": f"{row['姓']}{row['名']}", "death_year": _death_year(row), "work_id": wid, "title": title,
                        "ndc": row["分類番号"], "char_type": row["文字遣い種別"], "card_url": row["図書カードURL"], "zip_url": zurl,
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "char_count": n, "n_scenes": ns,
                        "source_manifests": sorted(src.get(key(aid), [])), "work_in_manifests": sorted(src_work.get(wid, []))})
            titles.add(base); scenes_total += ns
        if len(got) >= MIN_WORKS_PER_AUTHOR and scenes_total >= MIN_SCENES_PER_AUTHOR:
            accepted.extend(got); n_auth += 1
            print(f"[author {n_auth:2d}/{TARGET_AUTHORS}] {aid} {got[0]['author_ja']} works={len(got)} scenes={scenes_total} src={got[0]['source_manifests']}", flush=True)
        else:
            rejected.append({"author_id": aid, "author_ja": got[0]["author_ja"] if got else None, "works_ok": len(got), "scenes": scenes_total})
    accepted.sort(key=lambda e: (e["author_id"], e["work_id"]))
    out = {"stage": "P-2c", "seed": SEED, "selection_script": "scripts/p2c/p2c_select.py",
           "criteria": {"death_year": [S.DEATH_LO, S.DEATH_HI], "ndc": "913 (exclude K913)", "char_type": "新字新仮名", "copyright": "なし",
                        "chars": [S.MIN_CHARS, S.MAX_CHARS], "min_paragraphs": S.MIN_PARAS, "max_per_author": MAX_PER_AUTHOR,
                        "min_works_per_author": MIN_WORKS_PER_AUTHOR, "min_scenes_per_author": MIN_SCENES_PER_AUTHOR, "target_authors": TARGET_AUTHORS,
                        "identity_rule": "same 生年月日+没年月日 = same person; P-2b adopted persons excluded; existing-manifest authors allowed (R_xx / detector-v2 evaluation only, NOT Layer C)",
                        "excluded_p2b_authors": len(p2b_aids)},
           "n_authors": n_auth, "n_works": len(accepted), "works": accepted, "rejected_authors": rejected,
           "skipped_count": len(skipped), "skipped": skipped[:120],
           "limitation": "Authors overlap with other benchmarks (see source_manifests); use for R_xx and detector-v2 evaluation only."}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"done: {n_auth} authors, {len(accepted)} works, skipped {len(skipped)}, rejected authors {len(rejected)}")

if __name__ == "__main__":
    main()
