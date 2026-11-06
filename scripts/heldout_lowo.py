#!/usr/bin/env python
"""Phase 38-2: author identification on held-out works (not used to design the Phase 36 feature set).

Selection rule (fixed in docs/handoff/2026-11-06_phase38_preregistration.md before any held-out text was read):
for each atlas author, scan the official Aozora index in ascending work id and keep works with
  person id match, role 著者, work copyright flag なし, NDC containing 913, character type 新字新仮名,
  title (without subtitle) neither equal to, containing, nor contained in an existing manifest title,
  cleaned text >= 3000 characters and >= 5 windows of 600 characters;
the first 3 such works per author are used.  Texts are cached in data/aozora_cache/ (git-ignored); only ids, URLs
and SHA256 are written to data/aozora_heldout_manifest.json.

Evaluation: Fisher LDA trained on all manifest novel works (primary Phase 36 variant z24 + ortho_ext + narrative,
reg 1e-2; Phase 30 baseline z24 + ortho_ext reported alongside); each held-out work is assigned to the author with
the highest summed window score (same scoring as the LOWO scripts).

Output: docs/handoff/v10/p38_heldout_lowo.json
Usage:  cd python && uv run python ../scripts/heldout_lowo.py [--offline]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_aozora_style_atlas import featurize_all  # noqa: E402
from fetch_aozora import CACHE_DIR, DELAY_SEC, fetch_clean_text, load_index, load_manifest_texts  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.orthographic_prior import ortho_matrix_ext, ortho_matrix_narr  # noqa: E402

OUT = ROOT / "docs" / "handoff" / "v10" / "p38_heldout_lowo.json"
HELDOUT_MANIFEST = ROOT / "data" / "aozora_heldout_manifest.json"
REG = 1e-2
PER_AUTHOR = 3
MIN_CHARS = 3000
MIN_WINDOWS = 5


def title_clash(title: str, existing: list) -> bool:
    return any(title == t or title in t or t in title for t in existing)


def select_heldout(entries: list, offline: bool) -> list:
    index = load_index()
    existing = {}
    for e in entries:
        existing.setdefault(e["author_id"], []).append(e["title"])
    authors = sorted(existing)
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    chosen = []
    for pid in authors:
        rows = [r for r in index if r["人物ID"] == pid and r["役割フラグ"] == "著者" and r["作品著作権フラグ"] == "なし"
                and "913" in r["分類番号"] and r["文字遣い種別"] == "新字新仮名"
                and r["テキストファイルURL"].startswith("https://www.aozora.gr.jp/cards/")
                and not title_clash(r["作品名"], existing[pid])]
        rows.sort(key=lambda r: int(r["作品ID"]))
        seen_titles, got = set(), 0
        for r in rows:
            if got >= PER_AUTHOR:
                break
            if r["作品名"] in seen_titles:
                continue
            seen_titles.add(r["作品名"])
            cpath = CACHE_DIR / f"heldout_{pid}_{r['作品ID']}.txt"
            meta = cpath.with_suffix(".json")
            if cpath.exists() and meta.exists():
                text, zip_url = cpath.read_text(encoding="utf-8"), json.loads(meta.read_text(encoding="utf-8"))["zip_url"]
            elif offline:
                continue
            else:
                try:
                    f = fetch_clean_text(r)
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {pid} {r['作品ID']}: {exc}", file=sys.stderr)
                    continue
                time.sleep(DELAY_SEC)
                text, zip_url = f["text"], f["zip_url"]
                cpath.write_text(text, encoding="utf-8")
                meta.write_text(json.dumps({"zip_url": zip_url}), encoding="utf-8")
            n_win = len(sa.make_windows(text))
            if len(text) < MIN_CHARS or n_win < MIN_WINDOWS:
                print(f"[short] {labels[pid]} 「{r['作品名']}」 chars={len(text)} windows={n_win}")
                continue
            chosen.append({"author_id": pid, "author_ja": labels[pid], "title": r["作品名"], "work_id": r["作品ID"],
                           "char_type": r["文字遣い種別"], "ndc": r["分類番号"], "card_url": r["図書カードURL"],
                           "zip_url": zip_url, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                           "char_count": len(text), "n_windows": n_win, "genre": "heldout", "text": text})
            got += 1
            print(f"[ok] {labels[pid]} 「{r['作品名']}」 id={r['作品ID']} chars={len(text)} windows={n_win}")
        if got < PER_AUTHOR:
            print(f"[warn] {labels[pid]}: {got} held-out works")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    entries = load_manifest_texts("novel")
    held = select_heldout(entries, args.offline)
    HELDOUT_MANIFEST.write_text(json.dumps([{k: v for k, v in h.items() if k != "text"} for h in held],
                                           ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    aids = sorted({e["author_id"] for e in entries})
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    feats = featurize_all(entries)
    hfeats = featurize_all(held)
    win = {(e["author_id"], e["work_id"]): sa.make_windows(e["text"]) for e in entries + held}
    OX = {k: ortho_matrix_ext(v) for k, v in win.items()}
    NR = {k: ortho_matrix_narr(v) for k, v in win.items()}

    def Z(a, w):
        return (feats.get(a, {}).get(w) or hfeats[a][w])["Z"]

    variants = {
        "fisher_z24_ortho_ext": lambda a, w: np.hstack([Z(a, w), OX[(a, w)]]),
        "fisher_z24_ortho_ext_narr": lambda a, w: np.hstack([Z(a, w), OX[(a, w)], NR[(a, w)]]),
    }
    train = {a: [e["work_id"] for e in entries if e["author_id"] == a and len(feats[a][e["work_id"]]["Z"])] for a in aids}
    out = {"schema": "p38_heldout_lowo_v1", "primary": "fisher_z24_ortho_ext_narr", "reg": REG, "labels": labels,
           "n_train_works": sum(len(v) for v in train.values()), "n_heldout": len(held), "chance": 1.0 / len(aids),
           "heldout": [{k: h[k] for k in ("author_id", "title", "work_id", "char_count", "n_windows")} for h in held],
           "results": {}}
    for name, fx in variants.items():
        model = fit_fisher({a: np.concatenate([fx(a, w) for w in train[a]]) for a in aids}, reg=REG)
        rows = []
        for h in held:
            a, w = h["author_id"], h["work_id"]
            X = fx(a, w)
            assert len(X) == len(win[(a, w)]), (a, w)
            sc = model.scores(X)
            rank = sorted(sc, key=sc.get, reverse=True)
            rows.append({"author_id": a, "work_id": w, "predicted": rank[0], "correct": rank[0] == a,
                         "rank_of_truth": rank.index(a) + 1})
        n = len(rows)
        out["results"][name] = {
            "n": n, "n_correct": sum(r["correct"] for r in rows), "accuracy": sum(r["correct"] for r in rows) / n,
            "top3_accuracy": sum(r["rank_of_truth"] <= 3 for r in rows) / n,
            "per_author": {a: f"{sum(r['correct'] for r in rows if r['author_id'] == a)}/{sum(r['author_id'] == a for r in rows)}" for a in aids},
            "rows": rows}
        r = out["results"][name]
        print(f"{name}: {r['n_correct']}/{n} acc={r['accuracy']:.3f} top3={r['top3_accuracy']:.3f} {r['per_author']}")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
