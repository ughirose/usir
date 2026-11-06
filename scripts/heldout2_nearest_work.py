#!/usr/bin/env python
"""Phase 41: nearest-work scoring vs class-mean scoring on a second held-out set.

Criteria fixed in docs/handoff/2026-11-06_phase41_preregistration.md.  The second held-out set uses the Phase 38-2
selection rule, continuing in ascending work id after the works considered for the first set, with titles of the
training and first held-out works excluded.  Only ids, URLs and SHA256 are committed
(data/aozora_heldout2_manifest.json); texts stay in the git-ignored cache.

A: class mean (Phase 38-2).  B: per author, the projected mean of each training work; a held-out work is scored by
the best training work of that author, -0.5 * sum_w ||y_w - m_work||^2.

Output: docs/handoff/v10/p41_heldout2_nearest_work.json
Usage:  cd python && uv run python ../scripts/heldout2_nearest_work.py [--offline]
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
from heldout_lowo import MIN_CHARS, MIN_WINDOWS, PER_AUTHOR, REG, title_clash  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.orthographic_prior import ortho_matrix_ext, ortho_matrix_narr  # noqa: E402

OUT = ROOT / "docs" / "handoff" / "v10" / "p41_heldout2_nearest_work.json"
SET1 = ROOT / "data" / "aozora_heldout_manifest.json"
SET2 = ROOT / "data" / "aozora_heldout2_manifest.json"


def load_cached(pid: str, wid: str):
    c = CACHE_DIR / f"heldout_{pid}_{wid}.txt"
    m = c.with_suffix(".json")
    if c.exists() and m.exists():
        return c.read_text(encoding="utf-8"), json.loads(m.read_text(encoding="utf-8"))["zip_url"]
    return None


def select_set2(entries: list, set1: list, offline: bool) -> list:
    index = load_index()
    titles = {}
    for e in entries + set1:
        titles.setdefault(e["author_id"], []).append(e["title"])
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    last1 = {}
    for h in set1:
        last1[h["author_id"]] = max(last1.get(h["author_id"], 0), int(h["work_id"]))
    chosen = []
    for pid in sorted(labels):
        rows = [r for r in index if r["人物ID"] == pid and r["役割フラグ"] == "著者" and r["作品著作権フラグ"] == "なし"
                and "913" in r["分類番号"] and r["文字遣い種別"] == "新字新仮名"
                and r["テキストファイルURL"].startswith("https://www.aozora.gr.jp/cards/")
                and int(r["作品ID"]) > last1.get(pid, 0)  # continue after the works considered for set 1
                and not title_clash(r["作品名"], titles[pid])]
        rows.sort(key=lambda r: int(r["作品ID"]))
        seen, got = set(), 0
        for r in rows:
            if got >= PER_AUTHOR:
                break
            if r["作品名"] in seen:
                continue
            seen.add(r["作品名"])
            got_text = load_cached(pid, r["作品ID"])
            if got_text is None:
                if offline:
                    continue
                try:
                    f = fetch_clean_text(r)
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {pid} {r['作品ID']}: {exc}", file=sys.stderr)
                    continue
                time.sleep(DELAY_SEC)
                c = CACHE_DIR / f"heldout_{pid}_{r['作品ID']}.txt"
                c.write_text(f["text"], encoding="utf-8")
                c.with_suffix(".json").write_text(json.dumps({"zip_url": f["zip_url"]}), encoding="utf-8")
                got_text = (f["text"], f["zip_url"])
            text, zip_url = got_text
            nw = len(sa.make_windows(text))
            if len(text) < MIN_CHARS or nw < MIN_WINDOWS:
                print(f"[short] {labels[pid]} 「{r['作品名']}」 chars={len(text)} windows={nw}")
                continue
            chosen.append({"author_id": pid, "author_ja": labels[pid], "title": r["作品名"], "work_id": r["作品ID"],
                           "ndc": r["分類番号"], "card_url": r["図書カードURL"], "zip_url": zip_url,
                           "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "char_count": len(text),
                           "n_windows": nw, "genre": "heldout2", "text": text})
            got += 1
            print(f"[ok] {labels[pid]} 「{r['作品名']}」 id={r['作品ID']} chars={len(text)} windows={nw}")
        if got < PER_AUTHOR:
            print(f"[warn] {labels[pid]}: {got} works in set 2")
    return chosen


def evaluate(fm, train_works: dict, sets: dict, fx) -> dict:
    work_means = {a: [fm.project(X).mean(axis=0) for X in ws] for a, ws in train_works.items()}
    out = {}
    for sname, works in sets.items():
        res = {"A_class_mean": [], "B_nearest_work": []}
        for h in works:
            Y = fm.project(fx(h))
            a = h["author_id"]
            sA = {c: float(-0.5 * np.sum((Y - m) ** 2)) for c, m in fm.means.items()}
            sB = {c: max(float(-0.5 * np.sum((Y - m) ** 2)) for m in ms) for c, ms in work_means.items()}
            for k, s in (("A_class_mean", sA), ("B_nearest_work", sB)):
                rank = sorted(s, key=s.get, reverse=True)
                res[k].append({"author_id": a, "work_id": h["work_id"], "title": h["title"], "predicted": rank[0],
                               "correct": rank[0] == a, "rank_of_truth": rank.index(a) + 1})
        out[sname] = {k: {"n": len(v), "n_correct": sum(r["correct"] for r in v),
                          "accuracy": sum(r["correct"] for r in v) / len(v) if v else None,
                          "top3_accuracy": sum(r["rank_of_truth"] <= 3 for r in v) / len(v) if v else None,
                          "rows": v} for k, v in res.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    entries = load_manifest_texts("novel")
    set1 = json.loads(SET1.read_text(encoding="utf-8"))
    for h in set1:
        h["text"] = load_cached(h["author_id"], h["work_id"])[0]
    set2 = select_set2(entries, set1, args.offline)
    SET2.write_text(json.dumps([{k: v for k, v in h.items() if k != "text"} for h in set2], ensure_ascii=False, indent=1)
                    + "\n", encoding="utf-8")
    feats = featurize_all(entries)
    hf1, hf2 = featurize_all(set1), featurize_all(set2)

    def fx(e, fz=None):
        fz = fz or (feats if e.get("genre") not in ("heldout", "heldout2") else (hf1 if e["genre"] == "heldout" else hf2))
        w = sa.make_windows(e["text"])
        return np.hstack([fz[e["author_id"]][e["work_id"]]["Z"], ortho_matrix_ext(w), ortho_matrix_narr(w)])

    for h in set1:
        h["genre"] = "heldout"
    aids = sorted({e["author_id"] for e in entries})
    train_works = {a: [fx(e, feats) for e in entries if e["author_id"] == a and len(feats[a][e["work_id"]]["Z"])] for a in aids}
    fm = fit_fisher({a: np.concatenate(ws) for a, ws in train_works.items()}, reg=REG)
    out = {"schema": "p41_heldout2_nearest_work_v1", "reg": REG, "n_set1": len(set1), "n_set2": len(set2),
           "set2": [{k: h[k] for k in ("author_id", "title", "work_id", "char_count", "n_windows")} for h in set2],
           "results": evaluate(fm, train_works, {"set1": set1, "set2": set2}, fx)}
    r2 = out["results"]["set2"]
    out["criteria"] = {"c1_B_gt_A": r2["B_nearest_work"]["n_correct"] > r2["A_class_mean"]["n_correct"],
                       "c2_B_ge_070": (r2["B_nearest_work"]["accuracy"] or 0) >= 0.70}
    out["criteria"]["all_met"] = all(out["criteria"].values())
    out["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for s, v in out["results"].items():
        for k, x in v.items():
            print(s, k, f"{x['n_correct']}/{x['n']}", round(x["accuracy"], 3), "top3", round(x["top3_accuracy"], 3))
    print(out["criteria"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
