#!/usr/bin/env python3
"""P-2b Task A': Layer C 置換。R1〜R3 の 48 件を、青空文庫・戦後作品コーパス（data/aozora_p2b_manifest.json）から
抽出した非可逆プロファイルへ置換した 64 件（R1〜R3 = 戦後 48 件 + R4 = 既存 16 件）を生成する。

- 原文は data/aozora_cache/p2b_*.txt（gitignore 対象、リポジトリ外扱い）から読み、出力には数値プロファイルと出所
  （作品 ID・著者 ID・場面番号・窓の SHA-256）のみを書く。原文の直書きは無い。
- 旧 64 件 (tier_c_modern_benchmark_expanded.jsonl) は evidence として不変。
- 窓の選定 seed = 20261117 + 100。著者ごとに 4 窓（作品が複数あれば作品を巡回）、各窓は 400〜800 字、「。」「」」「！」「？」のいずれかで終わる。
- レジスター対応: R1〜R3 の元の意図（現代口語小説・随筆・技術文書）に対応する戦後 NDC913 作品は小説のみで、
  R2・R3 に相当する文体は取得元に無い。したがって 48 件を著者均等（各 4 窓）で R1〜R3 の枠に巡回割当し、
  R1〜R3 のラベルは「別レジスター」を意味しない（全て戦後小説）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from usir_py.v5 import split_work_into_scenes  # noqa: E402
from usir_py.v5.modern_text_normalizer import extract_tier_c_irreversible_profile  # noqa: E402

MANIFEST = REPO / "data" / "aozora_p2b_manifest.json"
OLD = REPO / "data" / "usir_v5_eval" / "tier_c_modern_benchmark_expanded.jsonl"
NEW = REPO / "data" / "usir_v5_eval" / "tier_c_modern_benchmark_expanded_v2_postwar.jsonl"
MAPPING = REPO / "data" / "usir_v5_eval" / "tier_c_v2_postwar_mapping.json"
SEED = 20261117 + 100
PER_AUTHOR = 4
LO, HI = 400, 800
TERMS = "。」！？"  # 窓の終端（v1 は「。」のみ → 佐野昌一で 3 窓しか取れず、終端に 」！？ を追加）
REG_NAMES = {"R1": "R1_postwar_fiction_slot", "R2": "R2_postwar_fiction_slot", "R3": "R3_postwar_fiction_slot"}


def window(scene: str):
    t = scene.strip()
    if len(t) > HI:
        t = t[:HI]
        k = max(t.rfind(c) for c in TERMS)
        if k == -1:
            return None
        t = t[: k + 1]
    if len(t) < LO or t[-1] not in TERMS:
        return None
    return t


def main():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    works = man["works"]
    authors = sorted({w["author_id"] for w in works})
    assert len(authors) >= 12
    authors = authors[:12] if len(authors) > 12 else authors
    rng = np.random.default_rng(SEED)
    picked = {}  # aid -> list of dict(work_id, scene_index, text)
    for a in authors:
        wl = [w for w in works if w["author_id"] == a]
        cands = {}
        for w in wl:
            t = (REPO / "data" / "aozora_cache" / f"p2b_{a}_{w['work_id']}.txt").read_text(encoding="utf-8")
            lst = []
            for i, s in enumerate(split_work_into_scenes(t)):
                x = window(s)
                if x is not None:
                    lst.append((i, x))
            cands[w["work_id"]] = lst
        chosen, used = [], set()
        order = [w["work_id"] for w in wl]
        turn = 0
        while len(chosen) < PER_AUTHOR:
            wid = order[turn % len(order)]; turn += 1
            avail = [c for c in cands[wid] if (wid, c[0]) not in used]
            if not avail:
                if turn > 50:
                    raise RuntimeError(f"author {a}: not enough windows")
                continue
            i, x = avail[int(rng.integers(0, len(avail)))]
            used.add((wid, i)); chosen.append({"work_id": wid, "scene_index": i, "text": x})
        picked[a] = chosen
    # slot assignment: register r (0..2), slot i (0..15) -> author index (r*16+i) % 12, k-th window of that author
    counters = {a: 0 for a in authors}
    out, mapping = [], []
    for r_i, reg in enumerate(["R1", "R2", "R3"]):
        for i in range(16):
            a = authors[(r_i * 16 + i) % len(authors)]
            w = picked[a][counters[a]]; counters[a] += 1
            sid = f"{reg}_{i + 1:02d}"
            prof = extract_tier_c_irreversible_profile(w["text"])
            sha = hashlib.sha256(w["text"].encode("utf-8")).hexdigest()
            rec = {"sample_id": sid, "register": reg, "register_name": REG_NAMES[reg], "char_len": len(w["text"]),
                   "source_type": "real_tier_c_aozora_postwar",
                   "source_description": f"青空文庫 人物ID {a} 作品ID {w['work_id']} 場面 {w['scene_index']}（戦後作品、没年1946-1955）",
                   "provenance": {"author_id": a, "work_id": w["work_id"], "scene_index": w["scene_index"], "window_sha256": sha},
                   "profile": prof}
            out.append(rec)
            mapping.append({"sample_id": sid, "author_id": a, "work_id": w["work_id"], "scene_index": w["scene_index"],
                            "window_sha256": sha, "char_len": len(w["text"])})
    assert all(c == PER_AUTHOR for c in counters.values()), counters
    # R4: unchanged from old file
    for line in OLD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["register"] == "R4":
                out.append(r)
    assert len(out) == 64 and len({r["sample_id"] for r in out}) == 64
    NEW.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    old_ids = {json.loads(l)["sample_id"]: json.loads(l) for l in OLD.read_text(encoding="utf-8").splitlines() if l.strip()}
    MAPPING.write_text(json.dumps({
        "stage": "P-2b", "seed": SEED, "window_chars": [LO, HI], "per_author": PER_AUTHOR, "n_authors": len(authors),
        "register_correspondence": "none (R2/R3 counterparts do not exist in postwar NDC913 fiction); 48 windows assigned round-robin by author across R1-R3 slots",
        "replaced": [{"sample_id": m["sample_id"], "old_source_description": old_ids[m["sample_id"]]["source_description"],
                      "old_register_name": old_ids[m["sample_id"]]["register_name"], **m} for m in mapping],
        "r4_unchanged": 16,
        "evidence_old_file": "data/usir_v5_eval/tier_c_modern_benchmark_expanded.jsonl (unchanged, profiles of the 48 pre-replacement samples retained)",
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    from collections import Counter
    print("authors", len(authors), "records", len(out), Counter(r["register"] for r in out))
    print("dialogue_ratio R1-R3 mean", np.mean([r["profile"]["dialogue_ratio"] for r in out if r["register"] != "R4"]).round(3))


if __name__ == "__main__":
    main()
