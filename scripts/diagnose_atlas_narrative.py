#!/usr/bin/env python
"""Phase 36-1: LOWO author identification with narrative-channel orthographic statistics.

Pre-specified before the run (docs/handoff/2026-11-06_phase36_pinpoint.md):
  primary variant  fisher_z24_ortho_ext_narr = z24 + ortho_ext(27) + narrative(4)   (55 dims), reg = 1e-2
  narrative(4)     = narr_log_mean_sent_len, narr_log_p90_sent_len, narr_commas_per_sent, dialogue_char_ratio
                     (dialogue 「...」 replaced by one placeholder; window-boundary fragments dropped)
Also reported (not used for the decision): the Phase 30 baseline re-run, narrative alone, ortho_ext + narrative,
and the reg grid of Phase 30 for the primary variant.  The decoder metric artifacts/style_atlas/fisher_v1.json is
NOT modified.

Output: docs/handoff/v10/p36_lowo_narrative.json
Usage:  cd python && uv run python ../scripts/diagnose_atlas_narrative.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_aozora_style_atlas import featurize_all  # noqa: E402
from fetch_aozora import load_manifest_texts  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.orthographic_prior import NARR_NAMES, narrative_sentence_lengths, ortho_matrix_ext, ortho_matrix_narr  # noqa: E402

OUT = ROOT / "docs" / "handoff" / "v10" / "p36_lowo_narrative.json"
REG = 1e-2
REG_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
PRIMARY = "fisher_z24_ortho_ext_narr"


def lowo(aids, works, fx, reg):
    rows = []
    for a in aids:
        for held in works[a]:
            Xc = {b: np.concatenate([fx(b, w) for w in works[b] if not (b == a and w == held)]) for b in aids}
            sc = fit_fisher(Xc, reg=reg).scores(fx(a, held))
            rank = sorted(sc, key=sc.get, reverse=True)
            rows.append({"author_id": a, "work_id": held, "predicted": rank[0], "correct": rank[0] == a,
                         "rank_of_truth": rank.index(a) + 1})
    n = len(rows)
    return {"accuracy": sum(r["correct"] for r in rows) / n, "n_correct": sum(r["correct"] for r in rows), "n": n,
            "top3_accuracy": sum(r["rank_of_truth"] <= 3 for r in rows) / n,
            "per_author": {a: f"{sum(r['correct'] for r in rows if r['author_id'] == a)}/{sum(r['author_id'] == a for r in rows)}" for a in aids},
            "rows": rows}


def main() -> None:
    t0 = time.time()
    entries = load_manifest_texts("novel")
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    aids = sorted(labels)
    feats = featurize_all(entries)
    texts = {(e["author_id"], e["work_id"]): e["text"] for e in entries}
    works = {a: [e["work_id"] for e in entries if e["author_id"] == a and len(feats[a][e["work_id"]]["Z"])] for a in aids}
    win = {(a, w): sa.make_windows(texts[(a, w)]) for a in aids for w in works[a]}
    OX = {k: ortho_matrix_ext(v) for k, v in win.items()}
    NR = {k: ortho_matrix_narr(v) for k, v in win.items()}
    for (a, w) in win:
        assert len(OX[(a, w)]) == len(NR[(a, w)]) == len(feats[a][w]["Z"]), (a, w)
    variants = {
        "fisher_z24_ortho_ext": lambda a, w: np.hstack([feats[a][w]["Z"], OX[(a, w)]]),
        "fisher_narr": lambda a, w: NR[(a, w)],
        "fisher_ortho_ext_narr": lambda a, w: np.hstack([OX[(a, w)], NR[(a, w)]]),
        PRIMARY: lambda a, w: np.hstack([feats[a][w]["Z"], OX[(a, w)], NR[(a, w)]]),
    }
    out = {"schema": "p36_lowo_narrative_v1", "primary": PRIMARY, "reg": REG, "narr_names": list(NARR_NAMES),
           "labels": labels, "n_works": sum(len(v) for v in works.values()), "chance": 1.0 / len(aids), "lowo": {}}
    for name, fx in variants.items():
        out["lowo"][name] = lowo(aids, works, fx, REG)
        r = out["lowo"][name]
        print(f"LOWO {name}: {r['n_correct']}/{r['n']} acc={r['accuracy']:.3f} top3={r['top3_accuracy']:.3f} {r['per_author']}")
    out["reg_sensitivity_primary"] = {}
    for reg in REG_GRID:
        r = lowo(aids, works, variants[PRIMARY], reg)
        out["reg_sensitivity_primary"][f"{reg:g}"] = {"n_correct": r["n_correct"], "accuracy": r["accuracy"]}
        print(f"reg {reg:g}: {r['n_correct']}/{r['n']}")
    # author-level narrative statistics (pooled over all windows; whole-text narrative sentence lengths too)
    by = {}
    for a in aids:
        M = np.concatenate([NR[(a, w)] for w in works[a]])
        whole = [x for w in works[a] for x in narrative_sentence_lengths(texts[(a, w)])]
        by[a] = {n: float(M[:, i].mean()) for i, n in enumerate(NARR_NAMES)}
        by[a]["whole_text_narr_mean_sent_len"] = float(np.mean(whole)) if whole else 0.0
        by[a]["whole_text_narr_p90_sent_len"] = float(np.percentile(whole, 90)) if whole else 0.0
    out["narr_by_author"] = by
    out["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, out["elapsed_sec"], "s")


if __name__ == "__main__":
    main()
