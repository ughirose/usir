#!/usr/bin/env python
"""Build the Aozora 24-D style atlas (Phase 20).

Inputs : data/aozora_manifest.json + data/aozora_cache/ (run scripts/fetch_aozora.py first), genre "novel".
Outputs: artifacts/style_atlas/<author_id>.usir           (Tier-M, shrinkage="scaled", alpha=0.75)
         artifacts/style_atlas/legacy/<author_id>.usir    (Tier-M, shrinkage="legacy", 0.2*S + 0.8*I)
         artifacts/style_atlas/atlas.json                 (profiles, MW2 / single-Gaussian W2 matrices, LOWO)

Windows: 600 chars, stride 300 (python/usir_py/style_atlas.py).  Validation: leave-one-WORK-out
nearest-author classification; the held-out author's genome is re-distilled without that work and
the work is assigned to the author whose genome gives the largest summed window log-likelihood.

Usage:  cd python && uv run python ../scripts/build_aozora_style_atlas.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_aozora import CACHE_DIR, load_manifest_texts  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "style_atlas"


def featurize_all(entries: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """{author_id: {work_id: features}} with an on-disk cache keyed by the cleaned-text sha256."""
    cache_path = CACHE_DIR / "window_features_v1.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    out: Dict[str, Dict[str, Dict]] = {}
    t0 = time.time()
    for e in entries:
        key = (e["sha256"], sa.WINDOW_CHARS, sa.STRIDE_CHARS)
        if key not in cache:
            cache[key] = sa.featurize_text(e["text"])
            print(f"  featurized {e['author_ja']} 「{e['title']}」: {len(cache[key]['Z'])} windows ({time.time() - t0:.0f}s)")
        out.setdefault(e["author_id"], {})[e["work_id"]] = cache[key]
    cache_path.write_bytes(pickle.dumps(cache))
    return out


def lowo(author_ids, works_by_author, feats, shrinkage: str, full_gmms: Dict[str, sa.Gmm],
         mode_assign: str = "rule", dialogue: Dict = None) -> Dict:
    rows = []
    for aid in author_ids:
        wids = [w for w in works_by_author[aid] if len(feats[aid][w]["Z"])]
        for held in wids:
            rest = [(w, feats[aid][w]["Z"]) for w in wids if w != held]
            dlg = {w: dialogue[(aid, w)] for w, _ in rest} if mode_assign == "quantile" else None
            reduced = sa.Gmm.from_binary(sa.distill_genome(rest, shrinkage=shrinkage, mode_assign=mode_assign, dialogue=dlg))
            gm = dict(full_gmms)
            gm[aid] = reduced
            Z = feats[aid][held]["Z"]
            sc = {a: float(g.loglik(Z).sum()) for a, g in gm.items()}
            ranking = sorted(sc, key=sc.get, reverse=True)
            rows.append({
                "author_id": aid, "work_id": held, "n_windows": int(len(Z)),
                "predicted": ranking[0], "correct": ranking[0] == aid,
                "rank_of_truth": ranking.index(aid) + 1,
            })
    n = len(rows)
    per_author = {a: {"n": sum(r["author_id"] == a for r in rows),
                      "correct": sum(r["correct"] for r in rows if r["author_id"] == a)} for a in author_ids}
    return {
        "shrinkage": shrinkage,
        "mode_assign": mode_assign,
        "n_works": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "top3_accuracy": sum(r["rank_of_truth"] <= 3 for r in rows) / n,
        "chance": 1.0 / len(author_ids),
        "majority_baseline": max(v["n"] for v in per_author.values()) / n,
        "per_author": per_author,
        "rows": rows,
    }


def main_quantile() -> None:
    """Phase 26.1: quantile 3-mode genomes -> artifacts/style_atlas/quantile/ (existing files untouched)."""
    qdir = OUT_DIR / "quantile"
    entries = load_manifest_texts("novel")
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    author_ids = sorted(labels)
    feats = featurize_all(entries)
    works_by_author = {a: [e["work_id"] for e in entries if e["author_id"] == a] for a in author_ids}
    author_works = {a: [(e["work_id"], e["text"]) for e in entries if e["author_id"] == a] for a in author_ids}
    dialogue = {(e["author_id"], e["work_id"]): sa.quote_ratios(e["text"]) for e in entries}

    built = sa.build_atlas(author_works, qdir, features=feats, mode_assign="quantile")
    failed = [(a, sh) for a, v in built.items() for sh, au in v["audit"].items() if not au["passed"]]
    print("cleanroom gate (quantile):", "all passed" if not failed else f"FAILED {failed}")

    gm_q = {"scaled": sa.load_atlas_gmms(qdir), "legacy": sa.load_atlas_gmms(qdir / "legacy")}
    gm_rule = {"scaled": sa.load_atlas_gmms(OUT_DIR), "legacy": sa.load_atlas_gmms(OUT_DIR / "legacy")}
    dist_q = {sh: sa.distance_matrices(g) for sh, g in gm_q.items()}
    lowo_q = {sh: lowo(author_ids, works_by_author, feats, sh, gm_q[sh], "quantile", dialogue) for sh in gm_q}
    lowo_rule = {sh: lowo(author_ids, works_by_author, feats, sh, gm_rule[sh]) for sh in gm_rule}
    for tag, res in (("quantile", lowo_q), ("rule", lowo_rule)):
        for sh, r in res.items():
            print(f"LOWO[{tag}/{sh}] acc={r['accuracy']:.3f} top3={r['top3_accuracy']:.3f} n={r['n_works']}")

    def mw2_vs_w2(d):
        a, b = np.array(d["mw2"]), np.array(d["w2_single_gaussian"])
        iu = np.triu_indices(len(a), 1)
        diff = np.abs(a - b)[iu]
        return {"mean_abs_diff": float(diff.mean()), "max_abs_diff": float(diff.max()),
                "mean_mw2": float(a[iu].mean()), "mean_w2": float(b[iu].mean())}

    authors = {}
    for a in author_ids:
        authors[a] = {
            "label_ja": labels[a],
            "n_windows": built[a]["profile"]["n_windows"],
            "mode_weights_quantile": {sh: [float(x) for x in gm_q[sh][a].weights] for sh in gm_q},
            "mode_weights_rule": [float(x) for x in gm_rule["scaled"][a].weights],
            "centroid_distance_between_modes": {
                f"{i}-{j}": float(np.linalg.norm(gm_q["scaled"][a].means[i] - gm_q["scaled"][a].means[j]))
                for i in range(3) for j in range(i + 1, 3)},
            "cleanroom_gate": {sh: au["passed"] for sh, au in built[a]["audit"].items()},
        }
    out = {
        "mode_assign": "quantile (Dialogue top25% by quote ratio; Climax top25% of rest by z(g_shock)+z(||p||); rest Narration)",
        "authors": authors,
        "mw2_vs_single_gaussian": {"quantile": {sh: mw2_vs_w2(d) for sh, d in dist_q.items()},
                                   "rule": {sh: mw2_vs_w2(sa.distance_matrices(g)) for sh, g in gm_rule.items()}},
        "distances_quantile": dist_q,
        "lowo": {"quantile": lowo_q, "rule": lowo_rule},
    }
    (qdir / "atlas.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + chr(10), encoding="utf-8")
    print(f"-> {qdir / 'atlas.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-assign", choices=["rule", "quantile"], default="rule")
    if ap.parse_args().mode_assign == "quantile":
        return main_quantile()
    entries = load_manifest_texts("novel")
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    titles = {(e["author_id"], e["work_id"]): e["title"] for e in entries}
    author_ids = sorted(labels)
    print(f"{len(entries)} novel works, {len(author_ids)} authors")

    feats = featurize_all(entries)
    works_by_author = {a: [e["work_id"] for e in entries if e["author_id"] == a] for a in author_ids}
    author_works = {a: [(e["work_id"], e["text"]) for e in entries if e["author_id"] == a] for a in author_ids}

    built = sa.build_atlas(author_works, OUT_DIR, features=feats)
    failed = [(a, sh) for a, v in built.items() for sh, au in v["audit"].items() if not au["passed"]]
    print("cleanroom gate:", "all passed" if not failed else f"FAILED {failed}")

    gm_scaled = sa.load_atlas_gmms(OUT_DIR)
    gm_legacy = sa.load_atlas_gmms(OUT_DIR / "legacy")
    dist_scaled = sa.distance_matrices(gm_scaled)
    dist_legacy = sa.distance_matrices(gm_legacy)

    res_scaled = lowo(author_ids, works_by_author, feats, "scaled", gm_scaled)
    res_legacy = lowo(author_ids, works_by_author, feats, "legacy", gm_legacy)
    for r in (res_scaled, res_legacy):
        print(f"LOWO[{r['shrinkage']}] acc={r['accuracy']:.3f} top3={r['top3_accuracy']:.3f} "
              f"chance={r['chance']:.3f} majority={r['majority_baseline']:.3f} n={r['n_works']}")

    atlas = {
        "window_chars": sa.WINDOW_CHARS,
        "stride_chars": sa.STRIDE_CHARS,
        "shrinkage_primary": "scaled (alpha=0.75)",
        "distance_primary": "MW2 (mixture Wasserstein, Delon-Desolneux, 3x3 LP); entries are MW2 (sqrt of MW2^2)",
        "authors": {
            a: {
                "label_ja": labels[a],
                "works": [titles[(a, w)] for w in works_by_author[a]],
                "n_works": len(works_by_author[a]),
                **built[a]["profile"],
                "cleanroom_gate": {sh: au["passed"] for sh, au in built[a]["audit"].items()},
                "binary_sha256": __import__("hashlib").sha256((OUT_DIR / f"{a}.usir").read_bytes()).hexdigest(),
            } for a in author_ids
        },
        "distances": {"scaled": dist_scaled, "legacy": dist_legacy},
        "lowo": {"scaled": res_scaled, "legacy": res_legacy},
    }
    (OUT_DIR / "atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {OUT_DIR / 'atlas.json'}")


if __name__ == "__main__":
    main()
