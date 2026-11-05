#!/usr/bin/env python
"""Measure ||p||_2 and Shock-group mean q[3,7,11] per chunk for one genre of the Aozora manifest.

Reproduces the per-chunk quantities used for the novel QDA calibration
(python/usir_py/manifold_geodesic.py DOMAIN_QDA_PROFILES["novel"]; python/tests/test_manifold_geodesic.py
section 1b): extract_usir_24d(chunk) -> ||p||_2 and groups["IV_Shock"] = mean(q[3], q[7], q[11]).
The 40-chunk golden corpus has 162-441 characters per chunk (mean 281), so texts are cut into
sentence-aligned chunks of the same size (target 260 chars, hard maximum 440, minimum 160).

It does NOT modify any QDA constant.

Usage:
    cd python && uv run python ../scripts/measure_qda_domain.py --genre article
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from usir_py.extractor_24d import extract_usir_24d, split_sentences  # noqa: E402

TARGET, MAX_LEN, MIN_LEN = 260, 440, 160
QUANTILE_KEYS = [("min", 0.0), ("q05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95), ("max", 1.0)]


def chunk_text(text: str, target: int = TARGET, max_len: int = MAX_LEN, min_len: int = MIN_LEN) -> List[str]:
    """Greedy sentence-aligned chunks of ``target``..``max_len`` characters; a short tail is dropped."""
    sents = [s.strip() for s in split_sentences(text.replace("\n", "")) if s.strip()]
    chunks: List[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + len(s) > max_len:
            chunks.append(cur)
            cur = ""
        cur += s
        if len(cur) >= target:
            chunks.append(cur)
            cur = ""
    if len(cur) >= min_len:
        chunks.append(cur)
    return [c for c in chunks if min_len <= len(c) <= max_len]


def quantiles(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {k: float(np.quantile(arr, q)) for k, q in QUANTILE_KEYS} | {"mean": float(arr.mean()), "n": int(arr.size)}


def measure(texts: List[Dict]) -> Dict:
    chunks_out = []
    for e in texts:
        for i, ch in enumerate(chunk_text(e["text"])):
            ext = extract_usir_24d(ch)
            q = np.asarray(ext["q"])
            p = np.asarray(ext["p"])
            chunks_out.append({
                "work": e["title"],
                "author_id": e["author_id"],
                "index": i,
                "chars": len(ch),
                "p_norm": float(np.linalg.norm(p)),
                "shock_mean": float(np.mean([q[3], q[7], q[11]])),
            })
    pn = [c["p_norm"] for c in chunks_out]
    sh = [c["shock_mean"] for c in chunks_out]
    per_work = {}
    for title in sorted({c["work"] for c in chunks_out}):
        sub = [c for c in chunks_out if c["work"] == title]
        per_work[title] = {"p_norm": quantiles([c["p_norm"] for c in sub]), "shock_mean": quantiles([c["shock_mean"] for c in sub])}
    return {
        "n_chunks": len(chunks_out),
        "p_norm": quantiles(pn),
        "shock_mean": quantiles(sh),
        "per_work": per_work,
        "chunks": chunks_out,
    }


def golden_reference() -> Dict:
    """Same two quantities on the 40-chunk golden corpus, per phase (reference for the novel profile)."""
    from usir_py.corpus_40_benchmark import CORPUS

    ref: Dict[str, Dict[str, List[float]]] = {}
    for item in CORPUS:
        ext = extract_usir_24d(item["text"])
        q = np.asarray(ext["q"])
        d = ref.setdefault(item["phase"], {"p_norm": [], "shock_mean": []})
        d["p_norm"].append(float(np.linalg.norm(ext["p"])))
        d["shock_mean"].append(float(np.mean([q[3], q[7], q[11]])))
    return {ph: {"p_norm": quantiles(v["p_norm"]), "shock_mean": quantiles(v["shock_mean"])} for ph, v in ref.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", default="article")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "qda_calibration" / "article_features.json"))
    args = ap.parse_args()

    from fetch_aozora import load_manifest_texts

    texts = load_manifest_texts(args.genre)
    res = measure(texts)
    res = {
        "genre": args.genre,
        "chunking": {"target": TARGET, "max": MAX_LEN, "min": MIN_LEN, "unit": "characters, sentence-aligned"},
        "n_works": len(texts),
        **res,
        "golden_corpus_reference_by_phase": golden_reference(),
        "current_novel_constants": {"sigma_crit_low": 1.40, "sigma_crit_high": 3.90, "shock_mean_max": 0.60},
        "current_article_constants": {"sigma_crit_low": 0.30, "sigma_crit_high": 1.40},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"genre={args.genre} works={len(texts)} chunks={res['n_chunks']}")
    print("p_norm    ", {k: round(v, 3) for k, v in res["p_norm"].items()})
    print("shock_mean", {k: round(v, 3) for k, v in res["shock_mean"].items()})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
