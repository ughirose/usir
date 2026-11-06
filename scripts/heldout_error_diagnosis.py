#!/usr/bin/env python
"""Phase 40 (local, post hoc): why the Phase 38-2 held-out errors go to one author.

For a held-out work with windows x_w (standardised), the Fisher score difference between the predicted class P and the
true class T is exactly
    S_P - S_T = n * (xbar - mu_mid) . v,   v = W (m_P - m_T),   mu_mid = (mu_P + mu_T) / 2
(mu_c: standardised feature mean of class c's training windows; m_c = mu_c W).  This splits the gap into one term per
feature.  The script reports the gap per feature group (24D, extended orthographic, narrative) and the largest terms.
It also reports, for every held-out work, how far its mean is from its author's training mean (Mahalanobis in the
projected space), to see whether the errors are works atypical of their author.

This is a diagnostic after seeing the result; nothing here changes the instrument.
Output: docs/handoff/v10/p40_heldout_error_diagnosis.json
Usage:  cd python && uv run python ../scripts/heldout_error_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_aozora_style_atlas import featurize_all  # noqa: E402
from fetch_aozora import CACHE_DIR, load_manifest_texts  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.orthographic_prior import NARR_NAMES, ORTHO_EXT_NAMES, ortho_matrix_ext, ortho_matrix_narr  # noqa: E402

OUT = ROOT / "docs" / "handoff" / "v10" / "p40_heldout_error_diagnosis.json"
RES = ROOT / "docs" / "handoff" / "v10" / "p38_heldout_lowo.json"
MAN = ROOT / "data" / "aozora_heldout_manifest.json"
REG = 1e-2


def main() -> None:
    entries = load_manifest_texts("novel")
    held = json.loads(MAN.read_text(encoding="utf-8"))
    for h in held:
        h["text"] = (CACHE_DIR / f"heldout_{h['author_id']}_{h['work_id']}.txt").read_text(encoding="utf-8")
    res = json.loads(RES.read_text(encoding="utf-8"))["results"]["fisher_z24_ortho_ext_narr"]["rows"]
    feats, hfeats = featurize_all(entries), featurize_all(held)
    names = [f"z24_{i}" for i in range(24)] + list(ORTHO_EXT_NAMES) + list(NARR_NAMES)
    groups = {"z24": range(0, 24), "ortho_ext": range(24, 24 + len(ORTHO_EXT_NAMES)),
              "narrative": range(24 + len(ORTHO_EXT_NAMES), len(names))}

    def X(e, fz):
        w = sa.make_windows(e["text"])
        return np.hstack([fz[e["author_id"]][e["work_id"]]["Z"], ortho_matrix_ext(w), ortho_matrix_narr(w)])

    aids = sorted({e["author_id"] for e in entries})
    train = {a: np.concatenate([X(e, feats) for e in entries if e["author_id"] == a and len(feats[a][e["work_id"]]["Z"])])
             for a in aids}
    fm = fit_fisher(train, reg=REG)
    assert len(names) == fm.W.shape[0], (len(names), fm.W.shape)
    std = {a: ((v - fm.center) / fm.scale).mean(axis=0) for a, v in train.items()}
    rows, typ = [], []
    for h in held:
        Xh = X(h, hfeats)
        Y = fm.project(Xh)
        a = h["author_id"]
        typ.append({"author_id": a, "work_id": h["work_id"], "title": h["title"], "n_windows": len(Xh),
                    "dist_to_own_mean": float(np.linalg.norm(Y.mean(axis=0) - fm.means[a])),
                    "dist_to_nearest_other": float(min(np.linalg.norm(Y.mean(axis=0) - fm.means[b]) for b in aids if b != a))})
    by_work = {(r["author_id"], r["work_id"]): r for r in res}
    for h in held:
        r = by_work[(h["author_id"], h["work_id"])]
        if r["correct"]:
            continue
        T, P = r["author_id"], r["predicted"]
        Xs = (X(h, hfeats) - fm.center) / fm.scale
        n = len(Xs)
        v = fm.W @ (fm.means[P] - fm.means[T])
        term = n * (Xs.mean(axis=0) - 0.5 * (std[P] + std[T])) * v
        sc = fm.scores(X(h, hfeats))
        assert abs(term.sum() - (sc[P] - sc[T])) < 1e-6 * max(1.0, abs(sc[P] - sc[T])), (term.sum(), sc[P] - sc[T])
        top = np.argsort(-term)[:6]
        rows.append({"author_id": T, "predicted": P, "work_id": h["work_id"], "title": h["title"], "n_windows": n,
                     "gap": float(term.sum()), "gap_per_window": float(term.sum() / n),
                     "by_group": {g: float(term[list(ix)].sum()) for g, ix in groups.items()},
                     "top_terms": [{"feature": names[j], "term": float(term[j]),
                                    "heldout_std_mean": float(Xs[:, j].mean()), "true_author_std_mean": float(std[T][j]),
                                    "predicted_author_std_mean": float(std[P][j])} for j in top]})
    d_ok = [t["dist_to_own_mean"] for t in typ if by_work[(t["author_id"], t["work_id"])]["correct"]]
    d_bad = [t["dist_to_own_mean"] for t in typ if not by_work[(t["author_id"], t["work_id"])]["correct"]]
    out = {"schema": "p40_heldout_error_diagnosis_v1", "post_hoc": True, "errors": rows, "typicality": typ,
           "dist_to_own_mean_correct": {"mean": float(np.mean(d_ok)), "max": float(np.max(d_ok))},
           "dist_to_own_mean_errors": {"mean": float(np.mean(d_bad)), "min": float(np.min(d_bad))}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r["title"], r["n_windows"], round(r["gap"], 2), {k: round(v, 2) for k, v in r["by_group"].items()})
        for t in r["top_terms"][:4]:
            print("   ", t["feature"], round(t["term"], 2), round(t["heldout_std_mean"], 2), round(t["true_author_std_mean"], 2),
                  round(t["predicted_author_std_mean"], 2))
    print("dist own mean: correct", out["dist_to_own_mean_correct"], "errors", out["dist_to_own_mean_errors"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
