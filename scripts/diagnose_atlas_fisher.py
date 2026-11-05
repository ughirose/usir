#!/usr/bin/env python
"""Phase 30: offline diagnostics of the Aozora atlas and the class-balanced Fisher metric.

1. Orthographic prior per author (mean sentence length etc.) on the real texts.
2. Sensitivity of DeltaLL_AB (genome log-likelihood A minus B, quantile 3-mode genomes) on real windows:
   window-level ROC-AUC and Cohen's d for A-works vs B-works, per-work means for all authors.  Works of A and B are
   scored with that author's genome re-distilled WITHOUT the work (no self-scoring).
3. Leave-one-work-out author identification: existing GMM (atlas.json) vs class-balanced Fisher LDA on
   z24 / orthographic prior (7) / extended prior (27: + function-kana and sentence-final character frequencies),
   alone and with z24 (reg fixed in advance at 1e-2; all variants reported).

Output: docs/handoff/v10/p30_atlas_diagnostics.json
Usage:  cd python && uv run python ../scripts/diagnose_atlas_fisher.py
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
from usir_py.orthographic_prior import ORTHO_EXT_NAMES, ORTHO_NAMES, ortho_matrix, ortho_matrix_ext, ortho_summary  # noqa: E402

QDIR = ROOT / "artifacts" / "style_atlas" / "quantile"
OUT = ROOT / "docs" / "handoff" / "v10" / "p30_atlas_diagnostics.json"
PAIRS = [("000879", "001383"), ("000879", "000035"), ("000148", "001383")]
REG = 1e-2


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def main() -> None:
    t0 = time.time()
    entries = load_manifest_texts("novel")
    labels = {e["author_id"]: e["author_ja"] for e in entries}
    aids = sorted(labels)
    feats = featurize_all(entries)
    texts = {(e["author_id"], e["work_id"]): e["text"] for e in entries}
    dialogue = {k: sa.quote_ratios(t) for k, t in texts.items()}
    works = {a: [e["work_id"] for e in entries if e["author_id"] == a and len(feats[a][e["work_id"]]["Z"])] for a in aids}
    O = {(a, w): ortho_matrix(sa.make_windows(texts[(a, w)])) for a in aids for w in works[a]}
    OX = {(a, w): ortho_matrix_ext(sa.make_windows(texts[(a, w)])) for a in aids for w in works[a]}
    for (a, w), M in O.items():
        assert len(M) == len(feats[a][w]["Z"]), (a, w)
    out = {"n_works": sum(len(v) for v in works.values()), "labels": labels, "reg": REG}

    # 1. orthographic prior per author
    out["ortho_by_author"] = {a: ortho_summary([x for w in works[a] for x in sa.make_windows(texts[(a, w)])]) for a in aids}

    # 2. DeltaLL sensitivity on real windows
    gm = sa.load_atlas_gmms(QDIR)

    def genome_wo(a, held):
        rest = [(w, feats[a][w]["Z"]) for w in works[a] if w != held]
        dlg = {w: dialogue[(a, w)] for w, _ in rest}
        return sa.Gmm.from_binary(sa.distill_genome(rest, shrinkage="scaled", mode_assign="quantile", dialogue=dlg))

    dll = {}
    for A, B in PAIRS:
        per_work, win_by_author = {}, {}
        for a in aids:
            for w in works[a]:
                Z = feats[a][w]["Z"]
                gA = genome_wo(A, w) if a == A else gm[A]
                gB = genome_wo(B, w) if a == B else gm[B]
                d = gA.loglik(Z) - gB.loglik(Z)
                per_work[f"{a}/{w}"] = {"mean": float(d.mean()), "n": int(len(d))}
                win_by_author.setdefault(a, []).append(d)
        wa, wb = np.concatenate(win_by_author[A]), np.concatenate(win_by_author[B])
        work_a = np.array([per_work[f"{A}/{w}"]["mean"] for w in works[A]])
        work_b = np.array([per_work[f"{B}/{w}"]["mean"] for w in works[B]])
        dll[f"{A}_vs_{B}"] = {
            "window_auc_A_over_B": auc(wa, wb), "window_cohen_d": cohen_d(wa, wb),
            "window_mean_A": float(wa.mean()), "window_mean_B": float(wb.mean()),
            "window_sd_A": float(wa.std(ddof=1)), "window_sd_B": float(wb.std(ddof=1)),
            "work_means_A": work_a.tolist(), "work_means_B": work_b.tolist(),
            "works_A_positive": int((work_a > 0).sum()), "works_B_negative": int((work_b < 0).sum()),
            "per_author_window_mean": {a: float(np.concatenate(v).mean()) for a, v in win_by_author.items()},
            "per_work": per_work,
        }
        print(f"dLL {A} vs {B}: AUC={dll[f'{A}_vs_{B}']['window_auc_A_over_B']:.3f} d={dll[f'{A}_vs_{B}']['window_cohen_d']:.2f}")
    out["delta_ll_real_text"] = dll

    # mode-centroid distances between the two genomes of the main pair (scaled, standardised by window SD)
    allZ = np.concatenate([feats[a][w]["Z"] for a in aids for w in works[a]])
    sd = allZ.std(axis=0) + 1e-9
    gA, gB = gm["000879"], gm["001383"]
    out["mode_centroid_dist_A_B_std_units"] = [[float(np.linalg.norm((gA.means[i] - gB.means[j]) / sd))
                                                for j in range(len(gB.means))] for i in range(len(gA.means))]
    out["author_mean_dist_A_B_std_units"] = float(np.linalg.norm(
        (np.concatenate([feats["000879"][w]["Z"] for w in works["000879"]]).mean(0)
         - np.concatenate([feats["001383"][w]["Z"] for w in works["001383"]]).mean(0)) / sd))

    # 3. LOWO
    variants = {
        "fisher_z24": lambda a, w: feats[a][w]["Z"],
        "fisher_ortho": lambda a, w: O[(a, w)],
        "fisher_z24_ortho": lambda a, w: np.hstack([feats[a][w]["Z"], O[(a, w)]]),
        "fisher_ortho_ext": lambda a, w: OX[(a, w)],
        "fisher_z24_ortho_ext": lambda a, w: np.hstack([feats[a][w]["Z"], OX[(a, w)]]),
    }
    lowo = {}
    for name, fx in variants.items():
        rows = []
        for a in aids:
            for held in works[a]:
                Xc = {}
                for b in aids:
                    ws = [w for w in works[b] if not (b == a and w == held)]
                    Xc[b] = np.concatenate([fx(b, w) for w in ws])
                fm = fit_fisher(Xc, reg=REG)
                sc = fm.scores(fx(a, held))
                rank = sorted(sc, key=sc.get, reverse=True)
                rows.append({"author_id": a, "work_id": held, "predicted": rank[0], "correct": rank[0] == a,
                             "rank_of_truth": rank.index(a) + 1})
        n = len(rows)
        lowo[name] = {
            "accuracy": sum(r["correct"] for r in rows) / n,
            "top3_accuracy": sum(r["rank_of_truth"] <= 3 for r in rows) / n,
            "per_author": {a: f"{sum(r['correct'] for r in rows if r['author_id'] == a)}/{sum(r['author_id'] == a for r in rows)}" for a in aids},
            "rows": rows,
        }
        print(f"LOWO {name}: acc={lowo[name]['accuracy']:.3f} top3={lowo[name]['top3_accuracy']:.3f} {lowo[name]['per_author']}")
    prev = json.loads((QDIR / "atlas.json").read_text(encoding="utf-8"))
    base = prev["lowo"]["quantile"]["scaled"]
    lowo["gmm_quantile_scaled_phase26"] = {"accuracy": base["accuracy"], "top3_accuracy": base["top3_accuracy"],
                                           "per_author": {a: f"{v['correct']}/{v['n']}" for a, v in base["per_author"].items()}}
    out["lowo"] = lowo
    out["chance"] = 1.0 / len(aids)

    # 4. Fisher-space score difference DeltaF_AB = s_A - s_B per window (primary variant), works of A and B scored by a
    #    metric fitted without that work.
    fx = variants["fisher_z24_ortho_ext"]
    allX = {b: np.concatenate([fx(b, w) for w in works[b]]) for b in aids}
    full = fit_fisher(allX, reg=REG)
    dF = {}
    for A, B in PAIRS:
        wins = {}
        for a in (A, B):
            for held in works[a]:
                fm = fit_fisher({b: np.concatenate([fx(b, w) for w in works[b] if not (b == a and w == held)])
                                 for b in aids}, reg=REG)
                Y = fm.project(fx(a, held))
                d = -0.5 * ((Y - fm.means[A]) ** 2).sum(1) + 0.5 * ((Y - fm.means[B]) ** 2).sum(1)
                wins.setdefault(a, []).append(d)
        wa, wb = np.concatenate(wins[A]), np.concatenate(wins[B])
        dF[f"{A}_vs_{B}"] = {"window_auc_A_over_B": auc(wa, wb), "window_cohen_d": cohen_d(wa, wb),
                             "work_means_A": [float(x.mean()) for x in wins[A]],
                             "work_means_B": [float(x.mean()) for x in wins[B]]}
        print(f"dF {A} vs {B}: AUC={dF[f'{A}_vs_{B}']['window_auc_A_over_B']:.3f} d={dF[f'{A}_vs_{B}']['window_cohen_d']:.2f}")
    out["delta_fisher_real_text"] = dF

    # 5. export the metric fitted on all works (consumed by the decoder's genome coupling, Phase 31)
    art = {
        "schema": "usir_fisher_atlas_v1", "feature": "z24 + ortho_ext", "feature_names": [f"z{i}" for i in range(24)] + list(ORTHO_EXT_NAMES),
        "reg": REG, "center": full.center.tolist(), "scale": full.scale.tolist(), "W": full.W.tolist(),
        "eigvals": full.eigvals.tolist(), "class_means_proj": {a: m.tolist() for a, m in full.means.items()},
        "mu_bar_proj": full.mu_bar.tolist(),
        "author_feature_means": {a: allX[a].mean(0).tolist() for a in aids},
        "ortho_by_author": out["ortho_by_author"], "labels": labels,
        "lowo_accuracy": lowo["fisher_z24_ortho_ext"]["accuracy"],
    }
    (ROOT / "artifacts" / "style_atlas" / "fisher_v1.json").write_text(json.dumps(art, ensure_ascii=False), encoding="utf-8")
    out["ortho_names"] = list(ORTHO_NAMES)
    out["ortho_ext_names"] = list(ORTHO_EXT_NAMES)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, out["elapsed_sec"], "s")


if __name__ == "__main__":
    main()
