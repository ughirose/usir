#!/usr/bin/env python
"""Fable v5 grand ablation: deterministic, LLM-free style features on the Aozora corpus (106 works, 10 authors).

Corpus (same as Phases 36 / 38 / 41):
  train     46 novels            data/aozora_manifest.json (genre == "novel")
  held-out  30 + 30 works        data/aozora_heldout_manifest.json, data/aozora_heldout2_manifest.json
The 13 "article" works are not used.

Protocol, fixed before the first run (see nishiki docs/handoff/2026-11-11_fable_v5_grand_shootout.md):
  LOWO      leave one of the 46 train works out, fit everything on the other 45, classify the held one.
  Held-out  fit on the 46, classify the 60.
  No statistic of a test work enters scaling, imputation, a classifier, an author PPM model or an author pool.
  Features computed inside one work (fractal, RQA, LSC, self-adaptive PPM surprisal) use that work's text only.
  Hyper-parameters are constants below and were not tuned on any result.

Systems:
  A   window-level 55-dim Fisher (Phase 36 primary; usir_py.fisher_atlas.fit_fisher, reg 1e-2)
  A2  work-mean 55-dim vector + shrinkage LDA                       (reference only)
  B   nishiki-ppm: grammar-skeleton PPM-D cross-entropy, argmin over author models
  C   nishiki-fractal + nishiki-rqa work features -> LDA / 3-NN / logistic regression
  D   nishiki-ot: sliced Wasserstein-2 between 55-dim window clouds (nearest author pool / nearest work / Sinkhorn)
  L   log(character count) only -> logistic regression             (length-confound control)
  E   ensemble: per test work, each system's 10 author scores are z-scored across the 10 authors and summed
      with equal weights (no weights are learned).  E = A+B+C+D, E_new = B+C+D.

Rust side: C:/dev/nishiki, `cargo build --release -p nishiki-hardcore-cli`.

Usage:  cd python && uv run python ../experiments/fable_v5_grand_shootout.py [--reuse]
Output: docs/handoff/v10/fable_v5_grand_shootout.json, docs/handoff/v10/fable_v5_grand_shootout_tables.md
        docs/handoff/v10/fable_v5_followup.json, docs/handoff/v10/fable_v5_followup_tables.md (follow-up)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_aozora import CACHE_DIR, load_manifest_texts  # noqa: E402
from usir_py import style_atlas as sa  # noqa: E402
from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.orthographic_prior import NARR_NAMES, ORTHO_EXT_NAMES, ortho_matrix_ext, ortho_matrix_narr  # noqa: E402

NISHIKI = Path("C:/dev/nishiki")
EXE = NISHIKI / "target" / "release" / "nishiki-hardcore.exe"
OUT_DIR = ROOT / "docs" / "handoff" / "v10"
OUT = OUT_DIR / "fable_v5_grand_shootout.json"
TABLES = OUT_DIR / "fable_v5_grand_shootout_tables.md"
WORK_DIR = ROOT / "experiments" / "_fable_v5_work"

# ---- fixed protocol constants -------------------------------------------------------------------------------
REG = 1e-2            # Fisher regularisation (Phase 36)
KNN_K = 3
LOGREG_C = 1.0
PPM_ORDER = 5
SW_K = 128
SEED = 20261111
RQA_TARGET_RR = 0.05
SURROGATES = 8
SINKHORN_EPS_PER_DIM = 0.1   # eps = 0.1 * d in standardised squared-distance units
SINKHORN_POINTS = 96
SINKHORN_ITERS = 100

# Work-level features of system C.  Counts (n_chars, n_sentences, n_skeleton_tokens) and the fit-quality
# diagnostics (*_min_r2) are deliberately excluded.
C_FEATURES = [
    "sp_mean_bits", "sp_hurst", "sp_delta_alpha", "sp_dh_memory", "sp_hvg_kl", "sp_hvg_assort", "sp_hvg_mean_degree",
    "sl_hurst", "sl_delta_alpha", "sl_dh_memory", "sl_hvg_kl", "sl_hvg_assort", "sl_hvg_mean_degree",
    "rqa_det", "rqa_lam", "rqa_tt", "rqa_l_mean", "rqa_l_max_ratio",
    "lsc_word_mean", "lsc_word_sd", "lsc_skel_mean", "lsc_skel_sd",
]
C_GROUPS = {
    "ppm_surprisal_fractal": [f for f in C_FEATURES if f.startswith("sp_")],
    "sentence_length_fractal": [f for f in C_FEATURES if f.startswith("sl_")],
    "rqa": [f for f in C_FEATURES if f.startswith("rqa_")],
    "lsc": [f for f in C_FEATURES if f.startswith("lsc_")],
}
FEATURE_NAMES_55 = [f"z{i}" for i in range(24)] + list(ORTHO_EXT_NAMES) + list(NARR_NAMES)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def load_works():
    """All 106 works: dict with id, author, split, text, path, sha256."""
    works = []
    for e in load_manifest_texts("novel"):
        works.append({**e, "split": "train", "path": CACHE_DIR / f"novel_{e['author_id']}_{e['work_id']}.txt"})
    for split, name in (("heldout1", "aozora_heldout_manifest.json"), ("heldout2", "aozora_heldout2_manifest.json")):
        for e in json.loads((ROOT / "data" / name).read_text(encoding="utf-8")):
            p = CACHE_DIR / f"heldout_{e['author_id']}_{e['work_id']}.txt"
            text = p.read_text(encoding="utf-8")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != e["sha256"]:
                raise ValueError(f"sha256 mismatch: {p}")
            works.append({**e, "split": split, "path": p, "text": text})
    for w in works:
        w["id"] = f"{w['split']}:{w['author_id']}:{w['work_id']}"
        # the Rust side reads the same file; make sure it is the text the manifest describes
        assert w["path"].read_text(encoding="utf-8") == w["text"], w["path"]
    ids = [w["id"] for w in works]
    assert len(set(ids)) == len(ids)
    return works


def window_features(works):
    """55-dim window matrix per work (z24 from the read-only cache, ortho_ext + narr recomputed)."""
    cache_path = CACHE_DIR / "window_features_v1.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    X = {}
    n_recomputed = 0
    for w in works:
        key = (w["sha256"], sa.WINDOW_CHARS, sa.STRIDE_CHARS)
        if key in cache:
            Z = cache[key]["Z"]
        else:
            Z = sa.featurize_text(w["text"])["Z"]
            n_recomputed += 1
        win = sa.make_windows(w["text"])
        M = np.hstack([Z, ortho_matrix_ext(win), ortho_matrix_narr(win)])
        assert M.shape == (len(win), 55) and len(win) > 0, (w["id"], M.shape)
        assert np.isfinite(M).all(), w["id"]
        X[w["id"]] = M
    return X, n_recomputed


def run_rust(cmd: str, job: dict, name: str, reuse: bool):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jp, op = WORK_DIR / f"{name}_job.json", WORK_DIR / f"{name}_out.json"
    if not (reuse and op.exists()):
        jp.write_text(json.dumps(job), encoding="utf-8")
        t0 = time.time()
        r = subprocess.run([str(EXE), cmd, str(jp), str(op)], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3 * 3600)
        (WORK_DIR / f"{name}_stderr.txt").write_text(r.stderr, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"nishiki-hardcore {cmd} failed ({r.returncode}): {r.stderr[-2000:]}")
        print(f"[rust] {cmd} {name}: {time.time() - t0:.0f}s")
    return json.loads(op.read_text(encoding="utf-8"))


def zscore_scores(s: dict) -> dict:
    """z-score one work's author scores across the authors (uses nothing but that score vector)."""
    v = np.array(list(s.values()), dtype=float)
    sd = v.std()
    return {a: (float((x - v.mean()) / sd) if sd > 0 else 0.0) for a, x in s.items()}


def summarize(rows, aids):
    """rows: list of {id, author, scores: {author: higher-is-better}} -> metrics."""
    out_rows = []
    for r in rows:
        rank = sorted(aids, key=lambda a: (-r["scores"][a], a))
        z = zscore_scores(r["scores"])
        best_other = max(z[a] for a in aids if a != r["author"])
        out_rows.append({"id": r["id"], "author": r["author"], "predicted": rank[0], "correct": rank[0] == r["author"],
                         "rank_of_truth": rank.index(r["author"]) + 1, "margin_z": z[r["author"]] - best_other})
    n = len(out_rows)
    k = sum(r["correct"] for r in out_rows)
    conf = {a: {b: 0 for b in aids} for a in aids}
    for r in out_rows:
        conf[r["author"]][r["predicted"]] += 1
    return {"n": n, "n_correct": k, "accuracy": round(k / n, 4), "wilson95": wilson(k, n),
            "top3_accuracy": round(sum(r["rank_of_truth"] <= 3 for r in out_rows) / n, 4),
            "per_author": {a: f"{sum(r['correct'] for r in out_rows if r['author'] == a)}/"
                              f"{sum(r['author'] == a for r in out_rows)}" for a in aids},
            "confusion": conf, "rows": out_rows}


def sklearn_scores(kind: str, Xtr, ytr, Xte, aids):
    """Fit imputer + scaler + classifier on the training fold only; return per-class scores for the test rows."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = {"lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
           "knn": lambda: KNeighborsClassifier(n_neighbors=KNN_K, weights="distance"),
           "logreg": lambda: LogisticRegression(C=LOGREG_C, max_iter=5000)}[kind]()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)
    pipe.fit(Xtr, ytr)
    P = pipe.predict_proba(Xte)
    classes = list(pipe.classes_)
    out = []
    for row in P:
        s = {a: -1e9 for a in aids}
        for c, p in zip(classes, row):
            s[c] = float(np.log(max(p, 1e-300)))
        out.append(s)
    return out


def eval_feature_classifier(kind, feat, works, aids):
    """LOWO over train + held-out, for a work-level feature matrix `feat` (id -> vector)."""
    train = [w for w in works if w["split"] == "train"]
    held = [w for w in works if w["split"] != "train"]
    lowo_rows = []
    for t in train:
        rest = [w for w in train if w["id"] != t["id"]]
        s = sklearn_scores(kind, np.array([feat[w["id"]] for w in rest]), [w["author_id"] for w in rest],
                           np.array([feat[t["id"]]]), aids)[0]
        lowo_rows.append({"id": t["id"], "author": t["author_id"], "scores": s})
    ss = sklearn_scores(kind, np.array([feat[w["id"]] for w in train]), [w["author_id"] for w in train],
                        np.array([feat[w["id"]] for w in held]), aids)
    held_rows = [{"id": w["id"], "author": w["author_id"], "scores": s} for w, s in zip(held, ss)]
    return lowo_rows, held_rows


def pack(lowo_rows, held_rows, works, aids):
    split = {w["id"]: w["split"] for w in works}
    return {"lowo": summarize(lowo_rows, aids),
            "heldout": summarize(held_rows, aids),
            "heldout1": summarize([r for r in held_rows if split[r["id"]] == "heldout1"], aids),
            "heldout2": summarize([r for r in held_rows if split[r["id"]] == "heldout2"], aids),
            "_rows": {"lowo": lowo_rows, "heldout": held_rows}}


def ensemble(parts, works, aids):
    res = {}
    for part in ("lowo", "heldout"):
        rows = []
        by = [{r["id"]: r for r in p["_rows"][part]} for p in parts]
        for wid, r0 in by[0].items():
            zs = [zscore_scores(b[wid]["scores"]) for b in by]
            rows.append({"id": wid, "author": r0["author"], "scores": {a: sum(z[a] for z in zs) for a in aids}})
        res[part] = rows
    return pack(res["lowo"], res["heldout"], works, aids)


# ---- follow-up (2026-11-11, second run): three targeted corrections, system C dropped ------------------------
# Fixed before this run:
#   B_core / B_collapsed   skeleton with symbols collapsed (core: keep the six marks 、。！？「」; collapsed: one <SYM>)
#   B_cal_center           H(X||M_a) - mu_a,  mu_a = mean of H(w||M_a) over the fold's training works NOT by a
#   B_cal_z                (H - mu_a) / sd_a  with the same training works
#   B_equalized            every author model trained on the same number of tokens (the fold's smallest author)
#   D9_*                   sliced W2 after projecting windows with the fold's own Fisher (55 -> 9), K = 128
#   E3_equal               z-sum of A, B_cal_center, D9_sw2_author_pool (no learned weights)
#   E3_stacked             non-negative weights of the same three, fitted by conditional logit on the
#                          leave-one-out scores of the fold's training works (nested: both the outer test
#                          work and the inner work are excluded from every model that scores the inner work)
STACK_L2 = 0.01
FOLLOWUP_OUT = OUT_DIR / "fable_v5_followup.json"
FOLLOWUP_TABLES = OUT_DIR / "fable_v5_followup_tables.md"
SOSEKI, KAJII, AKUTAGAWA = "000148", "000074", "000879"


def followup(works, X, aids, labels, systems, reuse):
    from scipy.optimize import minimize

    t0 = time.time()
    train = [w for w in works if w["split"] == "train"]
    held = [w for w in works if w["split"] != "train"]
    author_of = {w["id"]: w["author_id"] for w in works}
    train_ids = [w["id"] for w in train]
    held_ids = [w["id"] for w in held]

    ppm = run_rust("ppm-table", {
        "works": [{"id": w["id"], "path": str(w["path"]), "author": w["author_id"],
                   "split": "train" if w["split"] == "train" else "heldout"} for w in works],
        "ppm_order": PPM_ORDER, "modes": ["surface", "core", "collapsed"], "double_loo": ["surface"],
    }, "ppm_table", reuse)["modes"]

    def ppm_fold(mode, q, excl):
        """Scores of work q under the models trained on train minus excl.  Returns H, mu, sd per author."""
        table = ppm[mode]["table"]
        H, mu, sd = {}, {}, {}
        for a in aids:
            row = table[a + "|" + ";".join(sorted(i for i in excl if author_of[i] == a))]
            H[a] = row[q]
            cal = [row[w] for w in train_ids if w not in excl and author_of[w] != a]
            mu[a], sd[a] = float(np.mean(cal)), float(np.std(cal))
        return H, mu, sd

    def b_scores(mode, kind, q, excl):
        H, mu, sd = ppm_fold(mode, q, excl)
        if kind == "plain":
            return {a: -H[a] for a in aids}
        if kind == "center":
            return {a: -(H[a] - mu[a]) for a in aids}
        return {a: -(H[a] - mu[a]) / sd[a] for a in aids}

    def b_system(mode, kind):
        lo = [{"id": t, "author": author_of[t], "scores": b_scores(mode, kind, t, {t})} for t in train_ids]
        ho = [{"id": h, "author": author_of[h], "scores": b_scores(mode, kind, h, set())} for h in held_ids]
        return pack(lo, ho, works, aids)

    fu = {"A_fisher55_windows": systems["A_fisher55_windows"], "B_ppm_skeleton": systems["B_ppm_skeleton"]}
    chk = b_system("surface", "plain")
    assert chk["lowo"]["n_correct"] == fu["B_ppm_skeleton"]["lowo"]["n_correct"]
    assert chk["heldout"]["n_correct"] == fu["B_ppm_skeleton"]["heldout"]["n_correct"]
    fu["B_core"] = b_system("core", "plain")
    fu["B_collapsed"] = b_system("collapsed", "plain")
    fu["B_cal_center"] = b_system("surface", "center")
    fu["B_cal_z"] = b_system("surface", "z")
    fu["B_collapsed_cal_center"] = b_system("collapsed", "center")

    def eq_rows(ids, fold_of):
        return [{"id": i, "author": author_of[i],
                 "scores": {a: -ppm["surface"]["equalized"][fold_of(i)]["h"][i][a] for a in aids}} for i in ids]
    fu["B_equalized"] = pack(eq_rows(train_ids, lambda i: "lowo:" + i), eq_rows(held_ids, lambda i: "heldout"), works, aids)

    def fold_name(excl):
        return "|".join(sorted(excl)) or "heldout"

    # Fisher models per training set (train minus excl); excl has 0, 1 or 2 works.
    folds, a_scores = [], {}          # a_scores[(fold name, q)] = scores

    def add_fold(excl, queries):
        m = fit_fisher({a: np.concatenate([X[i] for i in train_ids if i not in excl and author_of[i] == a])
                        for a in aids}, reg=REG)
        for q in queries:
            a_scores[(fold_name(excl), q)] = m.scores(X[q])
        folds.append({"name": fold_name(excl), "train": [i for i in train_ids if i not in excl], "query": list(queries),
                      "projection": {"center": m.center.tolist(), "scale": m.scale.tolist(), "W": m.W.tolist()}})

    add_fold(set(), held_ids)
    for t in train_ids:
        add_fold({t}, [t])
    for i, t in enumerate(train_ids):
        for w in train_ids[i + 1:]:
            add_fold({t, w}, [t, w])
    print("fisher fits:", len(folds), "(%.0fs)" % (time.time() - t0))
    sw_out = run_rust("sw2", {"clouds": [{"id": w["id"], "author": w["author_id"], "points": X[w["id"]].tolist()}
                                         for w in works], "folds": folds, "k": SW_K, "seed": SEED}, "sw2_fisher9", reuse)
    d9 = {(r["fold"], r["query"]): r for r in sw_out}

    def d9_scores(q, excl, mode="pool"):
        r = d9[(fold_name(excl), q)]
        if mode == "pool":
            return {a: -r["to_author_pool"][a] for a in aids}
        return {a: -min(d for k, d in r["to_work"].items() if author_of[k] == a) for a in aids}

    for mode, name in (("pool", "D9_sw2_author_pool"), ("work", "D9_sw2_nearest_work")):
        fu[name] = pack([{"id": t, "author": author_of[t], "scores": d9_scores(t, {t}, mode)} for t in train_ids],
                        [{"id": h, "author": author_of[h], "scores": d9_scores(h, set(), mode)} for h in held_ids],
                        works, aids)
    # sanity: the per-fold Fisher scores reproduce system A
    chkA = pack([{"id": t, "author": author_of[t], "scores": a_scores[(fold_name({t}), t)]} for t in train_ids],
                [{"id": h, "author": author_of[h], "scores": a_scores[("heldout", h)]} for h in held_ids], works, aids)
    assert (chkA["lowo"]["n_correct"], chkA["heldout"]["n_correct"]) == (
        fu["A_fisher55_windows"]["lowo"]["n_correct"], fu["A_fisher55_windows"]["heldout"]["n_correct"])

    def triple(q, excl):
        """z-scored author scores of the three base systems for work q, models trained on train minus excl."""
        return [zscore_scores(a_scores[(fold_name(excl), q)]), zscore_scores(b_scores("surface", "center", q, excl)),
                zscore_scores(d9_scores(q, excl))]

    def fit_weights(samples):
        """samples: list of (triple, true author).  Non-negative conditional-logit weights."""
        Z = np.array([[[z[a] for a in aids] for z in tr] for tr, _ in samples])   # (n, 3, 10)
        y = np.array([aids.index(a) for _, a in samples])

        def nll(wv):
            logits = np.tensordot(wv, Z, axes=(0, 1))                                  # (n, 10)
            logits = logits - logits.max(axis=1, keepdims=True)
            lp = logits[np.arange(len(y)), y] - np.log(np.exp(logits).sum(axis=1))
            return -lp.sum() + STACK_L2 * float(np.sum(wv ** 2))
        return minimize(nll, np.ones(3), method="L-BFGS-B", bounds=[(0, None)] * 3).x

    def combine(tr, wv):
        return {a: float(sum(wk * z[a] for wk, z in zip(wv, tr))) for a in aids}

    def fixed(wv):
        return pack([{"id": t, "author": author_of[t], "scores": combine(triple(t, {t}), wv)} for t in train_ids],
                    [{"id": h, "author": author_of[h], "scores": combine(triple(h, set()), wv)} for h in held_ids],
                    works, aids)

    fu["E3_equal_A_Bcal_D9"] = fixed(np.array([1.0, 1.0, 1.0]))
    fu["E2_equal_A_Bcal"] = fixed(np.array([1.0, 1.0, 0.0]))
    fu["E2_equal_A_D9"] = fixed(np.array([1.0, 0.0, 1.0]))
    fu["E2_equal_Bcal_D9"] = fixed(np.array([0.0, 1.0, 1.0]))
    # stacking: LOWO uses double exclusion for the inner scores, held-out uses the plain LOWO scores of the 46
    st_lo, weights_lowo = [], []
    for t in train_ids:
        wv = fit_weights([(triple(w, {t, w}), author_of[w]) for w in train_ids if w != t])
        weights_lowo.append(wv)
        st_lo.append({"id": t, "author": author_of[t], "scores": combine(triple(t, {t}), wv)})
    w_held = fit_weights([(triple(w, {w}), author_of[w]) for w in train_ids])
    st_ho = [{"id": h, "author": author_of[h], "scores": combine(triple(h, set()), w_held)} for h in held_ids]
    fu["E3_stacked_A_Bcal_D9"] = pack(st_lo, st_ho, works, aids)
    WL = np.array(weights_lowo)

    def absorbed(sysres):
        """How many works of each non-Soseki author are predicted as Soseki."""
        res = {}
        for part in ("lowo", "heldout"):
            conf = sysres[part]["confusion"]
            res[part] = {"non_soseki_predicted_soseki": sum(conf[a][SOSEKI] for a in aids if a != SOSEKI),
                         "kajii": conf[KAJII][SOSEKI], "akutagawa": conf[AKUTAGAWA][SOSEKI],
                         "soseki_recall": "%d/%d" % (conf[SOSEKI][SOSEKI], sum(conf[SOSEKI].values()))}
        return res

    tok = {a: int(sum(ppm["surface"]["tokens"][i] for i in train_ids if author_of[i] == a)) for a in aids}
    _, mu_full, sd_full = ppm_fold("surface", held_ids[0], set())
    out = {
        "schema": "fable_v5_followup_v1",
        "protocol": {"stack_l2": STACK_L2, "ppm_order": PPM_ORDER, "sw_k": SW_K, "seed": SEED, "reg": REG,
                     "n_sw2_folds": len(folds)},
        "labels": labels,
        "skeleton_vocab": {m: ppm[m]["vocab"] for m in ppm},
        "equalized_n_min_heldout": ppm["surface"]["equalized"]["heldout"]["n_min"],
        "train_tokens_by_author": tok,
        "calibration_full_models": {a: {"mu": round(mu_full[a], 4), "sd": round(sd_full[a], 4)} for a in aids},
        "stack_weights": {"order": ["A", "B_cal_center", "D9_sw2_author_pool"],
                          "heldout": [round(float(x), 4) for x in w_held],
                          "lowo_mean": [round(float(x), 4) for x in WL.mean(axis=0)],
                          "lowo_min": [round(float(x), 4) for x in WL.min(axis=0)],
                          "lowo_max": [round(float(x), 4) for x in WL.max(axis=0)]},
        "systems": {k: {p: v[p] for p in ("lowo", "heldout", "heldout1", "heldout2")} for k, v in fu.items()},
        "soseki_absorption": {k: absorbed(v) for k, v in fu.items() if k.startswith("B_")},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    FOLLOWUP_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["| system | LOWO (46) | 95% CI | held-out (60) | 95% CI | set1 (30) | set2 (30) |", "|---|---|---|---|---|---|---|"]
    for k, v in out["systems"].items():
        lo, ho = v["lowo"], v["heldout"]
        L.append("| %s | %d/%d = %.3f | %.2f–%.2f | %d/%d = %.3f | %.2f–%.2f | %d/%d | %d/%d |" % (
            k, lo["n_correct"], lo["n"], lo["accuracy"], lo["wilson95"][0], lo["wilson95"][1],
            ho["n_correct"], ho["n"], ho["accuracy"], ho["wilson95"][0], ho["wilson95"][1],
            v["heldout1"]["n_correct"], v["heldout1"]["n"], v["heldout2"]["n_correct"], v["heldout2"]["n"]))
    L += ["", "### absorption into Soseki (system B variants)", "",
          "| system | LOWO: non-Soseki works predicted Soseki (of 41) | Kajii to Soseki (of 5) | Akutagawa to Soseki (of 5) "
          "| Soseki recall | held-out: non-Soseki predicted Soseki (of 54) | Kajii to Soseki (of 6) "
          "| Akutagawa to Soseki (of 6) | Soseki recall |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in out["soseki_absorption"].items():
        lo, ho = v["lowo"], v["heldout"]
        L.append("| %s | %d | %d | %d | %s | %d | %d | %d | %s |" % (
            k, lo["non_soseki_predicted_soseki"], lo["kajii"], lo["akutagawa"], lo["soseki_recall"],
            ho["non_soseki_predicted_soseki"], ho["kajii"], ho["akutagawa"], ho["soseki_recall"]))
    L += ["", "### training tokens and calibration offsets (full author models, surface skeleton)", "",
          "| author | training tokens | mu_a (bit/token; mean cross-entropy of the other authors' training works) | sd_a |",
          "|---|---|---|---|"]
    for a in sorted(aids, key=lambda a: tok[a]):
        L.append("| %s | %d | %.3f | %.3f |" % (labels[a], tok[a], mu_full[a], sd_full[a]))
    L += ["", "### per author (LOWO / held-out)", "", "| system | " + " | ".join(labels[a] for a in aids) + " |",
          "|---|" + "---|" * len(aids)]
    for k, v in out["systems"].items():
        L.append("| " + k + " | " + " | ".join(v["lowo"]["per_author"][a] + " / " + v["heldout"]["per_author"][a]
                                               for a in aids) + " |")
    for k in ("B_collapsed", "B_cal_center", "B_equalized", "D9_sw2_author_pool", "E3_equal_A_Bcal_D9",
              "E3_stacked_A_Bcal_D9"):
        for part in ("lowo", "heldout"):
            conf = out["systems"][k][part]["confusion"]
            L += ["", "### confusion: %s / %s (rows = truth, columns = predicted)" % (k, part), "",
                  "| truth / pred | " + " | ".join(labels[a] for a in aids) + " |", "|---|" + "---|" * len(aids)]
            for a in aids:
                L.append("| " + labels[a] + " | " + " | ".join(str(conf[a][b]) for b in aids) + " |")
    FOLLOWUP_TABLES.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("---- follow-up ----")
    for k, v in out["systems"].items():
        print("%-32s LOWO %2d/46  held-out %2d/60 (set1 %d/30, set2 %d/30)" % (
            k, v["lowo"]["n_correct"], v["heldout"]["n_correct"], v["heldout1"]["n_correct"], v["heldout2"]["n_correct"]))
    print("stack weights", out["stack_weights"])
    print("wrote", FOLLOWUP_OUT, "and", FOLLOWUP_TABLES, "(%.1fs)" % out["elapsed_sec"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse Rust outputs in experiments/_fable_v5_work if present")
    args = ap.parse_args()
    t0 = time.time()
    if not EXE.exists():
        raise SystemExit(f"{EXE} not found: cargo build --release -p nishiki-hardcore-cli (in {NISHIKI})")

    works = load_works()
    train = [w for w in works if w["split"] == "train"]
    held = [w for w in works if w["split"] != "train"]
    aids = sorted({w["author_id"] for w in train})
    labels = {w["author_id"]: w["author_ja"] for w in train}
    assert (len(train), len(held), len(aids)) == (46, 60, 10), (len(train), len(held), len(aids))
    assert all(w["author_id"] in aids for w in held)
    print(f"works: train {len(train)}, held-out {len(held)}, authors {len(aids)}, "
          f"chars {sum(len(w['text']) for w in works)}")

    X, n_recomputed = window_features(works)
    print(f"55-dim windows: {sum(len(v) for v in X.values())} ({n_recomputed} works had no cached z24)")
    systems = {}

    # ---- A: window-level Fisher (Phase 36 primary) ------------------------------------------------------
    def fisher_scores(train_ws, test_w):
        model = fit_fisher({a: np.concatenate([X[w["id"]] for w in train_ws if w["author_id"] == a]) for a in aids},
                           reg=REG)
        return model, model.scores(X[test_w["id"]]) if test_w is not None else None

    lowo_rows = []
    for t in train:
        _, s = fisher_scores([w for w in train if w["id"] != t["id"]], t)
        lowo_rows.append({"id": t["id"], "author": t["author_id"], "scores": s})
    full_model, _ = fisher_scores(train, None)
    held_rows = [{"id": w["id"], "author": w["author_id"], "scores": full_model.scores(X[w["id"]])} for w in held]
    systems["A_fisher55_windows"] = pack(lowo_rows, held_rows, works, aids)

    # ---- A2: work-mean 55-dim + LDA (reference) ----------------------------------------------------------
    mean55 = {w["id"]: X[w["id"]].mean(axis=0) for w in works}
    systems["A2_mean55_lda"] = pack(*eval_feature_classifier("lda", mean55, works, aids), works, aids)

    # ---- Rust: text job (features + skeleton PPM cross-entropy) ------------------------------------------
    text_out = run_rust("text", {
        "works": [{"id": w["id"], "path": str(w["path"]), "author": w["author_id"],
                   "split": "train" if w["split"] == "train" else "heldout"} for w in works],
        "ppm_order": PPM_ORDER, "surrogates": SURROGATES, "seed": SEED, "target_rr": RQA_TARGET_RR,
    }, "text", args.reuse)
    rust = {r["id"]: r for r in text_out["works"]}
    assert set(rust) == {w["id"] for w in works} and text_out["authors"] == aids

    # ---- B: skeleton PPM likelihood ratio -----------------------------------------------------------------
    def b_rows(ws):
        return [{"id": w["id"], "author": w["author_id"],
                 "scores": {a: -rust[w["id"]]["skeleton_cross_entropy"][a] for a in aids}} for w in ws]
    systems["B_ppm_skeleton"] = pack(b_rows(train), b_rows(held), works, aids)

    # ---- C: fractal + RQA work features --------------------------------------------------------------------
    def fmat(names):
        return {w["id"]: np.array([np.nan if rust[w["id"]]["features"][n] is None else rust[w["id"]]["features"][n]
                                   for n in names], dtype=float) for w in works}
    featC = fmat(C_FEATURES)
    for kind in ("logreg", "lda", "knn"):
        systems[f"C_fractal_rqa_{kind}"] = pack(*eval_feature_classifier(kind, featC, works, aids), works, aids)
    for g, names in C_GROUPS.items():
        systems[f"C_group_{g}_logreg"] = pack(*eval_feature_classifier("logreg", fmat(names), works, aids), works, aids)
    # length-only control
    featL = {w["id"]: np.array([math.log(rust[w["id"]]["features"]["n_chars"])]) for w in works}
    systems["L_length_only_logreg"] = pack(*eval_feature_classifier("logreg", featL, works, aids), works, aids)

    # ---- D: sliced Wasserstein between window clouds -------------------------------------------------------
    folds = [{"name": f"lowo:{t['id']}", "train": [w["id"] for w in train if w["id"] != t["id"]], "query": [t["id"]]}
             for t in train]
    folds.append({"name": "heldout", "train": [w["id"] for w in train], "query": [w["id"] for w in held]})
    sw_out = run_rust("sw2", {
        "clouds": [{"id": w["id"], "author": w["author_id"], "points": X[w["id"]].tolist()} for w in works],
        "folds": folds, "k": SW_K, "seed": SEED, "sinkhorn_eps": SINKHORN_EPS_PER_DIM * 55,
        "sinkhorn_max_points": SINKHORN_POINTS, "sinkhorn_max_iters": SINKHORN_ITERS,
    }, "sw2", args.reuse)
    sw = {r["query"]: r for r in sw_out}
    author_of = {w["id"]: w["author_id"] for w in works}
    assert set(sw) == {w["id"] for w in works}
    for w in train:  # leak check: a LOWO query never appears among its own references
        assert w["id"] not in sw[w["id"]]["to_work"]

    def d_rows(ws, mode):
        rows = []
        for w in ws:
            r = sw[w["id"]]
            if mode == "pool":
                s = {a: -r["to_author_pool"][a] for a in aids}
            elif mode == "sinkhorn":
                s = {a: -r["sinkhorn_to_author_pool"][a] for a in aids}
            else:  # nearest work of each author
                s = {a: -min(d for k, d in r["to_work"].items() if author_of[k] == a) for a in aids}
            rows.append({"id": w["id"], "author": w["author_id"], "scores": s})
        return rows
    systems["D_sw2_author_pool"] = pack(d_rows(train, "pool"), d_rows(held, "pool"), works, aids)
    systems["D_sw2_nearest_work"] = pack(d_rows(train, "work"), d_rows(held, "work"), works, aids)
    systems["D_sinkhorn_author_pool"] = pack(d_rows(train, "sinkhorn"), d_rows(held, "sinkhorn"), works, aids)
    sink_nc = sum(r["sinkhorn_not_converged"] for r in sw_out)

    # ---- E: equal-weight ensembles ---------------------------------------------------------------------------
    A, B, C, D = (systems[k] for k in ("A_fisher55_windows", "B_ppm_skeleton", "C_fractal_rqa_logreg",
                                         "D_sw2_author_pool"))
    systems["E_ensemble_ABCD"] = ensemble([A, B, C, D], works, aids)
    systems["E_new_BCD"] = ensemble([B, C, D], works, aids)
    systems["E_AB"] = ensemble([A, B], works, aids)

    # ---- orthogonality against the 55-dim baseline -----------------------------------------------------------
    ortho = {}
    all_ids = [w["id"] for w in works]
    train_ids = [w["id"] for w in train]

    def margins(sysname):
        m = {}
        for part in ("lowo", "heldout"):
            for r in systems[sysname][part]["rows"]:
                m[r["id"]] = (r["margin_z"], r["correct"])
        return m
    mA = margins("A_fisher55_windows")
    for name in ("B_ppm_skeleton", "C_fractal_rqa_logreg", "D_sw2_author_pool", "E_ensemble_ABCD", "E_new_BCD"):
        mX = margins(name)
        rho, p = spearmanr([mA[i][0] for i in all_ids], [mX[i][0] for i in all_ids])
        ortho[name] = {
            "spearman_margin_vs_A": round(float(rho), 4), "p_value": float(p), "n": len(all_ids),
            "agreement": {
                "both_correct": sum(mA[i][1] and mX[i][1] for i in all_ids),
                "only_A_correct": sum(mA[i][1] and not mX[i][1] for i in all_ids),
                "only_this_correct": sum((not mA[i][1]) and mX[i][1] for i in all_ids),
                "both_wrong": sum((not mA[i][1]) and (not mX[i][1]) for i in all_ids)}}
    # feature-level: each new work feature against the 55 work-mean baseline features (train works only)
    M55 = np.array([mean55[i] for i in train_ids])
    feat_corr = {}
    for j, n in enumerate(C_FEATURES):
        v = np.array([featC[i][j] for i in train_ids])
        ok = np.isfinite(v)
        best = (0.0, None)
        for k in range(55):
            if ok.sum() > 5 and np.std(M55[ok, k]) > 0 and np.std(v[ok]) > 0:
                rho = spearmanr(v[ok], M55[ok, k])[0]
                if abs(rho) > abs(best[0]):
                    best = (float(rho), FEATURE_NAMES_55[k])
        rl = spearmanr(v[ok], [math.log(rust[i]["features"]["n_chars"]) for i, o in zip(train_ids, ok) if o])[0]
        feat_corr[n] = {"max_abs_spearman_vs_55": round(best[0], 4), "with": best[1],
                        "spearman_vs_log_n_chars": round(float(rl), 4), "n_missing_all_works": int(sum(
                            rust[i]["features"][n] is None for i in all_ids))}
    absr = [abs(v["max_abs_spearman_vs_55"]) for v in feat_corr.values()]
    ortho["feature_level"] = {"per_feature": feat_corr, "median_max_abs_rho": round(float(np.median(absr)), 4),
                              "n_features_max_abs_rho_below_0.5": int(sum(r < 0.5 for r in absr)), "n_features": len(absr)}

    # ---- write -------------------------------------------------------------------------------------------------
    feat_summary = {}
    for n in list(C_FEATURES) + ["sp_min_r2", "sl_min_r2", "n_chars", "n_sentences", "n_skeleton_tokens"]:
        v = np.array([rust[i]["features"][n] for i in all_ids if rust[i]["features"][n] is not None], dtype=float)
        feat_summary[n] = {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
    out = {
        "schema": "fable_v5_grand_shootout_v1",
        "protocol": {"reg": REG, "knn_k": KNN_K, "logreg_C": LOGREG_C, "ppm_order": PPM_ORDER, "sw_k": SW_K,
                     "seed": SEED, "rqa_target_rr": RQA_TARGET_RR, "surrogates": SURROGATES,
                     "sinkhorn_eps": SINKHORN_EPS_PER_DIM * 55, "sinkhorn_points": SINKHORN_POINTS,
                     "sinkhorn_iters": SINKHORN_ITERS, "c_features": C_FEATURES},
        "corpus": {"n_train": len(train), "n_heldout1": sum(w["split"] == "heldout1" for w in works),
                   "n_heldout2": sum(w["split"] == "heldout2" for w in works),
                   "chars_total": sum(len(w["text"]) for w in works),
                   "chars_train": sum(len(w["text"]) for w in train),
                   "windows_total": int(sum(len(v) for v in X.values())), "z24_recomputed_works": n_recomputed},
        "labels": labels, "chance": 1.0 / len(aids),
        "rust": {"ppm_order": text_out["ppm_order"], "alphabet": text_out["alphabet"],
                 "skeleton_vocab": text_out["skeleton_vocab"], "skeleton_overflow": text_out["skeleton_overflow"],
                 "rqa_subsampled_works": sum(r["rqa_subsampled"] for r in text_out["works"]),
                 "sinkhorn_not_converged_pairs": sink_nc, "sinkhorn_pairs": len(sw_out) * len(aids)},
        "feature_summary": feat_summary,
        "systems": {k: {p: v[p] for p in ("lowo", "heldout", "heldout1", "heldout2")} for k, v in systems.items()},
        "orthogonality": ortho,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # markdown tables (numbers are copied from `out`, never retyped)
    L = ["| system | LOWO (46) | 95% CI | held-out (60) | 95% CI | set1 (30) | set2 (30) | LOWO top-3 | held-out top-3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in out["systems"].items():
        lo, ho = v["lowo"], v["heldout"]
        L.append(f"| {k} | {lo['n_correct']}/{lo['n']} = {lo['accuracy']:.3f} | {lo['wilson95'][0]:.2f}–{lo['wilson95'][1]:.2f} "
                 f"| {ho['n_correct']}/{ho['n']} = {ho['accuracy']:.3f} | {ho['wilson95'][0]:.2f}–{ho['wilson95'][1]:.2f} "
                 f"| {v['heldout1']['n_correct']}/{v['heldout1']['n']} | {v['heldout2']['n_correct']}/{v['heldout2']['n']} "
                 f"| {lo['top3_accuracy']:.3f} | {ho['top3_accuracy']:.3f} |")
    L += ["", "### per author (LOWO / held-out)", "", "| system | " + " | ".join(labels[a] for a in aids) + " |",
          "|---|" + "---|" * len(aids)]
    for k, v in out["systems"].items():
        L.append(f"| {k} | " + " | ".join(f"{v['lowo']['per_author'][a]} / {v['heldout']['per_author'][a]}" for a in aids) + " |")
    for k in ("A_fisher55_windows", "B_ppm_skeleton", "C_fractal_rqa_logreg", "D_sw2_author_pool", "E_ensemble_ABCD"):
        for part in ("lowo", "heldout"):
            conf = out["systems"][k][part]["confusion"]
            L += ["", f"### confusion: {k} / {part} (rows = truth, columns = predicted)", "",
                  "| truth \\ pred | " + " | ".join(labels[a] for a in aids) + " |", "|---|" + "---|" * len(aids)]
            for a in aids:
                L.append(f"| {labels[a]} | " + " | ".join(str(conf[a][b]) for b in aids) + " |")
    L += ["", "### orthogonality vs A (106 works)", "",
          "| system | Spearman of margin vs A | p | both correct | only A | only this | both wrong |", "|---|---|---|---|---|---|---|"]
    for k, v in ortho.items():
        if k != "feature_level":
            g = v["agreement"]
            L.append(f"| {k} | {v['spearman_margin_vs_A']:.3f} | {v['p_value']:.2g} | {g['both_correct']} | {g['only_A_correct']} "
                     f"| {g['only_this_correct']} | {g['both_wrong']} |")
    L += ["", "### system C features (46 train works): strongest Spearman with any of the 55 work-mean features, and with length", "",
          "| feature | max abs rho vs 55 | with | rho vs log n_chars | missing (of 106) | median (106) |", "|---|---|---|---|---|---|"]
    for n, v in feat_corr.items():
        L.append(f"| {n} | {v['max_abs_spearman_vs_55']:.3f} | {v['with']} | {v['spearman_vs_log_n_chars']:.3f} "
                 f"| {v['n_missing_all_works']} | {feat_summary[n]['median']:.4g} |")
    TABLES.write_text("\n".join(L) + "\n", encoding="utf-8")

    for k, v in out["systems"].items():
        print(f"{k:32s} LOWO {v['lowo']['n_correct']:2d}/46  held-out {v['heldout']['n_correct']:2d}/60 "
              f"(set1 {v['heldout1']['n_correct']}/30, set2 {v['heldout2']['n_correct']}/30)")
    print("wrote", OUT, "and", TABLES, f"({out['elapsed_sec']}s)")
    followup(works, X, aids, labels, systems, args.reuse)


if __name__ == "__main__":
    main()
