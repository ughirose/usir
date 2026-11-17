#!/usr/bin/env python3
"""P-2b Tasks B' (R_xx) and C' (perturbation ROC-AUC / FPR) on the Aozora postwar corpus.

All parameters are those pre-registered in docs/handoff/STATUS_p2b.md (登録 2〜5). Output is numeric only (no text).
Raw text is read from data/aozora_cache/ (gitignored) and never written out.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
for p in ("python", "scripts", "experiments"):
    sys.path.insert(0, str(REPO / p))
from scale_separated_level3 import extract_ortho24_kasane8 as FN  # noqa: E402
from usir_py.v5 import split_work_into_scenes  # noqa: E402
from usir_py.extractor_24d import PHYSICAL_SENSE_INTERVALS  # noqa: E402

MANIFEST = REPO / "data" / "aozora_p2b_manifest.json"
OUT = REPO / "scripts" / "p2b" / "p2b_results.json"
SEED = 20261117
SIGMA_FLOOR = 1e-6

# ---- pre-registered perturbation parameters (登録 4)
PRIMARY = {"ending": 1.0, "comma": 0.5, "sensory": 0.5}
SECONDARY = {"ending": [0.5], "comma": [0.25, 1.0], "sensory": [0.25, 1.0]}
SEED_OFFSET = {"ending": 0, "comma": 1, "sensory": 2}
_RE_ENDING = re.compile(r"[ぁ-ゖ]{1,3}。")
_SENSE = set()
for lo, hi in PHYSICAL_SENSE_INTERVALS:
    _SENSE.update(chr(c) for c in range(lo, hi + 1))


def perturb(kind: str, texts: list[str], rate: float) -> list[str]:
    rng = np.random.default_rng(SEED + SEED_OFFSET[kind])
    out = []
    for t in texts:
        if kind == "ending":
            out.append(_RE_ENDING.sub(lambda m: ("である。" if rng.random() < rate else m.group(0)), t))
        elif kind == "comma":
            pieces = []
            for ch in t:
                if ch == "、" and rng.random() < rate:
                    continue
                pieces.append(ch)
            out.append("".join(pieces))
        else:
            pieces = []
            for ch in t:
                if ch in _SENSE and rng.random() < rate:
                    continue
                pieces.append(ch)
            out.append("".join(pieces))
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats(x):
    mu = x.mean(0)
    sd = np.maximum(x.std(0, ddof=1), SIGMA_FLOOR)
    return mu, sd


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def summarize(v):
    v = np.asarray(v, float)
    q = np.percentile(v, [25, 50, 75])
    return {"n": len(v), "mean": float(v.mean()), "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]),
            "iqr": float(q[2] - q[0]), "min": float(v.min()), "max": float(v.max())}


def main():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    works = man["works"]
    rec = []  # per scene: aid, wid, order
    texts = []
    for w in works:  # manifest is sorted by (author_id, work_id)
        t = (REPO / "data" / "aozora_cache" / f"p2b_{w['author_id']}_{w['work_id']}.txt").read_text(encoding="utf-8")
        for s in split_work_into_scenes(t):
            rec.append((w["author_id"], w["work_id"]))
            texts.append(s)
    aids = np.array([r[0] for r in rec]); wids = np.array([r[1] for r in rec])
    X = np.stack([FN(t) for t in texts])
    n = len(X)
    authors = sorted(set(aids))
    name = {w["author_id"]: w["author_ja"] for w in works}
    res = {"seed": SEED, "n_scenes": int(n), "n_authors": len(authors), "n_works": len(set(wids)), "feature_dim": int(X.shape[1])}

    # ---------------- Task B': R_xx ----------------
    mu, sd = stats(X)
    Z = (X - mu) / sd

    def rxx_parity(Zs):
        return cos(Zs[0::2].mean(0), Zs[1::2].mean(0))

    per_author = {}
    for a in authors:
        idx = np.where(aids == a)[0]
        wl = sorted(set(wids[idx]))
        d = {"author": name[a], "n_scenes": int(len(idx)), "n_works": len(wl), "R_xx": rxx_parity(Z[idx])}
        # sensitivity: leave-author-out normalization
        oth = aids != a
        mu_o, sd_o = stats(X[oth])
        d["R_xx_leave_author_out_norm"] = rxx_parity((X[idx] - mu_o) / sd_o)
        # secondary: split by work (alternate works)
        if len(wl) >= 2:
            ha = np.isin(wids[idx], wl[0::2]); hb = np.isin(wids[idx], wl[1::2])
            d["R_xx_work_split"] = cos(Z[idx][ha].mean(0), Z[idx][hb].mean(0))
        per_author[a] = d
    vals = [d["R_xx"] for d in per_author.values()]
    # null: permute author labels over scenes, keep sizes
    g = np.random.default_rng(11)
    sizes = [int((aids == a).sum()) for a in authors]
    null_all, null_med = [], []
    for _ in range(2000):
        perm = g.permutation(n)
        pos, vv = 0, []
        for s in sizes:
            ii = np.sort(perm[pos:pos + s]); pos += s
            vv.append(rxx_parity(Z[ii]))
        null_all.extend(vv); null_med.append(float(np.median(vv)))
    res["B"] = {
        "per_author": per_author,
        "R_xx_summary": summarize(vals),
        "R_xx_leave_author_out_summary": summarize([d["R_xx_leave_author_out_norm"] for d in per_author.values()]),
        "R_xx_work_split_summary": summarize([d["R_xx_work_split"] for d in per_author.values() if "R_xx_work_split" in d]),
        "null_label_permutation": {"n_perm": 2000, "seed": 11, "per_author_value_quantiles": {
            "p50": float(np.percentile(null_all, 50)), "p95": float(np.percentile(null_all, 95)), "p99": float(np.percentile(null_all, 99))},
            "median_over_authors_p95": float(np.percentile(null_med, 95)), "median_over_authors_p50": float(np.percentile(null_med, 50))},
        "authors_with_R_xx_ge_0.70": int(sum(v >= 0.70 for v in vals)), "authors_with_R_xx_ge_0.80": int(sum(v >= 0.80 for v in vals)),
    }

    # ---------------- Task C': perturbation ROC-AUC ----------------
    def feats(tx):
        return np.stack([FN(t) for t in tx])

    def score(Xs, mu_, sd_):
        return np.linalg.norm((Xs - mu_) / sd_, axis=1)

    def run_levels(levels):
        P = {k: feats(perturb(k, texts, r)) for k, r in levels.items()}
        return P

    uw = sorted(set(wids))
    wid_index = {w: i for i, w in enumerate(uw)}
    work_of = np.array([wid_index[w] for w in wids])
    rngb = np.random.default_rng(7)
    boots = [rngb.integers(0, len(uw), len(uw)) for _ in range(2000)]
    members = [np.where(work_of == i)[0] for i in range(len(uw))]

    def boot_ci(s_neg, s_pos_by_kind):
        """Work-cluster bootstrap CI of AUC (scores fixed). s_pos_by_kind: kind -> scores aligned to scenes."""
        out = {k: [] for k in list(s_pos_by_kind) + ["pooled"]}
        for pick in boots:
            ii = np.concatenate([members[i] for i in pick])
            neg = s_neg[ii]
            pools = []
            for k, sp in s_pos_by_kind.items():
                out[k].append(auc(sp[ii], neg)); pools.append(sp[ii])
            out["pooled"].append(auc(np.concatenate(pools), neg))
        return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}

    def insample(levels, with_ci):
        P = run_levels(levels)
        s_neg = score(X, mu, sd)
        thr = float(np.percentile(s_neg, 90))
        s_pos = {k: score(P[k], mu, sd) for k in P}
        r = {"threshold_p90": thr, "fpr_at_op": float((s_neg >= thr).mean()), "per_type": {}}
        for k, sp in s_pos.items():
            r["per_type"][k] = {"auc": auc(sp, s_neg), "tpr_at_op": float((sp >= thr).mean()), "rate": levels[k]}
        allpos = np.concatenate(list(s_pos.values()))
        r["pooled"] = {"auc": auc(allpos, s_neg), "tpr_at_op": float((allpos >= thr).mean())}
        if with_ci:
            ci = boot_ci(s_neg, s_pos)
            for k in r["per_type"]:
                r["per_type"][k]["auc_ci95_work_cluster"] = ci[k]
            r["pooled"]["auc_ci95_work_cluster"] = ci["pooled"]
        return r, P

    primary, Pprim = insample(PRIMARY, True)
    res["C_primary_insample"] = {"levels": PRIMARY, **primary,
                                 "note": "FPR at the in-sample 90th percentile is ~10% by construction (circular)."}

    # secondary dose levels (in-sample, no CI)
    dose = {}
    for k, rates in SECONDARY.items():
        for rt in rates:
            lv = dict(PRIMARY); lv = {k: rt}
            r, _ = insample(lv, False)
            dose[f"{k}@{rt}"] = {"auc": r["per_type"][k]["auc"], "tpr_at_op": r["per_type"][k]["tpr_at_op"]}
    for k, rt in [("comma", 1.0), ("sensory", 1.0)]:
        pass
    res["C_secondary_dose_response_insample"] = dose

    # cross-fit by work (5 folds, seed 11)
    g2 = np.random.default_rng(11)
    perm = g2.permutation(len(uw))
    fold_of_work = np.empty(len(uw), int)
    for i, p in enumerate(perm):
        fold_of_work[p] = i % 5
    fold = fold_of_work[work_of]
    neg_n, pos_n = [], {k: [] for k in PRIMARY}
    fold_rows, fpr_hits, fpr_tot, widx_all = [], 0, 0, []
    tpr_hits = {k: 0 for k in PRIMARY}
    per_fold_auc = []
    for f in range(5):
        tr, te = fold != f, fold == f
        mu_f, sd_f = stats(X[tr])
        thr_f = float(np.percentile(score(X[tr], mu_f, sd_f), 90))
        sn = score(X[te], mu_f, sd_f)
        fpr_hits += int((sn >= thr_f).sum()); fpr_tot += int(te.sum())
        neg_n.append((sn / thr_f, np.where(te)[0]))
        fa = []
        for k in PRIMARY:
            sp = score(Pprim[k][te], mu_f, sd_f)
            pos_n[k].append(sp / thr_f); tpr_hits[k] += int((sp >= thr_f).sum())
            fa.append(auc(sp, sn))
        per_fold_auc.append({"fold": f, "n_scenes": int(te.sum()), "fpr": float((sn >= thr_f).mean()), "mean_auc_over_types": float(np.mean(fa))})
    neg_scores = np.empty(n); pos_scores = {k: np.empty(n) for k in PRIMARY}
    for f in range(5):
        te_idx = np.where(fold == f)[0]
        neg_scores[te_idx] = neg_n[f][0]
        for k in PRIMARY:
            pos_scores[k][te_idx] = pos_n[k][f]
    cf = {"folds": 5, "seed": 11, "fpr_heldout": fpr_hits / fpr_tot, "per_fold": per_fold_auc, "per_type": {}}
    for k in PRIMARY:
        cf["per_type"][k] = {"auc": auc(pos_scores[k], neg_scores), "tpr_at_op": tpr_hits[k] / fpr_tot}
    cf["pooled"] = {"auc": auc(np.concatenate(list(pos_scores.values())), neg_scores)}
    ci = boot_ci(neg_scores, pos_scores)
    for k in PRIMARY:
        cf["per_type"][k]["auc_ci95_work_cluster"] = ci[k]
    cf["pooled"]["auc_ci95_work_cluster"] = ci["pooled"]
    cf["note"] = "scores divided by each fold's own threshold before pooling; held-out works only."
    res["C_secondary_crossfit_heldout_works"] = cf

    res["passline"] = {"auc_min": 0.85, "fpr_max": 0.10}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # console summary (numbers only)
    print("n_scenes", n, "authors", len(authors), "works", len(uw))
    print("R_xx", json.dumps(res["B"]["R_xx_summary"]))
    print("null", json.dumps(res["B"]["null_label_permutation"]))
    print("C insample pooled", json.dumps(res["C_primary_insample"]["pooled"]), "fpr", res["C_primary_insample"]["fpr_at_op"])
    for k, v in res["C_primary_insample"]["per_type"].items():
        print("  ", k, json.dumps(v))
    print("crossfit fpr", cf["fpr_heldout"], "pooled", json.dumps(cf["pooled"]))
    for k, v in cf["per_type"].items():
        print("  cf", k, json.dumps(v))
    print("dose", json.dumps(dose))


if __name__ == "__main__":
    main()
