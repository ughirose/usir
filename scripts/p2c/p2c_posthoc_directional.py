#!/usr/bin/env python3
"""P-2c post-hoc EXPLORATORY diagnostic (NOT pre-registered; not used for pass/fail judgement).
Question: how much of the weak comma/sensory AUC is due to the registered two-sided |z| surface score?
Replaces the surface score by a one-sided (decrease-only) score, keeping everything else (folds, nested OP-B) identical.
Also reports the normal-theory ceiling AUC = Phi(d/sqrt(2)) for each surface quantity from the measured effect size d."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2c_analysis as P  # noqa: E402  (re-runs the registered analysis; deterministic; rewrites p2c_results.json identically)


def score_dir(fm, Xs, Qs):
    mu, sd, gm, gs, qm, qs = fm
    Z = (Xs - mu) / sd
    s_o = (np.linalg.norm(Z[:, P.ORTHO], axis=1) - gm[0]) / gs[0]
    s_k = (np.linalg.norm(Z[:, P.KASANE], axis=1) - gm[1]) / gs[1]
    qz = (qm - Qs) / qs  # decrease-only
    sub = np.maximum(s_o, s_k)
    surf = qz.max(1)
    return {"composite": np.maximum(sub, surf), "sub": sub, "surf": surf, "ortho": s_o, "kasane": s_k,
            "q1": qz[:, 0], "q2": qz[:, 1], "q3": qz[:, 2]}


P.score = score_dir
all_works = list(range(len(P.uw)))
fo = P.workset_folds(all_works, 11)
neg, pos, dose, tp = P.alloc()
fos = np.array([fo[w] for w in P.WI])
for f in range(5):
    trw = [w for w in all_works if fo[w] != f]
    fm, thrA, thrB = P.thresholds(trw, 11 + f)
    P.evaluate(fos == f, fm, thrA, thrB, neg, pos, dose, tp)
out = {"note": "post-hoc exploratory, decrease-only surface score; not pre-registered", "crossfit": {}}
for name, mask in (("all", np.ones(P.n, bool)), ("P-2b_existing", P.COH == 0), ("new", P.COH == 1)):
    r = P.report(mask, neg, pos, dose, tp)
    out["crossfit"][name] = {v: {"pooled_auc": e["pooled"]["auc"], "fpr_OP_B": e["fpr_OP_B"], "tpr_OP_B": e["pooled"]["tpr_OP_B"],
                                 "per_type_auc": {k: t["auc"] for k, t in e["per_type"].items()}} for v, e in r["variants"].items() if v in ("composite", "surf", "q1", "q2", "q3")}
res = json.loads(P.OUT.read_text(encoding="utf-8"))
es = res["C_diagnostics"]["surface_quantities"]["expected_effect_size_sd_units"]
sc = res["C_diagnostics"]["surface_quantities"]["scene_level_effect_size_sd_units_sd"]
# normal-theory ceiling for a single quantity: d = |mean shift| / sqrt(between-scene var + perturbed var) ~ |shift| / (sd * sqrt(1 + (sd_scene_shift)^2))
out["normal_theory_ceiling_auc"] = {"comma": float(norm.cdf(abs(es["comma"][0]) / np.sqrt(2))), "sensory": float(norm.cdf(abs(es["sensory"][1]) / np.sqrt(2))),
                                    "ending": float(norm.cdf(abs(es["ending"][2]) / np.sqrt(2))),
                                    "note": "AUC=Phi(d/sqrt(2)), d = mean shift in units of between-scene SD; assumes equal-variance normal; a ceiling for the single-quantity score, not a measured value"}
(Path(__file__).resolve().parent / "p2c_posthoc_directional.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for name, d_ in out["crossfit"].items():
    print(name, {v: (round(e["pooled_auc"], 3), round(e["fpr_OP_B"], 3), {k: round(a, 3) for k, a in e["per_type_auc"].items()}) for v, e in d_.items() if v in ("composite", "surf")})
print(out["normal_theory_ceiling_auc"])
