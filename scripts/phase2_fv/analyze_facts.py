"""Work 1: aggregate hand classification of every FP / FN (baseline annotations), verify coverage against the metric."""
import json, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0, str(S2))
from measure_stage_b import text_match
import classify_part1, classify_part2
CL = {**classify_part1.PART, **classify_part2.PART}
G = load_gold(); P, T = load_preds(S2 / "annotations")
rows, problems = [], []
for sid in G:
    g, a, t = G[sid], P[sid], T[sid]
    used = set(); res = []
    for f in a["fact_updates"]:
        hit = None
        for i, gf in enumerate(g["facts"]):
            if i in used or gf["predicate"] != f["predicate"] or not text_match(f["object"], gf["object"]): continue
            used.add(i); hit = i; break
        res.append(hit)
    fps = [k for k, h in enumerate(res) if h is None]
    fns = [i for i in range(len(g["facts"])) if i not in used]
    c = CL.get(sid, {"FP": {}, "FN": {}})
    if set(c["FP"]) != set(fps): problems.append((sid, "FP", sorted(set(fps) ^ set(c["FP"]))))
    if set(c["FN"]) != set(fns): problems.append((sid, "FN", sorted(set(fns) ^ set(c["FN"]))))
    for k in fps:
        f = a["fact_updates"][k]; lab = c["FP"].get(k, ("?", ""))
        rows.append({"scene_id": sid, "kind": "FP", "idx": k, "subject": f["subject"], "predicate": f["predicate"], "object": f["object"],
                     "basis": t["facts"][k]["basis"] if k < len(t["facts"]) else None, "class": lab[0], "reason": lab[1], "flags": list(lab[2:])})
    for i in fns:
        gf = g["facts"][i]; lab = c["FN"].get(i, ("?", ""))
        rows.append({"scene_id": sid, "kind": "FN", "idx": i, "subject": gf["subject"], "predicate": gf["predicate"], "object": gf["object"],
                     "class": lab[0], "reason": lab[1], "flags": list(lab[2:])})
if problems: print("COVERAGE PROBLEMS", problems); sys.exit(1)
fp = Counter(r["class"] for r in rows if r["kind"] == "FP"); fn = Counter(r["class"] for r in rows if r["kind"] == "FN")
pred_fp = Counter(r["predicate"] for r in rows if r["kind"] == "FP")
pred_fn = Counter(r["predicate"] for r in rows if r["kind"] == "FN")
flags = Counter(f for r in rows for f in r["flags"])
out = {"source_annotations": "scripts/phase2_stage2/annotations", "n_scenes": len(G),
       "FP_total": sum(fp.values()), "FP_by_class": dict(fp), "FN_total": sum(fn.values()), "FN_by_class": dict(fn),
       "FP_by_predicate": dict(pred_fp), "FN_by_predicate": dict(pred_fn), "flags": dict(flags),
       "class_legend": {"A":"本文に根拠がない作り話","B":"根拠はあるが粒度が違う(細かすぎ・粗すぎ。surf=意味は対応するがobject表層が不一致)","C":"述語の取り違え","D":"主語・対象の取り違え","E":"ゴールド側の漏れの疑い","M":"FNのみ: 対応する予測が場面内に無い(取りこぼし)"},
       "items": rows}
Path(HERE / "results").mkdir(exist_ok=True)
(HERE / "results/fact_error_classification.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("FP", sum(fp.values()), dict(fp)); print("FN", sum(fn.values()), dict(fn)); print(dict(flags)); print("pred FP", dict(pred_fp)); print("pred FN", dict(pred_fn))
