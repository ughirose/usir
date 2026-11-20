"""Compare fact outputs of baseline / control / fact_v2: survival of hand-classified baseline FPs, and which gold facts were matched."""
import json, sys
from collections import Counter
from common import *
sys.path.insert(0, str(S2))
from measure_stage_b import text_match
G = load_gold()
runs = {"baseline": load_preds(S2 / "annotations")[0], "control": load_preds(HERE / "results/run_control")[0], "fact_v2": load_preds(HERE / "results/run_fact_v2")[0]}
cls = json.loads((HERE / "results/fact_error_classification.json").read_text(encoding="utf-8"))["items"]
def key(f): return (f["subject"], f["predicate"])
def present(run, sid, subj, pred, obj):
    return any(f["predicate"] == pred and f["subject"] == subj and text_match(f["object"], obj) for f in runs[run][sid]["fact_updates"])
res = {}
for name in ("control", "fact_v2"):
    surv = Counter(); tot = Counter()
    for it in cls:
        if it["kind"] != "FP": continue
        tot[it["class"]] += 1
        if present(name, it["scene_id"], it["subject"], it["predicate"], it["object"]): surv[it["class"]] += 1
    res[name] = {"baseline_FP_still_emitted_by_class": {c: f"{surv[c]}/{tot[c]}" for c in sorted(tot)}, "total": f"{sum(surv.values())}/{sum(tot.values())}"}
# per-scene counts, empty scenes
for name, P in runs.items():
    cnt = [len(P[s]["fact_updates"]) for s in G]
    res.setdefault(name, {})["facts_per_scene"] = {"mean": round(sum(cnt)/len(cnt), 2), "max": max(cnt), "empty_scenes": sum(c == 0 for c in cnt), "total": sum(cnt)}
    res[name]["pred_by_predicate"] = dict(Counter(f["predicate"] for s in G for f in P[s]["fact_updates"]))
    res[name]["null_subject"] = sum(f["subject"] is None for s in G for f in P[s]["fact_updates"])
# gold facts matched: lenient, by run
for name, P in runs.items():
    m = set()
    for s in G:
        for i, gf in enumerate(G[s]["facts"]):
            if any(f["predicate"] == gf["predicate"] and text_match(f["object"], gf["object"]) for f in P[s]["fact_updates"]): m.add((s, i))
    res[name]["gold_matched_any"] = len(m)
b = {(s, i) for s in G for i, gf in enumerate(G[s]["facts"]) if any(f["predicate"] == gf["predicate"] and text_match(f["object"], gf["object"]) for f in runs["baseline"][s]["fact_updates"])}
v = {(s, i) for s in G for i, gf in enumerate(G[s]["facts"]) if any(f["predicate"] == gf["predicate"] and text_match(f["object"], gf["object"]) for f in runs["fact_v2"][s]["fact_updates"])}
c = {(s, i) for s in G for i, gf in enumerate(G[s]["facts"]) if any(f["predicate"] == gf["predicate"] and text_match(f["object"], gf["object"]) for f in runs["control"][s]["fact_updates"])}
res["gold_matched_overlap"] = {"baseline&control": len(b & c), "baseline|control": len(b | c), "baseline&fact_v2": len(b & v), "control&fact_v2": len(c & v), "fact_v2_only_vs_baseline": len(v - b), "baseline_only_vs_fact_v2": len(b - v)}
# fact_v2 remaining FPs (list)
fps = []
for s in G:
    used = set()
    for f in runs["fact_v2"][s]["fact_updates"]:
        hit = False
        for i, gf in enumerate(G[s]["facts"]):
            if i in used or gf["predicate"] != f["predicate"] or not text_match(f["object"], gf["object"]): continue
            used.add(i); hit = True; break
        if not hit: fps.append({"scene_id": s, "subject": f["subject"], "predicate": f["predicate"], "object": f["object"]})
res["fact_v2_FP_list"] = fps
(HERE / "results/variant_comparison.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "fact_v2_FP_list"}, ensure_ascii=False, indent=1))
for f in fps: print(f["scene_id"], f["subject"], f["predicate"], f["object"])
