import json, sys
from collections import defaultdict, Counter
from pathlib import Path
from common import *
G = load_gold(); P, T = load_preds(S2 / "annotations")
cs = []  # (sid, wid, cid, gv, pv|None)
for sid, g in G.items():
    pst = {s["character_id"]: s["values_6d"] for s in P[sid]["character_states"]}
    for cid, gv in g["states"].items():
        cs.append((sid, sid.split(":")[0], cid, gv, pst.get(cid)))
# persistence: for each gold nonzero axis, does the same character carry the same sign on that axis in >=1 other scene of the same work?
by = defaultdict(list)
for r in cs: by[(r[1], r[2])].append(r)
persist = tot = 0
for r in cs:
    for ax, y in enumerate(r[3]):
        if y == 0: continue
        tot += 1
        if any(o[0] != r[0] and o[3][ax] == y for o in by[(r[1], r[2])]): persist += 1
# categories of the 66 gold-nonzero axes
cat = Counter(); ax_cat = defaultdict(Counter)
for sid, wid, cid, gv, pv in cs:
    for ax, y in enumerate(gv):
        if y == 0: continue
        if pv is None: k = "character_not_predicted"
        elif pv[ax] == y: k = "same_sign"
        elif pv[ax] == 0: k = "pred_zero_on_axis" + ("(all-zero vector)" if all(v == 0 for v in pv) else "")
        else: k = "opposite_sign"
        cat[k] += 1; ax_cat[ax][k] += 1
# predicted vectors
pvs = [r for r in cs if r[4] is not None]
allzero = sum(all(v == 0 for v in r[4]) for r in pvs)
dense = [(r[0], r[2], r[4]) for r in pvs if sum(v != 0 for v in r[4]) >= 5]
extra_axes = sum(1 for r in pvs for x, y in zip(r[4], r[3]) if x != 0 and y == 0)
nz_pred = sum(1 for r in pvs for x in r[4] if x != 0)
# gold-zero vectors vs pred
res = {"gold_char_scenes": len(cs), "predicted_char_scenes": len(pvs), "not_predicted": len(cs) - len(pvs),
       "gold_nonzero_axes": tot, "gold_nonzero_axes_category": dict(cat),
       "gold_nonzero_axes_persisting_in_other_scenes_of_same_character": {"n": persist, "of": tot},
       "by_axis": {str(a): dict(c) for a, c in sorted(ax_cat.items())},
       "pred_vectors_all_zero": allzero, "pred_dense_vectors_(>=5 nonzero axes)": [{"scene": a, "char": b, "vec": c} for a, b, c in dense],
       "pred_nonzero_axes": nz_pred, "pred_nonzero_on_gold_zero_axes": extra_axes}
(HERE / "results/values_mismatch_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
