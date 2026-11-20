import json
from common import *
G = load_gold()
runs = {"baseline": load_preds(S2 / "annotations")[0], "control": load_preds(HERE / "results/run_control")[0], "values_ctx": load_preds(HERE / "results/run_values_ctx")[0]}
out = {}
for name, P in runs.items():
    d = dict(gold_char_scenes=0, matched=0, gold_nz_matched=0, same=0, pred0=0, opp=0, gold_nz_all=0, same_over_all_gold_nz=0, pred_nz=0, pred_nz_on_goldzero=0, allzero_vectors=0)
    for sid, g in G.items():
        pst = {s["character_id"]: s["values_6d"] for s in P[sid]["character_states"]}
        for cid, gv in g["states"].items():
            d["gold_char_scenes"] += 1
            d["gold_nz_all"] += sum(v != 0 for v in gv)
            if cid not in pst: continue
            pv = pst[cid]; d["matched"] += 1
            d["allzero_vectors"] += all(v == 0 for v in pv)
            for x, y in zip(pv, gv):
                if x: d["pred_nz"] += 1
                if x and not y: d["pred_nz_on_goldzero"] += 1
                if y:
                    d["gold_nz_matched"] += 1
                    if x == y: d["same"] += 1; d["same_over_all_gold_nz"] += 1
                    elif x == 0: d["pred0"] += 1
                    else: d["opp"] += 1
    d["same_rate_on_matched"] = round(d["same"] / d["gold_nz_matched"], 3)
    d["same_rate_over_all_gold_nz"] = round(d["same"] / d["gold_nz_all"], 3)
    out[name] = d
(HERE / "results/values_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
for k, v in out.items(): print(k, v)
