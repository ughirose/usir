import json, sys
from pathlib import Path
from common import *
sys.path.insert(0, str(S2))
from measure_stage_b import text_match
W = load_works(); G = load_gold(); P, T = load_preds(S2 / "annotations")
lines = []
for sid in G:
    wid, si = sid.split(":s"); si = int(si)
    txt = W[wid]["scenes"][si]["scene_text"]
    g, a, t = G[sid], P[sid], T[sid]
    used=set(); res=[]
    for k,f in enumerate(a["fact_updates"]):
        hit=None
        for i,gf in enumerate(g["facts"]):
            if i in used or gf["predicate"]!=f["predicate"] or not text_match(f["object"],gf["object"]): continue
            used.add(i); hit=i; break
        res.append(hit)
    lines.append(f"\n===== {sid} roster={[c['id'] for c in W[wid]['characters']]}\n[TEXT]\n{txt}\n")
    for k,f in enumerate(a["fact_updates"]):
        b = t["facts"][k]["basis"] if k < len(t["facts"]) else "?"
        lines.append(f"  PRED#{k} {'TP(g%d)'%res[k] if res[k] is not None else 'FP'} {f['subject']} {f['predicate']} 「{f['object']}」 ev={f['evidence']} basis={b}")
    for i,gf in enumerate(g["facts"]):
        if i not in used: lines.append(f"  GOLD#{i} FN {gf['subject']} {gf['predicate']} 「{gf['object']}」")
        else: lines.append(f"  GOLD#{i} matched {gf['subject']} {gf['predicate']} 「{gf['object']}」")
Path(sys.argv[1]).write_text("\n".join(lines), encoding="utf-8")
print(len(lines))
