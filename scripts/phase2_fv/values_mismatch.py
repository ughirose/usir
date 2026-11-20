import json, sys
from common import *
G = load_gold(); P, T = load_preds(S2 / "annotations")
AX = ["生死","秩序","世俗/神秘","利己/献身","理性/情念","宿命/自由"]
rows = []
for sid, g in G.items():
    pst = {s["character_id"]: s for s in P[sid]["character_states"]}
    tr = {s["id"]: s["basis"] for s in T[sid]["states"]}
    for cid, gv in g["states"].items():
        if cid not in pst:
            rows.append((sid, cid, gv, None, None)); continue
        rows.append((sid, cid, gv, pst[cid]["values_6d"], tr.get(cid)))
n=len(rows); miss=sum(r[3] is None for r in rows); print(n, "gold char-scenes; not predicted:", miss)
tot=same=opp=zero_pred=0
for sid,cid,gv,pv,b in rows:
    if pv is None: continue
    for x,y in zip(pv,gv):
        if y!=0:
            tot+=1
            if x==y: same+=1
            elif x==0: zero_pred+=1
            else: opp+=1
print("gold nonzero axes", tot, "same", same, "pred 0", zero_pred, "opposite", opp)
# predicted nonzero where gold zero
extra=sum(1 for r in rows if r[3] for x,y in zip(r[3],r[2]) if x!=0 and y==0)
print("pred nonzero on gold-zero axes", extra)
for r in rows:
    if r[3] is None: continue
    d=sum(1 for x,y in zip(r[3],r[2]) if x!=y)
    print(r[0], r[1], "gold",r[2],"pred",r[3],"diff",d, "| basis:", (r[4] or "")[:60])
