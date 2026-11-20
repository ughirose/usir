import json, sys
from collections import defaultdict
from pathlib import Path
G = json.loads(Path("scripts/phase2_stage2/gold/stage_b_gold.json").read_text(encoding="utf-8"))["scenes"]
def load(d):
    P = {}
    for f in sorted(Path(d).glob("*.annotations.json")):
        for a in json.loads(f.read_text(encoding="utf-8")): P[a["contract"]["scene_id"]] = a
    return P
sgn = lambda x: (x>0)-(x<0)
def agg(P, mode):
    """per (work,character): aggregate gold & pred values over the character's gold scenes."""
    gs = defaultdict(lambda: [0]*6); ps = defaultdict(lambda: [0]*6); gpos=defaultdict(lambda:[0]*6); gneg=defaultdict(lambda:[0]*6); ppos=defaultdict(lambda:[0]*6); pneg=defaultdict(lambda:[0]*6)
    for sid, g in G.items():
        w = sid.split(":")[0]
        pst = {s["character_id"]: s["values_6d"] for s in P[sid]["character_states"]}
        for c, gv in g["states"].items():
            for i in range(6):
                gs[(w,c)][i] += gv[i]; gpos[(w,c)][i] += gv[i]>0; gneg[(w,c)][i] += gv[i]<0
                pv = pst.get(c, [0]*6)
                ps[(w,c)][i] += pv[i]; ppos[(w,c)][i] += pv[i]>0; pneg[(w,c)][i] += pv[i]<0
    n = hit = opp = 0; rows=[]
    for k in gs:
        for i in range(6):
            if mode == "sum": gsign, psign = sgn(gs[k][i]), sgn(ps[k][i])
            else: # any: gold nonzero if any scene nonzero; sign by majority
                gsign = sgn(gpos[k][i]-gneg[k][i]); psign = sgn(ppos[k][i]-pneg[k][i])
            if gsign == 0: continue
            n += 1; hit += psign == gsign; opp += psign == -gsign
            rows.append((k, i, gsign, psign))
    return n, hit, opp, rows
if __name__ == "__main__":
    for name, d in (("local", "scripts/phase2_stage2/annotations"), ("strong", "scripts/phase2_ceiling/annotations")):
        P = load(d)
        for mode in ("sum", "majority"):
            n, h, o, _ = agg(P, mode); print(name, mode, "gold_nz", n, "same", h, "opposite", o)
