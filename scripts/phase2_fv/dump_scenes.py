import json, sys
from pathlib import Path
from common import *
W = load_works(); G = load_gold(); P, T = load_preds(S2 / "annotations")
out = Path(sys.argv[1]); out.mkdir(exist_ok=True, parents=True)
for sid in G:
    wid, si = sid.split(":s"); si = int(si)
    txt = W[wid]["scenes"][si]["scene_text"]
    (out / (sid.replace(":", "_") + ".txt")).write_text(txt, encoding="utf-8")
print(len(G), "scenes;", {w: len(W[w]["scenes"]) for w in W})
print(sum(len(W[sid.split(':s')[0]]["scenes"][int(sid.split(':s')[1])]["scene_text"]) for sid in G), "chars total")
