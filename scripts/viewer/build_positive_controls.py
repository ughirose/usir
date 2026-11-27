"""Positive controls: take 6 location-type S1 items, insert an explicit contradiction sentence at the start of the flagged scene text.
The claim stays the same. A sensitive judge must answer 'real'. Output: measure/batches_pos/batch_P0.json, measure/pos_key.json"""
import json
from pathlib import Path
M = Path(__file__).resolve().parent / "measure"
key = json.loads((M / "items.json").read_text(encoding="utf-8"))["items"]
src = [i for i in key if i["stratum"] == "S1" and i["rule"] in ("move_destination_mismatch", "place_conflict")][:6]
INS = ["そのとき当人は、遠い山の頂の岩に腰をかけて、ひとりで煙草を吹かしていた。", "ところが当人はもう、前の場所とは別の町の旅籠で、膳に向かって飯を食っていた。"]
out = []
for n, it in enumerate(src):
    ctx = [dict(c) for c in it["context"]]
    ctx[-1]["text"] = INS[n % 2] + "\n" + ctx[-1]["text"]
    out.append({"id": f"P{n:02d}", "claim": it["claim"], "context": ctx})
(M / "batches_pos").mkdir(exist_ok=True)
(M / "batches_pos" / "batch_P0.json").write_text(json.dumps({"items": out}, ensure_ascii=False, indent=1), encoding="utf-8")
(M / "pos_key.json").write_text(json.dumps([{"id": o["id"], "from": s["id"], "work": s["work"], "rule": s["rule"]} for o, s in zip(out, src)], ensure_ascii=False, indent=1), encoding="utf-8")
print(len(out), [s["work"] + ":" + s["rule"] for s in src])
