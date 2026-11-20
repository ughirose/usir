"""Speaker attribution accuracy of annotation sets against the gold utterance lists (read after annotations were committed).
Gold: modern frozen gold (F: inbox) and F9 gold.json. Global utterance order = concatenation over scenes (checked against counts).
attributed: speaker == gold speaker. not_speech / undetermined: reported separately (does the annotation abstain with speaker null?).
Usage: python score_utterances.py <ann_dir> <out_json>"""
import json, sys, collections
from pathlib import Path
MOD = "F:/マイドライブ/Claude-Sync/inbox/2026-11-20_modern_gold_frozen.json"
F9 = "C:/dev/nishiki/crates/nishiki-discourse/tests/data/f9_gold/gold.json"
WORKS = {"modern": ["001798_056779", "001801_001786", "001875_059897"], "f9dev": ["000035_000301", "001383_056643", "000050_003586", "000081_000456"]}
gold = {}
for path in (MOD, F9):
    for w in json.loads(Path(path).read_text(encoding="utf-8"))["works"]: gold[w["id"]] = w["utterances"]
ann_dir = Path(sys.argv[1]); res = {}
tot = collections.Counter(); per = {}
for grp, ids in WORKS.items():
    for wid in ids:
        anns = json.loads((ann_dir/f"{wid}.annotations.json").read_text(encoding="utf-8"))
        flat = [u for a in anns for u in a["utterances"]]
        g = gold[wid]; assert len(flat) == len(g), (wid, len(flat), len(g))
        c = collections.Counter()
        for p, gu in zip(flat, g):
            st = gu["status"]
            if st == "attributed":
                c["attr_n"] += 1; c["attr_ok"] += p["speaker_id"] == gu["speaker"]
                c["attr_null"] += p["speaker_id"] is None
                if p["evidence"] == "ambiguous": c["attr_pred_ambiguous"] += 1
            else:
                c[f"{st}_n"] += 1; c[f"{st}_abstain"] += p["speaker_id"] is None
        per[wid] = dict(c); tot.update(c)
        for k, v in c.items(): res.setdefault(grp, collections.Counter())[k] += v
out = {"all": dict(tot), "groups": {k: dict(v) for k, v in res.items()}, "per_work": per}
a = tot
out["attributed_accuracy"] = round(a["attr_ok"]/a["attr_n"], 4)
print(json.dumps(out, ensure_ascii=False, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
