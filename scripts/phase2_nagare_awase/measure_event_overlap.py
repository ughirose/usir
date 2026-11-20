"""Event top3 overlap vs gold. overlap = |pred ∩ gold| / 3 per scene (order ignored).

Reports: scenes with a filled slot (primary), all 35 scenes with null slot counted as 0 (conservative), per work,
and two baselines: uniform-random top3 (expected 3/16) and constant top3 = the 3 most frequent gold codes (in-sample, optimistic).
Awase and mood have no gold: reported as unmeasured.
Usage: python measure_event_overlap.py <annotations_dir> <out_json>
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ann_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
gold = json.loads((HERE / "gold/event_top3_gold.json").read_text(encoding="utf-8"))["scenes"]

rows, per_work = [], collections.defaultdict(list)
for wid in sorted({k.split(":")[0] for k in gold}):
    anns = json.loads((ann_dir / f"{wid}.annotations.json").read_text(encoding="utf-8"))
    for a in anns:
        sid = a["contract"]["scene_id"]
        nag = a["scene_meta"].get("nagare")
        g = gold[sid]
        if nag is None:
            rows.append({"scene_id": sid, "pred": None, "gold": g, "overlap_codes": None, "overlap": None})
            continue
        pred = [x["code"] for x in nag["event_top3"]]
        k = len(set(pred) & set(g))
        rows.append({"scene_id": sid, "pred": pred, "gold": g, "overlap_codes": k, "overlap": k / 3,
                     "top1_match": pred[0] == g[0]})
        per_work[wid].append(k / 3)

filled = [r for r in rows if r["overlap"] is not None]
freq = collections.Counter(c for g in gold.values() for c in g)
const = [c for c, _ in freq.most_common(3)]
const_ov = [len(set(const) & set(g)) / 3 for g in gold.values()]
res = {
    "n_scenes": len(rows),
    "n_filled": len(filled),
    "mean_overlap_filled": round(sum(r["overlap"] for r in filled) / len(filled), 4) if filled else None,
    "mean_overlap_all35_null_as_0": round(sum((r["overlap"] or 0) for r in rows) / len(rows), 4),
    "distribution_filled": dict(sorted(collections.Counter(r["overlap_codes"] for r in filled).items())),
    "top1_match_filled": sum(1 for r in filled if r["top1_match"]),
    "per_work_mean_filled": {w: round(sum(v) / len(v), 4) for w, v in sorted(per_work.items())},
    "baseline_random_expected": round(3 / 16, 4),
    "baseline_constant_top3": {"codes": const, "mean_overlap_all35_in_sample": round(sum(const_ov) / len(const_ov), 4)},
    "mood_gold": "unmeasured (no gold)",
    "awase_gold": "unmeasured (no gold)",
    "scenes": rows,
}
out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "scenes"}, ensure_ascii=False, indent=1))
