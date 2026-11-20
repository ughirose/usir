"""Extra scoring: per-person trait aggregation (same rule as v3 report: gold trait = majority sign over gold scenes,
ties -> 0 excluded; evaluated on characters having a prediction) and a per-scene diff dump for error analysis."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from trait_agg import G, load, sgn, agg
sys.path.insert(0, str(Path(__file__).parents[1]/"phase2_stage2"))
from measure_stage_b import text_match
ROOT = Path(__file__).parents[2]
def no_alias_chars(ann_dir):
    s = set()
    for f in Path(ann_dir).glob("*.characters.json"):
        for c in json.loads(f.read_text(encoding="utf-8")):
            if not c["aliases"]: s.add((f.name.split(".")[0], c["id"]))
    return s
if __name__ == "__main__":
    P = load("scripts/phase2_ceiling/annotations")
    n, h, o, rows = agg(P, "majority")
    excl = no_alias_chars("scripts/phase2_stage2/annotations")
    sub = [r for r in rows if r[0] not in excl]
    out = {"person_unit_all_gold_nonzero": {"n": n, "same": h, "opposite": o, "zero_pred": n-h-o},
           "chars_without_alias_in_local_roster": sorted(map(list, excl)),
           "person_unit_comparable_subset": {"n": len(sub), "same": sum(r[2]==r[3] for r in sub), "opposite": sum(r[3]==-r[2] for r in sub)},
           "rows": [{"work": r[0][0], "char": r[0][1], "axis": r[1], "gold": r[2], "pred": r[3]} for r in rows]}
    # per scene diff
    diff = []
    for sid, g in G.items():
        a = P[sid]
        used = set(); fp = []
        for f in a["fact_updates"]:
            m = None
            for i, gf in enumerate(g["facts"]):
                if i not in used and gf["predicate"] == f["predicate"] and text_match(f["object"], gf["object"]): m = i; break
            if m is None: fp.append(f)
            else: used.add(m)
        fn = [gf for i, gf in enumerate(g["facts"]) if i not in used]
        diff.append({"scene_id": sid, "tp": len(used), "fp": fp, "fn": fn,
                     "present_gold": g["present"], "present_pred": a["scene_meta"]["present_character_ids"],
                     "order": [g["relative_order"], a["scene_meta"]["time"]["relative_order"]],
                     "explicit_time": [g["explicit_time"], a["scene_meta"]["time"].get("explicit_time")],
                     "location": [g["location_surfaces"], a["scene_meta"]["location_id"]],
                     "states_gold": g["states"], "states_pred": {s["character_id"]: s["values_6d"] for s in a["character_states"]}})
    out["scene_diff"] = diff
    Path("scripts/phase2_ceiling/results/extra_metrics_strong.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print({k: v for k, v in out.items() if k not in ("rows", "scene_diff")})
