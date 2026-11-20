import json, glob
same = tot = 0
for f in sorted(glob.glob("scripts/phase2_stage2/annotations_rerun/*.annotations.json")):
    x = json.load(open(f, encoding="utf-8")); y = json.load(open(f.replace("_rerun", ""), encoding="utf-8"))
    for p, q in zip(x, y):
        tot += 1; same += (p["scene_meta"], p["character_states"], p["fact_updates"]) == (q["scene_meta"], q["character_states"], q["fact_updates"])
json.dump({"scenes_identical": same, "scenes": tot}, open("scripts/phase2_stage2/results/rerun_comparison.json", "w"))
print(same, tot)
