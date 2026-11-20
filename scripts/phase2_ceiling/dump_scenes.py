"""Dump scene texts + rosters (sensor inputs only; does NOT read stage_b_gold.json)."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"python")); sys.path.insert(0, str(REPO/"scripts/phase2")); sys.path.insert(0, str(REPO/"scripts/step1/modern_gold"))
sys.path.insert(0, str(REPO/"scripts/phase2_stage2"))
from build_annotations import segment
import run_full_annotation as R
out = {}
for group, src in R.SOURCES.items():
    gold = json.loads(Path(src["gold"]).read_text(encoding="utf-8"))
    gw = {w["id"]: w for w in gold["works"]}
    for wid in src["ids"]:
        seg = segment(R.NISHIKI, Path(src["text_dir"])/f"{wid}.txt")
        w = gw[wid]
        out[wid] = {"group": group, "roster": R.roster_entries(w["characters"]),
                    "narration": w["narration"], "narrator": w.get("narrator"), "pov": w.get("pov"),
                    "scenes": [{"scene_id": f"{wid}:s{i}", "scene_hash": s["scene_hash"], "text": s["scene_text"]} for i, s in enumerate(seg["scenes"])]}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print({k: len(v["scenes"]) for k, v in out.items()}, sum(len(v["scenes"]) for v in out.values()))
