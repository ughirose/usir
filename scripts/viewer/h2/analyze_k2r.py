"""Pack5 K2R: count detections of the 9 K2 insertions in a diagnose output and dump the annotation facts that decide them.
usage: python h2/analyze_k2r.py <final_dir> <work_id>   (e.g. step2_work/K2/K2final 000129_000689_h2b); prints one JSON line per run to stdout.
Insertion -> detection criterion (rule spans of the unmodified `nishiki diagnose` output):
  Q1 quote_bracket stray_closer near the Q1 offset, Q2 unclosed near Q2, Q3 stray_closer near Q3 (span start within +-3 chars of the inserted bracket position)
  M1 move_destination_mismatch for character zushio (厨子王 -> 由良の港, end of scene 3); M2 for anju with known_location 山の寺 (end of scene 5)
  P1 place_conflict between scenes 3 and 4 (same_explicit_time); P2 place_conflict between scenes 5 and 6
  V1 pov_contamination attributed_to kikori in scene 4; V2 pov_contamination attributed_to jiro in scene 6
Pure counting; no change to diagnose or its inputs."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
t = (H / "text_k2.txt").read_text(encoding="utf-8")
ins = json.loads((H / "insertions_k2.json").read_text(encoding="utf-8"))

fd, wid = Path(sys.argv[1]), sys.argv[2]
d = json.loads((fd / f"{wid}.diagnose.json").read_text(encoding="utf-8"))
a = json.loads((fd / f"{wid}.annotations.json").read_text(encoding="utf-8"))
spans = [s for l in d["lenses"] for s in l.get("spans", [])]
scores = {}
for l in d["lenses"]:
    scores.update(l.get("scores", {}))

# inserted bracket positions in text_k2: Q1 "覚えた。」", Q2 "しておけ大夫", Q3 "聞いた。」二郎"
q1 = t.index("安寿は汐を汲み覚えた。」") + len("安寿は汐を汲み覚えた。")
q2 = t.index("どうにでも勝手にしておけ大夫") + len("どうにでも勝手にしておけ") - 20  # unclosed: the opening bracket is earlier in the same utterance
q3 = t.index("それを二郎が通りかかって聞いた。」") + len("それを二郎が通りかかって聞いた。")


def has(rule, pred):
    return any(s["rule"] == rule and pred(s) for s in spans)


def qb(kind, pos, tol):
    return has("quote_bracket", lambda s: s["basis"].get("kind") == kind and abs(s["start"] - pos) <= tol)


# Q2 (unclosed): the span is on the opening bracket of the utterance whose closer was removed; locate that opening bracket
o2 = t.rfind("「", 0, t.index("どうにでも勝手にしておけ大夫"))
res = {
    "Q1_stray_closer": qb("stray_closer", q1, 3),
    "Q2_unclosed": qb("unclosed_at_eof", o2, 3) or qb("unclosed_recovered", o2, 3),
    "Q3_stray_closer": qb("stray_closer", q3, 3),
    "M1_zushio": has("move_destination_mismatch", lambda s: s["basis"].get("character") == "zushio"),
    "M2_anju_山の寺": has("move_destination_mismatch", lambda s: s["basis"].get("character") == "anju" and "山の寺" in str(s["basis"].get("known_location"))),
    "P1_scenes3-4": has("place_conflict", lambda s: s["scene"] == 4 and str(s["basis"].get("previous_scene")) == "3"),
    "P2_scenes5-6": has("place_conflict", lambda s: s["scene"] == 6 and str(s["basis"].get("previous_scene")) == "5"),
    "V1_kikori_scene4": has("pov_contamination", lambda s: s["scene"] == 4 and s["basis"].get("attributed_to") == "kikori"),
    "V2_jiro_scene6": has("pov_contamination", lambda s: s["scene"] == 6 and s["basis"].get("attributed_to") == "jiro"),
}
facts = []
for i in range(3, 7):
    m = a[i]["scene_meta"]
    mv = [(f.get("subject"), f.get("object"), f.get("evidence")) for f in a[i].get("fact_updates", []) if f.get("predicate") == "MOVED_TO"]
    facts.append({"scene": i, "location": m.get("location_id"), "pov": m.get("pov_character"), "explicit_time": (m.get("time") or {}).get("explicit_time"), "elapsed_hint": (m.get("time") or {}).get("elapsed_hint"),
                  "relative_order": (m.get("time") or {}).get("relative_order"), "present": m.get("present_character_ids"), "moved_to": mv})
other = [(s["rule"], s["scene"], s["basis"].get("character") or s["basis"].get("kind")) for s in spans
         if s["rule"] in ("move_destination_mismatch", "place_conflict", "pov_contamination", "quote_bracket")]
print(json.dumps({"detected": res, "n_detected": sum(res.values()), "facts": facts,
                  "all_rule_spans": other, "counts": {k: v for k, v in scores.items() if any(w in k for w in ("place_conflict", "move_dest", "location_change", "pov_contam"))}},
                 ensure_ascii=False))
