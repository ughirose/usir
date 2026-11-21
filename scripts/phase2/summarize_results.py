"""Collect per-work diagnose outputs into one results JSON + findings list (numbers are copied, not recomputed)."""
import json, sys, hashlib
from pathlib import Path
R = Path(__file__).resolve().parent
T = R.parents[0] / "step1/modern_gold"
works = ["001798_056779", "001801_001786", "001875_059897"]
out = {"schema": "phase2.step1.results/1", "works": {}, "findings": []}
for w in works:
    d = json.loads((R / f"results/diagnose/{w}.diagnose.json").read_text(encoding="utf-8"))
    text = (T / f"{w}.txt").read_text(encoding="utf-8")
    ann = json.loads((R / f"annotations/{w}.annotations.json").read_text(encoding="utf-8"))
    lenses = {}
    for l in d["lenses"]:
        lenses[l["id"]] = {"status": l["status"], "reason": l.get("reason"), "scores": l["scores"], "span_count": len(l["spans"]),
                           "span_by_rule": {}, "notes": l["notes"]}
        for s in l["spans"]:
            lenses[l["id"]]["span_by_rule"][s["rule"]] = lenses[l["id"]]["span_by_rule"].get(s["rule"], 0) + 1
            out["findings"].append({"work_id": w, "lens": l["id"], "rule": s["rule"], "scene": s["scene"], "score": s["score"],
                                    "start": s["start"], "end": s["end"], "basis": s["basis"], "text": text[s["start"]:s["end"]]})
    out["works"][w] = {"schema": d["schema"], "input": d["input"], "warnings": d.get("warnings", []),
                       "annotated_utterances": sum(len(a["utterances"]) for a in ann),
                       "lenses": lenses}
(R / "results/phase2_step1_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(len(out["findings"]), "findings")
