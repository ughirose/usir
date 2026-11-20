"""Aggregate Stage A audits from the per-work trace files (numbers are copied from traces, not recomputed by hand)."""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); out = {"works": {}}
tot = {"scenes": 0, "llm_sec": 0.0, "present": 0, "states": 0, "facts": 0, "rejected": 0, "schema_errors": 0, "parse_failures": 0, "reasons": {}}
for f in sorted(d.glob("*.trace.json")):
    t = json.loads(f.read_text(encoding="utf-8")); wid = f.name.split(".")[0]
    a = t["audits"]
    w = {"scenes": len(a), "llm_sec": round(sum(x["elapsed_sec"] for x in a), 1),
         "present": sum(x["kept"]["present"] for x in a), "states": sum(x["kept"]["states"] for x in a), "facts": sum(x["kept"]["facts"] for x in a),
         "rejected": sum(len(x["rejected"]) for x in a), "schema_errors": sum(len(x["schema_errors"]) for x in a),
         "parse_failures": sum(1 for x in a if not x["parse_ok"]), "reasons": {}}
    for x in a:
        for r in x["rejected"]:
            k = f"{r['kind']}:{r['reason']}"; w["reasons"][k] = w["reasons"].get(k, 0) + 1
    out["works"][wid] = w
    for k in ("scenes", "llm_sec", "present", "states", "facts", "rejected", "schema_errors", "parse_failures"):
        tot[k] += w[k]
    for k, v in w["reasons"].items(): tot["reasons"][k] = tot["reasons"].get(k, 0) + v
tot["llm_sec"] = round(tot["llm_sec"], 1); out["total"] = tot
(d / "stageA_summary_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(tot, ensure_ascii=False))
