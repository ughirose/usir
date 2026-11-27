"""Aggregate judge answers (measure/answers/*.json) against the key (measure/items.json).
hit = verdict 'real'. strict = real / n ; lenient = (real + unclear) / n.
usage: python aggregate.py  -> measure/results.json + printed tables"""
import json
from collections import defaultdict
from pathlib import Path

M = Path(__file__).resolve().parent / "measure"
key = json.loads((M / "items.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in key["items"]}
ans = {}
for f in sorted((M / "answers").glob("batch_*.json")):
    for a in json.loads(f.read_text(encoding="utf-8"))["answers"]:
        ans[a["id"]] = a
missing = sorted(set(items) - set(ans))


def tally(sel):
    n = len(sel)
    r = sum(ans[i["id"]]["verdict"] == "real" for i in sel)
    u = sum(ans[i["id"]]["verdict"] == "unclear" for i in sel)
    return {"n": n, "real": r, "unclear": u, "not_real": n - r - u, "strict": (r / n if n else None), "lenient": ((r + u) / n if n else None)}


done = [i for i in items.values() if i["id"] in ans]
out = {"missing": missing, "overall": {}, "by_work_S1": {}, "by_rule_S1": {}}
for st in ("S1", "S2", "S3", "C"):
    out["overall"][st] = tally([i for i in done if i["stratum"] == st])
for w in sorted({i["work"] for i in done}):
    out["by_work_S1"][w] = tally([i for i in done if i["stratum"] == "S1" and i["work"] == w])
by = defaultdict(list)
for i in done:
    if i["stratum"] == "S1":
        by[i["rule"]].append(i)
out["by_rule_S1"] = {k: tally(v) for k, v in sorted(by.items())}
(M / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("missing", missing)
for k, v in out["overall"].items():
    print(k, v)
for k, v in out["by_work_S1"].items():
    print(k, v)
for k, v in out["by_rule_S1"].items():
    print(k, v)
