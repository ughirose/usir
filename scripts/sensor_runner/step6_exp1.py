"""Phase 1.5 Stage 2 Step 6, experiment 1: new-instruction cold annotation (sandbox run s6a) vs the committed seed annotation (old instructions).
Per scene: whole-entry equality (canonical JSON, ignoring annotator / provenance metadata only) and per-field equality.
Writes results/phase15_stage2_step6_exp1.json."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import find_nishiki, load_work, segment  # noqa: E402

WORKS = ["000035_000301", "001798_056779"]
META = ("annotator", "provenance")
FIELDS = ["present", "pov_character", "location", "facts", "trait_changes", "ledger_additions", "time", "event_top3", "mood_top3", "awase"]


def load(root: Path, scene_hash: str):
    fs = sorted(glob.glob(str(root / "l2" / scene_hash / "*.json")))
    assert fs, (root, scene_hash)
    es = [json.loads(Path(f).read_text(encoding="utf-8"))["entry"] for f in fs]
    assert all(e == es[0] for e in es), "seed entries of one scene differ"
    return es[0]


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def main() -> int:
    nis = find_nishiki(None)
    old_root, new_root = HERE / "cache", HERE / "step2_work" / "s6a" / "cache"
    res = {"works": {}, "totals": {"scenes": 0, "whole_equal": 0, "whole_equal_incl_meta": 0, "field_diff_scenes": {f: 0 for f in FIELDS}}, "examples": []}
    for w in WORKS:
        seg = segment(nis, load_work(w)["text_abs"])
        rows = []
        for s in seg["scenes"]:
            o, n = load(old_root, s["scene_hash"]), load(new_root, s["scene_hash"])
            strip = lambda e: {k: v for k, v in e.items() if k not in META}  # noqa: E731
            diffs = [f for f in FIELDS if canon(o.get(f)) != canon(n.get(f))]
            row = {"scene": s["index"], "whole_equal": canon(strip(o)) == canon(strip(n)), "whole_equal_incl_meta": canon(o) == canon(n), "differs": diffs,
                   "old_annotator": o.get("annotator"), "new_annotator": n.get("annotator")}
            rows.append(row)
            res["totals"]["scenes"] += 1
            res["totals"]["whole_equal"] += row["whole_equal"]
            res["totals"]["whole_equal_incl_meta"] += row["whole_equal_incl_meta"]
            for f in diffs:
                res["totals"]["field_diff_scenes"][f] += 1
                res["examples"].append({"work": w, "scene": s["index"], "field": f, "old": o.get(f), "new": n.get(f)})
        res["works"][w] = {"scenes": len(rows), "rows": rows}
    write = HERE / "results" / "phase15_stage2_step6_exp1.json"
    write.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    t = res["totals"]
    print(t)
    for w, d in res["works"].items():
        print(w, [(r["scene"], r["whole_equal"], r["differs"]) for r in d["rows"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
