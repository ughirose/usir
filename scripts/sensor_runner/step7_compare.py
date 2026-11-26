"""Phase 1.5 Stage 2 Step 7: old-instruction cold annotation (sandbox run s7a) compared two ways, same items as Step 6.
 (1) s7a vs s6a (new instruction, new method)  = effect of the one instruction line
 (2) s7a vs seed (old instruction, old method) = effect of method / format
Counts: whole-entry equality (canonical JSON, ignoring annotator / provenance), per-field difference scenes, semantic agreement, up to 3 examples.
Writes results/phase15_stage2_step7_compare.json. Measurement only."""
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
SEM = ["present_ids", "pov", "location", "event_codes", "event_top1", "mood_codes", "mood_top1", "trait_set", "facts_count", "awase4", "relative_order"]


def load(root: Path, scene_hash: str):
    fs = sorted(glob.glob(str(root / "l2" / scene_hash / "*.json")))
    assert fs, (root, scene_hash)
    es = [json.loads(Path(f).read_text(encoding="utf-8"))["entry"] for f in fs]
    assert all(e == es[0] for e in es), "entries of one scene differ"
    return es[0]


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def sem(e):
    tr = set()
    for t in e.get("trait_changes") or []:
        for c in t.get("changes") or []:
            tr.add(canon(c) if not isinstance(c, dict) else canon({"c": t.get("character_id"), "axis": c.get("axis"), "delta": c.get("delta")}))
    aw = e.get("awase") or {}
    ev, mo = e.get("event_top3") or [], e.get("mood_top3") or []
    return {
        "present_ids": sorted(p["id"] for p in e.get("present") or []),
        "pov": e.get("pov_character"),
        "location": e.get("location"),
        "event_codes": sorted(x["code"] for x in ev),
        "event_top1": ev[0]["code"] if ev else None,
        "mood_codes": sorted(x["code"] for x in mo),
        "mood_top1": mo[0]["code"] if mo else None,
        "trait_set": sorted(tr),
        "facts_count": len(e.get("facts") or []),
        "awase4": [aw.get(k) for k in ("F_valence", "F_arousal", "f_valence", "f_arousal")],
        "relative_order": (e.get("time") or {}).get("relative_order"),
    }


def compare(ref_root: Path, new_root: Path, nis) -> dict:
    res = {"totals": {"scenes": 0, "whole_equal": 0, "field_diff_scenes": {f: 0 for f in FIELDS}, "sem_diff_scenes": {k: 0 for k in SEM}, "sem_all_equal": 0}, "rows": [], "examples": []}
    strip = lambda e: {k: v for k, v in e.items() if k not in META}  # noqa: E731
    for w in WORKS:
        seg = segment(nis, load_work(w)["text_abs"])
        for s in seg["scenes"]:
            r, n = load(ref_root, s["scene_hash"]), load(new_root, s["scene_hash"])
            fd = [f for f in FIELDS if canon(r.get(f)) != canon(n.get(f))]
            sr, sn = sem(r), sem(n)
            sd = [k for k in SEM if sr[k] != sn[k]]
            t = res["totals"]
            t["scenes"] += 1
            t["whole_equal"] += canon(strip(r)) == canon(strip(n))
            t["sem_all_equal"] += not sd
            for f in fd:
                t["field_diff_scenes"][f] += 1
            for k in sd:
                t["sem_diff_scenes"][k] += 1
            res["rows"].append({"work": w, "scene": s["index"], "whole_equal": canon(strip(r)) == canon(strip(n)), "field_diffs": fd, "sem_diffs": sd,
                                "ref_annotator": r.get("annotator"), "new_annotator": n.get("annotator")})
            for k in sd:
                if len(res["examples"]) < 3:
                    res["examples"].append({"work": w, "scene": s["index"], "item": k, "ref": sr[k], "new": sn[k]})
    return res


def main() -> int:
    nis = find_nishiki(None)
    s7a = HERE / "step2_work" / "s7a" / "cache"
    out = {
        "A_instruction_line_effect (s7a old-instr vs s6a new-instr)": compare(HERE / "step2_work" / "s6a" / "cache", s7a, nis),
        "B_method_effect (s7a old-instr vs seed old-instr/old-method)": compare(HERE / "cache", s7a, nis),
    }
    (HERE / "results" / "phase15_stage2_step7_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k)
        print(" ", v["totals"])
        for r in v["rows"]:
            print("  ", r["work"], r["scene"], r["whole_equal"], r["field_diffs"], r["sem_diffs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
