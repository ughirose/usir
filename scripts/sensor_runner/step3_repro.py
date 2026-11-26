"""Phase 1.5 Stage 2 Step 3: reproduce the Step 2 `ep` condition with a deterministic stub annotator and count rejected attempts.

Condition: work 001801_001786, committed ledger EMPTY, epoch-1 loop (prepare -> annotate -> ingest) until nothing is pending.
The stub behaves like the real annotator did in Step 2: every pack shows the frozen (empty) ledger, so it lists every visible
character in ledger_additions (baseline = zeros) in every scene. Prints JSON: calls, rejected attempts, reasons, pending set.
Usage: python step3_repro.py [--work 001801_001786]"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from cache import Cache  # noqa: E402
from common import find_nishiki, load_work, read_json, segment, write_json  # noqa: E402
from epoch import commit_epoch  # noqa: E402
from ingest import ingest_file  # noqa: E402
from prepare import prepare_work  # noqa: E402


def stub(pack: dict) -> dict:
    t = pack["scene_text"]
    basis = t[:4]
    vis = [c["id"] for c in pack["roster"]]
    pov = vis[0] if vis else None
    return {
        "schema": "sensor.strong/1", "instruction_version": common.INSTRUCTION_VERSION, "scene_hash": pack["scene_hash"],
        "annotator": "stub", "provenance": "session",
        "time": {"relative_order": "continuous", "explicit_time": None, "elapsed_hint": None, "evidence": "inferred"},
        "location": None, "pov_character": pov,
        "present": [{"id": c, "basis": basis} for c in vis],
        "ledger_additions": {c: [0] * 6 for c in vis if c not in pack["ledger_baseline"]},
        "trait_changes": [], "facts": [],
        "event_top3": [{"code": "E02", "basis": basis}, {"code": "E03", "basis": basis}, {"code": "E10", "basis": basis}],
        "mood_top3": [{"code": "M05", "basis": basis}, {"code": "M12", "basis": basis}, {"code": "M10", "basis": basis}],
        "awase": {"F_valence": 0, "F_arousal": 1, "f_valence": 0, "f_arousal": -1, "F_basis": basis, "f_basis": basis},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="001801_001786")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="step3_repro_"))
    cache = Cache(tmp / "cache")
    cache.put_ledger(a.work, {})
    seg = segment(find_nishiki(None), load_work(a.work)["text_abs"])
    calls, rejected, reasons, order = 0, 0, [], []
    while True:
        r = prepare_work(a.work, cache, seg=seg, packs_dir=tmp / "packs", defer_blocked=True)
        if not r["packs_written"]:
            break
        for p in r["packs_written"]:
            pack = read_json(Path(p["pack"]))
            out = tmp / "out" / f"{pack['scene_hash'][:12]}.strong.json"
            write_json(out, stub(pack))
            res = ingest_file(out, cache, tmp / "packs", tmp / "rejects")
            calls += 1
            order.append({"scene": pack["scene_index"], "additions_in_entry": sorted(stub(pack)["ledger_additions"]), "accepted": res["accepted"]})
            if not res["accepted"]:
                rejected += 1
                reasons += res["reasons"]
                break
        else:
            continue
        break
    st = commit_epoch(cache, a.work)
    print(json.dumps({"work": a.work, "scenes": len(seg["scenes"]), "epoch1_calls": calls, "rejected_attempts": rejected, "reasons": reasons,
                      "per_scene": order, "pending_after": sorted(cache.pending_additions(a.work)) if st["status"] != "committed" else [],
                      "commit": {"status": st["status"], "applied": st.get("applied")}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
