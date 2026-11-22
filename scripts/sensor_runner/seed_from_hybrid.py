"""Seed the cache from the existing 総仕上げ annotations (phase2/hybrid-full-run @ 6043212).

Reads (from a checkout of that branch, --hybrid-root):
  scripts/phase2_hybrid/annotations/<work>.{annotations,basis,characters}.json
  scripts/phase2_ceiling/scenes_input.json        (narration / narrator / pov per work)
and writes into this branch:
  works/<work>.work.json, works/<work>.characters.json     work registry
  local/<work>.utterances.json                             speaker/addressee/evidence per scene_hash (provenance: seed)
  cache/strong/<scene_hash>.json                           strong annotation (provenance: seed:hybrid-full-run@6043212)
  cache/ledger/<work>.traits.json                          baseline = per-character per-axis mode over the work's scenes
Scene state values are stored as variations from that baseline (delta = scene value - baseline), so merge reproduces
the original values_6d exactly. Source basis strings exist only for event/mood/awase; other seed items have basis=null.

Note: the hybrid utterances were annotated by the strong model, not by the local model. They are stored in the local-side
file only so the merge path can be exercised end to end; provenance says so.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from cache import Cache
from common import LOCAL_DIR, REPO, STRONG_SCHEMA, INSTRUCTION_VERSION, WORKS_DIR, find_nishiki, read_json, segment, write_json

SEED_PROV = "seed:hybrid-full-run@6043212"
F9_TEXT_DIR = "C:/dev/nishiki/crates/nishiki-discourse/tests/data/f9_gold"
MODERN = {"001798_056779", "001801_001786", "001875_059897"}


def baseline_mode(vectors: list[list[int]]) -> list[int]:
    out = []
    for ax in range(6):
        c = collections.Counter(v[ax] for v in vectors)
        out.append(sorted(c, key=lambda x: (-c[x], abs(x), x))[0])
    return out


def seed_work(wid: str, hroot: Path, cache: Cache, nishiki: str, scenes_input: dict) -> dict:
    ann_dir = hroot / "scripts/phase2_hybrid/annotations"
    anns = read_json(ann_dir / f"{wid}.annotations.json")
    basis = read_json(ann_dir / f"{wid}.basis.json")
    chars = read_json(ann_dir / f"{wid}.characters.json")
    meta = scenes_input[wid]
    text_path = f"scripts/step1/modern_gold/{wid}.txt" if wid in MODERN else f"{F9_TEXT_DIR}/{wid}.txt"
    spec = {
        "work_id": wid,
        "text_path": text_path,
        "narration": meta["narration"],
        "narrator": meta.get("narrator"),
        "pov": meta.get("pov"),
        "source": SEED_PROV,
    }
    write_json(WORKS_DIR / f"{wid}.work.json", spec)
    write_json(WORKS_DIR / f"{wid}.characters.json", chars)

    tp = Path(text_path)
    seg = segment(nishiki, tp if tp.is_absolute() else REPO / tp)
    assert len(seg["scenes"]) == len(anns) == len(basis), wid
    # ledger baseline
    per_char: dict[str, list[list[int]]] = collections.defaultdict(list)
    for a in anns:
        for s in a["character_states"]:
            per_char[s["character_id"]].append(s["values_6d"])
    baseline = {cid: baseline_mode(v) for cid, v in sorted(per_char.items())}
    cache.put_ledger(wid, baseline, note=f"{SEED_PROV}: per-axis mode of the 総仕上げ scene values")

    local_scenes = {}
    for sc, a, b in zip(seg["scenes"], anns, basis):
        assert a["contract"]["scene_hash"] == sc["scene_hash"], (wid, sc["index"])
        sm = a["scene_meta"]
        t = sm["time"]
        facts = []
        for f in a["fact_updates"]:
            r = {"subject": f["subject"], "predicate": f["predicate"], "evidence": f["evidence"], "basis": None}
            if f["predicate"] == "STATE_CHANGED":
                r.update({"target": None, "category": "unclassified", "text": f["object"]})
            else:
                r["object"] = f["object"]
            facts.append(r)
        tc = []
        for s in a["character_states"]:
            base = baseline[s["character_id"]]
            changes = [{"axis": ax, "delta": v - base[ax], "basis": None} for ax, v in enumerate(s["values_6d"]) if v != base[ax]]
            tc.append({"character_id": s["character_id"], "evidence": s["evidence"], "changes": changes})
        assert [e[0] for e in b["event"]] == [x["code"] for x in sm["nagare"]["event_top3"]], (wid, sc["index"])
        aw = sm["awase"]
        entry = {
            "schema": STRONG_SCHEMA,
            "instruction_version": INSTRUCTION_VERSION,
            "scene_hash": sc["scene_hash"],
            "annotator": a["contract"]["model_id"],
            "provenance": SEED_PROV,
            "time": {
                "relative_order": t["relative_order"],
                "explicit_time": t.get("explicit_time"),
                "elapsed_hint": t.get("elapsed_hint"),
                "evidence": t["evidence"],
            },
            "location": sm["location_id"],
            "pov_character": sm["pov_character"],
            "present": [{"id": p, "basis": None} for p in sm["present_character_ids"]],
            "ledger_additions": {},
            "trait_changes": tc,
            "facts": facts,
            "event_top3": [{"code": c, "basis": bs} for c, bs in b["event"]],
            "mood_top3": [{"code": c, "basis": bs} for c, bs in b["mood"]],
            "awase": {
                "F_valence": aw["f_event_valence"], "F_arousal": aw["f_event_arousal"],
                "f_valence": aw["f_style_valence"], "f_arousal": aw["f_style_arousal"],
                "F_basis": b["awase_basis"]["F"], "f_basis": b["awase_basis"]["f"],
            },
        }
        cache.put(entry)
        local_scenes[sc["scene_hash"]] = [
            {"utterance_index": u["utterance_index"], "speaker_id": u["speaker_id"], "addressee_id": u["addressee_id"], "evidence": u["evidence"]}
            for u in a["utterances"]
        ]
    write_json(LOCAL_DIR / f"{wid}.utterances.json", {
        "schema": "sensor.local_utterances/1", "work_id": wid,
        "provenance": f"{SEED_PROV} (utterances annotated by the strong model; stand-in for local-model output)",
        "scenes": local_scenes,
    })
    m = cache.write_manifest(wid, seg["scenes"], seg["splitter"])
    return {"work_id": wid, "scenes": m["scene_count"], "cached": m["cached_count"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-root", required=True, help="checkout of usir phase2/hybrid-full-run (6043212)")
    ap.add_argument("--nishiki", default=None)
    a = ap.parse_args()
    hroot = Path(a.hybrid_root)
    nishiki = find_nishiki(a.nishiki)
    scenes_input = read_json(hroot / "scripts/phase2_ceiling/scenes_input.json")
    cache = Cache()
    for wid in sorted(scenes_input):
        print(seed_work(wid, hroot, cache, nishiki, scenes_input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
