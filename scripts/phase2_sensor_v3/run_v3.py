"""sensor v3 runner: 7 works (3 modern + 4 F9 Dev), 35 scenes. One scene call per scene + one trait call per character.

Usage: python run_v3.py <out_dir>   (seed 42, temperature 0, qwen2.5:14b; same setting every run)
Output per work: <wid>.annotations.json (list of scene annotations), <wid>.traits.json, <wid>.trace.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts/phase2"))
sys.path.insert(0, str(REPO / "scripts/phase2_stage2"))

from build_annotations import segment  # noqa: E402
from usir_py.sensor_eval.ollama_client import call_ollama_chat  # noqa: E402
from usir_py.sensor_eval import scene_annotator_v3 as v3  # noqa: E402

import run_full_annotation as base  # noqa: E402  (SOURCES, roster_entries, NISHIKI, MODEL)

EXCERPT_CAP = 3000


def excerpts_for(char: dict, seg: dict) -> str:
    names = [a for a in char["aliases"] if a]
    if not names:
        return ""
    items = []
    for si, sc in enumerate(seg["scenes"]):
        t = sc["scene_text"]
        for s in sc["sentences"]:
            sent = t[s["local"]["start"]:s["local"]["end"]].strip()
            if any(n in sent for n in names):
                items.append(f"(場面{si + 1}) {sent}")
    if sum(len(x) for x in items) > EXCERPT_CAP:  # even subsample across the work, keep order
        step = sum(len(x) for x in items) / EXCERPT_CAP
        keep, acc, nxt = [], 0.0, 0.0
        for x in items:
            if acc >= nxt:
                keep.append(x)
                nxt += step * 0 + 0
            acc += len(x) / step
        items = keep
        while sum(len(x) for x in items) > EXCERPT_CAP:
            items.pop(len(items) // 2)
    return "\n".join(items)


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    only = sys.argv[2:] or None
    summary = {"model": base.MODEL, "prompt_hash": v3.prompt_hash(), "works": {}}
    t_all = time.perf_counter()
    for group, src in base.SOURCES.items():
        gold = json.loads(Path(src["gold"]).read_text(encoding="utf-8"))
        gw = {w["id"]: w for w in gold["works"]}
        for wid in src["ids"]:
            if only and wid not in only:
                continue
            w = gw[wid]
            seg = segment(base.NISHIKI, Path(src["text_dir"]) / f"{wid}.txt")
            scenes = seg["scenes"]
            ids = [c["id"] for c in w["characters"]]
            roster = base.roster_entries(w["characters"])
            t0 = time.perf_counter()
            anns, audits, raws = [], [], []
            for si, sc in enumerate(scenes):
                sents = [(s["index"], s["local"]["start"], s["local"]["end"]) for s in sc["sentences"]]
                prev_tail = scenes[si - 1]["scene_text"][-120:] if si > 0 else None
                msgs = v3.scene_messages(roster, sc["numbered_text"], si, len(scenes), prev_tail)
                resp = call_ollama_chat(base.MODEL, msgs, v3.scene_format_schema(ids, len(sents)), num_ctx=9216, num_predict=3072,
                                        seed=42, temperature=0.0, timeout_sec=300.0)
                cleaned, audit = v3.validate_scene_v3(resp.parsed_json, ids, sc["scene_text"], sents)
                sec = cleaned["sections"]
                sid = f"{wid}:s{si}"
                anns.append({
                    "contract": {"schema_version": "sensor.v3-draft", "scene_id": sid, "scene_hash": sc["scene_hash"], "work_id": wid,
                                 "scene_index": si, "scene_count": len(scenes), "model_id": base.MODEL, "seed": 42,
                                 "prompt_hash": v3.prompt_hash()[:16]},
                    "scene_meta": {"time": sec["time"], "location_id": sec["location_id"], "present_character_ids": sec["present_character_ids"]},
                    "fact_updates": sec["fact_updates"],
                    "value_events": sec["value_events"],
                })
                audit["scene_id"] = sid
                audit["elapsed_sec"] = resp.elapsed_sec
                audits.append({**audit, "trace": cleaned["trace"]})
                raws.append({"scene_id": sid, "raw_content": resp.raw_content, "parse_error": resp.parse_error})
                print(wid, si, audit["kept"], "rej", len(audit["rejected"]), f"{resp.elapsed_sec}s", flush=True)
            traits, trait_audit = {}, {}
            for c in roster:
                ex = excerpts_for(c, seg)
                if not ex:
                    trait_audit[c["id"]] = "no_mentions"
                    continue
                resp = call_ollama_chat(base.MODEL, v3.trait_messages(c, ex), v3.trait_format_schema(c["id"]), num_ctx=9216,
                                        num_predict=512, seed=42, temperature=0.0, timeout_sec=300.0)
                tr, why = v3.validate_trait(resp.parsed_json, ex)
                if tr:
                    traits[c["id"]] = tr
                trait_audit[c["id"]] = why or "ok"
                raws.append({"trait_id": c["id"], "excerpt_chars": len(ex), "raw_content": resp.raw_content, "parse_error": resp.parse_error})
                print(wid, "trait", c["id"], trait_audit[c["id"]], f"{resp.elapsed_sec}s", flush=True)
            (out_dir / f"{wid}.annotations.json").write_text(v3.dumps(anns), encoding="utf-8")
            (out_dir / f"{wid}.traits.json").write_text(v3.dumps({"traits": traits, "audit": trait_audit}), encoding="utf-8")
            (out_dir / f"{wid}.trace.json").write_text(v3.dumps({"audits": audits, "raw": raws}), encoding="utf-8")
            rr: dict[str, int] = {}
            for a in audits:
                for r in a["rejected"]:
                    rr[f"{r['kind']}:{r['reason'].split(':')[0]}"] = rr.get(f"{r['kind']}:{r['reason'].split(':')[0]}", 0) + 1
            summary["works"][wid] = {"group": group, "scenes": len(scenes), "elapsed_sec": round(time.perf_counter() - t0, 1),
                                     "parse_failures": sum(1 for a in audits if not a["parse_ok"]), "rejected_reasons": rr,
                                     "traits_ok": sum(1 for v in trait_audit.values() if v == "ok"), "trait_audit": trait_audit}
    summary["total_elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "summary.json").write_text(v3.dumps(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
