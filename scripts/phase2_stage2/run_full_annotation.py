"""Phase 2 stage 2 / Stage A runner: sensor.v2 full annotation for 3 modern + 4 F9 Dev works.

Frozen parts (not re-run): utterance speaker/addressee/evidence/first_person from the frozen
v5+postprocess+speaker-recheck outputs (modern: scripts/phase2/results/sensor_run, which phase2 step1
verified identical to the frozen modern result; F9 Dev: scripts/step1/results/v5_threestage).
New parts (this script): one qwen2.5:14b call per scene (seed 42, temperature 0) for
scene_meta / character_states / fact_updates, then deterministic validation.
register.politeness is re-assigned by Policy X (as in phase2 step1).

Usage: python run_full_annotation.py <out_dir> [--only WORK_ID ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts/phase2"))
sys.path.insert(0, str(REPO / "scripts/step1/modern_gold"))

from janome.tokenizer import Tokenizer  # noqa: E402
from build_annotations import POLITE_MAP, PRONOUN_ALIASES, segment  # noqa: E402
from build_modern_gold import classify_politeness_surface  # noqa: E402
from usir_py.sensor_eval.ollama_client import call_ollama_chat  # noqa: E402
from usir_py.sensor_eval.scene_annotator import (  # noqa: E402
    build_scene_format_schema,
    build_scene_messages,
    scene_prompt_hash,
    validate_annotation_schema,
    validate_scene_output,
)

NISHIKI = "C:/dev/nishiki-phase2/target/release/nishiki.exe"
SCHEMA = Path("C:/dev/usir-sensor-v2/schemas/sensor_v2.schema.json")
MODERN_GOLD = Path("F:/マイドライブ/Claude-Sync/inbox/2026-11-20_modern_gold_frozen.json")
F9_DIR = Path("C:/dev/nishiki/crates/nishiki-discourse/tests/data/f9_gold")
SOURCES = {
    "modern": dict(
        gold=MODERN_GOLD,
        text_dir=REPO / "scripts/step1/modern_gold",
        sensor=REPO / "scripts/phase2/results/sensor_run/qwen2.5_14b_dev.json",
        ids=["001798_056779", "001801_001786", "001875_059897"],
    ),
    "f9dev": dict(
        gold=F9_DIR / "gold.json",
        text_dir=F9_DIR,
        sensor=REPO / "scripts/step1/results/v5_threestage/qwen2.5_14b_dev.json",
        ids=["000035_000301", "001383_056643", "000050_003586", "000081_000456"],
    ),
}
MODEL = "qwen2.5:14b"


def roster_entries(characters: list[dict]) -> list[dict]:
    """Same rule as phase2 step1 (pronoun aliases and aliases shared by several characters removed); name defaults to id."""
    surf = lambda c: set(c["aliases"]) | {c.get("name", c["id"])}  # noqa: E731
    count: dict[str, int] = {}
    for c in characters:
        for a in surf(c):
            count[a] = count.get(a, 0) + 1
    out = []
    for c in characters:
        al = {a for a in surf(c) if a not in PRONOUN_ALIASES and count[a] == 1}
        out.append({"id": c["id"], "aliases": sorted(al, key=lambda s: (-len(s), s)), "gender": c["gender"], "honored": False})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    tok = Tokenizer()
    pr_hash = scene_prompt_hash()
    sp = out_dir / "stageA_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if a.only and sp.exists() else {"model": MODEL, "scene_prompt_hash": pr_hash, "works": {}}
    t_all = time.perf_counter()
    for group, src in SOURCES.items():
        gold = json.loads(Path(src["gold"]).read_text(encoding="utf-8"))
        gold_sha = hashlib.sha256(Path(src["gold"]).read_bytes()).hexdigest()
        sensor = json.loads(Path(src["sensor"]).read_text(encoding="utf-8"))
        gw = {w["id"]: w for w in gold["works"]}
        for wid in src["ids"]:
            if a.only and wid not in a.only:
                continue
            w = gw[wid]
            txt = Path(src["text_dir"]) / f"{wid}.txt"
            seg = segment(NISHIKI, txt)
            scenes = seg["scenes"]
            preds = sensor["raw_outputs"][wid]["parsed_utterances"]
            flat = [(si, u) for si, sc in enumerate(scenes) for u in sc["utterances"]]
            assert len(flat) == len(preds) == len(w["utterances"]), (wid, len(flat), len(preds), len(w["utterances"]))
            ids = [c["id"] for c in w["characters"]]
            roster_out = roster_entries(w["characters"])
            narrator = w["narrator"] if w["narration"] == "first" else None
            pov = narrator or w.get("pov")
            anns, trace_all, audit_all, raw_all = [], [], [], []
            t0 = time.perf_counter()
            for si, sc in enumerate(scenes):
                scene_text = sc["scene_text"]
                prev_tail = scenes[si - 1]["scene_text"][-120:] if si > 0 else None
                msgs = build_scene_messages(w["characters"], scene_text, si, len(scenes), prev_tail)
                resp = call_ollama_chat(
                    MODEL, msgs, build_scene_format_schema(ids), num_ctx=9216, num_predict=3072, seed=42,
                    temperature=0.0, timeout_sec=300.0,
                )
                sents = [(x["index"], x["local"]["start"], x["local"]["end"]) for x in sc["sentences"]]  # stage 4: fact sentence_index
                cleaned, audit = validate_scene_output(resp.parsed_json, ids, scene_text, sentences=sents)
                sec, trace = cleaned["sections"], cleaned["trace"]
                utts = []
                for k, (sj, u) in enumerate(flat):
                    if sj != si:
                        continue
                    p = preds[k]
                    assert p["utterance_index"] == k
                    chars = list(scene_text)
                    quote = "".join(chars[u["local"]["start"] - 1 : u["local"]["end"] + 1])
                    gu = w["utterances"][k]
                    if "quote_text" in gu:
                        assert quote == gu["quote_text"], (wid, k)
                    else:  # F9 gold stores only the head of each quote
                        m = min(len(quote) - 1, len(gu["head"]), 6); assert quote[1 : 1 + m] == gu["head"][:m], (wid, k, quote[:12], gu["head"][:6])
                    pol = classify_politeness_surface(quote, tok)
                    utts.append(
                        {
                            "utterance_index": u["index"],
                            "speaker_id": p["speaker_id"],
                            "addressee_id": p["addressee_id"],
                            "evidence": p["evidence"],
                            "register": {"politeness": POLITE_MAP[pol["politeness"]], "first_person": p["register"]["first_person"]},
                        }
                    )
                ann = {
                    "contract": {
                        "schema_version": "sensor.v2",
                        "splitter_version": seg["splitter"],
                        "scene_id": f"{wid}:s{si}",
                        "scene_hash": sc["scene_hash"],
                        "ledger_hash": f"gold-roster:{gold_sha[:16]}",
                        "model_id": MODEL,
                        "prompt_hash": f"utt:{sensor['prompt_hash'][:16]}+scene:{pr_hash[:16]}",
                        "work_id": wid,
                        "scene_index": si,
                        "scene_count": len(scenes),
                        "char_len": sc["char_len"],
                        "seed": 42,
                    },
                    "scene_meta": {
                        "time": sec["time"],
                        "location_id": sec["location_id"],
                        "present_character_ids": sec["present_character_ids"],
                        "pov_character": pov,
                    },
                    "utterances": utts,
                    "character_states": sec["character_states"],
                    "fact_updates": sec["fact_updates"],
                }
                errs = validate_annotation_schema(ann, schema, ids)
                audit["schema_errors"] = errs
                audit["scene_id"] = ann["contract"]["scene_id"]
                audit["elapsed_sec"] = resp.elapsed_sec
                anns.append(ann)
                trace_all.append({"scene_id": ann["contract"]["scene_id"], **trace})
                audit_all.append(audit)
                raw_all.append({"scene_id": ann["contract"]["scene_id"], "raw_content": resp.raw_content, "parse_error": resp.parse_error,
                                "prompt_eval_count": resp.prompt_eval_count, "eval_count": resp.eval_count, "format_mode": resp.format_mode})
                print(wid, si, audit["kept"], "rej", len(audit["rejected"]), "schema_err", len(errs), f"{resp.elapsed_sec}s", flush=True)
            (out_dir / f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (out_dir / f"{wid}.characters.json").write_text(json.dumps(roster_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (out_dir / f"{wid}.trace.json").write_text(json.dumps({"traces": trace_all, "audits": audit_all, "raw": raw_all}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["works"][wid] = {
                "group": group,
                "scenes": len(scenes),
                "elapsed_sec": round(time.perf_counter() - t0, 1),
                "kept": {k: sum(x["kept"][k] for x in audit_all) for k in ("present", "states", "facts")},
                "rejected": sum(len(x["rejected"]) for x in audit_all),
                "rejected_reasons": {},
                "schema_errors": sum(len(x["schema_errors"]) for x in audit_all),
                "parse_failures": sum(1 for x in audit_all if not x["parse_ok"]),
            }
            rr = summary["works"][wid]["rejected_reasons"]
            for x in audit_all:
                for r in x["rejected"]:
                    key = f"{r['kind']}:{r['reason']}"
                    rr[key] = rr.get(key, 0) + 1
    summary["total_elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "stageA_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
