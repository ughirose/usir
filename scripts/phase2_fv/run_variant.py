"""One-shot re-annotation of the gold scenes with a prompt variant (control | fact_v2 | values_ctx).

Writes annotations in the Stage A format (utterance part copied from the baseline) so that
scripts/phase2_stage2/measure_stage_b.py works unchanged. seed 42, temperature 0, qwen2.5:14b (same as baseline).
"""
import argparse
import json
import time
from pathlib import Path

from common import *  # noqa: F401,F403
from prompts_fv import *  # noqa: F401,F403
from usir_py.sensor_eval.ollama_client import call_ollama_chat
from usir_py.sensor_eval.scene_annotator import build_scene_format_schema, validate_scene_output

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["control", "fact_v2", "values_ctx"])
ap.add_argument("out_dir")
a = ap.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
W = load_works()
G = load_gold()
need: dict[str, set[int]] = {}
for sid in G:
    wid, si = sid.split(":s")
    need.setdefault(wid, set()).add(int(si))
SYS = {"control": None, "fact_v2": SYSTEM_PROMPT_FACT_V2, "values_ctx": SYSTEM_PROMPT_VALUES_CTX}[a.variant]
t_all = time.perf_counter()
log = {"variant": a.variant, "model": "qwen2.5:14b", "seed": 42, "temperature": 0.0, "scenes_called": 0}
for wid, w in W.items():
    base = json.loads((S2 / "annotations" / f"{wid}.annotations.json").read_text(encoding="utf-8"))
    ids = [c["id"] for c in w["characters"]]
    scenes = w["scenes"]
    history, anns, traces, raws = [], [], [], []
    for si, sc in enumerate(scenes):
        ann = json.loads(json.dumps(base[si]))
        # values_ctx runs every scene (history needs the preceding scenes); the others run only the gold scenes
        if a.variant != "values_ctx" and si not in need.get(wid, ()):
            anns.append(ann)
            continue
        prev_tail = scenes[si - 1]["scene_text"][-120:] if si > 0 else None
        msgs = build_messages(a.variant, w["roster_out"], sc["scene_text"], si, len(scenes), prev_tail, w["characters"], history)
        resp = call_ollama_chat("qwen2.5:14b", msgs, build_scene_format_schema(ids), num_ctx=9216, num_predict=3072, seed=42, temperature=0.0, timeout_sec=300.0)
        cleaned, audit = validate_scene_output(resp.parsed_json, ids, sc["scene_text"])
        sec = cleaned["sections"]
        ann["scene_meta"].update({"time": sec["time"], "location_id": sec["location_id"], "present_character_ids": sec["present_character_ids"]})
        ann["character_states"] = sec["character_states"]
        ann["fact_updates"] = sec["fact_updates"]
        anns.append(ann)
        traces.append({"scene_id": ann["contract"]["scene_id"], **cleaned["trace"]})
        raws.append({"scene_id": ann["contract"]["scene_id"], "raw_content": resp.raw_content, "parse_error": resp.parse_error,
                     "rejected": audit["rejected"], "elapsed_sec": resp.elapsed_sec, "prompt_eval_count": resp.prompt_eval_count})
        history.append({"scene": si, "states": sec["character_states"], "facts": sec["fact_updates"]})
        log["scenes_called"] += 1
        print(wid, si, "facts", len(sec["fact_updates"]), "states", len(sec["character_states"]), "rej", len(audit["rejected"]), f"{resp.elapsed_sec}s", flush=True)
    (out / f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / f"{wid}.trace.json").write_text(json.dumps({"traces": traces, "raw": raws}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
log["total_elapsed_sec"] = round(time.perf_counter() - t_all, 1)
log["prompt_sha256"] = prompt_hash(SYS) if SYS else "scene_v1 (unchanged)"
(out / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("done", log["total_elapsed_sec"])
