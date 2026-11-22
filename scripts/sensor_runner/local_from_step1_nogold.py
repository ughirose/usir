"""Local-side utterance annotation for a work that has NO F9 gold, using the Step 1 three-stage pipeline
(stage 1 qwen2.5:14b JSON extraction -> stage 2 deterministic postprocess -> stage 3 per-utterance speaker recheck),
with the settings of scripts/step1/step1_eval_config_v5_threestage.json.

Difference from local_from_step1.py: no gold is read at all. The work is cut into scene-aligned chunks
(<= --max-chars characters and <= --max-quotes outer quotes) because the frozen config (num_ctx 9216, num_predict 4096)
cannot take a whole 30k-character work in one call. Each chunk is run as one pseudo-work with placeholder gold
(status=undetermined, speaker=None); predictions never use the placeholder.
Writes scripts/sensor_runner/local/<work>.utterances.json (sensor.local_utterances/1) after every chunk, and
raw per-chunk records to --out-dir.

Usage: python local_from_step1_nogold.py WORK_ID [--max-chars 6000] [--max-quotes 55] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, find_nishiki, load_work, segment, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "python"))
from usir_py.sensor_eval.f9_loader import F9Character, F9UtteranceGold, F9Work, number_outer_quotes  # noqa: E402
from usir_py.sensor_eval.ollama_client import call_ollama_chat, parse_model_spec  # noqa: E402
from usir_py.sensor_eval.postprocess import PostprocessConfig, apply_postprocessing_to_work  # noqa: E402
from usir_py.sensor_eval.prompt import build_ollama_utterances_format_schema, build_work_messages  # noqa: E402
from usir_py.sensor_eval.run_step1_f9 import _extract_utterances, load_eval_config  # noqa: E402
from usir_py.sensor_eval.speaker_recheck import SpeakerRecheckConfig, apply_speaker_recheck_to_work  # noqa: E402

CONFIG = REPO / "scripts" / "step1" / "step1_eval_config_v5_threestage.json"


def make_chunks(scenes, max_chars, max_quotes):
    chunks, cur, c_chars, c_q = [], [], 0, 0
    for s in scenes:
        nq, nc = len(s["utterances"]), len(s["scene_text"])
        if cur and (c_chars + nc > max_chars or c_q + nq > max_quotes):
            chunks.append(cur)
            cur, c_chars, c_q = [], 0, 0
        cur.append(s)
        c_chars += nc
        c_q += nq
    if cur:
        chunks.append(cur)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_id")
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-quotes", type=int, default=55)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--local-out", default=None)
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else HERE / "results" / "local_step1" / a.work_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_eval_config(CONFIG)
    model_spec = cfg["models"][0]
    model_name, think, _ = parse_model_spec(model_spec)
    opts = cfg["per_model_options"].get(model_spec, {})
    num_ctx, num_predict = int(opts.get("num_ctx", cfg["num_ctx"])), int(opts.get("num_predict", cfg["num_predict"]))
    seed, timeout = int(opts.get("seed", cfg["seed"])), float(opts.get("timeout", cfg["timeout"]))
    pp_cfg = PostprocessConfig.from_dict(cfg["postprocess"])
    sr_cfg = SpeakerRecheckConfig.from_dict(cfg["speaker_recheck"])
    fmt = build_ollama_utterances_format_schema()

    work = load_work(a.work_id)
    seg = segment(find_nishiki(), work["text_abs"])
    chars = [F9Character(id=c["id"], aliases=list(c.get("aliases", [])), gender=str(c.get("gender", "Unknown"))) for c in work["characters"]]
    chunks = make_chunks(seg["scenes"], a.max_chars, a.max_quotes)
    print(f"scenes={len(seg['scenes'])} chunks={len(chunks)} quotes={sum(len(s['utterances']) for s in seg['scenes'])}", flush=True)

    local_out = Path(a.local_out) if a.local_out else HERE / "local" / f"{a.work_id}.utterances.json"
    scenes_out: dict[str, list[dict]] = {}
    totals = {"elapsed_wall_s": 0.0, "stage1_s": 0.0, "stage3_s": 0.0, "stage1_prompt_tok": 0, "stage1_eval_tok": 0,
              "stage3_prompt_tok": 0, "stage3_eval_tok": 0, "pp_forced": 0, "pp_changed": 0, "pp_downgraded": 0,
              "sr_target": 0, "sr_rechecked": 0, "sr_changed": 0, "sr_rejected": 0, "ctx_warn_chunks": 0, "parse_error_chunks": 0}
    chunk_log = []
    t_all = time.time()
    for ci, ch in enumerate(chunks):
        t0 = time.time()
        text = "\n\n".join(s["scene_text"] for s in ch)
        numbered, spans = number_outer_quotes(text)
        exp = sum(len(s["utterances"]) for s in ch)
        if len(spans) != exp:
            raise SystemExit(f"chunk {ci}: outer quotes {len(spans)} != segment utterances {exp}")
        gold = [
            F9UtteranceGold(n=i, open_offset=sp[0], close_offset=sp[1], head="", status="undetermined", speaker=None, quote_text=text[sp[0]:sp[1]])
            for i, sp in enumerate(spans)
        ]
        w = F9Work(
            work_id=f"{a.work_id}#c{ci}", author=None, title=None, narration=str(work.get("narration") or "third"),
            narrator=work.get("narrator"), pov=work.get("pov"), characters=chars, text=text, numbered_text=numbered,
            scene_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(), utterances=gold,
        )
        resp = call_ollama_chat(model=model_name, messages=build_work_messages(w, prompt_version=cfg["prompt_version"]),
                                format_schema=fmt, ollama_url=cfg["ollama_url"], num_ctx=num_ctx, num_predict=num_predict,
                                seed=seed, temperature=0.0, timeout_sec=timeout, think=think)
        raw = _extract_utterances(resp.parsed_json)
        s2, pp_audit = apply_postprocessing_to_work(w, raw, pp_cfg) if pp_cfg.is_enabled else (raw, None)
        final, sr_audit = apply_speaker_recheck_to_work(w, s2, sr_cfg, ollama_url=cfg["ollama_url"]) if sr_cfg.is_enabled else (s2, None)
        by_n = {int(u["utterance_index"]): u for u in final or []}
        p_tok, e_tok = int(resp.prompt_eval_count or 0), int(resp.eval_count or 0)
        ctx_warn = bool(p_tok + e_tok >= num_ctx or e_tok >= num_predict)
        missing = sum(1 for i in range(exp) if i not in by_n)
        n = 0
        for s in ch:
            rows = []
            for u in s["utterances"]:
                p = by_n.get(n)
                rows.append({"utterance_index": u["index"], "speaker_id": p.get("speaker_id") if p else None,
                             "addressee_id": p.get("addressee_id") if p else None,
                             "evidence": (p.get("evidence") if p else None) or "ambiguous"})
                n += 1
            scenes_out[s["scene_hash"]] = rows
        dt = time.time() - t0
        totals["stage1_s"] += resp.elapsed_sec
        totals["stage1_prompt_tok"] += p_tok
        totals["stage1_eval_tok"] += e_tok
        if pp_audit:
            totals["pp_forced"] += len(pp_audit["inline_non_speech_forced"])
            totals["pp_changed"] += sum(1 for r in pp_audit["inline_non_speech_forced"] if r["changed"])
            totals["pp_downgraded"] += len(pp_audit["explicit_downgraded"])
        if sr_audit:
            totals["stage3_s"] += float(sr_audit["elapsed_sec"])
            totals["stage3_prompt_tok"] += int(sr_audit["prompt_eval_count"])
            totals["stage3_eval_tok"] += int(sr_audit["eval_count"])
            totals["sr_target"] += int(sr_audit["target_count"])
            totals["sr_rechecked"] += int(sr_audit["rechecked_count"])
            totals["sr_changed"] += int(sr_audit["changed_count"])
            totals["sr_rejected"] += int(sr_audit["rejected_count"])
        totals["ctx_warn_chunks"] += int(ctx_warn)
        totals["parse_error_chunks"] += int(bool(resp.parse_error))
        row = {"chunk": ci, "scenes": [s["index"] for s in ch], "chars": len(text), "quotes": exp, "missing_predictions": missing,
               "wall_s": round(dt, 1), "stage1_prompt_tok": p_tok, "stage1_eval_tok": e_tok, "ctx_warning": ctx_warn,
               "parse_error": resp.parse_error}
        chunk_log.append(row)
        write_json(out_dir / f"chunk_{ci:03d}.json", {**row, "stage1_parsed": raw, "final": final, "pp_audit": pp_audit, "sr_audit": sr_audit})
        write_json(local_out, {"schema": "sensor.local_utterances/1",
                               "provenance": f"step1-threestage-nogold:{model_spec} (chunked; stage1 + postprocess + stage3 recheck_v2), config {CONFIG.name}, chunks_done {ci + 1}/{len(chunks)}",
                               "scenes": scenes_out})
        print(json.dumps(row, ensure_ascii=False), flush=True)
    totals["elapsed_wall_s"] = round(time.time() - t_all, 1)
    write_json(out_dir / "summary.json", {"work_id": a.work_id, "model": model_spec, "num_ctx": num_ctx, "num_predict": num_predict,
                                           "max_chars": a.max_chars, "max_quotes": a.max_quotes, "chunks": chunk_log, "totals": totals})
    print("DONE", json.dumps(totals), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
