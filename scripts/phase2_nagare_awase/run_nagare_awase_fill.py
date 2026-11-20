"""Fill scene_meta.nagare / awase of the stage4 annotations (7 works, 35 scenes) with one extra call per scene.

Frozen configuration reused unchanged: qwen2.5:14b, seed 42, temperature 0, num_ctx 9216 (as phase2_stage2 Stage A), same segmenter.
Existing annotation content (utterances / states / facts) is copied verbatim from scripts/phase2_stage4/annotations.
Usage: python run_nagare_awase_fill.py <out_dir> [--only WORK_ID ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts/phase2"))
sys.path.insert(0, str(HERE))
from build_annotations import segment  # noqa: E402
from usir_py.sensor_eval.ollama_client import call_ollama_chat  # noqa: E402
from usir_py.sensor_eval.scene_annotator import validate_annotation_schema  # noqa: E402
import nagare_awase_prompt as P  # noqa: E402

NISHIKI = "C:/dev/nishiki-phase2/target/release/nishiki.exe"
SCHEMA = REPO / "schemas/sensor_v2.schema.json"
SRC = REPO / "scripts/phase2_stage4/annotations"
TEXT_DIRS = [REPO / "scripts/step1/modern_gold", Path("C:/dev/nishiki/crates/nishiki-discourse/tests/data/f9_gold")]
WORKS = ["000035_000301", "000050_003586", "000081_000456", "001383_056643", "001798_056779", "001801_001786", "001875_059897"]
MODEL = "qwen2.5:14b"
MAX_RETRY = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    t_all = time.perf_counter()
    sp = out / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if a.only and sp.exists() else {"model": MODEL, "prompt_hash": P.prompt_hash(), "works": {}}
    for wid in WORKS:
        if a.only and wid not in a.only:
            continue
        txt = next(d / f"{wid}.txt" for d in TEXT_DIRS if (d / f"{wid}.txt").exists())
        scenes = segment(NISHIKI, txt)["scenes"]
        anns = json.loads((SRC / f"{wid}.annotations.json").read_text(encoding="utf-8"))
        roster = [c["id"] for c in json.loads((SRC / f"{wid}.characters.json").read_text(encoding="utf-8"))]
        assert len(scenes) == len(anns), (wid, len(scenes), len(anns))
        trace, nag, awa, t0 = [], 0, 0, time.perf_counter()
        for si, (sc, ann) in enumerate(zip(scenes, anns)):
            assert sc["scene_hash"] == ann["contract"]["scene_hash"], (wid, si)
            msgs = P.build_messages(sc["scene_text"], si, len(scenes))
            attempts, best = [], None
            for attempt in range(1 + MAX_RETRY):
                resp = call_ollama_chat(MODEL, msgs, P.format_schema(), num_ctx=9216, num_predict=2048, seed=42,
                                        temperature=0.0, timeout_sec=300.0)
                nslot, aslot, audit = P.validate(resp.parsed_json, sc["scene_text"])
                filled = (nslot is not None) + (aslot is not None)
                attempts.append({"attempt": attempt, "nagare": nslot is not None, "awase": aslot is not None, "rejected": audit["rejected"]})
                if best is None or filled > best[0]:
                    best = (filled, resp, nslot, aslot, audit)
                if filled == 2:
                    break
                fb = "; ".join(sorted({f"{r['kind']}:{r['reason']}" for r in audit["rejected"]})) or "parse_failure"
                msgs = msgs + [{"role": "assistant", "content": resp.raw_content},
                               {"role": "user", "content": f"前回の出力は検証で却下されました（{fb}）。コードは重複させず3つ異なるものにし、basis は本文からそのままコピーした語句にして、全体を出力し直してください。"}]
            _, resp, nslot, aslot, audit = best
            audit["attempts"] = attempts
            ann["scene_meta"]["nagare"] = nslot
            ann["scene_meta"]["awase"] = aslot
            ann["contract"]["prompt_hash"] += f"+nagare_awase:{P.prompt_hash()[:16]}"
            errs = validate_annotation_schema(ann, schema, roster)
            audit["schema_errors"] = errs
            nag += nslot is not None
            awa += aslot is not None
            trace.append({"scene_id": ann["contract"]["scene_id"], "elapsed_sec": resp.elapsed_sec, "raw_content": resp.raw_content,
                          "parse_error": resp.parse_error, "prompt_eval_count": resp.prompt_eval_count,
                          "eval_count": resp.eval_count, **audit})
            print(wid, si, "nagare" if nslot else "NAGARE-NULL", "awase" if aslot else "AWASE-NULL", "schema_err", len(errs),
                  f"{resp.elapsed_sec}s", flush=True)
        (out / f"{wid}.annotations.json").write_text(json.dumps(anns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / f"{wid}.trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["works"][wid] = {"scenes": len(scenes), "nagare_filled": nag, "awase_filled": awa,
                                 "elapsed_sec": round(time.perf_counter() - t0, 1),
                                 "schema_errors": sum(len(t["schema_errors"]) for t in trace)}
    summary["total_elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
