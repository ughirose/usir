"""Generate the local-side utterance annotation for one work with the Step 1 three-stage pipeline
(stage 1 qwen2.5:14b JSON extraction -> stage 2 deterministic postprocess -> stage 3 per-utterance speaker recheck),
using the settings of scripts/step1/step1_eval_config_v5_threestage.json, and write
scripts/sensor_runner/local/<work>.utterances.json (sensor.local_utterances/1: scene_hash -> [utterance]).

Gold is read only by the Step 1 loader for outer-quote numbering (the pipeline's predictions do not use gold speakers).
Usage: python local_from_step1.py WORK_ID [--out-dir DIR] [--local-out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, find_nishiki, segment, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "python"))
from usir_py.sensor_eval.f9_loader import load_f9_dataset  # noqa: E402
from usir_py.sensor_eval.postprocess import PostprocessConfig  # noqa: E402
from usir_py.sensor_eval.run_step1_f9 import load_eval_config, run_evaluation  # noqa: E402
from usir_py.sensor_eval.speaker_recheck import SpeakerRecheckConfig  # noqa: E402

CONFIG = REPO / "scripts" / "step1" / "step1_eval_config_v5_threestage.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_id")
    ap.add_argument("--out-dir", default=str(HERE / "results" / "local_step1"))
    ap.add_argument("--local-out", default=None)
    a = ap.parse_args()

    cfg = load_eval_config(CONFIG)
    ds = load_f9_dataset(gold_dir=cfg["gold_dir"], split_file=cfg["split_file"], split="holdout", unlock_holdout=True)
    works = [w for w in ds.works if w.work_id == a.work_id]
    if not works:
        raise SystemExit(f"work {a.work_id} not in holdout split")
    ds = type(ds)(split=ds.split, gold_dir=ds.gold_dir, gold_json_sha256=ds.gold_json_sha256, split_version=ds.split_version, works=works)

    model_spec = cfg["models"][0]
    run_evaluation(
        dataset=ds,
        model_specs=[model_spec],
        out_dir=Path(a.out_dir),
        ollama_url=cfg["ollama_url"],
        num_ctx=cfg["num_ctx"],
        num_predict=cfg["num_predict"],
        seed=cfg["seed"],
        timeout_sec=cfg["timeout"],
        dry_run=False,
        prompt_version=cfg["prompt_version"],
        per_model_options=cfg.get("per_model_options"),
        postprocess=PostprocessConfig.from_dict(cfg["postprocess"]),
        speaker_recheck=SpeakerRecheckConfig.from_dict(cfg["speaker_recheck"]),
    )
    res = json.loads(next(Path(a.out_dir).glob("*_holdout.json")).read_text(encoding="utf-8"))
    preds = res["raw_outputs"][a.work_id]["parsed_utterances"]
    by_n = {int(u["utterance_index"]): u for u in preds or []}

    # map work-wide quote number -> (scene_hash, scene-local utterance index) by segment order
    seg = segment(find_nishiki(), Path(json.loads((HERE / "works" / f"{a.work_id}.work.json").read_text(encoding="utf-8"))["text_path"]))
    scenes: dict[str, list[dict]] = {}
    n = 0
    for s in seg["scenes"]:
        rows = []
        for u in s["utterances"]:
            p = by_n.get(n)
            rows.append({
                "utterance_index": u["index"],
                "speaker_id": p.get("speaker_id") if p else None,
                "addressee_id": p.get("addressee_id") if p else None,
                "evidence": (p.get("evidence") if p else None) or "ambiguous",
            })
            n += 1
        scenes[s["scene_hash"]] = rows
    if n != len(ds.works[0].utterances):
        raise SystemExit(f"quote count mismatch: segment={n} gold-loader={len(ds.works[0].utterances)}")
    out = Path(a.local_out) if a.local_out else HERE / "local" / f"{a.work_id}.utterances.json"
    write_json(out, {
        "schema": "sensor.local_utterances/1",
        "provenance": f"step1-threestage:{model_spec} (stage1 + postprocess + stage3 recheck_v2), config {CONFIG.name}, run {res['run_timestamp']}",
        "scenes": scenes,
    })
    print(f"wrote {out} utterances={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
