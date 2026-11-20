"""Round 6 offline error analysis (0 LLM calls) and Stage 3 offline synthesis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.sensor_eval.f9_loader import load_f9_dataset
from usir_py.sensor_eval.features import extract_work_utterance_features
from usir_py.sensor_eval.run_step1_f9 import load_eval_config
from usir_py.sensor_eval.speaker_recheck import (
    SpeakerRecheckConfig,
    extract_in_quote_aliases,
    is_recheck_target,
    replay_saved_result_with_speaker_recheck,
)

DEFAULT_TWOS_STAGE_PATH = (
    REPO_ROOT / "scripts" / "step1" / "results" / "v5_twostage" / "qwen2.5_14b_dev.json"
)
DEFAULT_THREESTAGE_CONFIG_PATH = (
    REPO_ROOT / "scripts" / "step1" / "step1_eval_config_v5_threestage.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "scripts" / "step1" / "results" / "v5_threestage"


def run_error_analysis(
    gold_dir: Path,
    split_file: Path,
    saved_twostage_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Analyze the Round 5 Dev errors and inferred targets offline without calling Ollama."""
    dataset = load_f9_dataset(gold_dir=gold_dir, split_file=split_file, split="dev")
    saved = json.loads(saved_twostage_path.read_text(encoding="utf-8"))
    works_by_id = {w.work_id: w for w in dataset.works}

    pred_maps: dict[str, dict[int, dict[str, Any]]] = {}
    feats_by_work = {w.work_id: extract_work_utterance_features(w) for w in dataset.works}
    for work in dataset.works:
        utts = (saved.get("raw_outputs") or {}).get(work.work_id, {}).get("parsed_utterances") or []
        pred_maps[work.work_id] = {
            item["utterance_index"]: item
            for item in utts
            if isinstance(item, dict) and isinstance(item.get("utterance_index"), int)
        }

    error_rows: list[dict[str, Any]] = []
    confusion_counter: Counter[tuple[str, str, str]] = Counter()
    inversion_count = 0

    for err in saved["metrics"]["errors"]:
        wid = err["work_id"]
        idx = int(err["utterance_index"])
        work = works_by_id[wid]
        gold_u = work.utterances[idx]
        p_map = pred_maps[wid]
        pred_u = p_map.get(idx) or {}
        feat = feats_by_work[wid][idx]

        pred_spk = pred_u.get("speaker_id")
        pred_addr = pred_u.get("addressee_id")
        is_inversion = (gold_u.speaker is not None and pred_addr == gold_u.speaker)
        if is_inversion:
            inversion_count += 1
        confusion_counter[(wid, str(gold_u.speaker), str(pred_spk))] += 1

        prev_u = p_map.get(idx - 1) if idx > 0 else None
        next_u = p_map.get(idx + 1) if idx + 1 < len(work.utterances) else None
        in_quote = extract_in_quote_aliases(gold_u.quote_text, work.characters)

        error_rows.append(
            {
                "work_id": wid,
                "title": work.title,
                "narration": work.narration,
                "utterance_index": idx,
                "gold_speaker": gold_u.speaker,
                "pred_speaker": pred_spk,
                "pred_addressee": pred_addr,
                "gold_equals_pred_addressee": is_inversion,
                "pred_evidence": pred_u.get("evidence"),
                "outcome": err["outcome"],
                "quote_text": gold_u.quote_text,
                "pre_speech_verb": feat.pre_speech_verb,
                "post_speech_verb": feat.post_speech_verb,
                "in_quote_aliases": [{"character_id": cid, "alias": al} for cid, al in in_quote],
                "prev_utterance": (
                    {
                        "utterance_index": idx - 1,
                        "gold_status": work.utterances[idx - 1].status,
                        "gold_speaker": work.utterances[idx - 1].speaker,
                        "pred_speaker": prev_u.get("speaker_id"),
                        "pred_addressee": prev_u.get("addressee_id"),
                        "pred_evidence": prev_u.get("evidence"),
                    }
                    if prev_u is not None
                    else None
                ),
                "next_utterance": (
                    {
                        "utterance_index": idx + 1,
                        "gold_status": work.utterances[idx + 1].status,
                        "gold_speaker": work.utterances[idx + 1].speaker,
                        "pred_speaker": next_u.get("speaker_id"),
                        "pred_addressee": next_u.get("addressee_id"),
                        "pred_evidence": next_u.get("evidence"),
                    }
                    if next_u is not None
                    else None
                ),
            }
        )

    confusion_pairs = [
        {
            "work_id": wid,
            "title": works_by_id[wid].title,
            "gold_speaker": g_spk,
            "pred_speaker": p_spk,
            "count": count,
        }
        for (wid, g_spk, p_spk), count in sorted(confusion_counter.items())
    ]

    inferred_per_work: list[dict[str, Any]] = []
    total_targets = 0
    total_targets_attr = 0
    total_targets_undet = 0
    total_targets_not_speech = 0

    for work in dataset.works:
        p_map = pred_maps[work.work_id]
        t_all = 0
        t_attr = 0
        t_undet = 0
        t_ns = 0
        for u in work.utterances:
            pred_u = p_map.get(u.n)
            if is_recheck_target(pred_u):
                t_all += 1
                if u.status == "attributed":
                    t_attr += 1
                elif u.status == "undetermined":
                    t_undet += 1
                elif u.status == "not_speech":
                    t_ns += 1
        total_targets += t_all
        total_targets_attr += t_attr
        total_targets_undet += t_undet
        total_targets_not_speech += t_ns
        inferred_per_work.append(
            {
                "work_id": work.work_id,
                "title": work.title,
                "narration": work.narration,
                "total_quotes": len(work.utterances),
                "recheck_targets_total": t_all,
                "recheck_targets_attributed": t_attr,
                "recheck_targets_undetermined": t_undet,
                "recheck_targets_not_speech": t_ns,
            }
        )

    by_work = [
        {
            "work_id": bw["work_id"],
            "title": bw["title"],
            "narration": bw["narration"],
            "attributed_total": bw["attributed_total"],
            "overall_errors": bw["overall_errors"],
            "overall_error_rate": bw["overall_error_rate"],
            "explicit_errors": bw["explicit_errors"],
            "explicit_denominator": bw["explicit_denominator"],
            "inferred_errors": bw["by_evidence"]["inferred"]["errors"],
            "inferred_count": bw["by_evidence"]["inferred"]["count"],
            "not_speech_abstained": bw["not_speech_abstained"],
            "not_speech_total": bw["not_speech_total"],
            "undetermined_abstained": bw["undetermined_abstained"],
            "undetermined_total": bw["undetermined_total"],
        }
        for bw in saved["metrics"]["by_work"]
    ]

    ov = saved["metrics"]["overall"]
    payload = {
        "source_result_file": str(saved_twostage_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "split": saved["split"],
        "model_spec": saved["model_spec"],
        "overall": {
            "explicit_error_rate": ov["explicit_error_rate"],
            "explicit_errors": ov["explicit_errors"],
            "explicit_denominator": ov["explicit_denominator"],
            "overall_error_rate": ov["overall_error_rate"],
            "overall_errors": ov["overall_errors"],
            "attributed_total": ov["attributed_total"],
            "wrong_speaker_count": ov["error_breakdown"]["wrong_speaker"],
        },
        "error_count": len(error_rows),
        "addressee_inversion_count": inversion_count,
        "confusion_pairs": confusion_pairs,
        "by_work": by_work,
        "inferred_target_counts": {
            "total": total_targets,
            "attributed": total_targets_attr,
            "undetermined": total_targets_undet,
            "not_speech": total_targets_not_speech,
            "per_work": inferred_per_work,
        },
        "errors": error_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def run_offline_synthesis(
    config_path: Path,
    saved_twostage_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Run Stage 3 speaker recheck offline on saved Round 5 Stage 1+2 Dev predictions."""
    cfg = load_eval_config(config_path)
    dataset = load_f9_dataset(
        gold_dir=cfg["gold_dir"],
        split_file=cfg["split_file"],
        split="dev",
    )
    saved = json.loads(saved_twostage_path.read_text(encoding="utf-8"))
    sr_cfg = SpeakerRecheckConfig.from_dict(cfg.get("speaker_recheck"))
    if not sr_cfg.enabled:
        raise ValueError(f"speaker_recheck.enabled must be true in {config_path}")

    replay = replay_saved_result_with_speaker_recheck(
        dataset=dataset,
        saved_report=saved,
        config=sr_cfg,
        ollama_url=str(cfg.get("ollama_url", "http://127.0.0.1:11434")),
    )
    payload = {
        "source_result_file": str(saved_twostage_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "config_file": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "split": dataset.split,
        "gold_json_sha256": dataset.gold_json_sha256,
        **replay,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Round 6 offline analysis and Stage 3 synthesis.")
    parser.add_argument(
        "--mode",
        choices=["error-analysis", "offline-synthesis"],
        required=True,
    )
    parser.add_argument(
        "--saved-twostage",
        default=str(DEFAULT_TWOS_STAGE_PATH),
        help="Path to saved Round 5 Stage 1+2 Dev JSON",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_THREESTAGE_CONFIG_PATH),
        help="Path to Round 6 3-stage config JSON",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for JSON analysis artifacts",
    )
    args = parser.parse_args(argv)

    saved_path = Path(args.saved_twostage)
    cfg_path = Path(args.config)
    out_dir = Path(args.out_dir)

    if args.mode == "error-analysis":
        cfg = load_eval_config(cfg_path)
        payload = run_error_analysis(
            gold_dir=Path(cfg["gold_dir"]),
            split_file=Path(cfg["split_file"]),
            saved_twostage_path=saved_path,
            out_path=out_dir / "error_analysis_dev.json",
        )
        print(
            json.dumps(
                {
                    "mode": "error-analysis",
                    "out_file": str(out_dir / "error_analysis_dev.json"),
                    "error_count": payload["error_count"],
                    "addressee_inversion_count": payload["addressee_inversion_count"],
                    "inferred_target_counts": payload["inferred_target_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    payload = run_offline_synthesis(
        config_path=cfg_path,
        saved_twostage_path=saved_path,
        out_path=out_dir / "offline_synthesis_dev.json",
    )
    b_ov = payload["before_metrics"]["overall"]
    a_ov = payload["after_metrics"]["overall"]
    print(
        json.dumps(
            {
                "mode": "offline-synthesis",
                "out_file": str(out_dir / "offline_synthesis_dev.json"),
                "recheck_summary": payload["recheck_summary"],
                "before_overall_errors": b_ov["overall_errors"],
                "before_overall_error_rate": b_ov["overall_error_rate"],
                "after_explicit_errors": a_ov["explicit_errors"],
                "after_explicit_denominator": a_ov["explicit_denominator"],
                "after_explicit_error_rate": a_ov["explicit_error_rate"],
                "after_overall_errors": a_ov["overall_errors"],
                "after_overall_error_rate": a_ov["overall_error_rate"],
                "after_gate": payload["after_gate"],
                "fixed_errors": payload["diff"]["fixed_errors"],
                "regressions": payload["diff"]["regressions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
