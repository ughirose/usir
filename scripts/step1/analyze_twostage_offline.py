"""Offline analysis of Stage 2 deterministic post-processing on saved v5 Dev outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.sensor_eval.f9_loader import load_f9_dataset
from usir_py.sensor_eval.features import extract_work_utterance_features
from usir_py.sensor_eval.postprocess import (
    PostprocessConfig,
    analyze_inline_non_speech_thresholds,
    replay_saved_result_with_postprocess,
)

THRESHOLDS: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 15, 20)


def _compact_summary(replay: dict[str, Any]) -> dict[str, Any]:
    ov = replay["metrics"]["overall"]
    return {
        "postprocess": replay["postprocess"],
        "audit_totals": {
            "inline_non_speech_forced_count": replay["audit"]["inline_non_speech_forced_count"],
            "inline_non_speech_changed_count": replay["audit"]["inline_non_speech_changed_count"],
            "explicit_downgraded_count": replay["audit"]["explicit_downgraded_count"],
        },
        "explicit_error_rate": ov["explicit_error_rate"],
        "explicit_errors": ov["explicit_errors"],
        "explicit_denominator": ov["explicit_denominator"],
        "explicit_coverage": ov["explicit_coverage"],
        "overall_error_rate": ov["overall_error_rate"],
        "overall_errors": ov["overall_errors"],
        "overall_error_rate_raw_speaker": ov["overall_error_rate_raw_speaker"],
        "overall_errors_raw_speaker": ov["overall_errors_raw_speaker"],
        "attributed_total": ov["attributed_total"],
        "politeness_agreement_rate": ov["politeness_agreement_rate"],
        "ambiguous_rate": ov["ambiguous_rate"],
        "ambiguous_attributed_count": ov["ambiguous_attributed_count"],
        "ambiguous_rate_all_quotes": ov["ambiguous_rate_all_quotes"],
        "ambiguous_all_quotes_count": ov["ambiguous_all_quotes_count"],
        "not_speech_abstention_rate": ov["not_speech_abstention_rate"],
        "not_speech_abstained": ov["not_speech_abstained"],
        "not_speech_total": ov["not_speech_total"],
        "undetermined_abstention_rate": ov["undetermined_abstention_rate"],
        "undetermined_abstained": ov["undetermined_abstained"],
        "undetermined_total": ov["undetermined_total"],
        "by_evidence": ov["by_evidence"],
        "by_narration": {
            nar: {
                "explicit_error_rate": nb["explicit_error_rate"],
                "explicit_errors": nb["explicit_errors"],
                "explicit_denominator": nb["explicit_denominator"],
                "overall_error_rate": nb["overall_error_rate"],
                "overall_errors": nb["overall_errors"],
                "attributed_total": nb["attributed_total"],
            }
            for nar, nb in replay["metrics"]["by_narration"].items()
        },
        "by_work": [
            {
                "work_id": bw["work_id"],
                "title": bw["title"],
                "narration": bw["narration"],
                "explicit_errors": bw["explicit_errors"],
                "explicit_denominator": bw["explicit_denominator"],
                "explicit_error_rate": bw["explicit_error_rate"],
                "overall_errors": bw["overall_errors"],
                "attributed_total": bw["attributed_total"],
                "overall_error_rate": bw["overall_error_rate"],
                "inferred_errors": bw["by_evidence"]["inferred"]["errors"],
                "inferred_count": bw["by_evidence"]["inferred"]["count"],
                "ambiguous_attributed_count": bw["ambiguous_attributed_count"],
                "not_speech_abstained": bw["not_speech_abstained"],
                "not_speech_total": bw["not_speech_total"],
                "undetermined_abstained": bw["undetermined_abstained"],
                "undetermined_total": bw["undetermined_total"],
            }
            for bw in replay["metrics"]["by_work"]
        ],
        "gate": replay["gate"],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gold_dir = Path(r"C:\dev\nishiki\crates\nishiki-discourse\tests\data\f9_gold")
    split_file = REPO_ROOT / "scripts" / "step1" / "f9_split.json"
    v5_result_path = REPO_ROOT / "scripts" / "step1" / "results" / "v5" / "qwen2.5_14b_dev.json"
    out_path = (
        REPO_ROOT / "scripts" / "step1" / "results" / "v5_twostage" / "offline_analysis_dev.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_f9_dataset(gold_dir=gold_dir, split_file=split_file, split="dev")
    saved_v5 = json.loads(v5_result_path.read_text(encoding="utf-8"))

    baseline_replay = replay_saved_result_with_postprocess(
        dataset, saved_v5, PostprocessConfig()
    )
    rule_a_replay = replay_saved_result_with_postprocess(
        dataset,
        saved_v5,
        PostprocessConfig(explicit_evidence_gate=True, inline_non_speech_filter=False),
    )

    # Breakdown of Rule (A) downgrades by gold status and correctness.
    a_downgrades: list[dict[str, Any]] = []
    for work in dataset.works:
        gold_by_n = {u.n: u for u in work.utterances}
        s1_by_n = {
            u["utterance_index"]: u
            for u in saved_v5["raw_outputs"][work.work_id]["parsed_utterances"]
        }
        w_audit = rule_a_replay["audit"]["per_work"][work.work_id]["explicit_downgraded"]
        for rec in w_audit:
            idx = rec["utterance_index"]
            gu = gold_by_n[idx]
            s1 = s1_by_n[idx]
            a_downgrades.append(
                {
                    "work_id": work.work_id,
                    "title": work.title,
                    "utterance_index": idx,
                    "gold_status": gu.status,
                    "gold_speaker": gu.speaker,
                    "pred_speaker": s1["speaker_id"],
                    "stage1_correct": (
                        s1["speaker_id"] == gu.speaker if gu.status == "attributed" else False
                    ),
                }
            )

    inline_quotes: list[dict[str, Any]] = []
    for work in dataset.works:
        feats = extract_work_utterance_features(work)
        s1_by_n = {
            u["utterance_index"]: u
            for u in saved_v5["raw_outputs"][work.work_id]["parsed_utterances"]
        }
        for gu, feat in zip(work.utterances, feats):
            if feat.position != "inline_embedded":
                continue
            s1 = s1_by_n[gu.n]
            ctx = work.text[
                max(0, gu.open_offset - 25) : min(len(work.text), gu.close_offset + 25)
            ].replace("\n", "\\n")
            inline_quotes.append(
                {
                    "work_id": work.work_id,
                    "title": work.title,
                    "utterance_index": gu.n,
                    "status": gu.status,
                    "gold_speaker": gu.speaker,
                    "quote_chars": feat.quote_chars,
                    "pre_speech_verb": feat.pre_speech_verb,
                    "post_speech_verb": feat.post_speech_verb,
                    "pre_speaker_and_verb": feat.pre_speaker_and_verb,
                    "post_speaker_and_verb": feat.post_speaker_and_verb,
                    "v5_speaker": s1["speaker_id"],
                    "v5_evidence": s1["evidence"],
                    "quote_text": gu.quote_text,
                    "context": ctx,
                }
            )

    sweep_b_only = analyze_inline_non_speech_thresholds(
        dataset, saved_v5, THRESHOLDS, include_explicit_gate=False
    )
    sweep_ab = analyze_inline_non_speech_thresholds(
        dataset, saved_v5, THRESHOLDS, include_explicit_gate=True
    )

    payload = {
        "source_result_file": str(v5_result_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_spec": saved_v5["model_spec"],
        "prompt_version": saved_v5["prompt_version"],
        "prompt_hash": saved_v5["prompt_hash"],
        "v5_baseline": _compact_summary(baseline_replay),
        "v5_plus_rule_a": _compact_summary(rule_a_replay),
        "rule_a_downgrade_breakdown": {
            "total": len(a_downgrades),
            "attributed_correct": sum(
                1 for d in a_downgrades if d["gold_status"] == "attributed" and d["stage1_correct"]
            ),
            "attributed_wrong": sum(
                1
                for d in a_downgrades
                if d["gold_status"] == "attributed" and not d["stage1_correct"]
            ),
            "undetermined": sum(1 for d in a_downgrades if d["gold_status"] == "undetermined"),
            "not_speech": sum(1 for d in a_downgrades if d["gold_status"] == "not_speech"),
            "details": a_downgrades,
        },
        "inline_embedded_quotes": inline_quotes,
        "rule_b_threshold_sweep_b_only": sweep_b_only,
        "rule_b_threshold_sweep_a_plus_b": sweep_ab,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
