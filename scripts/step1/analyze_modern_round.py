"""Deterministic offline analysis for Track A Modern Japanese Round (0 LLM calls)."""

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
from usir_py.sensor_eval.metrics import (
    POLITENESS_3WAY_DECIDED,
    POLITENESS_5_TO_3_MAP,
    evaluate_model_predictions,
    evaluate_step1_gate,
)
from usir_py.sensor_eval.run_step1_f9 import load_eval_config

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "step1" / "step1_eval_config_modern_v5_threestage.json"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "scripts" / "step1" / "results" / "modern_v5_threestage" / "qwen2.5_14b_dev.json"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "scripts" / "step1" / "results" / "modern_v5_threestage" / "summary_dev.json"
)
DEFAULT_OUT_PATH = (
    REPO_ROOT
    / "scripts"
    / "step1"
    / "results"
    / "modern_v5_threestage"
    / "modern_round_analysis.json"
)
F9_DEV_SUMMARY_PATH = REPO_ROOT / "scripts" / "step1" / "results" / "v5_threestage" / "summary_dev.json"
F9_HOLDOUT_SUMMARY_PATH = (
    REPO_ROOT / "scripts" / "step1" / "results" / "v5_threestage" / "summary_holdout.json"
)


def _compact_stage_metrics(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    ov = metrics["overall"]
    return {
        "explicit_error_rate": ov["explicit_error_rate"],
        "explicit_errors": ov["explicit_errors"],
        "explicit_denominator": ov["explicit_denominator"],
        "overall_error_rate": ov["overall_error_rate"],
        "overall_errors": ov["overall_errors"],
        "overall_error_rate_raw_speaker": ov["overall_error_rate_raw_speaker"],
        "overall_errors_raw_speaker": ov["overall_errors_raw_speaker"],
        "attributed_total": ov["attributed_total"],
        "all_quotes_total": ov["all_quotes_total"],
        "inferred_error_rate": ov["by_evidence"]["inferred"]["error_rate"],
        "inferred_errors": ov["by_evidence"]["inferred"]["errors"],
        "inferred_count": ov["by_evidence"]["inferred"]["count"],
        "ambiguous_rate": ov["ambiguous_rate"],
        "ambiguous_attributed_count": ov["ambiguous_attributed_count"],
        "ambiguous_rate_all_quotes": ov["ambiguous_rate_all_quotes"],
        "ambiguous_all_quotes_count": ov["ambiguous_all_quotes_count"],
        "not_speech_abstention_rate": ov["not_speech_abstention_rate"],
        "not_speech_abstained": ov["not_speech_abstained"],
        "not_speech_total": ov["not_speech_total"],
        "politeness_agreement_rate": ov["politeness_agreement_rate"],
        "politeness_evaluation": ov.get("politeness_evaluation"),
        "by_narration": metrics["by_narration"],
        "by_work": metrics["by_work"],
        "gate": gate,
    }


def _index_by_utterance(items: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not items:
        return {}
    return {
        int(u["utterance_index"]): u
        for u in items
        if isinstance(u, dict) and isinstance(u.get("utterance_index"), int)
    }


def _compact_f9_comparison(summary_path: Path, label: str) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    models = raw.get("models") or []
    if not models:
        return None
    m = models[0]
    return {
        "label": label,
        "split": raw.get("split"),
        "work_ids_evaluated": raw.get("work_ids_evaluated"),
        "attributed_total": m.get("attributed_total"),
        "explicit_error_rate": m.get("explicit_error_rate"),
        "explicit_errors": m.get("explicit_errors"),
        "explicit_denominator": m.get("explicit_denominator"),
        "overall_error_rate": m.get("overall_error_rate"),
        "overall_errors": m.get("overall_errors"),
        "overall_error_rate_raw_speaker": m.get("overall_error_rate_raw_speaker"),
        "overall_errors_raw_speaker": m.get("overall_errors_raw_speaker"),
        "ambiguous_rate": m.get("ambiguous_rate"),
        "ambiguous_attributed_count": m.get("ambiguous_attributed_count"),
        "ambiguous_rate_all_quotes": m.get("ambiguous_rate_all_quotes"),
        "not_speech_abstention_rate": m.get("not_speech_abstention_rate"),
        "not_speech_abstained": m.get("not_speech_abstained"),
        "not_speech_total": m.get("not_speech_total"),
        "politeness_agreement_rate": m.get("politeness_agreement_rate"),
        "by_narration": m.get("by_narration"),
        "postprocess_summary": m.get("postprocess_summary"),
        "speaker_recheck_summary": m.get("speaker_recheck_summary"),
        "gate": m.get("gate"),
        "elapsed_sec": m.get("elapsed_sec"),
        "prompt_eval_count": m.get("prompt_eval_count"),
        "eval_count": m.get("eval_count"),
    }


def build_analysis(
    config_path: Path,
    result_path: Path,
    summary_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    cfg = load_eval_config(config_path)
    gold_json_path = Path(cfg["gold_json"])
    dataset = load_f9_dataset(
        gold_dir=cfg["gold_dir"],
        split_file=cfg["split_file"],
        split=cfg.get("split", "dev"),
        gold_json_path=gold_json_path,
    )
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    summary_json = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    gold_raw = json.loads(gold_json_path.read_text(encoding="utf-8"))
    gold_raw_by_work = {w["id"]: w for w in gold_raw["works"]}

    raw_outputs = saved["raw_outputs"]
    stage1_preds: dict[str, list[dict[str, Any]] | None] = {}
    stage2_preds: dict[str, list[dict[str, Any]] | None] = {}
    stage3_preds: dict[str, list[dict[str, Any]] | None] = {}
    for w in dataset.works:
        rec = raw_outputs[w.work_id]
        stage1_preds[w.work_id] = rec.get("stage1_parsed_utterances")
        stage2_preds[w.work_id] = rec.get("stage2_parsed_utterances")
        stage3_preds[w.work_id] = rec.get("parsed_utterances")

    pol_mode = str(cfg.get("politeness_mode", "3way_decided"))
    model_id = str(saved["model_spec"])
    prompt_hash = str(saved["prompt_hash"])

    m_s1 = evaluate_model_predictions(
        dataset.works, stage1_preds, model_id=model_id, prompt_hash=prompt_hash, politeness_mode=pol_mode
    )
    m_s2 = evaluate_model_predictions(
        dataset.works, stage2_preds, model_id=model_id, prompt_hash=prompt_hash, politeness_mode=pol_mode
    )
    m_s3 = evaluate_model_predictions(
        dataset.works, stage3_preds, model_id=model_id, prompt_hash=prompt_hash, politeness_mode=pol_mode
    )

    stage_progression = {
        "stage1_raw": _compact_stage_metrics(m_s1, evaluate_step1_gate(m_s1)),
        "stage1_plus_stage2": _compact_stage_metrics(m_s2, evaluate_step1_gate(m_s2)),
        "stage1_plus_stage2_plus_stage3_final": _compact_stage_metrics(m_s3, evaluate_step1_gate(m_s3)),
    }

    # Stage 2 (A) & (B) item-level audit.
    rule_b_items: list[dict[str, Any]] = []
    rule_a_items: list[dict[str, Any]] = []
    stage3_items: list[dict[str, Any]] = []
    not_speech_traces: list[dict[str, Any]] = []
    decided_politeness_mismatches: list[dict[str, Any]] = []
    ambiguous_gold_items: list[dict[str, Any]] = []
    ambiguous_rule_counts: Counter[str] = Counter()

    works_by_id = {w.work_id: w for w in dataset.works}

    for work in dataset.works:
        wid = work.work_id
        rec = raw_outputs[wid]
        s1_map = _index_by_utterance(rec.get("stage1_parsed_utterances"))
        s2_map = _index_by_utterance(rec.get("stage2_parsed_utterances"))
        s3_map = _index_by_utterance(rec.get("parsed_utterances"))
        feats = extract_work_utterance_features(work)
        feat_by_idx = {f.utterance_index: f for f in feats}
        gold_utts_raw = {int(u["n"]): u for u in gold_raw_by_work[wid]["utterances"]}

        pp_audit = rec.get("postprocess_audit") or {}
        rule_b_indices = {int(r["utterance_index"]): r for r in pp_audit.get("inline_non_speech_forced", [])}
        rule_a_indices = {int(r["utterance_index"]): r for r in pp_audit.get("explicit_downgraded", [])}

        for r_idx, r in sorted(rule_b_indices.items()):
            gu = work.utterances[r_idx]
            s1_u = s1_map.get(r_idx) or {}
            s1_spk = s1_u.get("speaker_id")
            s1_ev = s1_u.get("evidence")
            if gu.status == "not_speech":
                kind = "abstained_not_speech" if r.get("changed") else "preserved_abstained"
            elif gu.status == "undetermined":
                kind = "abstained_undetermined" if r.get("changed") else "preserved_abstained"
            elif s1_spk == gu.speaker and s1_ev != "ambiguous":
                kind = "collateral_broke_correct"
            elif s1_ev == "ambiguous" or s1_spk is None:
                kind = "preserved_abstained"
            else:
                kind = "collateral_masked_wrong"
            rule_b_items.append(
                {
                    "work_id": wid,
                    "title": work.title,
                    "utterance_index": r_idx,
                    "quote_chars": r["quote_chars"],
                    "quote_text": gu.quote_text,
                    "gold_status": gu.status,
                    "gold_speaker": gu.speaker,
                    "stage1_speaker": s1_spk,
                    "stage1_evidence": s1_ev,
                    "changed_by_rule_b": r["changed"],
                    "impact_class": kind,
                }
            )

        for r_idx, r in sorted(rule_a_indices.items()):
            gu = work.utterances[r_idx]
            s1_u = s1_map.get(r_idx) or {}
            s1_spk = s1_u.get("speaker_id")
            if gu.status != "attributed":
                kind = f"downgraded_{gu.status}"
            elif s1_spk == gu.speaker:
                kind = "downgraded_correct_explicit"
            else:
                kind = "downgraded_wrong_explicit"
            rule_a_items.append(
                {
                    "work_id": wid,
                    "title": work.title,
                    "utterance_index": r_idx,
                    "quote_text": gu.quote_text,
                    "gold_status": gu.status,
                    "gold_speaker": gu.speaker,
                    "stage1_speaker": s1_spk,
                    "pre_speaker_and_verb": r.get("pre_speaker_and_verb"),
                    "post_speaker_and_verb": r.get("post_speaker_and_verb"),
                    "impact_class": kind,
                }
            )

        sr_audit = rec.get("speaker_recheck_audit") or {}
        sr_dec_by_idx = {
            int(d["utterance_index"]): d
            for d in sr_audit.get("decisions", [])
            if isinstance(d, dict) and isinstance(d.get("utterance_index"), int)
        }
        for dec_idx, dec in sorted(sr_dec_by_idx.items()):
            if not (dec.get("contradiction_found") is True or dec.get("changed") or not dec.get("accepted")):
                continue
            gu = work.utterances[dec_idx]
            before_spk = dec.get("before_speaker")
            after_spk = dec.get("after_speaker")
            if gu.status != "attributed":
                s3_cls = "non_attributed"
            elif before_spk == after_spk:
                s3_cls = "unchanged"
            elif before_spk != gu.speaker and after_spk == gu.speaker:
                s3_cls = "fixed"
            elif before_spk == gu.speaker and after_spk != gu.speaker:
                s3_cls = "regressed"
            else:
                s3_cls = "wrong_to_wrong"
            stage3_items.append(
                {
                    "work_id": wid,
                    "title": work.title,
                    "utterance_index": dec_idx,
                    "quote_text": gu.quote_text,
                    "gold_status": gu.status,
                    "gold_speaker": gu.speaker,
                    "before_speaker": before_spk,
                    "after_speaker": after_spk,
                    "proposed_speaker": dec.get("proposed_speaker"),
                    "contradiction_found": dec.get("contradiction_found"),
                    "accepted": dec.get("accepted"),
                    "changed": dec.get("changed"),
                    "rejected_reason": dec.get("rejected_reason"),
                    "provisional_speaker_check": dec.get("provisional_speaker_check"),
                    "impact_class": s3_cls,
                }
            )

        for gu in work.utterances:
            u_raw = gold_utts_raw[gu.n]
            s1_u = s1_map.get(gu.n) or {}
            s2_u = s2_map.get(gu.n) or {}
            s3_u = s3_map.get(gu.n) or {}
            feat = feat_by_idx[gu.n]

            if gu.status == "not_speech":
                s3_pol = (s3_u.get("register") or {}).get("politeness")
                abstained = (s3_u.get("evidence") == "ambiguous") or (s3_u.get("speaker_id") is None)
                not_speech_traces.append(
                    {
                        "work_id": wid,
                        "title": work.title,
                        "utterance_index": gu.n,
                        "quote_text": gu.quote_text,
                        "speaker_cue": u_raw.get("speaker_cue"),
                        "gold_politeness": gu.gold_politeness,
                        "gold_politeness_rule_id": u_raw.get("politeness_rule_id"),
                        "features": {
                            "position": feat.position,
                            "quote_chars": feat.quote_chars,
                            "pre_has_narration": feat.pre_has_narration,
                            "pre_speakers": list(feat.pre_speakers),
                            "pre_speech_verb": feat.pre_speech_verb,
                            "pre_speaker_and_verb": feat.pre_speaker_and_verb,
                            "post_speakers": list(feat.post_speakers),
                            "post_speech_verb": feat.post_speech_verb,
                            "post_speaker_and_verb": feat.post_speaker_and_verb,
                        },
                        "stage1": s1_u,
                        "stage2": s2_u,
                        "stage3_final": s3_u,
                        "matched_rule_a": gu.n in rule_a_indices,
                        "matched_rule_b": gu.n in rule_b_indices,
                        "stage3_rechecked": gu.n in sr_dec_by_idx,
                        "stage3_decision": sr_dec_by_idx.get(gu.n),
                        "abstained_in_final": abstained,
                        "over_attributed_in_final": not abstained,
                        "final_politeness": s3_pol,
                    }
                )
                continue

            if gu.status != "attributed":
                continue

            pred_pol = (s3_u.get("register") or {}).get("politeness")
            mapped_pred = POLITENESS_5_TO_3_MAP.get(pred_pol)
            rule_id = str(u_raw.get("politeness_rule_id") or "")
            reason = str(u_raw.get("politeness_reason") or "")

            if gu.gold_politeness in POLITENESS_3WAY_DECIDED:
                if mapped_pred != gu.gold_politeness:
                    decided_politeness_mismatches.append(
                        {
                            "work_id": wid,
                            "title": work.title,
                            "utterance_index": gu.n,
                            "quote_text": gu.quote_text,
                            "gold_politeness": gu.gold_politeness,
                            "politeness_rule_id": rule_id,
                            "politeness_reason": reason,
                            "pred_politeness": pred_pol,
                            "pred_mapped_3way": mapped_pred,
                            "pred_evidence": s3_u.get("evidence"),
                            "pred_speaker": s3_u.get("speaker_id"),
                            "gold_speaker": gu.speaker,
                        }
                    )
            elif gu.gold_politeness == "ambiguous":
                ambiguous_rule_counts[rule_id] += 1
                ambiguous_gold_items.append(
                    {
                        "work_id": wid,
                        "title": work.title,
                        "utterance_index": gu.n,
                        "quote_text": gu.quote_text,
                        "gold_politeness": gu.gold_politeness,
                        "politeness_rule_id": rule_id,
                        "politeness_reason": reason,
                        "pred_politeness": pred_pol,
                        "pred_mapped_3way": mapped_pred,
                        "orion_freeze_changed": (wid == "001875_059897" and gu.n == 18),
                        "is_silent_turn": (wid == "001875_059897" and gu.n == 23),
                    }
                )

    final_speaker_errors: list[dict[str, Any]] = []
    for err in m_s3["errors"]:
        wid = err["work_id"]
        idx = int(err["utterance_index"])
        work = works_by_id[wid]
        gu = work.utterances[idx]
        rec = raw_outputs[wid]
        s1_u = _index_by_utterance(rec.get("stage1_parsed_utterances")).get(idx) or {}
        s2_u = _index_by_utterance(rec.get("stage2_parsed_utterances")).get(idx) or {}
        s3_u = _index_by_utterance(rec.get("parsed_utterances")).get(idx) or {}
        pred_addr = s3_u.get("addressee_id")
        final_speaker_errors.append(
            {
                "work_id": wid,
                "title": work.title,
                "narration": work.narration,
                "utterance_index": idx,
                "quote_text": gu.quote_text,
                "gold_speaker": gu.speaker,
                "stage1_speaker": s1_u.get("speaker_id"),
                "stage1_evidence": s1_u.get("evidence"),
                "stage2_speaker": s2_u.get("speaker_id"),
                "stage2_evidence": s2_u.get("evidence"),
                "final_speaker": s3_u.get("speaker_id"),
                "final_evidence": s3_u.get("evidence"),
                "pred_addressee": pred_addr,
                "gold_equals_pred_addressee": (gu.speaker is not None and pred_addr == gu.speaker),
                "pred_politeness": (s3_u.get("register") or {}).get("politeness"),
                "outcome": err["outcome"],
            }
        )

    payload = {
        "provenance": {
            "config_file": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "result_file": str(result_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "summary_file": str(summary_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "gold_json_path": str(gold_json_path),
            "gold_json_sha256": dataset.gold_json_sha256,
            "split": saved["split"],
            "split_version": saved["split_version"],
            "work_ids_evaluated": saved["work_ids_evaluated"],
            "model_spec": saved["model_spec"],
            "prompt_version": saved["prompt_version"],
            "prompt_hash": saved["prompt_hash"],
            "postprocess": saved.get("postprocess"),
            "speaker_recheck": saved.get("speaker_recheck"),
            "speaker_recheck_summary": saved.get("speaker_recheck_summary"),
            "runtime_totals": saved.get("runtime_totals"),
            "schema_validation": m_s3["schema_validation"],
            "non_roster_speaker_count": m_s3["non_roster_speaker_count"],
            "null_string_speaker_count": m_s3["null_string_speaker_count"],
            "summary_json_gate": (summary_json.get("models") or [{}])[0].get("gate"),
        },
        "stage_progression": stage_progression,
        "stage2_audit": {
            "rule_a_downgrades_total": len(rule_a_items),
            "rule_a_impact_counts": dict(Counter(item["impact_class"] for item in rule_a_items)),
            "rule_a_items": rule_a_items,
            "rule_b_forced_total": len(rule_b_items),
            "rule_b_changed_total": sum(1 for item in rule_b_items if item["changed_by_rule_b"]),
            "rule_b_impact_counts": dict(Counter(item["impact_class"] for item in rule_b_items)),
            "rule_b_items": rule_b_items,
        },
        "stage3_audit": {
            "summary": saved.get("speaker_recheck_summary"),
            "impact_counts": dict(Counter(item["impact_class"] for item in stage3_items)),
            "notable_items": stage3_items,
        },
        "final_speaker_errors": {
            "count": len(final_speaker_errors),
            "raw_speaker_errors": m_s3["overall"]["overall_errors_raw_speaker"],
            "items": final_speaker_errors,
        },
        "politeness_audit": {
            "overall": m_s3["overall"].get("politeness_evaluation"),
            "by_work": [
                {
                    "work_id": bw["work_id"],
                    "title": bw["title"],
                    "narration": bw["narration"],
                    "attributed_total": bw["attributed_total"],
                    "politeness_evaluation": bw.get("politeness_evaluation"),
                }
                for bw in m_s3["by_work"]
            ],
            "decided_mismatches_count": len(decided_politeness_mismatches),
            "decided_mismatches": decided_politeness_mismatches,
            "ambiguous_gold_total": len(ambiguous_gold_items),
            "ambiguous_gold_by_rule_id": dict(sorted(ambiguous_rule_counts.items())),
            "ambiguous_gold_items": ambiguous_gold_items,
        },
        "not_speech_trace": not_speech_traces,
        "f9_comparison": {
            "f9_dev": _compact_f9_comparison(F9_DEV_SUMMARY_PATH, "F9 Dev (Round 6)"),
            "f9_holdout": _compact_f9_comparison(F9_HOLDOUT_SUMMARY_PATH, "F9 Holdout (Round 6)"),
            "modern_frozen": _compact_f9_comparison(summary_path, "Modern Round (Frozen Gold)"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Offline analysis for Track A Modern Japanese Round.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--result", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    payload = build_analysis(
        config_path=Path(args.config),
        result_path=Path(args.result),
        summary_path=Path(args.summary),
        out_path=Path(args.out),
    )
    final_s = payload["stage_progression"]["stage1_plus_stage2_plus_stage3_final"]
    print(
        json.dumps(
            {
                "out_file": args.out,
                "explicit_error_rate": final_s["explicit_error_rate"],
                "overall_error_rate": final_s["overall_error_rate"],
                "overall_errors": final_s["overall_errors"],
                "attributed_total": final_s["attributed_total"],
                "politeness_evaluation": final_s["politeness_evaluation"],
                "not_speech_abstention_rate": final_s["not_speech_abstention_rate"],
                "gate": final_s["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
