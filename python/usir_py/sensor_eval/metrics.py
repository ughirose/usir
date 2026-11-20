"""Metrics computation and sensor.v2 JSON Schema validation for Step 1 evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from usir_py.sensor_eval.f9_loader import F9Work

EVIDENCE_LEVELS = ("explicit", "inferred", "ambiguous")
POLITENESS_LEVELS = ("plain", "polite", "honorific", "rough", "unknown")
POLITENESS_3WAY_GOLD_LEVELS = ("polite", "plain", "ambiguous")
POLITENESS_3WAY_DECIDED = frozenset({"polite", "plain"})
POLITENESS_5_TO_3_MAP: dict[str, str] = {
    "polite": "polite",
    "honorific": "polite",
    "plain": "plain",
    "rough": "plain",
    "unknown": "ambiguous",
}
POLITENESS_MODES = ("exact", "3way_decided")
FORBIDDEN_NULL_LITERALS = frozenset({"null", "none", "nil", "undefined"})
STEP1_PASS_THRESHOLDS = {
    "explicit_error_rate_lt": 0.05,
    "overall_error_rate_lt": 0.10,
    "politeness_agreement_rate_ge": 0.90,
}


def is_forbidden_null_literal(value: Any) -> bool:
    """Return True if value is a string masquerading as null/empty (e.g. 'null')."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped == "" or stripped.lower() in FORBIDDEN_NULL_LITERALS


def validate_speaker_id(speaker_id: Any, roster_ids: set[str]) -> str | None:
    """Validate speaker_id against null-string literals and character roster."""
    if speaker_id is None:
        return None
    if not isinstance(speaker_id, str):
        return "invalid_type"
    if is_forbidden_null_literal(speaker_id):
        return "string_null_literal"
    if speaker_id not in roster_ids:
        return "not_in_roster"
    return None


def default_schema_path() -> Path:
    """Return path to schemas/sensor_v2.schema.json inside the repository."""
    return Path(__file__).resolve().parents[3] / "schemas" / "sensor_v2.schema.json"


def load_sensor_v2_validator(schema_path: Path | None = None) -> Draft202012Validator:
    """Load Draft 2020-12 JSON Schema validator for sensor.v2."""
    path = schema_path or default_schema_path()
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_sensor_v2_utterances(
    work: F9Work,
    utterances: list[Any],
    model_id: str,
    prompt_hash: str,
    validator: Draft202012Validator | None = None,
) -> list[str]:
    """Validate predicted utterances wrapped in a minimal sensor.v2 document."""
    val = validator or load_sensor_v2_validator()
    doc = {
        "contract": {
            "schema_version": "sensor.v2",
            "splitter_version": "f9_gold_quotes@1",
            "scene_id": f"f9:{work.work_id}",
            "scene_hash": work.scene_hash,
            "ledger_hash": "f9_gold_characters@1",
            "model_id": model_id or "unknown-model",
            "prompt_hash": prompt_hash or "unknown-prompt",
        },
        "scene_meta": {
            "time": {"relative_order": "unknown"},
            "location_id": None,
            "present_character_ids": [c.id for c in work.characters],
            "pov_character": work.pov,
        },
        "utterances": utterances,
        "character_states": [],
        "fact_updates": [],
    }
    errors: list[str] = [e.message for e in val.iter_errors(doc)]

    # Runtime invariant from Phase 0 contract §5.2: unique utterance_index within scene,
    # plus deterministic rejection of string-null literals and non-roster speaker_id.
    roster_ids = {c.id for c in work.characters}
    seen: set[int] = set()
    for item in utterances:
        if not isinstance(item, dict):
            continue
        idx = item.get("utterance_index")
        if isinstance(idx, int):
            if idx in seen:
                errors.append(f"Duplicate utterance_index={idx} in {work.work_id}")
            seen.add(idx)

        spk = item.get("speaker_id")
        spk_violation = validate_speaker_id(spk, roster_ids)
        if spk_violation == "string_null_literal":
            errors.append(
                f"Rejected string-null speaker_id={spk!r} as non-roster at utterance_index={idx} in {work.work_id}"
            )
        elif spk_violation == "not_in_roster":
            errors.append(
                f"Non-roster speaker_id={spk!r} at utterance_index={idx} in {work.work_id}"
            )

        addr = item.get("addressee_id")
        if is_forbidden_null_literal(addr):
            errors.append(
                f"Rejected string-null addressee_id={addr!r} at utterance_index={idx} in {work.work_id}"
            )

        reg = item.get("register")
        if isinstance(reg, dict) and is_forbidden_null_literal(reg.get("first_person")):
            errors.append(
                f"Rejected string-null first_person={reg.get('first_person')!r} at utterance_index={idx} in {work.work_id}"
            )

    if work.utterances:
        expected_indices = {u.n for u in work.utterances}
        missing_indices = sorted(expected_indices - seen)
        extra_indices = sorted(seen - expected_indices)
        if missing_indices:
            errors.append(
                f"Missing utterance_index(es) {missing_indices} in {work.work_id} "
                f"(expected {len(expected_indices)}, got {len(seen)})"
            )
        if extra_indices:
            errors.append(f"Out-of-range utterance_index(es) {extra_indices} in {work.work_id}")
    return errors


def _rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def _empty_bucket() -> dict[str, Any]:
    return {
        "attributed_total": 0,
        "all_quotes_total": 0,
        "correct_strict": 0,
        "correct_raw_speaker": 0,
        "wrong_speaker": 0,
        "ambiguous_evasion": 0,
        "ambiguous_evasion_with_correct_speaker": 0,
        "unattributed_null": 0,
        "missing_utterance": 0,
        "explicit_total": 0,
        "explicit_correct": 0,
        "explicit_errors": 0,
        "inferred_total": 0,
        "inferred_correct": 0,
        "inferred_errors": 0,
        "ambiguous_attributed": 0,
        "ambiguous_all_quotes": 0,
        "politeness_gold_total": 0,
        "politeness_correct": 0,
        "politeness_decided_total": 0,
        "politeness_decided_mapped_correct": 0,
        "politeness_decided_strict_correct": 0,
        "politeness_all_mapped_correct": 0,
        "politeness_all_raw_exact_correct": 0,
        "politeness_gold_distribution_attributed": Counter(),
        "politeness_confusion_3x5": {g: Counter() for g in POLITENESS_3WAY_GOLD_LEVELS},
        "excluded_total": 0,
        "excluded_abstained": 0,
        "excluded_over_attributed": 0,
        "not_speech_total": 0,
        "not_speech_abstained": 0,
        "not_speech_over_attributed": 0,
        "undetermined_total": 0,
        "undetermined_abstained": 0,
        "undetermined_over_attributed": 0,
        "non_roster_speaker_count": 0,
        "null_string_speaker_count": 0,
        "politeness_distribution_attributed": Counter(),
        "politeness_distribution_all": Counter(),
    }


def _finalize_bucket(b: dict[str, Any], politeness_mode: str = "exact") -> dict[str, Any]:
    attr_tot = b["attributed_total"]
    strict_err = attr_tot - b["correct_strict"]
    raw_err = attr_tot - b["correct_raw_speaker"]
    pol_tot = b["politeness_gold_total"]
    out = {
        "attributed_total": attr_tot,
        "all_quotes_total": b["all_quotes_total"],
        "explicit_error_rate": _rate(b["explicit_errors"], b["explicit_total"]),
        "explicit_correct": b["explicit_correct"],
        "explicit_errors": b["explicit_errors"],
        "explicit_denominator": b["explicit_total"],
        "explicit_coverage": _rate(b["explicit_total"], attr_tot),
        "overall_error_rate": _rate(strict_err, attr_tot),
        "overall_errors": strict_err,
        "overall_correct": b["correct_strict"],
        "overall_error_rate_raw_speaker": _rate(raw_err, attr_tot),
        "overall_errors_raw_speaker": raw_err,
        "overall_correct_raw_speaker": b["correct_raw_speaker"],
        "politeness_agreement_rate": _rate(b["politeness_correct"], pol_tot),
        "politeness_gold_available": pol_tot > 0,
        "politeness_gold_total": pol_tot,
        "politeness_correct": b["politeness_correct"],
        "ambiguous_rate": _rate(b["ambiguous_attributed"], attr_tot),
        "ambiguous_attributed_count": b["ambiguous_attributed"],
        "ambiguous_rate_all_quotes": _rate(b["ambiguous_all_quotes"], b["all_quotes_total"]),
        "ambiguous_all_quotes_count": b["ambiguous_all_quotes"],
        "excluded_total": b["excluded_total"],
        "excluded_abstention_rate": _rate(b["excluded_abstained"], b["excluded_total"]),
        "excluded_abstained": b["excluded_abstained"],
        "excluded_over_attributed": b["excluded_over_attributed"],
        "not_speech_total": b["not_speech_total"],
        "not_speech_abstention_rate": _rate(b["not_speech_abstained"], b["not_speech_total"]),
        "not_speech_abstained": b["not_speech_abstained"],
        "not_speech_over_attributed": b["not_speech_over_attributed"],
        "undetermined_total": b["undetermined_total"],
        "undetermined_abstention_rate": _rate(b["undetermined_abstained"], b["undetermined_total"]),
        "undetermined_abstained": b["undetermined_abstained"],
        "undetermined_over_attributed": b["undetermined_over_attributed"],
        "non_roster_speaker_count": b["non_roster_speaker_count"],
        "null_string_speaker_count": b["null_string_speaker_count"],
        "error_breakdown": {
            "wrong_speaker": b["wrong_speaker"],
            "ambiguous_evasion": b["ambiguous_evasion"],
            "ambiguous_evasion_with_correct_speaker": b["ambiguous_evasion_with_correct_speaker"],
            "unattributed_null": b["unattributed_null"],
            "missing_utterance": b["missing_utterance"],
        },
        "by_evidence": {
            "explicit": {
                "count": b["explicit_total"],
                "correct": b["explicit_correct"],
                "errors": b["explicit_errors"],
                "error_rate": _rate(b["explicit_errors"], b["explicit_total"]),
                "accuracy": _rate(b["explicit_correct"], b["explicit_total"]),
            },
            "inferred": {
                "count": b["inferred_total"],
                "correct": b["inferred_correct"],
                "errors": b["inferred_errors"],
                "error_rate": _rate(b["inferred_errors"], b["inferred_total"]),
                "accuracy": _rate(b["inferred_correct"], b["inferred_total"]),
            },
            "ambiguous": {
                "count": b["ambiguous_attributed"],
                "correct_raw_speaker": b["ambiguous_evasion_with_correct_speaker"],
            },
            "missing": {
                "count": b["missing_utterance"],
            },
        },
        "politeness_distribution_attributed": {
            k: b["politeness_distribution_attributed"].get(k, 0)
            for k in (*POLITENESS_LEVELS, "missing_or_invalid")
            if b["politeness_distribution_attributed"].get(k, 0) > 0 or k in POLITENESS_LEVELS
        },
        "politeness_distribution_all": {
            k: b["politeness_distribution_all"].get(k, 0)
            for k in (*POLITENESS_LEVELS, "missing_or_invalid")
            if b["politeness_distribution_all"].get(k, 0) > 0 or k in POLITENESS_LEVELS
        },
    }
    all_pol_total = sum(b["politeness_gold_distribution_attributed"].values())
    if all_pol_total > 0:
        threshold = STEP1_PASS_THRESHOLDS["politeness_agreement_rate_ge"]
        dec_tot = b["politeness_decided_total"]
        dec_mapped_rate = _rate(b["politeness_decided_mapped_correct"], dec_tot)
        dec_strict_rate = _rate(b["politeness_decided_strict_correct"], dec_tot)
        all_mapped_rate = _rate(b["politeness_all_mapped_correct"], all_pol_total)
        all_raw_rate = _rate(b["politeness_all_raw_exact_correct"], all_pol_total)

        gold_dist_keys = list(POLITENESS_3WAY_GOLD_LEVELS) + [
            k
            for k in sorted(b["politeness_gold_distribution_attributed"])
            if k not in POLITENESS_3WAY_GOLD_LEVELS
        ]
        pred_cols = (*POLITENESS_LEVELS, "missing_or_invalid")
        conf_keys = list(POLITENESS_3WAY_GOLD_LEVELS) + [
            k for k in sorted(b["politeness_confusion_3x5"]) if k not in POLITENESS_3WAY_GOLD_LEVELS
        ]
        out["politeness_evaluation"] = {
            "mode": politeness_mode,
            "mapping_5to3": dict(POLITENESS_5_TO_3_MAP),
            "decided_mapped_5to3": {
                "correct": b["politeness_decided_mapped_correct"],
                "total": dec_tot,
                "rate": dec_mapped_rate,
                "passed_ge_90": dec_mapped_rate is not None and dec_mapped_rate >= threshold,
            },
            "decided_strict_exact": {
                "correct": b["politeness_decided_strict_correct"],
                "total": dec_tot,
                "rate": dec_strict_rate,
                "passed_ge_90": dec_strict_rate is not None and dec_strict_rate >= threshold,
            },
            "all_attributed_mapped_5to3": {
                "correct": b["politeness_all_mapped_correct"],
                "total": all_pol_total,
                "rate": all_mapped_rate,
                "passed_ge_90": all_mapped_rate is not None and all_mapped_rate >= threshold,
            },
            "all_attributed_raw_exact": {
                "correct": b["politeness_all_raw_exact_correct"],
                "total": all_pol_total,
                "rate": all_raw_rate,
                "passed_ge_90": all_raw_rate is not None and all_raw_rate >= threshold,
            },
            "gold_distribution_attributed": {
                k: b["politeness_gold_distribution_attributed"].get(k, 0) for k in gold_dist_keys
            },
            "confusion_matrix_3x5": {
                g: {p: b["politeness_confusion_3x5"][g].get(p, 0) for p in pred_cols}
                for g in conf_keys
            },
        }
    return out


def evaluate_model_predictions(
    works: list[F9Work],
    predictions_by_work: dict[str, list[dict[str, Any]] | None],
    model_id: str,
    prompt_hash: str,
    schema_validator: Draft202012Validator | None = None,
    politeness_mode: str = "exact",
) -> dict[str, Any]:
    """Compute Step 1 metrics across works, narration types, and evidence levels."""
    if politeness_mode not in POLITENESS_MODES:
        raise ValueError(
            f"Unsupported politeness_mode {politeness_mode!r}; expected one of {POLITENESS_MODES}"
        )
    val = schema_validator or load_sensor_v2_validator()
    total_bucket = _empty_bucket()
    by_narration_buckets: dict[str, dict[str, Any]] = {
        "first": _empty_bucket(),
        "third": _empty_bucket(),
    }
    by_work_results: list[dict[str, Any]] = []
    schema_validation_by_work: dict[str, dict[str, Any]] = {}
    non_roster_speakers: list[dict[str, Any]] = []
    error_details: list[dict[str, Any]] = []
    excluded_over_attr_details: list[dict[str, Any]] = []

    for work in works:
        w_bucket = _empty_bucket()
        nar_bucket = by_narration_buckets.setdefault(work.narration, _empty_bucket())
        roster_ids = {c.id for c in work.characters}

        raw_utts = predictions_by_work.get(work.work_id)
        if raw_utts is None:
            schema_errors = ["No valid parsed utterances payload"]
            pred_list: list[Any] = []
        else:
            pred_list = list(raw_utts)
            schema_errors = validate_sensor_v2_utterances(
                work, pred_list, model_id=model_id, prompt_hash=prompt_hash, validator=val
            )

        pred_by_idx: dict[int, dict[str, Any]] = {}
        duplicate_indices: list[int] = []
        extra_indices: list[int] = []
        expected_n = len(work.utterances)
        for item in pred_list:
            if not isinstance(item, dict):
                continue
            idx = item.get("utterance_index")
            if not isinstance(idx, int):
                continue
            if idx in pred_by_idx:
                duplicate_indices.append(idx)
                continue
            if idx < 0 or idx >= expected_n:
                extra_indices.append(idx)
                continue
            pred_by_idx[idx] = item

        schema_validation_by_work[work.work_id] = {
            "schema_valid": len(schema_errors) == 0,
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors[:20],
            "predicted_count": len(pred_list),
            "expected_count": expected_n,
            "duplicate_indices": duplicate_indices,
            "extra_indices": extra_indices,
        }

        for gold_u in work.utterances:
            pred_u = pred_by_idx.get(gold_u.n)
            pred_speaker = pred_u.get("speaker_id") if pred_u else None
            pred_evidence = pred_u.get("evidence") if pred_u else None
            pred_reg = (pred_u.get("register") or {}) if pred_u else {}
            pred_politeness = pred_reg.get("politeness") if isinstance(pred_reg, dict) else None

            pol_key = pred_politeness if pred_politeness in POLITENESS_LEVELS else "missing_or_invalid"

            spk_violation = (
                validate_speaker_id(pred_speaker, roster_ids) if pred_u is not None else None
            )
            if spk_violation is not None:
                non_roster_speakers.append(
                    {
                        "work_id": work.work_id,
                        "utterance_index": gold_u.n,
                        "speaker_id": pred_speaker,
                        "reason": spk_violation,
                        "gold_status": gold_u.status,
                        "gold_speaker": gold_u.speaker,
                    }
                )
                for b in (total_bucket, w_bucket, nar_bucket):
                    b["non_roster_speaker_count"] += 1
                    if spk_violation == "string_null_literal":
                        b["null_string_speaker_count"] += 1

            for b in (total_bucket, w_bucket, nar_bucket):
                b["all_quotes_total"] += 1
                b["politeness_distribution_all"][pol_key] += 1
                if pred_evidence == "ambiguous":
                    b["ambiguous_all_quotes"] += 1

            if gold_u.status != "attributed":
                abstained = (
                    pred_u is not None
                    and spk_violation is None
                    and (pred_evidence == "ambiguous" or pred_speaker is None)
                )
                over_attr = pred_u is not None and not abstained and (
                    pred_speaker is not None or spk_violation is not None
                )
                for b in (total_bucket, w_bucket, nar_bucket):
                    b["excluded_total"] += 1
                    if abstained:
                        b["excluded_abstained"] += 1
                    elif over_attr:
                        b["excluded_over_attributed"] += 1
                    if gold_u.status == "not_speech":
                        b["not_speech_total"] += 1
                        if abstained:
                            b["not_speech_abstained"] += 1
                        elif over_attr:
                            b["not_speech_over_attributed"] += 1
                    elif gold_u.status == "undetermined":
                        b["undetermined_total"] += 1
                        if abstained:
                            b["undetermined_abstained"] += 1
                        elif over_attr:
                            b["undetermined_over_attributed"] += 1
                if over_attr:
                    excluded_over_attr_details.append(
                        {
                            "work_id": work.work_id,
                            "utterance_index": gold_u.n,
                            "gold_status": gold_u.status,
                            "head": gold_u.head,
                            "pred_speaker": pred_speaker,
                            "pred_evidence": pred_evidence,
                            "pred_politeness": pred_politeness,
                            "speaker_violation": spk_violation,
                        }
                    )
                continue

            # Gold attributed utterance evaluation.
            raw_match = (
                pred_u is not None
                and spk_violation is None
                and pred_speaker == gold_u.speaker
            )
            strict_match = raw_match and pred_evidence != "ambiguous"

            if pred_u is None:
                outcome = "missing_utterance"
            elif pred_evidence == "ambiguous":
                outcome = "ambiguous_evasion"
            elif raw_match:
                outcome = "correct"
            elif pred_speaker is None:
                outcome = "unattributed_null"
            else:
                outcome = "wrong_speaker"

            for b in (total_bucket, w_bucket, nar_bucket):
                b["attributed_total"] += 1
                b["politeness_distribution_attributed"][pol_key] += 1
                if raw_match:
                    b["correct_raw_speaker"] += 1
                if strict_match:
                    b["correct_strict"] += 1

                if outcome == "wrong_speaker":
                    b["wrong_speaker"] += 1
                elif outcome == "ambiguous_evasion":
                    b["ambiguous_evasion"] += 1
                    if raw_match:
                        b["ambiguous_evasion_with_correct_speaker"] += 1
                elif outcome == "unattributed_null":
                    b["unattributed_null"] += 1
                elif outcome == "missing_utterance":
                    b["missing_utterance"] += 1

                if pred_u is not None:
                    if pred_evidence == "explicit":
                        b["explicit_total"] += 1
                        if raw_match:
                            b["explicit_correct"] += 1
                        else:
                            b["explicit_errors"] += 1
                    elif pred_evidence == "inferred":
                        b["inferred_total"] += 1
                        if raw_match:
                            b["inferred_correct"] += 1
                        else:
                            b["inferred_errors"] += 1
                    elif pred_evidence == "ambiguous":
                        b["ambiguous_attributed"] += 1

                if gold_u.gold_politeness is not None:
                    mapped_pred = POLITENESS_5_TO_3_MAP.get(pred_politeness)
                    b["politeness_gold_distribution_attributed"][gold_u.gold_politeness] += 1
                    b["politeness_confusion_3x5"].setdefault(gold_u.gold_politeness, Counter())[
                        pol_key
                    ] += 1
                    if pred_politeness == gold_u.gold_politeness:
                        b["politeness_all_raw_exact_correct"] += 1
                    if mapped_pred == gold_u.gold_politeness:
                        b["politeness_all_mapped_correct"] += 1
                    if gold_u.gold_politeness in POLITENESS_3WAY_DECIDED:
                        b["politeness_decided_total"] += 1
                        if mapped_pred == gold_u.gold_politeness:
                            b["politeness_decided_mapped_correct"] += 1
                        if pred_politeness == gold_u.gold_politeness:
                            b["politeness_decided_strict_correct"] += 1

                    if politeness_mode == "exact":
                        b["politeness_gold_total"] += 1
                        if pred_politeness == gold_u.gold_politeness:
                            b["politeness_correct"] += 1
                    elif politeness_mode == "3way_decided":
                        if gold_u.gold_politeness in POLITENESS_3WAY_DECIDED:
                            b["politeness_gold_total"] += 1
                            if mapped_pred == gold_u.gold_politeness:
                                b["politeness_correct"] += 1

            if outcome != "correct":
                error_details.append(
                    {
                        "work_id": work.work_id,
                        "narration": work.narration,
                        "utterance_index": gold_u.n,
                        "head": gold_u.head,
                        "gold_speaker": gold_u.speaker,
                        "pred_speaker": pred_speaker,
                        "pred_evidence": pred_evidence,
                        "pred_politeness": pred_politeness,
                        "outcome": outcome,
                    }
                )

        w_summary = _finalize_bucket(w_bucket, politeness_mode=politeness_mode)
        w_summary["work_id"] = work.work_id
        w_summary["title"] = work.title
        w_summary["author"] = work.author
        w_summary["narration"] = work.narration
        w_summary["schema_valid"] = schema_validation_by_work[work.work_id]["schema_valid"]
        by_work_results.append(w_summary)

    return {
        "overall": _finalize_bucket(total_bucket, politeness_mode=politeness_mode),
        "by_narration": {
            nar: _finalize_bucket(bucket, politeness_mode=politeness_mode)
            for nar, bucket in by_narration_buckets.items()
        },
        "by_work": by_work_results,
        "schema_validation": {
            "all_works_valid": all(
                v["schema_valid"] for v in schema_validation_by_work.values()
            ),
            "valid_works": sum(1 for v in schema_validation_by_work.values() if v["schema_valid"]),
            "total_works": len(works),
            "per_work": schema_validation_by_work,
        },
        "non_roster_speaker_ids": non_roster_speakers,
        "non_roster_speaker_count": len(non_roster_speakers),
        "null_string_speaker_count": total_bucket["null_string_speaker_count"],
        "errors": error_details,
        "excluded_over_attributions": excluded_over_attr_details,
    }


def evaluate_step1_gate(
    metrics: dict[str, Any], require_politeness: bool = False
) -> dict[str, Any]:
    """Evaluate deterministic Step 1 pass/fail thresholds on computed metrics."""
    ov = metrics["overall"]
    sv = metrics["schema_validation"]
    non_roster = metrics.get("non_roster_speaker_ids", [])

    explicit_rate = ov["explicit_error_rate"]
    explicit_den = ov["explicit_denominator"]
    overall_rate = ov["overall_error_rate"]

    explicit_passed = (
        explicit_rate is not None
        and explicit_den > 0
        and explicit_rate < STEP1_PASS_THRESHOLDS["explicit_error_rate_lt"]
    )
    overall_passed = (
        overall_rate is not None
        and overall_rate < STEP1_PASS_THRESHOLDS["overall_error_rate_lt"]
    )
    validation_passed = bool(sv.get("all_works_valid", False)) and len(non_roster) == 0

    if ov.get("politeness_gold_available"):
        pol_rate = ov.get("politeness_agreement_rate")
        politeness_passed: bool | None = (
            pol_rate is not None
            and pol_rate >= STEP1_PASS_THRESHOLDS["politeness_agreement_rate_ge"]
        )
    else:
        politeness_passed = False if require_politeness else None

    speaker_gate_passed = explicit_passed and overall_passed and validation_passed
    passed = speaker_gate_passed and (
        politeness_passed is True if require_politeness else True
    )
    return {
        "passed": passed,
        "speaker_gate_passed": speaker_gate_passed,
        "explicit_passed": explicit_passed,
        "overall_passed": overall_passed,
        "validation_passed": validation_passed,
        "politeness_passed": politeness_passed,
        "thresholds": dict(STEP1_PASS_THRESHOLDS),
    }


def compute_politeness_pairwise_agreement(
    works: list[F9Work],
    model_predictions: dict[str, dict[str, list[dict[str, Any]] | None]],
    attributed_only: bool = True,
) -> dict[str, Any]:
    """Compute pairwise politeness label agreement across models (diagnostic when gold lacks labels)."""
    tags = list(model_predictions.keys())
    extracted: dict[str, dict[tuple[str, int], str | None]] = {}
    keys: list[tuple[str, int]] = []
    for work in works:
        for u in work.utterances:
            if attributed_only and u.status != "attributed":
                continue
            keys.append((work.work_id, u.n))

    for tag in tags:
        mapping: dict[tuple[str, int], str | None] = {}
        preds_by_work = model_predictions[tag]
        for work in works:
            utts = preds_by_work.get(work.work_id) or []
            idx_map = {
                u.get("utterance_index"): (u.get("register") or {}).get("politeness")
                for u in utts
                if isinstance(u, dict)
            }
            for u in work.utterances:
                if attributed_only and u.status != "attributed":
                    continue
                mapping[(work.work_id, u.n)] = idx_map.get(u.n)
        extracted[tag] = mapping

    pairs: list[dict[str, Any]] = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            t1, t2 = tags[i], tags[j]
            m1, m2 = extracted[t1], extracted[t2]
            both_present = 0
            agreed = 0
            for k in keys:
                p1, p2 = m1.get(k), m2.get(k)
                if p1 is not None and p2 is not None:
                    both_present += 1
                    if p1 == p2:
                        agreed += 1
            pairs.append(
                {
                    "model_a": t1,
                    "model_b": t2,
                    "compared": both_present,
                    "agreed": agreed,
                    "agreement_rate": _rate(agreed, both_present),
                }
            )
    return {
        "scope": "attributed_only" if attributed_only else "all_quotes",
        "total_items": len(keys),
        "pairs": pairs,
    }
