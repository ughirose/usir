"""Deterministic Stage 2 post-processing for Step 1 speaker/evidence outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from usir_py.sensor_eval.f9_loader import F9Dataset, F9Work
from usir_py.sensor_eval.features import UtteranceFeatures, extract_work_utterance_features
from usir_py.sensor_eval.metrics import evaluate_model_predictions, evaluate_step1_gate

_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "explicit_evidence_gate",
        "inline_non_speech_filter",
        "inline_non_speech_max_chars",
    }
)


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration toggling deterministic Stage 2 post-processing rules."""

    explicit_evidence_gate: bool = False
    inline_non_speech_filter: bool = False
    inline_non_speech_max_chars: int = 11

    def __post_init__(self) -> None:
        if not isinstance(self.explicit_evidence_gate, bool):
            raise ValueError(
                "explicit_evidence_gate must be a bool, "
                f"got {type(self.explicit_evidence_gate).__name__}"
            )
        if not isinstance(self.inline_non_speech_filter, bool):
            raise ValueError(
                "inline_non_speech_filter must be a bool, "
                f"got {type(self.inline_non_speech_filter).__name__}"
            )
        if (
            isinstance(self.inline_non_speech_max_chars, bool)
            or not isinstance(self.inline_non_speech_max_chars, int)
            or self.inline_non_speech_max_chars < 0
        ):
            raise ValueError(
                "inline_non_speech_max_chars must be a non-negative integer, "
                f"got {self.inline_non_speech_max_chars!r}"
            )

    @property
    def is_enabled(self) -> bool:
        """Return True when at least one Stage 2 post-processing component is active."""
        return self.explicit_evidence_gate or self.inline_non_speech_filter

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PostprocessConfig:
        """Parse and validate a config mapping; None yields the all-disabled default."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"postprocess config must be a dict, got {type(raw).__name__}")
        unknown = set(raw.keys()) - _ALLOWED_CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown postprocess config keys: {sorted(unknown)}")
        return cls(
            explicit_evidence_gate=raw.get("explicit_evidence_gate", False),
            inline_non_speech_filter=raw.get("inline_non_speech_filter", False),
            inline_non_speech_max_chars=raw.get("inline_non_speech_max_chars", 11),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain JSON-compatible dictionary."""
        return {
            "explicit_evidence_gate": self.explicit_evidence_gate,
            "inline_non_speech_filter": self.inline_non_speech_filter,
            "inline_non_speech_max_chars": self.inline_non_speech_max_chars,
        }


def _copy_utterance_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    copied = dict(item)
    reg = copied.get("register")
    if isinstance(reg, dict):
        copied["register"] = dict(reg)
    return copied


def _matches_inline_non_speech(feat: UtteranceFeatures, config: PostprocessConfig) -> bool:
    return (
        config.inline_non_speech_filter
        and feat.position == "inline_embedded"
        and feat.pre_speech_verb is None
        and feat.post_speech_verb is None
        and feat.quote_chars <= config.inline_non_speech_max_chars
    )


def apply_postprocessing_to_work(
    work: F9Work,
    utterances: list[dict[str, Any]] | None,
    config: PostprocessConfig,
    features: Sequence[UtteranceFeatures] | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Apply deterministic Stage 2 gates (A) and (B) to a single work's predictions."""
    if utterances is None:
        return None, {
            "work_id": work.work_id,
            "inline_non_speech_forced": [],
            "explicit_downgraded": [],
        }

    work_features = extract_work_utterance_features(work) if features is None else features
    feat_by_idx = {f.utterance_index: f for f in work_features}

    inline_forced: list[dict[str, Any]] = []
    explicit_downgraded: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []

    for raw_item in utterances:
        item = _copy_utterance_item(raw_item)
        if not isinstance(item, dict):
            processed.append(item)
            continue

        idx = item.get("utterance_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx not in feat_by_idx:
            processed.append(item)
            continue

        feat = feat_by_idx[idx]
        if _matches_inline_non_speech(feat, config):
            prev_speaker = item.get("speaker_id")
            prev_evidence = item.get("evidence")
            item["speaker_id"] = None
            item["evidence"] = "ambiguous"
            inline_forced.append(
                {
                    "utterance_index": idx,
                    "quote_chars": feat.quote_chars,
                    "prev_speaker": prev_speaker,
                    "prev_evidence": prev_evidence,
                    "changed": (prev_speaker is not None or prev_evidence != "ambiguous"),
                }
            )
        elif (
            config.explicit_evidence_gate
            and item.get("evidence") == "explicit"
            and not feat.pre_speaker_and_verb
            and not feat.post_speaker_and_verb
        ):
            item["evidence"] = "inferred"
            explicit_downgraded.append(
                {
                    "utterance_index": idx,
                    "speaker_id": item.get("speaker_id"),
                    "pre_speaker_and_verb": feat.pre_speaker_and_verb,
                    "post_speaker_and_verb": feat.post_speaker_and_verb,
                }
            )

        processed.append(item)

    return processed, {
        "work_id": work.work_id,
        "inline_non_speech_forced": inline_forced,
        "explicit_downgraded": explicit_downgraded,
    }


def apply_postprocessing_to_predictions(
    works: Sequence[F9Work],
    predictions_by_work: dict[str, list[dict[str, Any]] | None],
    config: PostprocessConfig,
) -> tuple[dict[str, list[dict[str, Any]] | None], dict[str, Any]]:
    """Apply Stage 2 post-processing across all works and aggregate audit totals."""
    processed_by_work: dict[str, list[dict[str, Any]] | None] = {}
    per_work_audits: dict[str, dict[str, Any]] = {}
    forced_total = 0
    changed_total = 0
    downgraded_total = 0

    for work in works:
        raw_utts = predictions_by_work.get(work.work_id)
        out_utts, w_audit = apply_postprocessing_to_work(work, raw_utts, config)
        processed_by_work[work.work_id] = out_utts
        per_work_audits[work.work_id] = w_audit
        forced_total += len(w_audit["inline_non_speech_forced"])
        changed_total += sum(1 for r in w_audit["inline_non_speech_forced"] if r["changed"])
        downgraded_total += len(w_audit["explicit_downgraded"])

    aggregate_audit = {
        "inline_non_speech_forced_count": forced_total,
        "inline_non_speech_changed_count": changed_total,
        "explicit_downgraded_count": downgraded_total,
        "per_work": per_work_audits,
    }
    return processed_by_work, aggregate_audit


def _extract_stage1_predictions(
    works: Sequence[F9Work], saved_report: dict[str, Any]
) -> dict[str, list[dict[str, Any]] | None]:
    raw_outputs = saved_report.get("raw_outputs") or {}
    preds: dict[str, list[dict[str, Any]] | None] = {}
    for work in works:
        rec = raw_outputs.get(work.work_id) or {}
        if "stage1_parsed_utterances" in rec:
            preds[work.work_id] = rec.get("stage1_parsed_utterances")
        else:
            preds[work.work_id] = rec.get("parsed_utterances")
    return preds


def replay_saved_result_with_postprocess(
    dataset: F9Dataset,
    saved_report: dict[str, Any],
    config: PostprocessConfig,
) -> dict[str, Any]:
    """Re-score saved Stage 1 predictions offline with a given PostprocessConfig."""
    stage1_preds = _extract_stage1_predictions(dataset.works, saved_report)
    processed_preds, audit = apply_postprocessing_to_predictions(dataset.works, stage1_preds, config)
    model_id = str(saved_report.get("model_spec") or "unknown-model")
    prompt_hash = str(saved_report.get("prompt_hash") or "")
    metrics = evaluate_model_predictions(
        works=dataset.works,
        predictions_by_work=processed_preds,
        model_id=model_id,
        prompt_hash=prompt_hash,
    )
    gate = evaluate_step1_gate(metrics)
    return {
        "postprocess": config.to_dict(),
        "audit": audit,
        "metrics": metrics,
        "gate": gate,
    }


def _is_abstained(pred_u: dict[str, Any] | None) -> bool:
    if pred_u is None:
        return False
    return pred_u.get("evidence") == "ambiguous" or pred_u.get("speaker_id") is None


def analyze_inline_non_speech_thresholds(
    dataset: F9Dataset,
    saved_report: dict[str, Any],
    thresholds: Sequence[int],
    *,
    include_explicit_gate: bool = False,
) -> list[dict[str, Any]]:
    """Sweep Rule (B) character thresholds offline on saved predictions and count collateral hits."""
    stage1_preds = _extract_stage1_predictions(dataset.works, saved_report)
    features_by_work = {w.work_id: extract_work_utterance_features(w) for w in dataset.works}
    stage1_by_work_idx: dict[str, dict[int, dict[str, Any]]] = {}
    for work in dataset.works:
        idx_map: dict[int, dict[str, Any]] = {}
        for item in stage1_preds.get(work.work_id) or []:
            if isinstance(item, dict) and isinstance(item.get("utterance_index"), int):
                idx_map[item["utterance_index"]] = item
        stage1_by_work_idx[work.work_id] = idx_map

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        cfg = PostprocessConfig(
            explicit_evidence_gate=include_explicit_gate,
            inline_non_speech_filter=True,
            inline_non_speech_max_chars=threshold,
        )
        replay = replay_saved_result_with_postprocess(dataset, saved_report, cfg)
        ov = replay["metrics"]["overall"]

        matched_total = 0
        matched_not_speech = 0
        newly_abstained_not_speech = 0
        matched_undetermined = 0
        newly_abstained_undetermined = 0
        collateral_details: list[dict[str, Any]] = []

        for work in dataset.works:
            feats = features_by_work[work.work_id]
            s1_map = stage1_by_work_idx[work.work_id]
            for gold_u, feat in zip(work.utterances, feats):
                if not _matches_inline_non_speech(feat, cfg):
                    continue
                matched_total += 1
                s1_u = s1_map.get(gold_u.n)
                was_abstained = _is_abstained(s1_u)
                if gold_u.status == "not_speech":
                    matched_not_speech += 1
                    if not was_abstained:
                        newly_abstained_not_speech += 1
                elif gold_u.status == "undetermined":
                    matched_undetermined += 1
                    if not was_abstained:
                        newly_abstained_undetermined += 1
                elif gold_u.status == "attributed":
                    s1_spk = s1_u.get("speaker_id") if s1_u else None
                    s1_ev = s1_u.get("evidence") if s1_u else None
                    collateral_details.append(
                        {
                            "work_id": work.work_id,
                            "title": work.title,
                            "utterance_index": gold_u.n,
                            "quote_chars": feat.quote_chars,
                            "gold_speaker": gold_u.speaker,
                            "quote_text": gold_u.quote_text,
                            "stage1_speaker": s1_spk,
                            "stage1_evidence": s1_ev,
                            "stage1_was_correct": (
                                s1_spk == gold_u.speaker and s1_ev != "ambiguous"
                            ),
                        }
                    )

        rows.append(
            {
                "threshold": threshold,
                "include_explicit_gate": include_explicit_gate,
                "matched_total": matched_total,
                "matched_not_speech": matched_not_speech,
                "newly_abstained_not_speech": newly_abstained_not_speech,
                "not_speech_abstained": ov["not_speech_abstained"],
                "not_speech_total": ov["not_speech_total"],
                "not_speech_abstention_rate": ov["not_speech_abstention_rate"],
                "matched_undetermined": matched_undetermined,
                "newly_abstained_undetermined": newly_abstained_undetermined,
                "collateral_attributed_count": len(collateral_details),
                "collateral_attributed_details": collateral_details,
                "explicit_downgraded_count": replay["audit"]["explicit_downgraded_count"],
                "explicit_error_rate": ov["explicit_error_rate"],
                "explicit_errors": ov["explicit_errors"],
                "explicit_denominator": ov["explicit_denominator"],
                "overall_error_rate": ov["overall_error_rate"],
                "overall_errors": ov["overall_errors"],
                "attributed_total": ov["attributed_total"],
                "ambiguous_rate": ov["ambiguous_rate"],
                "ambiguous_attributed_count": ov["ambiguous_attributed_count"],
                "ambiguous_rate_all_quotes": ov["ambiguous_rate_all_quotes"],
                "ambiguous_all_quotes_count": ov["ambiguous_all_quotes_count"],
                "gate": replay["gate"],
            }
        )
    return rows
