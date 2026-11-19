"""Step 1 evaluation harness for sensor.v2 utterance speaker & register annotation."""

from __future__ import annotations

from usir_py.sensor_eval.f9_loader import (
    F9Character,
    F9Dataset,
    F9UtteranceGold,
    F9Work,
    extract_outer_quote_spans,
    load_f9_dataset,
    number_outer_quotes,
)
from usir_py.sensor_eval.features import (
    UtteranceFeatures,
    extract_utterance_features_from_text,
    extract_work_utterance_features,
    format_utterance_features_block,
)
from usir_py.sensor_eval.metrics import (
    FORBIDDEN_NULL_LITERALS,
    STEP1_PASS_THRESHOLDS,
    compute_politeness_pairwise_agreement,
    evaluate_model_predictions,
    evaluate_step1_gate,
    is_forbidden_null_literal,
    validate_sensor_v2_utterances,
    validate_speaker_id,
)
from usir_py.sensor_eval.ollama_client import OllamaResponse, call_ollama_chat
from usir_py.sensor_eval.postprocess import (
    PostprocessConfig,
    analyze_inline_non_speech_thresholds,
    apply_postprocessing_to_predictions,
    apply_postprocessing_to_work,
    replay_saved_result_with_postprocess,
)
from usir_py.sensor_eval.prompt import (
    PROMPT_REGISTRY,
    PROMPT_VERSION,
    build_ollama_utterances_format_schema,
    build_work_messages,
    compute_prompt_hash,
    get_prompt,
)

__all__ = [
    "F9Character",
    "F9Dataset",
    "F9UtteranceGold",
    "F9Work",
    "FORBIDDEN_NULL_LITERALS",
    "OllamaResponse",
    "PROMPT_REGISTRY",
    "PROMPT_VERSION",
    "PostprocessConfig",
    "STEP1_PASS_THRESHOLDS",
    "UtteranceFeatures",
    "analyze_inline_non_speech_thresholds",
    "apply_postprocessing_to_predictions",
    "apply_postprocessing_to_work",
    "build_ollama_utterances_format_schema",
    "build_work_messages",
    "call_ollama_chat",
    "compute_politeness_pairwise_agreement",
    "compute_prompt_hash",
    "evaluate_model_predictions",
    "evaluate_step1_gate",
    "extract_outer_quote_spans",
    "extract_utterance_features_from_text",
    "extract_work_utterance_features",
    "format_utterance_features_block",
    "get_prompt",
    "is_forbidden_null_literal",
    "load_f9_dataset",
    "number_outer_quotes",
    "replay_saved_result_with_postprocess",
    "validate_sensor_v2_utterances",
    "validate_speaker_id",
]
