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
from usir_py.sensor_eval.metrics import (
    compute_politeness_pairwise_agreement,
    evaluate_model_predictions,
    validate_sensor_v2_utterances,
)
from usir_py.sensor_eval.ollama_client import OllamaResponse, call_ollama_chat
from usir_py.sensor_eval.prompt import (
    PROMPT_VERSION,
    build_ollama_utterances_format_schema,
    build_work_messages,
    compute_prompt_hash,
)

__all__ = [
    "F9Character",
    "F9Dataset",
    "F9UtteranceGold",
    "F9Work",
    "OllamaResponse",
    "PROMPT_VERSION",
    "build_ollama_utterances_format_schema",
    "build_work_messages",
    "call_ollama_chat",
    "compute_politeness_pairwise_agreement",
    "compute_prompt_hash",
    "evaluate_model_predictions",
    "extract_outer_quote_spans",
    "load_f9_dataset",
    "number_outer_quotes",
    "validate_sensor_v2_utterances",
]
