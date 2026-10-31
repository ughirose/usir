"""USIR Python Reference & Golden Implementation Package."""

from .corpus_40_benchmark import CORPUS, PHASE_ORDER, PHASE_NAMES_JA
from .extractor_24d import (
    extract_usir_24d,
    run_loocv,
    split_sentences,
    char_bigram_entropy,
    ABSTRACT_FILLERS,
    PHYSICAL_SENSE_CHARS,
    VOICED_CHARS,
    FORMAL_LOGIC_TERMS,
    FIGURATIVE_MARKERS,
    RHETORICAL_RUPTURE,
    APOPHATIC_VOID_OPS,
)
from .grpo_reward import compute_grpo_reward, USIR_CANONICAL_TARGETS

__all__ = [
    "CORPUS",
    "PHASE_ORDER",
    "PHASE_NAMES_JA",
    "extract_usir_24d",
    "run_loocv",
    "split_sentences",
    "char_bigram_entropy",
    "ABSTRACT_FILLERS",
    "PHYSICAL_SENSE_CHARS",
    "VOICED_CHARS",
    "FORMAL_LOGIC_TERMS",
    "FIGURATIVE_MARKERS",
    "RHETORICAL_RUPTURE",
    "APOPHATIC_VOID_OPS",
    "compute_grpo_reward",
    "USIR_CANONICAL_TARGETS",
]
