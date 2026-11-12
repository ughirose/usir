"""USIR Python Reference & Golden Implementation Package.

Implements safe PEP 562 deferred lazy loading to ensure submodules like
`usir_py.v5.u6_guards` can be imported in environments without optional
dependencies (e.g. scipy, torch, janome).
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = [
    "HTHPolyLexicon",
    "DualSenseEntry",
    "jump_distance_64bit",
    "lca_depth_64bit",
    "CORPUS",
    "PHASE_ORDER",
    "PHASE_NAMES_JA",
    "extract_usir_24d",
    "extract_24d",
    "extract_stage1_signals",
    "extract_stage2_multiresolution",
    "extract_multiresolution_24d",
    "fit_contrastive_fisher_metric",
    "embed_sentence_unit",
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
    "RedlineToken",
    "RhythmBarcode",
    "FactRetentionGuard",
    "RoiMetrics",
    "RedlineDiffResult",
    "PresalesUseCaseVariant",
    "PresalesUseCase",
    "extract_core_facts",
    "check_fact_retention",
    "compute_rhythm_barcode",
    "compute_redline_tokens",
    "render_redline_html",
    "compute_redline_diff_and_roi",
    "apply_deterministic_style_rewrite",
    "CASE_1_LITERARY",
    "CASE_2_ESSAY",
    "CASE_3_B2B_PROPOSAL",
    "PRESALES_SHOWCASE_CASES",
    "HSMMLongformDirector",
    "PhaseWell",
    "compute_cv_l",
    "compute_wundt_berlyne_utility",
    "AttachedLoRALinear",
    "attach_lora_to_model",
    "set_lora_enabled",
    "run_authentic_grpo_lora_forge",
    "SymplecticActivationSteerer",
    "align_polar_fourier_phase",
    "compute_polar_fourier_phase",
    "compute_packed_cholesky_24d",
    "unpack_cholesky_24d",
    "precision_matrix_from_packed_cholesky_24d",
    "compile_tier_m_binary",
    "load_tier_m_binary",
    "classify_mode_24d",
    "OSWEStreamingCompiler",
    "USIR_MAGIC",
    "TIER_M_BYTE_LENGTH",
    "ToulminAuditResult",
    "evaluate_toulmin_components",
    "compute_toulmin_potential",
    "compute_usir_style_energy",
    "compute_unified_field_energy",
    "WorldCraftOntologyRule",
    "WorldCraftArbitrator",
    "MomentumBank",
    "visual_functor_forward_24d",
    "visual_functor_inverse",
    "music_functor_forward_24d",
    "music_functor_inverse",
]

_DEFERRED_TABLE: dict[str, tuple[str, str]] = {
    # .corpus_40_benchmark
    "CORPUS": ("corpus_40_benchmark", "CORPUS"),
    "PHASE_ORDER": ("corpus_40_benchmark", "PHASE_ORDER"),
    "PHASE_NAMES_JA": ("corpus_40_benchmark", "PHASE_NAMES_JA"),
    # .extractor_24d
    "extract_usir_24d": ("extractor_24d", "extract_usir_24d"),
    "extract_24d": ("extractor_24d", "extract_24d"),
    "extract_stage1_signals": ("extractor_24d", "extract_stage1_signals"),
    "extract_stage2_multiresolution": ("extractor_24d", "extract_stage2_multiresolution"),
    "extract_multiresolution_24d": ("extractor_24d", "extract_multiresolution_24d"),
    "run_loocv": ("extractor_24d", "run_loocv"),
    "split_sentences": ("extractor_24d", "split_sentences"),
    "char_bigram_entropy": ("extractor_24d", "char_bigram_entropy"),
    "ABSTRACT_FILLERS": ("extractor_24d", "ABSTRACT_FILLERS"),
    "PHYSICAL_SENSE_CHARS": ("extractor_24d", "PHYSICAL_SENSE_CHARS"),
    "VOICED_CHARS": ("extractor_24d", "VOICED_CHARS"),
    "FORMAL_LOGIC_TERMS": ("extractor_24d", "FORMAL_LOGIC_TERMS"),
    "FIGURATIVE_MARKERS": ("extractor_24d", "FIGURATIVE_MARKERS"),
    "RHETORICAL_RUPTURE": ("extractor_24d", "RHETORICAL_RUPTURE"),
    "APOPHATIC_VOID_OPS": ("extractor_24d", "APOPHATIC_VOID_OPS"),
    # .manifold_geodesic
    "fit_contrastive_fisher_metric": ("manifold_geodesic", "fit_contrastive_fisher_metric"),
    "embed_sentence_unit": ("manifold_geodesic", "embed_sentence_unit"),
    # .grpo_reward
    "compute_grpo_reward": ("grpo_reward", "compute_grpo_reward"),
    "USIR_CANONICAL_TARGETS": ("grpo_reward", "USIR_CANONICAL_TARGETS"),
    # .presales_showcase
    "RedlineToken": ("presales_showcase", "RedlineToken"),
    "RhythmBarcode": ("presales_showcase", "RhythmBarcode"),
    "FactRetentionGuard": ("presales_showcase", "FactRetentionGuard"),
    "RoiMetrics": ("presales_showcase", "RoiMetrics"),
    "RedlineDiffResult": ("presales_showcase", "RedlineDiffResult"),
    "PresalesUseCaseVariant": ("presales_showcase", "PresalesUseCaseVariant"),
    "PresalesUseCase": ("presales_showcase", "PresalesUseCase"),
    "extract_core_facts": ("presales_showcase", "extract_core_facts"),
    "check_fact_retention": ("presales_showcase", "check_fact_retention"),
    "compute_rhythm_barcode": ("presales_showcase", "compute_rhythm_barcode"),
    "compute_redline_tokens": ("presales_showcase", "compute_redline_tokens"),
    "render_redline_html": ("presales_showcase", "render_redline_html"),
    "compute_redline_diff_and_roi": ("presales_showcase", "compute_redline_diff_and_roi"),
    "apply_deterministic_style_rewrite": ("presales_showcase", "apply_deterministic_style_rewrite"),
    "CASE_1_LITERARY": ("presales_showcase", "CASE_1_LITERARY"),
    "CASE_2_ESSAY": ("presales_showcase", "CASE_2_ESSAY"),
    "CASE_3_B2B_PROPOSAL": ("presales_showcase", "CASE_3_B2B_PROPOSAL"),
    "PRESALES_SHOWCASE_CASES": ("presales_showcase", "PRESALES_SHOWCASE_CASES"),
    # .hth_lexicon
    "DualSenseEntry": ("hth_lexicon", "DualSenseEntry"),
    "HTHPolyLexicon": ("hth_lexicon", "HTHPolyLexicon"),
    "jump_distance_64bit": ("hth_lexicon", "jump_distance_64bit"),
    "lca_depth_64bit": ("hth_lexicon", "lca_depth_64bit"),
    # .hsmm_longform_director
    "HSMMLongformDirector": ("hsmm_longform_director", "HSMMLongformDirector"),
    "PhaseWell": ("hsmm_longform_director", "PhaseWell"),
    "compute_cv_l": ("hsmm_longform_director", "compute_cv_l"),
    "compute_wundt_berlyne_utility": ("hsmm_longform_director", "compute_wundt_berlyne_utility"),
    # .lora_grpo_forge
    "AttachedLoRALinear": ("lora_grpo_forge", "AttachedLoRALinear"),
    "attach_lora_to_model": ("lora_grpo_forge", "attach_lora_to_model"),
    "set_lora_enabled": ("lora_grpo_forge", "set_lora_enabled"),
    "run_authentic_grpo_lora_forge": ("lora_grpo_forge", "run_authentic_grpo_lora_forge"),
    # .activation_steerer
    "SymplecticActivationSteerer": ("activation_steerer", "SymplecticActivationSteerer"),
    # .usir_forge
    "align_polar_fourier_phase": ("usir_forge", "align_polar_fourier_phase"),
    "compute_polar_fourier_phase": ("usir_forge", "compute_polar_fourier_phase"),
    "compute_packed_cholesky_24d": ("usir_forge", "compute_packed_cholesky_24d"),
    "unpack_cholesky_24d": ("usir_forge", "unpack_cholesky_24d"),
    "precision_matrix_from_packed_cholesky_24d": ("usir_forge", "precision_matrix_from_packed_cholesky_24d"),
    "compile_tier_m_binary": ("usir_forge", "compile_tier_m_binary"),
    "load_tier_m_binary": ("usir_forge", "load_tier_m_binary"),
    "classify_mode_24d": ("usir_forge", "classify_mode_24d"),
    "OSWEStreamingCompiler": ("usir_forge", "OSWEStreamingCompiler"),
    "USIR_MAGIC": ("usir_forge", "USIR_MAGIC"),
    "TIER_M_BYTE_LENGTH": ("usir_forge", "TIER_M_BYTE_LENGTH"),
    # .nishiki_unified_field
    "ToulminAuditResult": ("nishiki_unified_field", "ToulminAuditResult"),
    "evaluate_toulmin_components": ("nishiki_unified_field", "evaluate_toulmin_components"),
    "compute_toulmin_potential": ("nishiki_unified_field", "compute_toulmin_potential"),
    "compute_usir_style_energy": ("nishiki_unified_field", "compute_usir_style_energy"),
    "compute_unified_field_energy": ("nishiki_unified_field", "compute_unified_field_energy"),
    "WorldCraftOntologyRule": ("nishiki_unified_field", "WorldCraftOntologyRule"),
    "WorldCraftArbitrator": ("nishiki_unified_field", "WorldCraftArbitrator"),
    "MomentumBank": ("nishiki_unified_field", "MomentumBank"),
    "visual_functor_forward_24d": ("nishiki_unified_field", "visual_functor_forward_24d"),
    "visual_functor_inverse": ("nishiki_unified_field", "visual_functor_inverse"),
    "music_functor_forward_24d": ("nishiki_unified_field", "music_functor_forward_24d"),
    "music_functor_inverse": ("nishiki_unified_field", "music_functor_inverse"),
}


def __getattr__(name: str) -> Any:
    if name in _DEFERRED_TABLE:
        submod, obj = _DEFERRED_TABLE[name]
        mod = __import__(f"usir_py.{submod}", fromlist=[obj])
        val = getattr(mod, obj)
        globals()[name] = val
        return val
    try:
        mod = __import__(f"usir_py.{name}", fromlist=["*"])
        globals()[name] = mod
        return mod
    except ImportError:
        pass
    raise AttributeError(f"module 'usir_py' has no attribute '{name}'")
