"""USIR v5 scene annotation, six-pole atlas, and splitter package.

Implements safe PEP 562 deferred lazy loading so that standalone submodules
like `usir_py.v5.u6_guards` can be imported without eagerly importing
heavy numerical packages or other submodules.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AozoraCorpusSplitter",
    "build_scene_annotation",
    "compute_heuristic_signals",
    "extract_polarity_divergence_spans",
    "split_long_paragraph",
    "split_work_into_scenes",
    "NarrativeRGNormalizer",
    "normalized_hsic",
    "svd_lowdin_orthogonalize",
    "GoldilocksVoidFinder",
    "aggregate_work_profile_from_scenes",
]

_DEFERRED_TABLE: dict[str, tuple[str, str]] = {
    "AozoraCorpusSplitter": ("aozora_scene_splitter", "AozoraCorpusSplitter"),
    "build_scene_annotation": ("aozora_scene_splitter", "build_scene_annotation"),
    "compute_heuristic_signals": ("aozora_scene_splitter", "compute_heuristic_signals"),
    "extract_polarity_divergence_spans": ("aozora_scene_splitter", "extract_polarity_divergence_spans"),
    "split_long_paragraph": ("aozora_scene_splitter", "split_long_paragraph"),
    "split_work_into_scenes": ("aozora_scene_splitter", "split_work_into_scenes"),
    "NarrativeRGNormalizer": ("six_pole_atlas", "NarrativeRGNormalizer"),
    "normalized_hsic": ("six_pole_atlas", "normalized_hsic"),
    "svd_lowdin_orthogonalize": ("six_pole_atlas", "svd_lowdin_orthogonalize"),
    "GoldilocksVoidFinder": ("six_pole_atlas", "GoldilocksVoidFinder"),
    "aggregate_work_profile_from_scenes": ("six_pole_atlas", "aggregate_work_profile_from_scenes"),
}


def __getattr__(name: str) -> Any:
    if name in _DEFERRED_TABLE:
        submod, obj = _DEFERRED_TABLE[name]
        mod = __import__(f"usir_py.v5.{submod}", fromlist=[obj])
        val = getattr(mod, obj)
        globals()[name] = val
        return val
    try:
        mod = __import__(f"usir_py.v5.{name}", fromlist=["*"])
        globals()[name] = mod
        return mod
    except ImportError:
        pass
    raise AttributeError(f"module 'usir_py.v5' has no attribute '{name}'")
