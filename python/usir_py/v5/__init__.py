"""USIR v5 scene annotation, six-pole atlas, and splitter package."""

from usir_py.v5.aozora_scene_splitter import (
    AozoraCorpusSplitter,
    build_scene_annotation,
    compute_heuristic_signals,
    extract_polarity_divergence_spans,
    split_long_paragraph,
    split_work_into_scenes,
)
from usir_py.v5.six_pole_atlas import (
    GoldilocksVoidFinder,
    NarrativeRGNormalizer,
    aggregate_work_profile_from_scenes,
    normalized_hsic,
    svd_lowdin_orthogonalize,
)

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
