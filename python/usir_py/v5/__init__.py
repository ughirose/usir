"""USIR v5 scene annotation and splitter package."""

from usir_py.v5.aozora_scene_splitter import (
    AozoraCorpusSplitter,
    build_scene_annotation,
    compute_heuristic_signals,
    split_long_paragraph,
    split_work_into_scenes,
)

__all__ = [
    "AozoraCorpusSplitter",
    "build_scene_annotation",
    "compute_heuristic_signals",
    "split_long_paragraph",
    "split_work_into_scenes",
]
