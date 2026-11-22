"""Thin re-export so scripts can import the existing sensor.v2 validator and sentence resolver (python/usir_py)."""
from common import REPO  # noqa: F401  (ensures python/ is on sys.path)
from usir_py.sensor_eval.scene_annotator import resolve_sentence_index, validate_annotation_schema  # noqa: F401
