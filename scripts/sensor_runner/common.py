"""Shared helpers: paths, nishiki segment wrapper, text normalisation, canonical JSON."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO / "python", REPO / "scripts" / "phase2", REPO / "scripts" / "step1" / "modern_gold", REPO / "scripts" / "phase2_nagare_awase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CACHE_ROOT = HERE / "cache"
WORKS_DIR = HERE / "works"
LOCAL_DIR = HERE / "local"
SCHEMA_PATH = REPO / "schemas" / "sensor_v2.schema.json"

STRONG_SCHEMA = "sensor.strong/1"
PACK_SCHEMA = "sensor.pack/1"
MANIFEST_SCHEMA = "sensor.manifest/1"
# Bump when the annotation instructions / strong schema change in a way that invalidates cached entries.
INSTRUCTION_VERSION = "strong.v1"
SPLITTER = "usir_v5_splitter@1"
# Annotation settings that are part of every L2 key (changing the annotator model or sampling invalidates the cache).
# The strong annotation runs in external sessions, so the model id is declared here (env SENSOR_ANNOTATOR_MODEL_ID).
# Declared annotator of the committed seed cache (Step 8). The seed L2 keys contain this model_id, so it is the default; an explicit env value still overrides it.
DEFAULT_ANNOTATOR_MODEL_ID = "claude-code-subagent:alias=sonnet(exact-model-id-unverified)"
ANNOTATION_MODEL_ID = os.environ.get("SENSOR_ANNOTATOR_MODEL_ID", DEFAULT_ANNOTATOR_MODEL_ID)
ANNOTATION_SAMPLING = os.environ.get("SENSOR_ANNOTATOR_SAMPLING", "default")

NISHIKI_DEFAULTS = [
    "C:/dev/nishiki-hybrid/target/debug/nishiki.exe",
]

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub("", s)


def in_text(snippet: Any, norm_text: str) -> bool:
    return isinstance(snippet, str) and bool(snippet) and norm(snippet) in norm_text


def dumps(x: Any, indent: int | None = 1) -> str:
    return json.dumps(x, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"


def write_json(path: Path, obj: Any) -> None:
    """Atomic write: temp file in the same directory -> flush + fsync -> os.replace (design memo 3.4). Bytes are
    identical to the previous Path.write_text(dumps(obj), newline="\\n")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(dumps(obj).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha12(s: str) -> str:
    return s[:12]


def content_hash(obj: Any) -> str:
    return hashlib.sha256(dumps(obj, indent=None).encode("utf-8")).hexdigest()


def find_nishiki(explicit: str | None = None) -> str:
    cands = [explicit, os.environ.get("NISHIKI_BIN"), *NISHIKI_DEFAULTS]
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("nishiki binary with `segment`/`diagnose` not found; set NISHIKI_BIN or pass --nishiki")


def segment(nishiki: str, txt: Path) -> dict[str, Any]:
    """`nishiki segment` (nishiki.segment/1): scenes with scene_hash, [S#] sentences, [U#] utterances."""
    r = subprocess.run([nishiki, "segment", str(txt), "--format", "json"], capture_output=True, check=True)
    j = json.loads(r.stdout.decode("utf-8"))
    if j.get("schema") != "nishiki.segment/1":
        raise ValueError(f"unexpected segment schema: {j.get('schema')}")
    return j


def load_work(work_id: str) -> dict[str, Any]:
    """Work registry entry: text path (repo-relative or absolute), characters, narration/pov."""
    spec = read_json(WORKS_DIR / f"{work_id}.work.json")
    tp = Path(spec["text_path"])
    spec["text_abs"] = tp if tp.is_absolute() else REPO / tp
    spec["characters"] = read_json(WORKS_DIR / f"{work_id}.characters.json")
    return spec


def all_work_ids() -> list[str]:
    return sorted(p.name[: -len(".work.json")] for p in WORKS_DIR.glob("*.work.json"))
