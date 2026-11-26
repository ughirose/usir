"""Phase 1.5 Stage 1: cache key construction (design memo 2026-11-24 task6, sections 2-3).

All hashes are SHA-256 over canonical JSON (sorted keys, UTF-8, no indentation) or over normalised text.
Scene index and character offsets are never part of any key.
"""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

from common import INSTRUCTION_VERSION, REPO

STEP1_ALGO_BASE = "step1.local+policyx.v1"
L1_SCHEMA = "sensor.cache.l1/1"
L2_SCHEMA_V1 = "sensor.cache.l2/1"  # Stage 1 keyed entries (component-hash key); migrated, not read as hits
L2_SCHEMA = "sensor.cache.l2/2"  # Step 1: key = H(pack bytes view + annotation settings), see render.py
TAIL_SENTENCES = 2  # prev_tail window: last 2 sentences of the previous scene (C1/C2 lookback)
TAIL_FALLBACK_CHARS = 80


def normalize_text(s: str) -> str:
    """Strip a leading BOM and unify CRLF / CR to LF (same pre-processing as diagnose.rs Mapper)."""
    if s.startswith("﻿"):
        s = s[1:]
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon_hash(obj: Any) -> str:
    return _sha(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def scene_text_hash(scene_text: str) -> str:
    """Common base key: SHA256("scene_v1\\0" || normalised scene text). No index, no offsets."""
    return _sha(b"scene_v1\0" + normalize_text(scene_text).encode("utf-8"))


def roster_hash(characters: list[dict[str, Any]]) -> str:
    """Structural hash of characters.json (list order and every field count; any change invalidates all scenes)."""
    return canon_hash({"roster_v1": characters})


def ledger_key_hash(ledger: dict[str, list[int]]) -> str:
    """Hash of the traits.json baseline (whole ledger). Distinct from merge.ledger_hash, which is the output label."""
    return canon_hash({"ledger_v1": ledger})


def prev_tail_hash(prev_scene: dict[str, Any] | None) -> str:
    """Hash of the previous scene's tail lookback window (last TAIL_SENTENCES sentences). First scene: empty window."""
    if prev_scene is None:
        return canon_hash({"tail_v1": None})
    text = normalize_text(prev_scene["scene_text"])
    sents = prev_scene.get("sentences") or []
    if sents:
        start = sents[max(0, len(sents) - TAIL_SENTENCES)]["local"]["start"]
        tail = text[start:]
    else:
        tail = text[-TAIL_FALLBACK_CHARS:]
    return canon_hash({"tail_v1": tail})


def prior_state(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Structured prior summary used as the L2 Early Cutoff key (Gate condition 1).

    Contains only the principal state of the previous scene: location, set of present characters, POV character.
    If the previous scene is re-annotated and these three are unchanged, the chain of invalidation stops there.
    Not part of the key (by design): the previous scene's time, recent_facts and trait_state.
    """
    if entry is None:
        return {"status": "none"}
    return {
        "status": "cached",
        "location": entry.get("location"),
        "present": sorted(p["id"] for p in entry.get("present", [])),
        "pov_character": entry.get("pov_character"),
    }


def prior_state_hash(entry: dict[str, Any] | None) -> str:
    return canon_hash({"prior_v1": prior_state(entry)})


def l2_key(stext_hash: str, roster_h: str, ledger_h: str, prior_h: str, instruction_version: str = INSTRUCTION_VERSION) -> str:
    return canon_hash({
        "layer": "L2", "scene_text_hash": stext_hash, "instruction_version": instruction_version,
        "roster_hash": roster_h, "ledger_hash": ledger_h, "prior_state_hash": prior_h,
    })


def step1_algo_version() -> str:
    """Base label + hash of the sources and library version Step 1 local processing (Policy X, first-person) depends on."""
    from merge import FIRST_PERSON  # lazy: merge imports this module

    h = hashlib.sha256()
    h.update(json.dumps(FIRST_PERSON, ensure_ascii=False).encode("utf-8"))
    for p in (REPO / "scripts" / "phase2" / "build_annotations.py", REPO / "scripts" / "step1" / "modern_gold" / "build_modern_gold.py"):
        h.update(p.read_bytes() if p.exists() else b"missing")
    try:
        h.update(metadata.version("janome").encode())
    except metadata.PackageNotFoundError:
        h.update(b"janome-missing")
    return f"{STEP1_ALGO_BASE}+{h.hexdigest()[:12]}"


_STEP1_VER: str | None = None


def step1_version() -> str:
    global _STEP1_VER
    if _STEP1_VER is None:
        _STEP1_VER = step1_algo_version()
    return _STEP1_VER


def l1_key(stext_hash: str, roster_h: str, tail_h: str, local_h: str, algo_version: str | None = None) -> str:
    """L1 key. `local_h` (hash of the local-side speaker/addressee file entries for the scene) is an extra safety
    component beyond the memo: the local file is an input of Step 1 output, so a change in it must miss."""
    return canon_hash({
        "layer": "L1", "scene_text_hash": stext_hash, "roster_hash": roster_h, "prev_tail_hash": tail_h,
        "local_utt_hash": local_h, "step1_algo_version": algo_version or step1_version(),
    })
