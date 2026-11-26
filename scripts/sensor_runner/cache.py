"""Cache for the sensor runner (Phase 1.5 Stage 1: strict composite keys, design memo 2026-11-24 task6).

Layout (under CACHE_ROOT):
  l2/<scene_hash>/<l2_key>.json   strong-model annotation (sensor.strong/1) wrapped with its key parts
                                  key = scene_text_hash + instruction_version + roster_hash + ledger_hash + prior_state_hash
  l1/<l1_key>.json                local Step 1 utterance block (speaker / addressee / Policy X register) of one scene
                                  key = scene_text_hash + roster_hash + prev_tail_hash + step1_algo_version (+ local_utt_hash)
  strong/<scene_hash>.json        LEGACY, unkeyed entries (pre Stage 1). Adopted into l2/ on first resolve when
                                  legacy_adopt is on (the migrate command does this for the whole cache), then removed.
  manifests/<work_id>.json        ordered scene list of a work with cached/pending status
  ledger/<work_id>.traits.json    baseline 6-axis traits per character (scene side stores variations only)
Reads never raise on a damaged entry: parse errors, schema or key mismatches are a miss (the caller recomputes and
overwrites). Writes are atomic (temp file -> fsync -> os.replace). Nothing is deleted except by delete() or gc().
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from common import CACHE_ROOT, INSTRUCTION_VERSION, MANIFEST_SCHEMA, read_json, write_json
from keys import L1_SCHEMA, L2_SCHEMA


def _load(path: Path, schema: str | None = None) -> Any | None:
    """Read a cache file; any damage (missing, unparsable, wrong schema) is a miss."""
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if schema is not None and (not isinstance(j, dict) or j.get("schema") != schema):
        return None
    return j


class Cache:
    def __init__(self, root: Path | None = None, legacy_adopt: bool = True):
        self.root = Path(root) if root else CACHE_ROOT
        self.legacy_adopt = legacy_adopt

    # ---- legacy (unkeyed) entries -------------------------------------------------------------------------------
    def entry_path(self, scene_hash: str) -> Path:
        return self.root / "strong" / f"{scene_hash}.json"

    def get(self, scene_hash: str) -> dict[str, Any] | None:
        """Legacy unkeyed entry only. Keyed entries are read with resolve_l2()."""
        return _load(self.entry_path(scene_hash))

    def put(self, entry: dict[str, Any]) -> Path:
        """Legacy unkeyed write (kept for callers without key context; adopted on first resolve)."""
        p = self.entry_path(entry["scene_hash"])
        write_json(p, entry)
        return p

    def _l2_dir(self, scene_hash: str) -> Path:
        return self.root / "l2" / scene_hash

    def has(self, scene_hash: str) -> bool:
        """True if any entry (keyed or legacy) of the current instruction version exists for the scene text.
        Informational only: validity under the current roster / ledger / prior is decided by resolve_l2()."""
        e = self.get(scene_hash)
        if e is not None and e.get("instruction_version") == INSTRUCTION_VERSION:
            return True
        d = self._l2_dir(scene_hash)
        for p in d.glob("*.json") if d.exists() else []:
            w = _load(p, L2_SCHEMA)
            if w and w.get("entry", {}).get("instruction_version") == INSTRUCTION_VERSION:
                return True
        return False

    def delete(self, scene_hash: str) -> bool:
        found = False
        p = self.entry_path(scene_hash)
        if p.exists():
            p.unlink()
            found = True
        d = self._l2_dir(scene_hash)
        if d.exists():
            for q in d.glob("*.json"):
                q.unlink()
                found = True
            try:
                d.rmdir()
            except OSError:
                pass
        return found

    # ---- L2 (strong annotation) ---------------------------------------------------------------------------------
    def get_l2(self, scene_hash: str, key: str) -> dict[str, Any] | None:
        w = _load(self._l2_dir(scene_hash) / f"{key}.json", L2_SCHEMA)
        if not w or w.get("key") != key:
            return None
        e = w.get("entry")
        if not isinstance(e, dict) or e.get("scene_hash") != scene_hash or e.get("instruction_version") != INSTRUCTION_VERSION:
            return None
        return e

    def put_l2(self, scene_hash: str, key: str, entry: dict[str, Any], parts: dict[str, Any], source: str = "ingest") -> Path:
        p = self._l2_dir(scene_hash) / f"{key}.json"
        write_json(p, {"schema": L2_SCHEMA, "key": key, "scene_hash": scene_hash, "parts": parts, "source": source, "entry": entry})
        return p

    def resolve_l2(self, scene_hash: str, key: str, parts: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str]:
        """Return (entry, how). how: 'hit' | 'adopted' (legacy entry moved under `key`) | 'miss'."""
        e = self.get_l2(scene_hash, key)
        if e is not None:
            return e, "hit"
        if self.legacy_adopt:
            leg = self.get(scene_hash)
            if leg is not None and leg.get("instruction_version") == INSTRUCTION_VERSION and leg.get("scene_hash") == scene_hash:
                self.put_l2(scene_hash, key, leg, parts or {}, source="adopted-legacy")
                self.entry_path(scene_hash).unlink()
                return leg, "adopted"
        return None, "miss"

    def has_candidate(self, scene_hash: str, roster_h: str, ledger_h: str) -> bool:
        """True if an entry for this scene text exists that could become a hit once its predecessor is resolved
        (same instruction version, roster and ledger; only the prior state is still unknown). Legacy entries count."""
        leg = self.get(scene_hash)
        if leg is not None and leg.get("instruction_version") == INSTRUCTION_VERSION:
            return True
        d = self._l2_dir(scene_hash)
        for p in d.glob("*.json") if d.exists() else []:
            w = _load(p, L2_SCHEMA)
            pr = (w or {}).get("parts") or {}
            if w and pr.get("roster_hash") == roster_h and pr.get("ledger_hash") == ledger_h and pr.get("instruction_version") == INSTRUCTION_VERSION:
                return True
        return False

    # ---- L1 (local Step 1 utterance block) ----------------------------------------------------------------------
    def get_l1(self, key: str) -> list[dict[str, Any]] | None:
        w = _load(self.root / "l1" / f"{key}.json", L1_SCHEMA)
        if not w or w.get("key") != key or not isinstance(w.get("utterances"), list):
            return None
        return w["utterances"]

    def put_l1(self, key: str, utterances: list[dict[str, Any]], parts: dict[str, Any]) -> Path:
        p = self.root / "l1" / f"{key}.json"
        write_json(p, {"schema": L1_SCHEMA, "key": key, "parts": parts, "utterances": utterances})
        return p

    # ---- gc (explicit only; never called by prepare / merge / diagnose) ----------------------------------------
    def gc(self, max_age_days: float = 30.0, apply: bool = False, now: float | None = None) -> list[Path]:
        """List (and with apply=True delete) l1/l2 files whose mtime is older than max_age_days."""
        cutoff = (now if now is not None else time.time()) - max_age_days * 86400
        stale = [p for sub in ("l1", "l2") if (self.root / sub).exists() for p in (self.root / sub).rglob("*.json") if p.stat().st_mtime < cutoff]
        if apply:
            for p in stale:
                p.unlink()
        return stale

    # ---- ledger -------------------------------------------------------------------------------------------------
    def ledger_path(self, work_id: str) -> Path:
        return self.root / "ledger" / f"{work_id}.traits.json"

    def ledger(self, work_id: str) -> dict[str, list[int]]:
        p = self.ledger_path(work_id)
        return read_json(p)["baseline"] if p.exists() else {}

    def put_ledger(self, work_id: str, baseline: dict[str, list[int]], note: str = "") -> None:
        write_json(self.ledger_path(work_id), {"schema": "sensor.ledger/1", "work_id": work_id, "note": note, "baseline": baseline})

    # ---- manifest -----------------------------------------------------------------------------------------------
    def manifest_path(self, work_id: str) -> Path:
        return self.root / "manifests" / f"{work_id}.json"

    def manifest(self, work_id: str) -> dict[str, Any] | None:
        p = self.manifest_path(work_id)
        return read_json(p) if p.exists() else None

    def mark_cached(self, work_id: str, scene_hash: str, entry: dict[str, Any]) -> None:
        m = self.manifest(work_id)
        if m is None:
            return
        for r in m["scenes"]:
            if r["scene_hash"] == scene_hash:
                r["status"], r["provenance"], r["annotator"] = "cached", entry.get("provenance"), entry.get("annotator")
        m["cached_count"] = sum(r["status"] == "cached" for r in m["scenes"])
        write_json(self.manifest_path(work_id), m)

    def write_manifest(self, work_id: str, scenes: list[dict[str, Any]], splitter: str, resolved: list[dict[str, Any] | None] | None = None) -> dict[str, Any]:
        """`resolved[i]` is the strictly keyed entry of scene i (None = pending). Without it, falls back to has()."""
        rows = []
        for i, s in enumerate(scenes):
            if resolved is not None:
                e = resolved[i]
                ok = e is not None
            else:
                e = self.get(s["scene_hash"])
                ok = self.has(s["scene_hash"])
            rows.append(
                {
                    "index": s["index"],
                    "scene_id": f"{work_id}:s{s['index']}",
                    "scene_hash": s["scene_hash"],
                    "char_len": s["char_len"],
                    "status": "cached" if ok else "pending",
                    "provenance": e.get("provenance") if e and ok else None,
                    "annotator": e.get("annotator") if e and ok else None,
                }
            )
        m = {
            "schema": MANIFEST_SCHEMA,
            "work_id": work_id,
            "splitter_version": splitter,
            "instruction_version": INSTRUCTION_VERSION,
            "scene_count": len(rows),
            "cached_count": sum(r["status"] == "cached" for r in rows),
            "scenes": rows,
        }
        write_json(self.manifest_path(work_id), m)
        return m
