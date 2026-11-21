"""Cache keyed by scene_hash + per-work manifest + per-work trait ledger.

Layout (under CACHE_ROOT):
  strong/<scene_hash>.json        strong-model annotation (sensor.strong/1) for one scene text
  manifests/<work_id>.json        ordered scene list of a work with cached/pending status
  ledger/<work_id>.traits.json    baseline 6-axis traits per character (scene side stores variations only)
An entry is "cached" when it exists AND its instruction_version equals the current INSTRUCTION_VERSION.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from common import CACHE_ROOT, INSTRUCTION_VERSION, MANIFEST_SCHEMA, read_json, write_json


class Cache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else CACHE_ROOT

    def entry_path(self, scene_hash: str) -> Path:
        return self.root / "strong" / f"{scene_hash}.json"

    def get(self, scene_hash: str) -> dict[str, Any] | None:
        p = self.entry_path(scene_hash)
        return read_json(p) if p.exists() else None

    def has(self, scene_hash: str) -> bool:
        e = self.get(scene_hash)
        return e is not None and e.get("instruction_version") == INSTRUCTION_VERSION

    def put(self, entry: dict[str, Any]) -> Path:
        p = self.entry_path(entry["scene_hash"])
        write_json(p, entry)
        return p

    def delete(self, scene_hash: str) -> bool:
        p = self.entry_path(scene_hash)
        if p.exists():
            p.unlink()
            return True
        return False

    # ledger
    def ledger_path(self, work_id: str) -> Path:
        return self.root / "ledger" / f"{work_id}.traits.json"

    def ledger(self, work_id: str) -> dict[str, list[int]]:
        p = self.ledger_path(work_id)
        return read_json(p)["baseline"] if p.exists() else {}

    def put_ledger(self, work_id: str, baseline: dict[str, list[int]], note: str = "") -> None:
        write_json(self.ledger_path(work_id), {"schema": "sensor.ledger/1", "work_id": work_id, "note": note, "baseline": baseline})

    # manifest
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

    def write_manifest(self, work_id: str, scenes: list[dict[str, Any]], splitter: str) -> dict[str, Any]:
        rows = []
        for s in scenes:
            e = self.get(s["scene_hash"])
            ok = e is not None and e.get("instruction_version") == INSTRUCTION_VERSION
            rows.append(
                {
                    "index": s["index"],
                    "scene_id": f"{work_id}:s{s['index']}",
                    "scene_hash": s["scene_hash"],
                    "char_len": s["char_len"],
                    "status": "cached" if ok else "pending",
                    "provenance": e.get("provenance") if e else None,
                    "annotator": e.get("annotator") if e else None,
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
