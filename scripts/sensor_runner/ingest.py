"""ingest: validate session-written strong annotations (sensor.strong/1) against their pack (scene text, roster, ledger)
and the sensor.v2 schema, then store them in the cache. Invalid files are rejected whole, with reasons; nothing is repaired."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cache import Cache
from common import read_json, write_json
from merge import merge_scene
from prepare import PACKS_DIR
from strong import validate_strong

REJECTS_DIR = Path(__file__).resolve().parent / "rejects"


def find_pack(scene_hash: str, packs_dir: Path) -> Path | None:
    hits = sorted(Path(packs_dir).glob(f"*/{scene_hash[:12]}.pack.json"))
    for h in hits:
        if read_json(h)["scene_hash"] == scene_hash:
            return h
    return None


def ingest_file(path: Path, cache: Cache, packs_dir: Path | None = None, rejects_dir: Path | None = None) -> dict[str, Any]:
    path = Path(path)
    packs_dir = Path(packs_dir) if packs_dir else PACKS_DIR
    rejects_dir = Path(rejects_dir) if rejects_dir else REJECTS_DIR
    res: dict[str, Any] = {"file": str(path), "accepted": False, "reasons": []}

    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - report any parse failure as a reason
        res["reasons"] = [f"json_parse_error: {e}"]
        return _reject(res, path, rejects_dir)
    sh = entry.get("scene_hash") if isinstance(entry, dict) else None
    pack_path = find_pack(sh, packs_dir) if isinstance(sh, str) else None
    if pack_path is None:
        res["reasons"] = ["no_pack_for_scene_hash (run prepare first; hash must be of a pending scene)"]
        return _reject(res, path, rejects_dir)
    pack = read_json(pack_path)
    res["work_id"], res["scene_id"] = pack["work_id"], pack["scene_id"]

    # Validate against the committed (epoch-frozen) ledger: the same snapshot the pack was rendered from. A character that is
    # already pending (added by an earlier scene in this epoch) is therefore not 'already existing' here; add_pending keeps
    # the first baseline (first writer wins) and ignores the repeat.
    ledger = cache.ledger_committed(pack["work_id"])
    reasons = validate_strong(entry, pack["scene_text"], pack["roster_all_ids"], ledger, pack["scene_hash"])
    if not reasons:
        # dry merge (no local utterances) to confirm the result is a valid sensor.v2 record
        full_ledger = {**ledger, **(entry.get("ledger_additions") or {})}
        scene = {"scene_text": pack["scene_text"], "utterances": [], "sentences": [], "index": pack["scene_index"],
                 "scene_hash": pack["scene_hash"], "char_len": pack["char_len"]}
        _, st = merge_scene(pack["work_id"], scene, pack["scene_count"], entry, None, full_ledger, pack["roster_all_ids"])
        reasons = [f"sensor.v2: {e}" for e in st["errors"]]
    if reasons:
        res["reasons"] = reasons
        return _reject(res, path, rejects_dir)

    # The entry is stored under the key of the pack the annotator actually saw (pack bytes view at prepare time).
    parts = pack.get("l2_parts")
    if not parts or not pack.get("l2_key"):
        res["reasons"] = ["pack_prior_unresolved: the pack was generated while the previous scene was not annotated; "
                          "ingest that scene first, then re-run prepare and annotate the regenerated pack"]
        return _reject(res, path, rejects_dir)

    # ledger_additions are NOT applied to the frozen ledger now: they are recorded as pending and applied by `epoch commit`.
    adds = entry.get("ledger_additions") or {}
    res["ledger_additions_pending"] = cache.add_pending(pack["work_id"], adds, pack["scene_hash"]) if adds else []
    prior = pack.get("prior") or {}
    prior_facts = prior.get("recent_facts") if prior.get("status") == "cached" else None
    cache.put_l2(pack["scene_hash"], pack["l2_key"], entry, parts, prior_facts=prior_facts)
    cache.clear_forced(pack["work_id"], pack["scene_hash"])
    cache.mark_cached(pack["work_id"], pack["scene_hash"], entry)
    res["accepted"] = True
    return res


def _reject(res: dict[str, Any], path: Path, rejects_dir: Path) -> dict[str, Any]:
    write_json(rejects_dir / f"{path.stem}.reject.json", {"file": str(path), "reasons": res["reasons"]})
    return res


def ingest_paths(paths: list[Path], cache: Cache, packs_dir: Path | None = None, rejects_dir: Path | None = None) -> list[dict[str, Any]]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.rglob("*.strong.json")) if p.is_dir() else [p])
    return [ingest_file(f, cache, packs_dir, rejects_dir) for f in files]
