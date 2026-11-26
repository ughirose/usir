"""prepare: segment a work (nishiki segment), number sentences/utterances, hash scenes, and emit strong-model packs
only for scenes that are not cached under the current instruction version."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cache import Cache
from common import (
    LOCAL_DIR,
    REPO,
    SPLITTER,
    load_work,
    read_json,
    segment,
    write_json,
)
from render import apply_variations, prior_summary, relevant_roster  # noqa: F401  (re-exported for callers/tests)
from resolve import load_local_utterances, record_premise_changes, resolve_work, roster_changed_rows

PACKS_DIR = Path(__file__).resolve().parent / "packs"
OUT_DIR = Path(__file__).resolve().parent / "out"


def prepare_work(
    work_id: str,
    cache: Cache,
    nishiki: str | None = None,
    packs_dir: Path | None = None,
    limit: int | None = None,
    seg: dict[str, Any] | None = None,
    segment_fn: Callable[[str, Path], dict[str, Any]] = segment,
    local_path: Path | None = None,
    defer_blocked: bool = False,
) -> dict[str, Any]:
    """defer_blocked: do not emit packs for scenes whose predecessor is itself unresolved (their strict key is not
    computable yet; after the predecessor is ingested they are usually cache hits via the Early Cutoff)."""
    work = load_work(work_id)
    if seg is None:
        seg = segment_fn(nishiki, work["text_abs"])
    scenes = seg["scenes"]
    packs_dir = Path(packs_dir) if packs_dir else PACKS_DIR
    wdir = packs_dir / work_id
    wdir.mkdir(parents=True, exist_ok=True)

    lpath = Path(local_path) if local_path else LOCAL_DIR / f"{work_id}.utterances.json"
    local = load_local_utterances(work_id, lpath)
    resolved = resolve_work(work_id, cache, scenes, work["characters"], local=local)
    premise = record_premise_changes(cache, work_id, resolved)
    manifest = cache.write_manifest(work_id, scenes, seg.get("splitter", SPLITTER), resolved=[r.entry for r in resolved])
    pend_r = [r for r in resolved if r.entry is None]
    # A blocked scene (predecessor unresolved) is not annotated now when defer_blocked, or when an entry for its text
    # already exists that may still hit once the predecessor is resolved (Early Cutoff).
    skip = [r for r in pend_r if r.blocked and (defer_blocked or r.extra.get("candidate"))]
    deferred = [r.index for r in skip]
    pend_r = [r for r in pend_r if r not in skip]
    by_index = {r.index: r for r in resolved}
    pending = [r.scene for r in pend_r]
    if limit is not None:
        pending = pending[:limit]
    keep = {f"{s['scene_hash'][:12]}.pack.json" for s in pending}
    removed = 0
    for old in wdir.glob("*.pack.json"):
        if old.name not in keep:
            old.unlink()
            removed += 1

    written = []
    for s_ in pending:
        idx = s_["index"]
        rv = by_index[idx]
        h12 = s_["scene_hash"][:12]
        pack = dict(rv.pack)
        pack["output_path"] = f"scripts/sensor_runner/out/{work_id}/{h12}.strong.json"
        # strict L2 key parts at pack time; None when the predecessor was unresolved (ingest then rejects the pack)
        pack["l2_parts"] = rv.parts
        pack["l2_key"] = rv.key
        p = wdir / f"{h12}.pack.json"
        write_json(p, pack)
        written.append({"scene_index": idx, "scene_hash": s_["scene_hash"], "pack": str(p)})
    return {
        "work_id": work_id,
        "scene_count": len(scenes),
        "cached": manifest["cached_count"],
        "pending": len(scenes) - manifest["cached_count"],
        "packs_written": written,
        "stale_packs_removed": removed,
        "deferred_blocked": deferred,
        "premise_changed": [{"scene_index": r["scene_index"], "scene_hash": r["scene_hash"]} for r in premise],
        "roster_changed": [{"scene_index": r["scene_index"], "scene_hash": r["scene_hash"]} for r in roster_changed_rows(resolved)],
        "forced": sorted(r.index for r in resolved if r.extra.get("forced")),
        "l2": {"hit": sum(r.how == "hit" for r in resolved), "adopted": sum(r.how == "adopted" for r in resolved),
               "miss": sum(r.how == "miss" for r in resolved), "blocked": sum(r.blocked for r in resolved)},
    }
