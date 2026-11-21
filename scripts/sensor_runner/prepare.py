"""prepare: segment a work (nishiki segment), number sentences/utterances, hash scenes, and emit strong-model packs
only for scenes that are not cached under the current instruction version."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cache import Cache
from common import (
    INSTRUCTION_VERSION,
    LOCAL_DIR,
    PACK_SCHEMA,
    REPO,
    SPLITTER,
    load_work,
    read_json,
    segment,
    write_json,
)
from strong import INSTRUCTIONS, OUTPUT_TEMPLATE, instruction_hash

PACKS_DIR = Path(__file__).resolve().parent / "packs"
OUT_DIR = Path(__file__).resolve().parent / "out"


def relevant_roster(characters: list[dict[str, Any]], text: str, must: set[str]) -> list[dict[str, Any]]:
    out = []
    for c in characters:
        names = [c["id"], *c.get("aliases", [])]
        if c["id"] in must or any(n and n in text for n in names):
            out.append(c)
    return out


def apply_variations(baseline: dict[str, list[int]], strong: dict[str, Any]) -> dict[str, list[int]]:
    """Current trait vector of each character active in `strong` (ledger + scene variations, clamped)."""
    cur = {}
    for c in strong.get("trait_changes", []):
        vec = list(baseline.get(c["character_id"], [0] * 6))
        for ch in c.get("changes") or []:
            vec[ch["axis"]] += ch["delta"]
        cur[c["character_id"]] = [max(-1, min(1, v)) for v in vec]
    return cur


def prior_summary(prev_scene: dict[str, Any] | None, cache: Cache, ledger: dict[str, list[int]]) -> dict[str, Any]:
    if prev_scene is None:
        return {"status": "none"}
    e = cache.get(prev_scene["scene_hash"])
    if e is None or e.get("instruction_version") != INSTRUCTION_VERSION:
        return {"status": "pending", "scene_id_hint": prev_scene["index"],
                "note": "直前の場面は未注釈。先にその出力ファイルを作成・ingest してから、このパックを再生成すること。"}
    return {
        "status": "cached",
        "prev_scene_index": prev_scene["index"],
        "location": e.get("location"),
        "pov_character": e.get("pov_character"),
        "present": [p["id"] for p in e.get("present", [])],
        "time": e.get("time"),
        "recent_facts": [
            {k: f.get(k) for k in ("subject", "predicate", "object", "target", "category", "text")} for f in e.get("facts", [])[-5:]
        ],
        "trait_state": apply_variations(ledger, e),
    }


def prepare_work(
    work_id: str,
    cache: Cache,
    nishiki: str | None = None,
    packs_dir: Path | None = None,
    limit: int | None = None,
    seg: dict[str, Any] | None = None,
    segment_fn: Callable[[str, Path], dict[str, Any]] = segment,
    local_path: Path | None = None,
) -> dict[str, Any]:
    work = load_work(work_id)
    if seg is None:
        seg = segment_fn(nishiki, work["text_abs"])
    scenes = seg["scenes"]
    packs_dir = Path(packs_dir) if packs_dir else PACKS_DIR
    wdir = packs_dir / work_id
    wdir.mkdir(parents=True, exist_ok=True)

    manifest = cache.write_manifest(work_id, scenes, seg.get("splitter", SPLITTER))
    pending = [s for s, row in zip(scenes, manifest["scenes"]) if row["status"] == "pending"]
    if limit is not None:
        pending = pending[:limit]
    keep = {f"{s['scene_hash'][:12]}.pack.json" for s in pending}
    removed = 0
    for old in wdir.glob("*.pack.json"):
        if old.name not in keep:
            old.unlink()
            removed += 1

    lpath = Path(local_path) if local_path else LOCAL_DIR / f"{work_id}.utterances.json"
    local = read_json(lpath)["scenes"] if lpath.exists() else {}
    ledger = cache.ledger(work_id)
    chars = work["characters"]
    roster_ids = [c["id"] for c in chars]
    must_base = {work.get("narrator")} - {None} if work.get("narration") == "first" else set()
    if work.get("pov"):
        must_base.add(work["pov"])

    written = []
    for s in pending:
        idx = s["index"]
        prev = scenes[idx - 1] if idx > 0 else None
        prior = prior_summary(prev, cache, ledger)
        lu = local.get(s["scene_hash"], [])
        must = set(must_base) | set(prior.get("present", []))
        for u in lu:
            must |= {x for x in (u["speaker_id"], u["addressee_id"]) if x}
        roster = relevant_roster(chars, s["scene_text"], must)
        rid = {c["id"] for c in roster}
        h12 = s["scene_hash"][:12]
        pack = {
            "schema": PACK_SCHEMA,
            "instruction_version": INSTRUCTION_VERSION,
            "instruction_hash": instruction_hash(),
            "work_id": work_id,
            "scene_id": f"{work_id}:s{idx}",
            "scene_index": idx,
            "scene_count": len(scenes),
            "scene_hash": s["scene_hash"],
            "char_len": s["char_len"],
            "output_path": f"scripts/sensor_runner/out/{work_id}/{h12}.strong.json",
            "narration": work.get("narration"),
            "narrator": work.get("narrator"),
            "work_pov": work.get("pov"),
            "instructions": INSTRUCTIONS,
            "output_template": OUTPUT_TEMPLATE,
            "scene_text": s["scene_text"],
            "numbered_text": s["numbered_text"],
            "sentence_count": len(s["sentences"]),
            "utterance_count": len(s["utterances"]),
            "roster": roster,
            "roster_all_ids": roster_ids,
            "ledger_baseline": {k: v for k, v in ledger.items() if k in rid},
            "local_utterances": lu,
            "prior": prior,
        }
        p = wdir / f"{h12}.pack.json"
        write_json(p, pack)
        written.append({"scene_index": idx, "scene_hash": s["scene_hash"], "pack": str(p)})
    return {
        "work_id": work_id,
        "scene_count": len(scenes),
        "cached": manifest["cached_count"],
        "pending": len(scenes) - manifest["cached_count"],
        "packs_written": written,
        "stale_packs_removed": removed,
    }
