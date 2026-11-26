"""Pack rendering and the L2 key (Phase 1.5 Stage 2 Step 1).

The L2 key is computed from the rendered pack - the exact input the annotator is given - instead of from separately hashed
components, so "what the LLM sees" and "what the key covers" cannot drift apart:

  key = H(layer, key schema, instruction_version, annotator model id, sampling, pack view)

`pack view` is the pack with three kinds of fields normalised:
  * position / metadata fields are dropped: scene_id, scene_index, scene_count, scene_hash (derived from scene_text),
    output_path, l2_parts, l2_key. A scene index is a position, not content (Stage 1 invariant: inserting or removing
    a scene must not invalidate its neighbours);
  * `prior` is replaced by its core: present (sorted), pov_character and, only for works with a hand-curated location list,
    location_id. Free-text location, time, trait_state and recent_facts are shown to the annotator but are NOT keyed;
  * roster_all_ids (key schema pack_bytes/2) is dropped: it is shown to the annotator but only the characters that appear in
    the scene are keyed (roster, ledger_baseline, prior present). A change of the full list is recorded at hit time
    (roster_all_digest -> stale reason "roster_changed") instead of invalidating every scene;
  * roster is keyed as a set (sorted by id): the order of the character file is not content;
  * everything else is hashed as written: scene text, numbered text, roster (visible characters only, with aliases and
    every other field), ledger_baseline (visible characters only), local_utterances, instructions,
    output_template, narration settings.

`recent_facts` change detection ("premise changed") is separate: see prior_facts_digest().
"""
from __future__ import annotations

from typing import Any

import strong
from common import ANNOTATION_MODEL_ID, ANNOTATION_SAMPLING, INSTRUCTION_VERSION, PACK_SCHEMA
from keys import canon_hash
from locations import location_id

KEY_SCHEMA = "pack_bytes/2"
KEY_SCHEMA_V1 = "pack_bytes/1"  # Step 1: roster_all_ids inside the key; kept only to recompute old keys for migration
EXCLUDED_FIELDS = ("scene_id", "scene_index", "scene_count", "scene_hash", "output_path", "l2_parts", "l2_key")
EXCLUDED_FIELDS_V1 = EXCLUDED_FIELDS


def relevant_roster(characters: list[dict[str, Any]], text: str, must: set[str]) -> list[dict[str, Any]]:
    out = []
    for c in characters:
        names = [c["id"], *c.get("aliases", [])]
        if c["id"] in must or any(n and n in text for n in names):
            out.append(c)
    return out


def apply_variations(baseline: dict[str, list[int]], st: dict[str, Any]) -> dict[str, list[int]]:
    """Current trait vector of each character active in `st` (ledger + scene variations, clamped)."""
    cur = {}
    for c in st.get("trait_changes", []):
        vec = list(baseline.get(c["character_id"], [0] * 6))
        for ch in c.get("changes") or []:
            vec[ch["axis"]] += ch["delta"]
        cur[c["character_id"]] = [max(-1, min(1, v)) for v in vec]
    return cur


def prior_summary(prev_scene: dict[str, Any] | None, prev_entry: dict[str, Any] | None, ledger: dict[str, list[int]]) -> dict[str, Any]:
    """Prior shown to the annotator. `prev_entry` is the strictly resolved annotation of the previous scene (or None)."""
    if prev_scene is None:
        return {"status": "none"}
    e = prev_entry
    if e is None:
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


def work_context(work: dict[str, Any], characters: list[dict[str, Any]], ledger: dict[str, list[int]], local: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Everything build_pack needs besides the scene itself. `ledger` must be the committed (epoch-frozen) ledger."""
    must_base = {work.get("narrator")} - {None} if work.get("narration") == "first" else set()
    if work.get("pov"):
        must_base.add(work["pov"])
    return {"work": work, "work_id": work.get("work_id"), "characters": characters, "roster_ids": [c["id"] for c in characters],
            "ledger": ledger, "local": local, "must_base": must_base}


def build_pack(ctx: dict[str, Any], scene: dict[str, Any], scene_count: int, prior: dict[str, Any], work_id: str) -> dict[str, Any]:
    """The pack the annotator receives, without output_path / l2_parts / l2_key (added by prepare)."""
    work, idx = ctx["work"], scene["index"]
    lu = ctx["local"].get(scene["scene_hash"], [])
    must = set(ctx["must_base"]) | set(prior.get("present", []))
    for u in lu:
        must |= {x for x in (u["speaker_id"], u["addressee_id"]) if x}
    roster = relevant_roster(ctx["characters"], scene["scene_text"], must)
    rid = {c["id"] for c in roster}
    return {
        "schema": PACK_SCHEMA,
        "instruction_version": INSTRUCTION_VERSION,
        "instruction_hash": strong.instruction_hash(),
        "work_id": work_id,
        "scene_id": f"{work_id}:s{idx}",
        "scene_index": idx,
        "scene_count": scene_count,
        "scene_hash": scene["scene_hash"],
        "char_len": scene["char_len"],
        "narration": work.get("narration"),
        "narrator": work.get("narrator"),
        "work_pov": work.get("pov"),
        "instructions": strong.INSTRUCTIONS,
        "output_template": strong.OUTPUT_TEMPLATE,
        "scene_text": scene["scene_text"],
        "numbered_text": scene["numbered_text"],
        "sentence_count": len(scene["sentences"]),
        "utterance_count": len(scene["utterances"]),
        "roster": roster,
        "roster_all_ids": ctx["roster_ids"],
        "ledger_baseline": {k: v for k, v in ctx["ledger"].items() if k in rid},
        "local_utterances": lu,
        "prior": prior,
    }


def prior_core(prior: dict[str, Any], work_id: str) -> dict[str, Any]:
    """The part of the prior that is keyed: present + pov, plus location_id when the work has a curated list."""
    if prior.get("status") != "cached":
        return {"status": prior.get("status")}
    core: dict[str, Any] = {"status": "cached", "present": sorted(prior.get("present", [])), "pov_character": prior.get("pov_character")}
    lid = location_id(work_id, prior.get("location"))
    if lid is not None:
        core["location_id"] = lid
    return core


def prior_state_hash_v2(core: dict[str, Any]) -> str:
    return canon_hash({"prior_v2": core})


def prior_facts_digest(prior: dict[str, Any]) -> str | None:
    """Digest of the recent facts shown in the prior. None when the pack has no cached prior."""
    if prior.get("status") != "cached":
        return None
    return canon_hash({"prior_facts_v1": prior.get("recent_facts", [])})


def roster_all_digest(pack: dict[str, Any]) -> str:
    """Digest of the full character id list shown to the annotator (as a set). Stored with the entry, compared at hit time."""
    return canon_hash({"roster_all_v1": sorted(pack.get("roster_all_ids", []))})


def pack_view(pack: dict[str, Any], work_id: str, schema: str = KEY_SCHEMA) -> dict[str, Any]:
    v = {k: val for k, val in pack.items() if k not in EXCLUDED_FIELDS}
    v["prior"] = prior_core(pack["prior"], work_id)
    if schema == KEY_SCHEMA:
        v.pop("roster_all_ids", None)
        v["roster"] = sorted(pack.get("roster", []), key=lambda c: str(c.get("id")) if isinstance(c, dict) else str(c))
    elif schema != KEY_SCHEMA_V1:
        raise ValueError(f"unknown key schema {schema}")
    return v


def pack_key(pack: dict[str, Any], work_id: str, model_id: str | None = None, sampling: str | None = None, schema: str = KEY_SCHEMA) -> str:
    return canon_hash({
        "layer": "L2", "key_schema": schema, "instruction_version": pack.get("instruction_version", INSTRUCTION_VERSION),
        "model_id": model_id if model_id is not None else ANNOTATION_MODEL_ID,
        "sampling": sampling if sampling is not None else ANNOTATION_SAMPLING,
        "pack": pack_view(pack, work_id, schema),
    })


def key_parts(pack: dict[str, Any], work_id: str) -> dict[str, Any]:
    """Informational parts stored next to an entry (the key is the hash of the pack view, not of these)."""
    from keys import scene_text_hash

    core = prior_core(pack["prior"], work_id)
    return {
        "key_schema": KEY_SCHEMA,
        "scene_text_hash": scene_text_hash(pack["scene_text"]),
        "instruction_version": pack.get("instruction_version", INSTRUCTION_VERSION),
        "model_id": ANNOTATION_MODEL_ID,
        "sampling": ANNOTATION_SAMPLING,
        "pack_view_hash": canon_hash(pack_view(pack, work_id)),
        "prior_core": core,
        "prior_state_hash": prior_state_hash_v2(core),
        "prior_facts_digest": prior_facts_digest(pack["prior"]),
        "roster_all_digest": roster_all_digest(pack),
    }
