"""merge: local utterance annotation + strong-model annotation + Policy X register -> one sensor.v2 record per scene.

Ownership of fields (design, fixed):
  local model + deterministic : utterance_index, speaker_id, addressee_id, evidence   (from the local-side file)
  Policy X (deterministic)    : register.politeness (Janome surface rule), register.first_person (surface regex)
  strong model (cache)        : time, location, pov, presence (strong model; local ids absent from it are only counted), character_states (ledger + variations),
                                fact_updates, nagare, awase
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common import INSTRUCTION_VERSION, SCHEMA_PATH, SPLITTER, content_hash, read_json
from keys import canon_hash, l1_key, prev_tail_hash, roster_hash, scene_text_hash, step1_version
from resolve import record_premise_changes, resolve_work
from strong import instruction_hash

FIRST_PERSON = ["僕", "ぼく", "わたくし", "私", "わたし", "俺", "おれ", "あたし"]

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from janome.tokenizer import Tokenizer

        _tok = Tokenizer()
    return _tok


def policy_x_politeness(quote: str) -> str:
    """Policy X: surface sentence-final grammar (Janome). 'ambiguous' is written as sensor.v2 'unknown'."""
    from build_annotations import POLITE_MAP
    from build_modern_gold import classify_politeness_surface

    return POLITE_MAP[classify_politeness_surface(quote, _tokenizer())["politeness"]]


def first_person_surface(quote: str, speaker: str | None) -> str | None:
    if not speaker:
        return None
    return next((p for p in FIRST_PERSON if p in quote), None)


def load_local(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Local-side utterance annotations keyed by scene_hash.

    Accepts sensor.local_utterances/1 ({"scenes": {hash: [..]}}) or a sensor.v2 annotation list (contract.scene_hash)."""
    j = read_json(Path(path))
    if isinstance(j, list):
        return {a["contract"]["scene_hash"]: a["utterances"] for a in j}
    return j["scenes"]


def state_changed_object(f: dict[str, Any]) -> str:
    if f.get("category") in (None, "unclassified") or not f.get("target"):
        return f["text"]
    return f"[{f['category']}] {f['target']}: {f['text']}"


def ledger_hash(ledger: dict[str, list[int]]) -> str:
    return "ledger:" + content_hash(ledger)[:12]


def build_utterances(scene: dict[str, Any], local_utts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Local Step 1 block of one scene (L1 cache payload): local speaker/addressee/evidence + Policy X register.
    Pure function of the scene text, the scene's utterance spans and `local_utts`."""
    text = scene["scene_text"]
    local_by_idx = {u["utterance_index"]: u for u in (local_utts or [])}
    utts = []
    for u in scene["utterances"]:
        i = u["index"]
        quote = text[u["local"]["start"] - 1 : u["local"]["end"] + 1]
        lu = local_by_idx.get(i)
        if lu is None:
            lu = {"speaker_id": None, "addressee_id": None, "evidence": "ambiguous"}
        utts.append(
            {
                "utterance_index": i,
                "speaker_id": lu["speaker_id"],
                "addressee_id": lu["addressee_id"],
                "evidence": lu["evidence"],
                "register": {"politeness": policy_x_politeness(quote), "first_person": first_person_surface(quote, lu["speaker_id"])},
            }
        )
    return utts


def merge_scene(
    work_id: str,
    scene: dict[str, Any],
    scene_count: int,
    strong: dict[str, Any],
    local_utts: list[dict[str, Any]] | None,
    ledger: dict[str, list[int]],
    roster_ids: list[str],
    utts_cached: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (sensor.v2 record, stats). `scene` is one element of `nishiki segment` output.
    `utts_cached`: L1 hit payload (build_utterances output); when given, Step 1 is not recomputed."""
    from scene_annotator_shim import resolve_sentence_index, validate_annotation_schema

    text = scene["scene_text"]
    stats = {"missing_local_utterances": 0, "local_ids_absent_from_present": 0, "sentence_index_unresolved": 0}
    local_by_idx = {u["utterance_index"]: u for u in (local_utts or [])}
    stats["missing_local_utterances"] = sum(1 for u in scene["utterances"] if u["index"] not in local_by_idx)
    utts = utts_cached if utts_cached is not None else build_utterances(scene, local_utts)

    # Presence is the strong model's field. Local speakers/addressees that it left out are NOT added (that would hide
    # Sasae off_stage_speaker findings); they are counted so the session can review them.
    present = [p["id"] for p in strong["present"]]
    for u in utts:
        for k in ("speaker_id", "addressee_id"):
            v = u[k]
            if v is not None and v not in present:
                stats["local_ids_absent_from_present"] += 1

    states = []
    for c in strong["trait_changes"]:
        vec = list(ledger[c["character_id"]])
        for ch in c.get("changes") or []:
            vec[ch["axis"]] += ch["delta"]
        states.append({"character_id": c["character_id"], "values_6d": [max(-1, min(1, v)) for v in vec], "evidence": c["evidence"]})

    sentences = [(s["index"], s["local"]["start"], s["local"]["end"]) for s in scene["sentences"]]
    facts = []
    for f in strong["facts"]:
        o = state_changed_object(f) if f["predicate"] == "STATE_CHANGED" else f["object"]
        rec = {"subject": f["subject"], "predicate": f["predicate"], "object": o, "evidence": f["evidence"]}
        if f.get("basis"):
            si, why = resolve_sentence_index(f["basis"], text, sentences)
            if si is not None:
                rec["sentence_index"] = si
            else:
                stats["sentence_index_unresolved"] += 1
        facts.append(rec)

    t = strong["time"]
    time = {"relative_order": t["relative_order"], "evidence": t["evidence"]}
    for k in ("explicit_time", "elapsed_hint"):
        if t.get(k):
            time[k] = t[k]
    aw = strong["awase"]
    rec = {
        "contract": {
            "schema_version": "sensor.v2",
            "splitter_version": SPLITTER,
            "scene_id": f"{work_id}:s{scene['index']}",
            "scene_hash": scene["scene_hash"],
            "ledger_hash": ledger_hash(ledger),
            "model_id": strong["annotator"],
            "prompt_hash": f"strong:{INSTRUCTION_VERSION}:{instruction_hash()[:16]}",
            "work_id": work_id,
            "scene_index": scene["index"],
            "scene_count": scene_count,
            "char_len": scene["char_len"],
        },
        "scene_meta": {
            "time": time,
            "location_id": strong["location"],
            "present_character_ids": present,
            "pov_character": strong["pov_character"],
            "nagare": {
                "event_top3": [{"code": i["code"], "rank_weight": w} for i, w in zip(strong["event_top3"], (3, 2, 1))],
                "mood_top3": [{"code": i["code"], "rank_weight": w} for i, w in zip(strong["mood_top3"], (3, 2, 1))],
            },
            "awase": {
                "f_event_valence": aw["F_valence"],
                "f_event_arousal": aw["F_arousal"],
                "f_style_valence": aw["f_valence"],
                "f_style_arousal": aw["f_arousal"],
            },
        },
        "utterances": utts,
        "character_states": states,
        "fact_updates": facts,
    }
    schema = read_json(SCHEMA_PATH)
    errs = validate_annotation_schema(rec, schema, roster_ids)
    return rec, {**stats, "errors": errs}


def merge_work(work_id, cache, seg, local_path=None, partial=False, use_l1=True):
    """Merge every scene of a work. Returns (records, report). Raises if scenes are unresolved and not `partial`.

    L2 entries are resolved with strict keys (resolve.resolve_work); the local Step 1 block of each scene is served from
    the L1 cache (key: scene text + roster + previous-scene tail + Step 1 algorithm version + local-file entry)."""
    from common import LOCAL_DIR, load_work

    work = load_work(work_id)
    roster_ids = [c["id"] for c in work["characters"]]
    lp = Path(local_path) if local_path else LOCAL_DIR / f"{work_id}.utterances.json"
    local = load_local(lp) if lp.exists() else {}
    ledger = cache.ledger(work_id)
    scenes = seg["scenes"]
    resolved = resolve_work(work_id, cache, scenes, work["characters"], local=local)
    roster_h = roster_hash(work["characters"])
    algo = step1_version() if use_l1 else None
    records, report = [], {"work_id": work_id, "scenes": len(scenes), "merged": 0, "uncached": [], "errors": [],
                           "missing_local_utterances": 0, "local_ids_absent_from_present": 0, "sentence_index_unresolved": 0,
                           "l1": {"hit": 0, "miss": 0}, "l2": {"hit": 0, "adopted": 0, "miss": 0, "blocked": 0}}
    report["premise_changed"] = [{"scene_index": r["scene_index"], "scene_hash": r["scene_hash"]} for r in record_premise_changes(cache, work_id, resolved)]
    for s, rv in zip(scenes, resolved):
        report["l2"][rv.how] += 1
        e = rv.entry
        if e is None:
            report["uncached"].append(s["index"])
            continue
        utts = None
        if use_l1:
            k1 = l1_key(scene_text_hash(s["scene_text"]), roster_h, prev_tail_hash(scenes[s["index"] - 1] if s["index"] > 0 else None),
                        canon_hash({"local_v1": local.get(s["scene_hash"])}), algo)
            utts = cache.get_l1(k1)
            if utts is None:
                report["l1"]["miss"] += 1
                utts = build_utterances(s, local.get(s["scene_hash"]))
                cache.put_l1(k1, utts, {"scene_text_hash": scene_text_hash(s["scene_text"]), "roster_hash": roster_h, "step1_algo_version": algo})
            else:
                report["l1"]["hit"] += 1
        rec, st = merge_scene(work_id, s, len(scenes), e, local.get(s["scene_hash"]), ledger, roster_ids, utts_cached=utts)
        for k in ("missing_local_utterances", "local_ids_absent_from_present", "sentence_index_unresolved"):
            report[k] += st[k]
        report["errors"] += [(s["index"], x) for x in st["errors"]]
        records.append(rec)
        report["merged"] += 1
    if report["uncached"] and not partial:
        raise RuntimeError(f"{work_id}: uncached scenes {report['uncached']} (run prepare, annotate, ingest first)")
    return records, report
