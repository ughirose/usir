"""Cache for the sensor runner (Phase 1.5 Stage 1 layout; Stage 2 Step 1 keys).

Layout (under CACHE_ROOT):
  l2/<scene_hash>/<l2_key>.json   strong-model annotation (sensor.strong/1) wrapped with its key parts (sensor.cache.l2/2)
                                  key = H(rendered pack view + instruction_version + model id + sampling), see render.py.
                                  The wrapper also stores the prior facts shown to the annotator ("premise changed" check).
                                  Stage 1 entries (sensor.cache.l2/1, component-hash key) are never read as hits; they are
                                  adopted under the new key by migration only when their old key still matches.
  l1/<l1_key>.json                local Step 1 utterance block (speaker / addressee / Policy X register) of one scene
                                  key = scene_text_hash + roster_hash + prev_tail_hash + step1_algo_version (+ local_utt_hash)
  strong/<scene_hash>.json        LEGACY, unkeyed entries (pre Stage 1). Adopted into l2/ on first resolve when
                                  legacy_adopt is on (the migrate command does this for the whole cache), then removed.
  manifests/<work_id>.json        ordered scene list of a work with cached/pending status
  ledger/<work_id>.traits.json    COMMITTED baseline 6-axis traits per character (scene side stores variations only);
                                  prepare / resolve build packs from this frozen ledger
  ledger/<work_id>.pending.json   ledger_additions accepted by ingest during the current epoch (applied by `epoch commit`)
  ledger/<work_id>.epoch.json     epoch counter of the current run (max 2 commits)
  stale/<work_id>.json            "premise changed" scenes (reused annotation made under different prior facts) and
                                  scenes forced back to pending by `stale --reannotate`
Reads never raise on a damaged entry: parse errors, schema or key mismatches are a miss (the caller recomputes and
overwrites). Writes are atomic (temp file -> fsync -> os.replace). Nothing is deleted except by delete() or gc().
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from common import CACHE_ROOT, INSTRUCTION_VERSION, MANIFEST_SCHEMA, read_json, write_json
from keys import L1_SCHEMA, L2_SCHEMA, L2_SCHEMA_V1, L2_SCHEMA_V2


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
        return self.has_candidate(scene_hash)

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

    def put_l2(self, scene_hash: str, key: str, entry: dict[str, Any], parts: dict[str, Any], source: str = "ingest",
               prior_facts: list[dict[str, Any]] | None = None) -> Path:
        p = self._l2_dir(scene_hash) / f"{key}.json"
        write_json(p, {"schema": L2_SCHEMA, "key": key, "scene_hash": scene_hash, "parts": parts, "source": source,
                       "prior_facts": prior_facts, "entry": entry})
        return p

    def get_l2_wrapper(self, scene_hash: str, key: str) -> dict[str, Any] | None:
        """The stored wrapper (parts, source, prior_facts, entry) of a valid v2 entry, else None."""
        w = _load(self._l2_dir(scene_hash) / f"{key}.json", L2_SCHEMA)
        if not w or w.get("key") != key:
            return None
        e = w.get("entry")
        if not isinstance(e, dict) or e.get("scene_hash") != scene_hash or e.get("instruction_version") != INSTRUCTION_VERSION:
            return None
        return w

    def get_l2_v2(self, scene_hash: str, v2_key: str) -> dict[str, Any] | None:
        """Wrapper stored by Step 1 under its pack_bytes/1 key (used for migration only)."""
        w = _load(self._l2_dir(scene_hash) / f"{v2_key}.json", L2_SCHEMA_V2)
        if not w or w.get("key") != v2_key:
            return None
        e = w.get("entry")
        if not isinstance(e, dict) or e.get("scene_hash") != scene_hash or e.get("instruction_version") != INSTRUCTION_VERSION:
            return None
        return w

    def get_l2_v1(self, scene_hash: str, v1_key: str) -> dict[str, Any] | None:
        """Entry stored by Stage 1 under its component-hash key (used for migration only)."""
        w = _load(self._l2_dir(scene_hash) / f"{v1_key}.json", L2_SCHEMA_V1)
        if not w or w.get("key") != v1_key:
            return None
        e = w.get("entry")
        if not isinstance(e, dict) or e.get("scene_hash") != scene_hash or e.get("instruction_version") != INSTRUCTION_VERSION:
            return None
        return e

    def resolve_l2(self, scene_hash: str, key: str, parts: dict[str, Any] | None = None, v1_key: str | None = None,
                   v2_key: str | None = None) -> tuple[dict[str, Any] | None, str]:
        """Return (entry, how). how: 'hit' | 'adopted' | 'miss'.
        Adopted (moved under `key`): a legacy unkeyed entry; a Step 1 (l2/2) entry whose recomputed pack_bytes/1 key `v2_key`
        still matches (its stored prior-facts digest is kept; roster_all_ids were part of that key, so the roster digest is the
        current one); a Stage 1 (l2/1) entry whose old key `v1_key` still matches (history unknown: both digests None)."""
        e = self.get_l2(scene_hash, key)
        if e is not None:
            return e, "hit"
        if self.legacy_adopt:
            parts = parts or {}
            unknown = {**parts, "prior_facts_digest": None, "roster_all_digest": None, "premise_unknown": True}
            leg = self.get(scene_hash)
            if leg is not None and leg.get("instruction_version") == INSTRUCTION_VERSION and leg.get("scene_hash") == scene_hash:
                self.put_l2(scene_hash, key, leg, unknown, source="adopted-legacy")
                self.entry_path(scene_hash).unlink()
                return leg, "adopted"
            if v2_key is not None:
                w = self.get_l2_v2(scene_hash, v2_key)
                if w is not None:
                    old_parts = w.get("parts") or {}
                    self.put_l2(scene_hash, key, w["entry"], {**parts, "prior_facts_digest": old_parts.get("prior_facts_digest")},
                                source="migrated-v2", prior_facts=w.get("prior_facts"))
                    return w["entry"], "adopted"
            if v1_key is not None:
                old = self.get_l2_v1(scene_hash, v1_key)
                if old is not None:
                    self.put_l2(scene_hash, key, old, unknown, source="migrated-v1")
                    return old, "adopted"
        return None, "miss"

    def has_candidate(self, scene_hash: str) -> bool:
        """True if an entry for this scene text exists that may become a hit once its predecessor is resolved
        (same instruction version; whether the pack bytes still match is decided by resolve). Legacy and Stage 1 entries count."""
        leg = self.get(scene_hash)
        if leg is not None and leg.get("instruction_version") == INSTRUCTION_VERSION:
            return True
        d = self._l2_dir(scene_hash)
        for p in d.glob("*.json") if d.exists() else []:
            w = _load(p)
            if isinstance(w, dict) and w.get("schema") in (L2_SCHEMA, L2_SCHEMA_V2, L2_SCHEMA_V1) and w.get("entry", {}).get("instruction_version") == INSTRUCTION_VERSION:
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

    # ---- ledger (epoch-frozen) ----------------------------------------------------------------------------------
    def ledger_path(self, work_id: str) -> Path:
        return self.root / "ledger" / f"{work_id}.traits.json"

    def pending_path(self, work_id: str) -> Path:
        return self.root / "ledger" / f"{work_id}.pending.json"

    def epoch_path(self, work_id: str) -> Path:
        return self.root / "ledger" / f"{work_id}.epoch.json"

    def ledger_committed(self, work_id: str) -> dict[str, list[int]]:
        """The frozen ledger: packs, keys and resolution are built from this one for the whole epoch."""
        p = self.ledger_path(work_id)
        return read_json(p)["baseline"] if p.exists() else {}

    def pending_additions(self, work_id: str) -> dict[str, dict[str, Any]]:
        """{character: {"baseline": [6 ints], "scene_hash": ..., "seq": n}} accepted by ingest and not yet committed."""
        p = self.pending_path(work_id)
        return read_json(p).get("additions", {}) if p.exists() else {}

    def ledger(self, work_id: str) -> dict[str, list[int]]:
        """Effective ledger = committed baseline + pending additions (validation and merge see additions at once;
        pack building does not: it uses ledger_committed())."""
        led = dict(self.ledger_committed(work_id))
        for ch, rec in self.pending_additions(work_id).items():
            led.setdefault(ch, rec["baseline"])
        return led

    def put_ledger(self, work_id: str, baseline: dict[str, list[int]], note: str = "") -> None:
        """Write the committed ledger. Pending additions that are now part of it are dropped."""
        write_json(self.ledger_path(work_id), {"schema": "sensor.ledger/1", "work_id": work_id, "note": note, "baseline": baseline})
        pend = self.pending_additions(work_id)
        left = {k: v for k, v in pend.items() if k not in baseline}
        if left != pend:
            self._write_pending(work_id, left)

    def _write_pending(self, work_id: str, additions: dict[str, dict[str, Any]]) -> None:
        if additions:
            write_json(self.pending_path(work_id), {"schema": "sensor.ledger.pending/1", "work_id": work_id, "additions": additions})
        elif self.pending_path(work_id).exists():
            self.pending_path(work_id).unlink()

    def add_pending(self, work_id: str, additions: dict[str, list[int]], scene_hash: str) -> list[str]:
        """Record ledger_additions of an accepted entry without touching the committed ledger (first writer wins)."""
        pend = self.pending_additions(work_id)
        committed = self.ledger_committed(work_id)
        added = []
        for ch, vec in additions.items():
            if ch in committed or ch in pend:
                continue
            pend[ch] = {"baseline": list(vec), "scene_hash": scene_hash, "seq": len(pend) + 1}
            added.append(ch)
        if added:
            self._write_pending(work_id, pend)
        return added

    def epoch_state(self, work_id: str) -> dict[str, Any]:
        p = self.epoch_path(work_id)
        return read_json(p) if p.exists() else {"epoch": 0, "history": []}

    def write_epoch_state(self, work_id: str, st: dict[str, Any]) -> None:
        write_json(self.epoch_path(work_id), {"schema": "sensor.epoch/1", "work_id": work_id, **st})

    # ---- premise-changed registry / forced re-annotation -------------------------------------------------------
    def stale_path(self, work_id: str) -> Path:
        return self.root / "stale" / f"{work_id}.json"

    def stale_state(self, work_id: str) -> dict[str, Any]:
        p = self.stale_path(work_id)
        j = _load(p) if p.exists() else None
        j = j if isinstance(j, dict) else {}
        return {"forced": j.get("forced", []), "premise_changed": j.get("premise_changed", []), "roster_changed": j.get("roster_changed", [])}

    def write_stale_state(self, work_id: str, st: dict[str, Any]) -> None:
        write_json(self.stale_path(work_id), {"schema": "sensor.stale/1", "work_id": work_id, "forced": st["forced"], "premise_changed": st["premise_changed"], "roster_changed": st.get("roster_changed", [])})

    def forced(self, work_id: str) -> set[str]:
        return set(self.stale_state(work_id).get("forced", []))

    def force_reannotate(self, work_id: str, scene_hashes: list[str]) -> None:
        st = self.stale_state(work_id)
        st["forced"] = sorted(set(st.get("forced", [])) | set(scene_hashes))
        self.write_stale_state(work_id, st)

    def clear_forced(self, work_id: str, scene_hash: str) -> None:
        st = self.stale_state(work_id)
        if scene_hash in st.get("forced", []):
            st["forced"] = [h for h in st["forced"] if h != scene_hash]
            self.write_stale_state(work_id, st)

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
