"""Sequential L2 resolution of a work (design memo 3.2, Early Cutoff; Stage 2 Step 1 keys).

Each scene's pack is rendered from the frozen (committed) ledger and the resolved annotation of the previous scene; the L2 key
is the hash of that pack view (render.pack_key). Scene i can therefore only be resolved when scene i-1 is resolved (a hit, an
adopted entry, or just ingested): otherwise its pack's prior is unknown and the scene is 'blocked'. Once the predecessor is
annotated, a blocked scene is a hit again if the predecessor's keyed prior core (present, pov, optional location_id) did not
change - the chain of invalidation stops there (Early Cutoff).

Facts are not keyed. A reused entry whose stored prior facts differ from the facts shown now is reported as `premise_changed`
(recorded by prepare/merge, listed by `cli.py stale`); it is never re-annotated automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import render
from cache import Cache
from common import INSTRUCTION_VERSION, LOCAL_DIR, load_work
from keys import l2_key, ledger_key_hash, prior_state_hash, roster_hash, scene_text_hash


@dataclass
class Resolved:
    index: int
    scene: dict[str, Any]
    entry: dict[str, Any] | None = None
    how: str = "miss"  # hit | adopted | miss | blocked
    key: str | None = None
    parts: dict[str, Any] | None = None
    prev_entry: dict[str, Any] | None = None
    pack: dict[str, Any] | None = None  # the rendered pack (no output_path / l2_*), also for blocked scenes (prior pending)
    premise_changed: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.how == "blocked"


def load_local_utterances(work_id: str, local_path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    from merge import load_local  # lazy: merge imports this module

    lp = Path(local_path) if local_path else LOCAL_DIR / f"{work_id}.utterances.json"
    return load_local(lp) if lp.exists() else {}


def resolve_work(work_id: str, cache: Cache, scenes: list[dict[str, Any]], characters: list[dict[str, Any]],
                 local: dict[str, list[dict[str, Any]]] | None = None) -> list[Resolved]:
    work = load_work(work_id)
    work.setdefault("work_id", work_id)
    if local is None:
        local = load_local_utterances(work_id)
    ledger = cache.ledger_committed(work_id)
    ctx = render.work_context(work, characters, ledger, local)
    forced = cache.forced(work_id)
    # Stage 1 (component-hash) key pieces, used only to adopt old entries whose old key still matches
    roster_h, ledger_h = roster_hash(characters), ledger_key_hash(ledger)
    out: list[Resolved] = []
    prev: Resolved | None = None
    for s in scenes:
        r = Resolved(index=s["index"], scene=s)
        prev_scene = prev.scene if prev is not None else None
        if prev is not None and prev.entry is None:
            r.how = "blocked"
            r.extra["candidate"] = cache.has_candidate(s["scene_hash"])
            r.pack = render.build_pack(ctx, s, len(scenes), render.prior_summary(prev_scene, None, ledger), work_id)
        else:
            r.prev_entry = prev.entry if prev is not None else None
            r.pack = render.build_pack(ctx, s, len(scenes), render.prior_summary(prev_scene, r.prev_entry, ledger), work_id)
            r.parts = render.key_parts(r.pack, work_id)
            r.key = render.pack_key(r.pack, work_id)
            if s["scene_hash"] in forced:
                r.extra["forced"] = True
            else:
                v1_key = l2_key(scene_text_hash(s["scene_text"]), roster_h, ledger_h, prior_state_hash(r.prev_entry), INSTRUCTION_VERSION)
                adopt_parts = {**r.parts, "prior_facts_digest": None, "premise_unknown": True}
                r.entry, r.how = cache.resolve_l2(s["scene_hash"], r.key, adopt_parts, v1_key)
                if r.how == "hit":
                    w = cache.get_l2_wrapper(s["scene_hash"], r.key)
                    stored = ((w or {}).get("parts") or {}).get("prior_facts_digest")
                    now = r.parts["prior_facts_digest"]
                    if stored is not None and now is not None and stored != now:
                        r.premise_changed = {
                            "scene_index": s["index"], "scene_hash": s["scene_hash"], "key": r.key,
                            "stored_digest": stored, "current_digest": now,
                            "stored_prior_facts": (w or {}).get("prior_facts"), "current_prior_facts": r.pack["prior"].get("recent_facts", []),
                        }
        out.append(r)
        prev = r
    return out


def record_premise_changes(cache: Cache, work_id: str, resolved: list[Resolved]) -> list[dict[str, Any]]:
    """Persist the premise-changed list (sensor_runner-side metadata; the diagnose output is untouched). Returns the list."""
    rows = [r.premise_changed for r in resolved if r.premise_changed]
    st = cache.stale_state(work_id)
    if st.get("premise_changed") != rows:
        st["premise_changed"] = rows
        cache.write_stale_state(work_id, st)
    return rows
