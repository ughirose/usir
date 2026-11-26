"""Sequential L2 resolution of a work: computes each scene's strict key and looks it up (design memo 3.2, Early Cutoff).

Scene i's key contains prior_state_hash(entry of scene i-1). Scene i can therefore only be resolved when scene i-1 is
resolved (a hit, an adopted legacy entry, or just ingested). A scene whose predecessor is unresolved is 'blocked': its
key cannot be computed yet. Once the predecessor is annotated, a blocked scene is a hit again if the predecessor's
structured prior state (location, present set, POV) did not change - the chain of invalidation stops there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cache import Cache
from common import INSTRUCTION_VERSION
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
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.how == "blocked"


def resolve_work(work_id: str, cache: Cache, scenes: list[dict[str, Any]], characters: list[dict[str, Any]]) -> list[Resolved]:
    roster_h = roster_hash(characters)
    ledger_h = ledger_key_hash(cache.ledger(work_id))
    out: list[Resolved] = []
    prev: Resolved | None = None
    for s in scenes:
        r = Resolved(index=s["index"], scene=s)
        if prev is not None and prev.entry is None:
            r.how = "blocked"
            r.extra["candidate"] = cache.has_candidate(s["scene_hash"], roster_h, ledger_h)
        else:
            r.prev_entry = prev.entry if prev is not None else None
            parts = {
                "scene_text_hash": scene_text_hash(s["scene_text"]),
                "instruction_version": INSTRUCTION_VERSION,
                "roster_hash": roster_h,
                "ledger_hash": ledger_h,
                "prior_state_hash": prior_state_hash(r.prev_entry),
            }
            r.parts = parts
            r.key = l2_key(parts["scene_text_hash"], roster_h, ledger_h, parts["prior_state_hash"])
            r.entry, r.how = cache.resolve_l2(s["scene_hash"], r.key, parts)
        out.append(r)
        prev = r
    return out
