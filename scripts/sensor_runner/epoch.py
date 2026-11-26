"""Ledger epochs and "premise changed" handling (Phase 1.5 Stage 2 Step 1).

Epoch freeze: while a run is in progress, packs, keys and resolution use the COMMITTED ledger only. `ledger_additions` of
accepted entries are recorded as pending (ingest) and applied together by `commit_epoch`. After a commit, only scenes whose
rendered pack changed (a character added to the ledger is visible in the scene) miss the cache and are re-annotated in the
next epoch. A run is cut off after MAX_EPOCHS commits; whatever is still unresolved or still pending is reported, not looped.
"""
from __future__ import annotations

from typing import Any

from cache import Cache
from resolve import resolve_work

MAX_EPOCHS = 2


def epoch_status(cache: Cache, work_id: str) -> dict[str, Any]:
    st = cache.epoch_state(work_id)
    return {"work_id": work_id, "epoch": st.get("epoch", 0), "max_epochs": MAX_EPOCHS,
            "pending_additions": sorted(cache.pending_additions(work_id)), "history": st.get("history", [])}


def commit_epoch(cache: Cache, work_id: str, max_epochs: int = MAX_EPOCHS) -> dict[str, Any]:
    """Apply pending ledger additions to the committed ledger. status: committed | converged | cutoff."""
    st = cache.epoch_state(work_id)
    pend = cache.pending_additions(work_id)
    n = st.get("epoch", 0)
    if not pend:
        cache.write_epoch_state(work_id, {"epoch": 0, "history": st.get("history", [])})  # run finished: counter reset
        return {"status": "converged", "epoch": n, "applied": [], "unapplied": []}
    if n >= max_epochs:
        return {"status": "cutoff", "epoch": n, "applied": [], "unapplied": sorted(pend)}
    committed = cache.ledger_committed(work_id)
    led = {**committed, **{ch: rec["baseline"] for ch, rec in pend.items() if ch not in committed}}
    cache.put_ledger(work_id, led, note=f"epoch {n + 1} commit of ledger_additions")
    hist = [*st.get("history", []), {"epoch": n + 1, "applied": sorted(pend)}]
    cache.write_epoch_state(work_id, {"epoch": n + 1, "history": hist})
    return {"status": "committed", "epoch": n + 1, "applied": sorted(pend), "unapplied": []}


def reset_epochs(cache: Cache, work_id: str) -> None:
    """Start a new run (counter back to 0). Pending additions are kept and become the first commit of the new run."""
    st = cache.epoch_state(work_id)
    cache.write_epoch_state(work_id, {"epoch": 0, "history": st.get("history", [])})


def unresolved_scenes(cache: Cache, work_id: str, scenes: list[dict[str, Any]], characters: list[dict[str, Any]],
                      local: dict[str, list[dict[str, Any]]] | None = None) -> list[int]:
    """Scene indexes with no valid entry under the current committed ledger (pending annotation, incl. blocked)."""
    return [r.index for r in resolve_work(work_id, cache, scenes, characters, local=local) if r.entry is None]


def list_premise_changed(cache: Cache, work_id: str, scenes: list[dict[str, Any]], characters: list[dict[str, Any]],
                         local: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    return [r.premise_changed for r in resolve_work(work_id, cache, scenes, characters, local=local) if r.premise_changed]


def reannotate(cache: Cache, work_id: str, scene_hashes: list[str]) -> None:
    """Explicitly mark scenes for re-annotation (prepare then emits packs for them). Never called automatically."""
    cache.force_reannotate(work_id, scene_hashes)
