"""Phase 1.5 Stage 2 Step 4: a repeat of a pending ledger_addition with a different vector is ignored (first writer wins, as in
Step 3) but recorded for diagnosis: in the pending file while pending, then in the epoch history at commit."""
from __future__ import annotations

import sys
from pathlib import Path

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import epoch as epoch_mod  # noqa: E402
import prepare as prepare_mod  # noqa: E402
from test_sensor_runner import WID, do_ingest, env, good_entry  # noqa: E402,F401


def _step(env):
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert len(r["packs_written"]) == 1


def test_discarded_repeat_vector_is_recorded_and_first_vector_wins(env):
    cache, scenes = env["cache"], env["seg"]["scenes"]
    cache.put_ledger(WID, {})
    first = {"taro": [0, 0, 0, 0, 0, 0], "hanako": [1, 0, 0, -1, 0, 0]}
    _step(env)
    r0 = do_ingest(env, good_entry(scenes[0], facts=[], ledger_additions=first))
    assert r0["accepted"] and r0["ledger_additions_discarded"] == []
    _step(env)
    second = {"taro": [0, 0, 0, 0, 0, 1], "hanako": [1, 0, 0, -1, 0, 0]}  # taro differs, hanako identical
    r1 = do_ingest(env, good_entry(scenes[1], facts=[], ledger_additions=second))
    assert r1["accepted"] and r1["ledger_additions_pending"] == []
    assert r1["ledger_additions_discarded"] == [{"character": "taro", "scene_hash": scenes[1]["scene_hash"], "vector": [0, 0, 0, 0, 0, 1],
                                                "kept_vector": [0, 0, 0, 0, 0, 0], "kept_scene_hash": scenes[0]["scene_hash"]}]
    pend = cache.pending_additions(WID)
    assert pend["taro"]["baseline"] == [0, 0, 0, 0, 0, 0]  # behaviour unchanged: first writer wins
    assert cache.pending_discarded(WID) == r1["ledger_additions_discarded"]  # identical repeats (hanako) are not noise
    _step(env)
    assert do_ingest(env, good_entry(scenes[2], facts=[], ledger_additions=dict(first)))["accepted"]  # identical: nothing more recorded
    assert len(cache.pending_discarded(WID)) == 1
    st = epoch_mod.commit_epoch(cache, WID)
    assert st["status"] == "committed" and st["discarded_repeats"] == r1["ledger_additions_discarded"]
    assert cache.ledger_committed(WID)["taro"] == [0, 0, 0, 0, 0, 0]
    hist = cache.epoch_state(WID)["history"][-1]
    assert hist["discarded_repeats"] == r1["ledger_additions_discarded"] and hist["applied"] == ["hanako", "taro"]
    assert cache.pending_discarded(WID) == [] and not cache.pending_path(WID).exists()


def test_no_record_when_nothing_was_discarded(env):
    cache, scenes = env["cache"], env["seg"]["scenes"]
    cache.put_ledger(WID, {})
    _step(env)
    assert do_ingest(env, good_entry(scenes[0], facts=[], ledger_additions={"taro": [0] * 6}))["accepted"]
    st = epoch_mod.commit_epoch(cache, WID)
    assert st["discarded_repeats"] == [] and "discarded_repeats" not in cache.epoch_state(WID)["history"][-1]
