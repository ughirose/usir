"""Phase 1.5 Stage 2 Step 3: ingest validates against the frozen (committed) ledger the pack was rendered from;
a repeat of a pending ledger_addition is absorbed (first writer wins). Measurement logs are appended without loss."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import epoch as epoch_mod  # noqa: E402
import prepare as prepare_mod  # noqa: E402
from jsonl_append import append_json  # noqa: E402
from test_sensor_runner import WID, do_ingest, env, good_entry  # noqa: E402,F401

ADDS = {"taro": [0, 0, 0, 0, 0, 0], "hanako": [1, 0, 0, -1, 0, 0]}


def _next_pack_scene(env):
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert len(r["packs_written"]) == 1
    return r["packs_written"][0]["scene_index"]


def test_repeating_a_pending_addition_is_accepted_first_writer_wins(env):
    """Same condition as the Step 2 run `ep`: empty ledger, 3 scenes, scene 0 adds two characters, scene 1 adds them again."""
    cache = env["cache"]
    cache.put_ledger(WID, {})  # empty committed ledger
    scenes = env["seg"]["scenes"]
    assert _next_pack_scene(env) == 0
    e0 = good_entry(scenes[0], facts=[], ledger_additions=dict(ADDS))
    assert do_ingest(env, e0)["accepted"]
    assert set(cache.pending_additions(WID)) == {"taro", "hanako"} and cache.ledger_committed(WID) == {}
    assert _next_pack_scene(env) == 1
    pack1 = json.loads(next((env["packs"] / WID).glob("*.pack.json")).read_text(encoding="utf-8"))
    assert pack1["ledger_baseline"] == {}  # the annotator still sees the frozen (empty) ledger, so it adds again
    again = {"taro": [0, 0, 0, 0, 0, 1], "hanako": [0, 0, 0, 0, 0, 1]}  # different vectors: ignored, not an overwrite
    res = do_ingest(env, good_entry(scenes[1], facts=[], ledger_additions=again))
    assert res["accepted"], res["reasons"]
    assert res["ledger_additions_pending"] == []  # nothing new recorded
    pend = cache.pending_additions(WID)
    assert pend["taro"]["baseline"] == ADDS["taro"] and pend["hanako"]["baseline"] == ADDS["hanako"]  # first writer wins
    assert _next_pack_scene(env) == 2
    assert do_ingest(env, good_entry(scenes[2], facts=[], ledger_additions=dict(ADDS)))["accepted"]
    st = epoch_mod.commit_epoch(cache, WID)
    assert st["status"] == "committed" and sorted(st["applied"]) == ["hanako", "taro"]


def test_committed_baselines_still_cannot_be_overwritten(env):
    scenes = env["seg"]["scenes"]
    assert _next_pack_scene(env) == 0
    res = do_ingest(env, good_entry(scenes[0], facts=[], ledger_additions={"taro": [0, 0, 0, 0, 0, 1]}))
    assert not res["accepted"] and any("baseline already exists" in r for r in res["reasons"])


def test_pending_character_is_unknown_to_trait_changes_unless_added_like_the_pack_shows(env):
    """Validation uses the pack's ledger: trait_changes for a character that exists only as pending still needs ledger_additions."""
    cache = env["cache"]
    cache.put_ledger(WID, {"hanako": [1, 0, 0, -1, 0, 0]})
    scenes = env["seg"]["scenes"]
    assert _next_pack_scene(env) == 0
    assert do_ingest(env, good_entry(scenes[0], facts=[], ledger_additions={"taro": [0] * 6}))["accepted"]
    assert _next_pack_scene(env) == 1
    res = do_ingest(env, good_entry(scenes[1], facts=[]))  # trait_changes for taro, no addition
    assert not res["accepted"] and any("no ledger baseline for 'taro'" in r for r in res["reasons"])
    assert do_ingest(env, good_entry(scenes[1], facts=[], ledger_additions={"taro": [0] * 6}))["accepted"]


def _writer(args):
    path, name, n = args
    for i in range(n):
        append_json(Path(path), {"pid": f"{name}.{i}", "event": "start", "t": float(i)})


def test_parallel_appends_lose_no_record(tmp_path):
    path = tmp_path / "timing.jsonl"
    procs, n = 8, 40
    with mp.get_context("spawn").Pool(procs) as pool:
        pool.map(_writer, [(str(path), f"p{k}", n) for k in range(procs)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == procs * n and all(l.strip() for l in lines)
    ids = {json.loads(l)["pid"] for l in lines}
    assert ids == {f"p{k}.{i}" for k in range(procs) for i in range(n)}
    assert not list(tmp_path.glob("*.lock"))
