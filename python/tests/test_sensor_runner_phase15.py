"""Phase 1.5 Stage 1: strict L1/L2 cache keys, invalidation rules, Early Cutoff, cache robustness, warm/cold parity."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import common  # noqa: E402
import ingest as ingest_mod  # noqa: E402
import keys  # noqa: E402
import merge as merge_mod  # noqa: E402
import prepare as prepare_mod  # noqa: E402
from cache import Cache  # noqa: E402
from resolve import resolve_work  # noqa: E402
from test_sensor_runner import TEXTS, WID, do_ingest, env, fake_seg, good_entry, prep  # noqa: E402,F401


def annotate_all(env, seg=None, **over):
    """Annotate every pending scene in index order (prepare -> ingest rounds), like the SESSION.md flow."""
    seg = seg or env["seg"]
    env["seg"] = seg
    n = 0
    while True:
        r = prepare_mod.prepare_work(WID, env["cache"], seg=seg, packs_dir=env["packs"], defer_blocked=True)
        if not r["packs_written"]:
            return n
        for p in r["packs_written"]:
            s = seg["scenes"][p["scene_index"]]
            e = good_entry(s, facts=[] if p["scene_index"] else good_entry(s)["facts"], **over)
            res = do_ingest(env, e)
            assert res["accepted"], res["reasons"]
            n += 1


def with_text(env, i, suffix):
    texts = list(TEXTS)
    texts[i] = texts[i] + suffix
    return fake_seg(texts)


# ---------------- keys ----------------
def test_scene_text_hash_normalises_and_ignores_position():
    a = "太郎は笑った。\n花子も笑った。"
    assert keys.scene_text_hash(a) == keys.scene_text_hash("﻿" + a.replace("\n", "\r\n")) == keys.scene_text_hash(a.replace("\n", "\r"))
    assert keys.scene_text_hash(a) != keys.scene_text_hash(a + "。")
    assert keys.scene_text_hash(a) == keys.scene_text_hash(a)  # no index / offset parameter exists


def test_roster_and_ledger_hash_sensitivity():
    r = [{"id": "a", "aliases": ["x"]}]
    assert keys.roster_hash(r) != keys.roster_hash([{"id": "a", "aliases": ["x", "y"]}])
    assert keys.roster_hash(r) == keys.roster_hash([{"aliases": ["x"], "id": "a"}])  # key order irrelevant
    assert keys.ledger_key_hash({"a": [0] * 6}) != keys.ledger_key_hash({"a": [0, 0, 0, 0, 0, 1]})


def test_prior_state_is_structured_and_ignores_facts_and_traits():
    e = good_entry(fake_seg(TEXTS)["scenes"][0])
    h = keys.prior_state_hash(e)
    e2 = json.loads(json.dumps(e))
    e2["facts"] = []
    e2["trait_changes"] = []
    e2["time"]["evidence"] = "explicit"
    e2["annotator"] = "other"
    assert keys.prior_state_hash(e2) == h
    e3 = json.loads(json.dumps(e))
    e3["present"].append({"id": "hanako", "basis": "花子"})
    assert keys.prior_state_hash(e3) != h
    e4 = json.loads(json.dumps(e))
    e4["location"] = "駅"
    assert keys.prior_state_hash(e4) != h
    e5 = json.loads(json.dumps(e))
    e5["pov_character"] = "hanako"
    assert keys.prior_state_hash(e5) != h
    assert keys.prior_state_hash(None) != h


def test_prev_tail_hash_depends_only_on_last_two_sentences():
    s = fake_seg(["A。B。C。D。"])["scenes"][0]
    h = keys.prev_tail_hash(s)
    s2 = fake_seg(["X。B。C。D。"])["scenes"][0]  # change outside the tail window
    assert keys.prev_tail_hash(s2) == h
    s3 = fake_seg(["A。B。C。E。"])["scenes"][0]
    assert keys.prev_tail_hash(s3) != h
    assert keys.prev_tail_hash(None) != h


# ---------------- L2 invalidation rules ----------------
def test_l2_local_edit_invalidates_that_scene_only_and_early_cutoff_stops_chain(env):
    assert annotate_all(env) == 3
    seg2 = with_text(env, 0, "太郎は花子を見た。")
    r = prepare_mod.prepare_work(WID, env["cache"], seg=seg2, packs_dir=env["packs"], defer_blocked=True)
    assert [p["scene_index"] for p in r["packs_written"]] == [0]
    assert r["deferred_blocked"] == [1, 2]
    env["seg"] = seg2
    assert do_ingest(env, good_entry(seg2["scenes"][0]))["accepted"]
    r = prepare_mod.prepare_work(WID, env["cache"], seg=seg2, packs_dir=env["packs"], defer_blocked=True)
    assert r["packs_written"] == [] and r["cached"] == 3  # prior state unchanged -> scenes 1 and 2 are hits


def test_l2_prior_state_change_invalidates_next_scene_only(env):
    annotate_all(env)
    seg2 = with_text(env, 0, "太郎は花子を見た。")
    env["seg"] = seg2
    prepare_mod.prepare_work(WID, env["cache"], seg=seg2, packs_dir=env["packs"], defer_blocked=True)
    changed = good_entry(seg2["scenes"][0], present=[{"id": "taro", "basis": "太郎"}, {"id": "hanako", "basis": "花子"}])
    assert do_ingest(env, changed)["accepted"]
    r = prepare_mod.prepare_work(WID, env["cache"], seg=seg2, packs_dir=env["packs"], defer_blocked=True)
    assert [p["scene_index"] for p in r["packs_written"]] == [1]  # scene 1 sees a different prior; scene 2 stays blocked
    assert do_ingest(env, good_entry(seg2["scenes"][1], facts=[]))["accepted"]
    r = prepare_mod.prepare_work(WID, env["cache"], seg=seg2, packs_dir=env["packs"], defer_blocked=True)
    assert r["packs_written"] == [] and r["cached"] == 3  # scene 1's own state unchanged -> cutoff before scene 2


def test_l2_roster_change_invalidates_every_scene(env):
    annotate_all(env)
    chars = common.read_json(common.WORKS_DIR / f"{WID}.characters.json")
    chars[0]["aliases"].append("たろう")
    common.write_json(common.WORKS_DIR / f"{WID}.characters.json", chars)
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"])
    assert (r["cached"], r["pending"]) == (0, 3)


def test_l2_ledger_change_invalidates_every_scene(env):
    annotate_all(env)
    led = env["cache"].ledger(WID)
    led["taro"] = [0, 0, 0, 0, 0, 1]
    env["cache"].put_ledger(WID, led)
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"])
    assert (r["cached"], r["pending"]) == (0, 3)


def test_instruction_version_is_part_of_the_key():
    a = keys.l2_key("s", "r", "l", "p", "strong.v1")
    assert a != keys.l2_key("s", "r", "l", "p", "strong.v2")


def test_ingest_rejects_pack_made_while_prior_unresolved(env):
    prep(env)  # compat mode: pack for scene 1 is written with prior pending
    r = do_ingest(env, good_entry(env["seg"]["scenes"][1], facts=[]))
    assert not r["accepted"] and "pack_prior_unresolved" in r["reasons"][0]


def test_ingest_stores_under_post_addition_ledger_key(env):
    prep(env)
    s = env["seg"]["scenes"][0]
    e = good_entry(s, ledger_additions={"ghost": [0, 1, 0, 0, 0, 0]})
    e["trait_changes"].append({"character_id": "ghost", "evidence": "inferred", "changes": []})
    assert do_ingest(env, e)["accepted"]
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"])
    assert 0 not in [p["scene_index"] for p in r["packs_written"]]  # scene 0 is a hit under the extended ledger


# ---------------- L1 ----------------
def test_l1_hit_miss_and_tail_cascade_is_one_scene(env):
    annotate_all(env)
    recs, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["l1"] == {"hit": 0, "miss": 3}
    recs_b, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["l1"] == {"hit": 3, "miss": 0} and recs_b == recs
    seg2 = with_text(env, 0, "太郎は花子を見た。")  # scene 0 changes at its tail
    env["seg"] = seg2
    annotate_all(env, seg2)
    _, rep = merge_mod.merge_work(WID, env["cache"], seg2)
    assert rep["l1"] == {"hit": 1, "miss": 2}  # scene 0 (text) and scene 1 (prev_tail); scene 2 untouched


def test_l1_survives_ledger_change_but_not_roster_change(env):
    annotate_all(env)
    merge_mod.merge_work(WID, env["cache"], env["seg"])
    led = env["cache"].ledger(WID)
    led["taro"] = [0, 0, 0, 0, 0, 1]
    env["cache"].put_ledger(WID, led)
    annotate_all(env)  # L2 recomputed for all scenes
    _, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["l1"] == {"hit": 3, "miss": 0}
    chars = common.read_json(common.WORKS_DIR / f"{WID}.characters.json")
    chars[1]["aliases"].append("はなこ")
    common.write_json(common.WORKS_DIR / f"{WID}.characters.json", chars)
    annotate_all(env)
    _, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["l1"] == {"hit": 0, "miss": 3}


def test_l1_output_equals_uncached_output(env):
    annotate_all(env)
    a, _ = merge_mod.merge_work(WID, env["cache"], env["seg"], use_l1=False)
    merge_mod.merge_work(WID, env["cache"], env["seg"])
    b, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["l1"]["hit"] == 3 and common.dumps(a) == common.dumps(b)


# ---------------- robustness ----------------
def test_damaged_entries_are_misses_and_get_overwritten(env):
    annotate_all(env)
    merge_mod.merge_work(WID, env["cache"], env["seg"])
    root = env["cache"].root
    l2 = sorted((root / "l2").rglob("*.json"))
    l1 = sorted((root / "l1").rglob("*.json"))
    l2[0].write_text("{broken", encoding="utf-8")
    w = json.loads(l2[1].read_text(encoding="utf-8"))
    w["schema"] = "sensor.cache.l2/0"
    l2[1].write_text(json.dumps(w), encoding="utf-8")
    l1[0].write_text("", encoding="utf-8")
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert r["cached"] < 3 and r["packs_written"]  # no exception; damaged scenes recomputed
    annotate_all(env)
    _, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["l1"]["miss"] >= 1 and rep["merged"] == 3
    for p in (root / "l2").rglob("*.json"):
        json.loads(p.read_text(encoding="utf-8"))  # all valid again


def test_writes_are_atomic_no_temp_files_left(env):
    annotate_all(env)
    merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert not list(env["cache"].root.rglob("*.tmp"))


def test_gc_is_explicit_only(env):
    annotate_all(env)
    merge_mod.merge_work(WID, env["cache"], env["seg"])
    root = env["cache"].root
    old = time.time() - 40 * 86400
    victims = sorted((root / "l1").rglob("*.json"))[:2]
    for p in victims:
        os.utime(p, (old, old))
    merge_mod.merge_work(WID, env["cache"], env["seg"])  # prepare/merge/diagnose paths never delete
    prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"])
    assert all(p.exists() for p in victims)
    assert sorted(env["cache"].gc(30)) == sorted(victims) and all(p.exists() for p in victims)  # dry run
    env["cache"].gc(30, apply=True)
    assert not any(p.exists() for p in victims)


def test_legacy_entry_is_adopted_once_then_strict(env):
    for s in env["seg"]["scenes"]:
        env["cache"].put(good_entry(s, facts=[]))
    r = prep(env)
    assert r["cached"] == 3 and not (env["cache"].root / "strong").exists() or not list((env["cache"].root / "strong").glob("*.json"))
    led = env["cache"].ledger(WID)
    led["taro"] = [0, 0, 0, 0, 0, 1]
    env["cache"].put_ledger(WID, led)
    assert prep(env)["cached"] == 0  # adopted entries now obey the strict key


# ---------------- committed seed cache + warm/cold parity (need the nishiki binary) ----------------
def _nishiki():
    try:
        return common.find_nishiki()
    except FileNotFoundError:
        return None


needs_nishiki = pytest.mark.skipif(_nishiki() is None, reason="nishiki binary missing")


@needs_nishiki
def test_committed_seed_cache_is_fully_strict_keyed():
    nis, cache, total = _nishiki(), Cache(legacy_adopt=False), 0
    assert not list((cache.root / "strong").glob("*.json")) if (cache.root / "strong").exists() else True
    for w in common.all_work_ids():
        work = common.load_work(w)
        seg = common.segment(nis, work["text_abs"])
        res = resolve_work(w, cache, seg["scenes"], work["characters"])
        assert all(r.how == "hit" for r in res), (w, [r.how for r in res])
        total += len(res)
    assert total == 35


@needs_nishiki
@pytest.mark.parametrize("where", ["tail", "mid"])
@pytest.mark.parametrize("variant", ["same", "state_change"])
def test_warm_partial_recompute_equals_cold_full_recompute(where, variant):
    import parity

    res = parity.run_case(_nishiki(), "000035_000301", 3, where, variant)
    assert res["parity_annotations_bytes_equal"] and res["parity_diagnose_bytes_equal"]
    assert res["parity_cold_l1off_equal"] and res["unchanged_rerun_equal_warmup"] and res["unchanged_rerun_llm_calls"] == 0
    assert res["cold"]["llm_calls"] == res["scene_count"]
    expect = [3] if variant == "same" else [3, 4]  # state_change: scene 4 sees a new prior; scene 5 is cut off
    assert res["warm"]["l2_recomputed_scenes"] == expect and res["warm"]["llm_calls"] == len(expect)
    assert res["warm"]["l1"]["miss"] == (2 if where == "tail" else 1)
