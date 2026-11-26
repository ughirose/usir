"""Phase 1.5 Stage 2 Step 1.5: roster_all_ids is shown to the annotator but is not part of the L2 key.
A change of the full character list is recorded at hit time (reason roster_changed) instead of invalidating every scene."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import common  # noqa: E402
import epoch as epoch_mod  # noqa: E402
import keys  # noqa: E402
import prepare as prepare_mod  # noqa: E402
import render  # noqa: E402
import resolve as resolve_mod  # noqa: E402
from cache import Cache  # noqa: E402
from test_sensor_runner import WID, do_ingest, env, good_entry  # noqa: E402,F401
from test_sensor_runner_phase15 import annotate_all  # noqa: E402
from test_sensor_runner_step1 import chars_of, key_of  # noqa: E402


def write_chars(env, chars):
    (common.WORKS_DIR / f"{WID}.characters.json").write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")


def keys_of(env, chars):
    """Resolver keys of all scenes with the given character file (scenes are all annotated, so every key is the one that is looked up)."""
    return [r for r in resolve_mod.resolve_work(WID, env["cache"], env["seg"]["scenes"], chars)]


def newcomer(alias, cid="newcomer"):
    return {"id": cid, "aliases": [alias], "gender": "Male", "honored": False}


def test_pack_still_shows_roster_all_ids_and_pack_content_is_unchanged_by_the_key_change(env):
    _, pack = key_of(env, 0)
    assert pack["roster_all_ids"] == ["taro", "hanako", "ghost"]  # displayed to the annotator, order as in the character file
    assert "roster_all_ids" not in render.pack_view(pack, WID)
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"])
    on_disk = json.loads(Path(r["packs_written"][0]["pack"]).read_text(encoding="utf-8"))
    assert on_disk["roster_all_ids"] == ["taro", "hanako", "ghost"]
    assert render.pack_key(on_disk, WID) == on_disk["l2_key"]


def test_adding_a_character_who_appears_in_no_scene_keeps_every_key_and_records_roster_changed(env):
    assert annotate_all(env) == 3
    cache, chars = env["cache"], chars_of(env)
    before = keys_of(env, chars)
    assert [r.how for r in before] == ["hit"] * 3 and all(r.roster_changed is None for r in before)
    write_chars(env, [*chars, newcomer("誰にも現れない名前")])
    after = keys_of(env, chars_of(env))
    assert [r.key for r in after] == [r.key for r in before]  # keys unchanged
    assert [r.how for r in after] == ["hit"] * 3  # all still cache hits
    assert [r.roster_changed["scene_index"] for r in after] == [0, 1, 2]
    assert all(r.roster_changed["reason"] == "roster_changed" for r in after)
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert r["packs_written"] == [] and r["pending"] == 0  # nothing is re-annotated automatically
    assert [x["scene_index"] for x in r["roster_changed"]] == [0, 1, 2]
    st = cache.stale_state(WID)
    assert [x["scene_index"] for x in st["roster_changed"]] == [0, 1, 2] and st["premise_changed"] == []
    assert "newcomer" in st["roster_changed"][0]["current_roster_all_ids"]


def test_adding_a_character_who_appears_in_one_scene_changes_only_that_scene_key(env):
    assert annotate_all(env) == 3
    chars = chars_of(env)
    before = keys_of(env, chars)
    write_chars(env, [*chars, newcomer("店", "shopkeeper")])  # "店" occurs only in scene 2
    after = keys_of(env, chars_of(env))
    assert after[0].key == before[0].key and after[1].key == before[1].key and after[2].key != before[2].key
    assert [r.how for r in after] == ["hit", "hit", "miss"]
    pack2 = after[2].pack
    assert "shopkeeper" in {c["id"] for c in pack2["roster"]} and "shopkeeper" in pack2["roster_all_ids"]
    # scene 0 and 1 are hits with roster_changed recorded; scene 2 is a plain miss (new key, no stored entry)
    assert [r.roster_changed is not None for r in after] == [True, True, False]


def test_removing_a_character_keeps_keys_of_scenes_without_that_character_and_records_roster_changed(env):
    assert annotate_all(env) == 3
    chars = chars_of(env)
    before = keys_of(env, chars)
    write_chars(env, [c for c in chars if c["id"] != "ghost"])  # ghost appears in no scene text
    after = keys_of(env, chars_of(env))
    assert [r.key for r in after] == [r.key for r in before] and [r.how for r in after] == ["hit"] * 3
    assert all(r.roster_changed is not None for r in after)
    # removing a character who is visible in a scene (hanako: scenes 0 and 2) changes exactly those keys
    # (scene 0 then misses, so later scenes are blocked in the resolver; compare the rendered pack keys directly)
    no_hanako = [c for c in chars if c["id"] != "hanako"]
    prior = {"status": "cached", "prev_scene_index": 0, "location": None, "pov_character": "taro", "present": ["taro"], "time": {}, "recent_facts": [], "trait_state": {}}
    assert key_of(env, 0, chars=no_hanako)[0] != before[0].key  # hanako appears in scene 0
    assert key_of(env, 1, prior=prior, chars=no_hanako)[0] == key_of(env, 1, prior=prior, chars=chars)[0]  # not in scene 1


def test_character_list_order_alone_does_not_change_keys_or_record_anything(env):
    assert annotate_all(env) == 3
    chars = chars_of(env)
    before = keys_of(env, chars)
    write_chars(env, list(reversed(chars)))
    after = keys_of(env, chars_of(env))
    assert [r.key for r in after] == [r.key for r in before] and [r.how for r in after] == ["hit"] * 3
    assert all(r.roster_changed is None for r in after)  # the digest is of the id set
    k, p = key_of(env, 0, chars=list(reversed(chars)))
    assert k == before[0].key and p["roster_all_ids"] == ["ghost", "hanako", "taro"]  # the annotator still sees the new order


def test_reannotation_of_roster_changed_scenes_is_explicit_and_clears_the_record(env):
    assert annotate_all(env) == 3
    cache, chars, scenes = env["cache"], chars_of(env), env["seg"]["scenes"]
    write_chars(env, [*chars, newcomer("誰にも現れない名前")])
    rows = epoch_mod.list_roster_changed(cache, WID, scenes, chars_of(env))
    assert [x["scene_index"] for x in rows] == [0, 1, 2]
    epoch_mod.reannotate(cache, WID, [scenes[1]["scene_hash"]])
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert [p["scene_index"] for p in r["packs_written"]] == [1]
    pack = json.loads(Path(r["packs_written"][0]["pack"]).read_text(encoding="utf-8"))
    assert "newcomer" in pack["roster_all_ids"]  # the re-annotation sees the current list
    assert do_ingest(env, good_entry(scenes[1], facts=[]))["accepted"]
    rows = epoch_mod.list_roster_changed(cache, WID, scenes, chars_of(env))
    assert [x["scene_index"] for x in rows] == [0, 2]  # scene 1 now carries the current digest


def test_new_entries_store_the_roster_digest_and_diagnose_inputs_have_no_flag(env):
    import merge as merge_mod

    annotate_all(env)
    w = env["cache"].get_l2_wrapper(env["seg"]["scenes"][0]["scene_hash"], keys_of(env, chars_of(env))[0].key)
    assert w["schema"] == keys.L2_SCHEMA == "sensor.cache.l2/3"
    assert w["parts"]["roster_all_digest"] == render.roster_all_digest({"roster_all_ids": ["hanako", "ghost", "taro"]})  # set semantics
    assert w["parts"]["key_schema"] == render.KEY_SCHEMA == "pack_bytes/2"
    write_chars(env, [*chars_of(env), newcomer("誰にも現れない名前")])
    out, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["roster_changed"] and not any("roster_changed" in json.dumps(r) for r in out)


def test_step1_entry_is_adopted_when_its_pack_bytes_1_key_matches_and_not_otherwise(env):
    """l2/2 entries (Step 1: roster_all_ids inside the key) migrate only if the recomputed old key matches."""
    cache = env["cache"]
    chars = chars_of(env)
    ctx = resolve_mod.render.work_context({**common.load_work(WID), "work_id": WID}, chars, cache.ledger_committed(WID), {})
    s0 = env["seg"]["scenes"][0]
    pack = render.build_pack(ctx, s0, 3, {"status": "none"}, WID)
    old_key = render.pack_key(pack, WID, schema=render.KEY_SCHEMA_V1)
    assert old_key != render.pack_key(pack, WID)
    entry = good_entry(s0, facts=[])
    facts_digest = "f" * 64
    d = cache.root / "l2" / s0["scene_hash"]
    d.mkdir(parents=True)
    (d / f"{old_key}.json").write_text(json.dumps({"schema": keys.L2_SCHEMA_V2, "key": old_key, "scene_hash": s0["scene_hash"],
                                                  "parts": {"prior_facts_digest": facts_digest}, "source": "ingest", "prior_facts": [{"x": 1}],
                                                  "entry": entry}), encoding="utf-8")
    ro = Cache(cache.root, legacy_adopt=False)
    assert resolve_mod.resolve_work(WID, ro, env["seg"]["scenes"], chars)[0].how == "miss"  # never read as a hit
    r = resolve_mod.resolve_work(WID, cache, env["seg"]["scenes"], chars)[0]
    assert r.how == "adopted" and r.entry == entry
    again = resolve_mod.resolve_work(WID, ro, env["seg"]["scenes"], chars)[0]
    w = ro.get_l2_wrapper(s0["scene_hash"], again.key)
    assert again.how == "hit" and w["source"] == "migrated-v2" and w["schema"] == keys.L2_SCHEMA
    assert w["parts"]["prior_facts_digest"] == facts_digest and w["prior_facts"] == [{"x": 1}]  # history kept
    assert w["parts"]["roster_all_digest"] == render.roster_all_digest(pack) and again.roster_changed is None
    # if the character list had changed since (old key no longer matches), nothing is adopted
    cache.delete(s0["scene_hash"])
    d.mkdir(parents=True)
    (d / f"{old_key}.json").write_text(json.dumps({"schema": keys.L2_SCHEMA_V2, "key": old_key, "scene_hash": s0["scene_hash"],
                                                  "parts": {}, "source": "ingest", "entry": entry}), encoding="utf-8")
    write_chars(env, [*chars, newcomer("誰にも現れない名前")])
    assert resolve_mod.resolve_work(WID, cache, env["seg"]["scenes"], chars_of(env))[0].how == "miss"
