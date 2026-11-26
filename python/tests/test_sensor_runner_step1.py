"""Phase 1.5 Stage 2 Step 1: pack-bytes L2 key, prior_state v2 (present + pov + optional location_id), facts not keyed
("premise changed" recorded instead), ledger epoch freeze, Stage 1 -> Step 1 key migration."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import common  # noqa: E402
import epoch as epoch_mod  # noqa: E402
import keys  # noqa: E402
import locations as locations_mod  # noqa: E402
import prepare as prepare_mod  # noqa: E402
import render  # noqa: E402
import resolve as resolve_mod  # noqa: E402
import strong  # noqa: E402
from cache import Cache  # noqa: E402
from test_sensor_runner import TEXTS, WID, do_ingest, env, fake_seg, good_entry  # noqa: E402,F401
from test_sensor_runner_phase15 import annotate_all  # noqa: E402


def chars_of(env):
    return common.read_json(common.WORKS_DIR / f"{WID}.characters.json")


def ctx_for(env, chars=None, ledger=None):
    work = common.load_work(WID)
    work["work_id"] = WID
    return render.work_context(work, chars if chars is not None else chars_of(env), ledger if ledger is not None else env["cache"].ledger_committed(WID), {})


def key_of(env, idx, prior=None, **ctx_kw):
    ctx = ctx_for(env, **ctx_kw)
    sc = env["seg"]["scenes"][idx]
    pack = render.build_pack(ctx, sc, len(env["seg"]["scenes"]), prior or {"status": "none"}, WID)
    return render.pack_key(pack, WID), pack


CACHED_PRIOR = {"status": "cached", "prev_scene_index": 0, "location": "駅", "pov_character": "taro", "present": ["taro"],
                "time": {"relative_order": "continuous"}, "recent_facts": [{"subject": "taro", "predicate": "MOVED_TO", "object": "駅"}],
                "trait_state": {"taro": [0, 0, 0, 0, 0, 0]}}


# ---------------- property: hidden vs visible characters ----------------
def test_hidden_character_ledger_change_does_not_change_key_visible_does(env):
    led = {**env["cache"].ledger_committed(WID), "ghost": [0, 1, 0, 0, 0, 0]}
    k0, p0 = key_of(env, 0, ledger=led)
    assert "ghost" not in p0["ledger_baseline"] and {c["id"] for c in p0["roster"]} == {"taro", "hanako"}
    assert key_of(env, 0, ledger={**led, "ghost": [1, 1, 1, 1, 1, 1]})[0] == k0  # hidden character
    assert key_of(env, 0, ledger={**led, "taro": [0, 0, 0, 0, 0, 1]})[0] != k0  # visible character
    assert key_of(env, 0, ledger={**led, "hanako": [1, 0, 0, -1, 0, 1]})[0] != k0
    # scene 1 shows only taro: hanako is hidden there
    k1, p1 = key_of(env, 1, prior=CACHED_PRIOR, ledger=led)
    assert "hanako" not in p1["ledger_baseline"]
    assert key_of(env, 1, prior=CACHED_PRIOR, ledger={**led, "hanako": [-1, 0, 0, 0, 0, 0]})[0] == k1


def test_roster_alias_and_attribute_changes_visible_vs_hidden(env):
    k0, _ = key_of(env, 0)
    chars = chars_of(env)

    def mod(i, f):
        c = copy.deepcopy(chars)
        f(c[i])
        return key_of(env, 0, chars=c)[0]

    assert mod(0, lambda c: c["aliases"].append("たろう")) != k0  # visible character, alias
    assert mod(0, lambda c: c.update(honored=True)) != k0  # visible character, other roster field
    assert mod(2, lambda c: c["aliases"].append("おばけ")) == k0  # hidden character, alias that does not match the text
    assert mod(2, lambda c: c["aliases"].append("駅")) != k0  # alias that makes the character visible in this scene
    c = copy.deepcopy(chars)
    c.append({"id": "newcomer", "aliases": ["新人"], "gender": "Male", "honored": False})
    assert key_of(env, 0, chars=c)[0] != k0  # roster_all_ids is shown to the annotator


def test_global_instruction_model_and_sampling_change_the_key(env, monkeypatch):
    k0, pack = key_of(env, 0)
    with monkeypatch.context() as m:
        m.setattr(strong, "INSTRUCTIONS", strong.INSTRUCTIONS + "追加の指示。")
        assert key_of(env, 0)[0] != k0
    assert key_of(env, 0)[0] == k0
    assert render.pack_key(pack, WID, model_id="m-other") != k0
    assert render.pack_key(pack, WID, sampling="temp0.7") != k0
    p2 = dict(pack, instruction_version="strong.v2")
    assert render.pack_key(p2, WID) != k0


# ---------------- property: any byte of the rendered pack ----------------
def test_one_byte_change_of_any_keyed_field_changes_the_key_and_excluded_fields_do_not(env):
    k0, pack = key_of(env, 1, prior=CACHED_PRIOR)
    view = render.pack_view(pack, WID)
    assert set(view) == set(pack) - set(render.EXCLUDED_FIELDS)
    for field in sorted(view):
        if field == "prior":
            continue
        p = copy.deepcopy(pack)
        v = p[field]
        if isinstance(v, str):
            p[field] = v + "x"
        elif isinstance(v, bool):
            p[field] = not v
        elif isinstance(v, int):
            p[field] = v + 1
        elif isinstance(v, list):
            p[field] = [*v, "x"]
        elif isinstance(v, dict):
            p[field] = {**v, "zz": 1}
        else:  # None
            p[field] = "x"
        assert render.pack_key(p, WID) != k0, field
    # a single character inside the scene text
    p = copy.deepcopy(pack)
    p["scene_text"] = p["scene_text"].replace("夜", "昼", 1)
    assert render.pack_key(p, WID) != k0
    for field in render.EXCLUDED_FIELDS:  # position / metadata
        p = copy.deepcopy(pack)
        p[field] = "changed"
        assert render.pack_key(p, WID) == k0, field


def test_pack_file_bytes_reproduce_the_stored_key(env):
    r = prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"])
    on_disk = json.loads(Path(r["packs_written"][0]["pack"]).read_text(encoding="utf-8"))
    assert render.pack_key(on_disk, WID) == on_disk["l2_key"]


# ---------------- prior core: present + pov (+ location_id) ----------------
def test_prior_core_is_present_and_pov_only_without_location_list(env):
    k, _ = key_of(env, 1, prior=CACHED_PRIOR)

    def with_prior(**over):
        return key_of(env, 1, prior={**CACHED_PRIOR, **over})[0]

    assert with_prior(present=["taro"], location="港") == k  # free-text location is not keyed
    assert with_prior(location=None) == k
    assert with_prior(time={"relative_order": "jump"}, trait_state={}) == k
    assert with_prior(recent_facts=[]) == k  # facts are not keyed
    assert with_prior(present=["hanako", "taro"]) != k
    assert with_prior(pov_character="hanako") != k
    assert with_prior(present=["taro", "hanako"]) == with_prior(present=["hanako", "taro"])  # set semantics


def test_location_id_is_keyed_only_for_works_with_a_hand_curated_list(env, tmp_path, monkeypatch):
    base, _ = key_of(env, 1, prior=CACHED_PRIOR)
    d = tmp_path / "loc"
    d.mkdir()
    (d / f"{WID}.json").write_text(json.dumps({
        "schema": "sensor.locations/1", "work_id": WID,
        "locations": [{"id": "L1", "label": "駅", "aliases": ["ステーション", "駅前"]}, {"id": "L2", "label": "港"}]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(locations_mod, "LOCATIONS_DIR", d)

    def k(loc):
        return key_of(env, 1, prior={**CACHED_PRIOR, "location": loc})[0]

    assert k("駅") == k("ステーション") == k("駅前") == k(" 駅。")  # surface variants of one listed place
    assert k("駅") != k("港")
    assert k(None) != k("駅") and k(None) == k("")  # "no location" is a legitimate value
    assert k("謎の場所") != k("駅") and k("謎の場所") != k(None)  # unlisted is distinct from none
    assert k("謎の場所") == k("別の謎の場所")  # unlisted places are not told apart
    assert k("駅") != base  # the curated list changes the key relative to a work without one


# ---------------- facts: not keyed, "premise changed" recorded ----------------
def test_facts_change_keeps_key_and_marks_premise_changed_until_explicit_reannotation(env):
    assert annotate_all(env) == 3
    cache = env["cache"]
    scenes = env["seg"]["scenes"]
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"])
    assert r["pending"] == 0 and r["premise_changed"] == []
    # explicitly re-annotate scene 0 with different facts (same present / pov)
    cache_scene0 = scenes[0]["scene_hash"]
    epoch_mod.reannotate(cache, WID, [cache_scene0])
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert [p["scene_index"] for p in r["packs_written"]] == [0] and r["forced"] == [0]
    new0 = good_entry(scenes[0], facts=[{"subject": "hanako", "predicate": "MOVED_TO", "object": "家", "evidence": "explicit", "basis": "花子"}])
    assert do_ingest(env, new0)["accepted"]
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert r["packs_written"] == [] and r["cached"] == 3 and r["forced"] == []  # scene 1 is still a hit: facts are not keyed
    assert r["premise_changed"] == [{"scene_index": 1, "scene_hash": scenes[1]["scene_hash"]}]
    st = cache.stale_state(WID)
    assert [x["scene_index"] for x in st["premise_changed"]] == [1]
    assert st["premise_changed"][0]["stored_prior_facts"] != st["premise_changed"][0]["current_prior_facts"]
    listed = epoch_mod.list_premise_changed(cache, WID, env["seg"]["scenes"], chars_of(env))
    assert [x["scene_index"] for x in listed] == [1]
    # nothing is re-annotated automatically; an explicit request emits exactly that pack
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert r["packs_written"] == []
    epoch_mod.reannotate(cache, WID, [scenes[1]["scene_hash"]])
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert [p["scene_index"] for p in r["packs_written"]] == [1]
    assert do_ingest(env, good_entry(scenes[1], facts=[]))["accepted"]
    r = prepare_mod.prepare_work(WID, cache, seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert r["pending"] == 0 and r["premise_changed"] == [] and r["forced"] == []


def test_premise_check_does_not_touch_the_diagnose_input_schema(env):
    """The flag lives only in cache/stale and in the prepare/merge reports; merged sensor.v2 records are unchanged."""
    import merge as merge_mod

    annotate_all(env)
    before, rep0 = merge_mod.merge_work(WID, env["cache"], env["seg"])
    epoch_mod.reannotate(env["cache"], WID, [env["seg"]["scenes"][0]["scene_hash"]])
    prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"], defer_blocked=True)
    assert do_ingest(env, good_entry(env["seg"]["scenes"][0], facts=[]))["accepted"]
    after, rep = merge_mod.merge_work(WID, env["cache"], env["seg"])
    assert rep["premise_changed"] and {k for r in after for k in r} == {k for r in before for k in r}
    assert not any("premise" in json.dumps(r) for r in after)


# ---------------- ledger epoch freeze ----------------
def _texts_with_ghost_late():
    return [TEXTS[0], TEXTS[1], "幽霊が現れた。太郎は驚いた。", "朝になった。太郎は去った。"]


def test_ledger_additions_are_pending_until_epoch_commit_and_do_not_recompute_earlier_scenes(env):
    seg = fake_seg(_texts_with_ghost_late())
    env["seg"] = seg
    cache = env["cache"]
    calls = 0
    while True:
        r = prepare_mod.prepare_work(WID, cache, seg=seg, packs_dir=env["packs"], defer_blocked=True)
        if not r["packs_written"]:
            break
        for p in r["packs_written"]:
            s = seg["scenes"][p["scene_index"]]
            e = good_entry(s, facts=[])
            if p["scene_index"] == 2:  # the annotator of scene 2 introduces a new character
                e["ledger_additions"] = {"ghost": [0, 1, 0, 0, 0, 0]}
                e["trait_changes"].append({"character_id": "ghost", "evidence": "inferred", "changes": []})
                e["present"].append({"id": "ghost", "basis": "幽霊"}) if False else None
            res = do_ingest(env, e)
            assert res["accepted"], res["reasons"]
            calls += 1
    assert calls == 4  # epoch 1: every scene annotated exactly once
    assert "ghost" not in cache.ledger_committed(WID) and "ghost" in cache.ledger(WID)  # frozen vs effective
    assert list(cache.pending_additions(WID)) == ["ghost"]
    # epoch 2: only the scene where ghost is visible misses
    st = epoch_mod.commit_epoch(cache, WID)
    assert st["status"] == "committed" and st["epoch"] == 1 and st["applied"] == ["ghost"]
    assert cache.ledger_committed(WID)["ghost"] == [0, 1, 0, 0, 0, 0] and cache.pending_additions(WID) == {}
    r = prepare_mod.prepare_work(WID, cache, seg=seg, packs_dir=env["packs"], defer_blocked=True)
    assert [p["scene_index"] for p in r["packs_written"]] == [2]  # scenes 0, 1 (ghost hidden) stay hits; 3 is cut off by the prior core
    pack = json.loads(Path(r["packs_written"][0]["pack"]).read_text(encoding="utf-8"))
    assert pack["ledger_baseline"]["ghost"] == [0, 1, 0, 0, 0, 0]
    assert do_ingest(env, good_entry(seg["scenes"][2], facts=[]))["accepted"]  # annotator now sees the baseline: no new additions
    r = prepare_mod.prepare_work(WID, cache, seg=seg, packs_dir=env["packs"], defer_blocked=True)
    assert r["pending"] == 0 and r["packs_written"] == []
    assert epoch_mod.commit_epoch(cache, WID)["status"] == "converged"
    assert cache.epoch_state(WID)["epoch"] == 0  # run finished, counter reset


def test_epoch_limit_cuts_off_and_reports_without_looping(env):
    cache = env["cache"]
    for i, name in enumerate(["a1", "a2", "a3"]):
        cache.add_pending(WID, {name: [0, 0, 0, 0, 0, i]}, f"h{i}")
        r = epoch_mod.commit_epoch(cache, WID)
        if i < 2:
            assert r["status"] == "committed" and r["epoch"] == i + 1
        else:
            assert r["status"] == "cutoff" and r["unapplied"] == ["a3"] and r["epoch"] == 2
    assert "a3" not in cache.ledger_committed(WID) and "a3" in cache.pending_additions(WID)
    epoch_mod.reset_epochs(cache, WID)
    assert epoch_mod.commit_epoch(cache, WID)["status"] == "committed"  # explicit new run applies it


def test_pending_additions_first_writer_wins_and_never_overwrite_committed(env):
    cache = env["cache"]
    assert cache.add_pending(WID, {"ghost": [1, 0, 0, 0, 0, 0], "taro": [1, 1, 1, 1, 1, 1]}, "h1") == ["ghost"]
    assert cache.add_pending(WID, {"ghost": [0, 0, 0, 0, 0, 1]}, "h2") == []
    assert cache.ledger(WID)["ghost"] == [1, 0, 0, 0, 0, 0] and cache.ledger(WID)["taro"] == [0] * 6


# ---------------- Stage 1 entries ----------------
def test_stage1_entry_is_adopted_under_the_new_key_when_its_old_key_matches(env):
    cache = env["cache"]
    seg = env["seg"]
    s0 = seg["scenes"][0]
    old_key = keys.l2_key(keys.scene_text_hash(s0["scene_text"]), keys.roster_hash(chars_of(env)),
                          keys.ledger_key_hash(cache.ledger_committed(WID)), keys.prior_state_hash(None))
    entry = good_entry(s0, facts=[])
    d = cache.root / "l2" / s0["scene_hash"]
    d.mkdir(parents=True)
    (d / f"{old_key}.json").write_text(json.dumps({"schema": keys.L2_SCHEMA_V1, "key": old_key, "scene_hash": s0["scene_hash"], "parts": {},
                                                  "source": "ingest", "entry": entry}), encoding="utf-8")
    ro = Cache(cache.root, legacy_adopt=False)
    assert resolve_mod.resolve_work(WID, ro, seg["scenes"], chars_of(env))[0].how == "miss"  # never read as a hit
    r = resolve_mod.resolve_work(WID, cache, seg["scenes"], chars_of(env))[0]
    assert r.how == "adopted" and r.entry == entry
    again = resolve_mod.resolve_work(WID, ro, seg["scenes"], chars_of(env))[0]
    assert again.how == "hit" and again.premise_changed is None  # history unknown -> not flagged
    w = ro.get_l2_wrapper(s0["scene_hash"], again.key)
    assert w["source"] == "migrated-v1" and w["parts"]["prior_facts_digest"] is None
    # a changed ledger (visible character) makes the old key mismatch: the Stage 1 entry is not adopted
    led = cache.ledger_committed(WID)
    cache.put_ledger(WID, {**led, "taro": [0, 0, 0, 0, 0, 1]})
    cache.delete(s0["scene_hash"])
    d.mkdir(parents=True)
    (d / f"{old_key}.json").write_text(json.dumps({"schema": keys.L2_SCHEMA_V1, "key": old_key, "scene_hash": s0["scene_hash"], "parts": {},
                                                  "source": "ingest", "entry": entry}), encoding="utf-8")
    assert resolve_mod.resolve_work(WID, cache, seg["scenes"], chars_of(env))[0].how == "miss"
