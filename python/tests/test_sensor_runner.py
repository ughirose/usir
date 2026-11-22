"""Tests for scripts/sensor_runner (prepare / ingest / merge / cache). Unit tests use a fake `nishiki segment` result;
the integration tests need the nishiki binary (NISHIKI_BIN or the default path) and the committed seed cache."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import common  # noqa: E402
import ingest as ingest_mod  # noqa: E402
import merge as merge_mod  # noqa: E402
import prepare as prepare_mod  # noqa: E402
from cache import Cache  # noqa: E402
from strong import STATE_CATEGORIES, validate_strong  # noqa: E402

WID = "tw"
TEXTS = [
    "太郎は東京の駅に着いた。花子が笑った。「こんにちは。」と太郎は言った。",
    "夜になって雨が降り出した。太郎は黙って傘を差した。",
    "翌朝、花子は店を失った。太郎は驚いた。",
]


def fake_seg(texts: list[str]) -> dict:
    scenes = []
    for i, t in enumerate(texts):
        sents, pos, k = [], 0, 0
        for part in t.split("。"):
            if not part:
                continue
            end = pos + len(part) + 1
            sents.append({"index": k, "local": {"start": pos, "end": end}})
            pos, k = end, k + 1
        utts = []
        q = t.find("「")
        if q >= 0:
            utts.append({"index": 0, "local": {"start": q + 1, "end": t.find("」")}})
        scenes.append({
            "index": i, "span": {"start": 0, "end": len(t)}, "char_len": len(t),
            "scene_hash": hashlib.sha256(t.encode()).hexdigest(), "scene_text": t,
            "sentences": sents, "utterances": utts, "numbered_text": "".join(f"[S{s['index']}]x" for s in sents),
        })
    return {"schema": "nishiki.segment/1", "splitter": "usir_v5_splitter@1", "scenes": scenes}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    works = tmp_path / "works"
    works.mkdir()
    (works / f"{WID}.work.json").write_text(json.dumps({"work_id": WID, "text_path": "x.txt", "narration": "third", "narrator": None, "pov": None}), encoding="utf-8")
    chars = [{"id": "taro", "aliases": ["太郎"], "gender": "Male", "honored": False},
             {"id": "hanako", "aliases": ["花子"], "gender": "Female", "honored": False},
             {"id": "ghost", "aliases": ["幽霊"], "gender": "Male", "honored": False}]
    (works / f"{WID}.characters.json").write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(common, "WORKS_DIR", works)
    monkeypatch.setattr(prepare_mod, "LOCAL_DIR", tmp_path / "local")
    cache = Cache(tmp_path / "cache")
    cache.put_ledger(WID, {"taro": [0, 0, 0, 0, 0, 0], "hanako": [1, 0, 0, -1, 0, 0]})
    return {"cache": cache, "packs": tmp_path / "packs", "rejects": tmp_path / "rejects", "seg": fake_seg(TEXTS), "tmp": tmp_path}


def good_entry(seg_scene: dict, **over) -> dict:
    t = seg_scene["scene_text"]
    e = {
        "schema": "sensor.strong/1", "instruction_version": common.INSTRUCTION_VERSION, "scene_hash": seg_scene["scene_hash"],
        "annotator": "test", "provenance": "session",
        "time": {"relative_order": "continuous", "explicit_time": None, "elapsed_hint": None, "evidence": "inferred"},
        "location": None, "pov_character": "taro",
        "present": [{"id": "taro", "basis": "太郎"}],
        "ledger_additions": {},
        "trait_changes": [{"character_id": "taro", "evidence": "inferred", "changes": [{"axis": 3, "delta": 1, "basis": "太郎"}]}],
        "facts": [{"subject": "taro", "predicate": "MOVED_TO", "object": "駅", "evidence": "explicit", "basis": "駅に着いた"}],
        "event_top3": [{"code": "E02", "basis": "太郎"}, {"code": "E03", "basis": "太郎"}, {"code": "E10", "basis": "太郎"}],
        "mood_top3": [{"code": "M05", "basis": "太郎"}, {"code": "M12", "basis": "太郎"}, {"code": "M10", "basis": "太郎"}],
        "awase": {"F_valence": 0, "F_arousal": 1, "f_valence": 0, "f_arousal": -1, "F_basis": "太郎", "f_basis": "太郎"},
    }
    assert "太郎" in t
    e.update(over)
    return e


def prep(env, **kw):
    return prepare_mod.prepare_work(WID, env["cache"], seg=env["seg"], packs_dir=env["packs"], **kw)


def test_prepare_all_pending_then_none_after_cached(env):
    r = prep(env)
    assert (r["cached"], r["pending"], len(r["packs_written"])) == (0, 3, 3)
    for s in env["seg"]["scenes"]:
        env["cache"].put(good_entry(s))
    r = prep(env)
    assert (r["cached"], r["pending"], len(r["packs_written"])) == (3, 0, 0)
    assert list((env["packs"] / WID).glob("*.pack.json")) == []  # stale packs removed


def test_prepare_only_missing_scene_after_removal(env):
    for s in env["seg"]["scenes"]:
        env["cache"].put(good_entry(s))
    victim = env["seg"]["scenes"][1]["scene_hash"]
    assert env["cache"].delete(victim)
    r = prep(env)
    assert [p["scene_index"] for p in r["packs_written"]] == [1]
    assert [p.name for p in (env["packs"] / WID).glob("*.pack.json")] == [victim[:12] + ".pack.json"]


def test_prepare_changed_scene_only(env):
    for s in env["seg"]["scenes"]:
        env["cache"].put(good_entry(s))
    texts = list(TEXTS)
    texts[2] = texts[2] + "花子は泣いた。"
    env["seg"] = fake_seg(texts)
    r = prep(env)
    assert [p["scene_index"] for p in r["packs_written"]] == [2]


def test_prepare_limit_and_pack_content(env):
    r = prep(env, limit=2)
    assert len(r["packs_written"]) == 2 and r["pending"] == 3
    pack = json.loads(Path(r["packs_written"][0]["pack"]).read_text(encoding="utf-8"))
    assert pack["scene_text"] == TEXTS[0] and pack["roster_all_ids"] == ["taro", "hanako", "ghost"]
    assert {c["id"] for c in pack["roster"]} == {"taro", "hanako"}  # ghost not mentioned
    assert pack["prior"]["status"] == "none" and pack["utterance_count"] == 1
    pack2 = json.loads(Path(r["packs_written"][1]["pack"]).read_text(encoding="utf-8"))
    assert pack2["prior"]["status"] == "pending"


def test_prior_summary_uses_cached_previous_scene(env):
    env["cache"].put(good_entry(env["seg"]["scenes"][0], location=None))
    r = prep(env)
    pack = json.loads(Path(r["packs_written"][0]["pack"]).read_text(encoding="utf-8"))
    assert pack["scene_index"] == 1 and pack["prior"]["status"] == "cached"
    assert pack["prior"]["trait_state"]["taro"] == [0, 0, 0, 1, 0, 0]
    assert pack["prior"]["present"] == ["taro"]


def write_out(env, entry, name="a.strong.json"):
    p = env["tmp"] / "out" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return p


def do_ingest(env, entry):
    return ingest_mod.ingest_file(write_out(env, entry), env["cache"], env["packs"], env["rejects"])


def test_ingest_accepts_valid_and_updates_cache_and_manifest(env):
    prep(env)
    s = env["seg"]["scenes"][0]
    r = do_ingest(env, good_entry(s))
    assert r["accepted"], r["reasons"]
    assert env["cache"].has(s["scene_hash"])
    m = env["cache"].manifest(WID)
    assert m["cached_count"] == 1 and m["scenes"][0]["status"] == "cached"


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e["facts"][0].update(basis="存在しない語句"), "facts[0]: basis not verbatim"),
    (lambda e: e["present"][0].update(id="nobody"), "not in roster"),
    (lambda e: e["present"][0].update(basis=None), "present[0]: basis"),
    (lambda e: e["event_top3"][1].update(code="E02"), "duplicate codes"),
    (lambda e: e["event_top3"][0].update(code="E99"), "bad code"),
    (lambda e: e["mood_top3"].pop(), "need exactly 3"),
    (lambda e: e["awase"].update(F_valence=3), "awase.F_valence"),
    (lambda e: e["awase"].update(f_basis="言い換え"), "awase.f_basis"),
    (lambda e: e["time"].update(relative_order="later"), "relative_order"),
    (lambda e: e["time"].update(explicit_time="三月"), "time.explicit_time"),
    (lambda e: e.update(location="どこか"), "location"),
    (lambda e: e["facts"][0].update(predicate="DIED"), "predicate"),
    (lambda e: e["facts"].append({"subject": None, "predicate": "STATE_CHANGED", "target": "太郎", "category": "mood", "text": "x", "evidence": "explicit", "basis": "太郎"}), "category"),
    (lambda e: e["facts"].append({"subject": None, "predicate": "STATE_CHANGED", "target": "太郎", "category": "unclassified", "text": "x", "evidence": "explicit", "basis": "太郎"}), "category"),
    (lambda e: e["trait_changes"][0]["changes"][0].update(delta=0), "delta"),
    (lambda e: e["trait_changes"][0]["changes"][0].update(axis=6), "axis"),
    (lambda e: e["trait_changes"].append({"character_id": "ghost", "evidence": "inferred", "changes": []}), "no ledger baseline"),
    (lambda e: e.update(ledger_additions={"taro": [0, 0, 0, 0, 0, 1]}), "already exists"),
    (lambda e: e.update(instruction_version="strong.v0"), "instruction_version"),
    (lambda e: e.update(scene_hash="0" * 64), "scene_hash"),
])
def test_ingest_rejects_with_reason(env, mutate, needle):
    prep(env)
    s = env["seg"]["scenes"][0]
    e = good_entry(s)
    mutate(e)
    if "scene_hash" in e and e["scene_hash"] == "0" * 64:
        r = do_ingest(env, e)
        assert not r["accepted"] and "no_pack_for_scene_hash" in r["reasons"][0]
        return
    r = do_ingest(env, e)
    assert not r["accepted"], "expected rejection"
    assert any(needle in x for x in r["reasons"]), r["reasons"]
    assert not env["cache"].has(s["scene_hash"])  # nothing stored
    assert list(env["rejects"].glob("*.reject.json"))


def test_ingest_rejects_bad_json(env):
    prep(env)
    p = env["tmp"] / "bad.strong.json"
    p.write_text("{not json", encoding="utf-8")
    r = ingest_mod.ingest_file(p, env["cache"], env["packs"], env["rejects"])
    assert not r["accepted"] and r["reasons"][0].startswith("json_parse_error")


def test_ledger_additions_extend_ledger_without_overwrite(env):
    prep(env)
    s = env["seg"]["scenes"][0]
    e = good_entry(s, ledger_additions={"ghost": [0, 1, 0, 0, 0, 0]})
    e["trait_changes"].append({"character_id": "ghost", "evidence": "inferred", "changes": []})
    r = do_ingest(env, e)
    assert r["accepted"], r["reasons"]
    assert env["cache"].ledger(WID)["ghost"] == [0, 1, 0, 0, 0, 0]
    assert env["cache"].ledger(WID)["taro"] == [0, 0, 0, 0, 0, 0]


def test_seed_provenance_allows_missing_basis_but_session_does_not(env):
    s = env["seg"]["scenes"][0]
    e = good_entry(s, provenance="seed:test")
    e["facts"][0]["basis"] = None
    assert validate_strong(e, s["scene_text"], ["taro", "hanako", "ghost"], {"taro": [0] * 6}, s["scene_hash"]) == []
    e["provenance"] = "session"
    assert validate_strong(e, s["scene_text"], ["taro", "hanako", "ghost"], {"taro": [0] * 6}, s["scene_hash"])


def test_merge_scene_fields(env):
    s = env["seg"]["scenes"][0]
    st = good_entry(
        s,
        trait_changes=[{"character_id": "hanako", "evidence": "explicit", "changes": [{"axis": 0, "delta": 2, "basis": "花子"}, {"axis": 3, "delta": 1, "basis": "花子"}]},
                       {"character_id": "taro", "evidence": "inferred", "changes": []}],
        facts=[{"subject": "taro", "predicate": "MOVED_TO", "object": "駅", "evidence": "explicit", "basis": "駅に着いた"},
               {"subject": "hanako", "predicate": "STATE_CHANGED", "target": "花子", "category": "emotion", "text": "笑った", "evidence": "explicit", "basis": "花子が笑った"}],
    )
    local = [{"utterance_index": 0, "speaker_id": "taro", "addressee_id": "hanako", "evidence": "explicit"}]
    ledger = env["cache"].ledger(WID)
    rec, stats = merge_mod.merge_scene(WID, s, 3, st, local, ledger, ["taro", "hanako", "ghost"])
    assert stats["errors"] == []
    # ledger hanako [1,0,0,-1,0,0] + variations -> axis0: 1+2 clamped to 1, axis3: -1+1 = 0
    assert rec["character_states"][0]["values_6d"] == [1, 0, 0, 0, 0, 0]
    assert rec["character_states"][1]["values_6d"] == [0] * 6
    assert rec["fact_updates"][1]["object"] == "[emotion] 花子: 笑った"
    assert rec["fact_updates"][0]["sentence_index"] == 0 and rec["fact_updates"][1]["sentence_index"] == 1
    u = rec["utterances"][0]
    assert (u["speaker_id"], u["addressee_id"], u["evidence"]) == ("taro", "hanako", "explicit")
    assert u["register"]["politeness"] in ("plain", "polite", "unknown") and u["register"]["first_person"] is None
    # hanako is addressee but not in strong.present: counted, not added
    assert rec["scene_meta"]["present_character_ids"] == ["taro"] and stats["local_ids_absent_from_present"] == 1
    assert rec["scene_meta"]["nagare"]["event_top3"][0] == {"code": "E02", "rank_weight": 3}
    assert rec["scene_meta"]["awase"]["f_style_arousal"] == -1
    assert rec["contract"]["model_id"] == "test" and rec["contract"]["scene_id"] == f"{WID}:s0"


def test_merge_flags_missing_local_utterances(env):
    s = env["seg"]["scenes"][0]
    rec, stats = merge_mod.merge_scene(WID, s, 3, good_entry(s), None, env["cache"].ledger(WID), ["taro", "hanako", "ghost"])
    assert stats["missing_local_utterances"] == 1
    assert rec["utterances"][0]["speaker_id"] is None and rec["utterances"][0]["evidence"] == "ambiguous"


def test_state_changed_object_unclassified_is_plain_text():
    assert merge_mod.state_changed_object({"category": "unclassified", "target": None, "text": "ただの文"}) == "ただの文"


def test_merge_work_requires_all_cached(env):
    for s in env["seg"]["scenes"][:2]:
        env["cache"].put(good_entry(s))
    with pytest.raises(RuntimeError, match="uncached scenes"):
        merge_mod.merge_work(WID, env["cache"], env["seg"])
    recs, rep = merge_mod.merge_work(WID, env["cache"], env["seg"], partial=True)
    assert rep["merged"] == 2 and rep["uncached"] == [2] and len(recs) == 2


def test_state_categories_closed_set():
    assert "other" in STATE_CATEGORIES and "unclassified" not in STATE_CATEGORIES


# ---------------- integration (seed cache + nishiki binary) ----------------
def _nishiki():
    try:
        return common.find_nishiki()
    except FileNotFoundError:
        return None


needs_env = pytest.mark.skipif(_nishiki() is None or not (common.CACHE_ROOT / "manifests").exists(), reason="nishiki binary or seed cache missing")


@needs_env
def test_seed_all_scenes_cached_in_manifests():
    # the expected total is counted from the manifests' scene rows (it grows with every live run), not a fixed number
    total = cached = expected = 0
    for w in common.all_work_ids():
        m = Cache().manifest(w)
        assert m["scene_count"] == len(m["scenes"]), w
        expected += len(m["scenes"])
        total += m["scene_count"]
        cached += m["cached_count"]
        assert all(r["status"] == "cached" for r in m["scenes"]), w
        assert all(Cache().has(r["scene_hash"]) for r in m["scenes"])
    assert expected > 0
    assert (total, cached) == (expected, expected)


@needs_env
def test_seed_prepare_delta_and_diagnose_all_lenses_ok(tmp_path):
    from diagnose import all_ok, run_diagnose

    nis = _nishiki()
    cache_copy = tmp_path / "cache"
    shutil.copytree(common.CACHE_ROOT, cache_copy)
    cache = Cache(cache_copy)
    w = "000035_000301"
    work = common.load_work(w)
    seg = common.segment(nis, work["text_abs"])
    # nothing pending on the seeded cache
    r = prepare_mod.prepare_work(w, cache, seg=seg, packs_dir=tmp_path / "p")
    assert r["pending"] == 0 and r["packs_written"] == []
    # remove one scene -> exactly that scene's pack
    victim = seg["scenes"][3]["scene_hash"]
    cache.delete(victim)
    r = prepare_mod.prepare_work(w, cache, seg=seg, packs_dir=tmp_path / "p")
    assert [p["scene_index"] for p in r["packs_written"]] == [3]
    # diagnose of the merged seed: 6 lenses ok for every work
    seed = Cache()
    for wid in common.all_work_ids():
        wk = common.load_work(wid)
        sg = common.segment(nis, wk["text_abs"])
        recs, rep = merge_mod.merge_work(wid, seed, sg)
        assert rep["errors"] == []
        (tmp_path / f"{wid}.annotations.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        (tmp_path / f"{wid}.characters.json").write_text(json.dumps(wk["characters"], ensure_ascii=False), encoding="utf-8")
        d = run_diagnose(nis, wk["text_abs"], tmp_path / f"{wid}.characters.json", tmp_path / f"{wid}.annotations.json")
        assert all_ok(d), (wid, d["lenses"])


def test_nogold_warns_when_stage1_returns_all_speakers_null():
    from local_from_step1_nogold import all_null_scene_warnings

    scenes = [(0, "h0", 2), (1, "h1", 3), (2, "h2", 0), (3, "h3", 1)]
    stage1 = [
        {"utterance_index": 0, "speaker_id": "a"}, {"utterance_index": 1, "speaker_id": None},  # s0: one speaker -> no warning
        {"utterance_index": 2, "speaker_id": None}, {"utterance_index": 3, "speaker_id": None},  # s1: all null (index 4 missing)
    ]  # s2 has no utterances; s3 (index 5) has no prediction at all -> counts as null
    final = [{"utterance_index": 2, "speaker_id": "a"}]
    w = all_null_scene_warnings(scenes, stage1, final)
    assert [(x["scene_index"], x["utterances"], x["final_speaker_null"]) for x in w] == [(1, 3, 2), (3, 1, 1)]
    assert all(x["warning"] == "stage1_all_speakers_null" for x in w)
    assert all_null_scene_warnings(scenes[:1], stage1) == []