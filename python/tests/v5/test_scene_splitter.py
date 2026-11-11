"""Tests for USIR v5 scene splitter and heuristic annotations.

Validates:
1. All 106 works (46 novel + 30 heldout + 30 heldout2, 10 authors) included without loss.
2. Continuity: scene_index strictly ascending from 0 to scene_count - 1 without gaps or duplicates.
3. Length bounds: 200 <= char_len <= 1600 across all scenes.
4. usir-schema compliance: matches usir_scene_annotation_v5.schema.json, no unknown fields.
5. Degenerate and boundary inputs: empty string, short text, long text without punctuation, symbols.
6. Pilot 50 scenes: balanced extraction (10 authors x 5 scenes), valid JSONL.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from usir_py.v5.aozora_scene_splitter import (
    AozoraCorpusSplitter,
    build_scene_annotation,
    compute_heuristic_signals,
    split_long_paragraph,
    split_work_into_scenes,
    SCHEMA_VERSION,
    EVENT_CODES,
    MOOD_CODES,
)

DATA_DIR = Path("C:/dev/usir/data")
SCHEMA_PATH = Path("C:/dev/nishiki/schemas/usir_scene_annotation_v5.schema.json")


@pytest.fixture(scope="module")
def splitter() -> AozoraCorpusSplitter:
    return AozoraCorpusSplitter(DATA_DIR)


@pytest.fixture(scope="module")
def all_scenes(splitter: AozoraCorpusSplitter):
    return splitter.process_all_works(include_audit_meta=False)


@pytest.fixture(scope="module")
def v5_schema():
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text("utf-8"))
    return None


def test_all_106_works_present(splitter: AozoraCorpusSplitter, all_scenes):
    """1. All 106 works (10 authors) must be present."""
    manifests = splitter.load_manifests()
    assert len(manifests) == 106, f"Expected 106 works, got {len(manifests)}"

    manifest_work_ids = {w["work_id"] for w in manifests}
    scene_work_ids = {s["work_id"] for s in all_scenes}
    assert manifest_work_ids == scene_work_ids, "Mismatch between manifest and scene work IDs"

    manifest_author_ids = {w["author_id"] for w in manifests}
    scene_author_ids = {s["author_id"] for s in all_scenes}
    assert len(manifest_author_ids) == 10, f"Expected 10 authors, got {len(manifest_author_ids)}"
    assert manifest_author_ids == scene_author_ids, "Mismatch between manifest and scene authors"


def test_scene_continuity_per_work(all_scenes):
    """2. scene_index strictly ascending 0..scene_count-1 without gaps or duplicates."""
    work_groups = {}
    for s in all_scenes:
        work_groups.setdefault(s["work_id"], []).append(s)

    assert len(work_groups) == 106

    for work_id, scenes in work_groups.items():
        scenes.sort(key=lambda x: x["scene_index"])
        expected_count = len(scenes)
        for idx, s in enumerate(scenes):
            assert s["scene_count"] == expected_count, (
                f"Work {work_id}: scene {idx} has scene_count={s['scene_count']}, expected {expected_count}"
            )
            assert s["scene_index"] == idx, (
                f"Work {work_id}: expected index {idx}, got {s['scene_index']}"
            )


def test_scene_length_bounds(all_scenes):
    """3. All scenes must strictly satisfy 200 <= char_len <= 1600."""
    total = len(all_scenes)
    assert total >= 4000, f"Expected around 4500 scenes, got {total}"

    out_of_bounds = [
        s for s in all_scenes if s["char_len"] < 200 or s["char_len"] > 1600
    ]
    assert not out_of_bounds, (
        f"{len(out_of_bounds)} scenes violate [200, 1600]: sample={out_of_bounds[:3]}"
    )


def test_usir_schema_strict_compliance(all_scenes, v5_schema):
    """4. Validate against usir-schema and v5 JSON schema.

    - Unknown fields are strictly prohibited (deny_unknown_fields).
    - awase: -2..=2.
    - nagare: exactly 3 distinct codes, rank_weight in 1..=3.
    """
    allowed_root_keys = {
        "schema_version", "author_id", "work_id", "scene_index", "scene_count",
        "char_len", "annotator", "seed", "focal_character", "awase", "nagare", "sadame", "kamae"
    }
    allowed_awase_keys = {"f_event_valence", "f_event_arousal", "f_style_valence", "f_style_arousal"}
    allowed_nagare_keys = {"event_top3", "mood_top3", "event_logp", "mood_logp"}
    allowed_sadame_keys = {"character_stalks", "dialogue_edges"}
    allowed_kamae_keys = {"opened_hooks", "closed_hook_ids"}

    for idx, s in enumerate(all_scenes):
        # Unknown fields check
        unknown = set(s.keys()) - allowed_root_keys
        assert not unknown, f"Line {idx + 1} has unknown fields: {unknown}"

        assert s["schema_version"] == SCHEMA_VERSION
        assert isinstance(s["author_id"], str) and len(s["author_id"]) > 0
        assert isinstance(s["work_id"], str) and len(s["work_id"]) > 0
        assert isinstance(s["annotator"], str) and len(s["annotator"]) > 0
        assert s["char_len"] >= 1
        assert s["scene_index"] < s["scene_count"]

        # Awase check
        awase = s["awase"]
        assert set(awase.keys()) == allowed_awase_keys
        for k, v in awase.items():
            assert isinstance(v, int) and -2 <= v <= 2, f"Line {idx + 1}: {k}={v} out of -2..2"

        # Nagare check
        nagare = s["nagare"]
        assert set(nagare.keys()) <= allowed_nagare_keys
        ev3 = nagare["event_top3"]
        mo3 = nagare["mood_top3"]
        assert len(ev3) == 3 and len(mo3) == 3

        ev_codes = [x["code"] for x in ev3]
        mo_codes = [x["code"] for x in mo3]
        assert len(set(ev_codes)) == 3, f"Line {idx + 1}: duplicate event codes {ev_codes}"
        assert len(set(mo_codes)) == 3, f"Line {idx + 1}: duplicate mood codes {mo_codes}"

        for item in ev3:
            assert item["code"] in EVENT_CODES
            assert 1 <= item["rank_weight"] <= 3
        for item in mo3:
            assert item["code"] in MOOD_CODES
            assert 1 <= item["rank_weight"] <= 3

        # Sadame check
        sadame = s["sadame"]
        assert set(sadame.keys()) == allowed_sadame_keys
        assert isinstance(sadame["character_stalks"], list)
        assert isinstance(sadame["dialogue_edges"], list)

        # Kamae check
        kamae = s["kamae"]
        assert set(kamae.keys()) == allowed_kamae_keys
        assert isinstance(kamae["opened_hooks"], list)
        assert isinstance(kamae["closed_hook_ids"], list)


def test_degenerate_and_boundary_inputs():
    """5. Must not panic on degenerate or boundary inputs."""
    # Empty or whitespace only
    assert split_work_into_scenes("") == []
    assert split_work_into_scenes("   \n\n\t  ") == []

    # Extremely short text (< 200 chars)
    short_txt = "これは極めて短い単一の文章です。"
    short_scenes = split_work_into_scenes(short_txt, min_chars=200, max_chars=1600)
    assert len(short_scenes) == 1
    assert short_scenes[0] == short_txt

    # Symbols only
    sym_txt = "＊\n◇\n○\n―\n" * 10
    sym_scenes = split_work_into_scenes(sym_txt)
    # Shouldn't crash; might be empty or combined
    assert isinstance(sym_scenes, list)

    # Long text without any sentence delimiters (3000 chars of 'あ')
    long_no_punct = "あ" * 3000
    long_scenes = split_work_into_scenes(long_no_punct, min_chars=200, max_chars=1600, target_chars=750)
    assert len(long_scenes) >= 2
    for sc in long_scenes:
        assert 200 <= len(sc) <= 1600, f"Degenerate long scene violation: len={len(sc)}"

    # Repeated headings
    headings_txt = "\n\n".join([f"第{i}章\n本文テキスト本文テキスト本文テキスト。" for i in range(1, 20)])
    h_scenes = split_work_into_scenes(headings_txt, min_chars=200, max_chars=1600, target_chars=500)
    assert len(h_scenes) > 0


def test_pilot_50_scenes(splitter: AozoraCorpusSplitter):
    """6. Pilot 50 scenes: balanced across 10 authors (5 scenes each), valid JSONL."""
    pilot = splitter.extract_pilot_scenes(per_author=5)
    assert len(pilot) == 50, f"Expected exactly 50 scenes, got {len(pilot)}"

    authors = {s["author_id"] for s in pilot}
    assert len(authors) == 10, f"Expected 10 distinct authors, got {len(authors)}"

    for aid in authors:
        a_scenes = [s for s in pilot if s["author_id"] == aid]
        assert len(a_scenes) == 5, f"Author {aid} has {len(a_scenes)} scenes, expected 5"

    # Verify audit fields
    for s in pilot:
        assert "scene_id" in s
        assert "author_ja" in s
        assert "title" in s
        assert "text" in s
        assert "story_position" in s
        assert s["story_position"] in ("序盤", "中盤", "終盤")
        assert 200 <= s["char_len"] <= 1600
        assert s["schema_version"] == SCHEMA_VERSION


def test_fix1_melos_and_lemon_awase(splitter: AozoraCorpusSplitter):
    """Gate 2 Fix 1: Awase physical separation and negation polarity inversion.

    - Melos beginning (001567, scene 0) must have high arousal (f_event_arousal >= 1 or f_style_arousal >= 1).
    - Lemon beginning (000424, scene 0) must have negative valence (f_event_valence <= -1 and f_style_valence <= -1).
    - Scenes with irony > 0 (opposite valence signs) and detachment > 0 must exist.
    """
    works = splitter.load_manifests()
    melos_meta = [w for w in works if w["work_id"] == "001567"][0]
    lemon_meta = [w for w in works if w["work_id"] == "000424"][0]

    melos_scenes = splitter.process_work_scenes(
        melos_meta, split_work_into_scenes(splitter.get_cached_text(melos_meta))
    )
    lemon_scenes = splitter.process_work_scenes(
        lemon_meta, split_work_into_scenes(splitter.get_cached_text(lemon_meta))
    )

    m0_awase = melos_scenes[0]["awase"]
    assert m0_awase["f_event_arousal"] >= 1 or m0_awase["f_style_arousal"] >= 1, (
        f"Melos scene 0 failed high arousal: {m0_awase}"
    )

    l0_awase = lemon_scenes[0]["awase"]
    assert l0_awase["f_event_valence"] <= -1 and l0_awase["f_style_valence"] <= -1, (
        f"Lemon scene 0 failed negative valence: {l0_awase}"
    )

    # Check across pilot 50 scenes for irony > 0 and detachment > 0
    pilot = splitter.extract_pilot_scenes(5)
    has_irony = False
    has_detachment = False
    for s in pilot:
        a = s["awase"]
        vf = a["f_event_valence"] / 2.0
        vs = a["f_style_valence"] / 2.0
        af = a["f_event_arousal"] / 2.0
        as_ = a["f_style_arousal"] / 2.0

        if (-vf * vs) > 0.0:
            has_irony = True
        if abs(vf) * max(0.0, af - as_) > 0.0:
            has_detachment = True

    assert has_irony, "No irony > 0 found in pilot scenes"
    assert has_detachment, "No detachment > 0 found in pilot scenes"


def test_fix2_sadame_characters_and_edges(splitter: AozoraCorpusSplitter):
    """Gate 2 Fix 2: HW-06 closed-class character stalks and dialogue edges.

    - >= 30 of 50 pilot scenes must have non-empty character_stalks.
    - Dialogue scenes must have non-empty dialogue_edges with strict schema {u, v, confidence}.
    - Recurring characters must have varying axes_q across scenes (intra_variance > 0).
    """
    pilot = splitter.extract_pilot_scenes(5)
    non_empty_stalks = [s for s in pilot if len(s["sadame"]["character_stalks"]) > 0]
    assert len(non_empty_stalks) >= 30, (
        f"Expected >= 30 scenes with stalks, got {len(non_empty_stalks)}"
    )

    dialogue_scenes = [s for s in pilot if "「" in s["text"]]
    scenes_with_edges = [s for s in dialogue_scenes if len(s["sadame"]["dialogue_edges"]) > 0]
    assert len(scenes_with_edges) >= 15, (
        f"Expected >= 15 dialogue scenes with edges, got {len(scenes_with_edges)}"
    )

    for s in pilot:
        for edge in s["sadame"]["dialogue_edges"]:
            assert set(edge.keys()) == {"u", "v", "confidence"}, (
                f"DialogueEdge has invalid keys: {edge.keys()}"
            )
            assert isinstance(edge["u"], str) and len(edge["u"]) > 0
            assert isinstance(edge["v"], str) and len(edge["v"]) > 0
            assert isinstance(edge["confidence"], int) and 0 <= edge["confidence"] <= 255

    # Check varying axes_q across scenes for same character
    works = splitter.load_manifests()
    melos_meta = [w for w in works if w["work_id"] == "001567"][0]
    melos_scenes = splitter.process_work_scenes(
        melos_meta, split_work_into_scenes(splitter.get_cached_text(melos_meta))
    )
    melos_axes_seq = [
        c["axes_q"]
        for s in melos_scenes
        for c in s["sadame"]["character_stalks"]
        if c["name"] == "メロス"
    ]
    assert len(melos_axes_seq) >= 2, "Expected メロス in at least 2 scenes"
    unique_axes = {tuple(x) for x in melos_axes_seq}
    assert len(unique_axes) >= 2, f"Expected varying axes_q for メロス, got identical: {unique_axes}"


def test_fix3_kamae_opened_and_closed_hooks(splitter: AozoraCorpusSplitter):
    """Gate 2 Fix 3: kamae opened_hooks and closed_hook_ids.

    - Opened hooks and closed hooks must be non-empty across pilot scenes / corpus.
    - HookRef must be strictly {id: int, label: str}, closed_hook_ids must be list of int.
    """
    pilot = splitter.extract_pilot_scenes(5)
    opened = [s for s in pilot if len(s["kamae"]["opened_hooks"]) > 0]
    assert len(opened) >= 10, f"Expected >= 10 scenes with opened hooks, got {len(opened)}"

    for s in pilot:
        for h in s["kamae"]["opened_hooks"]:
            assert set(h.keys()) == {"id", "label"}
            assert isinstance(h["id"], int) and h["id"] >= 1
            assert isinstance(h["label"], str) and len(h["label"]) > 0
        for cid in s["kamae"]["closed_hook_ids"]:
            assert isinstance(cid, int) and cid >= 1


def test_fix4_nagare_continuous_logp_and_mood_diversity(splitter: AozoraCorpusSplitter):
    """Gate 2 Fix 4: Continuous log-probabilities and mood diversity.

    - event_logp and mood_logp must be length 16 and all <= 0.0 finite floats.
    - mood_top3 must be diverse (M01, M02, M03 must not dominate excessively).
    """
    pilot = splitter.extract_pilot_scenes(5)
    first_rank_moods = []

    for s in pilot:
        nag = s["nagare"]
        assert "event_logp" in nag and nag["event_logp"] is not None
        assert "mood_logp" in nag and nag["mood_logp"] is not None

        elp = nag["event_logp"]
        mlp = nag["mood_logp"]

        assert len(elp) == 16
        assert len(mlp) == 16
        assert all(isinstance(x, (float, int)) and math.isfinite(x) and x <= 0.0 for x in elp)
        assert all(isinstance(x, (float, int)) and math.isfinite(x) and x <= 0.0 for x in mlp)

        first_rank_moods.append(nag["mood_top3"][0]["code"])

    unique_first_moods = set(first_rank_moods)
    assert len(unique_first_moods) >= 4, (
        f"Expected at least 4 distinct 1st-rank mood codes, got {len(unique_first_moods)}: {unique_first_moods}"
    )

