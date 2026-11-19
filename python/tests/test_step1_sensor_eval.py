"""Unit tests for the Step 1 sensor.v2 F9 evaluation harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from usir_py.sensor_eval.f9_loader import (
    F9Character,
    F9UtteranceGold,
    F9Work,
    extract_outer_quote_spans,
    load_f9_dataset,
    number_outer_quotes,
)
from usir_py.sensor_eval.metrics import (
    compute_politeness_pairwise_agreement,
    evaluate_model_predictions,
    validate_sensor_v2_utterances,
)
from usir_py.sensor_eval.ollama_client import call_ollama_chat, parse_json_content
from usir_py.sensor_eval.prompt import build_ollama_utterances_format_schema, compute_prompt_hash


def _make_synthetic_gold_dir(tmp_path: Path) -> tuple[Path, Path]:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    w1_text = "地の文。\n「おや『本当』かい」と男が言った。\n「ええ、本当です」\n"
    w2_text = "「行こう」\n「ああ」\n"
    (gold_dir / "w1.txt").write_text(w1_text, encoding="utf-8")
    (gold_dir / "w2.txt").write_text(w2_text, encoding="utf-8")

    s1 = extract_outer_quote_spans(w1_text)
    s2 = extract_outer_quote_spans(w2_text)
    gold_obj = {
        "note": "synthetic",
        "works": [
            {
                "id": "w1",
                "narration": "third",
                "narrator": None,
                "pov": "otoko",
                "characters": [
                    {"id": "otoko", "aliases": ["男"], "gender": "Male"},
                    {"id": "onna", "aliases": ["女"], "gender": "Female"},
                ],
                "utterances": [
                    {"n": 0, "open": s1[0][0], "head": "おや『本当』か", "status": "attributed", "speaker": "otoko"},
                    {"n": 1, "open": s1[1][0], "head": "ええ、本当です", "status": "attributed", "speaker": "onna"},
                ],
            },
            {
                "id": "w2",
                "narration": "first",
                "narrator": "boku",
                "pov": None,
                "characters": [
                    {"id": "boku", "aliases": ["僕"], "gender": "Male"},
                    {"id": "kimi", "aliases": ["君"], "gender": "Male"},
                ],
                "utterances": [
                    {"n": 0, "open": s2[0][0], "head": "行こう", "status": "attributed", "speaker": "boku"},
                    {"n": 1, "open": s2[1][0], "head": "ああ", "status": "undetermined", "speaker": None},
                ],
            },
        ],
    }
    gold_bytes = json.dumps(gold_obj, ensure_ascii=False).encode("utf-8")
    (gold_dir / "gold.json").write_bytes(gold_bytes)
    sha = hashlib.sha256(gold_bytes).hexdigest()

    split_path = tmp_path / "split.json"
    split_obj = {
        "split_version": "test_v1",
        "gold_json_sha256": sha,
        "dev": {"work_ids": ["w1"]},
        "holdout": {"sealed": True, "work_ids": ["w2"]},
    }
    split_path.write_text(json.dumps(split_obj), encoding="utf-8")
    return gold_dir, split_path


def test_holdout_is_sealed_by_default(tmp_path: Path) -> None:
    gold_dir, split_path = _make_synthetic_gold_dir(tmp_path)
    with pytest.raises(PermissionError, match="sealed"):
        load_f9_dataset(gold_dir, split_path, split="holdout")
    ds_dev = load_f9_dataset(gold_dir, split_path, split="dev")
    assert [w.work_id for w in ds_dev.works] == ["w1"]
    ds_hold = load_f9_dataset(gold_dir, split_path, split="holdout", unlock_holdout=True)
    assert [w.work_id for w in ds_hold.works] == ["w2"]


def test_real_f9_split_manifest_integrity() -> None:
    split_path = Path(__file__).resolve().parents[2] / "scripts" / "step1" / "f9_split.json"
    cfg = json.loads(split_path.read_text(encoding="utf-8"))
    dev_ids = set(cfg["dev"]["work_ids"])
    hold_ids = set(cfg["holdout"]["work_ids"])
    assert len(dev_ids) == 4
    assert len(hold_ids) == 4
    assert dev_ids.isdisjoint(hold_ids)
    assert cfg["holdout"]["sealed"] is True
    assert cfg["dev"]["summary"]["attributed_utterances"] == 90
    assert cfg["holdout"]["summary"]["attributed_utterances"] == 96
    assert (
        cfg["dev"]["summary"]["attributed_utterances"]
        + cfg["holdout"]["summary"]["attributed_utterances"]
        == 186
    )


def test_outer_quote_numbering_handles_nested_and_multiline() -> None:
    txt = "導入。\n「『カ』という\n所へ行くの」と子。\n「そうかい」"
    numbered, spans = number_outer_quotes(txt)
    assert len(spans) == 2
    assert txt[spans[0][0] : spans[0][1]] == "「『カ』という\n所へ行くの」"
    assert txt[spans[1][0] : spans[1][1]] == "「そうかい」"
    assert "[U0]「『カ』という" in numbered
    assert "[U1]「そうかい」" in numbered


def test_metrics_computation_with_and_without_politeness_gold() -> None:
    work = F9Work(
        work_id="w_test",
        author="a",
        title="t",
        narration="third",
        narrator=None,
        pov="otoko",
        characters=[
            F9Character(id="otoko", aliases=["男"], gender="Male"),
            F9Character(id="onna", aliases=["女"], gender="Female"),
        ],
        text="x",
        numbered_text="x",
        scene_hash="0" * 64,
        utterances=[
            # 0: attributed, explicit & right -> ok
            F9UtteranceGold(0, 0, 1, "h0", "attributed", "otoko", "「h0」", "plain"),
            # 1: attributed, explicit & wrong speaker -> explicit error + overall error
            F9UtteranceGold(1, 2, 3, "h1", "attributed", "onna", "「h1」", "polite"),
            # 2: attributed, ambiguous evasion (even if speaker happens to match!) -> ambiguous + overall error
            F9UtteranceGold(2, 4, 5, "h2", "attributed", "otoko", "「h2」", "plain"),
            # 3: attributed, inferred & right -> ok
            F9UtteranceGold(3, 6, 7, "h3", "attributed", "onna", "「h3」", "polite"),
            # 4: undetermined, correctly marked ambiguous -> excluded_abstained
            F9UtteranceGold(4, 8, 9, "h4", "undetermined", None, "「h4」", None),
            # 5: not_speech, wrongly attributed -> excluded_over_attributed
            F9UtteranceGold(5, 10, 11, "h5", "not_speech", None, "「h5」", None),
        ],
    )
    preds = {
        "w_test": [
            {
                "utterance_index": 0,
                "speaker_id": "otoko",
                "addressee_id": "onna",
                "evidence": "explicit",
                "register": {"politeness": "plain", "first_person": None},
            },
            {
                "utterance_index": 1,
                "speaker_id": "otoko",
                "addressee_id": None,
                "evidence": "explicit",
                "register": {"politeness": "plain", "first_person": None},
            },
            {
                "utterance_index": 2,
                "speaker_id": "otoko",
                "addressee_id": None,
                "evidence": "ambiguous",
                "register": {"politeness": "plain", "first_person": None},
            },
            {
                "utterance_index": 3,
                "speaker_id": "onna",
                "addressee_id": "otoko",
                "evidence": "inferred",
                "register": {"politeness": "polite", "first_person": None},
            },
            {
                "utterance_index": 4,
                "speaker_id": None,
                "addressee_id": None,
                "evidence": "ambiguous",
                "register": {"politeness": "unknown", "first_person": None},
            },
            {
                "utterance_index": 5,
                "speaker_id": "otoko",
                "addressee_id": None,
                "evidence": "inferred",
                "register": {"politeness": "unknown", "first_person": None},
            },
        ]
    }
    res = evaluate_model_predictions([work], preds, "m1", compute_prompt_hash())
    ov = res["overall"]
    assert res["schema_validation"]["all_works_valid"] is True
    assert ov["attributed_total"] == 4
    assert ov["explicit_denominator"] == 2
    assert ov["explicit_errors"] == 1
    assert ov["explicit_error_rate"] == pytest.approx(0.5)
    assert ov["overall_errors"] == 2  # idx 1 wrong_speaker, idx 2 ambiguous_evasion
    assert ov["overall_error_rate"] == pytest.approx(0.5)
    assert ov["overall_errors_raw_speaker"] == 1
    assert ov["overall_error_rate_raw_speaker"] == pytest.approx(0.25)
    assert ov["politeness_gold_available"] is True
    assert ov["politeness_correct"] == 3
    assert ov["politeness_agreement_rate"] == pytest.approx(0.75)
    assert ov["ambiguous_rate"] == pytest.approx(0.25)
    assert ov["excluded_total"] == 2
    assert ov["excluded_abstained"] == 1
    assert ov["excluded_over_attributed"] == 1

    # Without politeness gold labels, politeness_agreement_rate must be None.
    work_no_pol = F9Work(
        work_id="w_test",
        author="a",
        title="t",
        narration="third",
        narrator=None,
        pov="otoko",
        characters=work.characters,
        text="x",
        numbered_text="x",
        scene_hash="0" * 64,
        utterances=[
            F9UtteranceGold(u.n, u.open_offset, u.close_offset, u.head, u.status, u.speaker, u.quote_text, None)
            for u in work.utterances
        ],
    )
    res_no_pol = evaluate_model_predictions([work_no_pol], preds, "m1", compute_prompt_hash())
    assert res_no_pol["overall"]["politeness_gold_available"] is False
    assert res_no_pol["overall"]["politeness_agreement_rate"] is None

    pw = compute_politeness_pairwise_agreement([work_no_pol], {"m1": preds, "m2": preds})
    assert pw["pairs"][0]["agreement_rate"] == pytest.approx(1.0)


def test_sensor_v2_validation_rejects_invalid_utterances() -> None:
    work = F9Work(
        work_id="w_val",
        author=None,
        title=None,
        narration="third",
        narrator=None,
        pov=None,
        characters=[F9Character(id="a", aliases=[], gender="Unknown")],
        text="x",
        numbered_text="x",
        scene_hash="a" * 64,
        utterances=[],
    )
    invalid = [
        {
            "utterance_index": 0,
            "speaker_id": "",  # violates minLength: 1
            "addressee_id": None,
            "evidence": "explicit",
            "register": {"politeness": "casual", "first_person": None},  # invalid enum
            "extra_key": 123,  # additionalProperties: false
        }
    ]
    errs = validate_sensor_v2_utterances(work, invalid, "m", "h")
    assert len(errs) >= 3


def test_parse_json_and_ollama_mock() -> None:
    fenced = '```json\n{"utterances": []}\n```'
    parsed, err = parse_json_content(fenced)
    assert err is None
    assert parsed == {"utterances": []}

    broken, err2 = parse_json_content("{not json")
    assert broken is None
    assert err2 is not None and "JSONDecodeError" in err2

    fake_api_resp = {
        "message": {"content": '{"utterances": [{"utterance_index": 0}]}'},
        "prompt_eval_count": 100,
        "eval_count": 20,
    }
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=fake_api_resp):
        resp = call_ollama_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            format_schema=build_ollama_utterances_format_schema(),
            think=False,
        )
    assert resp.parse_error is None
    assert resp.parsed_json == {"utterances": [{"utterance_index": 0}]}
    assert resp.think is False
    assert resp.format_mode == "json_schema"


def test_merge_result_files_to_summary(tmp_path: Path) -> None:
    from usir_py.sensor_eval.run_step1_f9 import merge_result_files_to_summary, run_evaluation

    gold_dir, split_path = _make_synthetic_gold_dir(tmp_path)
    ds = load_f9_dataset(gold_dir, split_path, split="dev")
    out_dir = tmp_path / "out"

    resp_a = {
        "message": {
            "content": json.dumps(
                {
                    "utterances": [
                        {
                            "utterance_index": 0,
                            "speaker_id": "otoko",
                            "addressee_id": "onna",
                            "evidence": "explicit",
                            "register": {"politeness": "plain", "first_person": None},
                        },
                        {
                            "utterance_index": 1,
                            "speaker_id": "onna",
                            "addressee_id": "otoko",
                            "evidence": "inferred",
                            "register": {"politeness": "polite", "first_person": None},
                        },
                    ]
                }
            )
        },
        "prompt_eval_count": 50,
        "eval_count": 30,
    }
    resp_b = {
        "message": {
            "content": json.dumps(
                {
                    "utterances": [
                        {
                            "utterance_index": 0,
                            "speaker_id": "otoko",
                            "addressee_id": "onna",
                            "evidence": "explicit",
                            "register": {"politeness": "rough", "first_person": None},
                        },
                        {
                            "utterance_index": 1,
                            "speaker_id": "onna",
                            "addressee_id": "otoko",
                            "evidence": "inferred",
                            "register": {"politeness": "polite", "first_person": None},
                        },
                    ]
                }
            )
        },
        "prompt_eval_count": 60,
        "eval_count": 35,
    }
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=resp_a):
        run_evaluation(ds, ["m_a:1b"], out_dir, "http://127.0.0.1:11434", 4096, 512, 42, 30.0, False)
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=resp_b):
        run_evaluation(ds, ["m_b:2b"], out_dir, "http://127.0.0.1:11434", 4096, 512, 42, 30.0, False)

    merged = merge_result_files_to_summary(ds, ["m_a:1b", "m_b:2b"], out_dir)
    assert [m["model_spec"] for m in merged["models"]] == ["m_a:1b", "m_b:2b"]
    assert merged["politeness_pairwise_agreement"]["pairs"][0]["agreement_rate"] == pytest.approx(0.5)

