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
from usir_py.sensor_eval.features import (
    extract_utterance_features_from_text,
    extract_work_utterance_features,
    find_speakers_in_text,
    find_speech_verb,
    format_utterance_features_block,
)
from usir_py.sensor_eval.metrics import (
    compute_politeness_pairwise_agreement,
    evaluate_model_predictions,
    evaluate_step1_gate,
    is_forbidden_null_literal,
    validate_sensor_v2_utterances,
    validate_speaker_id,
)
from usir_py.sensor_eval.ollama_client import call_ollama_chat, parse_json_content
from usir_py.sensor_eval.postprocess import (
    PostprocessConfig,
    apply_postprocessing_to_work,
    replay_saved_result_with_postprocess,
)
from usir_py.sensor_eval.prompt import (
    build_ollama_utterances_format_schema,
    build_work_messages,
    compute_prompt_hash,
    get_prompt,
)


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


def test_validator_rejects_string_null_and_non_roster_speaker_id() -> None:
    work = F9Work(
        work_id="w_null",
        author=None,
        title=None,
        narration="third",
        narrator=None,
        pov=None,
        characters=[F9Character(id="otoko", aliases=["男"], gender="Male")],
        text="「あ」「い」「う」「え」「お」「か」",
        numbered_text="[U0]「あ」[U1]「い」[U2]「う」[U3]「え」[U4]「お」[U5]「か」",
        scene_hash="b" * 64,
        utterances=[
            F9UtteranceGold(0, 0, 3, "あ", "attributed", "otoko", "「あ」"),
            F9UtteranceGold(1, 3, 6, "い", "attributed", "otoko", "「い」"),
            F9UtteranceGold(2, 6, 9, "う", "attributed", "otoko", "「う」"),
            F9UtteranceGold(3, 9, 12, "え", "attributed", "otoko", "「え」"),
            F9UtteranceGold(4, 12, 15, "お", "attributed", "otoko", "「お」"),
            F9UtteranceGold(5, 15, 18, "か", "undetermined", None, "「か」"),
        ],
    )
    assert is_forbidden_null_literal("null") is True
    assert is_forbidden_null_literal(" NULL ") is True
    assert is_forbidden_null_literal("none") is True
    assert is_forbidden_null_literal("otoko") is False
    assert validate_speaker_id("null", {"otoko"}) == "string_null_literal"
    assert validate_speaker_id("unknown_char", {"otoko"}) == "not_in_roster"
    assert validate_speaker_id("otoko", {"otoko"}) is None
    assert validate_speaker_id(None, {"otoko"}) is None

    bad_utts = [
        {
            "utterance_index": 0,
            "speaker_id": "null",
            "addressee_id": "null",
            "evidence": "inferred",
            "register": {"politeness": "plain", "first_person": "null"},
        },
        {
            "utterance_index": 1,
            "speaker_id": "NULL",
            "addressee_id": None,
            "evidence": "explicit",
            "register": {"politeness": "plain", "first_person": None},
        },
        {
            "utterance_index": 2,
            "speaker_id": " null ",
            "addressee_id": None,
            "evidence": "inferred",
            "register": {"politeness": "plain", "first_person": None},
        },
        {
            "utterance_index": 3,
            "speaker_id": "none",
            "addressee_id": None,
            "evidence": "inferred",
            "register": {"politeness": "plain", "first_person": None},
        },
        {
            "utterance_index": 4,
            "speaker_id": "unknown_char",
            "addressee_id": None,
            "evidence": "inferred",
            "register": {"politeness": "plain", "first_person": None},
        },
        {
            "utterance_index": 5,
            "speaker_id": "null",
            "addressee_id": None,
            "evidence": "ambiguous",
            "register": {"politeness": "unknown", "first_person": None},
        },
    ]
    errs = validate_sensor_v2_utterances(work, bad_utts, "m", "h")
    assert any("Rejected string-null speaker_id='null'" in e for e in errs)
    assert any("Rejected string-null addressee_id='null'" in e for e in errs)
    assert any("Rejected string-null first_person='null'" in e for e in errs)
    assert any("Non-roster speaker_id='unknown_char'" in e for e in errs)

    metrics = evaluate_model_predictions([work], {"w_null": bad_utts}, "m", "h")
    assert metrics["schema_validation"]["all_works_valid"] is False
    assert metrics["non_roster_speaker_count"] == 6
    assert metrics["null_string_speaker_count"] == 5
    reasons = [entry["reason"] for entry in metrics["non_roster_speaker_ids"]]
    assert reasons == [
        "string_null_literal",
        "string_null_literal",
        "string_null_literal",
        "string_null_literal",
        "not_in_roster",
        "string_null_literal",
    ]
    assert metrics["overall"]["excluded_abstained"] == 0
    assert metrics["overall"]["excluded_over_attributed"] == 1
    gate = evaluate_step1_gate(metrics)
    assert gate["validation_passed"] is False
    assert gate["speaker_gate_passed"] is False


def test_prompt_versions_and_v1_hash_regression() -> None:
    assert (
        compute_prompt_hash(prompt_version="v1")
        == "7912bad97f799ce526ad60553adaac5456002c650f6981880f44bdb0b433de6a"
    )
    ver_v2, text_v2 = get_prompt("v2")
    assert ver_v2 == "step1_utterances_v2"
    assert compute_prompt_hash(prompt_version="v2") != compute_prompt_hash(prompt_version="v1")

    work = F9Work(
        work_id="w_p",
        author="a",
        title="t",
        narration="third",
        narrator=None,
        pov=None,
        characters=[F9Character(id="otoko", aliases=["男"], gender="Male")],
        text="「あ」",
        numbered_text="[U0]「あ」",
        scene_hash="c" * 64,
        utterances=[F9UtteranceGold(0, 0, 3, "あ", "attributed", "otoko", "「あ」")],
    )
    msgs_v2 = build_work_messages(work, prompt_version="v2")
    sys_msg = msgs_v2[0]["content"]
    assert sys_msg == text_v2
    assert "後置伝達節" in sys_msg
    assert "非発話引用" in sys_msg
    assert "ambiguous" in sys_msg
    assert "厳格基準" in sys_msg

    ver_v3, text_v3 = get_prompt("v3")
    assert ver_v3 == "step1_utterances_v3"
    h1 = compute_prompt_hash(prompt_version="v1")
    h2 = compute_prompt_hash(prompt_version="v2")
    h3 = compute_prompt_hash(prompt_version="v3")
    assert len({h1, h2, h3}) == 3
    msgs_v3 = build_work_messages(work, prompt_version="v3")
    assert msgs_v3[0]["content"] == text_v3


def test_validator_rejects_missing_and_extra_utterance_indices() -> None:
    work = F9Work(
        work_id="w_miss",
        author=None,
        title=None,
        narration="third",
        narrator=None,
        pov=None,
        characters=[F9Character(id="otoko", aliases=["男"], gender="Male")],
        text="「あ」「い」",
        numbered_text="[U0]「あ」[U1]「い」",
        scene_hash="d" * 64,
        utterances=[
            F9UtteranceGold(0, 0, 3, "あ", "attributed", "otoko", "「あ」"),
            F9UtteranceGold(1, 3, 6, "い", "attributed", "otoko", "「い」"),
        ],
    )
    preds_missing = [
        {
            "utterance_index": 0,
            "speaker_id": "otoko",
            "addressee_id": None,
            "evidence": "explicit",
            "register": {"politeness": "plain", "first_person": None},
        }
    ]
    errs_missing = validate_sensor_v2_utterances(work, preds_missing, "m", "h")
    assert any("Missing utterance_index(es) [1]" in e for e in errs_missing)

    m_missing = evaluate_model_predictions([work], {"w_miss": preds_missing}, "m", "h")
    assert m_missing["schema_validation"]["all_works_valid"] is False
    assert evaluate_step1_gate(m_missing)["validation_passed"] is False

    preds_extra = preds_missing + [
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
            "evidence": "inferred",
            "register": {"politeness": "plain", "first_person": None},
        },
    ]
    errs_extra = validate_sensor_v2_utterances(work, preds_extra, "m", "h")
    assert any("Out-of-range utterance_index(es) [2]" in e for e in errs_extra)


def test_evaluate_step1_gate_boundaries() -> None:
    def _mk_metrics(
        exp_err: float | None,
        exp_den: int,
        ov_err: float | None,
        all_valid: bool = True,
        non_roster: int = 0,
    ) -> dict:
        return {
            "overall": {
                "explicit_error_rate": exp_err,
                "explicit_denominator": exp_den,
                "overall_error_rate": ov_err,
                "politeness_gold_available": False,
                "politeness_agreement_rate": None,
            },
            "schema_validation": {"all_works_valid": all_valid},
            "non_roster_speaker_ids": [{"x": i} for i in range(non_roster)],
        }

    g_pass = evaluate_step1_gate(_mk_metrics(0.049, 20, 0.099))
    assert g_pass["speaker_gate_passed"] is True
    assert g_pass["passed"] is True
    assert g_pass["politeness_passed"] is None

    assert evaluate_step1_gate(_mk_metrics(0.05, 20, 0.05))["speaker_gate_passed"] is False
    assert evaluate_step1_gate(_mk_metrics(0.0, 20, 0.10))["speaker_gate_passed"] is False
    assert evaluate_step1_gate(_mk_metrics(0.0, 0, 0.0))["speaker_gate_passed"] is False
    assert evaluate_step1_gate(_mk_metrics(0.0, 20, 0.0, non_roster=1))["speaker_gate_passed"] is False
    assert evaluate_step1_gate(_mk_metrics(0.0, 20, 0.0, all_valid=False))["speaker_gate_passed"] is False


def test_cli_config_and_check_gate(tmp_path: Path) -> None:
    from usir_py.sensor_eval.run_step1_f9 import main

    gold_dir, split_path = _make_synthetic_gold_dir(tmp_path)
    out_dir = tmp_path / "cli_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "gold_dir": str(gold_dir),
                "split_file": str(split_path),
                "split": "dev",
                "prompt_version": "v2",
                "out_dir": str(out_dir),
                "models": [{"spec": "m_pass:1b", "num_ctx": 4096, "num_predict": 512}],
            }
        ),
        encoding="utf-8",
    )

    resp_pass = {
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
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=resp_pass):
        rc_pass = main(["--config", str(cfg_path), "--check-gate"])
    assert rc_pass == 0

    resp_fail = {
        "message": {
            "content": json.dumps(
                {
                    "utterances": [
                        {
                            "utterance_index": 0,
                            "speaker_id": "onna",
                            "addressee_id": None,
                            "evidence": "explicit",
                            "register": {"politeness": "plain", "first_person": None},
                        },
                        {
                            "utterance_index": 1,
                            "speaker_id": "otoko",
                            "addressee_id": None,
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
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=resp_fail):
        rc_fail = main(["--config", str(cfg_path), "--check-gate"])
    assert rc_fail == 2


def test_features_position_ownership_speakers_verbs_and_no_leakage() -> None:
    text = (
        "彼は所謂「自由」を嫌った。\n"
        "　「もう時間だよ」ジョバンニは叫びました。\n"
        "「ええ、\n本当です」\n"
        "信一がこう云い出したので、仙吉が\n"
        "「やりましょう」\n"
        "\n"
        "「誰だ」\n"
    )
    spans = extract_outer_quote_spans(text)
    assert len(spans) == 5
    chars = [
        F9Character(id="giovanni", aliases=["ジョバンニ"], gender="Male"),
        F9Character(id="shinichi", aliases=["信一"], gender="Male"),
        F9Character(id="senkichi", aliases=["仙吉"], gender="Male"),
        F9Character(id="boku", aliases=["僕"], gender="Male"),
    ]
    feats = extract_utterance_features_from_text(
        text=text, spans=spans, characters=chars, narration="first", narrator="boku"
    )
    assert [f.position for f in feats] == [
        "inline_embedded",
        "line_head_with_trailing",
        "standalone_line",
        "standalone_line",
        "standalone_line",
    ]
    assert [f.quote_chars for f in feats] == [2, 6, 8, 6, 2]

    # U1 post-quote reporting clause does not leak into U2 pre-window.
    assert feats[1].post_speakers == ("giovanni",)
    assert feats[1].post_speech_verb == "叫び"
    assert feats[1].post_speaker_and_verb is True
    assert feats[2].pre_has_narration is False
    assert feats[2].pre_speakers == ()
    assert feats[2].pre_speech_verb is None
    assert feats[2].pre_speaker_and_verb is False

    # U2's newline-separated reporting clause splits off trailing subject intro for U3.
    assert feats[2].post_speakers == ("shinichi",)
    assert feats[2].post_speech_verb == "云い出"
    assert feats[2].post_speaker_and_verb is True
    assert feats[3].pre_speakers == ("senkichi",)
    assert feats[3].pre_speech_verb is None

    # Blank line before U4 clears pre_has_narration.
    assert feats[4].pre_has_narration is False
    assert feats[4].pre_speakers == ()

    # Speaker & verb matching helpers: 1st-person guard, negations, noun vs verb.
    assert find_speakers_in_text("私的な用事だが、私は言った", chars, "first", "boku") == ("boku",)
    assert find_speakers_in_text("公私の別なく私は笑った", chars, "third", None) == ()
    assert find_speech_verb("彼は言いませんでした") is None
    assert find_speech_verb("彼女は言わない") is None
    assert find_speech_verb("挨拶でもした") is None
    assert find_speech_verb("話を聞いた") is None
    assert find_speech_verb("と尋ねた") == "尋ね"
    assert find_speech_verb("と聞きました") == "聞きまし"
    assert find_speech_verb("と挨拶した") == "挨拶し"

    # Zero gold-label leakage: altering gold status/speaker must not alter extracted features.
    numbered, _ = number_outer_quotes(text)
    utts_a = [
        F9UtteranceGold(i, s[0], s[1], "h", "attributed", "giovanni", text[s[0] : s[1]])
        for i, s in enumerate(spans)
    ]
    utts_b = [
        F9UtteranceGold(i, s[0], s[1], "h", "not_speech", None, text[s[0] : s[1]])
        for i, s in enumerate(spans)
    ]
    w_a = F9Work("w", None, None, "first", "boku", None, chars, text, numbered, "0" * 64, utts_a)
    w_b = F9Work("w", None, None, "first", "boku", None, chars, text, numbered, "0" * 64, utts_b)
    assert extract_work_utterance_features(w_a) == extract_work_utterance_features(w_b)


def test_prompt_v4_strict_ablation_over_v3() -> None:
    ver_v3, sys_v3 = get_prompt("v3")
    ver_v4, sys_v4 = get_prompt("v4")
    assert ver_v3 == "step1_utterances_v3"
    assert ver_v4 == "step1_utterances_v4"
    assert sys_v4 == sys_v3

    hashes = {compute_prompt_hash(prompt_version=v) for v in ("v1", "v2", "v3", "v4")}
    assert len(hashes) == 4

    text = "「おはよう」と男が言った。\n彼は「自由」を嫌った。\n"
    numbered, spans = number_outer_quotes(text)
    chars = [F9Character(id="otoko", aliases=["男"], gender="Male")]
    utts = [
        F9UtteranceGold(0, spans[0][0], spans[0][1], "おはよう", "attributed", "otoko", "「おはよう」"),
        F9UtteranceGold(1, spans[1][0], spans[1][1], "自由", "not_speech", None, "「自由」"),
    ]
    work = F9Work("w_ab", None, None, "third", None, None, chars, text, numbered, "e" * 64, utts)

    msgs_v3 = build_work_messages(work, prompt_version="v3")
    msgs_v4 = build_work_messages(work, prompt_version="v4")
    assert msgs_v4[0] == msgs_v3[0]

    feat_block = format_utterance_features_block(extract_work_utterance_features(work))
    expected_user_v4 = msgs_v3[1]["content"].replace(
        "\n\n上記 `[U0]`",
        f"\n\n{feat_block}\n\n上記 `[U0]`",
    )
    assert msgs_v4[1]["content"] == expected_user_v4
    assert "| U0 | line_head_with_trailing | 4 | - | - | false | otoko | 言っ | true |" in feat_block
    assert "| U1 | inline_embedded | 2 | - | - | false | - | - | false |" in feat_block


def test_prompt_v5_strict_ablation_over_v4() -> None:
    ver_v3, sys_v3 = get_prompt("v3")
    ver_v4, sys_v4 = get_prompt("v4")
    ver_v5, sys_v5 = get_prompt("v5")
    assert ver_v3 == "step1_utterances_v3"
    assert ver_v4 == "step1_utterances_v4"
    assert ver_v5 == "step1_utterances_v5"
    assert sys_v5 == sys_v4 == sys_v3

    hashes = {compute_prompt_hash(prompt_version=v) for v in ("v1", "v2", "v3", "v4", "v5")}
    assert len(hashes) == 5

    text = (
        "自分の机に向って、娘が言った。\n"
        "「お帰りなさい」とツネ子が呟いた。\n"
        "彼は「自由」を嫌った。\n"
    )
    numbered, spans = number_outer_quotes(text)
    chars = [
        F9Character(id="yozo", aliases=["葉蔵"], gender="Male"),
        F9Character(id="musume", aliases=["娘"], gender="Female"),
        F9Character(id="tsuneko", aliases=["ツネ子"], gender="Female"),
    ]
    utts = [
        F9UtteranceGold(0, spans[0][0], spans[0][1], "お帰りなさい", "attributed", "tsuneko", "「お帰りなさい」"),
        F9UtteranceGold(1, spans[1][0], spans[1][1], "自由", "not_speech", None, "「自由」"),
    ]
    work = F9Work("w_ab5", None, None, "first", "yozo", None, chars, text, numbered, "f" * 64, utts)

    msgs_v3 = build_work_messages(work, prompt_version="v3")
    msgs_v4 = build_work_messages(work, prompt_version="v4")
    msgs_v5 = build_work_messages(work, prompt_version="v5")
    assert msgs_v5[0] == msgs_v4[0] == msgs_v3[0]

    feat_block_v5 = format_utterance_features_block(
        extract_work_utterance_features(work), include_speaker_ids=False
    )
    expected_user_v5 = msgs_v3[1]["content"].replace(
        "\n\n上記 `[U0]`",
        f"\n\n{feat_block_v5}\n\n上記 `[U0]`",
    )
    assert msgs_v5[1]["content"] == expected_user_v5
    assert "| id | pos | chars | pre_verb | pre_spk_verb | post_verb | post_spk_verb |" in feat_block_v5
    assert "pre_spk |" not in feat_block_v5
    assert "post_spk |" not in feat_block_v5
    assert "| U0 | line_head_with_trailing | 6 | 言っ | true | 呟い | true |" in feat_block_v5
    assert "| U1 | inline_embedded | 2 | - | false | - | false |" in feat_block_v5
    for forbidden in ("yozo", "musume", "tsuneko", "葉蔵", "娘", "ツネ子", "自分"):
        assert forbidden not in feat_block_v5


def test_not_speech_and_undetermined_abstention_breakdown() -> None:
    work = F9Work(
        work_id="w_exc",
        author=None,
        title=None,
        narration="first",
        narrator="yozo",
        pov=None,
        characters=[F9Character(id="yozo", aliases=["葉蔵"], gender="Male")],
        text="x",
        numbered_text="x",
        scene_hash="f" * 64,
        utterances=[
            F9UtteranceGold(0, 0, 3, "h0", "attributed", "yozo", "「あ」"),
            F9UtteranceGold(1, 3, 6, "h1", "not_speech", None, "「い」"),
            F9UtteranceGold(2, 6, 9, "h2", "not_speech", None, "「う」"),
            F9UtteranceGold(3, 9, 12, "h3", "undetermined", None, "「え」"),
        ],
    )
    preds = {
        "w_exc": [
            {
                "utterance_index": 0,
                "speaker_id": "yozo",
                "addressee_id": None,
                "evidence": "explicit",
                "register": {"politeness": "plain", "first_person": None},
            },
            {
                "utterance_index": 1,
                "speaker_id": None,
                "addressee_id": None,
                "evidence": "ambiguous",
                "register": {"politeness": "unknown", "first_person": None},
            },
            {
                "utterance_index": 2,
                "speaker_id": "yozo",
                "addressee_id": None,
                "evidence": "inferred",
                "register": {"politeness": "plain", "first_person": None},
            },
            {
                "utterance_index": 3,
                "speaker_id": None,
                "addressee_id": None,
                "evidence": "ambiguous",
                "register": {"politeness": "unknown", "first_person": None},
            },
        ]
    }
    res = evaluate_model_predictions([work], preds, "m", compute_prompt_hash(prompt_version="v4"))
    ov = res["overall"]
    assert ov["excluded_total"] == 3
    assert ov["excluded_abstained"] == 2
    assert ov["excluded_abstention_rate"] == pytest.approx(2 / 3)
    assert ov["not_speech_total"] == 2
    assert ov["not_speech_abstained"] == 1
    assert ov["not_speech_over_attributed"] == 1
    assert ov["not_speech_abstention_rate"] == pytest.approx(0.5)
    assert ov["undetermined_total"] == 1
    assert ov["undetermined_abstained"] == 1
    assert ov["undetermined_over_attributed"] == 0
    assert ov["undetermined_abstention_rate"] == pytest.approx(1.0)


def test_postprocess_config_validation_and_defaults() -> None:
    default_cfg = PostprocessConfig()
    assert default_cfg.explicit_evidence_gate is False
    assert default_cfg.inline_non_speech_filter is False
    assert default_cfg.inline_non_speech_max_chars == 11
    assert default_cfg.is_enabled is False
    assert PostprocessConfig.from_dict(None) == default_cfg

    enabled_cfg = PostprocessConfig.from_dict(
        {
            "explicit_evidence_gate": True,
            "inline_non_speech_filter": True,
            "inline_non_speech_max_chars": 8,
        }
    )
    assert enabled_cfg.is_enabled is True
    assert enabled_cfg.to_dict() == {
        "explicit_evidence_gate": True,
        "inline_non_speech_filter": True,
        "inline_non_speech_max_chars": 8,
    }

    with pytest.raises(ValueError, match="dict"):
        PostprocessConfig.from_dict(["invalid"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown postprocess config keys"):
        PostprocessConfig.from_dict({"unknown_flag": True})
    with pytest.raises(ValueError, match="non-negative integer"):
        PostprocessConfig.from_dict({"inline_non_speech_max_chars": True})
    with pytest.raises(ValueError, match="non-negative integer"):
        PostprocessConfig.from_dict({"inline_non_speech_max_chars": -1})
    with pytest.raises(ValueError, match="bool"):
        PostprocessConfig.from_dict({"explicit_evidence_gate": 1})


def test_postprocess_explicit_evidence_gate_and_inline_non_speech_filter() -> None:
    text = (
        "男が言った。\n"
        "「おはよう」\n"
        "「こんばんは」\n"
        "「おやすみ」\n"
        "彼は所謂「自由」を嫌った。\n"
        "彼は所謂「一二三四五六」を嫌った。\n"
        "誰かが「はい」と言った。\n"
        "「あ」\n"
    )
    numbered, spans = number_outer_quotes(text)
    assert len(spans) == 7
    chars = [
        F9Character(id="otoko", aliases=["男"], gender="Male"),
        F9Character(id="onna", aliases=["女"], gender="Female"),
    ]
    utts_gold_a = [
        F9UtteranceGold(i, s[0], s[1], "h", "attributed", "otoko", text[s[0] : s[1]])
        for i, s in enumerate(spans)
    ]
    utts_gold_b = [
        F9UtteranceGold(i, s[0], s[1], "h", "not_speech", None, text[s[0] : s[1]])
        for i, s in enumerate(spans)
    ]
    work_a = F9Work("w_pp", None, None, "third", None, None, chars, text, numbered, "1" * 64, utts_gold_a)
    work_b = F9Work("w_pp", None, None, "third", None, None, chars, text, numbered, "1" * 64, utts_gold_b)

    raw_preds = [
        {"utterance_index": 0, "speaker_id": "otoko", "addressee_id": None, "evidence": "explicit", "register": {"politeness": "plain", "first_person": None}},
        {"utterance_index": 1, "speaker_id": "onna", "addressee_id": None, "evidence": "explicit", "register": {"politeness": "polite", "first_person": None}},
        {"utterance_index": 2, "speaker_id": "otoko", "addressee_id": None, "evidence": "inferred", "register": {"politeness": "plain", "first_person": None}},
        {"utterance_index": 3, "speaker_id": "otoko", "addressee_id": None, "evidence": "inferred", "register": {"politeness": "plain", "first_person": None}},
        {"utterance_index": 4, "speaker_id": "otoko", "addressee_id": None, "evidence": "inferred", "register": {"politeness": "plain", "first_person": None}},
        {"utterance_index": 5, "speaker_id": "otoko", "addressee_id": None, "evidence": "explicit", "register": {"politeness": "plain", "first_person": None}},
        {"utterance_index": 6, "speaker_id": "otoko", "addressee_id": None, "evidence": "inferred", "register": {"politeness": "plain", "first_person": None}},
    ]
    snapshot = json.loads(json.dumps(raw_preds))
    cfg = PostprocessConfig(
        explicit_evidence_gate=True,
        inline_non_speech_filter=True,
        inline_non_speech_max_chars=5,
    )
    out_a, audit_a = apply_postprocessing_to_work(work_a, raw_preds, cfg)
    out_b, audit_b = apply_postprocessing_to_work(work_b, raw_preds, cfg)

    # Zero mutation of original inputs and zero gold-label leakage.
    assert raw_preds == snapshot
    assert out_a == out_b
    assert audit_a == audit_b
    assert out_a is not None

    # U0: pre_speaker_and_verb=True -> keeps explicit.
    assert out_a[0]["evidence"] == "explicit" and out_a[0]["speaker_id"] == "otoko"
    # U1: no speaker+verb -> downgraded to inferred, speaker preserved.
    assert out_a[1]["evidence"] == "inferred" and out_a[1]["speaker_id"] == "onna"
    # U2: already inferred -> unchanged.
    assert out_a[2]["evidence"] == "inferred" and out_a[2]["speaker_id"] == "otoko"
    # U3: inline_embedded, no verb, 2 chars <= 5 -> forced null/ambiguous.
    assert out_a[3]["speaker_id"] is None and out_a[3]["evidence"] == "ambiguous"
    # U4: inline_embedded, no verb, 6 chars > 5 -> untouched.
    assert out_a[4]["speaker_id"] == "otoko" and out_a[4]["evidence"] == "inferred"
    # U5: inline_embedded, 2 chars <= 5, BUT post_speech_verb="言っ" (post_spk=()) -> (B) skips, (A) downgrades!
    assert out_a[5]["speaker_id"] == "otoko" and out_a[5]["evidence"] == "inferred"
    # U6: standalone_line, 1 char <= 5 -> (B) skips because position != inline_embedded.
    assert out_a[6]["speaker_id"] == "otoko" and out_a[6]["evidence"] == "inferred"

    assert [r["utterance_index"] for r in audit_a["inline_non_speech_forced"]] == [3]
    assert [r["utterance_index"] for r in audit_a["explicit_downgraded"]] == [1, 5]

    # None input stays None; missing indices are never fabricated.
    none_out, none_audit = apply_postprocessing_to_work(work_a, None, cfg)
    assert none_out is None
    assert none_audit["inline_non_speech_forced"] == []
    partial_out, _ = apply_postprocessing_to_work(work_a, [raw_preds[1]], cfg)
    assert partial_out is not None and len(partial_out) == 1
    assert partial_out[0]["evidence"] == "inferred"


def test_postprocess_cli_config_toggles_and_threshold(tmp_path: Path) -> None:
    from usir_py.sensor_eval.run_step1_f9 import main

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    w1_text = "「おはよう」\n彼は「自由」を嫌った。\n"
    (gold_dir / "w1.txt").write_text(w1_text, encoding="utf-8")
    s1 = extract_outer_quote_spans(w1_text)
    gold_obj = {
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
                    {"n": 0, "open": s1[0][0], "head": "おはよう", "status": "attributed", "speaker": "otoko"},
                    {"n": 1, "open": s1[1][0], "head": "自由", "status": "not_speech", "speaker": None},
                ],
            }
        ]
    }
    gold_bytes = json.dumps(gold_obj, ensure_ascii=False).encode("utf-8")
    (gold_dir / "gold.json").write_bytes(gold_bytes)
    sha = hashlib.sha256(gold_bytes).hexdigest()
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "split_version": "test_pp_v1",
                "gold_json_sha256": sha,
                "dev": {"work_ids": ["w1"]},
                "holdout": {"sealed": True, "work_ids": ["w1"]},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_pp"
    cfg_path = tmp_path / "cfg_pp.json"
    cfg_path.write_text(
        json.dumps(
            {
                "gold_dir": str(gold_dir),
                "split_file": str(split_path),
                "split": "dev",
                "prompt_version": "v5",
                "out_dir": str(out_dir),
                "postprocess": {
                    "explicit_evidence_gate": True,
                    "inline_non_speech_filter": True,
                    "inline_non_speech_max_chars": 2,
                },
                "models": [{"spec": "m_pp:1b", "num_ctx": 4096, "num_predict": 512}],
            }
        ),
        encoding="utf-8",
    )

    # U0: wrong speaker + explicit (no verb/speaker cue -> Rule A downgrades to inferred -> explicit_den=0).
    # U1: not_speech inline 2 chars -> Rule B forces null/ambiguous -> not_speech_abstained=1/1.
    fake_resp = {
        "message": {
            "content": json.dumps(
                {
                    "utterances": [
                        {
                            "utterance_index": 0,
                            "speaker_id": "onna",
                            "addressee_id": None,
                            "evidence": "explicit",
                            "register": {"politeness": "plain", "first_person": None},
                        },
                        {
                            "utterance_index": 1,
                            "speaker_id": "otoko",
                            "addressee_id": None,
                            "evidence": "inferred",
                            "register": {"politeness": "unknown", "first_person": None},
                        },
                    ]
                }
            )
        },
        "prompt_eval_count": 40,
        "eval_count": 20,
    }
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=fake_resp):
        rc = main(["--config", str(cfg_path), "--check-gate"])
    assert rc == 2  # explicit_denominator == 0 and overall_error_rate == 1.0

    report = json.loads((out_dir / "m_pp_1b_dev.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary_dev.json").read_text(encoding="utf-8"))
    assert report["postprocess"] == {
        "explicit_evidence_gate": True,
        "inline_non_speech_filter": True,
        "inline_non_speech_max_chars": 2,
    }
    assert report["postprocess_summary"] == {
        "inline_non_speech_forced_count": 1,
        "inline_non_speech_changed_count": 1,
        "explicit_downgraded_count": 1,
    }
    assert summary["models"][0]["postprocess"] == report["postprocess"]
    assert summary["models"][0]["postprocess_summary"] == report["postprocess_summary"]
    raw_w1 = report["raw_outputs"]["w1"]
    assert raw_w1["stage1_parsed_utterances"][0]["evidence"] == "explicit"
    assert raw_w1["parsed_utterances"][0]["evidence"] == "inferred"
    assert raw_w1["parsed_utterances"][1]["speaker_id"] is None
    assert raw_w1["parsed_utterances"][1]["evidence"] == "ambiguous"
    assert report["metrics"]["overall"]["not_speech_abstained"] == 1

    # CLI flags override config: disable Rule (A), lower Rule (B) max_chars to 1.
    out_dir_override = tmp_path / "out_pp_override"
    with patch("usir_py.sensor_eval.ollama_client._post_chat", return_value=fake_resp):
        rc2 = main(
            [
                "--config",
                str(cfg_path),
                "--out-dir",
                str(out_dir_override),
                "--no-explicit-evidence-gate",
                "--inline-non-speech-max-chars",
                "1",
            ]
        )
    assert rc2 == 0
    rep2 = json.loads((out_dir_override / "m_pp_1b_dev.json").read_text(encoding="utf-8"))
    assert rep2["metrics"]["overall"]["explicit_errors"] == 1
    assert rep2["metrics"]["overall"]["not_speech_abstained"] == 0

    # Verify offline replay helper agrees with live pipeline output.
    ds = load_f9_dataset(gold_dir, split_path, split="dev")
    replayed = replay_saved_result_with_postprocess(
        ds, rep2, PostprocessConfig.from_dict(report["postprocess"])
    )
    assert replayed["metrics"]["overall"] == report["metrics"]["overall"]



