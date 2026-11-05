"""evaluate_llm_judge real-model wiring (M15.3), tested with mock judge callables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))

from run_v4_bidirectional_experiment import evaluate_llm_judge, parse_judge_scores  # noqa: E402

TEXT = "雨上がりの古寺に朝が来た。苔の匂いが濃くなり、鐘の音が遠くへ消えた。しかし庭には誰もいなかった。"
GOOD = "COHERENCE: 4.0\nLOGICAL_VALIDITY: 3.5\nSOPHISTICATION: 4.5\nCLEANLINESS: 5.0\n"


def _run(fn=None):
    return evaluate_llm_judge(TEXT, "novel", "雨上がりの古寺", judge_fn=fn)


def test_default_is_offline_only():
    r = _run()
    assert r["judge_source"] == "offline_rubric"
    assert r["judge_llm"] is None and r["judge_llm_raw"] is None
    assert r["judge_rubric"]["normalized_quality"] == r["normalized_quality"]


def test_llm_scores_recorded_with_rubric():
    base = _run()
    r = _run(lambda p: GOOD)
    assert r["judge_source"] == "llm"
    assert r["judge_llm"]["total_score"] == 17.0
    assert r["judge_llm"]["normalized_quality"] == 81.2
    assert r["normalized_quality"] == 81.2
    assert r["judge_rubric"] == base["judge_rubric"]
    assert r["judge_llm_raw"] == GOOD


def test_prompt_passed_to_judge():
    seen = []
    _run(lambda p: seen.append(p) or GOOD)
    assert seen and "COHERENCE" in seen[0] and TEXT in seen[0]


def test_clamp_and_formats():
    p = parse_judge_scores("COHERENCE: 9\nLOGICAL_VALIDITY：0.2\nSOPHISTICATION = 4/5\nCLEANLINESS: **3.5**")
    assert p == {"coherence_flow": 5.0, "logical_validity": 1.0,
                 "structural_sophistication": 4.0, "absence_of_slop": 3.5}


def test_malformed_replies_fall_back():
    base = _run()
    for bad in ["", "   ", "I cannot grade this.", "COHERENCE: high\nLOGICAL_VALIDITY: 3",
                "COHERENCE: 4\nLOGICAL_VALIDITY: 4\nSOPHISTICATION: 4", None, 123]:
        assert parse_judge_scores(bad) is None
        r = _run(lambda p, b=bad: b)
        assert r["judge_llm"] is None
        assert r["judge_source"] == "offline_rubric"
        assert r["normalized_quality"] == base["normalized_quality"]


def test_judge_exception_is_contained():
    def boom(p):
        raise RuntimeError("x")

    r = _run(boom)
    assert r["judge_llm"] is None and r["judge_llm_raw"].startswith("<error")
