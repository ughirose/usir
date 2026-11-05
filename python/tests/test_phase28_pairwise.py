import re

from usir_py.pairwise_judge import (
    DEFAULT_AXES,
    build_pairwise_prompt,
    evaluate_pairwise,
    parse_pairwise_reply,
    tournament,
)

N = len(DEFAULT_AXES)


def _split(prompt):
    t1 = re.search(r"作品1:\n\"\"\"\n(.*?)\n\"\"\"", prompt, re.S).group(1)
    t2 = re.search(r"作品2:\n\"\"\"\n(.*?)\n\"\"\"", prompt, re.S).group(1)
    return t1, t2


def marker_judge(marker="良"):
    def fn(prompt):
        t1, t2 = _split(prompt)
        w = "作品1" if marker in t1 and marker not in t2 else "作品2" if marker in t2 and marker not in t1 else None
        if w is None:
            return "\n".join(f"軸{i+1}: 引き分け 0" for i in range(N))
        return "\n".join(f"軸{i+1}: {w} 2" for i in range(N))
    return fn


def biased_judge(prompt):
    return "\n".join(f"軸{i+1}: 作品1 2" for i in range(N))


def test_prompt_contains_texts_and_axes():
    p = build_pairwise_prompt("AAA", "BBB", None)
    assert "AAA" in p and "BBB" in p and "引き分け" in p
    for a in DEFAULT_AXES:
        assert a in p
    assert "軸5:" in p


def test_parse_basic_and_fullwidth():
    r = "軸1: 作品2 2\n軸2: 作品1 1\n軸3: 引き分け 0\n軸4：作品１ ２\n軸5: 作品2 0"
    assert parse_pairwise_reply(r, 5) == [-2, 1, 0, 2, -1]


def test_parse_extra_text_and_missing():
    r = "評価します。\n軸1: 作品1 2\n理由...\n軸2: 作品2 1\n"
    assert parse_pairwise_reply(r, 2) == [2, -1]
    assert parse_pairwise_reply(r, 3) is None
    assert parse_pairwise_reply("", 2) is None
    assert parse_pairwise_reply(None, 2) is None


def test_perspective_flip_and_consistency():
    res = evaluate_pairwise("良い文章", "普通の文", marker_judge())
    assert res["parse_failures"] == 0
    assert res["overall_mean"] == 2.0
    assert all(res["position_consistent"])
    res2 = evaluate_pairwise("普通の文", "良い文章", marker_judge())
    assert res2["overall_mean"] == -2.0
    assert res2["scores_ba_as_a"] == [-2] * N


def test_position_bias_cancels():
    res = evaluate_pairwise("x", "y", biased_judge)
    assert res["overall_mean"] == 0.0
    assert res["per_axis_mean"] == [0.0] * N
    assert not any(res["position_consistent"])


def test_parse_failure_excludes_order():
    calls = []

    def fn(prompt):
        calls.append(1)
        if len(calls) == 1:
            return "garbage"
        return "\n".join(f"軸{i+1}: 作品2 1" for i in range(N))  # B,A order: 作品2 = A -> +1... after negation of 作品2 -> A is 作品2
    res = evaluate_pairwise("a", "b", fn)
    assert res["parse_failures"] == 1
    assert res["scores_ab"] is None
    # second order: 作品2 wins => -1 in B-perspective => +1 for A
    assert res["overall_mean"] == 1.0
    assert res["position_consistent"] == [None] * N

    res_all = evaluate_pairwise("a", "b", lambda p: "nope")
    assert res_all["parse_failures"] == 2
    assert res_all["overall_mean"] is None


def test_judge_exception_counts_as_failure():
    def boom(p):
        raise RuntimeError("x")
    res = evaluate_pairwise("a", "b", boom)
    assert res["parse_failures"] == 2


def test_tournament_aggregation():
    texts = {"g": "良い", "m": "普通", "n": "普通2"}
    pairs = [("g", "m"), ("g", "n"), ("m", "n")]
    out = tournament(texts, pairs, marker_judge())
    s = out["summary"]
    assert s["g"]["wins"] == 2 and s["g"]["losses"] == 0
    assert s["m"]["losses"] == 1 and s["m"]["ties"] == 1
    assert s["n"]["losses"] == 1 and s["n"]["ties"] == 1
    assert s["g"]["mean_score_diff"] == 2.0
    assert s["m"]["mean_score_diff"] == -1.0
    assert len(out["matches"]) == 3


def test_tournament_position_bias_all_ties():
    out = tournament({"a": "1", "b": "2"}, [("a", "b")], biased_judge)
    assert out["summary"]["a"]["ties"] == 1 and out["summary"]["b"]["ties"] == 1
