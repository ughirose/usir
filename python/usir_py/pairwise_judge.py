"""Phase 28.1: blind pairwise LLM judge with order swap.

Pure Python + numpy. No torch import. A judge_fn is Callable[[str], str].
Scores are signed ints in [-2, 2] from the perspective of the first text.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_AXES: Tuple[str, ...] = (
    "文長リズムの緩急と呼吸",
    "表現の自然さ（造語・破綻のなさ）",
    "情景の鮮明さと余韻",
    "クライマックスの迫力",
    "総合的な文学的魅力",
)

_CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def build_pairwise_prompt(text_1: str, text_2: str, axes: Optional[Sequence[str]] = None) -> str:
    axes = list(axes) if axes is not None else list(DEFAULT_AXES)
    axis_lines = "\n".join(
        f"軸{i + 1}（{_CIRCLED[i] if i < len(_CIRCLED) else i + 1}）: {a}" for i, a in enumerate(axes)
    )
    fmt_lines = "\n".join(f"軸{i + 1}: <作品1|作品2|引き分け> <0|1|2>" for i in range(len(axes)))
    return f"""[匿名ペア比較評価]
あなたは公平な第三者の文学評価者です。次の2つの作品は出所不明で匿名です。順序や長さで判断せず、内容のみで比較してください。

作品1:
\"\"\"
{text_1}
\"\"\"

作品2:
\"\"\"
{text_2}
\"\"\"

評価軸:
{axis_lines}

各軸について、優れている方（作品1 / 作品2 / 引き分け）と優劣の差（margin: 0=ほぼ同等, 1=やや優れる, 2=明確に優れる）を答えてください。
出力は以下の形式のみとし、軸ごとに1行、説明文は書かないでください。
{fmt_lines}
例: 軸1: 作品2 2
"""


def _normalize(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF10 <= o <= 0xFF19:
            out.append(chr(o - 0xFF10 + ord("0")))
        elif ch in "：":
            out.append(":")
        else:
            out.append(ch)
    return "".join(out)


_LINE_RE = re.compile(
    r"軸\s*([0-9]+)\s*(?:（[^）]*）)?\s*[:=]\s*\**\s*(作品\s*1|作品\s*2|引き分け|引分け|引分|同等)\s*[,、\s]*\**\s*([0-9])?"
)


def parse_pairwise_reply(reply: Any, n_axes: int) -> Optional[List[int]]:
    """Parse to per-axis signed scores (+: 作品1 wins). None if any axis is missing."""
    if not isinstance(reply, str) or not reply.strip():
        return None
    text = _normalize(reply)
    found: Dict[int, int] = {}
    for m in _LINE_RE.finditer(text):
        idx = int(m.group(1))
        if idx < 1 or idx > n_axes or idx in found:
            continue
        verdict = re.sub(r"\s", "", m.group(2))
        margin = int(m.group(3)) if m.group(3) is not None else 1
        margin = min(max(margin, 0), 2)
        if verdict == "作品1":
            found[idx] = max(margin, 1)
        elif verdict == "作品2":
            found[idx] = -max(margin, 1)
        else:
            found[idx] = 0
    if len(found) < n_axes or any(i not in found for i in range(1, n_axes + 1)):
        return None
    return [found[i] for i in range(1, n_axes + 1)]


def _call(judge_fn: Callable[[str], str], prompt: str) -> Any:
    try:
        return judge_fn(prompt)
    except Exception as exc:  # judge failure must not break evaluation
        return f"<error: {type(exc).__name__}>"


def evaluate_pairwise(
    text_a: str,
    text_b: str,
    judge_fn: Callable[[str], str],
    axes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Judge (A,B) and (B,A); report scores from A's perspective."""
    axes_l = list(axes) if axes is not None else list(DEFAULT_AXES)
    n = len(axes_l)
    reply_ab = _call(judge_fn, build_pairwise_prompt(text_a, text_b, axes_l))
    reply_ba = _call(judge_fn, build_pairwise_prompt(text_b, text_a, axes_l))
    s_ab = parse_pairwise_reply(reply_ab, n)
    s_ba_raw = parse_pairwise_reply(reply_ba, n)
    s_ba = [-v for v in s_ba_raw] if s_ba_raw is not None else None
    failures = int(s_ab is None) + int(s_ba is None)
    used = [s for s in (s_ab, s_ba) if s is not None]

    if used:
        arr = np.array(used, dtype=float)
        per_axis = [float(x) for x in arr.mean(axis=0)]
        overall = float(arr.mean())
    else:
        per_axis = [None] * n  # type: ignore[list-item]
        overall = None

    if s_ab is not None and s_ba is not None:
        consistent = [bool(np.sign(a) == np.sign(b)) for a, b in zip(s_ab, s_ba)]
    else:
        consistent = [None] * n  # type: ignore[list-item]

    return {
        "axes": axes_l,
        "scores_ab": s_ab,
        "scores_ba_as_a": s_ba,
        "per_axis_mean": per_axis,
        "overall_mean": overall,
        "position_consistent": consistent,
        "raw_replies": [reply_ab if isinstance(reply_ab, str) else None,
                        reply_ba if isinstance(reply_ba, str) else None],
        "parse_failures": failures,
    }


def tournament(
    texts: Dict[str, str],
    pairs: Sequence[Tuple[str, str]],
    judge_fn: Callable[[str], str],
    axes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run evaluate_pairwise over pairs; win/loss/tie by sign of overall_mean."""
    stats: Dict[str, Dict[str, Any]] = {
        k: {"wins": 0, "losses": 0, "ties": 0, "score_sum": 0.0, "n": 0} for k in texts
    }
    matches = []
    skipped = 0
    for a, b in pairs:
        r = evaluate_pairwise(texts[a], texts[b], judge_fn, axes)
        matches.append({"a": a, "b": b, **r})
        ov = r["overall_mean"]
        if ov is None:
            skipped += 1
            continue
        if ov > 0:
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
        elif ov < 0:
            stats[b]["wins"] += 1
            stats[a]["losses"] += 1
        else:
            stats[a]["ties"] += 1
            stats[b]["ties"] += 1
        stats[a]["score_sum"] += ov
        stats[b]["score_sum"] -= ov
        stats[a]["n"] += 1
        stats[b]["n"] += 1
    summary = {}
    for k, s in stats.items():
        summary[k] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "ties": s["ties"],
            "mean_score_diff": (s["score_sum"] / s["n"]) if s["n"] else None,
            "n_matches": s["n"],
        }
    return {"summary": summary, "matches": matches, "skipped_matches": skipped}
