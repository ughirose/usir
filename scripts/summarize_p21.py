#!/usr/bin/env python
"""Phase 21: summarize docs/handoff/v7/p21_*.json into docs/handoff/v7/p21_summary.json and a Markdown report
with per-run statistics, condition means, and every full text.

Gate (user-specified): distinct char bigram >= 0.55, CV_L >= 0.40, polite/meta/paren/fragment/kanji-adj 連用止め = 0.
"""
from __future__ import annotations

import ast
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "docs/handoff/v7"


def _gate(r: dict) -> bool:
    reg = r.get("register") or {}
    return (r.get("distinct_char_bigram", 0) >= 0.55 and r.get("cv_l", 0) >= 0.40
            and not any(reg.get(k, 0) for k in ("polite", "meta", "paren"))
            and r.get("fragment_hits", 0) == 0 and r.get("kanji_iadj_renyoutome", 0) == 0)


def main() -> None:
    rows, texts = [], []
    for f in sorted(V7.glob("p21_*.json")):
        if f.name == "p21_summary.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for key, r in d["runs"].items():
            if r.get("error"):
                rows.append({"file": f.stem, "run": key, "model": r.get("model", d.get("generator")), "error": r["error"]})
                continue
            j, jr = r.get("judge_llm") or {}, r.get("judge_rubric") or {}
            row = {
                "file": f.stem, "run": key, "model": r["model"].split("/")[-1], "condition": r["condition"], "seed": r["seed"],
                "chars": r["chars"], "cv_l": r["cv_l"], "bigram": r["distinct_char_bigram"],
                "polite": r["register"]["polite"], "meta": r["register"]["meta"], "paren": r["register"]["paren"],
                "fragment": r["fragment_hits"], "adj_renyo": r["kanji_iadj_renyoutome"],
                "demo_max_para": max(r["demonstratives_total_per_paragraph"]),
                "dll_AB": r["delta_ll_AB"], "ll_A": r["ll_A"], "ll_B": r["ll_B"],
                "judge_llm": j.get("normalized_quality"), "judge_rubric": jr.get("normalized_quality"),
                "judge_source": r.get("judge_source"), "gate": _gate(r), "sec": r.get("elapsed_sec"),
            }
            rows.append(row)
            paras = r["paragraph_texts"]
            paras = ast.literal_eval(paras) if isinstance(paras, str) else paras
            beats = r["paragraph_beats"]
            beats = ast.literal_eval(beats) if isinstance(beats, str) else beats
            texts.append((row, paras, beats))
    ok = [r for r in rows if "error" not in r]
    groups: dict = {}
    for r in ok:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    agg = []
    for (m, c), g in sorted(groups.items()):
        mean = lambda k: round(st.mean(x[k] for x in g if x[k] is not None), 3) if any(x[k] is not None for x in g) else None  # noqa: E731
        agg.append({"model": m, "condition": c, "n": len(g), **{k: mean(k) for k in
                    ("chars", "cv_l", "bigram", "dll_AB", "judge_llm", "judge_rubric", "demo_max_para")},
                    "gate_pass": f"{sum(x['gate'] for x in g)}/{len(g)}"})
    (V7 / "p21_summary.json").write_text(json.dumps({"runs": rows, "by_condition": agg}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    L = ["# Phase 21 プロット駆動短編 生成結果 (2026-11-05, 記入者: Claude Code)", "",
         "条件: ctrl=ステアリングなし / A=芥川ゲノム(000879, α=0.75) / B=太宰ゲノム(000035, α=0.75) / A_legacy=芥川ゲノム(0.2S+0.8I)。",
         "ΔLL_AB = 生成文の窓平均対数尤度 (A − B)。A でステアした文が A に近づくなら A の ΔLL が ctrl・B より大きい。",
         "Judge: Qwen2.5-7B-Instruct（3B 出力は交差採点、7B 出力は自己採点でバイアスあり）。rubric はオフライン規則採点。", "",
         "## 条件別平均", "", "| model | condition | n | chars | CV_L | bigram | ΔLL_AB | judge_llm | judge_rubric | 指示語最大/段落 | gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in agg:
        L.append(f"| {a['model']} | {a['condition']} | {a['n']} | {a['chars']} | {a['cv_l']} | {a['bigram']} | {a['dll_AB']} | "
                 f"{a['judge_llm']} | {a['judge_rubric']} | {a['demo_max_para']} | {a['gate_pass']} |")
    L += ["", "## 全実行", "", "| run | model | chars | CV_L | bigram | 敬/メ/括 | 断片 | 連用止め | ΔLL_AB | judge_llm | rubric | gate | sec |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['file']}:{r['run']} | {r['model']} | ERROR: {str(r['error'])[:80]} |||||||||||")
            continue
        L.append(f"| {r['run']} | {r['model']} | {r['chars']} | {r['cv_l']:.3f} | {r['bigram']:.3f} | {r['polite']}/{r['meta']}/{r['paren']} | "
                 f"{r['fragment']} | {r['adj_renyo']} | {r['dll_AB']:.2f} | {r['judge_llm']} | {r['judge_rubric']} | {'○' if r['gate'] else '×'} | {r['sec']} |")
    L += ["", "## 全文", ""]
    for row, paras, beats in texts:
        L += [f"### {row['model']} / {row['condition']} / seed {row['seed']}", ""]
        for k, p in enumerate(paras):
            b = beats[k] if k < len(beats) else ""
            L += [f"*［{b}］*", "", p.strip(), ""]
    (V7 / "p21_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for a in agg:
        print(a)


if __name__ == "__main__":
    main()
