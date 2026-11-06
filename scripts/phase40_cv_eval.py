#!/usr/bin/env python
"""Phase 40: sentence-length CV (CV_L) as the primary metric for steering on literary prompts.

Criteria fixed in docs/handoff/2026-11-06_phase40_preregistration.md before the runs. CV_L is the extractor's len_cv
on each text with the fragment after the last sentence end removed (strip_incomplete_tail). Comparisons are paired
by (prompt, seed) within each series and pooled over the series given by --series (Phase 40: 4 series, n = 48 pairs).

Output: docs/handoff/v10/<out>
Usage:  cd python && uv run python ../scripts/phase40_cv_eval.py --prefix p40c_q3_8b --out p40_cv_eval.json
        (--prefix p39c_q3_8b gives the exploratory Phase 39 values that motivated the test)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from usir_py.extractor_24d import extract_usir_24d  # noqa: E402
from usir_py.lora_grpo_forge import strip_incomplete_tail  # noqa: E402

V10 = ROOT / "docs" / "handoff" / "v10"
DEFAULT_SERIES = ("s1", "s2", "s3")
ALPHAS_DEFAULT = ("0.08", "0.12")
CAL, RAND = "steer_cal_last_alpha_0.12", "steer_rand_last_alpha_0.12"


def records(d: dict, cond: str) -> list:
    if cond == "base":
        return d["stages"]["base"]["records"]
    return d["stages"]["steering"]["conditions"][cond]["records"]


def paired(x: list, y: list) -> dict:
    diff = [a - b for a, b in zip(x, y)]
    w = wilcoxon(x, y, alternative="greater", zero_method="wilcox")
    return {"mean_diff": st.mean(diff), "n_positive": sum(v > 0 for v in diff), "n": len(diff),
            "wilcoxon_p_greater": float(w.pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="p40c_q3_8b")
    ap.add_argument("--out", default="p40_cv_eval.json")
    ap.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES))
    ap.add_argument("--alphas", nargs="+", default=list(ALPHAS_DEFAULT), help="first value is the secondary (report-only) alpha")
    args = ap.parse_args()
    SERIES = tuple(args.series)
    a2 = [a for a in args.alphas if a != "0.12"][0]
    CONDS = ("base", f"steer_cal_last_alpha_{a2}", f"steer_rand_last_alpha_{a2}", CAL, RAND)
    cv = {c: [] for c in CONDS}
    out = {"schema": "p40_cv_eval_v1", "prefix": args.prefix, "secondary_alpha": a2, "series": {}}
    for s in SERIES:
        d = json.loads((V10 / f"{args.prefix}_{s}.json").read_text(encoding="utf-8"))
        assert not d["provenance"]["mock_used"] and not d["errors"], s
        ser, keys = {"seed": d["config"]["seed"], "prompt_set": d["config"].get("prompt_set"), "conditions": {}}, None
        for c in CONDS:
            recs = sorted(records(d, c), key=lambda r: r["prompt_idx"])
            k = [(r["prompt_idx"], r["seed"]) for r in recs]
            assert len(recs) == 12 and (keys is None or k == keys), (s, c)
            keys = k
            v = [float(extract_usir_24d(strip_incomplete_tail(r["text"]))["len_cv"]) for r in recs]
            cv[c] += v
            ser["conditions"][c] = {
                "cv_mean": st.mean(v),
                "degenerate": sum(r["metrics"]["distinct_char_bigram"] < 0.5 for r in recs),
                "ref_ppl_mean": st.mean(r["ref_ppl_completion"] for r in recs),
                "truncated": sum(r["text"].rstrip()[-1:] not in "。！？!?」" for r in recs),
            }
        ser["cal_minus_rand_mean"] = ser["conditions"][CAL]["cv_mean"] - ser["conditions"][RAND]["cv_mean"]
        out["series"][s] = ser
    out["pooled"] = {c: {"cv_mean": st.mean(cv[c]), "cv_median": st.median(cv[c]),
                         "degenerate": sum(out["series"][s]["conditions"][c]["degenerate"] for s in SERIES),
                         "ref_ppl_mean": st.mean(out["series"][s]["conditions"][c]["ref_ppl_mean"] for s in SERIES)}
                     for c in CONDS}
    out["paired"] = {"cal012_vs_base": paired(cv[CAL], cv["base"]), "cal012_vs_rand012": paired(cv[CAL], cv[RAND]),
                     "cal_secondary_vs_base": paired(cv[f"steer_cal_last_alpha_{a2}"], cv["base"]),
                     "cal_secondary_vs_rand_secondary": paired(cv[f"steer_cal_last_alpha_{a2}"], cv[f"steer_rand_last_alpha_{a2}"])}
    P, pr = out["pooled"], out["paired"]
    nser = sum(out["series"][s]["cal_minus_rand_mean"] > 0 for s in SERIES)
    need = len(SERIES) // 2 + 1  # strict majority of series
    deg = P[CAL]["degenerate"] + P[f"steer_cal_last_alpha_{a2}"]["degenerate"]
    out["criteria"] = {
        "c1_cal_gt_base": pr["cal012_vs_base"]["mean_diff"] > 0 and pr["cal012_vs_base"]["wilcoxon_p_greater"] < 0.05,
        "c2_cal_gt_rand": pr["cal012_vs_rand012"]["mean_diff"] > 0 and pr["cal012_vs_rand012"]["wilcoxon_p_greater"] < 0.05,
        "c3_series_cal_gt_rand": nser, "c3_needed": need, "c3_met": nser >= need,
        "c4_degenerate_calibrated": deg, "c4_met": deg == 0,
        "c5_ref_ppl_cal": P[CAL]["ref_ppl_mean"], "c5_limit": P["base"]["ref_ppl_mean"] + 0.5,
        "c5_met": P[CAL]["ref_ppl_mean"] <= P["base"]["ref_ppl_mean"] + 0.5,
    }
    c = out["criteria"]
    c["all_met"] = all(c[k] for k in ("c1_cal_gt_base", "c2_cal_gt_rand", "c3_met", "c4_met", "c5_met"))
    (V10 / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in P.items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    print(json.dumps(pr, indent=1))
    print(json.dumps(c, indent=1))
    print("wrote", V10 / args.out)


if __name__ == "__main__":
    main()
