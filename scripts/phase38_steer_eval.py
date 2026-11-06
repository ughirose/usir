#!/usr/bin/env python
"""Phase 38-1: evaluate the re-run of Phase C steering (max_new_tokens 1024, 3 seed series) against the criteria
fixed in docs/handoff/2026-11-06_phase38_preregistration.md before the runs.

Primary metric: QDA Deadzone (Phase_I_Deadzone) recomputed on each text with the fragment after the last sentence end
removed (strip_incomplete_tail), pooled over 3 series (n = 36 per condition).
Criteria: (1) pooled calibrated a0.12 < random a0.12 and < base; (2) calibrated < random in >= 2 of 3 series;
(3) no degenerate calibrated output (distinct char bigram < 0.5); (4) mean ref PPL of calibrated <= base + 0.5.
Report only: alpha 0.08, CV_L < 0.28, one-sided exact McNemar, truncation counts, latency.

Output: docs/handoff/v10/p38_steer_eval.json
Usage:  cd python && uv run python ../scripts/phase38_steer_eval.py [--prefix p39c_q3_8b --out p39_steer_eval.json]
        (Phase 39 uses the same script and criteria on the literary prompt set)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from usir_py.extractor_24d import extract_usir_24d  # noqa: E402
from usir_py.lora_grpo_forge import strip_incomplete_tail  # noqa: E402
from usir_py.manifold_geodesic import compute_three_phase_qda_distance  # noqa: E402

V10 = ROOT / "docs" / "handoff" / "v10"
OUT = V10 / "p38_steer_eval.json"
SERIES = ("s1", "s2", "s3")
ENDS = "。！？!?"
CONDS = ("base", "steer_cal_last_alpha_0.08", "steer_rand_last_alpha_0.08",
         "steer_cal_last_alpha_0.12", "steer_rand_last_alpha_0.12")
CAL, RAND = "steer_cal_last_alpha_0.12", "steer_rand_last_alpha_0.12"


def dz(text: str) -> bool:
    return compute_three_phase_qda_distance(text, domain="novel")["phase_regime"] == "Phase_I_Deadzone"


def mcnemar_one_sided(x: list, y: list) -> dict:
    """H1: x is in the Deadzone less often than y (paired). b = y only, c = x only; p = P(B >= b | b + c, 1/2)."""
    b = sum(1 for a, o in zip(x, y) if o and not a)
    c = sum(1 for a, o in zip(x, y) if a and not o)
    n = b + c
    p = sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n if n else 1.0
    return {"only_other": b, "only_calibrated": c, "p_one_sided": p}


def records(d: dict, cond: str) -> list:
    if cond == "base":
        return d["stages"]["base"]["records"]
    return d["stages"]["steering"]["conditions"][cond]["records"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="p38c_q3_8b")
    ap.add_argument("--out", default=OUT.name)
    args = ap.parse_args()
    out = {"schema": "p38_steer_eval_v1", "prefix": args.prefix, "criteria_doc": "docs/handoff/2026-11-06_phase38_preregistration.md",
           "series": {}, "pooled": {}}
    flags = {c: [] for c in CONDS}
    for s in SERIES:
        d = json.loads((V10 / f"{args.prefix}_{s}.json").read_text(encoding="utf-8"))
        assert d["config"]["max_new_tokens"] == 1024 and not d["provenance"]["mock_used"] and not d["errors"], s
        ser = {"seed": d["config"]["seed"], "prompt_set": d["config"].get("prompt_set", "expository"), "total_wall_sec": d["total_wall_sec"], "conditions": {}}
        keys = None
        for c in CONDS:
            recs = sorted(records(d, c), key=lambda r: r["prompt_idx"])
            assert all(not r.get("error") for r in recs) and len(recs) == 12, (s, c)
            k = [(r["prompt_idx"], r["seed"]) for r in recs]
            assert keys is None or k == keys, (s, c)
            keys = k
            f = [dz(strip_incomplete_tail(r["text"])) for r in recs]
            flags[c] += f
            ser["conditions"][c] = {
                "deadzone_stripped": sum(f),
                "deadzone_full": sum(dz(r["text"]) for r in recs),
                "deadzone_json": sum(bool(r["metrics"]["in_deadzone"]) for r in recs),
                "cv_lt_028_stripped": sum(extract_usir_24d(strip_incomplete_tail(r["text"]))["len_cv"] < 0.28 for r in recs),
                "truncated": sum(r["text"].rstrip()[-1:] not in ENDS for r in recs),
                "hit_token_limit": sum(r["n_new_tokens"] >= 1024 for r in recs),
                "degenerate": sum(r["metrics"]["distinct_char_bigram"] < 0.5 for r in recs),
                "min_bigram": min(r["metrics"]["distinct_char_bigram"] for r in recs),
                "ref_ppl_mean": st.mean(r["ref_ppl_completion"] for r in recs),
                "n_new_tokens_mean": st.mean(r["n_new_tokens"] for r in recs),
                "ms_per_token_mean": st.mean(r["ms_per_token"] for r in recs),
            }
        out["series"][s] = ser
    P = out["pooled"]
    for c in CONDS:
        v = [out["series"][s]["conditions"][c] for s in SERIES]
        P[c] = {k: sum(x[k] for x in v) for k in ("deadzone_stripped", "deadzone_full", "deadzone_json",
                                                   "cv_lt_028_stripped", "truncated", "hit_token_limit", "degenerate")}
        P[c]["ref_ppl_mean"] = st.mean(x["ref_ppl_mean"] for x in v)  # equal n per series
        P[c]["n"] = 36
    base_ppl = P["base"]["ref_ppl_mean"]
    per_series = [out["series"][s]["conditions"][CAL]["deadzone_stripped"] < out["series"][s]["conditions"][RAND]["deadzone_stripped"]
                  for s in SERIES]
    deg = sum(P[c]["degenerate"] for c in CONDS if "cal" in c)
    out["criteria"] = {
        "c1_pooled_cal_lt_rand_and_base": P[CAL]["deadzone_stripped"] < P[RAND]["deadzone_stripped"]
        and P[CAL]["deadzone_stripped"] < P["base"]["deadzone_stripped"],
        "c2_series_cal_lt_rand": sum(per_series), "c2_met": sum(per_series) >= 2,
        "c3_degenerate_calibrated": deg, "c3_met": deg == 0,
        "c4_ref_ppl_cal": P[CAL]["ref_ppl_mean"], "c4_limit": base_ppl + 0.5, "c4_met": P[CAL]["ref_ppl_mean"] <= base_ppl + 0.5,
    }
    cr = out["criteria"]
    cr["all_met"] = cr["c1_pooled_cal_lt_rand_and_base"] and cr["c2_met"] and cr["c3_met"] and cr["c4_met"]
    out["mcnemar"] = {"cal012_vs_rand012": mcnemar_one_sided(flags[CAL], flags[RAND]),
                      "cal012_vs_base": mcnemar_one_sided(flags[CAL], flags["base"]),
                      "cal008_vs_rand008": mcnemar_one_sided(flags["steer_cal_last_alpha_0.08"], flags["steer_rand_last_alpha_0.08"])}
    out_path = V10 / args.out
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in CONDS:
        print(c, P[c], [out["series"][s]["conditions"][c]["deadzone_stripped"] for s in SERIES])
    print(json.dumps(out["criteria"], indent=1))
    print(json.dumps(out["mcnemar"], indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
