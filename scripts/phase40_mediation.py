#!/usr/bin/env python
"""Exploratory (post hoc, after Phase 38-1): which part of the extractor moves when steering lowers the QDA Deadzone
count on expository prompts.

For each output (tail-stripped, as in Phase 38-1) it records the Deadzone flag, ||p||, cadence_alive, the three
conditions of the extractor's flat-slop gate (mean sentence length > 40, CV_L < 0.28, abstract fillers > 0.015 or
sluggish endings > 0.15), and the abstract / sluggish rates.  It then tabulates, per condition, how often each gate
condition holds, and for base→calibrated pairs that leave the Deadzone, which gate conditions changed.

Output: docs/handoff/v10/p40_mediation.json
Usage:  cd python && uv run python ../scripts/phase40_mediation.py [--prefix p38c_q3_8b]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from usir_py.extractor_24d import ABSTRACT_FILLERS, extract_usir_24d  # noqa: E402
from usir_py.lora_grpo_forge import strip_incomplete_tail  # noqa: E402
from usir_py.manifold_geodesic import compute_three_phase_qda_distance  # noqa: E402
from usir_py.orthographic_prior import sentence_lengths  # noqa: E402

V10 = ROOT / "docs" / "handoff" / "v10"
CONDS = ("base", "steer_cal_last_alpha_0.12", "steer_rand_last_alpha_0.12")


def feats(text: str) -> dict:
    t = strip_incomplete_tail(text)
    f = extract_usir_24d(t)
    L = sentence_lengths(t)
    mean_len = float(np.mean(L)) if L else 0.0
    g_len, g_cv = mean_len > 40.0, float(f["len_cv"]) < 0.28
    g_lex = float(f["abstract_b_mean"]) > 0.015 or float(f["sluggish_mean"]) > 0.15
    return {"deadzone": compute_three_phase_qda_distance(t, domain="novel")["phase_regime"] == "Phase_I_Deadzone",
            "p_norm": float(np.linalg.norm(f["p"])), "cadence_alive": float(f["cadence_alive"]),
            "mean_len": mean_len, "len_cv": float(f["len_cv"]), "abstract_b": float(f["abstract_b_mean"]),
            "sluggish": float(f["sluggish_mean"]), "g_len": g_len, "g_cv": g_cv, "g_lex": g_lex,
            "gate_on": g_len and g_cv and g_lex}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="p38c_q3_8b")
    ap.add_argument("--series", nargs="+", default=["s1", "s2", "s3"])
    args = ap.parse_args()
    F = {c: [] for c in CONDS}
    W = {c: {w: 0 for w in ABSTRACT_FILLERS} for c in CONDS}
    for s in args.series:
        d = json.loads((V10 / f"{args.prefix}_{s}.json").read_text(encoding="utf-8"))
        for c in CONDS:
            recs = d["stages"]["base"]["records"] if c == "base" else d["stages"]["steering"]["conditions"][c]["records"]
            F[c] += [feats(r["text"]) for r in sorted(recs, key=lambda r: r["prompt_idx"])]
            for r in recs:
                s_t = strip_incomplete_tail(r["text"])
                for w in ABSTRACT_FILLERS:
                    W[c][w] += s_t.count(w)
    summ = {}
    for c, v in F.items():
        summ[c] = {"n": len(v), "deadzone": sum(x["deadzone"] for x in v), "gate_on": sum(x["gate_on"] for x in v),
                   "deadzone_and_gate_on": sum(x["deadzone"] and x["gate_on"] for x in v)}
        for k in ("g_len", "g_cv", "g_lex"):
            summ[c][k] = sum(x[k] for x in v)
        for k in ("p_norm", "cadence_alive", "mean_len", "len_cv", "abstract_b", "sluggish"):
            summ[c][k + "_mean"] = st.mean(x[k] for x in v)
    exits = {}
    for c in CONDS[1:]:
        pairs = [(b, s) for b, s in zip(F["base"], F[c]) if b["deadzone"] and not s["deadzone"]]
        exits[c] = {"n_exit": len(pairs),
                    "gate_turned_off": sum(b["gate_on"] and not s["gate_on"] for b, s in pairs),
                    "g_len_turned_off": sum(b["g_len"] and not s["g_len"] for b, s in pairs),
                    "g_cv_turned_off": sum(b["g_cv"] and not s["g_cv"] for b, s in pairs),
                    "g_lex_turned_off": sum(b["g_lex"] and not s["g_lex"] for b, s in pairs),
                    "n_enter": sum(1 for b, s in zip(F["base"], F[c]) if not b["deadzone"] and s["deadzone"])}
    out = {"schema": "p40_mediation_v1", "post_hoc": True, "prefix": args.prefix, "summary": summ, "exits_from_base": exits,
           "filler_counts": {c: {"total": sum(v.values()), **{w: n for w, n in v.items() if n}} for c, v in W.items()}}
    (V10 / "p40_mediation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
