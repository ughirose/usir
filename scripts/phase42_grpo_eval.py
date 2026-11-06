#!/usr/bin/env python
"""Phase 42: evaluate GRPO with the tail-stripped shaped reward against the criteria fixed in
docs/handoff/2026-11-06_phase42_preregistration.md (this script was written before the run was launched).

Criteria: (1) mean reward of the last 10 steps >= first 10 steps + 0.02; (2) merged-LoRA QDA Deadzone on
tail-stripped text < base (n = 12 each); (3) merged LoRA: no degenerate output (bigram < 0.5) and mean ref PPL <=
base + 0.5.  Report only (stored rollouts): share of rollouts ending inside a sentence, share without any sentence
end, position of the last sentence end, correlation of reward with CV_L and with cadence_alive, cadence_alive
distribution, and the reward each rollout would have received without tail stripping.

Output: docs/handoff/v10/p42_grpo_eval.json
Usage:  cd python && uv run python ../scripts/phase42_grpo_eval.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from usir_py.extractor_24d import extract_usir_24d  # noqa: E402
from usir_py.lora_grpo_forge import strip_incomplete_tail  # noqa: E402
from usir_py.manifold_geodesic import compute_three_phase_qda_distance  # noqa: E402

V10 = ROOT / "docs" / "handoff" / "v10"
SRC = V10 / "p42g_q3_8b.json"
OUT = V10 / "p42_grpo_eval.json"
ENDS = "。！？!?"


def dz(t: str) -> bool:
    return compute_three_phase_qda_distance(t, domain="novel")["phase_regime"] == "Phase_I_Deadzone"


def shaped_unstripped(text: str, ppl_ref: float, thr: float, alpha: float) -> float:
    """Same formula as run_phase33_phase_c.make_shaped_reward with strip_tail=False."""
    if not text or not text.strip():
        return 0.0
    f = extract_usir_24d(text)
    arg = float(np.clip(alpha * (float(ppl_ref) - float(thr)), -50.0, 50.0))
    return float(np.clip(float(f["len_cv"]) / 0.45, 0.05, 1.2) * float(f["cadence_alive"]) / (1.0 + math.exp(arg)))


def corr(a: list, b: list):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    d = json.loads(Path(args.src).read_text(encoding="utf-8"))
    assert not d["provenance"]["mock_used"], "mock"
    g = d["stages"]["grpo_lora"]
    steps = g["train"]["result"]["steps"]
    base, lm = d["stages"]["base"]["records"], g["lora_merged"]["records"]
    assert len(base) == len(lm) == 12
    first, last = st.mean(x["mean_reward"] for x in steps[:10]), st.mean(x["mean_reward"] for x in steps[-10:])
    sets = {}
    for name, recs in (("base", base), ("lora_merged", lm)):
        sets[name] = {"deadzone_stripped": sum(dz(strip_incomplete_tail(r["text"])) for r in recs),
                      "deadzone_full": sum(dz(r["text"]) for r in recs),
                      "degenerate": sum(r["metrics"]["distinct_char_bigram"] < 0.5 for r in recs),
                      "ref_ppl_mean": st.mean(r["ref_ppl_completion"] for r in recs),
                      "truncated": sum(r["text"].rstrip()[-1:] not in ENDS for r in recs),
                      "cv_mean_stripped": st.mean(extract_usir_24d(strip_incomplete_tail(r["text"]))["len_cv"] for r in recs)}
    crit = {"c1_reward_rise": last - first, "c1_met": last - first >= 0.02,
            "c2_met": sets["lora_merged"]["deadzone_stripped"] < sets["base"]["deadzone_stripped"],
            "c3_met": sets["lora_merged"]["degenerate"] == 0
            and sets["lora_merged"]["ref_ppl_mean"] <= sets["base"]["ref_ppl_mean"] + 0.5}
    crit["all_met"] = crit["c1_met"] and crit["c2_met"] and crit["c3_met"]
    ro = [dict(r, step=x["step"]) for x in steps for r in x.get("rollouts", [])]
    rep = {"n_rollouts": len(ro)}
    if ro:
        thr, alpha = g["ppl_barrier"]["threshold"], d["config"]["alpha_ppl"]
        cvs, cads, rew, unst, endpos = [], [], [], [], []
        for r in ro:
            t = r["text"]
            s = strip_incomplete_tail(t)
            f = extract_usir_24d(s) if s.strip() else {"len_cv": 0.0, "cadence_alive": 0.0}
            cvs.append(float(f["len_cv"]))
            cads.append(float(f["cadence_alive"]))
            rew.append(r["reward"])
            unst.append(shaped_unstripped(t, r["ppl_ref"], thr, alpha))
            last_end = max((t.rfind(c) for c in ENDS), default=-1)
            endpos.append((last_end + 1) / len(t) if t else 0.0)
        half = len(ro) // 2

        def part(ix):
            return {"ends_inside_sentence": sum(ro[i]["text"].rstrip()[-1:] not in ENDS for i in ix) / len(ix),
                    "no_sentence_end": sum(not any(c in ro[i]["text"] for c in ENDS) for i in ix) / len(ix),
                    "last_end_position_mean": st.mean(endpos[i] for i in ix),
                    "reward_mean": st.mean(rew[i] for i in ix), "reward_unstripped_mean": st.mean(unst[i] for i in ix)}
        rep.update({"first_half": part(range(half)), "second_half": part(range(half, len(ro))),
                    "corr_reward_cv": corr(rew, cvs), "corr_reward_cadence": corr(rew, cads),
                    "cadence_alive_hist": {"lt_0.26": sum(c < 0.26 for c in cads),
                                           "0.26_0.65": sum(0.26 <= c <= 0.65 for c in cads),
                                           "gt_0.65": sum(c > 0.65 for c in cads)},
                    "corr_reward_stripped_vs_unstripped": corr(rew, unst)})
    # post-run additions (report only, after the reviewer's note): noise of the 10-step means
    mr = np.array([x["mean_reward"] for x in steps], float)
    rng = np.random.default_rng(0)
    obs = mr[-10:].mean() - mr[:10].mean()
    perm = [(lambda q: q[-10:].mean() - q[:10].mean())(rng.permutation(mr)) for _ in range(20000)]
    noise = {"step_sd": float(mr.std(ddof=1)), "perm_p_last10_minus_first10": float(np.mean([x >= obs for x in perm])),
             "first20": float(mr[:20].mean()), "last20": float(mr[20:].mean())}
    if ro:
        P_ = sorted({r["prompt_idx"] for r in ro})
        X = np.array([[r["step"]] + [1.0 if r["prompt_idx"] == k else 0.0 for k in P_] for r in ro], float)
        y = np.array([r["reward"] for r in ro], float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        s2 = float(res @ res) / (len(y) - X.shape[1])
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]))
        noise.update({"slope_per_step_prompt_fe": float(beta[0]), "slope_se": se,
                      "slope_ci95": [float(beta[0] - 1.96 * se), float(beta[0] + 1.96 * se)],
                      "note": "OLS on rollout rewards with prompt fixed effects; rollouts within a step are not independent"})
    out = {"schema": "p42_grpo_eval_v1", "noise": noise, "reward_first10": first, "reward_last10": last, "sets": sets,
           "criteria": crit, "rollouts": rep}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
