#!/usr/bin/env python3
"""P-2c Task D: recalibrate the 0.9 SD auxiliary threshold with negative (no-revision) pairs. Parameters = STATUS_p2c.md 登録 D (frozen).
N1 = v1.1 Primary text_before + typo-scale edits (synthetic negatives); N0 = identical pairs (sanity). Positives = P-1 proxy labels (not ground truth).
Raw text stays in memory; only numbers are written."""
from __future__ import annotations
import difflib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))
import run_stage_b13_analysis as b13  # noqa: E402

EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")
DS = REPO / "data/usir_v5_eval/real_revision_pairs_v1_1.jsonl"
C216 = REPO / "docs/handoff/v10/paragraph_noise_floor_all_216_canonical.json"
OUT = Path(__file__).resolve().parent / "p2c_taskD_results.json"
THRS = [0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25]

rows = [json.loads(line) for line in DS.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 439
prim = [r for r in rows if r["same_orthography"]]
assert len(prim) == 405


def abs_max_of(rs):
    recs = b13._run_rust_harvest_batch(EXE, C216, rs)
    out = []
    for rec in recs:
        pd = rec["noise_floor_ratio_kasane11"].get("per_dim") or {}
        v = [abs(float(x)) for x in pd.values() if x is not None]
        out.append(max(v) if v else np.nan)
    return np.array(out)


def edit_text(t, k, rng):
    s = list(t)
    for _ in range(k):
        cand = [i for i, ch in enumerate(s) if ch != "\n"]
        if len(cand) < 3:
            break
        i = int(cand[rng.integers(0, len(cand))])
        kind = ("delete", "duplicate", "swap")[int(rng.integers(0, 3))]
        if kind == "delete":
            del s[i]
        elif kind == "duplicate":
            s.insert(i, s[i])
        elif i + 1 < len(s) and s[i + 1] != "\n":
            s[i], s[i + 1] = s[i + 1], s[i]
        else:
            del s[i]
    return "".join(s)


def ccr_of(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = ra = rb = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            ins += j2 - j1
        elif tag == "delete":
            dele += i2 - i1
        else:
            ra += i2 - i1
            rb += j2 - j1
    return (ins + dele + max(ra, rb)) / max(len(a), len(b), 1)


def auc(pos, neg):
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# positives: all 439 real pairs, proxy labels as in P-1
am_real = abs_max_of(rows)
ccr = np.array([ccr_of(r["text_before"], r["text_after"]) for r in rows])
sub = ccr >= np.median(ccr)
# negatives
rng = np.random.default_rng(20261717)
n1_rows, n1_k = [], []
for i, r in enumerate(prim):
    k = 1 + (i % 3)
    n1_rows.append({"text_before": r["text_before"], "text_after": edit_text(r["text_before"], k, rng), "pair_id": f"N1_{i}"})
    n1_k.append(k)
n0_rows = [{"text_before": r["text_before"], "text_after": r["text_before"], "pair_id": f"N0_{i}"} for i, r in enumerate(prim)]
am_n1 = abs_max_of(n1_rows)
try:
    am_n0 = abs_max_of(n0_rows)
    n0_note = "records emitted"
except AssertionError as exc:  # harvest-diff emits no record for identical (unchanged) pairs
    am_n0 = np.full(len(n0_rows), np.nan)
    n0_note = f"harvest-diff emits no record for identical pairs ({exc}); identical pairs are treated as unchanged and never flagged"
n1_k = np.array(n1_k)
band = np.array([r["band"] for r in prim])
n1_ccr = np.array([ccr_of(a["text_before"], a["text_after"]) for a in n1_rows])
valid1 = ~np.isnan(am_n1)

res = {"n_real_pairs": len(rows), "n_proxy_substantive": int(sub.sum()), "n_N1": len(prim), "n_N1_aligned_valid": int(valid1.sum()),
       "n_N0": len(prim), "N0_note": n0_note, "N0_n_nan": int(np.isnan(am_n0).sum()),
       "N1_edit_scale": {"char_change_ratio_median": float(np.median(n1_ccr)), "max": float(n1_ccr.max())},
       "caveat": "N1 is synthetic (typo-scale edits), not real no-revision pairs; positives are proxy labels (char_change_ratio >= median), not ground truth."}
res["N1_abs_max_quantiles"] = {str(q): float(np.nanpercentile(am_n1, q)) for q in (50, 75, 90, 95, 99, 100)}
res["N1_abs_max_by_k"] = {str(k): {"n": int((n1_k == k).sum()), "median": float(np.nanmedian(am_n1[n1_k == k])),
                                   "p95": float(np.nanpercentile(am_n1[n1_k == k], 95))} for k in (1, 2, 3)}
res["N1_abs_max_by_band"] = {b: {"n": int((band == b).sum()), "median": float(np.nanmedian(am_n1[band == b])), "p95": float(np.nanpercentile(am_n1[band == b], 95)),
                                 "flag_rate_at_1.0": float((am_n1[(band == b) & valid1] > 1.0).mean()) if (band == b).any() else None} for b in sorted(set(band))}
res["real_abs_max_quantiles_proxy_substantive"] = {str(q): float(np.nanpercentile(am_real[sub], q)) for q in (5, 25, 50, 75)}
res["auc_proxy_substantive_vs_N1"] = auc(am_real[sub], am_n1)
res["auc_proxy_substantive_vs_proxy_nonsubstantive"] = auc(am_real[sub], am_real[~sub])
tab = {}
for t in THRS:
    ok = ~np.isnan(am_real)
    tab[str(t)] = {"sensitivity_proxy": float((am_real[sub & ok] > t).mean()), "n_flag_real": int((am_real[ok] > t).sum()),
                   "N1_flag_rate": float((am_n1[valid1] > t).mean()), "N1_specificity": float(1 - (am_n1[valid1] > t).mean()),
                   "proxy_nonsubstantive_flag_rate": float((am_real[(~sub) & ok] > t).mean())}
res["threshold_table"] = tab
# specificity-targeted thresholds from N1 empirical distribution
res["N1_threshold_for_specificity"] = {str(s): float(np.nanpercentile(am_n1, 100 * s)) for s in (0.90, 0.95, 0.99)}
res["sensitivity_at_N1_specificity"] = {str(s): float((am_real[sub & ~np.isnan(am_real)] > np.nanpercentile(am_n1, 100 * s)).mean()) for s in (0.90, 0.95, 0.99)}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: res[k] for k in ("n_proxy_substantive", "n_N1_aligned_valid", "N0_note", "N0_n_nan", "N1_abs_max_quantiles", "N1_abs_max_by_k", "N1_abs_max_by_band",
                                       "auc_proxy_substantive_vs_N1", "auc_proxy_substantive_vs_proxy_nonsubstantive", "N1_threshold_for_specificity", "sensitivity_at_N1_specificity")}, indent=1, ensure_ascii=False))
for t, v in tab.items():
    print(t, {k: round(x, 3) for k, x in v.items()})
