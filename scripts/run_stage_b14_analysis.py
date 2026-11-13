#!/usr/bin/env python
"""USIR v5 Stage B-14: evaluation rule v2 (2026-11-13) -- mixed pairs evaluated in 8D.

Tasks:
  - Task 2/3: re-run nishiki ``harvest-diff`` on real_revision_pairs_v1.jsonl (415) under the F-17 floor with the
              v2 binary, assert Python (``f14_hybrid_noise_floor.evaluate_f14_hybrid_pair``) == Rust on all 415
              (0 discrepancies required), re-run with the pre-v2 binary (rule v1) as the "before" snapshot, and
              write docs/handoff/v10/real_revision_pairs_v1_eval_rule_v2_summary.json with before/after tables.
  - Task 4:   recompute the Set 1 (10,859 adjacent pairs) mixed-pair (1,590) aggregates under v1 / v2 (8D) / 7D
              diagnostic (evidence level; no artifact JSON besides the embedded block).

Frozen artifacts (never overwritten): real_revision_pairs_v1_f17_harvest_summary.json, sparsity_audit_106works_canonical*.json,
paragraph_noise_floor_all_novel.json.

Usage: python scripts/run_stage_b14_analysis.py --exe-v2 <new nishiki.exe> --exe-v1 <pre-v2 nishiki.exe>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_stage_b13_analysis import (  # noqa: E402
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    EXPECTED_F17_SHA256,
    F17_NOISE_FLOOR_PATH,
    OUT_SUMMARY_PATH as B13_SUMMARY_PATH,
    _build_pair_record,
    _quantiles,
    _run_rust_harvest_batch,
)
from usir_py.v5.f14_hybrid_noise_floor import (  # noqa: E402
    KASANE11_DIM_NAMES,
    KASANE8_ACTIVE_DIM_NAMES,
    SHORT_DEAD_DIMS,
    compute_sha256,
    evaluate_f14_hybrid_pair,
    load_f14_noise_floor,
    select_length_band,
)

OUT_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_eval_rule_v2_summary.json"
SET1_TSV = Path(r"C:\dev\nishiki\docs\handoff\evidence\2026-11-13_f18\set1_paragraphs.tsv")
DC7 = tuple(d for d in KASANE11_DIM_NAMES if not d.startswith("ac_"))
FROZEN = [
    B13_SUMMARY_PATH,
    REPO_ROOT / "docs/handoff/v10/sparsity_audit_106works_canonical.json",
    REPO_ROOT / "docs/handoff/v10/sparsity_audit_106works_canonical_v2.json",
    F17_NOISE_FLOOR_PATH,
]


def _med(v: Sequence[float]):
    return float(np.median(v)) if len(v) else None


def _mean(v: Sequence[float]):
    return float(np.mean(v)) if len(v) else None


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def metric_block(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-pair aggregates required by the stage spec (all values are over pairs in the group)."""
    n = len(items)
    f = lambda k: [it[k] for it in items]  # noqa: E731
    return {
        "n_pairs": n,
        "n_evaluated_distribution": {str(k): sum(1 for it in items if it["n_evaluated"] == k) for k in sorted({it["n_evaluated"] for it in items})},
        "abs_median": {"median": _r(_med(f("abs_ratio_median"))), "mean": _r(_mean(f("abs_ratio_median")))},
        "abs_mean": {"median": _r(_med(f("abs_ratio_mean"))), "mean": _r(_mean(f("abs_ratio_mean")))},
        "abs_max": {"median": _r(_med(f("abs_ratio_max"))), "mean": _r(_mean(f("abs_ratio_max")))},
        "frac_above_1sd": {"mean": _r(_mean(f("frac_above_1sd"))), "median": _r(_med(f("frac_above_1sd")))},
        "frac_above_2sd": {"mean": _r(_mean(f("frac_above_2sd"))), "median": _r(_med(f("frac_above_2sd")))},
        "quantiles": {
            "abs_max": _quantiles(f("abs_ratio_max")),
            "abs_mean": _quantiles(f("abs_ratio_mean")),
        },
    }


def groups_of(metrics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mixed = lambda m: m["mixed_window_pair"]  # noqa: E731
    minor = lambda m: m["dl_over_lmax"] < 0.15 and m["dice_similarity"] >= 0.85  # noqa: E731
    g = {
        "overall": metrics,
        "band_S": [m for m in metrics if m["band"] == "S"],
        "band_M": [m for m in metrics if m["band"] == "M"],
        "band_L": [m for m in metrics if m["band"] == "L"],
        "mixed_window_pairs": [m for m in metrics if mixed(m)],
        "non_mixed_pairs": [m for m in metrics if not mixed(m)],
        "non_mixed_both_single_window": [m for m in metrics if max(m["window_counts"]) <= 1],
        "non_mixed_both_multi_window": [m for m in metrics if min(m["window_counts"]) >= 2],
        "minor_band_all": [m for m in metrics if minor(m)],
        "minor_band_mixed": [m for m in metrics if minor(m) and mixed(m)],
        "minor_band_non_mixed": [m for m in metrics if minor(m) and not mixed(m)],
    }
    for b in ("S", "M", "L"):
        g[f"band_{b}_mixed"] = [m for m in metrics if m["band"] == b and mixed(m)]
        g[f"band_{b}_non_mixed"] = [m for m in metrics if m["band"] == b and not mixed(m)]
    # Primary (B-13 task 3): same_orthography == True (381)
    prim = [m for m in metrics if m["same_orthography"]]
    g["primary_same_orthography"] = prim
    g["primary_mixed"] = [m for m in prim if mixed(m)]
    g["primary_non_mixed"] = [m for m in prim if not mixed(m)]
    g["primary_minor_band"] = [m for m in prim if minor(m)]
    g["primary_minor_band_mixed"] = [m for m in prim if minor(m) and mixed(m)]
    return g


def diag7_block(mixed_items: List[Dict[str, Any]], recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mx, fr = [], []
    for it in mixed_items:
        d = recs[it["pair_id"]]["noise_floor_ratio_kasane11"]["diagnostic_7d_no_crest"]
        assert d["n_evaluated"] == 7 and d["diagnostic_only"] is True
        mx.append(float(d["max"]))
        fr.append(d["n_above_1"] / 7.0)
    return {
        "diagnostic_only": True,
        "note": "7D (dc 7 dims, ac_mean_crest and ac_* excluded) is a diagnostic value, NOT canonical. Canonical for mixed pairs is 8D.",
        "n_pairs": len(mx),
        "abs_max": {"median": _r(_med(mx)), "mean": _r(_mean(mx))},
        "frac_above_1sd": {"mean": _r(_mean(fr)), "median": _r(_med(fr))},
    }


def python_rust_parity(rows, recs, nf_data, prov) -> List[Dict[str, Any]]:
    bad = []
    for row, rec in zip(rows, recs):
        py = evaluate_f14_hybrid_pair(row["text_before"], row["text_after"], nf_data, delta_kasane11=rec["delta_kasane11"], provenance=prov)
        rn = rec["noise_floor_ratio_kasane11"]
        errs = []
        if py["window_counts"] != rec["window_counts"]:
            errs.append("window_counts")
        for k_py, k_rs in (("n_evaluated", "n_evaluated"), ("band", "band"), ("ac_sigma_source", "ac_sigma_source"), ("n_eval_rule", "n_eval_rule"), ("eval_rule", "eval_rule")):
            if py[k_py] != rn[k_rs] and not (k_py == "eval_rule" and py[k_py] == rec.get("eval_rule") == rn.get("eval_rule")):
                errs.append(k_py)
        if rec.get("eval_rule") != "v2" or rn.get("eval_rule") != "v2" or py["eval_rule"] != "v2":
            errs.append("eval_rule_value")
        if sorted(py["excluded_structurally_dead_dims"]) != sorted(rn["excluded_structurally_dead_dims"]):
            errs.append("excluded_dims")
        pr = py["ratios"]
        for k in ("max", "mean", "median"):
            if abs(pr[k] - rn[k]) > 1e-12:
                errs.append(k)
        if pr["n_above_1"] != rn["n_above_1"] or pr["argmax"] != rn["argmax"]:
            errs.append("n_above_1/argmax")
        if set(pr["per_dim"]) != set(rn["per_dim"]) or any(abs(pr["per_dim"][d] - rn["per_dim"][d]) > 1e-12 for d in pr["per_dim"]):
            errs.append("per_dim")
        d7r, d7p = rn.get("diagnostic_7d_no_crest"), py["diagnostic_7d_no_crest"]
        if (d7r is None) != (d7p is None):
            errs.append("diag7_presence")
        elif d7r is not None:
            for k in ("max", "mean", "median"):
                if abs(d7p[k] - d7r[k]) > 1e-12:
                    errs.append("diag7_" + k)
            if d7p["n_above_1"] != d7r["n_above_1"] or d7p["n_evaluated"] != d7r["n_evaluated"]:
                errs.append("diag7_counts")
        if bool(row["mixed_window_pair"]) != bool(rec["mixed_window_pair"]) or (d7r is not None) != bool(rec["mixed_window_pair"]):
            errs.append("mixed_flag")
        if errs:
            bad.append({"pair_id": row["pair_id"], "errors": sorted(set(errs))})
    return bad


# ----------------------------- Task 4: Set 1 -----------------------------

def set1_mixed_recompute(nf_data) -> Dict[str, Any]:
    lines = SET1_TSV.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    names = head[5:]
    assert tuple(names) == KASANE11_DIM_NAMES, names
    rows = [l.split("\t") for l in lines[1:]]
    w = np.array([int(r[0]) for r in rows])
    chars = np.array([int(r[3]) for r in rows])
    wins = np.array([int(r[4]) for r in rows])
    X = np.array([[float(x) for x in r[5:]] for r in rows])
    idx = np.where(w[1:] == w[:-1])[0]
    d = X[idx + 1] - X[idx]
    lmax = np.maximum(chars[idx], chars[idx + 1])
    wp, wn = wins[idx], wins[idx + 1]
    mixed = (wp >= 2) != (wn >= 2)
    both_single = (wp <= 1) & (wn <= 1)
    both_multi = (wp >= 2) & (wn >= 2)
    bands = nf_data["length_bands"]["bands"]
    bsel = [select_length_band(int(l), bands) for l in lmax]

    def sig_for(i, use_ac_multi: bool) -> Dict[str, float]:
        k = bsel[i]["kasane11"]
        s = dict(k["per_dim_std"])
        if use_ac_multi:
            for dn, v in (k.get("ac_multi_window_per_dim_std") or {}).items():
                s[dn] = v
        return s

    def eval_pair(i, dims, use_ac_multi):
        s = sig_for(i, use_ac_multi)
        vals = []
        for dn in dims:
            sv = s.get(dn)
            if isinstance(sv, (int, float)) and sv > 0:
                vals.append(abs(d[i, names.index(dn)]) / sv)
        return vals

    def agg(sel_idx, dims, use_ac_multi):
        mx, mn, md, f1, f2, ne, crest_arg = [], [], [], [], [], [], 0
        for i in sel_idx:
            v = eval_pair(i, dims, use_ac_multi)
            mx.append(max(v)); mn.append(sum(v) / len(v)); md.append(statistics.median(v))
            f1.append(sum(1 for x in v if x > 1.0) / len(v)); f2.append(sum(1 for x in v if x >= 2.0) / len(v)); ne.append(len(v))
        return {
            "n_pairs": len(mx),
            "n_evaluated_distribution": {str(k): int(sum(1 for x in ne if x == k)) for k in sorted(set(ne))},
            "abs_median": {"median": _r(_med(md)), "mean": _r(_mean(md))},
            "abs_mean": {"median": _r(_med(mn)), "mean": _r(_mean(mn))},
            "abs_max": {"median": _r(_med(mx)), "mean": _r(_mean(mx))},
            "frac_above_1sd": {"mean": _r(_mean(f1))},
            "frac_above_2sd": {"mean": _r(_mean(f2))},
        }

    mixed_idx = list(np.where(mixed)[0])
    out: Dict[str, Any] = {
        "source": "C:/dev/nishiki/docs/handoff/evidence/2026-11-13_f18/set1_paragraphs.tsv (F-18 evidence; same adjacent-pair construction as paragraph_noise)",
        "sigma": "F-17 paragraph_noise_floor_all_novel.json per-band per_dim_std, ac_* from ac_multi_window_per_dim_std (v1 11D). NOTE: sigma was generated from this same Set 1 (in-sample).",
        "n_adjacent_pairs": int(len(idx)),
        "n_both_single": int(both_single.sum()), "n_mixed": int(mixed.sum()), "n_both_multi": int(both_multi.sum()),
        "mixed_v1_11d": agg(mixed_idx, KASANE11_DIM_NAMES, True),
        "mixed_v2_8d_canonical": agg(mixed_idx, KASANE8_ACTIVE_DIM_NAMES, False),
        "mixed_7d_diagnostic_only": agg(mixed_idx, DC7, False),
        "by_band": {},
        "non_mixed_unchanged_check": {},
    }
    for bn in ("S", "M", "L"):
        sel = [i for i in mixed_idx if bsel[i]["name"] == bn]
        if sel:
            out["by_band"][bn] = {
                "mixed_v1_11d": agg(sel, KASANE11_DIM_NAMES, True),
                "mixed_v2_8d_canonical": agg(sel, KASANE8_ACTIVE_DIM_NAMES, False),
                "mixed_7d_diagnostic_only": agg(sel, DC7, False),
            }
    # 7D/8D ratio of abs_max medians (F-18 reported 0.942) and mean-of-pairs
    a8 = out["mixed_v2_8d_canonical"]["abs_max"]["median"]; a7 = out["mixed_7d_diagnostic_only"]["abs_max"]["median"]
    out["ratio_7d_over_8d_abs_max_median"] = _r(a7 / a8, 4)
    # crest share: fraction of mixed pairs where crest is argmax (8D) and where crest alone exceeds 1 sigma
    cr_arg = 0; cr_only = 0
    for i in mixed_idx:
        s = sig_for(i, False)
        r8 = {dn: abs(d[i, names.index(dn)]) / s[dn] for dn in KASANE8_ACTIVE_DIM_NAMES}
        if max(r8, key=r8.get) == "ac_mean_crest":
            cr_arg += 1
        if r8["ac_mean_crest"] > 1.0 and all(r8[dn] <= 1.0 for dn in DC7):
            cr_only += 1
    out["crest_argmax_fraction_8d"] = _r(cr_arg / len(mixed_idx))
    out["crest_only_above_1sd_fraction_8d"] = _r(cr_only / len(mixed_idx))
    # 8D pairs unaffected: [1,1] and [>=2,>=2] pairs are identical under v1 and v2
    for nm, mask in (("both_single", both_single), ("both_multi", both_multi)):
        out["non_mixed_unchanged_check"][nm] = int(mask.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe-v2", required=True, type=Path)
    ap.add_argument("--exe-v1", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    assert args.out.resolve() != B13_SUMMARY_PATH.resolve(), "refuse to overwrite the frozen F-17 summary"
    frozen_before = {str(p): compute_sha256(p) for p in FROZEN if p.exists()}

    assert compute_sha256(DATASET_PATH) == EXPECTED_DATASET_SHA256
    assert compute_sha256(F17_NOISE_FLOOR_PATH) == EXPECTED_F17_SHA256
    rows = [json.loads(l) for l in DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 415
    nf_data, prov = load_f14_noise_floor(F17_NOISE_FLOOR_PATH)

    recs_v2 = _run_rust_harvest_batch(args.exe_v2, F17_NOISE_FLOOR_PATH, rows)
    recs_v1 = _run_rust_harvest_batch(args.exe_v1, F17_NOISE_FLOOR_PATH, rows)

    parity_bad = python_rust_parity(rows, recs_v2, nf_data, prov)
    assert not parity_bad, f"Python/Rust v2 parity discrepancies: {parity_bad[:5]}"

    m_v1 = [_build_pair_record(r, c) for r, c in zip(rows, recs_v1)]
    m_v2 = [_build_pair_record(r, c) for r, c in zip(rows, recs_v2)]

    # Baseline check: pre-v2 binary reproduces the frozen F-17 summary (overall median of abs_ratio_max etc.)
    b13 = json.loads(B13_SUMMARY_PATH.read_text(encoding="utf-8"))
    b13_overall = b13["distributions_f17"]["overall"]["ratio_distributions"]["per_pair_abs_ratio_max"]["median"]
    assert abs(b13_overall - round(_med([m["abs_ratio_max"] for m in m_v1]), 4)) < 1e-4, (b13_overall, _med([m["abs_ratio_max"] for m in m_v1]))

    # Unchanged-pair invariants: non-mixed pairs are bit-identical between v1 and v2 outputs
    inv_diff = [c1["noise_floor_ratio_kasane11"]["per_dim"] != c2["noise_floor_ratio_kasane11"]["per_dim"]
                for r, c1, c2 in zip(rows, recs_v1, recs_v2) if not r["mixed_window_pair"]]
    n_non_mixed = len(inv_diff)
    n_non_mixed_changed = int(sum(inv_diff))
    assert n_non_mixed_changed == 0, "non-mixed pairs changed between v1 and v2"
    mixed_nev_v1 = sorted({c["noise_floor_ratio_kasane11"]["n_evaluated"] for r, c in zip(rows, recs_v1) if r["mixed_window_pair"]})
    mixed_nev_v2 = sorted({c["noise_floor_ratio_kasane11"]["n_evaluated"] for r, c in zip(rows, recs_v2) if r["mixed_window_pair"]})

    g_v1, g_v2 = groups_of(m_v1), groups_of(m_v2)
    recs_by_id = {m["pair_id"]: c for m, c in zip(m_v2, recs_v2)}
    comparison = {}
    for k in g_v1:
        comparison[k] = {"before_rule_v1": metric_block(g_v1[k]) if g_v1[k] else None, "after_rule_v2": metric_block(g_v2[k]) if g_v2[k] else None}
        if "mixed" in k or k in ("minor_band_all", "overall") or k.startswith("band_"):
            mixed_items = [m for m in g_v2[k] if m["mixed_window_pair"]]
            if mixed_items:
                comparison[k]["mixed_subset_7d_diagnostic_only"] = diag7_block(mixed_items, recs_by_id)
                comparison[k]["mixed_subset_8d_canonical_v2"] = metric_block(mixed_items)
                comparison[k]["mixed_subset_11d_v1"] = metric_block([m for m in g_v1[k] if m["mixed_window_pair"]])

    set1 = set1_mixed_recompute(nf_data)

    frozen_after = {str(p): compute_sha256(p) for p in FROZEN if p.exists()}
    assert frozen_before == frozen_after

    summary = {
        "stage": "B-14",
        "eval_rule": "v2",
        "eval_rule_description": "Evaluation rule v2 (2026-11-13, B-14): 11D only if both sides >= 2 windows; [1,1] and mixed pairs are 8D (ac_* excluded from numerator and denominator). ac_* sigma population unchanged (both-multi pairs only). Pool criterion, margin 1.25 and sigma table values unchanged. Rationale: B-13 (1-window ac_* == 0.0 in 476/476; mixed 11D gives 'value - 0' artifact), F-18 (ac_mean_crest inflates 1.7-2.0x across window counts; kept in canonical 8D, 7D diagnostic only).",
        "dataset_path": "data/usir_v5_eval/real_revision_pairs_v1.jsonl",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "f17_noise_floor_path": "docs/handoff/v10/paragraph_noise_floor_all_novel.json",
        "f17_noise_floor_sha256": EXPECTED_F17_SHA256,
        "before_snapshot": "docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json (frozen, not modified)",
        "before_reproduced_by_pre_v2_binary": True,
        "python_rust_parity": {"n_pairs": len(rows), "discrepancy_count": len(parity_bad), "discrepancies": parity_bad},
        "invariants": {
            "non_mixed_pairs": n_non_mixed,
            "non_mixed_pairs_changed_v1_to_v2": n_non_mixed_changed,
            "mixed_n_evaluated_values_v1": mixed_nev_v1,
            "mixed_n_evaluated_values_v2": mixed_nev_v2,
            "frozen_artifact_sha256_unchanged": frozen_after,
        },
        "diagnostic_7d_note": "7D (crest excluded) appears only as 'mixed_subset_7d_diagnostic_only' / 'mixed_7d_diagnostic_only'. It is diagnostic only and not canonical.",
        "comparison": comparison,
        "primary_subset_note": "primary_* groups are the same_orthography == True subset (381 pairs; B-13 task 3 conclusion).",
        "set1_mixed_recompute": set1,
        "per_pair_v2": [
            {k: m[k] for k in ("pair_id", "band", "window_counts", "mixed_window_pair", "same_orthography", "dl_over_lmax", "dice_similarity", "n_evaluated", "abs_ratio_median", "abs_ratio_mean", "abs_ratio_max", "argmax_dim", "n_above_1sd", "n_above_2sd")}
            for m in m_v2
        ],
        "governance_note": "Gate 3a status and Gate 3b freeze are unchanged by this stage. No Level 3b rating is stated or changed.",
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.out)
    print("parity discrepancies:", len(parity_bad))
    for k in ("overall", "mixed_window_pairs", "non_mixed_pairs", "minor_band_all", "minor_band_mixed", "primary_same_orthography"):
        c = comparison[k]
        a, b = c["before_rule_v1"], c["after_rule_v2"]
        print(k, a["n_pairs"], "abs_max med v1/v2:", a["abs_max"]["median"], b["abs_max"]["median"])
    print("set1:", json.dumps({k: set1[k] for k in ("n_adjacent_pairs", "n_mixed", "ratio_7d_over_8d_abs_max_median", "crest_argmax_fraction_8d", "crest_only_above_1sd_fraction_8d")}))


if __name__ == "__main__":
    main()
