#!/usr/bin/env python
"""USIR v5 Stage B-13: F-17 Re-measurement, Mixed-Pair 11D->8D->7D Ablation, and Orthography Sensitivity Analysis.

Tasks covered:
  - Task 1: Re-run harvest-diff on real_revision_pairs_v1.jsonl (415 pairs) under F-17 noise floor
            (docs/handoff/v10/paragraph_noise_floor_all_novel.json), verify 0/415 Python-Rust parity
            discrepancy, verify n_evaluated=8 (196 pairs) bit-level invariance vs F-15, check
            mathematical predictions for velocity/max_sis, count 1SD threshold crossings, and
            split band_L_both_single_window (50) vs band_L_both_multi_window (61).
  - Task 2: 3-stage ablation (11D -> 8D -> 7D) across all 415 pairs (especially mixed=84 vs
            non_mixed_multi=135 vs non_mixed_single=196), empirical verification of 1-window ac_*
            zeros, stratification (1) by window transition pattern, and stratification (2) by
            length change rate (dL/Lmax) and 2-gram Dice bins.
  - Task 3: Sensitivity analysis comparing same_orthography==True (381 pairs),
            same_orthography==False (34 pairs, ginga_initial_vs_final), and core set
            (same_orthography==True and dL/Lmax <= 0.30).

Output:
  - docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from usir_py.v5.f14_hybrid_noise_floor import (
    KASANE8_ACTIVE_DIM_NAMES,
    KASANE11_DIM_NAMES,
    compute_sha256,
    evaluate_f14_hybrid_pair,
    load_f14_noise_floor,
    select_length_band,
)

DATASET_PATH = REPO_ROOT / "data" / "usir_v5_eval" / "real_revision_pairs_v1.jsonl"
EXPECTED_DATASET_SHA256 = "62da75271f453bd3a8f161d916fdf7b7b87138b773b564b737f90aee85b6b8e4"

F15_NOISE_FLOOR_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_f15_all_novel.json"
EXPECTED_F15_SHA256 = "5a6095659d4ef7defeab238f7ffb7aed3a980d6c8a9a83d0ca229f4c11a05910"

F17_NOISE_FLOOR_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "paragraph_noise_floor_all_novel.json"
EXPECTED_F17_SHA256 = "0575b64a75a8af8dd1ad0db81bdf4d8898c473ce8ac1876c6e7e2f8b61a6f604"

F15_SUMMARY_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_f15_harvest_summary.json"
OUT_SUMMARY_PATH = REPO_ROOT / "docs" / "handoff" / "v10" / "real_revision_pairs_v1_f17_harvest_summary.json"

NISHIKI_EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")
DC7_DIM_NAMES: Tuple[str, ...] = KASANE11_DIM_NAMES[:7]


def _quantiles(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "iqr": None,
            "max": None,
            "mean": None,
        }
    arr = np.array(vals, dtype=float)
    q25 = float(np.percentile(arr, 25))
    med = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    return {
        "count": int(len(arr)),
        "min": round(float(np.min(arr)), 4),
        "q25": round(q25, 4),
        "median": round(med, 4),
        "q75": round(q75, 4),
        "iqr": round(q75 - q25, 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def _run_rust_harvest_batch(
    nishiki_exe: Path,
    noise_floor_path: Path,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run Rust `nishiki harvest-diff` in a single batch using unique unchanged sentinels."""
    b_lines: List[str] = []
    a_lines: List[str] = []
    for idx, r in enumerate(rows):
        b_lines.append(r["text_before"])
        a_lines.append(r["text_after"])
        sent = f"SENTINEL_ANCHOR_UNCHANGED_BLOCK_{idx:04d}_" + ("あ" * 40)
        b_lines.append(sent)
        a_lines.append(sent)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        v1 = td_path / "v1.txt"
        v2 = td_path / "v2.txt"
        out = td_path / "out.jsonl"
        v1.write_text("\n\n".join(b_lines), encoding="utf-8")
        v2.write_text("\n\n".join(a_lines), encoding="utf-8")
        subprocess.run(
            [
                str(nishiki_exe),
                "harvest-diff",
                str(v1),
                str(v2),
                "--noise-floor",
                str(noise_floor_path),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        recs = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]

    assert len(recs) == len(rows), f"Expected {len(rows)} records, got {len(recs)}"
    for r, rec in zip(rows, recs):
        assert rec["kind"] == "pair", f"Unexpected non-pair record for {r['pair_id']}: {rec.get('kind')}"
        assert rec["text_before"] == r["text_before"] and rec["text_after"] == r["text_after"]
    return recs


def _build_pair_record(
    row: Dict[str, Any],
    rust_rec: Dict[str, Any],
) -> Dict[str, Any]:
    rn = rust_rec["noise_floor_ratio_kasane11"]
    per_dim_map = rn.get("per_dim") or {}
    abs_ratios = [abs(float(v)) for v in per_dim_map.values() if v is not None]
    n_eval = int(rn["n_evaluated"])
    n_above_1sd = int(rn["n_above_1"])
    n_above_2sd = int(sum(1 for r in abs_ratios if r >= 2.0))
    l_before = int(row["len_before"])
    l_after = int(row["len_after"])
    l_max = int(row["l_max"])
    dl_over_lmax = abs(l_after - l_before) / max(l_before, l_after)
    return {
        "pair_id": row["pair_id"],
        "work_group_id": row["work_group_id"],
        "band": rn["band"],
        "l_before": l_before,
        "l_after": l_after,
        "l_max": l_max,
        "l_min": int(row["l_min"]),
        "dl_over_lmax": dl_over_lmax,
        "dice_similarity": float(row["dice_similarity"]),
        "window_counts": rust_rec["window_counts"],
        "mixed_window_pair": bool(row["mixed_window_pair"]),
        "variant_relationship": row["variant_relationship"],
        "same_orthography": bool(row["same_orthography"]),
        "n_evaluated": n_eval,
        "ac_sigma_source": rn.get("ac_sigma_source"),
        "abs_ratio_median": float(rn["median"]),
        "abs_ratio_mean": float(rn["mean"]),
        "abs_ratio_max": float(rn["max"]),
        "argmax_dim": rn.get("argmax"),
        "n_above_1sd": n_above_1sd,
        "n_above_2sd": n_above_2sd,
        "frac_above_1sd": n_above_1sd / n_eval,
        "frac_above_2sd": n_above_2sd / n_eval,
        "per_dim": {k: float(v) if v is not None else None for k, v in per_dim_map.items()},
    }


def _summarize_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_eval_dist: Dict[str, int] = {"8": 0, "10": 0, "11": 0}
    for it in items:
        k = str(it["n_evaluated"])
        n_eval_dist[k] = n_eval_dist.get(k, 0) + 1
    return {
        "n_pairs": len(items),
        "n_evaluated_distribution": n_eval_dist,
        "mixed_window_count": sum(1 for it in items if it["mixed_window_pair"]),
        "ratio_distributions": {
            "per_pair_abs_ratio_median": _quantiles([it["abs_ratio_median"] for it in items]),
            "per_pair_abs_ratio_mean": _quantiles([it["abs_ratio_mean"] for it in items]),
            "per_pair_abs_ratio_max": _quantiles([it["abs_ratio_max"] for it in items]),
            "per_pair_frac_above_1sd": _quantiles([it["frac_above_1sd"] for it in items]),
            "per_pair_frac_above_2sd": _quantiles([it["frac_above_2sd"] for it in items]),
        },
    }


def _group_definitions(metrics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "overall": metrics,
        "band_S": [m for m in metrics if m["band"] == "S"],
        "band_M": [m for m in metrics if m["band"] == "M"],
        "band_L": [m for m in metrics if m["band"] == "L"],
        "mixed_window_pairs": [m for m in metrics if m["mixed_window_pair"]],
        "non_mixed_both_single_window": [m for m in metrics if max(m["window_counts"]) <= 1],
        "non_mixed_both_multi_window": [m for m in metrics if min(m["window_counts"]) >= 2],
        "band_M_n_eval_8": [m for m in metrics if m["band"] == "M" and m["n_evaluated"] == 8],
        "band_M_n_eval_11": [m for m in metrics if m["band"] == "M" and m["n_evaluated"] == 11],
        "band_L_mixed_window": [m for m in metrics if m["band"] == "L" and m["mixed_window_pair"]],
        "band_L_both_single_window": [
            m for m in metrics if m["band"] == "L" and not m["mixed_window_pair"] and max(m["window_counts"]) <= 1
        ],
        "band_L_both_multi_window": [
            m for m in metrics if m["band"] == "L" and not m["mixed_window_pair"] and min(m["window_counts"]) >= 2
        ],
    }


def _eval_dim_subset(
    row: Dict[str, Any],
    rust_rec: Dict[str, Any],
    bands: Sequence[Dict[str, Any]],
    dim_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    band = select_length_band(int(row["l_max"]), bands)
    assert band is not None
    base_sigma = dict(band["kasane11"]["per_dim_std"])
    ac_multi = band["kasane11"].get("ac_multi_window_per_dim_std") or {}
    merged = dict(base_sigma)
    for k, v in ac_multi.items():
        if v is not None and v > 0:
            merged[k] = v

    delta = rust_rec["delta_kasane11"]
    vals: List[float] = []
    per_dim: Dict[str, float] = {}
    for d in dim_names:
        s = merged.get(d)
        if s is None or float(s) <= 0.0:
            return None
        rv = abs(float(delta[d])) / float(s)
        vals.append(rv)
        per_dim[d] = rv

    argmax_dim = max(per_dim, key=per_dim.get)
    n_ev = len(vals)
    n_above_1 = sum(1 for v in vals if v > 1.0)
    n_above_2 = sum(1 for v in vals if v >= 2.0)
    return {
        "abs_median": float(statistics.median(vals)),
        "abs_mean": float(sum(vals) / n_ev),
        "abs_max": float(max(vals)),
        "argmax_dim": argmax_dim,
        "frac_above_1sd": n_above_1 / n_ev,
        "frac_above_2sd": n_above_2 / n_ev,
        "per_dim": per_dim,
    }


def _summarize_ablation_subset(
    items: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    bands: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_pairs": len(items)}
    for label, dims in (("11D", KASANE11_DIM_NAMES), ("8D", KASANE8_ACTIVE_DIM_NAMES), ("7D", DC7_DIM_NAMES)):
        evs = [_eval_dim_subset(r, rec, bands, dims) for r, rec in items]
        valid = [e for e in evs if e is not None]
        argmax_counts: Dict[str, int] = {}
        for e in valid:
            ad = e["argmax_dim"]
            argmax_counts[ad] = argmax_counts.get(ad, 0) + 1
        out[label] = {
            "n_evaluated_pairs": len(valid),
            "n_skipped_zero_sigma": len(evs) - len(valid),
            "abs_median": _quantiles([e["abs_median"] for e in valid]),
            "abs_mean": _quantiles([e["abs_mean"] for e in valid]),
            "abs_max": _quantiles([e["abs_max"] for e in valid]),
            "frac_above_1sd": _quantiles([e["frac_above_1sd"] for e in valid]),
            "frac_above_2sd": _quantiles([e["frac_above_2sd"] for e in valid]),
            "argmax_dim_counts": dict(sorted(argmax_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    return out


def _dl_bin(r: Dict[str, Any]) -> str:
    dl = abs(int(r["len_after"]) - int(r["len_before"])) / max(int(r["len_before"]), int(r["len_after"]))
    if dl < 0.05:
        return "lt_5pct"
    if dl < 0.15:
        return "5_to_15pct"
    if dl < 0.30:
        return "15_to_30pct"
    return "ge_30pct"


def _dice_bin(r: Dict[str, Any]) -> str:
    d = float(r["dice_similarity"])
    if d >= 0.85:
        return "ge_0.85_minor"
    if d >= 0.60:
        return "0.60_to_0.85_moderate"
    return "0.25_to_0.60_major"


def _window_transition_stratum(r: Dict[str, Any]) -> str:
    wb, wa = r["window_counts"]
    w_min, w_max = min(wb, wa), max(wb, wa)
    if w_min == 1 and w_max == 1:
        return "non_mixed_single_1_1"
    if w_min == 1 and w_max == 2:
        return "mixed_adjacent_1_2"
    if w_min == 1 and w_max >= 3:
        return "mixed_jump_1_3plus"
    if w_min >= 2 and wb == wa:
        return "non_mixed_multi_same_w"
    return "non_mixed_multi_diff_w"


def _compute_f15_vs_f17_diff(
    dist15: Dict[str, Any],
    dist17: Dict[str, Any],
) -> Dict[str, Any]:
    diff_table: Dict[str, Any] = {}
    metric_keys = (
        "per_pair_abs_ratio_median",
        "per_pair_abs_ratio_mean",
        "per_pair_abs_ratio_max",
        "per_pair_frac_above_1sd",
        "per_pair_frac_above_2sd",
    )
    for grp, g17 in dist17.items():
        g15 = dist15.get(grp)
        if g15 is None:
            continue
        assert g15["n_pairs"] == g17["n_pairs"], f"Pair count mismatch for {grp}"
        grp_diff: Dict[str, Any] = {"n_pairs": g17["n_pairs"]}
        for mk in metric_keys:
            r15 = g15["ratio_distributions"][mk]
            r17 = g17["ratio_distributions"][mk]
            grp_diff[mk] = {
                "f15_median": r15["median"],
                "f17_median": r17["median"],
                "delta_median": round(r17["median"] - r15["median"], 4) if r15["median"] is not None else None,
                "f15_mean": r15["mean"],
                "f17_mean": r17["mean"],
                "delta_mean": round(r17["mean"] - r15["mean"], 4) if r15["mean"] is not None else None,
                "f15_max": r15["max"],
                "f17_max": r17["max"],
                "delta_max": round(r17["max"] - r15["max"], 4) if r15["max"] is not None else None,
            }
        diff_table[grp] = grp_diff
    return diff_table


def main() -> None:
    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    assert F15_NOISE_FLOOR_PATH.exists(), f"Missing F-15 JSON: {F15_NOISE_FLOOR_PATH}"
    assert F17_NOISE_FLOOR_PATH.exists(), f"Missing F-17 JSON: {F17_NOISE_FLOOR_PATH}"
    assert NISHIKI_EXE.is_file(), f"Missing nishiki binary: {NISHIKI_EXE}"

    ds_sha = compute_sha256(DATASET_PATH)
    f15_sha = compute_sha256(F15_NOISE_FLOOR_PATH)
    f17_sha = compute_sha256(F17_NOISE_FLOOR_PATH)
    assert ds_sha == EXPECTED_DATASET_SHA256, f"Dataset SHA mismatch: {ds_sha}"
    assert f15_sha == EXPECTED_F15_SHA256, f"F-15 SHA mismatch: {f15_sha}"
    assert f17_sha == EXPECTED_F17_SHA256, f"F-17 SHA mismatch: {f17_sha}"

    rows = [json.loads(ln) for ln in DATASET_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 415, f"Expected 415 rows, got {len(rows)}"

    nf17_data, prov17 = load_f14_noise_floor(F17_NOISE_FLOOR_PATH)
    bands17 = nf17_data["length_bands"]["bands"]

    # Verify F-17 per-band ac_* effective values
    band_m = next(b for b in bands17 if b["name"] == "M")
    band_l = next(b for b in bands17 if b["name"] == "L")
    ac_m = band_m["kasane11"]["ac_multi_window_per_dim_std"]
    ac_l = band_l["kasane11"]["ac_multi_window_per_dim_std"]
    assert abs(ac_m["ac_mean_velocity"] - 134.29641924737427) < 1e-9
    assert abs(ac_m["ac_max_sis"] - 0.12661474438213988) < 1e-9
    assert abs(ac_m["ac_coupling"] - 1.4078678674874479) < 1e-9
    assert abs(ac_l["ac_mean_velocity"] - 102.57321663587159) < 1e-9
    assert abs(ac_l["ac_max_sis"] - 0.14935699218887902) < 1e-9
    assert abs(ac_l["ac_coupling"] - 1.0668175754568485) < 1e-9

    print("Running Rust harvest-diff batch for F-15 and F-17...")
    rust_r15 = _run_rust_harvest_batch(NISHIKI_EXE, F15_NOISE_FLOOR_PATH, rows)
    rust_r17 = _run_rust_harvest_batch(NISHIKI_EXE, F17_NOISE_FLOOR_PATH, rows)

    # Task 1.1: Python vs Rust parity on F-17 (0/415)
    parity_discrepancies: List[Dict[str, Any]] = []
    for row, rec17 in zip(rows, rust_r17):
        py17 = evaluate_f14_hybrid_pair(
            row["text_before"],
            row["text_after"],
            nf17_data,
            delta_kasane11=rec17["delta_kasane11"],
            provenance=prov17,
            apply_dice_gate=True,
        )
        rn17 = rec17["noise_floor_ratio_kasane11"]
        meta_ok = (
            py17["window_counts"] == rec17["window_counts"]
            and py17["n_evaluated"] == rn17["n_evaluated"]
            and py17["band"] == rn17["band"]
            and py17["ac_sigma_source"] == rn17["ac_sigma_source"]
        )
        num_ok = (
            abs(py17["ratios"]["max"] - rn17["max"]) <= 1e-12
            and abs(py17["ratios"]["mean"] - rn17["mean"]) <= 1e-12
            and abs(py17["ratios"]["median"] - rn17["median"]) <= 1e-12
            and py17["ratios"]["n_above_1"] == rn17["n_above_1"]
        )
        if not (meta_ok and num_ok):
            parity_discrepancies.append({"pair_id": row["pair_id"], "meta_ok": meta_ok, "num_ok": num_ok})
    assert len(parity_discrepancies) == 0, f"Parity discrepancies found: {parity_discrepancies}"

    # Task 1.2: Verification gate (n_evaluated=8 bit-invariance vs F-15)
    n_eval8_pairs = 0
    n_eval8_diffs: List[str] = []
    for row, rec15, rec17 in zip(rows, rust_r15, rust_r17):
        if rec17["noise_floor_ratio_kasane11"]["n_evaluated"] == 8:
            n_eval8_pairs += 1
            if rec15["noise_floor_ratio_kasane11"] != rec17["noise_floor_ratio_kasane11"]:
                n_eval8_diffs.append(row["pair_id"])
    assert n_eval8_pairs == 196, f"Expected 196 n_eval=8 pairs, got {n_eval8_pairs}"
    assert len(n_eval8_diffs) == 0, f"n_eval=8 diffs found: {n_eval8_diffs}"

    # Task 1.3: Mathematical predictions & 1SD threshold crossings
    exact_1sd_boundary_hits = 0
    vel_cross_pairs: List[Dict[str, Any]] = []
    sis_cross_pairs: List[Dict[str, Any]] = []
    multipliers: Dict[str, List[float]] = {
        "velocity_M": [],
        "velocity_L": [],
        "max_sis_M": [],
        "max_sis_L": [],
        "coupling_ML": [],
    }
    for row, rec15, rec17 in zip(rows, rust_r15, rust_r17):
        rn15 = rec15["noise_floor_ratio_kasane11"]
        rn17 = rec17["noise_floor_ratio_kasane11"]
        if rn17["n_evaluated"] != 11:
            continue
        band = rn17["band"]
        v15, v17 = float(rn15["per_dim"]["ac_mean_velocity"]), float(rn17["per_dim"]["ac_mean_velocity"])
        s15, s17 = float(rn15["per_dim"]["ac_max_sis"]), float(rn17["per_dim"]["ac_max_sis"])
        c15, c17 = float(rn15["per_dim"]["ac_coupling"]), float(rn17["per_dim"]["ac_coupling"])
        for val in (v15, v17, s15, s17, c15, c17):
            if val == 1.0:
                exact_1sd_boundary_hits += 1
        if v15 > 0.0:
            multipliers[f"velocity_{band}"].append(v17 / v15)
        if s15 > 0.0:
            multipliers[f"max_sis_{band}"].append(s17 / s15)
        if c15 > 0.0:
            multipliers["coupling_ML"].append(c17 / c15)

        if (v15 >= 1.0) != (v17 >= 1.0):
            vel_cross_pairs.append(
                {
                    "pair_id": row["pair_id"],
                    "band": band,
                    "mixed_window_pair": bool(row["mixed_window_pair"]),
                    "same_orthography": bool(row["same_orthography"]),
                    "direction": "up_ge_1" if v17 >= 1.0 else "down_lt_1",
                    "f15_ratio": round(v15, 6),
                    "f17_ratio": round(v17, 6),
                }
            )
        if (s15 >= 1.0) != (s17 >= 1.0):
            sis_cross_pairs.append(
                {
                    "pair_id": row["pair_id"],
                    "band": band,
                    "mixed_window_pair": bool(row["mixed_window_pair"]),
                    "same_orthography": bool(row["same_orthography"]),
                    "direction": "up_ge_1" if s17 >= 1.0 else "down_lt_1",
                    "f15_ratio": round(s15, 6),
                    "f17_ratio": round(s17, 6),
                }
            )

    metrics15 = [_build_pair_record(r, rec) for r, rec in zip(rows, rust_r15)]
    metrics17 = [_build_pair_record(r, rec) for r, rec in zip(rows, rust_r17)]

    groups15 = _group_definitions(metrics15)
    groups17 = _group_definitions(metrics17)
    dist15 = {k: _summarize_group(v) for k, v in groups15.items()}
    dist17 = {k: _summarize_group(v) for k, v in groups17.items()}
    f15_vs_f17_comparison = _compute_f15_vs_f17_diff(dist15, dist17)

    # Task 2: 3-stage ablation (11D -> 8D -> 7D) + hypothesis verification + stratifications
    paired_all = list(zip(rows, rust_r17))
    one_window_sides_total = 0
    one_window_ac_nonzero = {
        "ac_mean_velocity": 0,
        "ac_max_sis": 0,
        "ac_coupling": 0,
        "ac_mean_crest": 0,
    }
    for row, rec in paired_all:
        wb, wa = rec["window_counts"]
        for w, feat in ((wb, rec["features_before"]), (wa, rec["features_after"])):
            if w <= 1:
                one_window_sides_total += 1
                for d in one_window_ac_nonzero:
                    if abs(float(feat[d])) > 0.0:
                        one_window_ac_nonzero[d] += 1

    ablation_groups = {
        "overall": paired_all,
        "mixed_window_pairs": [(r, rec) for r, rec in paired_all if r["mixed_window_pair"]],
        "non_mixed_both_multi_window": [(r, rec) for r, rec in paired_all if min(r["window_counts"]) >= 2],
        "non_mixed_both_single_window": [(r, rec) for r, rec in paired_all if max(r["window_counts"]) <= 1],
    }
    ablation_summary = {k: _summarize_ablation_subset(v, bands17) for k, v in ablation_groups.items()}

    # Stratification 1: window transition patterns (mixed=84 + non_mixed_multi=135)
    strat1_buckets: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {
        "mixed_adjacent_1_2": [],
        "mixed_jump_1_3plus": [],
        "non_mixed_multi_same_w": [],
        "non_mixed_multi_diff_w": [],
        "non_mixed_single_1_1": [],
    }
    for r, rec in paired_all:
        strat1_buckets[_window_transition_stratum(r)].append((r, rec))
    strat1_summary = {k: _summarize_ablation_subset(v, bands17) for k, v in strat1_buckets.items()}

    # Stratification 2: dL/Lmax (4 bins) x Dice (3 bins), comparing mixed vs non-mixed
    dl_bins = ("lt_5pct", "5_to_15pct", "15_to_30pct", "ge_30pct")
    dice_bins = ("ge_0.85_minor", "0.60_to_0.85_moderate", "0.25_to_0.60_major")
    strat2_cells: Dict[str, Any] = {}
    for db in dl_bins:
        for cb in dice_bins:
            cell_key = f"{db}__{cb}"
            cell_items = [(r, rec) for r, rec in paired_all if _dl_bin(r) == db and _dice_bin(r) == cb]
            if not cell_items:
                continue
            mixed_c = [(r, rec) for r, rec in cell_items if r["mixed_window_pair"]]
            non_mixed_c = [(r, rec) for r, rec in cell_items if not r["mixed_window_pair"]]
            non_mixed_multi_c = [(r, rec) for r, rec in non_mixed_c if min(r["window_counts"]) >= 2]
            non_mixed_single_c = [(r, rec) for r, rec in non_mixed_c if max(r["window_counts"]) <= 1]
            strat2_cells[cell_key] = {
                "n_total": len(cell_items),
                "mixed": _summarize_ablation_subset(mixed_c, bands17) if mixed_c else None,
                "non_mixed_all": _summarize_ablation_subset(non_mixed_c, bands17) if non_mixed_c else None,
                "non_mixed_multi": _summarize_ablation_subset(non_mixed_multi_c, bands17) if non_mixed_multi_c else None,
                "non_mixed_single": _summarize_ablation_subset(non_mixed_single_c, bands17) if non_mixed_single_c else None,
            }

    # Controlled minor-revision band: dL/Lmax < 15% and Dice >= 0.85
    minor_band_items = [
        (r, rec) for r, rec in paired_all if _dl_bin(r) in ("lt_5pct", "5_to_15pct") and _dice_bin(r) == "ge_0.85_minor"
    ]
    minor_band_comparison = {
        "definition": "dl_over_lmax < 0.15 and dice_similarity >= 0.85",
        "n_total": len(minor_band_items),
        "mixed": _summarize_ablation_subset(
            [(r, rec) for r, rec in minor_band_items if r["mixed_window_pair"]], bands17
        ),
        "non_mixed_all": _summarize_ablation_subset(
            [(r, rec) for r, rec in minor_band_items if not r["mixed_window_pair"]], bands17
        ),
        "non_mixed_multi": _summarize_ablation_subset(
            [(r, rec) for r, rec in minor_band_items if min(r["window_counts"]) >= 2], bands17
        ),
        "non_mixed_single": _summarize_ablation_subset(
            [(r, rec) for r, rec in minor_band_items if max(r["window_counts"]) <= 1], bands17
        ),
    }

    # Task 3: Sensitivity analysis (same_orthography == True [381], False [34], Core [<= 0.30])
    def _build_cohort_report(name: str, pred: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
        c_indices = [i for i, r in enumerate(rows) if pred(r)]
        c_rows = [rows[i] for i in c_indices]
        c_r15 = [metrics15[i] for i in c_indices]
        c_r17 = [metrics17[i] for i in c_indices]
        c_paired = [(rows[i], rust_r17[i]) for i in c_indices]

        g15 = _group_definitions(c_r15)
        g17 = _group_definitions(c_r17)
        d15 = {k: _summarize_group(v) for k, v in g15.items()}
        d17 = {k: _summarize_group(v) for k, v in g17.items()}

        c_ablation = {
            "overall": _summarize_ablation_subset(c_paired, bands17),
            "mixed_window_pairs": _summarize_ablation_subset(
                [(r, rec) for r, rec in c_paired if r["mixed_window_pair"]], bands17
            ),
            "non_mixed_both_multi_window": _summarize_ablation_subset(
                [(r, rec) for r, rec in c_paired if min(r["window_counts"]) >= 2], bands17
            ),
            "non_mixed_both_single_window": _summarize_ablation_subset(
                [(r, rec) for r, rec in c_paired if max(r["window_counts"]) <= 1], bands17
            ),
        }
        c_minor = [
            (r, rec)
            for r, rec in c_paired
            if _dl_bin(r) in ("lt_5pct", "5_to_15pct") and _dice_bin(r) == "ge_0.85_minor"
        ]
        return {
            "cohort_name": name,
            "n_pairs": len(c_rows),
            "n_work_groups": len({r["work_group_id"] for r in c_rows}),
            "band_counts": {
                "S": sum(1 for r in c_rows if r["band"] == "S"),
                "M": sum(1 for r in c_rows if r["band"] == "M"),
                "L": sum(1 for r in c_rows if r["band"] == "L"),
            },
            "mixed_window_count": sum(1 for r in c_rows if r["mixed_window_pair"]),
            "f17_distributions": d17,
            "f15_vs_f17_comparison": _compute_f15_vs_f17_diff(d15, d17),
            "ablation_11d_8d_7d": c_ablation,
            "minor_revision_band_comparison": {
                "n_total": len(c_minor),
                "mixed": _summarize_ablation_subset([(r, rec) for r, rec in c_minor if r["mixed_window_pair"]], bands17),
                "non_mixed_all": _summarize_ablation_subset(
                    [(r, rec) for r, rec in c_minor if not r["mixed_window_pair"]], bands17
                ),
            },
        }

    cohorts = {
        "all_415": _build_cohort_report("all_415", lambda r: True),
        "same_orthography_true_381": _build_cohort_report(
            "same_orthography_true_381", lambda r: bool(r["same_orthography"])
        ),
        "same_orthography_false_34": _build_cohort_report(
            "same_orthography_false_34", lambda r: not bool(r["same_orthography"])
        ),
        "same_orthography_true_core_dl_le_30pct": _build_cohort_report(
            "same_orthography_true_core_dl_le_30pct",
            lambda r: bool(r["same_orthography"])
            and (abs(int(r["len_after"]) - int(r["len_before"])) / max(int(r["len_before"]), int(r["len_after"])) <= 0.30),
        ),
    }

    summary = {
        "stage": "B-13",
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": ds_sha,
        "f15_noise_floor_path": str(F15_NOISE_FLOOR_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "f15_noise_floor_sha256": f15_sha,
        "f17_noise_floor_path": str(F17_NOISE_FLOOR_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "f17_noise_floor_sha256": f17_sha,
        "f17_ac_multi_window_per_dim_std_verified": {
            "M": ac_m,
            "L": ac_l,
        },
        "rust_binary_used": str(NISHIKI_EXE),
        "python_rust_parity_discrepancy_count": len(parity_discrepancies),
        "python_rust_parity_discrepancies": parity_discrepancies,
        "verification_gate": {
            "n_evaluated_8_pair_count": n_eval8_pairs,
            "n_evaluated_8_bit_level_diff_count": len(n_eval8_diffs),
            "bit_identical_to_f15": len(n_eval8_diffs) == 0,
        },
        "mathematical_predictions_verification": {
            "n_evaluated_11_pair_count": 415 - n_eval8_pairs,
            "observed_multipliers": {
                k: {
                    "count": len(v),
                    "min": round(min(v), 8) if v else None,
                    "max": round(max(v), 8) if v else None,
                }
                for k, v in multipliers.items()
            },
            "threshold_1sd_crossings": {
                "exact_1_0_boundary_hits": exact_1sd_boundary_hits,
                "ac_mean_velocity_total": len(vel_cross_pairs),
                "ac_mean_velocity_up_ge_1": sum(1 for x in vel_cross_pairs if x["direction"] == "up_ge_1"),
                "ac_mean_velocity_down_lt_1": sum(1 for x in vel_cross_pairs if x["direction"] == "down_lt_1"),
                "ac_mean_velocity_pairs": vel_cross_pairs,
                "ac_max_sis_total": len(sis_cross_pairs),
                "ac_max_sis_up_ge_1": sum(1 for x in sis_cross_pairs if x["direction"] == "up_ge_1"),
                "ac_max_sis_down_lt_1": sum(1 for x in sis_cross_pairs if x["direction"] == "down_lt_1"),
                "ac_max_sis_pairs": sis_cross_pairs,
            },
        },
        "distributions_f17": dist17,
        "f15_vs_f17_comparison": f15_vs_f17_comparison,
        "task2_ablation_and_stratification": {
            "one_window_ac_empirical_check": {
                "one_window_sides_total": one_window_sides_total,
                "nonzero_counts": one_window_ac_nonzero,
            },
            "ablation_11d_8d_7d": ablation_summary,
            "stratification_1_window_transition": strat1_summary,
            "stratification_2_dl_and_dice_cells": strat2_cells,
            "stratification_2_minor_revision_band": minor_band_comparison,
        },
        "task3_orthography_and_core_sensitivity": cohorts,
        "governance_note": (
            "Descriptive distributional measurements only under F-17 noise floor. "
            "Gate 3a = SHIFT_QUANTIFIED_NOT_PASSED, Gate 3b = GATE3B_FROZEN. "
            "No classification or rating claims are made."
        ),
    }

    OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_sha = compute_sha256(OUT_SUMMARY_PATH)
    print(f"Wrote Stage B-13 summary -> {OUT_SUMMARY_PATH} (SHA-256: {out_sha})")


if __name__ == "__main__":
    main()
