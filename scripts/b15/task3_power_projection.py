#!/usr/bin/env python3
"""Stage B-15 Task 3: 検出力プロジェクション（F-18 実測ブートストラップ補間 + 1/sqrt(K) 近似）."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
F18_DIR = Path(r"F:\マイドライブ\Claude-Sync\inbox\2026-11-13_f18")
OUT_JSON_PATH = REPO_ROOT / "scripts" / "b15" / "b15_task3_projection.json"

TARGET_KS = [46, 60, 80, 100, 106, 150, 219, 300, 554]
DIMS = ["ac_mean_velocity", "ac_max_sis", "ac_coupling"]


def interp_log_k(xs: List[int], ys: List[float], target_k: int) -> float:
    if target_k <= xs[0]:
        return ys[0]
    if target_k >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= target_k <= xs[i + 1]:
            if xs[i] == target_k:
                return ys[i]
            if xs[i + 1] == target_k:
                return ys[i + 1]
            t = (math.log(target_k) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def main() -> None:
    d46 = json.loads((F18_DIR / "task1_power_K46_B10000.json").read_text(encoding="utf-8"))
    dscan = json.loads((F18_DIR / "task1_power_Kscan_B2000.json").read_text(encoding="utf-8"))
    dfine = json.loads((F18_DIR / "task1_power_Kfine_B2000.json").read_text(encoding="utf-8"))

    merged: Dict[int, Dict[str, Any]] = {}
    for src in (dscan, dfine):
        for k_str, v in src["empirical_by_K"].items():
            merged[int(k_str)] = v
    # Use B=10000 at K=46 as anchor for 1/sqrt(K) and exact K=46 row
    anchor_46 = d46["empirical_by_K"]["46"]

    ks_sorted = sorted(merged.keys())
    # For interpolation, keep B=2000 series consistent across K, and report B=10000 at K=46
    rows = []
    for k in TARGET_KS:
        row: Dict[str, Any] = {"K": k}
        for di, dim in enumerate(DIMS):
            if k == 46:
                p125_emp = anchor_46["by_margin"]["1.25"]["power"][di]
                m80_emp = anchor_46["m_80"][di]
            else:
                ys_pow = [merged[kk]["by_margin"]["1.25"]["power"][di] for kk in ks_sorted]
                ys_m80 = [merged[kk]["m_80"][di] for kk in ks_sorted]
                p125_emp = interp_log_k(ks_sorted, ys_pow, k)
                m80_emp = interp_log_k(ks_sorted, ys_m80, k)

            # 1/sqrt(K) scaling approximation on log-margin: ln(m_80(K)) ≈ ln(m_80(46)) * sqrt(46 / K)
            m80_46 = anchor_46["m_80"][di]
            m80_approx = math.exp(math.log(m80_46) * math.sqrt(46.0 / k))

            row[dim] = {
                "power_125_f18_interp": round(p125_emp, 4),
                "m80_f18_interp": round(m80_emp, 4),
                "m80_sqrt_k_approx": round(m80_approx, 4),
            }
        rows.append(row)

    out = {
        "note": "power_125_f18_interp and m80_f18_interp are log-K linear interpolations of F-18 B=2000 bootstrap scan (K=46 uses B=10000). m80_sqrt_k_approx uses ln(m80(K)) = ln(m80(46)) * sqrt(46/K) approximation.",
        "f18_grid_ks": ks_sorted,
        "projections": rows,
    }
    OUT_JSON_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'K':>4} | {'vel p1.25':>9} {'vel m80(F18)':>12} {'vel m80(1/√K)':>13} | {'sis p1.25':>9} {'sis m80(F18)':>12} {'sis m80(1/√K)':>13} | {'cpl p1.25':>9} {'cpl m80(F18)':>12} {'cpl m80(1/√K)':>13}")
    print("-" * 125)
    for r in rows:
        v, s, c = r["ac_mean_velocity"], r["ac_max_sis"], r["ac_coupling"]
        print(
            f"{r['K']:>4} | {v['power_125_f18_interp']:>9.3f} {v['m80_f18_interp']:>12.3f} {v['m80_sqrt_k_approx']:>13.3f} | "
            f"{s['power_125_f18_interp']:>9.3f} {s['m80_f18_interp']:>12.3f} {s['m80_sqrt_k_approx']:>13.3f} | "
            f"{c['power_125_f18_interp']:>9.3f} {c['m80_f18_interp']:>12.3f} {c['m80_sqrt_k_approx']:>13.3f}"
        )


if __name__ == "__main__":
    main()
