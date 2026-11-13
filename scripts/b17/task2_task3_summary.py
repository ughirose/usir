#!/usr/bin/env python3
"""Stage B-17 Task 2 & Task 3: 106作品 vs 216作品 ノイズ床比較・K_active/K_M/K_L 集計・coupling/max_sis CI 抽出."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

USIR_ROOT = Path(r"c:\dev\usir")
NISHIKI_ROOT = Path(r"c:\dev\nishiki")

p106 = USIR_ROOT / "docs/handoff/v10/paragraph_noise_floor_all_106.json"
p216 = USIR_ROOT / "docs/handoff/v10/paragraph_noise_floor_all_216.json"
tsv216 = NISHIKI_ROOT / "docs/handoff/evidence/2026-11-14_b17/set216_paragraphs.tsv"

d106 = json.loads(p106.read_text(encoding="utf-8"))
d216 = json.loads(p216.read_text(encoding="utf-8"))
sha216 = hashlib.sha256(p216.read_bytes()).hexdigest().upper()

# TSV から正確な K_active, K_M, K_L を計算
lines = tsv216.read_text(encoding="utf-8").splitlines()[1:]
rows = [ln.split("\t") for ln in lines]
works_m, works_l, works_active = set(), set(), set()
for i in range(len(rows) - 1):
    w0, w1 = int(rows[i][0]), int(rows[i + 1][0])
    if w0 != w1:
        continue
    c0, c1 = int(rows[i][3]), int(rows[i + 1][3])
    n0, n1 = int(rows[i][4]), int(rows[i + 1][4])
    lmax = max(c0, c1)
    if n0 >= 2 and n1 >= 2:
        if 251 <= lmax <= 500:
            works_m.add(w0)
            works_active.add(w0)
        elif lmax >= 501:
            works_l.add(w0)
            works_active.add(w0)

src106 = d106["kasane11"]["source"]
src216 = d216["kasane11"]["source"]

def band_stds(d: dict, bname: str) -> tuple[int, int, dict]:
    for b in d["length_bands"]["bands"]:
        if b["name"] == bname:
            k11 = b["kasane11"]
            stds = k11.get("ac_multi_window_band_specific_per_dim_std", k11["ac_multi_window_per_dim_std"])
            return b["n_pairs"], k11["ac_multi_window_n_pairs"], stds
    raise KeyError(bname)

summary = {
    "sha256_all_216": sha216,
    "106_meta": {
        "K": src106["n_works"],
        "n_authors": src106["n_authors"],
        "n_paragraphs": src106["n_paragraphs"],
        "n_pairs": d106["kasane11"]["n_pairs"],
        "K_active": 64,
        "K_M": 53,
        "K_L": 54,
    },
    "216_meta": {
        "K": src216["n_works"],
        "n_authors": src216["n_authors"],
        "excluded_works": len(src216["excluded_works"]),
        "n_paragraphs": src216["n_paragraphs"],
        "n_pairs": d216["kasane11"]["n_pairs"],
        "K_active": len(works_active),
        "K_M": len(works_m),
        "K_L": len(works_l),
    },
    "band_comparison": {},
    "ac_sigma_pool_216": d216["length_bands"]["ac_sigma_pool"],
}

dims = ["ac_mean_velocity", "ac_max_sis", "ac_coupling"]
for bname in ("M", "L"):
    n106_all, n106_m, s106 = band_stds(d106, bname)
    n216_all, n216_m, s216 = band_stds(d216, bname)
    summary["band_comparison"][bname] = {
        "n_all_106": n106_all,
        "n_multi_106": n106_m,
        "n_all_216": n216_all,
        "n_multi_216": n216_m,
        "dims": {
            dim: {
                "sigma_106": s106[dim],
                "sigma_216": s216[dim],
                "pct_change": (s216[dim] - s106[dim]) / s106[dim] * 100.0,
            }
            for dim in dims
        },
    }

out_path = USIR_ROOT / "scripts/b17/b17_task2_task3_summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"216 SHA-256: {sha216}")
print(f"Meta 216: K={summary['216_meta']['K']}, K_active={summary['216_meta']['K_active']}, K_M={summary['216_meta']['K_M']}, K_L={summary['216_meta']['K_L']}, authors={summary['216_meta']['n_authors']}, paras={summary['216_meta']['n_paragraphs']}, pairs={summary['216_meta']['n_pairs']}")
for bname in ("M", "L"):
    bc = summary["band_comparison"][bname]
    print(f"Band {bname}: n_all {bc['n_all_106']}->{bc['n_all_216']}, n_multi {bc['n_multi_106']}->{bc['n_multi_216']}")
    for dim in dims:
        dd = bc["dims"][dim]
        print(f"  {dim:18s}: 106={dd['sigma_106']:.5f} -> 216={dd['sigma_216']:.5f} ({dd['pct_change']:+.2f}%)")

print("\n=== Task 3: ac_sigma_pool (216) ===")
for dim in ["ac_coupling", "ac_max_sis"]:
    info = summary["ac_sigma_pool_216"]["dims"][dim]
    lr = info["log_ratio_criterion"]
    print(f"{dim}: ratio={lr['point_ratio']:.4f}, CI={lr['ci95_ratio']}, within_margin={lr['ci_within_equivalence_margin']}, pooled={info['pooled']}, frozen={info['pool_frozen']}")
