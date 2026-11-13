#!/usr/bin/env python
"""Stage F-21 Task1 Gate4: Python/Rust eval_rule v2 parity on real_revision_pairs_v1_1 (439) under the 216 candidate,
plus old/new floor comparison (F-17, 106 candidate, 216 candidate) with 1SD-crossing counts. Read-only for all frozen files."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "scripts"))
import run_stage_b13_analysis as b13, run_stage_b14_analysis as b14
from usir_py.v5.f14_hybrid_noise_floor import load_f14_noise_floor

EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")
DS = REPO / "data/usir_v5_eval/real_revision_pairs_v1_1.jsonl"
V10 = REPO / "docs/handoff/v10"
FLOORS = {"F17": V10 / "paragraph_noise_floor_all_novel.json", "C106": V10 / "paragraph_noise_floor_all_106.json", "C216": V10 / "paragraph_noise_floor_all_216.json", "C216_canonical": V10 / "paragraph_noise_floor_all_216_canonical.json"}
OUT = Path(__file__).resolve().parent / "f21_v11_parity_and_floor_table.json"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
rows = [json.loads(l) for l in DS.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(rows) == 439
res = {"dataset_sha256": sha(DS), "exe": str(EXE), "n_pairs": len(rows), "floors": {}}
per = {}
for name, p in FLOORS.items():
    recs = b13._run_rust_harvest_batch(EXE, p, rows)
    nf, prov = load_f14_noise_floor(p)
    bad = b14.python_rust_parity(rows, recs, nf, prov)
    amax, above, perdim = [], [], []
    for rec in recs:
        rn = rec["noise_floor_ratio_kasane11"]
        v = {k: abs(float(x)) for k, x in (rn.get("per_dim") or {}).items() if x is not None}
        perdim.append(v); amax.append(max(v.values()) if v else float("nan")); above.append(sum(1 for x in v.values() if x > 1))
    per[name] = (np.array(amax), np.array(above), perdim)
    res["floors"][name] = {"sha256": sha(p), "parity_mismatches": len(bad), "parity_detail": bad[:10],
        "abs_max_median": float(np.nanmedian(amax)), "abs_max_mean": float(np.nanmean(amax)),
        "pairs_abs_max_gt_1": int(np.sum(np.array(amax) > 1)), "n_above_1_total_dims": int(sum(above)),
        "n_above_1_median_per_pair": float(np.median(above))}
    print(name, "mismatch", len(bad), "median", res["floors"][name]["abs_max_median"], flush=True)
bands = [r["band"] for r in rows]; mixed = [bool(r["mixed_window_pair"]) for r in rows]
def cross(a, b):
    ma, mb = per[a][0] > 1, per[b][0] > 1
    pd_flip = sum(1 for da, db in zip(per[a][2], per[b][2]) for k in da if k in db and ((da[k] > 1) != (db[k] > 1)))
    return {"pairs_flip_total": int((ma != mb).sum()), "gt1_to_le1": int((ma & ~mb).sum()), "le1_to_gt1": int((~ma & mb).sum()),
            "dim_level_flips": int(pd_flip),
            "max_abs_change_in_abs_max": float(np.nanmax(np.abs(per[a][0] - per[b][0]))),
            "median_ratio_new_over_old_abs_max": float(np.nanmedian(per[b][0] / per[a][0]))}
res["crossings"] = {"F17_to_C216": cross("F17", "C216"), "C106_to_C216": cross("C106", "C216"), "F17_to_C106": cross("F17", "C106")}
res["by_band"] = {}
for bd in sorted(set(bands)) + ["mixed"]:
    idx = [i for i, (b, m) in enumerate(zip(bands, mixed)) if (m if bd == "mixed" else b == bd)]
    res["by_band"][bd] = {"n": len(idx), **{n: {"median_abs_max": float(np.nanmedian(per[n][0][idx])), "n_gt1": int((per[n][0][idx] > 1).sum())} for n in per}}
res["v1_subset_415"] = {n: {"n_gt1": int((per[n][0][:415] > 1).sum())} for n in per}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: res[k] for k in ("floors", "crossings", "by_band")}, indent=1, ensure_ascii=False))
