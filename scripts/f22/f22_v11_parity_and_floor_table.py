#!/usr/bin/env python
"""Stage F-22 Task2/3: after switching CANONICAL_F14_NOISE_FLOOR_PATH to the 216 canonical floor.
(1) Python(default path, path=None)/Rust eval_rule v2 parity on real_revision_pairs_v1_1 (439).
(2) F-17 vs 216 canonical: abs_max distribution, band-wise abs_max>1 counts, 1SD-crossing list (pair, dim, direction).
Read-only for all frozen files."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "scripts"))
import run_stage_b13_analysis as b13, run_stage_b14_analysis as b14
from usir_py.v5 import f14_hybrid_noise_floor as f14
from usir_py.v5.f14_hybrid_noise_floor import load_f14_noise_floor

EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")
DS = REPO / "data/usir_v5_eval/real_revision_pairs_v1_1.jsonl"
V10 = REPO / "docs/handoff/v10"
F17 = V10 / "paragraph_noise_floor_all_novel.json"
C216 = V10 / "paragraph_noise_floor_all_216_canonical.json"
OUT = Path(__file__).resolve().parent / "f22_v11_parity_and_floor_table.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

assert Path(f14.CANONICAL_F14_NOISE_FLOOR_PATH).resolve() == C216.resolve(), "default path is not the 216 canonical"
rows = [json.loads(l) for l in DS.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(rows) == 439
res = {"dataset_sha256": sha(DS), "exe": str(EXE), "n_pairs": len(rows),
       "default_path": str(f14.CANONICAL_F14_NOISE_FLOOR_PATH), "default_path_sha256": sha(f14.CANONICAL_F14_NOISE_FLOOR_PATH), "floors": {}}
per = {}
for name, p, use_default in (("F17", F17, False), ("C216_canonical", C216, True)):
    recs = b13._run_rust_harvest_batch(EXE, p, rows)
    nf, prov = load_f14_noise_floor(None if use_default else p)   # default-path load for the canonical case
    bad = b14.python_rust_parity(rows, recs, nf, prov)
    amax, perdim = [], []
    for rec in recs:
        rn = rec["noise_floor_ratio_kasane11"]
        v = {k: abs(float(x)) for k, x in (rn.get("per_dim") or {}).items() if x is not None}
        perdim.append(v); amax.append(max(v.values()) if v else float("nan"))
    a = np.array(amax); per[name] = (a, perdim)
    res["floors"][name] = {"sha256": sha(p), "loaded_via_default": use_default, "parity_mismatches": len(bad), "parity_detail": bad[:10],
        "abs_max_median": float(np.nanmedian(a)), "abs_max_mean": float(np.nanmean(a)), "abs_max_max": float(np.nanmax(a)),
        "abs_max_p95": float(np.nanpercentile(a, 95)), "pairs_abs_max_gt_1": int((a > 1).sum()),
        "n_above_1_total_dims": int(sum(sum(1 for x in d.values() if x > 1) for d in perdim))}
    print(name, "mismatch", len(bad), "median", res["floors"][name]["abs_max_median"], flush=True)
a, b = per["F17"][0], per["C216_canonical"][0]
ma, mb = a > 1, b > 1
flips = []
for i, (r, da, db) in enumerate(zip(rows, per["F17"][1], per["C216_canonical"][1])):
    if ma[i] != mb[i]:
        dims = [{"dim": k, "f17": da[k], "c216": db[k]} for k in da if k in db and ((da[k] > 1) != (db[k] > 1))]
        flips.append({"index": i, "pair_id": r["pair_id"], "band": r["band"], "mixed": bool(r["mixed_window_pair"]),
                      "abs_max_f17": float(a[i]), "abs_max_c216": float(b[i]),
                      "direction": "gt1->le1" if ma[i] else "le1->gt1", "flipped_dims": dims})
res["crossings_F17_to_C216"] = {"pairs_flip_total": len(flips), "gt1_to_le1": int((ma & ~mb).sum()), "le1_to_gt1": int((~ma & mb).sum()),
    "dim_level_flips": int(sum(len(f["flipped_dims"]) for f in flips)),
    "max_abs_change_in_abs_max": float(np.nanmax(np.abs(a - b))), "median_ratio_new_over_old_abs_max": float(np.nanmedian(b / a)),
    "median_abs_max_delta": float(np.nanmedian(b - a))}
res["crossings_F17_to_C216"]["dim_level_flips_all_pairs_f21_definition"] = int(sum(1 for da, db in zip(per["F17"][1], per["C216_canonical"][1]) for k in da if k in db and ((da[k] > 1) != (db[k] > 1))))
res["crossings_F17_to_C216"]["dim_level_flips_note"] = "dim_level_flips counts only dims inside the 14 pairs whose abs_max>1 flag flipped; the all_pairs value (F-21 definition) also counts dims that flipped in pairs whose abs_max stayed >1."
res["crossing_pairs"] = flips
bands = [r["band"] for r in rows]; mixed = [bool(r["mixed_window_pair"]) for r in rows]
res["by_band"] = {}
for bd in sorted(set(bands)) + ["mixed"]:
    idx = [i for i, (x, m) in enumerate(zip(bands, mixed)) if (m if bd == "mixed" else x == bd)]
    res["by_band"][bd] = {"n": len(idx), "F17": {"median_abs_max": float(np.nanmedian(a[idx])), "n_gt1": int((a[idx] > 1).sum())},
                          "C216_canonical": {"median_abs_max": float(np.nanmedian(b[idx])), "n_gt1": int((b[idx] > 1).sum())}}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: res[k] for k in ("floors", "crossings_F17_to_C216", "by_band")}, indent=1, ensure_ascii=False))
