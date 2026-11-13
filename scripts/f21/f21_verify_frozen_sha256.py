#!/usr/bin/env python
"""Stage F-21 Task4: frozen artifact SHA-256 gate (15 files from the B-17 baseline) + canonical file SHA + sigma/margin invariants."""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
base = json.loads((HERE.parent / "b17/frozen_sha256_baseline.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
files = []; ok = True
for f, exp in base.items():
    act = sha(f); files.append({"file": f, "expected": exp, "actual": act, "unchanged": act == exp}); ok &= act == exp
extra = {}
c216 = Path(r"C:\dev\usir\docs\handoff\v10\paragraph_noise_floor_all_216.json")
can_u = Path(r"C:\dev\usir\docs\handoff\v10\paragraph_noise_floor_all_216_canonical.json")
can_n = Path(r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_f21\paragraph_noise_floor_all_216_canonical.json")
extra["candidate_216_unchanged"] = sha(c216) == "07E43FDF11E3C91D641A08FF509AB9DDFE153C9F0DBE2030B3C81DFFF199888D"
extra["canonical_sha256"] = {"usir": sha(can_u), "nishiki": sha(can_n)}
extra["canonical_copies_identical"] = sha(can_u) == sha(can_n)
a = json.loads(c216.read_text(encoding="utf-8")); b = json.loads(can_u.read_text(encoding="utf-8"))
sig = lambda d: {n: d["length_bands"]["bands"][i]["kasane11"] for i, n in ((0, "S"), (1, "M"), (2, "L"))}
extra["sigma_table_identical_candidate_vs_canonical"] = sig(a) == sig(b) and a["kasane11"]["per_dim_std"] == b["kasane11"]["per_dim_std"]
extra["margin_ratio_in_canonical"] = b["length_bands"]["ac_sigma_pool"]["dims"]["ac_coupling"]["log_ratio_criterion"]["equivalence_margin_ratio"]
extra["margin_lower_upper"] = [round(1 / extra["margin_ratio_in_canonical"], 4), extra["margin_ratio_in_canonical"]]
extra["pooled_dims_unchanged"] = all(not d["pooled"] for d in b["length_bands"]["ac_sigma_pool"]["dims"].values())
out = {"all_frozen_unchanged": ok, "n_frozen": len(files), "files": files, **extra}
(HERE / "f21_frozen_sha256_verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=1)); assert ok and extra["candidate_216_unchanged"] and extra["sigma_table_identical_candidate_vs_canonical"]
