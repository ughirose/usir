#!/usr/bin/env python
"""Stage F-21 Task1: assemble the four-gate (Gemini sec.23.2) evaluation for the 216 candidate (read-only on all inputs)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
V10 = HERE.parents[1] / "docs/handoff/v10"
c216 = json.loads((V10 / "paragraph_noise_floor_all_216.json").read_text(encoding="utf-8"))
c106 = json.loads((V10 / "paragraph_noise_floor_all_106.json").read_text(encoding="utf-8"))
sig = json.loads((HERE / "f21_sigma_decomp.json").read_text(encoding="utf-8"))
par = json.loads((HERE / "f21_v11_parity_and_floor_table.json").read_text(encoding="utf-8"))
b17 = json.loads((HERE.parent / "b17/b17_task2_task3_summary.json").read_text(encoding="utf-8"))
def pool(c): return c["length_bands"]["ac_sigma_pool"]
def width(c, d, b): return pool(c)["dims"][d][b]["cluster_ci_width_over_sigma"]
out = {"stage": "F-21", "gates": {}}
conc = sig["concentration"]
g1 = {"K_total": c216["kasane11"]["source"]["n_works"], "n_authors_total": c216["kasane11"]["source"]["n_authors"], "n_authors_106": 10,
      "K_M": pool(c216)["dims"]["ac_max_sis"]["M"]["n_works_contributing"], "K_L": pool(c216)["dims"]["ac_max_sis"]["L"]["n_works_contributing"],
      "K_M_K_L_target": "約105〜110 (spec 23.2)", "K_M_shortfall_vs_105": 105 - pool(c216)["dims"]["ac_max_sis"]["M"]["n_works_contributing"],
      "K_L_shortfall_vs_105": 105 - pool(c216)["dims"]["ac_max_sis"]["L"]["n_works_contributing"],
      "authors_with_multi_window_pairs": {b: {"c106": conc[f"first106_{b}"]["n_authors"], "c216": conc[f"216_{b}"]["n_authors"]} for b in "ML"},
      "effective_n_authors_invHHI": {b: {"c106": conc[f"first106_{b}"]["effective_n_authors_invHHI"], "c216": conc[f"216_{b}"]["effective_n_authors_invHHI"]} for b in "ML"},
      "top1_author_pair_share": {b: {"c106": conc[f"first106_{b}"]["top1_author_share"], "c216": conc[f"216_{b}"]["top1_author_share"]} for b in "ML"},
      "top10_author_pair_share": {b: {"c106": conc[f"first106_{b}"]["top10_author_share"], "c216": conc[f"216_{b}"]["top10_author_share"]} for b in "ML"},
      "cluster_ci_width_over_sigma": {d: {b: {"c106": width(c106, d, b), "c216": width(c216, d, b)} for b in "ML"} for d in ("ac_mean_velocity", "ac_max_sis", "ac_coupling")},
      "authors_overlap_old_new": sig["authors_overlap_first106_last110"]}
out["gates"]["gate1_author_diversity"] = g1
d216 = pool(c216)["dims"]
g2 = {d: {"pooled": d216[d]["pooled"], "pool_frozen": d216[d].get("pool_frozen"), "decision": d216[d]["decision"],
          "ratio_point": d216[d]["log_ratio_criterion"]["point_ratio"], "ratio_ci95": d216[d]["log_ratio_criterion"]["ci95_ratio"],
          "ci_within_margin": d216[d]["log_ratio_criterion"]["ci_within_equivalence_margin"]} for d in d216}
g2["pooled_dims_in_bands"] = {b: c216["length_bands"]["bands"][i]["kasane11"]["ac_multi_window_pooled_dims"] for i, b in ((1, "M"), (2, "L"))}
g2["equivalence_margin_ratio"] = d216["ac_coupling"]["log_ratio_criterion"]["equivalence_margin_ratio"]
g2["b17_coupling_remeasure"] = b17 if isinstance(b17, dict) and len(json.dumps(b17)) < 3000 else "see b17_task2_task3_summary.json"
out["gates"]["gate2_pool_consistency"] = g2
out["gates"]["gate3_sigma_stability"] = {"subset_sigma": sig["subset_sigma"], "influence": sig["influence"], "max_sis_M_decomp": sig["max_sis_M_decomp"]}
out["gates"]["gate4_downstream_parity"] = {"floors": par["floors"], "crossings": par["crossings"], "by_band": par["by_band"]}
(HERE / "f21_gate_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: out["gates"][k] for k in ("gate1_author_diversity", "gate2_pool_consistency")}, indent=1, ensure_ascii=False)[:6000])
