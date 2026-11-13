#!/usr/bin/env python
"""Stage F-21 Task2: write the canonical 216-work noise floor as NEW files (never overwrites). Source: nishiki regen output
(identical to the B-17 candidate except the F-21 freeze_notes annotation, verified before this script is run)."""
import json, hashlib, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
SRC = Path(r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_f21\regen_all_216_with_notes.json")
DESTS = [REPO / "docs/handoff/v10/paragraph_noise_floor_all_216_canonical.json",
         Path(r"C:\dev\nishiki\docs\handoff\evidence\2026-11-14_f21\paragraph_noise_floor_all_216_canonical.json")]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
for d in DESTS:
    assert not d.exists(), f"refuse to overwrite {d}"
g = json.loads((REPO / "scripts/f21/f21_gate_eval.json").read_text(encoding="utf-8"))["gates"]
d = json.loads(SRC.read_text(encoding="utf-8"))
src = d["kasane11"]["source"]; pool = d["length_bands"]["ac_sigma_pool"]["dims"]["ac_max_sis"]
d["canonical_adoption"] = {
    "stage": "F-21", "status": "canonical", "adopted_on": "2026-11-14",
    "inherits_from": {
        "f17_floor": {"role": "history_and_benchmark_reference (frozen, unchanged)", "file": "paragraph_noise_floor_all_novel.json", "sha256": "0575B64A75A8AF8DD1AD0DB81BDF4D8898C473CE8AC1876C6E7E2F8B61A6F604"},
        "candidate_106": {"role": "intermediate_archive (frozen, unchanged)", "file": "paragraph_noise_floor_all_106.json", "sha256": "39DA6CC8C2E52ACE514B85C9C26FD79EF41DD2320682AFF736C426ABA969526E"},
        "candidate_216": {"role": "B-17 candidate from which this file derives (unchanged)", "file": "paragraph_noise_floor_all_216.json", "sha256": "07E43FDF11E3C91D641A08FF509AB9DDFE153C9F0DBE2030B3C81DFFF199888D",
                          "difference_from_candidate": "ac_sigma_pool.dims.ac_max_sis.freeze_notes (F-21 annotation) and this canonical_adoption block only; all numeric values identical (verified by parsed-JSON equality)"}},
    "population": {"K": src["n_works"], "K_active": 121, "K_M": pool["M"]["n_works_contributing"], "K_L": pool["L"]["n_works_contributing"],
                   "n_authors": src["n_authors"], "n_paragraphs": src["n_paragraphs"], "n_adjacent_pairs": d["kasane11"]["n_pairs"]},
    "rationale_four_gates": {
        "gate1_author_diversity": "PASS (note: K_M=97/K_L=93 are below the spec 23.2 target of ~105-110; authors 10->120, effective authors (inverse HHI over [>=2,>=2] pairs) M 4.40->8.52 / L 3.95->8.31)",
        "gate2_pool_consistency": "PASS (coupling band-specific, velocity band-specific (closed), max_sis frozen band-specific; margin 0.8-1.25 unchanged)",
        "gate3_sigma_stability": "PASS (leave-one-work-out max sigma change <=3.02% (max_sis M), leave-one-author-out <=4.8%; max_sis M +8.68% is attributable to higher variance in the 110 new-author works, sigma 0.1626 vs 0.1277)",
        "gate4_downstream_parity": "PASS (Python/Rust eval_rule v2 parity mismatches 0/439 on real_revision_pairs_v1_1 under the 216 floor)"},
    "unchanged_decisions": ["ac_coupling: band-specific (CI [1.2313, 1.4514] outside [0.8, 1.25])", "ac_mean_velocity: permanently band-specific (closed)", "ac_max_sis: pool frozen (always band-specific), see freeze_notes"],
}
text = json.dumps(d, indent=2, ensure_ascii=False) + "\n"
for dest in DESTS:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8", newline="\n")
    print(dest, sha(dest))
assert sha(DESTS[0]) == sha(DESTS[1])
