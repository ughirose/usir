"""USIR v5 Stage B-14: evaluation rule v2 (mixed pairs -> 8D) -- unit tests, Python/Rust parity on v1 (415), artifact checks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from usir_py.v5.f14_hybrid_noise_floor import (
    EVAL_RULE_VERSION,
    KASANE8_ACTIVE_DIM_NAMES,
    SHORT_DEAD_DIMS,
    evaluate_f14_hybrid_pair,
    is_eight_dim,
    is_eight_dim_v1,
    load_f14_noise_floor,
    n_eval_rule_v2,
    window_count_kasane,
)

NF = ROOT / "docs/handoff/v10/paragraph_noise_floor_all_novel.json"
V1 = ROOT / "data/usir_v5_eval/real_revision_pairs_v1.jsonl"
SUMMARY = ROOT / "docs/handoff/v10/real_revision_pairs_v1_eval_rule_v2_summary.json"
FROZEN_F17_SUMMARY = ROOT / "docs/handoff/v10/real_revision_pairs_v1_f17_harvest_summary.json"
FROZEN_SPARSITY = ROOT / "docs/handoff/v10/sparsity_audit_106works_canonical.json"
NEW_EXE = Path(r"C:\dev\nishiki\target\f14\release\nishiki.exe")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_rule_v2_three_states():
    assert EVAL_RULE_VERSION == "v2"
    assert is_eight_dim((1, 1)) and n_eval_rule_v2((1, 1)) == "both_single_window:8"
    for w in [(1, 2), (2, 1), (1, 7), (5, 1)]:
        assert is_eight_dim(w) and not is_eight_dim_v1(w)
        assert n_eval_rule_v2(w) == "mixed_window:8"
    for w in [(2, 2), (3, 2), (2, 9)]:
        assert not is_eight_dim(w) and n_eval_rule_v2(w) == "both_multi_window:11"


def test_mixed_pair_excludes_ac_from_numerator_and_denominator():
    nf, prov = load_f14_noise_floor(NF)
    single = "あ" * 260 + "。"
    multi = ("あ" * 129 + "。") * 3
    assert window_count_kasane(single) == 1 and window_count_kasane(multi) >= 2
    base = {d: 0.3 for d in ["dc_showing", "dc_staccato", "dc_polyphony", "dc_confession", "dc_cadence", "dc_speculation", "semantic_tension", "ac_mean_crest"]}
    d_a = {**base, "ac_mean_velocity": 5.0, "ac_max_sis": 5.0, "ac_coupling": 5.0}
    d_b = {**base, "ac_mean_velocity": -9e3, "ac_max_sis": 0.0, "ac_coupling": 77.0}
    ra = evaluate_f14_hybrid_pair(single, multi, nf, delta_kasane11=d_a, provenance=prov, apply_dice_gate=False)
    rb = evaluate_f14_hybrid_pair(single, multi, nf, delta_kasane11=d_b, provenance=prov, apply_dice_gate=False)
    assert ra["n_evaluated"] == 8 and ra["n_eval_rule"] == "mixed_window:8" and ra["eval_rule"] == "v2"
    assert set(ra["ratios"]["per_dim"]) == set(KASANE8_ACTIVE_DIM_NAMES)
    assert not set(SHORT_DEAD_DIMS) & set(ra["ratios"]["per_dim"])
    assert ra["ratios"] == rb["ratios"], "ac_* の値は混在ペアの比に影響しない"
    assert ra["ac_sigma_source"] == "not_applicable"
    d7 = ra["diagnostic_7d_no_crest"]
    assert d7["diagnostic_only"] is True and d7["n_evaluated"] == 7 and "ac_mean_crest" not in d7["dims"]
    # both-multi stays 11D and has no 7D block
    rc = evaluate_f14_hybrid_pair(multi, multi, nf, delta_kasane11={**d_a}, provenance=prov, apply_dice_gate=False)
    assert rc["n_evaluated"] == 11 and rc["n_eval_rule"] == "both_multi_window:11" and rc["diagnostic_7d_no_crest"] is None


def test_b14_summary_artifact_contents():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["stage"] == "B-14" and s["eval_rule"] == "v2"
    assert s["python_rust_parity"]["n_pairs"] == 415 and s["python_rust_parity"]["discrepancy_count"] == 0
    inv = s["invariants"]
    assert inv["non_mixed_pairs"] == 331 and inv["non_mixed_pairs_changed_v1_to_v2"] == 0
    assert inv["mixed_n_evaluated_values_v1"] == [11] and inv["mixed_n_evaluated_values_v2"] == [8]
    c = s["comparison"]
    mx = c["mixed_window_pairs"]
    assert mx["before_rule_v1"]["n_pairs"] == 84
    assert mx["before_rule_v1"]["abs_max"]["median"] == pytest.approx(1.5714, abs=1e-4)
    assert mx["after_rule_v2"]["abs_max"]["median"] == pytest.approx(1.0751, abs=1e-4)
    assert mx["mixed_subset_7d_diagnostic_only"]["diagnostic_only"] is True
    assert mx["mixed_subset_7d_diagnostic_only"]["abs_max"]["median"] == pytest.approx(0.9331, abs=1e-4)
    mn = c["minor_band_mixed"]
    assert mn["before_rule_v1"]["n_pairs"] == 9
    assert mn["before_rule_v1"]["abs_max"]["median"] == pytest.approx(2.3513, abs=1e-4)
    assert mn["after_rule_v2"]["abs_max"]["median"] == pytest.approx(0.3285, abs=1e-4)
    assert c["primary_same_orthography"]["after_rule_v2"]["n_pairs"] == 381
    assert c["non_mixed_pairs"]["before_rule_v1"] == c["non_mixed_pairs"]["after_rule_v2"]
    assert s["set1_mixed_recompute"]["n_mixed"] == 1590


def test_frozen_artifacts_untouched_by_b14():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    shas = s["invariants"]["frozen_artifact_sha256_unchanged"]
    for p in (FROZEN_F17_SUMMARY, FROZEN_SPARSITY, NF):
        assert shas[str(p)] == _sha(p)
    old = json.loads(FROZEN_F17_SUMMARY.read_text(encoding="utf-8"))
    assert old["stage"] == "B-13"
    assert old["distributions_f17"]["mixed_window_pairs"]["n_evaluated_distribution"]["11"] == 84


@pytest.mark.skipif(not NEW_EXE.is_file(), reason="nishiki v2 binary not built")
def test_python_rust_parity_on_v1_415_live():
    import run_stage_b13_analysis as b13
    import run_stage_b14_analysis as b14

    rows = [json.loads(l) for l in V1.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 415
    recs = b13._run_rust_harvest_batch(NEW_EXE, NF, rows)
    nf, prov = load_f14_noise_floor(NF)
    assert b14.python_rust_parity(rows, recs, nf, prov) == []
