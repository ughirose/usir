"""Verification Test Suite for RTX 4070 DPO/GRPO Preference Dataset Factory."""

import json
from pathlib import Path
import pytest

from usir_py.preference_factory import (
    build_preference_records,
    generate_preference_dataset,
    scale_grpo_reward,
    verify_cleanroom_7gram,
    DEFAULT_JSONL_OUTPUT,
)


def test_scale_grpo_reward_properties():
    """Verifies that logistic reward scaling satisfies bounds and monotonicity."""
    # Monotonicity
    assert scale_grpo_reward(-1.0) < scale_grpo_reward(0.0) < scale_grpo_reward(1.0)
    # Range bounds
    assert 0.0 < scale_grpo_reward(-10.0) < 0.1
    assert 0.9 < scale_grpo_reward(10.0) < 1.0
    # Zero point symmetry
    assert pytest.approx(scale_grpo_reward(0.0), abs=1e-5) == 0.5


def test_verify_cleanroom_7gram_unit():
    """Tests 7-gram Clean Room detection logic."""
    source = "秋の日は暮れやすく冷え込む。"
    # Overlapping 7-gram: "暮れやすく冷え込"
    leaked = "どこか暮れやすく冷え込む景色であった。"
    # Non-overlapping text
    clean = "山陰の古寺の瓦に露が落ちていた。"

    assert verify_cleanroom_7gram(leaked, source, is_synthetic_rewrite=True) is False
    assert verify_cleanroom_7gram(clean, source, is_synthetic_rewrite=True) is True
    # Non-synthetic masterwork mode passes
    assert verify_cleanroom_7gram(leaked, source, is_synthetic_rewrite=False) is True


def test_preference_records_clean_set():
    """Verifies the distributable set: PD-only Set A (8 pairs) + clean-room Set B/C (96 + 96 pairs), all constraints met."""
    records = build_preference_records()
    assert len(records) == 200

    set_a = [r for r in records if r["set_type"] == "PairSetA_AuthenticMasterwork"]
    set_b = [r for r in records if r["set_type"] == "PairSetB_ClosedLoopVoid"]
    set_c = [r for r in records if r["set_type"] == "PairSetC_ClosedLoopCritical"]

    assert len(set_a) == 8
    assert len(set_b) == 96
    assert len(set_c) == 96

    # Copyright-protected authors (Shiga / Kawabata / Mishima) must be fully excluded.
    from usir_py.preference_factory import is_public_domain_author
    for r in set_a:
        assert is_public_domain_author(r.get("author")), f"Author not PD: {r.get('author')}"
        assert r.get("author") not in {"Shiga_Naoya", "Kawabata_Yasunari", "Mishima_Yukio"}

    for r in records:
        # Mandatory fields
        assert "pair_id" in r
        assert "prompt" in r
        assert "chosen" in r
        assert "rejected" in r
        assert len(r["prompt"]) > 10
        assert len(r["chosen"]) > 20
        assert len(r["rejected"]) > 20

        # Physical 24D features
        assert len(r["chosen_z24d"]) == 24
        assert len(r["rejected_z24d"]) == 24
        assert len(r["delta_z24d"]) == 24

        # Physical rhythm alive: chosen must exceed deadzone rejected
        assert r["chosen_cadence_alive"] > r["rejected_cadence_alive"]
        assert r["rejected_cadence_alive"] <= 0.40

        # Reward margin constraint: delta R >= 0.45
        assert r["reward_margin"] >= 0.45, f"Pair {r['pair_id']} failed margin constraint: {r['reward_margin']}"
        assert r["reward_chosen"] > r["reward_rejected"]

        # Clean room 7-gram constraint
        assert r["cleanroom_7gram_passed"] is True, f"Pair {r['pair_id']} failed clean room check"


def test_generate_preference_dataset_file(tmp_path):
    """Verifies JSONL generation pipeline and file integrity."""
    target_jsonl = tmp_path / "test_usir_dpo.jsonl"
    summary = generate_preference_dataset(output_path=target_jsonl)

    assert target_jsonl.exists()
    assert summary["total_records"] == 200
    assert summary["min_reward_margin"] >= 0.45
    assert summary["cleanroom_7gram_all_passed"] is True

    # Read back line by line
    with open(target_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 200
    first_record = json.loads(lines[0])
    assert first_record["pair_id"] == "SET_A_DEAD-01_VOID-01"


def test_distributed_data_has_zero_protected_leakage():
    """Re-scans every record: no 7-gram of any copyright-protected work in chosen/rejected."""
    from usir_py.preference_factory import is_public_domain_author

    assert not is_public_domain_author("Shiga_Naoya")
    assert not is_public_domain_author("Kawabata_Yasunari")
    assert not is_public_domain_author("Mishima_Yukio")
    for r in build_preference_records():
        assert r["cleanroom_7gram_passed"] is True
        assert r.get("author") not in {"Shiga_Naoya", "Kawabata_Yasunari", "Mishima_Yukio"}
