"""Unit test suite for Workstream 1: Pre-Sales Redline Diff & ROI Showcase."""

import json
import pytest

from usir_py.presales_showcase import (
    RedlineToken,
    RhythmBarcode,
    FactRetentionGuard,
    RoiMetrics,
    RedlineDiffResult,
    compute_redline_tokens,
    render_redline_html,
    compute_rhythm_barcode,
    extract_core_facts,
    check_fact_retention,
    compute_redline_diff_and_roi,
    apply_deterministic_style_rewrite,
    CASE_1_LITERARY,
    CASE_2_ESSAY,
    CASE_3_B2B_PROPOSAL,
    PRESALES_SHOWCASE_CASES,
)

SAMPLE_DEADZONE = (
    "北国の冬の山峡を走る汽車の旅というテーマにおいて、窓の外には雪景色が広がっており、非常に静かな雰囲気が感じられるような状況であった。"
    "汽車が深い雪に覆われた山峡の線路をゆっくりと進んでいく中で、車内には乗客たちの会話もほとんどなく、落ち着いた時間が流れていた。"
    "やがて列車が暗いトンネルへと入っていくと、窓に映る乗客たちの表情には旅の疲れとともに様々な感情が浮かんでいるように思われた。"
)

SAMPLE_VOID_TUNED = (
    "雪が止んだ。北国の山峡を穿つ鉄路を、煤けた汽車がただ無言のまま這い進む。"
    "窓の向こうは白一色の深沈たる静寂である。"
    "やがて暗いトンネルを抜けると、粉雪に埋もれた小さな無人駅が、午後の光の底にただ寒々と佇んでいた。人の声はない。"
)


class TestRedlineTokensAndHtml:
    def test_redline_token_computation_and_invariants(self):
        tokens = compute_redline_tokens(SAMPLE_DEADZONE, SAMPLE_VOID_TUNED)
        assert len(tokens) > 0

        types = {t.type for t in tokens}
        assert "deleted_slop" in types
        assert "added_concrete" in types
        assert "unchanged" in types

        # Invariant 1: unchanged + deleted reconstructs before_text
        before_reconstructed = "".join(
            t.text for t in tokens if t.type in ("unchanged", "deleted_slop")
        )
        assert before_reconstructed == SAMPLE_DEADZONE

        # Invariant 2: unchanged + added reconstructs after_text
        after_reconstructed = "".join(
            t.text for t in tokens if t.type in ("unchanged", "added_concrete")
        )
        assert after_reconstructed == SAMPLE_VOID_TUNED

    def test_render_redline_html(self):
        tokens = compute_redline_tokens(SAMPLE_DEADZONE, SAMPLE_VOID_TUNED)
        html_output = render_redline_html(tokens)

        assert '<del class="usir-redline-del">' in html_output
        assert '<ins class="usir-redline-ins">' in html_output
        assert "<script>" not in html_output

    def test_redline_edge_cases(self):
        identical = compute_redline_tokens("テスト文章", "テスト文章")
        assert identical == [RedlineToken(type="unchanged", text="テスト文章")]

        empty = compute_redline_tokens("", "")
        assert empty == []

        added_only = compute_redline_tokens("", "新規")
        assert added_only == [RedlineToken(type="added_concrete", text="新規")]

        deleted_only = compute_redline_tokens("削除", "")
        assert deleted_only == [RedlineToken(type="deleted_slop", text="削除")]

class TestRhythmBarcode:
    def test_rhythm_barcode_transition(self):
        barcode = compute_rhythm_barcode(SAMPLE_DEADZONE, SAMPLE_VOID_TUNED)

        assert len(barcode.before_lengths) == 3
        assert len(barcode.after_lengths) == 5

        # Monotonous cadence before
        assert barcode.before_cv < 0.15

        # Living dynamic cadence after
        assert barcode.after_cv > 0.35

        # Normalized bars between 0.0 and 1.0
        for b in barcode.before_bars + barcode.after_bars:
            assert 0.0 <= b <= 1.0


class TestFactRetentionGuard:
    def test_fact_extraction_and_slop_exclusion(self):
        b2b_text = (
            "弊社の次世代クラウド基盤導入プロジェクトのご提案においては、年間運用コストについて最大で42%の削減を達成し、"
            "最短6ヶ月での完了とSLA 99.99%を保証します。"
        )
        facts = extract_core_facts(b2b_text)

        assert "42%" in facts
        assert "6ヶ月" in facts
        assert "SLA 99.99%" in facts
        assert "クラウド" in facts
        assert "基盤" in facts

        # Generic slop words excluded
        assert "こと" not in facts
        assert "提案" not in facts
        assert "プロジェクト" not in facts

    def test_fact_retention_scoring(self):
        facts = ["42%", "6ヶ月", "SLA 99.99%", "クラウド基盤"]
        target_good = (
            "新世代クラウド基盤により、コストを42%削減。移行期間は最短6ヶ月で完了し、SLA 99.99%を保証する。"
        )
        guard_good = check_fact_retention(facts, target_good)

        assert guard_good.fact_retention_rate == 1.0
        assert guard_good.missing_facts == []
        assert len(guard_good.retained_facts) == 4

        target_missing = "クラウド基盤を刷新し、運用コストを削減します。"
        guard_missing = check_fact_retention(facts, target_missing)

        assert guard_missing.fact_retention_rate < 1.0
        assert "42%" in guard_missing.missing_facts
        assert "6ヶ月" in guard_missing.missing_facts
        assert "SLA 99.99%" in guard_missing.missing_facts


class TestRoiMetrics:
    def test_roi_metrics_deterministic_and_variable(self):
        result = compute_redline_diff_and_roi(SAMPLE_DEADZONE, SAMPLE_VOID_TUNED, "III_Void")

        assert result.roi_metrics.estimated_dropoff_risk_before > 60.0
        assert result.roi_metrics.estimated_dropoff_risk_after < 30.0
        assert result.roi_metrics.slop_reduction_ratio > 0.7
        assert result.roi_metrics.clarity_and_impact_gain > 1.0

        # Verify not hardcoded
        another = compute_redline_diff_and_roi("短い文章。テストである。", SAMPLE_VOID_TUNED)
        assert (
            another.roi_metrics.estimated_dropoff_risk_before
            != result.roi_metrics.estimated_dropoff_risk_before
        )

class TestDeterministicRewriter:
    def test_deterministic_rewriter_all_modes(self):
        raw_input = (
            "この業務効率化というテーマにおいて、様々な観点から会議のあり方を見直すことが非常に重要であると考えられます。"
            "30分以内の意思決定を目指すことで、成果につながるように思われました。"
        )

        void_txt = apply_deterministic_style_rewrite(raw_input, "III_Void")
        critical_txt = apply_deterministic_style_rewrite(raw_input, "II_Critical")
        cryst_txt = apply_deterministic_style_rewrite(raw_input, "I_Crystalline")
        genesis_txt = apply_deterministic_style_rewrite(raw_input, "Genesis")

        # Slop removed
        assert "というテーマにおいて" not in void_txt
        assert "様々な観点から" not in void_txt
        assert "ように思われました" not in void_txt

        # Mode markers
        assert "静寂" in void_txt
        assert "！" in critical_txt
        assert "結論から示す" in cryst_txt
        assert "無音" in genesis_txt

        # 100% Fact retention guarantee
        facts = extract_core_facts(raw_input)
        for f in facts:
            assert f in void_txt
            assert f in critical_txt
            assert f in cryst_txt
            assert f in genesis_txt


class TestBuiltinShowcases:
    def test_builtin_showcases_integrity(self):
        assert "case_1_literary" in PRESALES_SHOWCASE_CASES
        assert "case_2_essay" in PRESALES_SHOWCASE_CASES
        assert "case_3_b2b_proposal" in PRESALES_SHOWCASE_CASES

        # Case 1
        c1 = CASE_1_LITERARY
        assert list(c1.variants.keys()) == ["III_Void", "II_Critical", "Genesis"]
        for variant in c1.variants.values():
            for fact in c1.core_facts:
                assert fact in variant.after_text
            assert variant.analysis.fact_retention_rate == 1.0
            assert variant.analysis.rhythm_barcode.after_cv > 0.35

        # Case 2
        c2 = CASE_2_ESSAY
        assert list(c2.variants.keys()) == ["III_Void", "II_Critical", "I_Crystalline", "Genesis"]
        for variant in c2.variants.values():
            for fact in c2.core_facts:
                assert fact in variant.after_text
            assert variant.analysis.fact_retention_rate == 1.0

        # Case 3
        c3 = CASE_3_B2B_PROPOSAL
        assert list(c3.variants.keys()) == ["I_Crystalline", "II_Critical", "III_Void", "Genesis"]
        for variant in c3.variants.values():
            for fact in c3.core_facts:
                assert fact in variant.after_text
            assert variant.analysis.fact_retention_rate == 1.0
            assert variant.analysis.roi_metrics.slop_reduction_ratio > 0.8

    def test_clean_room_compliance(self):
        banned_authors = ["志賀直哉", "川端康成", "三島由紀夫", "太宰治", "安部公房", "村上春樹"]
        for case in PRESALES_SHOWCASE_CASES.values():
            for author in banned_authors:
                assert author not in case.before_text
                for v in case.variants.values():
                    assert author not in v.after_text
                    assert author not in v.title
                    assert author not in v.subtitle
