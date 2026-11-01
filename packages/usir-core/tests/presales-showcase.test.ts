import { describe, it, expect } from 'vitest';
import {
  computeRedlineTokens,
  renderRedlineHtml,
  computeRhythmBarcode,
  extractCoreFacts,
  checkFactRetention,
  computeRedlineDiffAndRoi,
  applyDeterministicStyleRewrite,
  PRESALES_SHOWCASE_CASES,
  CASE_1_LITERARY,
  CASE_2_ESSAY,
  CASE_3_B2B_PROPOSAL,
} from '../src/index.js';

describe('Workstream 1: Pre-Sales Redline Diff & ROI Showcase Suite', () => {
  const sampleDeadzone =
    '北国の冬の山峡を走る汽車の旅というテーマにおいて、窓の外には雪景色が広がっており、非常に静かな雰囲気が感じられるような状況であった。' +
    '汽車が深い雪に覆われた山峡の線路をゆっくりと進んでいく中で、車内には乗客たちの会話もほとんどなく、落ち着いた時間が流れていた。' +
    'やがて列車が暗いトンネルへと入っていくと、窓に映る乗客たちの表情には旅の疲れとともに様々な感情が浮かんでいるように思われた。';

  const sampleVoidTuned =
    '雪が止んだ。北国の山峡を穿つ鉄路を、煤けた汽車がただ無言のまま這い進む。' +
    '窓の向こうは白一色の深沈たる静寂である。' +
    'やがて暗いトンネルを抜けると、粉雪に埋もれた小さな無人駅が、午後の光の底にただ寒々と佇んでいた。人の声はない。';

  describe('1. Redline Token Diff Engine & HTML Formatter', () => {
    it('computes clause/phrase-level diff with deleted_slop, added_concrete, and unchanged tokens', () => {
      const tokens = computeRedlineTokens(sampleDeadzone, sampleVoidTuned);
      expect(tokens.length).toBeGreaterThan(0);

      const hasDeleted = tokens.some((t) => t.type === 'deleted_slop');
      const hasAdded = tokens.some((t) => t.type === 'added_concrete');
      const hasUnchanged = tokens.some((t) => t.type === 'unchanged');

      expect(hasDeleted).toBe(true);
      expect(hasAdded).toBe(true);
      expect(hasUnchanged).toBe(true);

      // Invariant: joining unchanged + deleted equals beforeText
      const reconstructedBefore = tokens
        .filter((t) => t.type === 'unchanged' || t.type === 'deleted_slop')
        .map((t) => t.text)
        .join('');
      expect(reconstructedBefore).toBe(sampleDeadzone);

      // Invariant: joining unchanged + added equals afterText
      const reconstructedAfter = tokens
        .filter((t) => t.type === 'unchanged' || t.type === 'added_concrete')
        .map((t) => t.text)
        .join('');
      expect(reconstructedAfter).toBe(sampleVoidTuned);
    });

    it('renders clean HTML markup with semantic CSS classes', () => {
      const tokens = computeRedlineTokens(sampleDeadzone, sampleVoidTuned);
      const html = renderRedlineHtml(tokens);

      expect(html).toContain('<del class="usir-redline-del">');
      expect(html).toContain('<ins class="usir-redline-ins">');
      expect(html).not.toContain('<script>');
    });

    it('handles edge cases: identical, empty, and completely disjoint strings', () => {
      const identical = computeRedlineTokens('テスト文章', 'テスト文章');
      expect(identical).toEqual([{ type: 'unchanged', text: 'テスト文章' }]);

      const empty = computeRedlineTokens('', '');
      expect(empty).toEqual([]);

      const addedOnly = computeRedlineTokens('', '新規');
      expect(addedOnly).toEqual([{ type: 'added_concrete', text: '新規' }]);

      const deletedOnly = computeRedlineTokens('削除', '');
      expect(deletedOnly).toEqual([{ type: 'deleted_slop', text: '削除' }]);
    });
  });

  describe('2. Sentence Length Rhythm Barcode', () => {
    it('quantifies the transition from monotone cadence to living dynamic cadence', () => {
      const barcode = computeRhythmBarcode(sampleDeadzone, sampleVoidTuned);

      expect(barcode.beforeLengths.length).toBe(3);
      expect(barcode.afterLengths.length).toBe(5);

      // Deadzone has uniform monotonous lengths
      expect(barcode.beforeCv).toBeLessThan(0.15);

      // Tuned text has dynamic cadence
      expect(barcode.afterCv).toBeGreaterThan(0.35);

      // Normalized bar heights between 0 and 1
      for (const bar of [...barcode.beforeBars, ...barcode.afterBars]) {
        expect(bar).toBeGreaterThanOrEqual(0.0);
        expect(bar).toBeLessThanOrEqual(1.0);
      }
    });
  });

  describe('3. Fact & Proper-Noun Retention Guard', () => {
    it('extracts numbers, katakana terms, and core kanji domain anchors while excluding generic slop', () => {
      const b2bText =
        '弊社の次世代クラウド基盤導入プロジェクトのご提案においては、年間運用コストについて最大で42%の削減を達成し、最短6ヶ月での完了とSLA 99.99%を保証します。';
      const facts = extractCoreFacts(b2bText);

      expect(facts).toContain('42%');
      expect(facts).toContain('6ヶ月');
      expect(facts).toContain('SLA 99.99%');
      expect(facts).toContain('クラウド');
      expect(facts).toContain('基盤');

      // Generic slop nouns must be purged from core facts
      expect(facts).not.toContain('こと');
      expect(facts).not.toContain('提案');
      expect(facts).not.toContain('プロジェクト');
    });

    it('computes 1.0 retention when all core facts are preserved, and detects missing facts', () => {
      const facts = ['42%', '6ヶ月', 'SLA 99.99%', 'クラウド基盤'];
      const targetGood =
        '新世代クラウド基盤により、コストを42%削減。移行期間は最短6ヶ月で完了し、SLA 99.99%を保証する。';
      const guardGood = checkFactRetention(facts, targetGood);

      expect(guardGood.factRetentionRate).toBe(1.0);
      expect(guardGood.missingFacts).toEqual([]);
      expect(guardGood.retainedFacts.length).toBe(4);

      const targetMissing = 'クラウド基盤を刷新し、運用コストを削減します。';
      const guardMissing = checkFactRetention(facts, targetMissing);

      expect(guardMissing.factRetentionRate).toBeLessThan(1.0);
      expect(guardMissing.missingFacts).toContain('42%');
      expect(guardMissing.missingFacts).toContain('6ヶ月');
      expect(guardMissing.missingFacts).toContain('SLA 99.99%');
    });
  });

  describe('4. Pre-Sales ROI Metrics & Scorecard', () => {
    it('computes deterministic, variable dropoff risk and impact gain', () => {
      const result = computeRedlineDiffAndRoi(sampleDeadzone, sampleVoidTuned, 'III_Void');

      expect(result.roiMetrics.estimatedDropoffRiskBefore).toBeGreaterThan(60.0);
      expect(result.roiMetrics.estimatedDropoffRiskAfter).toBeLessThan(30.0);
      expect(result.roiMetrics.slopReductionRatio).toBeGreaterThan(0.7);
      expect(result.roiMetrics.clarityAndImpactGain).toBeGreaterThan(1.0);

      // Verify that changing the input text deterministically alters dropoff risk (NO hardcoded constant!)
      const anotherText = '短い文章。テストである。';
      const anotherResult = computeRedlineDiffAndRoi(anotherText, sampleVoidTuned);
      expect(anotherResult.roiMetrics.estimatedDropoffRiskBefore).not.toBe(
        result.roiMetrics.estimatedDropoffRiskBefore
      );
    });
  });

  describe('5. Deterministic Offline Structural Rewriter', () => {
    const rawInput =
      'この業務効率化というテーマにおいて、様々な観点から会議のあり方を見直すことが非常に重要であると考えられます。' +
      '30分以内の意思決定を目指すことで、成果につながるように思われました。';

    it('strips AI slop phrases and restructures text deterministically across modes', () => {
      const rewrittenVoid = applyDeterministicStyleRewrite(rawInput, 'III_Void');
      const rewrittenCritical = applyDeterministicStyleRewrite(rawInput, 'II_Critical');
      const rewrittenCrystalline = applyDeterministicStyleRewrite(rawInput, 'I_Crystalline');
      const rewrittenGenesis = applyDeterministicStyleRewrite(rawInput, 'Genesis');

      // Slop removed
      expect(rewrittenVoid).not.toContain('というテーマにおいて');
      expect(rewrittenVoid).not.toContain('様々な観点から');
      expect(rewrittenVoid).not.toContain('ように思われました');

      // Void features
      expect(rewrittenVoid).toContain('静寂');

      // Critical features
      expect(rewrittenCritical).toContain('！');

      // Crystalline features
      expect(rewrittenCrystalline).toContain('結論から示す');

      // Genesis features
      expect(rewrittenGenesis).toContain('無音');

      // 100% Fact Retention guarantee
      const facts = extractCoreFacts(rawInput);
      for (const f of facts) {
        expect(rewrittenVoid).toContain(f);
        expect(rewrittenCritical).toContain(f);
        expect(rewrittenCrystalline).toContain(f);
        expect(rewrittenGenesis).toContain(f);
      }
    });
  });

  describe('6. 3 Built-In Use Case Showcases', () => {
    it('provides case_1_literary, case_2_essay, and case_3_b2b_proposal with 100% fact retention', () => {
      expect(PRESALES_SHOWCASE_CASES.case_1_literary).toBeDefined();
      expect(PRESALES_SHOWCASE_CASES.case_2_essay).toBeDefined();
      expect(PRESALES_SHOWCASE_CASES.case_3_b2b_proposal).toBeDefined();

      // Case 1
      const c1 = CASE_1_LITERARY;
      expect(Object.keys(c1.variants)).toEqual(['III_Void', 'II_Critical', 'Genesis']);
      for (const variant of Object.values(c1.variants)) {
        for (const fact of c1.coreFacts) {
          expect(variant.afterText).toContain(fact);
        }
        expect(variant.analysis.factRetentionRate).toBe(1.0);
        expect(variant.analysis.rhythmBarcode.afterCv).toBeGreaterThan(0.35);
      }

      // Case 2
      const c2 = CASE_2_ESSAY;
      expect(Object.keys(c2.variants)).toEqual(['III_Void', 'II_Critical', 'I_Crystalline', 'Genesis']);
      for (const variant of Object.values(c2.variants)) {
        for (const fact of c2.coreFacts) {
          expect(variant.afterText).toContain(fact);
        }
        expect(variant.analysis.factRetentionRate).toBe(1.0);
      }

      // Case 3
      const c3 = CASE_3_B2B_PROPOSAL;
      expect(Object.keys(c3.variants)).toEqual(['I_Crystalline', 'II_Critical', 'III_Void', 'Genesis']);
      for (const variant of Object.values(c3.variants)) {
        for (const fact of c3.coreFacts) {
          expect(variant.afterText).toContain(fact);
        }
        expect(variant.analysis.factRetentionRate).toBe(1.0);
        expect(variant.analysis.roiMetrics.slopReductionRatio).toBeGreaterThan(0.8);
      }
    });

    it('strictly complies with Clean Room Public-Domain mandates (no copyrighted modern authors)', () => {
      const allText = JSON.stringify(PRESALES_SHOWCASE_CASES);
      const BANNED_MODERN_AUTHORS = ['志賀直哉', '川端康成', '三島由紀夫', '太宰治', '安部公房', '村上春樹'];
      for (const author of BANNED_MODERN_AUTHORS) {
        expect(allText).not.toContain(author);
      }
    });
  });
});
