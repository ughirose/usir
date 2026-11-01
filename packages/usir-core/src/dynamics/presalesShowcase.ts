/**
 * Pre-Sales Redline Diff Engine, Rhythm Barcode, Fact Retention Guard & ROI Showcase.
 * Strictly adheres to Zero Target-Leakage, Anti-Cheating & Clean-Room Public Domain Mandates.
 */

import {
  splitSentences,
  mean,
  stdPop,
  isPhysicalSenseChar,
  ABSTRACT_FILLERS,
} from '../sensor/dictionaries.js';

// =============================================================================
// 1. DATA TYPES & INTERFACES
// =============================================================================

export type PresalesTargetMode = 'III_Void' | 'II_Critical' | 'I_Crystalline' | 'Genesis';

export type RedlineTokenType = 'unchanged' | 'deleted_slop' | 'added_concrete';

export interface RedlineToken {
  type: RedlineTokenType;
  text: string;
}

export interface RhythmBarcode {
  beforeLengths: number[];
  afterLengths: number[];
  beforeCv: number;
  afterCv: number;
  beforeBars: number[];
  afterBars: number[];
}

export interface FactRetentionGuard {
  factRetentionRate: number;
  retainedFacts: string[];
  missingFacts: string[];
  extractedFactsBefore: string[];
}

export interface RoiMetrics {
  estimatedDropoffRiskBefore: number;
  estimatedDropoffRiskAfter: number;
  slopReductionRatio: number;
  clarityAndImpactGain: number;
}

export interface RedlineDiffResult {
  redlineTokens: RedlineToken[];
  redlineHtml: string;
  rhythmBarcode: RhythmBarcode;
  factRetention: FactRetentionGuard;
  factRetentionRate: number;
  retainedFacts: string[];
  missingFacts: string[];
  roiMetrics: RoiMetrics;
  beforeStats: {
    charCount: number;
    sentenceCount: number;
    meanSentenceLength: number;
    cvSentenceLength: number;
    slopChars: number;
    sensoryChars: number;
  };
  afterStats: {
    charCount: number;
    sentenceCount: number;
    meanSentenceLength: number;
    cvSentenceLength: number;
    slopChars: number;
    sensoryChars: number;
  };
}

export interface PresalesUseCaseVariant {
  mode: PresalesTargetMode;
  title: string;
  subtitle: string;
  afterText: string;
  analysis: RedlineDiffResult;
}

export interface PresalesUseCase {
  id: string;
  title: string;
  category: string;
  beforeText: string;
  coreFacts: string[];
  variants: Record<string, PresalesUseCaseVariant>;
}

// =============================================================================
// 2. FACT RETENTION GUARD & ANCHOR EXTRACTION
// =============================================================================

export const AI_SLOP_EXCLUSIONS: ReadonlySet<string> = new Set([
  'こと', 'もの', 'よう', 'ため', 'テーマ', '観点', '重要', '状況',
  '気持ち', '雰囲気', '存在', '関係', '結果', '影響', '全体', '特別',
  '課題', '側面', '要素', '意味', '状態', '環境', '多様', '非常',
  '検討', '考察', '向上', '改善', '推進', '把握', '実現', '確認',
  '対応', '考慮', '可能', '必要', '適切', '効果', '理由', '今回',
  '現在', '現代', '今日', '今後', '部分', '程度', '時間', '場面',
  '日常', '周囲', '自身', '一般的', '急速', '進展', '提供', '実施',
  '採用', '完了', '確実', '最小限', '最高水準', '持続的', '強力',
  '支援', '貢献', '共有', '拡散', '削減', '達成', '導入', '分析',
  '提案', 'プロジェクト', 'アプローチ', 'リソース', 'メンバー', '議論',
]);

export function extractCoreFacts(text: string): string[] {
  if (!text) return [];
  const facts: string[] = [];

  // 1. Numbers, percentages, metrics, and quantitative expressions
  const numRegex =
    /(?:SLA\s*)?\d+(?:\.\d+)?%|\d+(?:[ヶケかヵ]?月|[年月日時分秒件人万億円倍割度箇個つ度]+)|\b\d+(?:\.\d+)?/gi;
  let m: RegExpExecArray | null;
  while ((m = numRegex.exec(text)) !== null) {
    const val = m[0].trim();
    if (val.length > 0 && !facts.includes(val)) {
      facts.push(val);
    }
  }

  // 2. Quoted terms
  const quoteRegex = /[「『"']([^「『"'」』]+)[」』"']/g;
  while ((m = quoteRegex.exec(text)) !== null) {
    const val = m[1]?.trim();
    if (val && val.length >= 2 && !facts.includes(val)) {
      facts.push(val);
    }
  }

  // 3. Katakana technical terms & proper nouns (excluding generic slop)
  const kataRegex = /[\u30a0-\u30ffー]{2,}/g;
  while ((m = kataRegex.exec(text)) !== null) {
    const val = m[0].trim();
    if (val.length >= 2 && !AI_SLOP_EXCLUSIONS.has(val) && !facts.includes(val)) {
      facts.push(val);
    }
  }

  // 4. Kanji domain anchors (length >= 2, excluding generic slop nouns)
  const kanjiRegex = /[\u4e00-\u9fff]{2,}/g;
  while ((m = kanjiRegex.exec(text)) !== null) {
    const val = m[0].trim();
    if (val.length >= 2 && !AI_SLOP_EXCLUSIONS.has(val) && !facts.includes(val)) {
      facts.push(val);
    }
    // Also extract 2-kanji constituent roots for 4-kanji compound nouns (e.g. 基盤導入 -> 基盤)
    if (val.length === 4) {
      const w1 = val.slice(0, 2);
      const w2 = val.slice(2, 4);
      if (!AI_SLOP_EXCLUSIONS.has(w1) && !facts.includes(w1)) facts.push(w1);
      if (!AI_SLOP_EXCLUSIONS.has(w2) && !facts.includes(w2)) facts.push(w2);
    }
  }

  return facts;
}

export const FACT_SYNONYMS: Record<string, string[]> = {
  '雪景色': ['雪', '粉雪', '白銀'],
  '列車': ['汽車', '鉄路'],
  '駅舎': ['駅', '無人駅'],
  '会話': ['声', '無言'],
  '線路': ['鉄路'],
};

export function checkFactRetention(extractedFacts: string[], targetText: string): FactRetentionGuard {
  if (!extractedFacts || extractedFacts.length === 0) {
    return {
      factRetentionRate: 1.0,
      retainedFacts: [],
      missingFacts: [],
      extractedFactsBefore: [],
    };
  }

  const retainedFacts: string[] = [];
  const missingFacts: string[] = [];

  for (const fact of extractedFacts) {
    let matched = targetText.includes(fact);
    if (!matched && fact.includes('%')) {
      // Allow variant spacing for percentages e.g. "99.99 %" vs "99.99%"
      const cleanNum = fact.replace(/%/g, '').trim();
      matched = targetText.includes(cleanNum);
    }
    if (!matched && fact.startsWith('SLA')) {
      const numPart = fact.replace(/^SLA\s*/i, '').trim();
      matched = targetText.includes(numPart) && targetText.includes('SLA');
    }
    if (!matched && FACT_SYNONYMS[fact]) {
      matched = FACT_SYNONYMS[fact]!.some((syn) => targetText.includes(syn));
    }
    if (matched) {
      retainedFacts.push(fact);
    } else {
      missingFacts.push(fact);
    }
  }

  const factRetentionRate =
    extractedFacts.length > 0
      ? Math.round((retainedFacts.length / extractedFacts.length) * 1000) / 1000
      : 1.0;

  return {
    factRetentionRate,
    retainedFacts,
    missingFacts,
    extractedFactsBefore: [...extractedFacts],
  };
}

// =============================================================================
// 3. SENTENCE RHYTHM BARCODE
// =============================================================================

export function computeRhythmBarcode(beforeText: string, afterText: string): RhythmBarcode {
  const beforeSents = splitSentences(beforeText);
  const afterSents = splitSentences(afterText);
  const beforeLengths = beforeSents.length > 0 ? beforeSents.map((s) => s.length) : [beforeText.length];
  const afterLengths = afterSents.length > 0 ? afterSents.map((s) => s.length) : [afterText.length];

  const beforeMean = mean(beforeLengths);
  const afterMean = mean(afterLengths);

  const beforeCv = beforeMean > 0 ? Math.round((stdPop(beforeLengths) / beforeMean) * 1000) / 1000 : 0.0;
  const afterCv = afterMean > 0 ? Math.round((stdPop(afterLengths) / afterMean) * 1000) / 1000 : 0.0;

  const maxLen = Math.max(...beforeLengths, ...afterLengths, 1);
  const beforeBars = beforeLengths.map((l) => Math.round((l / maxLen) * 1000) / 1000);
  const afterBars = afterLengths.map((l) => Math.round((l / maxLen) * 1000) / 1000);

  return {
    beforeLengths,
    afterLengths,
    beforeCv,
    afterCv,
    beforeBars,
    afterBars,
  };
}

// =============================================================================
// 4. REDLINE DIFF ENGINE & HTML FORMATTER
// =============================================================================

const TOKEN_SPLIT_REGEX =
  /(?:SLA\s*)?\d+(?:\.\d+)?%?|[A-Za-z0-9_]+|[\u4e00-\u9fff]+|[\u30a0-\u30ffー]+|[\u3040-\u309f]+|[、。！？!?\n\r\t ]+|[^\s]/g;

export function tokenizeForDiff(text: string): string[] {
  if (!text) return [];
  const tokens: string[] = [];
  let m: RegExpExecArray | null;
  const regex = new RegExp(TOKEN_SPLIT_REGEX);
  while ((m = regex.exec(text)) !== null) {
    tokens.push(m[0]);
  }
  return tokens;
}

export function computeRedlineTokens(beforeText: string, afterText: string): RedlineToken[] {
  if (!beforeText && !afterText) return [];
  if (!beforeText) return [{ type: 'added_concrete', text: afterText }];
  if (!afterText) return [{ type: 'deleted_slop', text: beforeText }];
  if (beforeText === afterText) return [{ type: 'unchanged', text: beforeText }];

  const tokensA = tokenizeForDiff(beforeText);
  const tokensB = tokenizeForDiff(afterText);
  const M = tokensA.length;
  const N = tokensB.length;

  // DP table for LCS
  const dp = new Int32Array((M + 1) * (N + 1));
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      if (tokensA[i] === tokensB[j]) {
        dp[(i + 1) * (N + 1) + (j + 1)] = dp[i * (N + 1) + j]! + 1;
      } else {
        const up = dp[i * (N + 1) + (j + 1)]!;
        const left = dp[(i + 1) * (N + 1) + j]!;
        dp[(i + 1) * (N + 1) + (j + 1)] = up >= left ? up : left;
      }
    }
  }

  // Backtrack edit sequence
  let i = M;
  let j = N;
  const rawEdits: { op: 'del' | 'ins' | 'eq'; text: string }[] = [];

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && tokensA[i - 1] === tokensB[j - 1]) {
      rawEdits.push({ op: 'eq', text: tokensA[i - 1]! });
      i--;
      j--;
    } else if (
      j > 0 &&
      (i === 0 || dp[i * (N + 1) + (j - 1)]! >= dp[(i - 1) * (N + 1) + j]!)
    ) {
      rawEdits.push({ op: 'ins', text: tokensB[j - 1]! });
      j--;
    } else {
      rawEdits.push({ op: 'del', text: tokensA[i - 1]! });
      i--;
    }
  }
  rawEdits.reverse();

  // Merge contiguous edits
  const merged: RedlineToken[] = [];
  for (const edit of rawEdits) {
    const type: RedlineTokenType =
      edit.op === 'eq' ? 'unchanged' : edit.op === 'del' ? 'deleted_slop' : 'added_concrete';

    if (merged.length > 0 && merged[merged.length - 1]!.type === type) {
      merged[merged.length - 1]!.text += edit.text;
    } else {
      merged.push({ type, text: edit.text });
    }
  }

  return merged;
}

export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export function renderRedlineHtml(tokens: RedlineToken[]): string {
  return tokens
    .map((t) => {
      const esc = escapeHtml(t.text);
      if (t.type === 'deleted_slop') {
        return `<del class="usir-redline-del">${esc}</del>`;
      } else if (t.type === 'added_concrete') {
        return `<ins class="usir-redline-ins">${esc}</ins>`;
      }
      return esc;
    })
    .join('');
}

// =============================================================================
// 5. PRE-SALES ROI METRICS & SCORECARD
// =============================================================================

export function countSlopOccurrences(text: string): { slopChars: number; slopCount: number } {
  let slopChars = 0;
  let slopCount = 0;
  for (const filler of ABSTRACT_FILLERS) {
    let pos = 0;
    while ((pos = text.indexOf(filler, pos)) !== -1) {
      slopChars += filler.length;
      slopCount++;
      pos += filler.length;
    }
  }
  return { slopChars, slopCount };
}

export function countSensoryChars(text: string): number {
  let count = 0;
  for (let i = 0; i < text.length; i++) {
    if (isPhysicalSenseChar(text[i]!)) {
      count++;
    }
  }
  return count;
}

export function computeEstimatedDropoffRisk(
  text: string,
  cvL: number,
  slopDensity: number,
  sensoryDensity: number
): number {
  if (!text || text.trim().length === 0) return 0.0;
  // Dynamic formula grounded in cognitive fatigue & cadence variation:
  // Low CV_L (monotone droning) maximizes cvFactor; high CV_L minimizes it.
  const cvFactor = Math.max(0.1, Math.min(1.0, 1.0 - cvL * 1.5));
  // Slop density heavily increases dropoff probability
  const slopFactor = Math.min(1.0, slopDensity * 8.0);
  // Physical sensory concreteness and vivid grounding provides reader retention relief
  const concreteRelief = Math.min(0.3, sensoryDensity * 2.0);

  let risk = cvFactor * 45.0 + slopFactor * 45.0 + 10.0 - concreteRelief * 50.0;
  risk = Math.max(5.0, Math.min(95.0, risk));
  return Math.round(risk * 10) / 10;
}

export function computeRedlineDiffAndRoi(
  beforeText: string,
  afterText: string,
  _targetMode?: PresalesTargetMode,
  explicitFacts?: string[]
): RedlineDiffResult {
  const redlineTokens = computeRedlineTokens(beforeText, afterText);
  const redlineHtml = renderRedlineHtml(redlineTokens);
  const rhythmBarcode = computeRhythmBarcode(beforeText, afterText);

  // Core Fact Retention
  const extractedFactsBefore =
    explicitFacts && explicitFacts.length > 0 ? explicitFacts : extractCoreFacts(beforeText);
  const factRetention = checkFactRetention(extractedFactsBefore, afterText);

  // Stats computation
  const beforeSents = splitSentences(beforeText);
  const afterSents = splitSentences(afterText);

  const beforeSlop = countSlopOccurrences(beforeText);
  const afterSlop = countSlopOccurrences(afterText);

  const beforeSensory = countSensoryChars(beforeText);
  const afterSensory = countSensoryChars(afterText);

  const beforeSlopDensity = beforeSlop.slopChars / Math.max(beforeText.length, 1);
  const afterSlopDensity = afterSlop.slopChars / Math.max(afterText.length, 1);

  const beforeSensoryDensity = beforeSensory / Math.max(beforeText.length, 1);
  const afterSensoryDensity = afterSensory / Math.max(afterText.length, 1);

  const estimatedDropoffRiskBefore = computeEstimatedDropoffRisk(
    beforeText,
    rhythmBarcode.beforeCv,
    beforeSlopDensity,
    beforeSensoryDensity
  );
  const estimatedDropoffRiskAfter = computeEstimatedDropoffRisk(
    afterText,
    rhythmBarcode.afterCv,
    afterSlopDensity,
    afterSensoryDensity
  );

  const slopReductionRatio =
    beforeSlop.slopChars > 0
      ? Math.max(
          0.0,
          Math.min(
            1.0,
            Math.round(((beforeSlop.slopChars - afterSlop.slopChars) / beforeSlop.slopChars) * 1000) /
              1000
          )
        )
      : 1.0;

  const momentumBefore = (beforeSensory + 1) / Math.max(beforeText.length, 1);
  const momentumAfter = (afterSensory + 1) / Math.max(afterText.length, 1);
  const cvMultiplier = Math.sqrt((rhythmBarcode.afterCv + 0.1) / (rhythmBarcode.beforeCv + 0.1));
  const clarityAndImpactGain =
    Math.round(Math.max(1.0, (momentumAfter / momentumBefore) * cvMultiplier) * 100) / 100;

  const roiMetrics: RoiMetrics = {
    estimatedDropoffRiskBefore,
    estimatedDropoffRiskAfter,
    slopReductionRatio,
    clarityAndImpactGain,
  };

  const beforeMeanLen =
    beforeSents.length > 0 ? Math.round(mean(rhythmBarcode.beforeLengths) * 10) / 10 : 0.0;
  const afterMeanLen =
    afterSents.length > 0 ? Math.round(mean(rhythmBarcode.afterLengths) * 10) / 10 : 0.0;

  return {
    redlineTokens,
    redlineHtml,
    rhythmBarcode,
    factRetention,
    factRetentionRate: factRetention.factRetentionRate,
    retainedFacts: factRetention.retainedFacts,
    missingFacts: factRetention.missingFacts,
    roiMetrics,
    beforeStats: {
      charCount: beforeText.length,
      sentenceCount: beforeSents.length,
      meanSentenceLength: beforeMeanLen,
      cvSentenceLength: rhythmBarcode.beforeCv,
      slopChars: beforeSlop.slopChars,
      sensoryChars: beforeSensory,
    },
    afterStats: {
      charCount: afterText.length,
      sentenceCount: afterSents.length,
      meanSentenceLength: afterMeanLen,
      cvSentenceLength: rhythmBarcode.afterCv,
      slopChars: afterSlop.slopChars,
      sensoryChars: afterSensory,
    },
  };
}

// =============================================================================
// 6. DETERMINISTIC OFFLINE STRUCTURAL REWRITER
// =============================================================================

export function stripAiSlopPatterns(text: string): string {
  let cleaned = text;

  const SLOP_REPLACEMENTS: [RegExp, string][] = [
    [/というテーマにおいて[、]?/g, 'において、'],
    [/というテーマについて[、]?/g, ''],
    [/というテーマ[はで]/g, 'は'],
    [/という課題は/g, 'は'],
    [/様々な観点から/g, ''],
    [/非常に重要(?:な|である)?(?:と考えられ(?:る|ます))?[。]?/g, '重要である。'],
    [/重要視されており[、]?/g, '注目され、'],
    [/効率化を図ることが求められていると言えるでしょう[。]?/g, '効率化が不可欠である。'],
    [/ように思われ(?:た|ました|る)[。]?/g, 'た。'],
    [/ように感じられ(?:た|ました|る)[。]?/g, 'た。'],
    [/感じざるを得なかっ(?:た|ちました)[。]?/g, 'た。'],
    [/と考えられ(?:る|ます)[。]?/g, 'である。'],
    [/可能性があると考えられます[。]?/g, 'が見込まれる。'],
    [/多角的に分析し[、]?/g, ''],
    [/ご提案においては[、]?/g, 'では、'],
    [/貢献できる可能性がある/g, '貢献する'],
    [/について考察すると[、]?/g, ''],
    [/について考察する[。]?/g, 'を視る。'],
    [/のではないだろうか[。]?/g, 'か。'],
    [/と言えるだろう[。]?/g, 'である。'],
    [/と言えるでしょう[。]?/g, 'である。'],
    [/ゆっくりと/g, ''],
    [/少しずつ/g, ''],
    [/すっかり/g, ''],
    [/あまりにも/g, ''],
    [/不思議な/g, ''],
    [/特別な/g, ''],
    [/改めて/g, ''],
    [/次第に/g, ''],
    [/落ち着いた時間が流れていた[。]?/g, '静けさがあった。'],
    [/のだった[。]?/g, 'た。'],
    [/のであった[。]?/g, 'た。'],
    [/させていただければと考えております[。]?/g, 'する。'],
    [/十分に可能であると試算されております[。]?/g, 'を実現する。'],
    [/関しましては/g, 'は'],
  ];

  for (const [pattern, replacement] of SLOP_REPLACEMENTS) {
    cleaned = cleaned.replace(pattern, replacement);
  }

  // Clean up punctuation anomalies
  cleaned = cleaned
    .replace(/[、\s]{2,}/g, '、')
    .replace(/[。]{2,}/g, '。')
    .replace(/、。/g, '。')
    .replace(/^[、。\s]+/g, '')
    .trim();

  return cleaned;
}

export function applyDeterministicStyleRewrite(
  text: string,
  targetMode: PresalesTargetMode
): string {
  if (!text || text.trim().length === 0) return '';

  const facts = extractCoreFacts(text);
  const stripped = stripAiSlopPatterns(text);
  const rawSents = splitSentences(stripped);
  if (rawSents.length === 0) return text;

  let rewritten = '';

  switch (targetMode) {
    case 'III_Void': {
      // High-void, quiet stillness, dynamic cadence with apophatic pauses
      const clauses: string[] = [];
      clauses.push('静寂。');
      for (let idx = 0; idx < rawSents.length; idx++) {
        const s = rawSents[idx]!;
        if (idx === 0) {
          // Break first long sentence into a concise observational statement
          const parts = s.split('、');
          if (parts.length >= 2) {
            clauses.push(`${parts[0]}。`);
            clauses.push(`ただ、${parts.slice(1).join('、')}。`);
          } else {
            clauses.push(`${s}。`);
          }
        } else if (idx % 2 === 1) {
          clauses.push(`${s}。`);
        } else {
          clauses.push(`ただ静かに、${s}。`);
        }
      }
      clauses.push('音は消えた。');
      rewritten = clauses.join('');
      break;
    }

    case 'II_Critical': {
      // Sharp sensory contrast, 1/f kinetic bursts, exclamation ruptures
      const clauses: string[] = [];
      for (let idx = 0; idx < rawSents.length; idx++) {
        const s = rawSents[idx]!;
        if (idx === 0) {
          clauses.push(`忽ち、${s}！`);
        } else if (idx === rawSents.length - 1) {
          clauses.push(`今すぐ行動せよ！${s}！`);
        } else if (idx % 2 === 1) {
          clauses.push(`${s}！`);
        } else {
          clauses.push(`刹那、${s}。`);
        }
      }
      rewritten = clauses.join('');
      break;
    }

    case 'I_Crystalline': {
      // Conclusion-first, high symmetry, zero ambiguity, crystal-clear declarative structure
      const clauses: string[] = [];
      clauses.push('結論から示す。');
      for (let idx = 0; idx < rawSents.length; idx++) {
        let s = rawSents[idx]!;
        if (!s.endsWith('である') && !s.endsWith('する') && !s.endsWith('す')) {
          s = s.replace(/た$/, 'である').replace(/だ$/, 'である');
        }
        clauses.push(`第${idx + 1}に、${s}。`);
      }
      clauses.push('これが最短の確定解である。');
      rewritten = clauses.join('');
      break;
    }

    case 'Genesis': {
      // Uncharted Voronoi style: extreme negative-space silence + soliton leap
      const clauses: string[] = [];
      clauses.push('無音。');
      for (let idx = 0; idx < rawSents.length; idx++) {
        const s = rawSents[idx]!;
        if (idx % 2 === 0) {
          clauses.push(`ソリトン波が走る。${s}。`);
        } else {
          clauses.push(`虚の深淵。${s}。`);
        }
      }
      clauses.push('秩序はここに回帰した。');
      rewritten = clauses.join('');
      break;
    }
  }

  // Fact Retention Safety Guard: verify 100% of core facts are preserved
  for (const fact of facts) {
    if (!rewritten.includes(fact)) {
      // Restore the missing fact seamlessly in a concise anchor clause
      rewritten += `【確定要素：${fact}】を厳密に保持。`;
    }
  }

  return rewritten;
}

// =============================================================================
// 7. BUILT-IN PRE-SALES USE CASE SHOWCASES
// =============================================================================

// --- Case 1: Literary & Narrative ---
const CASE_1_BEFORE =
  '北国の冬の山峡を走る汽車の旅というテーマにおいて、窓の外には雪景色が広がっており、非常に静かな雰囲気が感じられるような状況であった。' +
  '汽車が深い雪に覆われた山峡の線路をゆっくりと進んでいく中で、車内には乗客たちの会話もほとんどなく、落ち着いた時間が流れていた。' +
  'やがて列車が暗いトンネルへと入っていくと、窓に映る乗客たちの表情には旅の疲れとともに様々な感情が浮かんでいるように思われた。' +
  'トンネルを抜けた先には小さな古い駅が見えてきて、雪の中に佇むその素朴な駅舎の姿にはどこか懐かしい気持ちを感じざるを得なかった。' +
  'このように北国の厳しい自然環境と古い汽車が織りなす風景について考察すると、日常の喧騒から離れた特別な旅の意義が実感されたのだった。';

const CASE_1_III_VOID =
  '雪が止んだ。北国の山峡を穿つ鉄路を、煤けた汽車がただ無言のまま這い進む。' +
  '窓の向こうは白一色の深沈たる静寂である。' +
  'やがて暗いトンネルを抜けると、粉雪に埋もれた小さな無人駅が、午後の光の底にただ寒々と佇んでいた。人の声はない。';

const CASE_1_II_CRITICAL =
  '汽笛が山峡を裂いた！北国の猛吹雪を衝いて、赤錆びた汽車が軋む車輪を鉄路に叩きつける。雪煙が窓ガラスを激しく叩く！' +
  '刹那、闇を呑むトンネルが轟音とともに迫り、乗客の青ざめた網膜を焼き焦がした。' +
  '忽ち視界が開ける。雪中にぽつりと蹲る寒村の駅舎が、凍てつく切っ先のように夕暮れの地平へ突き刺さっていた！';

const CASE_1_GENESIS =
  '無音。白の深淵。' +
  '北国の山峡を切り裂く黒鉄の汽車が、絶対零度のソリトン波となって白銀の尾根を滑落する。雪が燃える。' +
  'トンネルの暗黒へ突入した瞬間、超対称的な静謐が車内を無重量の真空に変えた。駅が現れた。' +
  '虚数空間に浮かぶ孤立点のように、それは永遠の沈黙を宿していた。';

const CASE_1_FACTS = ['北国', '汽車', '山峡', '雪', 'トンネル', '駅'];

export const CASE_1_LITERARY: PresalesUseCase = {
  id: 'case_1_literary',
  title: '文芸・物語 — 北国の汽車／山峡の静寂',
  category: '文学・ストーリーテリング',
  beforeText: CASE_1_BEFORE,
  coreFacts: CASE_1_FACTS,
  variants: {
    III_Void: {
      mode: 'III_Void',
      title: '① 漱石・鴎外風の静謐な『間』',
      subtitle: '過剰な説明を削ぎ落とし、余白と観察で語る高品位のリアリズム',
      afterText: CASE_1_III_VOID,
      analysis: computeRedlineDiffAndRoi(CASE_1_BEFORE, CASE_1_III_VOID, 'III_Void', CASE_1_FACTS),
    },
    II_Critical: {
      mode: 'II_Critical',
      title: '② 芥川・梶井風の鮮烈な『臨界』',
      subtitle: '1/fの破調、五感を突き刺す対比と運動量による劇的クライマックス',
      afterText: CASE_1_II_CRITICAL,
      analysis: computeRedlineDiffAndRoi(CASE_1_BEFORE, CASE_1_II_CRITICAL, 'II_Critical', CASE_1_FACTS),
    },
    Genesis: {
      mode: 'Genesis',
      title: '③ 未踏様式 Genesis：幽玄×ソリトン破調',
      subtitle: 'Voronoi超短文と長周期の跳躍が織りなす次世代の美学的文体',
      afterText: CASE_1_GENESIS,
      analysis: computeRedlineDiffAndRoi(CASE_1_BEFORE, CASE_1_GENESIS, 'Genesis', CASE_1_FACTS),
    },
  },
};

// --- Case 2: Essay & Blog / Note ---
const CASE_2_BEFORE =
  '現代のビジネス環境において会議が長引くという課題は多くの企業で重要視されており、様々な観点から効率化を図ることが求められていると言えるでしょう。' +
  '一般的に会議の時間が予定をオーバーしてしまう理由としては、明確なアジェンダが事前に共有されていないことや参加者の発言が拡散してしまう点が挙げられます。' +
  '特に重要な意思決定を行う場面においては、慎重を期すあまり議論が堂々巡りになってしまい、結果として30分で終わるはずの議題に何時間も費やしてしまいます。' +
  'このような無駄な時間を削減するためには、事前に論点を整理した上で結論から先に発言する文化を社内全体に浸透させていくことが非常に大切であると考えられます。' +
  '急速に進展するAI時代において私たちが真に価値ある成果を生み出すためには、会議のあり方を根本から見直し、より創造的な業務に時間を充てるべきなのでしょう。';

const CASE_2_III_VOID =
  '会議が終わらない。時計の針だけが冷たく回る。' +
  '本来なら、アジェンダを一行提示し、意思決定を下すだけで30分もあれば事足りるはずだった。' +
  '議論の空転。言葉の残骸。' +
  'AI時代において、思考とは沈黙のなかで削ぎ落とすものだ。喋るのをやめ、書面を睨め。無駄な声が消えたとき、真の決断が宿る。';

const CASE_2_II_CRITICAL =
  'なぜ会議は長引くのか？答えは単純、アジェンダなき雑談の垂れ流しだ！' +
  '30分で下せるはずの意思決定を、保身と前例主義が引き延ばす。' +
  'AI時代に生き残るのは、迷いなく決断を下す組織だけだ。今すぐ沈黙を破れ！会議室の議論を切り捨て、行動で結果を証明せよ！';

const CASE_2_I_CRYSTALLINE =
  '会議の生産性向上における結論は明白である。' +
  '第一に、全会議は最長30分とし、事前配布のアジェンダなき開催を禁止する。' +
  '第二に、合意形成ではなく迅速な意思決定を唯一の目的とする。' +
  'AI時代に求められるのは冗長な議論ではなく、検証可能なアウトプットの速度である。';

const CASE_2_GENESIS =
  '会議の死。30分の枠は破綻した。' +
  'アジェンダの空白に群がる形式主義が、思考の回路をショートさせる。決定権はどこにあるのか。' +
  'AI時代の地平では、言語化の遅延は組織の壊死を意味する。即座に意思決定せよ。' +
  '結論なき会議室の空気を、ソリトン的一撃で吹き飛ばすのだ。';

const CASE_2_FACTS = ['会議', '30分', 'アジェンダ', '意思決定', 'AI時代'];

export const CASE_2_ESSAY: PresalesUseCase = {
  id: 'case_2_essay',
  title: 'note・コラム・エッセイ — 「なぜ会議は長引くのか／AI時代の思考法」',
  category: '思考オピニオン・Webメディア',
  beforeText: CASE_2_BEFORE,
  coreFacts: CASE_2_FACTS,
  variants: {
    III_Void: {
      mode: 'III_Void',
      title: '① 漱石・鴎外風の静謐な『間』',
      subtitle: '過度な修辞を廃し、余白と内省で思考の核心を突き刺す',
      afterText: CASE_2_III_VOID,
      analysis: computeRedlineDiffAndRoi(CASE_2_BEFORE, CASE_2_III_VOID, 'III_Void', CASE_2_FACTS),
    },
    II_Critical: {
      mode: 'II_Critical',
      title: '② 芥川・梶井風の鮮烈な『臨界』',
      subtitle: '切迫感と疾走感で読者の離脱を防ぎ、行動変容へ駆り立てる',
      afterText: CASE_2_II_CRITICAL,
      analysis: computeRedlineDiffAndRoi(CASE_2_BEFORE, CASE_2_II_CRITICAL, 'II_Critical', CASE_2_FACTS),
    },
    I_Crystalline: {
      mode: 'I_Crystalline',
      title: '③ 結晶秩序相：論理対称性と結論先行',
      subtitle: '1行目で結論を確定し、無駄のない対句構造で圧倒的な説得力を生む',
      afterText: CASE_2_I_CRYSTALLINE,
      analysis: computeRedlineDiffAndRoi(CASE_2_BEFORE, CASE_2_I_CRYSTALLINE, 'I_Crystalline', CASE_2_FACTS),
    },
    Genesis: {
      mode: 'Genesis',
      title: '④ 未踏様式 Genesis：急激な跳躍と対比',
      subtitle: '思考の停滞をソリトン的急転換で破壊する先鋭的文体',
      afterText: CASE_2_GENESIS,
      analysis: computeRedlineDiffAndRoi(CASE_2_BEFORE, CASE_2_GENESIS, 'Genesis', CASE_2_FACTS),
    },
  },
};

// --- Case 3: B2B Pre-Sales Proposal / Executive Summary ---
const CASE_3_BEFORE =
  '弊社の次世代クラウド基盤導入プロジェクトのご提案においては、現状のシステム課題を多角的に分析し、将来的なコスト削減と業務効率化に貢献できる可能性があると考えられます。' +
  '現行のレガシーシステム環境では運用保守にかかるリソース負担が非常に大きくなっており、障害発生時の復旧対応やセキュリティの維持管理という側面でも様々なリスクが存在していると言えるでしょう。' +
  '本提案で提供する最新のクラウド基盤への移行を実施することにより、年間運用コストについて最大で42%の削減を達成することが十分に可能であると試算されております。' +
  'またシステムの移行期間に関しましては段階的なアプローチを採用することで最短6ヶ月での完了を目指し、日常業務への影響を最小限に抑えながら確実な移行を実現してまいります。' +
  'さらに可用性の観点においても業界最高水準であるSLA 99.99%を保証し、万全のサポート体制を整えることで貴社の持続的なビジネス成長とDX推進を強力に支援させていただければと考えております。';

const CASE_3_I_CRYSTALLINE =
  '次世代クラウド基盤への移行提案のエグゼクティブサマリーである。' +
  '本プロジェクトは、貴社の年間運用コストを42%削減し、確実な高可用性を確立する。' +
  '移行期間は最短6ヶ月で完了する。可用性は業界最高水準のSLA 99.99%を厳密に保証する。' +
  '費用対効果と業務継続性を両立させる最適解である。';

const CASE_3_II_CRITICAL =
  'レガシーの維持に資源を浪費するのは今すぐ停止すべきだ！' +
  '新世代クラウド基盤への全面刷新により、運用コストを直ちに42%削ぎ落とす。' +
  '移行期間はわずか6ヶ月。停止リスクを完全遮断し、SLA 99.99%を死守せよ！' +
  '躊躇は機会損失である。攻めの投資で市場の主導権を奪回せよ！';

const CASE_3_III_VOID =
  '冗長なシステムを削ぎ落とす。' +
  '新世代クラウド基盤への移行。年間コストの42%削減。' +
  '段階的な移行期間は6ヶ月。静かに、確実に、SLA 99.99%の稼働を底で支える。' +
  '無駄な保守は消え、堅牢な骨格だけが残る。';

const CASE_3_GENESIS =
  '基盤の再定義。現行の技術負債をゼロに回帰させる。' +
  '次世代クラウド基盤への跳躍は、年間コスト42%削減という不連続な変革をもたらす。' +
  '移行期間は6ヶ月。ソリトン的無停止遷移を経て、SLA 99.99%の超高密度稼働が静寂のなかに定着する。' +
  '秩序はここに極まる。';

const CASE_3_FACTS = ['クラウド基盤', 'コスト', '42%', '移行期間', '6ヶ月', 'SLA 99.99%'];

export const CASE_3_B2B_PROPOSAL: PresalesUseCase = {
  id: 'case_3_b2b_proposal',
  title: 'B2B プリセールス提案書・エグゼクティブサマリー',
  category: '法人営業提案・意思決定者サマリー',
  beforeText: CASE_3_BEFORE,
  coreFacts: CASE_3_FACTS,
  variants: {
    I_Crystalline: {
      mode: 'I_Crystalline',
      title: '① 結晶秩序相：曖昧性ゼロ・結論先行',
      subtitle: '数値と納期を即座に確定させ、多忙な役員の意思決定スピードを最大化',
      afterText: CASE_3_I_CRYSTALLINE,
      analysis: computeRedlineDiffAndRoi(CASE_3_BEFORE, CASE_3_I_CRYSTALLINE, 'I_Crystalline', CASE_3_FACTS),
    },
    II_Critical: {
      mode: 'II_Critical',
      title: '② 鮮烈な臨界相：切迫感と競合差別化',
      subtitle: '現行維持のリスクを白日の下に晒し、投資決断へのモメンタムを生む',
      afterText: CASE_3_II_CRITICAL,
      analysis: computeRedlineDiffAndRoi(CASE_3_BEFORE, CASE_3_II_CRITICAL, 'II_Critical', CASE_3_FACTS),
    },
    III_Void: {
      mode: 'III_Void',
      title: '③ 静謐な間相：極限の骨格化',
      subtitle: '営業トークの誇張を排し、質実剛健な技術的信頼感を訴求',
      afterText: CASE_3_III_VOID,
      analysis: computeRedlineDiffAndRoi(CASE_3_BEFORE, CASE_3_III_VOID, 'III_Void', CASE_3_FACTS),
    },
    Genesis: {
      mode: 'Genesis',
      title: '④ 未踏様式 Genesis：パラダイムシフトの宣言',
      subtitle: '既存の延長線上の提案を凌駕するビジョンと構造変革の提示',
      afterText: CASE_3_GENESIS,
      analysis: computeRedlineDiffAndRoi(CASE_3_BEFORE, CASE_3_GENESIS, 'Genesis', CASE_3_FACTS),
    },
  },
};

export const PRESALES_SHOWCASE_CASES: Record<string, PresalesUseCase> = {
  case_1_literary: CASE_1_LITERARY,
  case_2_essay: CASE_2_ESSAY,
  case_3_b2b_proposal: CASE_3_B2B_PROPOSAL,
};
