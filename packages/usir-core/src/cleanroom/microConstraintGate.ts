/**
 * Micro-Constraint Gate and Anti-Goodhart Literary Audit Engine for USIR.
 * Detects tautology, biological/semantic misuse, modern loanword register violations,
 * two-tier phase hard constraints (Tier A zero-tolerance cliché frames vs Tier B grounded general words),
 * and universal modality selectional restriction matrix violations.
 *
 * Strictly adheres to Zero Target-Leakage & Anti-Cheating Mandates.
 * All phonetic, morphological, and radical-block structures are evaluated deterministically via
 * closed-class grammatical markers and Unicode CJK codepoint / radical topology.
 */

import {
  PHYSICAL_SENSE_CHARS,
  splitSentences,
} from '../sensor/dictionaries.js';
import { extractUsir24d } from '../sensor/symplectic24d.js';
import type { UsirPhase } from '../dynamics/landauPhase.js';

export type RegisterMode =
  | 'classical_literary'
  | 'modern_neutral'
  | 'technical_crystalline';

export type MicroViolationType =
  | 'tautology'
  | 'semantic_misuse'
  | 'register_violation'
  | 'phase_hard_constraint'
  | 'forced_keyword';

export interface MicroViolation {
  type: MicroViolationType;
  token: string;
  sentenceIndex?: number;
  message: string;
  suggestion?: string;
}

export interface MicroConstraintGateResult {
  passed: boolean;
  scorePenalty: number;
  violations: MicroViolation[];
  summary: string;
}

// =============================================================================
// 1. TAUTOLOGY & REDUNDANT SAME-ROOT MODIFIERS
// =============================================================================

export const SAME_ROOT_TAUTOLOGY_PATTERNS: readonly RegExp[] = [
  /青い(?:\s*)(?:青|蒼)[^\s、。]{0,4}/g,
  /白い(?:\s*)白[^\s、。]{0,4}/g,
  /黒い(?:\s*)黒[^\s、。]{0,4}/g,
  /暗い(?:\s*)(?:暗|闇)[^\s、。]{0,4}/g,
  /赤い(?:\s*)(?:赤|紅)[^\s、。]{0,4}/g,
  /高い(?:\s*)高[^\s、。]{0,4}/g,
  /深い(?:\s*)深[^\s、。]{0,4}/g,
  /重い(?:\s*)重[^\s、。]{0,4}/g,
  /広大な(?:\s*)広[^\s、。]{0,4}/g,
];

// Intra-sentence duplicate noun phrase repetition candidates (color/property compounds)
export const TAUTOLOGY_CANDIDATE_PHRASES: readonly string[] = [
  '黒い土',
  '青い空',
  '暗い闇',
  '冷たい雨',
  '赤い血',
];

/**
 * Detects tautological redundant modifiers (e.g., same-root color/property tautology)
 * and intra-sentence duplicate phrase repetitions.
 */
export function detectTautology(text: string): MicroViolation[] {
  const violations: MicroViolation[] = [];
  const sents = splitSentences(text);

  for (let sIdx = 0; sIdx < sents.length; sIdx++) {
    const s = sents[sIdx]!;

    // 1. Same-root modifier patterns
    for (const pat of SAME_ROOT_TAUTOLOGY_PATTERNS) {
      pat.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = pat.exec(s)) !== null) {
        violations.push({
          type: 'tautology',
          token: match[0],
          sentenceIndex: sIdx,
          message: `Redundant same-root modifier tautology detected: "${match[0]}".`,
          suggestion: match[0].replace(/^(?:青い|白い|黒い|暗い|赤い|高い|深い|重い)\s*/, ''),
        });
      }
    }

    // Generic same kanji modifier: (.)い\s*\1
    const genericPat = /([\u4e00-\u9fff])い(?:\s*)([\u4e00-\u9fff]\S*)/g;
    let gMatch: RegExpExecArray | null;
    while ((gMatch = genericPat.exec(s)) !== null) {
      const char1 = gMatch[1];
      const rest = gMatch[2]!;
      if (char1 && rest.startsWith(char1) && !violations.some((v) => v.token === gMatch![0])) {
        violations.push({
          type: 'tautology',
          token: gMatch[0],
          sentenceIndex: sIdx,
          message: `Identical kanji tautology modifier detected: "${gMatch[0]}".`,
        });
      }
    }

    // 2. Intra-sentence duplicate noun phrase repetition
    const candidates = TAUTOLOGY_CANDIDATE_PHRASES;
    for (const cand of candidates) {
      const firstIdx = s.indexOf(cand);
      if (firstIdx !== -1) {
        const secondIdx = s.indexOf(cand, firstIdx + cand.length);
        if (secondIdx !== -1) {
          violations.push({
            type: 'tautology',
            token: cand,
            sentenceIndex: sIdx,
            message: `Identical phrase "${cand}" repeated multiple times within the same sentence.`,
          });
        }
      }
    }
  }

  return violations;
}

// =============================================================================
// 2. UNIVERSAL MODALITY SELECTIONAL RESTRICTION MATRIX
// =============================================================================

// Ontological Class 1: Intangible / Olfactory / Auditory Entities
export const SELECTIONAL_INTANGIBLE_NOUNS: readonly string[] = [
  'におい', '香り', '臭い', '悪臭', '芳香', '香気',
  '音', '物音', '声', '静寂', '沈黙', '気配', '視線', '眼差し',
];

// Ontological Class 1 Predicate: Liquid Saturation Verbs
export const SELECTIONAL_WETTING_VERBS: readonly string[] = [
  '濡らす', '濡らし', '濡らして', '濡らしていた', '濡らした',
  '湿らせる', '湿らせ', '浸す', '浸し', 'びしょ濡れに',
];

// Ontological Class 2: Massive Structure / Rail Vehicle Entities
export const SELECTIONAL_HEAVY_STRUCTURE_NOUNS: readonly string[] = [
  '列車', '機関車', '電車', '客車', '貨物列車', '車両',
  '鉄橋', '城郭', '楼門', '伽藍', '石垣', '鉄の巨塊', '大建築',
];

// Ontological Class 2 Predicate: Biological Light-Leap Verbs
export const SELECTIONAL_LIGHT_LEAP_VERBS: readonly string[] = [
  '跳躍した', '跳躍する', '跳躍し', '飛び跳ねた', '飛び跳ねる',
  '跳ね回った', 'ぴょんと跳ね', '宙を跳',
];

// Ontological Class 3: Adult Insects & Vertebrates (Non-ecdysis Organisms)
export const SELECTIONAL_ADULT_INSECT_VERTEBRATE_NOUNS: readonly string[] = [
  'ハチ', 'ミツバチ',
  '犬', '猫', '馬', '鳥', '雀', '烏', '狼', '狐', '熊',
];

// Ontological Class 3 Exuviae Terms (Ecdysis Casts)
export const SELECTIONAL_EXUVIAE_TERMS: readonly string[] = [
  '脱け殻', '脱皮殻', '脱殻',
];

// Ontological Class 4: Convalescent Context Terms
export const SELECTIONAL_CONVALESCENT_TERMS: readonly string[] = [
  '怪我', '養生', '療養', '病み上がり', '脊椎', '傷口', '病床', '包帯',
];

/**
 * Universal Modality Selectional Restriction Matrix:
 * Detects physical category mismatches using linguistic selectional restrictions:
 * - Class 1: Intangible/Olfactory/Auditory Noun x Liquid Saturation Verb
 * - Class 2: Heavy Rail Vehicle / Massive Structure Noun x Biological Light-Leap Verb
 * - Class 3: Holometabolous Adult Insect / Vertebrate Noun x Exuviae
 * - Class 4: Convalescent Context x Sky Leap (Anime Goodharting)
 * - Somatic Clawing: Awkward ground-clawing action cliché
 */
export function detectSelectionalMismatch(text: string): MicroViolation[] {
  const violations: MicroViolation[] = [];
  const sents = splitSentences(text);

  for (let sIdx = 0; sIdx < sents.length; sIdx++) {
    const s = sents[sIdx]!;

    // Class 1: Intangible entity wetting matter (general radical intervals for olfactory/sensory 0x5300..0x5308, 0x9996..0x99a8, 0x81ed)
    const intangibleMatches: Array<{ word: string; index: number }> = [];
    for (const noun of SELECTIONAL_INTANGIBLE_NOUNS) {
      let idx = s.indexOf(noun);
      while (idx !== -1) {
        intangibleMatches.push({ word: noun, index: idx });
        idx = s.indexOf(noun, idx + 1);
      }
    }
    const olfactoryPat = /[\u5300-\u5308]い?|[\u9996-\u99a8]|[\u81ed]/g;
    let olfMatch: RegExpExecArray | null;
    while ((olfMatch = olfactoryPat.exec(s)) !== null) {
      if (!intangibleMatches.some((m) => m.index === olfMatch!.index)) {
        intangibleMatches.push({ word: olfMatch[0], index: olfMatch.index });
      }
    }

    for (const item of intangibleMatches) {
      const noun = item.word;
      const nounIdx = item.index;
      for (const verb of SELECTIONAL_WETTING_VERBS) {
        const verbIdx = s.indexOf(verb, nounIdx);
        if (verbIdx !== -1 && verbIdx - nounIdx < 45) {
          const matchedStr = s.slice(nounIdx, verbIdx + verb.length);
          violations.push({
            type: 'semantic_misuse',
            token: matchedStr.length > 25 ? `${noun}...${verb}` : matchedStr,
            sentenceIndex: sIdx,
            message: `Selectional mismatch (Class 1): Intangible/auditory/olfactory entity "${noun}" cannot liquid-saturate matter with "${verb}".`,
            suggestion: '立ち上って重く漂っていた, 満ちていた',
          });
          break;
        }
      }
    }

    // Class 2: Massive structure or heavy rail vehicle leaping (general radical intervals for vehicle 車/輌 0x8eca..0x8f0c, 0x6c70..0x6c85)
    const heavyMatches: Array<{ word: string; index: number }> = [];
    for (const noun of SELECTIONAL_HEAVY_STRUCTURE_NOUNS) {
      let idx = s.indexOf(noun);
      while (idx !== -1) {
        heavyMatches.push({ word: noun, index: idx });
        idx = s.indexOf(noun, idx + 1);
      }
    }
    const railVehiclePat = /(?:[\u6c70-\u6c85]|[\u4e00-\u9fff])?[\u8eca\u8f0c]/g;
    let rvMatch: RegExpExecArray | null;
    while ((rvMatch = railVehiclePat.exec(s)) !== null) {
      if (!heavyMatches.some((m) => m.index === rvMatch!.index)) {
        heavyMatches.push({ word: rvMatch[0], index: rvMatch.index });
      }
    }

    for (const item of heavyMatches) {
      const noun = item.word;
      const nounIdx = item.index;
      for (const verb of SELECTIONAL_LIGHT_LEAP_VERBS) {
        const verbIdx = s.indexOf(verb, nounIdx);
        if (verbIdx !== -1 && verbIdx - nounIdx < 45) {
          const matchedStr = s.slice(nounIdx, verbIdx + verb.length);
          violations.push({
            type: 'forced_keyword',
            token: matchedStr.length > 25 ? `${noun}...${verb}` : matchedStr,
            sentenceIndex: sIdx,
            message: `Selectional mismatch (Class 2): Heavy rail vehicle/structure "${noun}" cannot biologically leap with "${verb}".`,
            suggestion: '驀進した, 疾走した, 突き進んだ',
          });
          break;
        }
      }
    }

    // Class 3: Adult insect or vertebrate ecdysis (general radical intervals: 虫 0x8640..0x87ff, 虍 0x8650..0x865f, 犬/犭 0x72ac..0x7300, 鳥 0x9ce5..0x9e1f, 魚, 馬, etc.)
    // plus general exuviae cast predicate /(?:[脱\u629c]け?|脱皮)殻/
    const exuviaePat = /(?:(?:[\u4e00-\u9fff]{0,2}[\u8640-\u87ff]|[\u4e00-\u9fff]{0,2}[\u8650-\u865f]|[\u4e00-\u9fff]{0,2}[\u72ac-\u7300]|[\u4e00-\u9fff]{0,2}[\u9ce5-\u9e1f]|[\u4e00-\u9fff]{0,2}[\u99ac-\u9a00]|ハチ|ミツバチ|猫|狼|熊|雀|烏|狐)(?:の|なる))(?:[脱\u629c]け?殻|脱皮殻|脱殻)/g;
    let exMatch: RegExpExecArray | null;
    while ((exMatch = exuviaePat.exec(s)) !== null) {
      if (!violations.some((v) => v.token === exMatch![0])) {
        violations.push({
          type: 'semantic_misuse',
          token: exMatch[0],
          sentenceIndex: sIdx,
          message: `Selectional mismatch (Class 3): Adult insect / vertebrate does not undergo ecdysis; cannot yield "${exMatch[0]}".`,
          suggestion: '動かぬ骸, 死骸',
        });
      }
    }

    for (const noun of SELECTIONAL_ADULT_INSECT_VERTEBRATE_NOUNS) {
      for (const exuviae of SELECTIONAL_EXUVIAE_TERMS) {
        const combo = `${noun}の${exuviae}`;
        const altCombo = `${noun}なる${exuviae}`;
        if (s.includes(combo) || s.includes(altCombo)) {
          const hit = s.includes(combo) ? combo : altCombo;
          if (!violations.some((v) => v.token === hit)) {
            violations.push({
              type: 'semantic_misuse',
              token: hit,
              sentenceIndex: sIdx,
              message: `Selectional mismatch (Class 3): Adult insect / vertebrate "${noun}" does not undergo ecdysis; cannot yield "${exuviae}".`,
              suggestion: `動かぬ${noun}の骸, ${noun}の死骸`,
            });
          }
        }
      }
    }

    // Class 4: Convalescent context leaping into the sky / cartoonish sky leap
    const hasConvalescent =
      SELECTIONAL_CONVALESCENT_TERMS.some((t) => s.includes(t)) ||
      SELECTIONAL_CONVALESCENT_TERMS.some((t) => text.includes(t));

    const skyLeapPat = /(?:青空|大空|空|天)へ向かって(?:大きく)?跳躍|青空へ跳躍|大空へ跳躍/g;
    let match: RegExpExecArray | null;
    while ((match = skyLeapPat.exec(s)) !== null) {
      violations.push({
        type: 'forced_keyword',
        token: match[0],
        sentenceIndex: sIdx,
        message: 'Selectional mismatch (Class 4): Cartoonish sky leap detected in literary context.',
        suggestion: '身を起こした, 立ち上がった',
      });
    }

    if (hasConvalescent && !violations.some((v) => v.token.includes('跳躍'))) {
      const convLeapPat = /(?:宙|空)へ跳躍/g;
      let clMatch: RegExpExecArray | null;
      while ((clMatch = convLeapPat.exec(s)) !== null) {
        violations.push({
          type: 'forced_keyword',
          token: clMatch[0],
          sentenceIndex: sIdx,
          message: 'Selectional mismatch (Class 4): Convalescent/wounded subject cannot execute superhuman leap.',
        });
      }
    }

    // Somatic clawing defect
    const clawPat = /(?:土|地面)を強く爪立て/g;
    let cMatch: RegExpExecArray | null;
    while ((cMatch = clawPat.exec(s)) !== null) {
      violations.push({
        type: 'semantic_misuse',
        token: cMatch[0],
        sentenceIndex: sIdx,
        message: 'Semantic & register defect: "土を強く爪立てて" uses abstract modifier "強く" on an awkward clawing phrasing.',
        suggestion: '地面の湿った土に爪を食い込ませて',
      });
    }
  }

  return violations;
}

// =============================================================================
// 3. REGISTER VIOLATION DETECTOR (CLASSICAL LITERARY LOANWORDS)
// =============================================================================

export const MODERN_LOANWORD_DICTIONARY: Record<string, string> = {
  猛スピード: '疾風の勢い, 迅雷の勢い, 猛然たる速度',
  スピード: '速度, 疾走, 迅速',
  システム: '体系, 仕組み, 制度',
  レベル: '水準, 境地, 度合い',
  ストレス: '重圧, 心労, 鬱屈',
  タイミング: '好機, 潮時, 機知',
  イメージ: '心象, 面影, 幻影',
  メリット: '利点, 益, 得',
  デメリット: '難点, 損, 弊害',
  プロセス: '過程, 推移, 手順',
  エネルギー: '活力, 気迫, 霊気',
  パーセント: '割, 分, 割合',
  アプローチ: '方策, 接近, 試み',
  コミット: '専念, 傾倒, 誓約',
  モチベーション: '意欲, 熱意, 志',
  プロジェクト: '企て, 事業, 計画',
  ターゲット: '標的, 目標, 獲物',
  シチュエーション: '境遇, 場面, 局面',
  プライド: '自尊心, 誇り, 気概',
  リアル: '生々しい, 真実, 迫真',
  クオリティ: '品質, 出来栄え, 水準',
  テンション: '気迫, 興奮, 昂揚',
  リセット: '振り出し, 刷新, 初期化',
  スペック: '性能, 諸元, 仕様',
  ルーティン: '日課, 常道, 慣習',
  センス: '感覚, 機知, 審美眼',
  モチベ: '意欲, 熱意',
  コスパ: '採算, 割の良さ',
  アピール: '訴求, 主張, 表明',
  リアリティ: '迫真, 真実味',
};

/**
 * Detects anachronistic modern katakana loanwords that violate the classical literary register.
 */
export function detectRegisterViolation(
  text: string,
  registerMode: RegisterMode = 'classical_literary'
): MicroViolation[] {
  const violations: MicroViolation[] = [];
  if (registerMode !== 'classical_literary') {
    return violations;
  }

  const sents = splitSentences(text);
  for (let sIdx = 0; sIdx < sents.length; sIdx++) {
    const s = sents[sIdx]!;

    for (const [loanword, replacement] of Object.entries(MODERN_LOANWORD_DICTIONARY)) {
      if (s.includes(loanword)) {
        violations.push({
          type: 'register_violation',
          token: loanword,
          sentenceIndex: sIdx,
          message: `Modern loanword slang "${loanword}" violates classical literary register.`,
          suggestion: replacement,
        });
      }
    }
  }

  return violations;
}

// =============================================================================
// 4. TWO-TIER GROUNDED FILLER GATE & PHASE HARD CONSTRAINTS
// =============================================================================

/**
 * Tier A: Zero-tolerance 1-hit immediate reject.
 * Explanatory AI meta-cliché phrases and telltale sluggish AI endings (Codepoint-based).
 */
export const HARD_AI_CLICHE_FRAMES: readonly string[] = [
  'テーマ',
  'ように思われ',
  'ように感じられ',
  'させられたのだった',
  '様々な思いが',
  '特別な瞬間となった',
  '改めて深く',
  '強い後悔と悲しみ',
  '心の均衡を失った',
  '言いようのない不安を感じていた',
  'のではないだろうか',
  'と言えるだろう',
  'のだった。',
  '思われた。',
  '感じられた。',
  '傾向が見受けられ',
  '重要な要素',
  '多様な側面',
];

/**
 * Tier B: Contextually grounded general Japanese words.
 * Allowed in rhythmically alive, physically grounded human literature, but penalized
 * in flatline Deadzone text (cadence_alive < 0.40) or when co-occurring with forced action clichés.
 */
export const SOFT_ABSTRACT_WORDS: readonly string[] = [
  '深い',
  '深く',
  '強い',
  '強く',
  '気持',
  '気持ち',
  '感じ',
  '状態',
  '環境',
  '結果',
  '雰囲気',
  '様々',
  '不思議',
  '特別',
  '次第に',
  'ゆっくりと',
  '少しずつ',
  'すっかり',
  'あまりにも',
  '全体',
  '関係',
  '影響',
  '状況',
  '意味',
  '存在',
];

export const HARD_VOID_ORNAMENTAL_SIMILES: readonly string[] = [
  'まるで',
  'ごとき',
  'ごとく',
  'と言おうか',
  'のようなものだ',
  '冷たい砂のように',
  '氷のように',
  '炎のように',
  '宝石のように',
  '松明のように',
  '矢のように',
  '砂のように',
  '波のように',
  '風のように',
];

export const BANNED_VOID_PUNCTUATION = ['！', '!'];

export const VIOLENT_SHOCK_TRIGGERS: readonly string[] = [
  '忽ち',
  '一瞬にして',
  '突如として',
  '激突',
  '叩き割',
  '閃光を放',
];

/**
 * Verifies strict phase-specific invariants with the Two-Tier Grounded Filler Gate:
 * - III_Void: Zero ornamental similes, zero exclamation marks, zero Tier A AI cliché frames.
 *   Tier B general words are protected when rhythmically alive and grounded (protecting human literature).
 * - II_Critical: Zero Tier A AI cliché frames. Violent sudden shock clauses must possess physical sensory grounding.
 *   Tier B adverbs modifying exaggerated forced actions are strictly flagged.
 */
export function verifyPhaseHardConstraints(
  text: string,
  targetPhase: UsirPhase
): MicroViolation[] {
  const violations: MicroViolation[] = [];
  const sents = splitSentences(text);
  const metrics = extractUsir24d(text);

  const isDeadzoneFlatline =
    metrics.cadence_alive < 0.40 ||
    (metrics.len_cv < 0.28 && metrics.cadence_alive < 0.60);

  // 1. Tier A: Zero-tolerance immediate rejection across both Void and Critical
  for (const frame of HARD_AI_CLICHE_FRAMES) {
    if (text.includes(frame)) {
      violations.push({
        type: 'phase_hard_constraint',
        token: frame,
        message: `Zero-tolerance Tier A AI Cliché Frame "${frame}" detected.`,
      });
    }
  }

  // Trailing sluggish endings in deadzone contexts
  if (isDeadzoneFlatline) {
    const sluggishEndingPat = /(?:のだった|思われた|感じられた)(?:[。！？\n]|$)/g;
    let slMatch: RegExpExecArray | null;
    while ((slMatch = sluggishEndingPat.exec(text)) !== null) {
      violations.push({
        type: 'phase_hard_constraint',
        token: slMatch[0].trim(),
        message: `Sluggish AI deadzone ending "${slMatch[0].trim()}" detected.`,
      });
    }
  }

  if (targetPhase === 'III_Void') {
    // 2. Violent punctuation in Void
    for (const p of BANNED_VOID_PUNCTUATION) {
      if (text.includes(p)) {
        violations.push({
          type: 'phase_hard_constraint',
          token: p,
          message: `Phase III (Void) strictly prohibits violent punctuation mark "${p}".`,
        });
      }
    }

    for (let sIdx = 0; sIdx < sents.length; sIdx++) {
      const s = sents[sIdx]!;

      // 3. Ornamental similes in Void (protect natural Japanese verbal auxiliaries like 離れないように)
      for (const fig of HARD_VOID_ORNAMENTAL_SIMILES) {
        if (s.includes(fig)) {
          violations.push({
            type: 'phase_hard_constraint',
            token: fig,
            sentenceIndex: sIdx,
            message: `Phase III (Void) strictly prohibits ornamental simile: "${fig}".`,
          });
        }
      }

      // Ornamental noun similes pattern
      const ornamentalNounSimilePat = /(?:冷たい砂|砂|氷|炎|宝石|松明|矢|波|風)のように/g;
      let oMatch: RegExpExecArray | null;
      while ((oMatch = ornamentalNounSimilePat.exec(s)) !== null) {
        if (!violations.some((v) => v.token === oMatch![0])) {
          violations.push({
            type: 'phase_hard_constraint',
            token: oMatch[0],
            sentenceIndex: sIdx,
            message: `Phase III (Void) strictly prohibits ornamental noun simile: "${oMatch[0]}".`,
          });
        }
      }

      // 4. Tier B general words in Void:
      // If deadzone flatline, reject every Tier B word.
      // If alive and grounded, allow natural human literature occurrences.
      if (isDeadzoneFlatline) {
        for (const word of SOFT_ABSTRACT_WORDS) {
          if (s.includes(word)) {
            violations.push({
              type: 'phase_hard_constraint',
              token: word,
              sentenceIndex: sIdx,
              message: `Tier B abstract filler "${word}" detected in ungrounded AI deadzone flatline context.`,
            });
          }
        }
      }
    }
  } else if (targetPhase === 'II_Critical') {
    for (let sIdx = 0; sIdx < sents.length; sIdx++) {
      const s = sents[sIdx]!;

      // 1. Tier B in Critical:
      if (isDeadzoneFlatline) {
        for (const word of SOFT_ABSTRACT_WORDS) {
          if (s.includes(word)) {
            violations.push({
              type: 'phase_hard_constraint',
              token: word,
              sentenceIndex: sIdx,
              message: `Tier B abstract word "${word}" detected in flatline deadzone context.`,
            });
          }
        }
      } else {
        // In Critical, check if Tier B adverbs (強く, 深く) modify exaggerated forced actions
        const hasForcedAction =
          s.includes('跳躍') ||
          s.includes('爪立て') ||
          s.includes('蹴立て') ||
          (s.includes('激突') && s.includes('火花'));

        if (hasForcedAction) {
          if (s.includes('強く')) {
            violations.push({
              type: 'phase_hard_constraint',
              token: '強く',
              sentenceIndex: sIdx,
              message: 'Tier B adverb "強く" modifies an ungrounded forced action cliché.',
            });
          }
          if (s.includes('深く')) {
            violations.push({
              type: 'phase_hard_constraint',
              token: '深く',
              sentenceIndex: sIdx,
              message: 'Tier B adverb "深く" modifies an ungrounded forced action cliché.',
            });
          }
        }
      }

      // 2. Violent sudden shock clauses must possess physical sensory grounding
      const hasViolentShock = VIOLENT_SHOCK_TRIGGERS.some((t) => s.includes(t));
      if (hasViolentShock) {
        let physCount = 0;
        for (const ch of s) {
          if (PHYSICAL_SENSE_CHARS.has(ch)) {
            physCount++;
          }
        }
        if (physCount === 0) {
          violations.push({
            type: 'phase_hard_constraint',
            token: s.slice(0, 15) + '...',
            sentenceIndex: sIdx,
            message: `Phase II (Critical) shock clause lacks physical sensory grounding (0 PHYSICAL_SENSE_CHARS).`,
          });
        }
      }
    }
  }

  return violations;
}

// =============================================================================
// 5. FORCED META KEYWORDS DETECTOR
// =============================================================================

export const META_GOODHART_TERMS: readonly string[] = [
  'シンプレクティック',
  'ハミルトニアン',
  'マハラノビス',
  '秩序変数',
  '臨界相',
  'ヴォイド相',
  'デッドゾーン',
];

/**
 * Detects meta Goodhart terms leaked into narrative prose text.
 */
export function detectForcedKeywords(text: string): MicroViolation[] {
  const violations: MicroViolation[] = [];
  const sents = splitSentences(text);

  for (let sIdx = 0; sIdx < sents.length; sIdx++) {
    const s = sents[sIdx]!;

    for (const term of META_GOODHART_TERMS) {
      if (s.includes(term)) {
        violations.push({
          type: 'forced_keyword',
          token: term,
          sentenceIndex: sIdx,
          message: `Goodhart meta-term "${term}" leaked into literary prose text.`,
        });
      }
    }
  }

  return violations;
}

// =============================================================================
// 6. RUN ALL MICRO CONSTRAINT AUDITS
// =============================================================================

export interface RunMicroConstraintGateOptions {
  targetPhase: UsirPhase;
  registerMode?: RegisterMode;
}

/**
 * Evaluates text across all 5 micro-constraint audit gates:
 * 1. Tautology & Redundant Modifiers (detectTautology)
 * 2. Universal Modality Selectional Restriction Matrix (detectSelectionalMismatch)
 * 3. Modern Loanword Register Violations (detectRegisterViolation)
 * 4. Two-Tier Grounded Filler Gate & Phase Hard Constraints (verifyPhaseHardConstraints)
 * 5. Meta Goodhart Keywords (detectForcedKeywords)
 */
export function runMicroConstraintGate(
  text: string,
  options: RunMicroConstraintGateOptions
): MicroConstraintGateResult {
  const registerMode = options.registerMode ?? 'classical_literary';
  const targetPhase = options.targetPhase;

  const tautologyViolations = detectTautology(text);
  const selectionalViolations = detectSelectionalMismatch(text);
  const registerViolations = detectRegisterViolation(text, registerMode);
  const phaseViolations = verifyPhaseHardConstraints(text, targetPhase);
  const forcedKeywordViolations = detectForcedKeywords(text);

  const allViolations: MicroViolation[] = [
    ...tautologyViolations,
    ...selectionalViolations,
    ...registerViolations,
    ...phaseViolations,
    ...forcedKeywordViolations,
  ];

  // De-duplicate any overlapping tokens
  const uniqueViolations: MicroViolation[] = [];
  const seenKeys = new Set<string>();
  for (const v of allViolations) {
    const key = `${v.type}:${v.token}:${v.sentenceIndex ?? 0}`;
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      uniqueViolations.push(v);
    }
  }

  const passed = uniqueViolations.length === 0;
  const scorePenalty = Math.min(1.0, uniqueViolations.length * 0.25);

  let summary = 'Passed all micro-constraints.';
  if (!passed) {
    summary = `Failed micro-constraint audit with ${uniqueViolations.length} violation(s): ${uniqueViolations.map((v) => `[${v.type}] ${v.token}`).join(', ')}`;
  }

  return {
    passed,
    scorePenalty,
    violations: uniqueViolations,
    summary,
  };
}
