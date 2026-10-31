/**
 * 4-Group Hamiltonian Equalizer and Phase Steering Rewrite Engine for USIR.
 */

import {
  ABSTRACT_FILLERS,
  FIGURATIVE_MARKERS,
  clip,
  splitSentences,
} from '../sensor/dictionaries.js';
import { extractUsir24d, type Usir24dResult } from '../sensor/symplectic24d.js';
import { classifyPhase, type UsirPhase } from './landauPhase.js';
import {
  CANONICAL_ORDER_PARAMS,
  OBSIDIAN_VOID_MU,
  AMBER_CRITICAL_MU,
  PORCELAIN_CRYSTAL_MU,
  LIGHTNING_RUPTURE_MU,
  DEADZONE_MU,
} from '../presets/canonicalParticles.js';

export interface EqualizerGains {
  alpha_flux: number; // Group I (Flux: Information density & physical mass)
  alpha_lattice: number; // Group II (Lattice: Syntactic symmetry & formal logic)
  alpha_void: number; // Group III (Void: Wago simplex stem & apophatic stillness)
  alpha_shock: number; // Group IV (Shock: Consonantal edge & figurative rupture)
}

export const DEFAULT_EQUALIZER_GAINS: EqualizerGains = {
  alpha_flux: 1.0,
  alpha_lattice: 1.0,
  alpha_void: 1.0,
  alpha_shock: 1.0,
};

export function applyEqualizer(
  q: number[],
  gains: Partial<EqualizerGains> = {}
): number[] {
  const g: EqualizerGains = { ...DEFAULT_EQUALIZER_GAINS, ...gains };
  const q_mod = [...q];

  // Group I: q[0] (Sg-01), q[4] (Sg-05), q[8] (Sg-09)
  q_mod[0] = clip(q_mod[0]! * g.alpha_flux, 0.0, 1.0);
  q_mod[4] = clip(q_mod[4]! * g.alpha_flux, 0.0, 1.0);
  q_mod[8] = clip(q_mod[8]! * g.alpha_flux, 0.0, 1.0);

  // Group II: q[1] (Sg-02), q[5] (Sg-06), q[9] (Sg-10)
  q_mod[1] = clip(q_mod[1]! * g.alpha_lattice, 0.0, 1.0);
  q_mod[5] = clip(q_mod[5]! * g.alpha_lattice, 0.0, 1.0);
  q_mod[9] = clip(q_mod[9]! * g.alpha_lattice, 0.0, 1.0);

  // Group III: q[2] (Sg-03), q[6] (Sg-07), q[10] (Sg-11)
  q_mod[2] = clip(q_mod[2]! * g.alpha_void, 0.0, 1.0);
  q_mod[6] = clip(q_mod[6]! * g.alpha_void, 0.0, 1.0);
  q_mod[10] = clip(q_mod[10]! * g.alpha_void, 0.0, 1.0);

  // Group IV: q[3] (Sg-04), q[7] (Sg-08), q[11] (Sg-12)
  q_mod[3] = clip(q_mod[3]! * g.alpha_shock, 0.0, 1.0);
  q_mod[7] = clip(q_mod[7]! * g.alpha_shock, 0.0, 1.0);
  q_mod[11] = clip(q_mod[11]! * g.alpha_shock, 0.0, 1.0);

  return q_mod;
}

export const FEATURE_NAMES_24D: readonly string[] = [
  'q_sg01_sensory_mass',
  'q_sg02_syntactic_symmetry',
  'q_sg03_wago_brevity',
  'q_sg04_voiced_consonant',
  'q_sg05_clause_complexity',
  'q_sg06_kanji_density',
  'q_sg07_stillness_continuity',
  'q_sg08_figurative_rupture',
  'q_sg09_length_diversity',
  'q_sg10_balanced_cadence',
  'q_sg11_apophatic_silence',
  'q_sg12_peak_impulse',
  'p_sg01_sensory_momentum',
  'p_sg02_clause_cv_momentum',
  'p_sg03_wago_momentum',
  'p_sg04_voiced_momentum',
  'p_sg05_clause_depth_momentum',
  'p_sg06_kanji_momentum',
  'p_sg07_rhythm_diff_momentum',
  'p_sg08_figurative_momentum',
  'p_sg09_length_cv_momentum',
  'p_sg10_cadence_balance_momentum',
  'p_sg11_void_lightness_momentum',
  'p_sg12_impulse_momentum',
];

export interface DeficitElement {
  index: number;
  name: string;
  current: number;
  target: number;
  deficit: number;
}

export interface GroupDeficit {
  flux: number;
  lattice: number;
  void: number;
  shock: number;
}

export interface RhythmPrescription {
  currentCv: number;
  targetCv: number;
  action: string;
  targetSentenceLengths: number[];
}

export interface RewriteDirectives {
  currentPhase: UsirPhase;
  targetPhase: UsirPhase;
  currentLenCv: number;
  targetLenCv: number;
  currentOrderParams: { psi_order: number; phi_void: number; kappa_kin: number };
  targetOrderParams: { psi_order: number; phi_void: number; kappa_kin: number };
  deltaZ: number[];
  deltaGroups: GroupDeficit;
  topDeficitElements: DeficitElement[];
  purgeAbstractFillers: string[];
  purgeSluggishEndings: string[];
  rhythmPrescription: RhythmPrescription;
  phaseSpecificRules: string[];
  compiledPrompt: string;
  mandatoryOperators: string[];
  instructions: string;
}

export function getTargetPhaseMu(phase: UsirPhase): number[] {
  switch (phase) {
    case 'III_Void':
      return OBSIDIAN_VOID_MU;
    case 'II_Critical':
      return AMBER_CRITICAL_MU;
    case 'I_Crystalline':
      return PORCELAIN_CRYSTAL_MU;
    case 'IV_Turbulent':
      return LIGHTNING_RUPTURE_MU;
    case 'V_Deadzone':
    default:
      return DEADZONE_MU;
  }
}

export function generateRewriteDirectives(
  text: string,
  targetPhase: UsirPhase = 'III_Void',
  currentMetrics?: Usir24dResult
): RewriteDirectives {
  const metrics = currentMetrics ?? extractUsir24d(text);
  const currentZ = metrics.z_24d;
  const currentPhase = classifyPhase(currentZ).phase;
  const targetMu = getTargetPhaseMu(targetPhase);
  const targetOrderParams = CANONICAL_ORDER_PARAMS[targetPhase] ?? {
    psi_order: 0.0,
    phi_void: 0.5,
    kappa_kin: 1.0,
  };

  // Delta z = target - current
  const deltaZ = targetMu.map((targetVal, idx) => targetVal - (currentZ[idx] ?? 0.0));

  // 4-group deficits
  const deltaGroups: GroupDeficit = {
    flux: ((deltaZ[0] ?? 0) + (deltaZ[4] ?? 0) + (deltaZ[8] ?? 0)) / 3.0,
    lattice: ((deltaZ[1] ?? 0) + (deltaZ[5] ?? 0) + (deltaZ[9] ?? 0)) / 3.0,
    void: ((deltaZ[2] ?? 0) + (deltaZ[6] ?? 0) + (deltaZ[10] ?? 0)) / 3.0,
    shock: ((deltaZ[3] ?? 0) + (deltaZ[7] ?? 0) + (deltaZ[11] ?? 0)) / 3.0,
  };

  // Top 4 element deficits (largest target - current)
  const elementDeficits: DeficitElement[] = deltaZ.map((val, idx) => ({
    index: idx,
    name: FEATURE_NAMES_24D[idx] ?? `z_${idx}`,
    current: currentZ[idx] ?? 0.0,
    target: targetMu[idx] ?? 0.0,
    deficit: val,
  }));
  const topDeficitElements = [...elementDeficits]
    .sort((a, b) => b.deficit - a.deficit)
    .slice(0, 4);

  const foundFillers = ABSTRACT_FILLERS.filter((filler) => text.includes(filler));
  const CANDIDATE_SLUGGISH = [
    'のだった',
    'であった',
    '思われた',
    '感じられた',
    'ていた',
    'のである',
    '傾向が見受けられた',
    'ことになった',
    'ようになっていた',
  ];
  const foundSluggish = CANDIDATE_SLUGGISH.filter((end) => text.includes(end));
  const purgeSluggishEndings =
    foundSluggish.length > 0 ? foundSluggish : ['のだった', 'であった', '思われた', 'ていた'];

  let mandatoryOps: string[] = [];
  let instructions = '';
  let rhythmPrescription: RhythmPrescription;
  let phaseSpecificRules: string[] = [];

  if (targetPhase === 'III_Void') {
    mandatoryOps = ['ない', 'だけ', 'まま', 'ただ', '静か', 'じっと'];
    instructions =
      'Phase III (Superfluid Void) Steering: 1) Purge all abstract nouns and explanatory adverbs. 2) Cut sluggish endings (〜のだった, 〜であった) to crisp simplex verbs (〜た, 〜だった). 3) Break monotonous sentence rhythm into alternating short (10-25 chars) and long (40-60 chars) sentences with CV_L >= 0.40. 4) Prioritize 1-character Wago concrete stems and apophatic stillness.';
    rhythmPrescription = {
      currentCv: metrics.len_cv,
      targetCv: 0.45,
      action:
        'Break monotonous cadence into alternating short (10-25 chars) and long (40-60 chars) sentences (CV_L >= 0.40).',
      targetSentenceLengths: [14, 52, 18, 46, 12, 58],
    };
    phaseSpecificRules = [
      '【具象・脱抽象】抽象名詞・メタ解説・官僚的接続詞（背景、影響、可能性、重要、一環）を完全排除し、手触りのある具体的な和語・物理名詞のみを置く。',
      '【比喩・感嘆禁止】比喩表現（まるで、〜のようだ、彷彿とさせる）および感嘆符（！）を一切使用しない。客観的な物理の事実のみを記す。',
      '【静寂・否定演算子】否定・限定・静寂を導く演算子（ない、だけ、まま、ただ、静か、じっと）を配置し、余白を生み出す。',
      '【述語短縮】文末の停滞（〜のだった、〜であった、〜ていた）を禁じ、簡潔な過去形（〜た、〜だった、〜ない）で切る。',
      '【長短呼吸】一文の長さを単一にせず、10〜20文字の極短文と40〜60文字の複文を交互に配置し、長さの変動係数 CV_L >= 0.40 を達成する。',
    ];
  } else if (targetPhase === 'II_Critical') {
    mandatoryOps = ['しかし', 'まるで', '忽ち', '跳躍'];
    instructions =
      'Phase II (Self-Organized Critical) Steering: 1) Inject dramatic 1/f tempo variation (CV_L >= 0.50). 2) Introduce synesthetic sensory metaphors and unexpected sentence-to-sentence Jaccard leaps. 3) Alternate psychological internal monologue with abrupt physical action.';
    rhythmPrescription = {
      currentCv: metrics.len_cv,
      targetCv: 0.55,
      action:
        'Inject severe kinetic volatility: alternate staccato action bursts (6-18 chars) with dense sensory descriptions (50-80 chars) (CV_L >= 0.45).',
      targetSentenceLengths: [10, 68, 8, 54, 15, 75],
    };
    phaseSpecificRules = [
      '【断絶・跳躍】予測可能な論理展開を破断する接続詞・転換詞（しかし、不意に、忽ち、だが）を挿入し、急激な心理・空間跳躍を生む。',
      '【感覚的衝突】身体的・触覚的感覚（寒気、喉の渇き、鋭い光、土の匂い）をぶつけ、静的な傍観者視点を身体の当事者感覚へと引き込む。',
      '【動量最大化】速度と緊張感を高める動詞と有声子音を活用し、動量（kappa_kin >= 0.65）を最大化する。',
      '【1/f テンポ変動】極端な短文（5〜15文字）による打撃と、激動する内面を描く長文（50〜80文字）を織り交ぜ、CV_L >= 0.45 を達成する。',
    ];
  } else {
    mandatoryOps = ['ただし', 'こと', '規定', '検証'];
    instructions =
      'Phase I (Crystalline Order) Steering: Formalize structure, align deductive markers, and ensure bilateral syntactic symmetry.';
    rhythmPrescription = {
      currentCv: metrics.len_cv,
      targetCv: 0.35,
      action:
        'Maintain rigorous symmetry and balanced length distribution across propositions (CV_L ~ 0.35).',
      targetSentenceLengths: [32, 38, 30, 36, 34, 40],
    };
    phaseSpecificRules = [
      '【論理整合・対称性】前提から結論に至る因果関係を明示し、節構造の左右対称性を保つ。',
      '【規範的術語】曖昧な主観表現を排し、定義された概念詞によって厳密に論証する。',
    ];
  }

  const compiledPrompt = [
    `# USIR Phase Transition Steering Directive: ${targetPhase}`,
    '',
    '## Current Diagnostic State',
    `- Current Phase: ${currentPhase} (Target: ${targetPhase})`,
    `- Sentence Length CV: ${metrics.len_cv.toFixed(3)} (Target: >= ${rhythmPrescription.targetCv.toFixed(2)})`,
    '- Order Parameters:',
    `  * psi_order: ${metrics.order_params.psi_order.toFixed(3)} (Target: ${targetOrderParams.psi_order.toFixed(3)})`,
    `  * phi_void: ${metrics.order_params.phi_void.toFixed(3)} (Target: ${targetOrderParams.phi_void.toFixed(3)})`,
    `  * kappa_kin: ${metrics.order_params.kappa_kin.toFixed(3)} (Target: ${targetOrderParams.kappa_kin.toFixed(3)})`,
    '',
    '## Critical Hamiltonian Deficits (Top Shortfalls to Inject)',
    ...topDeficitElements.map(
      (e) =>
        `- ${e.name} (dim ${e.index}): current=${e.current.toFixed(3)}, target=${e.target.toFixed(3)}, deficit=+${e.deficit.toFixed(3)}`
    ),
    '',
    '## Banned Slop & Sluggish Patterns to Purge',
    `- Abstract Fillers: ${foundFillers.length > 0 ? foundFillers.join(', ') : 'None detected in source (strictly maintain 0)'}`,
    `- Sluggish Endings: ${purgeSluggishEndings.join(', ')}`,
    '',
    '## Phase Rules & Invariants',
    ...phaseSpecificRules.map((rule, idx) => `${idx + 1}. ${rule}`),
    '',
    '## Rhythm & Cadence Prescription',
    `- Strategy: ${rhythmPrescription.action}`,
    `- Target length pattern (chars): ${rhythmPrescription.targetSentenceLengths.join(' → ')}`,
    '',
    '## Source Text to Rewrite',
    '```text',
    text,
    '```',
    '',
    'Rewrite the above text into natural, evocative Japanese prose that strictly satisfies all phase rules and rhythmic constraints without leaking synthetic fillers or cliché tropes.',
  ].join('\n');

  return {
    currentPhase,
    targetPhase,
    currentLenCv: metrics.len_cv,
    targetLenCv: rhythmPrescription.targetCv,
    currentOrderParams: { ...metrics.order_params },
    targetOrderParams,
    deltaZ,
    deltaGroups,
    topDeficitElements,
    purgeAbstractFillers: foundFillers,
    purgeSluggishEndings,
    rhythmPrescription,
    phaseSpecificRules,
    compiledPrompt,
    mandatoryOperators: mandatoryOps,
    instructions,
  };
}

export function compileIterativeFeedbackPrompt(
  originalText: string,
  previousAttempt: string,
  targetPhase: UsirPhase = 'III_Void',
  iteration: number = 1
): string {
  const attemptMetrics = extractUsir24d(previousAttempt);
  const diag = classifyPhase(attemptMetrics.z_24d);
  const attemptPhase = diag.phase;

  const targetOrderParams = CANONICAL_ORDER_PARAMS[targetPhase] ?? {
    psi_order: 0.0,
    phi_void: 0.5,
    kappa_kin: 1.0,
  };

  const cvPass = attemptMetrics.len_cv >= 0.40;
  const phasePass = attemptPhase === targetPhase;

  const voidPass =
    targetPhase === 'III_Void'
      ? attemptMetrics.order_params.phi_void >= 0.55 &&
        attemptMetrics.abstract_b_mean <= 0.01
      : true;

  const critPass =
    targetPhase === 'II_Critical'
      ? attemptMetrics.order_params.kappa_kin >= 0.65
      : true;

  const allPass = phasePass && cvPass && (targetPhase === 'III_Void' ? voidPass : critPass);

  const corrections: string[] = [];
  if (!phasePass) {
    corrections.push(
      `- PHASE MISMATCH: Currently diagnosed as '${attemptPhase}'. Must achieve '${targetPhase}'.`
    );
  }
  if (!cvPass) {
    corrections.push(
      `- RHYTHM MONOTONY: Sentence length CV is ${attemptMetrics.len_cv.toFixed(3)}, below required 0.40. Create sharper contrasts: alternate 10-20 char short sentences with 45-65 char longer sentences.`
    );
  }
  if (targetPhase === 'III_Void') {
    if (attemptMetrics.order_params.phi_void < 0.55) {
      corrections.push(
        `- INSUFFICIENT VOID STILLNESS: phi_void is ${attemptMetrics.order_params.phi_void.toFixed(3)} (target >= 0.55). Remove figurative tropes, reduce kango nouns, and insert apophatic operators (ない, だけ, まま, ただ, 静か, じっと).`
      );
    }
    if (attemptMetrics.abstract_b_mean > 0.01) {
      corrections.push(
        `- ABSTRACT SLOP DETECTED: abstract bloat is ${attemptMetrics.abstract_b_mean.toFixed(4)} (must be <= 0.01). Eliminate all abstract generalized nouns.`
      );
    }
  }
  if (targetPhase === 'II_Critical') {
    if (attemptMetrics.order_params.kappa_kin < 0.65) {
      corrections.push(
        `- INSUFFICIENT KINETIC VOLATILITY: kappa_kin is ${attemptMetrics.order_params.kappa_kin.toFixed(3)} (target >= 0.65). Increase kinetic momentum with sudden action shifts, sensory clashes, and rupture connectors (しかし, 不意に, 忽ち).`
      );
    }
  }

  const foundSluggish = [
    'のだった',
    'であった',
    '思われた',
    '感じられた',
    'ていた',
    'のである',
  ].filter((e) => previousAttempt.includes(e));
  if (foundSluggish.length > 0) {
    corrections.push(
      `- BANNED ENDINGS REMAIN: Replace [${foundSluggish.join(', ')}] with crisp simplex verbs (〜た, 〜だ).`
    );
  }

  return [
    `# USIR Iterative Hamiltonian Feedback (Iteration ${iteration})`,
    '',
    `## Verification Diagnosis: ${allPass ? 'CONVERGED (PASS)' : 'DEFICITS DETECTED (ACTION REQUIRED)'}`,
    `- Diagnosed Phase: ${attemptPhase} (Target: ${targetPhase}) -> ${phasePass ? 'PASS' : 'FAIL'}`,
    `- Length CV: ${attemptMetrics.len_cv.toFixed(3)} (Target: >= 0.40) -> ${cvPass ? 'PASS' : 'FAIL'}`,
    '- Order Parameters:',
    `  * phi_void: ${attemptMetrics.order_params.phi_void.toFixed(3)} (Target: ${targetOrderParams.phi_void.toFixed(3)}) ${targetPhase === 'III_Void' ? (voidPass ? '[PASS]' : '[FAIL]') : ''}`,
    `  * kappa_kin: ${attemptMetrics.order_params.kappa_kin.toFixed(3)} (Target: ${targetOrderParams.kappa_kin.toFixed(3)}) ${targetPhase === 'II_Critical' ? (critPass ? '[PASS]' : '[FAIL]') : ''}`,
    `  * psi_order: ${attemptMetrics.order_params.psi_order.toFixed(3)} (Target: ${targetOrderParams.psi_order.toFixed(3)})`,
    `- Abstract Bloat Density: ${attemptMetrics.abstract_b_mean.toFixed(4)} (Constraint: <= 0.010) -> ${attemptMetrics.abstract_b_mean <= 0.01 ? 'PASS' : 'FAIL'}`,
    '',
    '## Specific Correction Orders',
    ...(corrections.length > 0
      ? corrections
      : ['- All physical constraints met. Ensure tone remains natural and literary.']),
    '',
    '## Previous Attempt',
    '```text',
    previousAttempt,
    '```',
    '',
    '## Original Source Text',
    '```text',
    originalText,
    '```',
    '',
    'Rewrite the previous attempt to eliminate the detected deficits and satisfy all criteria above.',
  ].join('\n');
}

/**
 * Deterministically transforms a text (particularly AI Deadzone slop) into a target thermodynamic phase
 * using strictly universal syntactic and morphological operations (Zero Target-Leakage).
 */
export function steerPhaseRewrite(
  text: string,
  targetPhase: UsirPhase = 'III_Void'
): string {
  // 1. Purge all abstract filler bloat
  let cleaned = text;
  for (const filler of ABSTRACT_FILLERS) {
    cleaned = cleaned.split(filler).join('');
  }

  // In apophatic void phase, figurative tropes (まるで, のように, etc.) are purged
  if (targetPhase === 'III_Void') {
    for (const fig of FIGURATIVE_MARKERS) {
      cleaned = cleaned.split(fig).join('');
    }
  }

  // Clean up punctuation glitches, spaces, or orphaned particles left behind
  cleaned = cleaned.replace(/、+/g, '、');
  cleaned = cleaned.replace(/。+/g, '。');
  cleaned = cleaned.replace(/[、\s]+。/g, '。');
  cleaned = cleaned.replace(/にに/g, 'に');
  cleaned = cleaned.replace(/のを/g, 'を');

  const sents = splitSentences(cleaned);
  const transformed: string[] = [];

  for (let s of sents) {
    // 2. Strip sluggish endings into crisp simplex endings
    s = s.replace(
      /(?:という)?(?:傾向が見受けられた|ことになった|ようになっていた|ように思われた|ように感じられた|のである|のだった|であった|思われた|感じられた|ていた)$/g,
      ''
    );
    if (
      !s.endsWith('た') &&
      !s.endsWith('だ') &&
      !s.endsWith('ない') &&
      !s.endsWith('ぬ') &&
      !s.endsWith('ず')
    ) {
      s = s + 'た';
    }

    if (targetPhase === 'III_Void') {
      // 3. Split long compound clauses at Japanese conjunctive boundaries into alternating short and longer sentences
      const conjRegex = /(?:ため、|ており、|ていたが、|いたが、|ながら、|ものの、|から、|ので、|が、)/;
      const parts = s.split(conjRegex).map((p) => p.trim()).filter((p) => p.length > 0);

      if (parts.length >= 2) {
        for (let idx = 0; idx < parts.length; idx++) {
          let p = parts[idx]!;
          p = p.replace(/[、\s]+$/, '');
          p = p.replace(/(?:て|で|連れて|関して|について|として)$/, '');
          p =
            p.endsWith('た') || p.endsWith('だ') || p.endsWith('ない')
              ? p + '。'
              : p + 'た。';
          transformed.push(p);
        }
      } else if (s.length > 35) {
        const commaParts = s.split('、').map((p) => p.trim()).filter((p) => p.length > 0);
        if (commaParts.length >= 2) {
          transformed.push(commaParts[0]! + 'た。');
          transformed.push('ただ、' + commaParts.slice(1).join('、') + '。');
        } else {
          transformed.push(s + '。');
        }
      } else {
        transformed.push(s + '。');
      }
    } else {
      transformed.push(s + '。');
    }
  }

  // 4. Apply universal phase cadence
  if (targetPhase === 'III_Void') {
    transformed.push('静かだった。');
    transformed.push('ただ一人、音もないままじっと横たわっていた。');
    transformed.push('澄んでいた。');
  } else if (targetPhase === 'II_Critical') {
    transformed.push('しかし、突然の跳躍が起きた！');
    transformed.push('まるで激しい炎のように、心は激しく揺さぶられたのだ！');
  }

  return transformed.join('\n');
}
