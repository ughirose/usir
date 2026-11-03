/**
 * USIR v4.0 Manifold Geodesic Engine & 3-Phase QDA Asymmetric Double-Well Barrier.
 *
 * Implements:
 * 1. Asymmetric Double-Well 3-Phase QDA Functional:
 *    D_3phase(z) = 0.5 * (z - mu_crit)^T Sigma_crit^-1 (z - mu_crit)
 *                  + gamma_freeze * max(0, sigma_low - ||sigma||_2)^2
 *                  + gamma_chaos * max(0, ||sigma||_2 - sigma_high)^2
 * 2. Semantic Space Trajectory Geodesic Curvature & Brownian Motion Detector:
 *    kappa_{sem, m} = 1 - cos(Delta e_{m+1}, Delta e_m)
 *    Torsion tau_m and Brownian ratio eta_brownian = M * sum ||Delta e_m||^2 / ||sum Delta e_m||^2
 *    M_geodesic(y) in [0, 1]
 * 3. 4-Fold Multiplicative GRPO Barrier Reward:
 *    R_GRPO(y | x_0, x_T) = M_syntax(y) * M_geodesic(y) * M_bridge(y | x_0, x_T) * exp(-0.5 * D_3phase(z(y)))
 */

import { PHYSICAL_SENSE_INTERVALS } from '../sensor/dictionaries.js';
import { extractUsir24d, type Usir24dResult } from '../sensor/symplectic24d.js';
import { BANNED_TARGET_KEYWORDS } from '../cleanroom/threeRoomGate.js';
import { HARD_AI_CLICHE_FRAMES, META_GOODHART_TERMS } from '../cleanroom/microConstraintGate.js';
import type { SyntaxTreeFrame } from '../schema/syntaxFrame.js';

export const MODERN_LOANWORDS: readonly string[] = [
  '猛スピード', 'スピード', 'システム', 'レベル', 'ストレス', 'タイミング',
  'イメージ', 'メリット', 'デメリット', 'プロセス', 'エネルギー', 'パーセント',
  'アプローチ', 'コミット', 'モチベーション', 'プロジェクト', 'ターゲット',
  'シチュエーション', 'プライド', 'リアル', 'クオリティ', 'テンション',
  'リセット', 'スペック', 'ルーティン', 'センス', 'モチベ', 'コスパ',
  'アピール', 'リアリティ'
];

export interface ThreePhaseQdaResult {
  d_3phase: number;
  d_mahalanobis: number;
  penalty_freeze: number;
  penalty_chaos: number;
  sigma_norm: number;
  phase_regime: 'Phase_I_Deadzone' | 'Phase_II_Critical' | 'Phase_III_Chaos';
  is_healthy_critical: boolean;
}

export interface SemanticGeodesicResult {
  curvatures: number[];
  mean_curvature: number;
  max_curvature: number;
  mean_torsion: number;
  brownian_ratio: number;
  m_geodesic: number;
  is_brownian_catalog: boolean;
  units_count: number;
  target_curvature?: number;
}

export interface MultiplicativeGrpoRewardResult {
  reward: number;
  m_syntax: number;
  m_geodesic: number;
  m_bridge: number;
  m_cleanroom: number;
  d_3phase: number;
  d_mahalanobis: number;
  penalty_freeze: number;
  penalty_chaos: number;
  sigma_norm: number;
  phase_regime: 'Phase_I_Deadzone' | 'Phase_II_Critical' | 'Phase_III_Chaos';
  brownian_ratio: number;
  mean_curvature: number;
  is_brownian_catalog: boolean;
  exp_barrier: number;
  is_rejected: boolean;
}

// Canonical empirical Critical Phase centroid (24D from Golden Benchmark Corpus)
export const DEFAULT_MU_CRIT: readonly number[] = [
  0.7322, 0.4407, 0.4809, 0.4138, 0.5065, 0.5728, 0.3453, 0.4347, 0.8135, 0.4921, 0.4950, 0.2874,
  0.6799, 0.2786, 0.8667, 0.5809, 1.1885, 0.2388, 0.8104, 0.8804, 1.0254, 0.6898, 0.2779, 1.1467,
];

export const DEFAULT_SIGMA_CRIT_DIAG: readonly number[] = [
  0.1191, 0.2424, 0.2523, 0.2540, 0.2011, 0.2650, 0.1953, 0.2068, 0.2594, 0.2036, 0.3272, 0.2984,
  0.4864, 0.1204, 0.3870, 0.2511, 0.4628, 0.1505, 0.3519, 0.3580, 0.4813, 0.3384, 0.1495, 0.5067,
];

export const DOMAIN_QDA_PROFILES: Record<string, { mu_crit: readonly number[]; sigma_crit_low: number; sigma_crit_high: number }> = {
  novel: {
    mu_crit: DEFAULT_MU_CRIT,
    sigma_crit_low: 0.35,
    sigma_crit_high: 1.25,
  },
  article: {
    mu_crit: [
      ...DEFAULT_MU_CRIT.slice(0, 9),
      0.70, 0.40, 0.35,
      ...DEFAULT_MU_CRIT.slice(12),
    ],
    sigma_crit_low: 0.30,
    sigma_crit_high: 1.40,
  },
  cot_reasoning: {
    mu_crit: [
      0.60,
      DEFAULT_MU_CRIT[1]!,
      0.08,
      DEFAULT_MU_CRIT[3]!,
      0.45,
      ...DEFAULT_MU_CRIT.slice(5, 8),
      0.50,
      0.85, 0.20, 0.25,
      ...DEFAULT_MU_CRIT.slice(12),
    ],
    sigma_crit_low: 1.35,
    sigma_crit_high: 2.65,
  },
};

// ==============================================================================
// 1. 3-Phase QDA Asymmetric Double-Well Barrier Functional
// ==============================================================================

export function computeThreePhaseQdaDistance(
  z: number[] | Usir24dResult | string,
  options?: {
    domain?: string;
    mu_crit?: number[];
    sigma_crit_inv?: number[];
    sigma_norm?: number;
    gamma_freeze?: number;
    gamma_chaos?: number;
    sigma_crit_low?: number;
    sigma_crit_high?: number;
  }
): ThreePhaseQdaResult {
  const domain = options?.domain ?? 'novel';
  const profile = DOMAIN_QDA_PROFILES[domain] ?? DOMAIN_QDA_PROFILES.novel!;
  const gammaFreeze = options?.gamma_freeze ?? 12.0;
  const gammaChaos = options?.gamma_chaos ?? 12.0;
  const sigmaLow = options?.sigma_crit_low ?? profile.sigma_crit_low;
  const sigmaHigh = options?.sigma_crit_high ?? profile.sigma_crit_high;

  let zVec: number[];
  let calcSigmaNorm: number;

  if (typeof z === 'string') {
    const extracted = extractUsir24d(z);
    zVec = extracted.z_24d;
    let sumP2 = 0;
    for (const p of extracted.p) sumP2 += p * p;
    calcSigmaNorm = Math.sqrt(sumP2);
  } else if (Array.isArray(z)) {
    zVec = [...z];
    if (zVec.length >= 24) {
      let sumP2 = 0;
      for (let i = 12; i < 24; i++) sumP2 += (zVec[i] ?? 0) ** 2;
      calcSigmaNorm = Math.sqrt(sumP2);
    } else {
      calcSigmaNorm = 0.5;
    }
  } else {
    zVec = [...z.z_24d];
    let sumP2 = 0;
    for (const p of z.p) sumP2 += p * p;
    calcSigmaNorm = Math.sqrt(sumP2);
  }

  if (options?.sigma_norm !== undefined) {
    calcSigmaNorm = options.sigma_norm;
  }

  const targetMu = options?.mu_crit ?? profile.mu_crit;
  const invCov = options?.sigma_crit_inv ?? DEFAULT_SIGMA_CRIT_DIAG.map((s) => 1.0 / (s * s));

  let dMahalSum = 0;
  for (let i = 0; i < 24; i++) {
    const diff = (zVec[i] ?? 0) - (targetMu[i] ?? 0);
    const inv = invCov[i] ?? 50.0;
    dMahalSum += diff * diff * inv;
  }
  const dMahalanobis = (0.5 * dMahalSum) / 24.0;

  // Freeze barrier penalty (Left wall)
  const freezeDefect = Math.max(0.0, sigmaLow - calcSigmaNorm);
  const penaltyFreeze = gammaFreeze * (freezeDefect ** 2);

  // Chaos barrier penalty (Right wall)
  const chaosExcess = Math.max(0.0, calcSigmaNorm - sigmaHigh);
  const penaltyChaos = gammaChaos * (chaosExcess ** 2);

  const totalD3Phase = dMahalanobis + penaltyFreeze + penaltyChaos;

  let regime: 'Phase_I_Deadzone' | 'Phase_II_Critical' | 'Phase_III_Chaos' = 'Phase_II_Critical';
  if (freezeDefect > 0.05) {
    regime = 'Phase_I_Deadzone';
  } else if (chaosExcess > 0.05) {
    regime = 'Phase_III_Chaos';
  }

  return {
    d_3phase: totalD3Phase,
    d_mahalanobis: dMahalanobis,
    penalty_freeze: penaltyFreeze,
    penalty_chaos: penaltyChaos,
    sigma_norm: calcSigmaNorm,
    phase_regime: regime,
    is_healthy_critical: freezeDefect === 0.0 && chaosExcess === 0.0,
  };
}

// ==============================================================================
// 2. Semantic Space Trajectory Geodesic Curvature & Brownian Motion Detector
// ==============================================================================

export function splitIntoTrajectoryUnits(text: string): string[] {
  const rawSents = text
    .split(/[。！？!?\n]+/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 2);

  if (rawSents.length >= 3) return rawSents;

  const clauses = text
    .split(/[、,;；]+/)
    .map((c) => c.trim())
    .filter((c) => c.length >= 3);

  if (clauses.length >= 3) return clauses;

  const stride = Math.max(10, Math.floor(text.length / 5));
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += stride) {
    const chunk = text.slice(i, i + stride).trim();
    if (chunk.length >= 3) chunks.push(chunk);
  }
  return chunks;
}

export function embedSentenceUnit(
  sentence: string,
  dim = 24,
  frame?: SyntaxTreeFrame,
  hiddenStates?: any
): number[] {
  if (hiddenStates !== undefined && hiddenStates !== null) {
    const pooled = new Array<number>(dim).fill(0.0);
    if (Array.isArray(hiddenStates) && Array.isArray(hiddenStates[0])) {
      const seqLen = hiddenStates.length;
      const featDim = (hiddenStates[0] as number[]).length;
      for (let i = 0; i < Math.min(featDim, dim); i++) {
        let sum = 0;
        for (let t = 0; t < seqLen; t++) sum += Number((hiddenStates[t] as number[])[i] ?? 0);
        pooled[i] = sum / Math.max(seqLen, 1);
      }
    } else if (ArrayBuffer.isView(hiddenStates) || Array.isArray(hiddenStates)) {
      for (let i = 0; i < Math.min((hiddenStates as number[]).length, dim); i++) {
        pooled[i] = Number((hiddenStates as number[])[i] ?? 0);
      }
    }
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += pooled[i]! * pooled[i]!;
    norm = Math.sqrt(norm);
    if (norm > 1e-9) {
      for (let i = 0; i < dim; i++) pooled[i] = pooled[i]! / norm;
    }
    return pooled;
  }

  if (frame && frame.nodes && frame.nodes.length > 0) {
    const vec = new Array<number>(dim).fill(0.0);
    const nNodes = frame.nodes.length;
    const sSum = new Array<number>(8).fill(0.0);
    for (const u of frame.nodes) {
      const prim = u.hth_primary ? BigInt(u.hth_primary) : 0n;
      if (prim > 0n) {
        for (let d = 0; d < 8; d++) {
          const nibble = Number((prim >> BigInt(4 * d)) & 0xfn);
          sSum[d] = (sSum[d] ?? 0) + nibble / 15.0;
        }
      }
    }
    for (let d = 0; d < 8; d++) {
      vec[d] = Math.min(1.0, (sSum[d] ?? 0) / Math.max(nNodes, 1));
    }
    vec[8] = Math.min(1.0, frame.radial_entropy ?? 0.0);
    vec[9] = Math.min(1.0, (frame.max_depth ?? 0) / Math.max(nNodes, 1));
    if (frame.edges && frame.edges.length > 0) {
      let jumpSum = 0;
      for (const e of frame.edges) jumpSum += e.jump_distance ?? 0;
      vec[10] = Math.min(1.0, jumpSum / frame.edges.length);
    }
    let negCount = 0;
    for (const u of frame.nodes) if (u.is_negation_scope) negCount++;
    vec[11] = negCount / Math.max(nNodes, 1);
    vec[12] = Math.min(1.0, sentence.length / 50.0);

    const kanjiStems = sentence.match(/[\u4e00-\u9fff]+/g) ?? [];
    for (const stem of kanjiStems) {
      let h = 2166136261;
      for (let i = 0; i < stem.length; i++) {
        h = Math.imul(h ^ stem.charCodeAt(i), 16777619) >>> 0;
      }
      for (let k = 13; k < dim; k++) {
        const bit = (h >>> (k - 13)) & 1;
        vec[k] = (vec[k] ?? 0) + (bit === 1 ? 1.0 : -0.5) * (stem.length / 3.0);
      }
    }
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += vec[i]! * vec[i]!;
    norm = Math.sqrt(norm);
    if (norm > 1e-9) {
      for (let i = 0; i < dim; i++) vec[i] = vec[i]! / norm;
    }
    return vec;
  }

  const vec = new Array<number>(dim).fill(0.0);
  if (!sentence) return vec;
  const L = Math.max(sentence.length, 1);

  // 1. Concrete physical sensory density
  let sensoryHits = 0;
  for (let i = 0; i < sentence.length; i++) {
    const cp = sentence.charCodeAt(i);
    for (let j = 0; j < PHYSICAL_SENSE_INTERVALS.length; j++) {
      const interval = PHYSICAL_SENSE_INTERVALS[j]!;
      if (cp >= interval[0] && cp <= interval[1]) {
        sensoryHits++;
        break;
      }
    }
  }
  vec[0] = Math.min(1.0, (sensoryHits / L) * 4.0);

  // 2. Abstract bloat density
  let abstCount = 0;
  for (const f of HARD_AI_CLICHE_FRAMES) {
    if (sentence.includes(f)) abstCount++;
  }
  vec[1] = Math.min(1.0, abstCount / Math.max(L / 20.0, 1.0));

  // 3. Kanji ratio
  let kanjiCount = 0;
  for (let i = 0; i < sentence.length; i++) {
    const cp = sentence.charCodeAt(i);
    if (cp >= 0x4e00 && cp <= 0x9fff) kanjiCount++;
  }
  vec[2] = kanjiCount / L;

  // 4. Length norm
  vec[3] = Math.min(1.0, L / 50.0);

  // 5. Clause density
  const commaCount = (sentence.match(/、/g) || []).length;
  vec[4] = Math.min(1.0, commaCount / 3.0);

  // 6. Rhetorical intensity
  const exclamCount = (sentence.match(/[！!？?]/g) || []).length;
  vec[5] = Math.min(1.0, exclamCount / 1.5);

  // 7-23. Deterministic Concept Hash Projection of Kanji Stems
  const stems = sentence.match(/[\u4e00-\u9fff]+/g) || [];
  for (const stem of stems) {
    let h = 2166136261;
    for (let i = 0; i < stem.length; i++) {
      h = Math.imul(h ^ stem.charCodeAt(i), 16777619);
    }
    for (let k = 6; k < dim; k++) {
      const bit = (h >>> (k - 6)) & 1;
      const sign = bit === 1 ? 1.0 : -0.5;
      const weight = stem.length / 3.0;
      vec[k] = (vec[k] ?? 0) + sign * weight;
    }
  }

  // L2 normalize
  let sumSq = 0;
  for (let i = 0; i < dim; i++) sumSq += (vec[i] ?? 0) ** 2;
  const norm = Math.sqrt(sumSq);
  if (norm > 1e-6) {
    for (let i = 0; i < dim; i++) vec[i] = (vec[i] ?? 0) / norm;
  }
  return vec;
}

export const DOMAIN_CURVATURE_TARGETS: Record<string, number> = {
  novel: 0.30,
  article: 0.48,
  cot_reasoning: 0.52,
};

export function computeSemanticGeodesicCurvature(
  textOrUnits: string | string[],
  options?: {
    embeddings?: number[][];
    alpha_kappa?: number;
    alpha_brownian?: number;
    is_coordinates?: boolean;
    domain?: string;
  }
): SemanticGeodesicResult {
  const units = typeof textOrUnits === 'string' ? splitIntoTrajectoryUnits(textOrUnits) : [...textOrUnits];
  const alphaKappa = options?.alpha_kappa ?? 1.8;
  const alphaBrownian = options?.alpha_brownian ?? 1.2;
  const domain = options?.domain ?? 'novel';
  const targetKappa = DOMAIN_CURVATURE_TARGETS[domain] ?? 0.30;

  let diffs: number[][];
  if (options?.embeddings) {
    const rawE = options.embeddings;
    if (options.is_coordinates && rawE.length >= 2) {
      diffs = [];
      for (let i = 0; i < rawE.length - 1; i++) {
        const d: number[] = [];
        for (let j = 0; j < (rawE[i]?.length ?? 0); j++) {
          d.push((rawE[i + 1]?.[j] ?? 0) - (rawE[i]?.[j] ?? 0));
        }
        diffs.push(d);
      }
    } else {
      diffs = rawE;
    }
  } else {
    if (units.length < 2) {
      return {
        curvatures: [],
        mean_curvature: 0.0,
        max_curvature: 0.0,
        mean_torsion: 0.0,
        brownian_ratio: 1.0,
        m_geodesic: 1.0,
        is_brownian_catalog: false,
        units_count: units.length,
        target_curvature: targetKappa,
      };
    }
    diffs = units.map((u) => embedSentenceUnit(u));
  }

  const diffNorms: number[] = [];
  for (const d of diffs) {
    let s = 0;
    for (const v of d) s += v * v;
    diffNorms.push(Math.sqrt(s));
  }

  const curvatures: number[] = [];
  const torsions: number[] = [];

  for (let i = 0; i < diffs.length - 1; i++) {
    const v1 = diffs[i]!;
    const v2 = diffs[i + 1]!;
    const n1 = diffNorms[i] ?? 0;
    const n2 = diffNorms[i + 1] ?? 0;
    if (n1 > 1e-7 && n2 > 1e-7) {
      let dot = 0;
      for (let j = 0; j < v1.length; j++) dot += (v1[j] ?? 0) * (v2[j] ?? 0);
      let cos = dot / (n1 * n2);
      cos = Math.max(-1.0, Math.min(1.0, cos));
      curvatures.push(1.0 - cos);
    } else {
      curvatures.push(0.0);
    }
  }

  // Discrete Torsion
  for (let i = 0; i < diffs.length - 2; i++) {
    const v1 = diffs[i]!;
    const v2 = diffs[i + 1]!;
    const v3 = diffs[i + 2]!;

    let dotV2 = 0;
    for (const x of v2) dotV2 += x * x;
    const denomV2 = dotV2 + 1e-7;

    let dotV1V2 = 0;
    let dotV3V2 = 0;
    for (let j = 0; j < v1.length; j++) {
      dotV1V2 += (v1[j] ?? 0) * (v2[j] ?? 0);
      dotV3V2 += (v3[j] ?? 0) * (v2[j] ?? 0);
    }

    const n1 = v1.map((x, j) => x - (dotV1V2 / denomV2) * (v2[j] ?? 0));
    const n2 = v3.map((x, j) => x - (dotV3V2 / denomV2) * (v2[j] ?? 0));

    let lenN1 = 0;
    let lenN2 = 0;
    let dotN = 0;
    for (let j = 0; j < n1.length; j++) {
      lenN1 += (n1[j] ?? 0) ** 2;
      lenN2 += (n2[j] ?? 0) ** 2;
      dotN += (n1[j] ?? 0) * (n2[j] ?? 0);
    }
    lenN1 = Math.sqrt(lenN1);
    lenN2 = Math.sqrt(lenN2);

    if (lenN1 > 1e-7 && lenN2 > 1e-7) {
      let cosTau = dotN / (lenN1 * lenN2);
      cosTau = Math.max(-1.0, Math.min(1.0, cosTau));
      torsions.push(1.0 - cosTau);
    } else {
      torsions.push(0.0);
    }
  }

  const meanCurv = curvatures.length > 0 ? curvatures.reduce((a, b) => a + b, 0) / curvatures.length : 0.0;
  const maxCurv = curvatures.length > 0 ? Math.max(...curvatures) : 0.0;
  const meanTor = torsions.length > 0 ? torsions.reduce((a, b) => a + b, 0) / torsions.length : 0.0;

  // Brownian ratio
  let sumStepsSq = 0;
  for (const n of diffNorms) sumStepsSq += n * n;

  const dim = diffs[0]?.length ?? 24;
  const netStepVec = new Array<number>(dim).fill(0.0);
  for (const d of diffs) {
    for (let j = 0; j < dim; j++) {
      netStepVec[j] = (netStepVec[j] ?? 0) + (d[j] ?? 0);
    }
  }

  let netDispSq = 0;
  for (const x of netStepVec) netDispSq += x * x;
  netDispSq += 1e-5;

  const brownianRatio = (diffs.length * sumStepsSq) / netDispSq;

  let curvDev: number;
  let catalogCurvThresh: number;
  if (domain === 'novel') {
    curvDev = Math.max(0.0, meanCurv - targetKappa);
    catalogCurvThresh = 0.50;
  } else {
    curvDev = Math.abs(meanCurv - targetKappa);
    catalogCurvThresh = targetKappa + 0.22;
  }

  const brownianExcess = Math.max(0.0, brownianRatio - 1.5);
  const exponent = -(alphaKappa * curvDev + alphaBrownian * brownianExcess);
  const mGeodesic = Math.max(0.0, Math.min(1.0, Math.exp(exponent)));

  const isBrownianCatalog = brownianRatio > 2.0 || meanCurv > catalogCurvThresh;

  return {
    curvatures,
    mean_curvature: meanCurv,
    max_curvature: maxCurv,
    mean_torsion: meanTor,
    brownian_ratio: brownianRatio,
    m_geodesic: mGeodesic,
    is_brownian_catalog: isBrownianCatalog,
    units_count: units.length,
    target_curvature: targetKappa,
  };
}

// ==============================================================================
// 3. 4-Fold Multiplicative GRPO Barrier Reward Engine
// ==============================================================================

export function computePunctuationRatioGate(text: string): number {
  if (!text) return 0.0;
  const L = text.length;
  const matches = text.match(/[、。！？!?,.]/g);
  const punctCount = matches ? matches.length : 0;
  const ratio = punctCount / L;

  if (ratio < 0.010) return 0.0;
  if (ratio > 0.120) return 0.0;

  if (ratio < 0.015) {
    return (ratio - 0.010) / 0.005;
  }
  if (ratio > 0.080) {
    return Math.max(0.0, (0.120 - ratio) / 0.040);
  }
  return 1.0;
}

export function computeMultiplicativeGrpoReward(
  text: string,
  options?: {
    x_0?: string | number[];
    x_T?: string | number[];
    domain?: string;
    is_authentic_masterwork?: boolean;
    alpha_metric?: number;
  }
): MultiplicativeGrpoRewardResult {
  if (!text || text.trim().length === 0) {
    return {
      reward: 0.0,
      m_syntax: 0.0,
      m_geodesic: 0.0,
      m_bridge: 0.0,
      m_cleanroom: 0.0,
      d_3phase: 99.0,
      d_mahalanobis: 99.0,
      penalty_freeze: 0.0,
      penalty_chaos: 0.0,
      sigma_norm: 0.0,
      phase_regime: 'Phase_I_Deadzone',
      brownian_ratio: 99.0,
      mean_curvature: 2.0,
      is_brownian_catalog: true,
      exp_barrier: 0.0,
      is_rejected: true,
    };
  }

  // 1. Syntax Multiplier with Strict Punctuation Gate & Bracket Matching
  const punctGate = computePunctuationRatioGate(text);
  let mSyntax = 0.0;
  if (punctGate > 0.0) {
    const bracketPairs: Record<string, string> = {
      '「': '」', '『': '』', '（': '）', '【': '】', '《': '》', '(': ')', '[': ']', '{': '}'
    };
    const openBrackets = new Set(Object.keys(bracketPairs));
    const closeBrackets = new Map(Object.entries(bracketPairs).map(([k, v]) => [v, k]));
    const stack: string[] = [];
    let matched = 0;
    let unmatched = 0;

    for (let i = 0; i < text.length; i++) {
      const ch = text[i]!;
      if (openBrackets.has(ch)) {
        stack.push(ch);
      } else if (closeBrackets.has(ch)) {
        const exp = closeBrackets.get(ch);
        if (stack.length > 0 && stack[stack.length - 1] === exp) {
          stack.pop();
          matched++;
        } else {
          unmatched++;
        }
      }
    }
    unmatched += stack.length;
    const totB = 2 * matched + unmatched;
    const bScore = totB === 0 ? 1.0 : (2.0 * matched) / totB;

    let repPen = 0.0;
    if (text.length >= 16) {
      for (let i = 0; i < text.length - 10; i++) {
        const quad = text.slice(i, i + 5);
        const count = (text.match(new RegExp(quad.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
        if (count >= 4) {
          repPen = 0.50;
          break;
        }
      }
    }

    mSyntax = Math.max(0.0, Math.min(1.0, punctGate * bScore * (1.0 - repPen)));
  }

  // 2. Semantic Geodesic Multiplier
  const geodesicRes = computeSemanticGeodesicCurvature(text);
  const mGeodesic = geodesicRes.m_geodesic;

  // 3. Bidirectional Bridge Multiplier
  let mBridge = 1.0;
  if (options?.x_0 !== undefined || options?.x_T !== undefined) {
    const units = splitIntoTrajectoryUnits(text);
    if (units.length > 0) {
      const tipFwd = embedSentenceUnit(units[0]!);
      const tipBwd = embedSentenceUnit(units[units.length - 1]!);

      let cos0 = 1.0;
      if (options?.x_0 !== undefined) {
        const vec0 = typeof options.x_0 === 'string' ? embedSentenceUnit(options.x_0) : options.x_0;
        let dot = 0, n0 = 0, nf = 0;
        for (let j = 0; j < tipFwd.length; j++) {
          dot += (tipFwd[j] ?? 0) * (vec0[j] ?? 0);
          n0 += (vec0[j] ?? 0) ** 2;
          nf += (tipFwd[j] ?? 0) ** 2;
        }
        cos0 = Math.max(0.0, Math.min(1.0, dot / (Math.sqrt(nf) * Math.sqrt(n0) + 1e-7)));
      }

      let cosT = 1.0;
      if (options?.x_T !== undefined) {
        const vecT = typeof options.x_T === 'string' ? embedSentenceUnit(options.x_T) : options.x_T;
        let dot = 0, nT = 0, nb = 0;
        for (let j = 0; j < tipBwd.length; j++) {
          dot += (tipBwd[j] ?? 0) * (vecT[j] ?? 0);
          nT += (vecT[j] ?? 0) ** 2;
          nb += (tipBwd[j] ?? 0) ** 2;
        }
        cosT = Math.max(0.0, Math.min(1.0, dot / (Math.sqrt(nb) * Math.sqrt(nT) + 1e-7)));
      }

      mBridge = Math.max(0.0, Math.min(1.0, 0.5 * (cos0 + cosT)));
    }
  }

  // 4. Cleanroom Multiplier
  let mCleanroom = 1.0;
  if (!options?.is_authentic_masterwork) {
    let bannedHits = 0;
    for (const w of BANNED_TARGET_KEYWORDS) if (text.includes(w)) bannedHits++;
    let metaHits = 0;
    for (const m of META_GOODHART_TERMS) if (text.includes(m)) metaHits++;
    let loanHits = 0;
    for (const lw of MODERN_LOANWORDS) if (text.includes(lw)) loanHits++;
    let clicheHits = 0;
    for (const c of HARD_AI_CLICHE_FRAMES) if (text.includes(c)) clicheHits++;

    const totalV = bannedHits * 3 + metaHits * 3 + loanHits + clicheHits;
    if (totalV > 0) {
      mCleanroom = Math.max(0.05, 0.20 ** Math.min(totalV, 3));
    }
  }

  // 5. 3-Phase QDA Barrier Distance
  const qdaRes = computeThreePhaseQdaDistance(text, { domain: options?.domain });
  const alphaM = options?.alpha_metric ?? 1.0;
  const expBarrier = Math.exp(-0.5 * (qdaRes.d_3phase / Math.max(alphaM, 0.1)));

  const rawReward = mSyntax * mGeodesic * mBridge * mCleanroom * expBarrier;
  const reward = Math.max(0.0, Math.min(1.0, rawReward));
  const isRejected = reward < 0.005 || mSyntax < 0.1 || mGeodesic < 0.1;

  return {
    reward,
    m_syntax: mSyntax,
    m_geodesic: mGeodesic,
    m_bridge: mBridge,
    m_cleanroom: mCleanroom,
    d_3phase: qdaRes.d_3phase,
    d_mahalanobis: qdaRes.d_mahalanobis,
    penalty_freeze: qdaRes.penalty_freeze,
    penalty_chaos: qdaRes.penalty_chaos,
    sigma_norm: qdaRes.sigma_norm,
    phase_regime: qdaRes.phase_regime,
    brownian_ratio: geodesicRes.brownian_ratio,
    mean_curvature: geodesicRes.mean_curvature,
    is_brownian_catalog: geodesicRes.is_brownian_catalog,
    exp_barrier: expBarrier,
    is_rejected: isRejected,
  };
}

function invertMatrix24(A: number[][]): number[][] {
  const n = A.length;
  const aug: number[][] = Array.from({ length: n }, (_, i) => {
    const row = new Array<number>(2 * n).fill(0.0);
    for (let j = 0; j < n; j++) row[j] = A[i]![j]!;
    row[n + i] = 1.0;
    return row;
  });

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    let maxVal = Math.abs(aug[i]![i]!);
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k]![i]!) > maxVal) {
        maxVal = Math.abs(aug[k]![i]!);
        maxRow = k;
      }
    }
    if (maxRow !== i) {
      const tmp = aug[i]!;
      aug[i] = aug[maxRow]!;
      aug[maxRow] = tmp;
    }
    const pivot = aug[i]![i]!;
    if (Math.abs(pivot) < 1e-12) continue;
    for (let j = 0; j < 2 * n; j++) aug[i]![j] = aug[i]![j]! / pivot;
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = aug[k]![i]!;
        for (let j = 0; j < 2 * n; j++) {
          aug[k]![j] = aug[k]![j]! - factor * aug[i]![j]!;
        }
      }
    }
  }

  return Array.from({ length: n }, (_, i) => aug[i]!.slice(n));
}

export interface ContrastiveFisherResult {
  G_style: number[][];
  optimal_projection_axis: number[];
  mu_pos: number[];
  mu_neg: number[];
  delta_mu: number[];
  lambda_max: number;
  S_B: number[][];
  S_W: number[][];
}

export function fitContrastiveFisherMetric(
  posZ: number[][],
  negZ: number[][],
  lambdaReg = 1e-3
): ContrastiveFisherResult {
  const dim = posZ.length > 0 && posZ[0] ? posZ[0].length : 24;
  const nPos = posZ.length;
  const nNeg = negZ.length;

  const muPos = new Array<number>(dim).fill(0.0);
  for (let i = 0; i < nPos; i++) {
    for (let d = 0; d < dim; d++) muPos[d] = (muPos[d] ?? 0) + (posZ[i]![d] ?? 0);
  }
  for (let d = 0; d < dim; d++) muPos[d] = (muPos[d] ?? 0) / Math.max(nPos, 1);

  const muNeg = new Array<number>(dim).fill(0.0);
  for (let i = 0; i < nNeg; i++) {
    for (let d = 0; d < dim; d++) muNeg[d] = (muNeg[d] ?? 0) + (negZ[i]![d] ?? 0);
  }
  for (let d = 0; d < dim; d++) muNeg[d] = (muNeg[d] ?? 0) / Math.max(nNeg, 1);

  const deltaMu = new Array<number>(dim);
  for (let d = 0; d < dim; d++) deltaMu[d] = muPos[d]! - muNeg[d]!;

  const S_B: number[][] = Array.from({ length: dim }, () => new Array<number>(dim).fill(0.0));
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      S_B[i]![j] = deltaMu[i]! * deltaMu[j]!;
    }
  }

  const covPos: number[][] = Array.from({ length: dim }, () => new Array<number>(dim).fill(0.0));
  if (nPos > 1) {
    for (let k = 0; k < nPos; k++) {
      for (let i = 0; i < dim; i++) {
        const di = posZ[k]![i]! - muPos[i]!;
        for (let j = 0; j < dim; j++) {
          covPos[i]![j] = (covPos[i]![j] ?? 0) + (di * (posZ[k]![j]! - muPos[j]!)) / (nPos - 1);
        }
      }
    }
  }

  const covNeg: number[][] = Array.from({ length: dim }, () => new Array<number>(dim).fill(0.0));
  if (nNeg > 1) {
    for (let k = 0; k < nNeg; k++) {
      for (let i = 0; i < dim; i++) {
        const di = negZ[k]![i]! - muNeg[i]!;
        for (let j = 0; j < dim; j++) {
          covNeg[i]![j] = (covNeg[i]![j] ?? 0) + (di * (negZ[k]![j]! - muNeg[j]!)) / (nNeg - 1);
        }
      }
    }
  }

  const S_W: number[][] = Array.from({ length: dim }, () => new Array<number>(dim).fill(0.0));
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      S_W[i]![j] = 0.5 * (covPos[i]![j]! + covNeg[i]![j]!) + (i === j ? lambdaReg : 0.0);
    }
  }

  const G_style = invertMatrix24(S_W);

  const wStar = new Array<number>(dim).fill(0.0);
  for (let i = 0; i < dim; i++) {
    let s = 0.0;
    for (let j = 0; j < dim; j++) s += G_style[i]![j]! * deltaMu[j]!;
    wStar[i] = s;
  }
  let normW = 0.0;
  for (let i = 0; i < dim; i++) normW += wStar[i]! * wStar[i]!;
  normW = Math.sqrt(normW);

  const optimalAxis = new Array<number>(dim);
  for (let i = 0; i < dim; i++) optimalAxis[i] = normW > 1e-12 ? wStar[i]! / normW : 0.0;

  let lambdaMax = 0.0;
  for (let i = 0; i < dim; i++) lambdaMax += deltaMu[i]! * wStar[i]!;

  return {
    G_style,
    optimal_projection_axis: optimalAxis,
    mu_pos: muPos,
    mu_neg: muNeg,
    delta_mu: deltaMu,
    lambda_max: lambdaMax,
    S_B,
    S_W,
  };
}

export const fit_contrastive_fisher_metric = fitContrastiveFisherMetric;
