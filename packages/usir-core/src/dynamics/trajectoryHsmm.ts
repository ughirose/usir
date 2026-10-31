/**
 * 3-Mode HSMM Trajectory Engine & Deadzone Sinkhole Detector for USIR.
 * Decodes dynamic narrative modes (Setup, Rupture, Deep Void) via Viterbi,
 * estimates Markov transition matrices and Cholesky factors into Tier M 4096B binary,
 * and surgically pinpoints localized Deadzone Sinkholes.
 */

import {
  extractUsir24d,
} from '../sensor/symplectic24d.js';
import {
  classifyPhase,
  type UsirPhase,
} from './landauPhase.js';
import {
  generateRewriteDirectives,
  type RewriteDirectives,
} from './equalizer.js';
import {
  splitSentences,
} from '../sensor/dictionaries.js';
import {
  encodeTierM,
  type TierMProfile,
} from '../schema/usirBinarySchema.js';
import {
  AMBER_CRITICAL_MU,
  OBSIDIAN_VOID_MU,
  PORCELAIN_CRYSTAL_MU,
} from '../presets/canonicalParticles.js';

// =============================================================================
// TYPES & INTERFACES
// =============================================================================

export interface TrajectoryWindow {
  windowIndex: number;
  sentenceRange: [number, number]; // [startSentence, endSentence] inclusive
  charRange: [number, number];
  paragraphIndex: number;
  windowText: string;
  z_24d: number[];
  q: number[];
  p: number[];
  order_params: {
    psi_order: number;
    phi_void: number;
    kappa_kin: number;
  };
  groups: {
    I_Flux: number;
    II_Lattice: number;
    III_Void: number;
    IV_Shock: number;
  };
  len_cv: number;
  abstract_b_mean: number;
  phase: UsirPhase;
}

export type NarrativeMode = 0 | 1 | 2;
export const NARRATIVE_MODE_NAMES: readonly string[] = [
  'Mode 0 (Setup / 静・序)',
  'Mode 1 (Rupture / 動・破)',
  'Mode 2 (Deep Void / 余韻・間)',
];

export interface HsmmDecodingResult {
  viterbiPath: NarrativeMode[];
  modeNames: string[];
  transitionMatrix: number[][]; // 3x3 stochastic matrix
  modeWeights: number[]; // 3 floats summing to 1.0
  modeCentroids: number[][]; // 3 x 24 floats
  choleskyFactors: number[][]; // 3 x 300 floats
  windowCount: number;
}

export interface DeadzoneSinkhole {
  paragraphIndex: number;
  sentenceRange: [number, number];
  textExcerpt: string;
  localCvL: number;
  localKappa: number;
  localAbstractB: number;
  localPhase: UsirPhase;
  surgicalDirective: RewriteDirectives;
}

export interface DeadzoneSinkholeResult {
  hasSinkholes: boolean;
  sinkholes: DeadzoneSinkhole[];
  sinkholeCount: number;
}

// =============================================================================
// 1. SLIDING WINDOW TRAJECTORY EXTRACTION
// =============================================================================

interface ParsedSentence {
  index: number;
  text: string;
  paragraphIndex: number;
  charStart: number;
  charEnd: number;
}

/**
 * Extracts a sliding-window trajectory wave across a prose text.
 * Tracks character offsets, paragraph boundaries, 24D symplectic states, and Landau phases.
 */
export function extractTrajectoryWave(
  text: string,
  options?: { windowSize?: number; stride?: number }
): TrajectoryWindow[] {
  const windowSize = Math.max(1, options?.windowSize ?? 3);
  const stride = Math.max(1, options?.stride ?? 1);

  // Parse paragraphs and assign paragraph indices to sentences
  const rawParagraphs = text.split(/\n+/);
  const allSentences: ParsedSentence[] = [];
  let globalCharOffset = 0;
  let sentIdx = 0;

  for (let pIdx = 0; pIdx < rawParagraphs.length; pIdx++) {
    const pText = rawParagraphs[pIdx]!;
    if (pText.trim().length === 0) {
      globalCharOffset += pText.length + 1;
      continue;
    }

    const pSents = splitSentences(pText);
    for (const s of pSents) {
      const start = text.indexOf(s, globalCharOffset);
      const charStart = start !== -1 ? start : globalCharOffset;
      const charEnd = charStart + s.length;
      allSentences.push({
        index: sentIdx++,
        text: s,
        paragraphIndex: pIdx,
        charStart,
        charEnd,
      });
      globalCharOffset = charEnd;
    }
  }

  const N = allSentences.length;
  if (N === 0) {
    return [];
  }

  const windows: TrajectoryWindow[] = [];
  const effectiveWindow = Math.min(windowSize, N);

  for (let start = 0; start <= N - effectiveWindow; start += stride) {
    const slice = allSentences.slice(start, start + effectiveWindow);
    const windowText = slice.map((s) => s.text).join('。') + '。';
    const metrics = extractUsir24d(windowText);
    const diag = classifyPhase(metrics.z_24d);

    const first = slice[0]!;
    const last = slice[slice.length - 1]!;

    windows.push({
      windowIndex: windows.length,
      sentenceRange: [first.index, last.index],
      charRange: [first.charStart, last.charEnd],
      paragraphIndex: first.paragraphIndex,
      windowText,
      z_24d: metrics.z_24d,
      q: metrics.q,
      p: metrics.p,
      order_params: metrics.order_params,
      groups: metrics.groups,
      len_cv: metrics.len_cv,
      abstract_b_mean: metrics.abstract_b_mean,
      phase: diag.phase,
    });
  }

  // Handle single-sentence fallback if loop did not execute
  if (windows.length === 0 && N > 0) {
    const slice = allSentences;
    const windowText = slice.map((s) => s.text).join('。') + '。';
    const metrics = extractUsir24d(windowText);
    const diag = classifyPhase(metrics.z_24d);
    windows.push({
      windowIndex: 0,
      sentenceRange: [slice[0]!.index, slice[slice.length - 1]!.index],
      charRange: [slice[0]!.charStart, slice[slice.length - 1]!.charEnd],
      paragraphIndex: slice[0]!.paragraphIndex,
      windowText,
      z_24d: metrics.z_24d,
      q: metrics.q,
      p: metrics.p,
      order_params: metrics.order_params,
      groups: metrics.groups,
      len_cv: metrics.len_cv,
      abstract_b_mean: metrics.abstract_b_mean,
      phase: diag.phase,
    });
  }

  return windows;
}

// =============================================================================
// 2. 3-MODE HSMM VITERBI DECODING & TIER M PACKING
// =============================================================================

function computeEmissionScores(w: TrajectoryWindow): [number, number, number] {
  const shock = w.groups.IV_Shock;
  const voidG = w.groups.III_Void;
  const lattice = w.groups.II_Lattice;
  const flux = w.groups.I_Flux;
  const phi = w.order_params.phi_void;
  const kappa = w.order_params.kappa_kin;

  // Mode 0: Setup (Low shock, moderate flux/lattice, balanced quietude)
  const score0 = (lattice + flux) * 1.5 - shock * 2.5 - Math.abs(phi - 0.25) * 1.0;

  // Mode 1: Rupture (High Shock, High Kinetic Volatility kappa_kin >= 0.65)
  const score1 = shock * 3.0 + kappa * 1.2 - phi * 1.8;

  // Mode 2: Deep Void (High Apophatic Void phi_void >= 0.50, low shock)
  const score2 = phi * 3.2 + voidG * 1.8 - shock * 2.8;

  return [score0, score1, score2];
}

/**
 * Maps sliding windows to the 3 Narrative Modes via Viterbi decoding:
 * - Mode 0 (Setup / 静・序)
 * - Mode 1 (Rupture / 動・破)
 * - Mode 2 (Deep Void / 余韻・間)
 */
export function decodeHsmmModes(wave: TrajectoryWindow[]): HsmmDecodingResult {
  const T = wave.length;
  if (T === 0) {
    const I3 = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    return {
      viterbiPath: [],
      modeNames: [],
      transitionMatrix: I3,
      modeWeights: [1 / 3, 1 / 3, 1 / 3],
      modeCentroids: [PORCELAIN_CRYSTAL_MU, AMBER_CRITICAL_MU, OBSIDIAN_VOID_MU],
      choleskyFactors: [
        new Array(300).fill(0.0),
        new Array(300).fill(0.0),
        new Array(300).fill(0.0),
      ],
      windowCount: 0,
    };
  }

  // Initial prior
  const logPi = [Math.log(0.6), Math.log(0.2), Math.log(0.2)];

  // Transition prior (Setup -> Rupture -> Deep Void preference with self-transitions)
  const transPrior = [
    [0.65, 0.25, 0.1],
    [0.1, 0.55, 0.35],
    [0.15, 0.1, 0.75],
  ];
  const logA = transPrior.map((row) => row.map((p) => Math.log(p)));

  // Compute log emission probabilities via softmax
  const logB: number[][] = new Array(T).fill(0).map(() => [0, 0, 0]);
  for (let t = 0; t < T; t++) {
    const rawScores = computeEmissionScores(wave[t]!);
    const maxScore = Math.max(...rawScores);
    const expScores = rawScores.map((s) => Math.exp(3.0 * (s - maxScore)));
    const sumExp = expScores.reduce((acc, v) => acc + v, 0);
    const probs = expScores.map((v) => Math.max(v / sumExp, 1e-6));
    logB[t] = probs.map((p) => Math.log(p));
  }

  // Viterbi Trellis
  const V: number[][] = new Array(T).fill(0).map(() => [-1e18, -1e18, -1e18]);
  const backpointer: number[][] = new Array(T).fill(0).map(() => [0, 0, 0]);

  for (let m = 0; m < 3; m++) {
    V[0]![m] = logPi[m]! + logB[0]![m]!;
  }

  for (let t = 1; t < T; t++) {
    for (let m = 0; m < 3; m++) {
      let maxVal = -1e18;
      let bestPrev = 0;
      for (let j = 0; j < 3; j++) {
        const val = V[t - 1]![j]! + logA[j]![m]!;
        if (val > maxVal) {
          maxVal = val;
          bestPrev = j;
        }
      }
      V[t]![m] = maxVal + logB[t]![m]!;
      backpointer[t]![m] = bestPrev;
    }
  }

  // Backtrack
  let bestLast = 0;
  let maxFinal = -1e18;
  for (let m = 0; m < 3; m++) {
    if (V[T - 1]![m]! > maxFinal) {
      maxFinal = V[T - 1]![m]!;
      bestLast = m;
    }
  }

  const viterbiPath: NarrativeMode[] = new Array<NarrativeMode>(T);
  viterbiPath[T - 1] = bestLast as NarrativeMode;
  for (let t = T - 2; t >= 0; t--) {
    const nextState = viterbiPath[t + 1]!;
    viterbiPath[t] = backpointer[t + 1]![nextState]! as NarrativeMode;
  }

  // 1. Empirical Transition Matrix A with Laplace smoothing
  const counts: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let t = 0; t < T - 1; t++) {
    const s_from = viterbiPath[t]!;
    const s_to = viterbiPath[t + 1]!;
    counts[s_from]![s_to]! += 1;
  }

  const alpha = 0.1;
  const A: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) {
    const rowSum = counts[r]![0]! + counts[r]![1]! + counts[r]![2]! + 3 * alpha;
    for (let c = 0; c < 3; c++) {
      A[r]![c] = (counts[r]![c]! + alpha) / rowSum;
    }
  }

  // 2. Mode weights pi
  const modeCounts: [number, number, number] = [0, 0, 0];
  for (const s of viterbiPath) {
    modeCounts[s] += 1;
  }
  const totalModeSum = T + 3 * alpha;
  const modeWeights = [
    (modeCounts[0] + alpha) / totalModeSum,
    (modeCounts[1] + alpha) / totalModeSum,
    (modeCounts[2] + alpha) / totalModeSum,
  ];

  // 3. Mode centroids in R^24
  const modeCentroids: number[][] = [];
  const fallbacks = [PORCELAIN_CRYSTAL_MU, AMBER_CRITICAL_MU, OBSIDIAN_VOID_MU];

  for (let k = 0; k < 3; k++) {
    const matchingWindows = wave.filter((_, idx) => viterbiPath[idx] === k);
    if (matchingWindows.length === 0) {
      modeCentroids.push([...fallbacks[k]!]);
    } else {
      const centroid = new Array<number>(24).fill(0.0);
      for (let d = 0; d < 24; d++) {
        let sum = 0.0;
        for (const w of matchingWindows) {
          sum += w.z_24d[d]!;
        }
        centroid[d] = sum / matchingWindows.length;
      }
      modeCentroids.push(centroid);
    }
  }

  // 4. Cholesky factors (3 x 300 lower-triangular floats)
  const choleskyFactors: number[][] = [];
  for (let k = 0; k < 3; k++) {
    const factor: number[] = [];
    const centroid = modeCentroids[k]!;
    // Build diagonal precision factor: L_ii = sqrt(1 / (variance + 0.1))
    const matchingWindows = wave.filter((_, idx) => viterbiPath[idx] === k);

    const diagScales = new Array<number>(24).fill(1.0);
    for (let d = 0; d < 24; d++) {
      if (matchingWindows.length >= 2) {
        const vals = matchingWindows.map((w) => w.z_24d[d]!);
        const v = vals.reduce((acc, x) => acc + Math.pow(x - centroid[d]!, 2), 0) / vals.length;
        diagScales[d] = Math.sqrt(1.0 / (v + 0.05));
      } else {
        diagScales[d] = 1.0;
      }
    }

    for (let i = 0; i < 24; i++) {
      for (let j = 0; j <= i; j++) {
        if (i === j) {
          factor.push(diagScales[i]!);
        } else {
          factor.push(0.0);
        }
      }
    }
    choleskyFactors.push(factor);
  }

  return {
    viterbiPath,
    modeNames: viterbiPath.map((m) => NARRATIVE_MODE_NAMES[m] ?? `Mode ${m}`),
    transitionMatrix: A,
    modeWeights,
    modeCentroids,
    choleskyFactors,
    windowCount: T,
  };
}

/**
 * Decodes the narrative HSMM trajectory of a text and encodes it into a valid 4096-byte Tier M profile.
 */
export function packTrajectoryToTierM(
  text: string,
  options?: { windowSize?: number; stride?: number }
): Uint8Array {
  const wave = extractTrajectoryWave(text, options);
  const hsmm = decodeHsmmModes(wave);

  const profile: TierMProfile = {
    version: 1,
    flags: 0,
    transition_matrix: hsmm.transitionMatrix,
    weights: hsmm.modeWeights,
    centroids: hsmm.modeCentroids,
    cholesky_factors: hsmm.choleskyFactors,
  };

  return encodeTierM(profile);
}

// =============================================================================
// 3. DEADZONE SINKHOLE DETECTOR & SURGICAL DIRECTIVES
// =============================================================================

/**
 * Scans a multi-paragraph text to pinpoint exact "Deadzone Sinkholes" where rhythm flatlines
 * or slop sediment forms, producing targeted surgical rewrite directives.
 */
export function detectDeadzoneSinkholes(text: string): DeadzoneSinkholeResult {
  const paragraphs = text.split(/\n+/).map((p) => p.trim()).filter((p) => p.length > 0);
  const sinkholes: DeadzoneSinkhole[] = [];

  let globalSentenceCount = 0;

  for (let pIdx = 0; pIdx < paragraphs.length; pIdx++) {
    const pText = paragraphs[pIdx]!;
    const pSents = splitSentences(pText);
    const sentStart = globalSentenceCount;
    const sentEnd = globalSentenceCount + Math.max(pSents.length - 1, 0);
    globalSentenceCount += pSents.length;

    const metrics = extractUsir24d(pText);
    const diag = classifyPhase(metrics.z_24d);

    const isSinkhole =
      diag.phase === 'V_Deadzone' ||
      (metrics.len_cv < 0.28 && metrics.order_params.kappa_kin < 0.25) ||
      (metrics.abstract_b_mean > 0.015 && metrics.len_cv < 0.32);

    if (isSinkhole) {
      const directive = generateRewriteDirectives(pText, 'III_Void', metrics);
      sinkholes.push({
        paragraphIndex: pIdx,
        sentenceRange: [sentStart, sentEnd],
        textExcerpt: pText.length > 80 ? pText.slice(0, 80) + '...' : pText,
        localCvL: metrics.len_cv,
        localKappa: metrics.order_params.kappa_kin,
        localAbstractB: metrics.abstract_b_mean,
        localPhase: diag.phase,
        surgicalDirective: directive,
      });
    }
  }

  return {
    hasSinkholes: sinkholes.length > 0,
    sinkholes,
    sinkholeCount: sinkholes.length,
  };
}
