/**
 * Universal Style Genome (USIR v2.0) Pure Symplectic Instrumentation
 * Module 2: Wavelet-Hermite, Fourier-Poisson MCA & RQA 12-Element Encoder
 *
 * Implements:
 * 1. Base structural signals s_m in R^4 per clause (Flux, Lattice, Void, Shock).
 * 2. OPEN-08 Ghost History Warm-Start & James-Stein shrinkage padding.
 * 3. Strictly orthogonal Haar wavelet 3-scale decomposition (Micro, Meso, Macro) (Theorem 2.1).
 * 4. Morphological Component Analysis (MCA): Fourier D_F + Poisson-Soliton D_S via ISTA.
 * 5. Wiener-Hermite polynomials (He_2, He_3, He_4) and RQA (DET & I_arch).
 * 6. 24D Symplectic state z = (q, p) in [0, 1]^12 x R_{>=0}^12.
 */

import {
  HTHLexicon,
  type MusicNoteEvent,
  extractMusicFunctorSignals,
  discreteJumpDistance,
} from './hthLexicon.js';

// ============================================================================
// Interfaces
// ============================================================================

export interface WaveletDecompositionResult {
  micro: Float64Array;
  meso: Float64Array;
  macro: Float64Array;
  coeffs: Float64Array;
}

export interface MCAResult {
  fourierComp: Float64Array;
  solitonComp: Float64Array;
  cF: Float64Array;
  cS: Float64Array;
  residual: Float64Array;
  // Aliases for interoperability
  fourier_comp?: Float64Array;
  soliton_comp?: Float64Array;
  c_F?: Float64Array;
  c_S?: Float64Array;
}

export interface RQAResult {
  det: number;
  iArch: number;
  i_arch?: number;
  lam?: number;
  recurrenceMatrix?: Float64Array[];
}

export interface SymplecticStateResult {
  q: Float64Array; // 12D position coordinates in [0, 1]^12
  p: Float64Array; // 12D conjugate momentum coordinates in R_{>=0}^12
  z: Float64Array; // 24D symplectic state [q, p]
}

export interface LogarithmicScalesResult {
  m: number;
  logM: number;
  log_m: number;
  scaleMicro: number;
  scale_micro: number;
  scaleMeso: number;
  scale_meso: number;
  scaleMacro: number;
  scale_macro: number;
  tertiles: [number, number, number];
  microLevel: number;
  micro_level: number;
  mesoLevel: number;
  meso_level: number;
  macroLevel: number;
  macro_level: number;
}

// ============================================================================
// Constants & Closed Grammatical Sets
// ============================================================================

const VOICED_OBSTRUENTS = new Set([
  'が','ぎ','ぐ','げ','ご','ざ','じ','ず','ぜ','ぞ',
  'だ','ぢ','づ','で','ど','ば','び','ぶ','べ','ぼ',
  'ガ','ギ','グ','ゲ','ゴ','ザ','ジ','ズ','ゼ','ゾ',
  'ダ','ヂ','ヅ','デ','ド','バ','ビ','ブ','ベ','ボ','ヴ'
]);

const GEMINATES = new Set(['っ', 'ッ']);

const CLOSED_ADVERSATIVE = new Set([
  'しかし', 'だが', 'けれども', 'けれど', 'でも', 'ところが', 'ものの', 'とはいえ'
]);

const CLOSED_LOGICAL = new Set([
  'から', 'ので', 'ため', 'ゆえに', 'したがって', 'よって', 'ならば', 'たら', 'れば'
]);

// ============================================================================
// Helper Math Functions
// ============================================================================

function vecMean(v: ArrayLike<number>): number {
  if (v.length === 0) return 0.0;
  let sum = 0.0;
  for (let i = 0; i < v.length; i++) sum += v[i]!;
  return sum / v.length;
}

function vecStd(v: ArrayLike<number>): number {
  if (v.length <= 1) return 0.0;
  const m = vecMean(v);
  let vSum = 0.0;
  for (let i = 0; i < v.length; i++) {
    const diff = v[i]! - m;
    vSum += diff * diff;
  }
  return Math.sqrt(vSum / v.length);
}

function vecNorm(v: ArrayLike<number>): number {
  let s = 0.0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  return Math.sqrt(s);
}

function diffRms(v: ArrayLike<number>): number {
  if (v.length <= 1) return 0.0;
  let s = 0.0;
  for (let i = 0; i < v.length - 1; i++) {
    const d = v[i + 1]! - v[i]!;
    s += d * d;
  }
  return Math.sqrt(s / (v.length - 1));
}

export function padToPowerOf2(signal: Float64Array, minLen: number = 8): Float64Array {
  const m = signal.length;
  let target = minLen;
  while (target < m) target <<= 1;
  if (m === target) return signal;

  const padded = new Float64Array(target);
  padded.set(signal);
  let filled = m;
  while (filled < target) {
    const take = Math.min(target - filled, m);
    for (let i = 0; i < take; i++) {
      padded[filled + i] = signal[m - 1 - i]!;
    }
    filled += take;
  }
  return padded;
}

// ============================================================================
// Gap 2: Logarithmic Self-Similar Renormalization & Multiscale Filters
// ============================================================================

export function computeLogarithmicScales(m: number): LogarithmicScalesResult {
  const mClamped = Math.max(Math.floor(m), 2);
  const logM = Math.log2(mClamped);
  const scaleMicro = logM / 3.0;
  const scaleMeso = (2.0 * logM) / 3.0;
  const scaleMacro = logM;

  const microLevel = Math.max(1, Math.round(scaleMicro));
  const mesoLevel = Math.max(microLevel + 1, Math.round(scaleMeso));
  const macroLevel = Math.max(mesoLevel + 1, Math.round(scaleMacro));

  return {
    m: Math.floor(m),
    logM,
    log_m: logM,
    scaleMicro,
    scale_micro: scaleMicro,
    scaleMeso,
    scale_meso: scaleMeso,
    scaleMacro,
    scale_macro: scaleMacro,
    tertiles: [scaleMicro, scaleMeso, scaleMacro],
    microLevel,
    micro_level: microLevel,
    mesoLevel,
    meso_level: mesoLevel,
    macroLevel,
    macro_level: macroLevel,
  };
}

export function renormalizeDecimateSignals(
  signals: Float64Array[] | number[][],
  maxLength: number = 1024
): Float64Array[] {
  let cur: Float64Array[] = signals.map((row) =>
    row instanceof Float64Array ? new Float64Array(row) : new Float64Array(row)
  );
  if (cur.length === 0) return cur;
  const numChannels = cur[0]!.length;

  while (cur.length > maxLength) {
    const m = cur.length;
    const nPairs = Math.floor(m / 2);
    const hasRem = m % 2 !== 0;
    const nextSig: Float64Array[] = [];

    for (let k = 0; k < nPairs; k++) {
      const rowA = cur[2 * k]!;
      const rowB = cur[2 * k + 1]!;
      const rowOut = new Float64Array(numChannels);

      if (numChannels >= 4) {
        for (let c = 0; c < 3; c++) {
          rowOut[c] = 0.5 * (rowA[c]! + rowB[c]!);
        }
        rowOut[3] = Math.max(rowA[3]!, rowB[3]!);
        for (let c = 4; c < numChannels; c++) {
          rowOut[c] = 0.5 * (rowA[c]! + rowB[c]!);
        }
      } else {
        for (let c = 0; c < numChannels; c++) {
          rowOut[c] = 0.5 * (rowA[c]! + rowB[c]!);
        }
      }
      nextSig.push(rowOut);
    }

    if (hasRem) {
      nextSig.push(new Float64Array(cur[m - 1]!));
    }

    cur = nextSig;
  }

  return cur;
}

// ============================================================================
// Strictly Orthogonal Haar Wavelet Filter Bank (Theorem 2.1)
// ============================================================================

export function buildHaarOrthonormalMatrix(n: number): number[][] {
  if (n < 8 || (n & (n - 1)) !== 0) {
    throw new Error(`Wavelet length N must be a power of 2 >= 8, got ${n}`);
  }
  const w: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0.0));

  // 1. Level 1 details (Micro, j=1): n/2 wavelets of support 2
  const invSqrt2 = 1.0 / Math.SQRT2;
  for (let k = 0; k < n / 2; k++) {
    w[k]![2 * k] = invSqrt2;
    w[k]![2 * k + 1] = -invSqrt2;
  }

  // 2. Level 2 details (Meso, j=2): n/4 wavelets of support 4
  const offset2 = n / 2;
  for (let k = 0; k < n / 4; k++) {
    w[offset2 + k]![4 * k] = 0.5;
    w[offset2 + k]![4 * k + 1] = 0.5;
    w[offset2 + k]![4 * k + 2] = -0.5;
    w[offset2 + k]![4 * k + 3] = -0.5;
  }

  // 3. Level 3 details (Macro detail): n/8 wavelets of support 8
  const offset3 = (3 * n) / 4;
  const invSqrt8 = 1.0 / Math.sqrt(8.0);
  for (let k = 0; k < n / 8; k++) {
    for (let j = 0; j < 4; j++) {
      w[offset3 + k]![8 * k + j] = invSqrt8;
      w[offset3 + k]![8 * k + 4 + j] = -invSqrt8;
    }
  }

  // 4. Level 3 scaling approximations (Macro coarse): n/8 wavelets of support 8
  const offset4 = (7 * n) / 8;
  for (let k = 0; k < n / 8; k++) {
    for (let j = 0; j < 8; j++) {
      w[offset4 + k]![8 * k + j] = invSqrt8;
    }
  }

  return w;
}

export function orthogonalWaveletDecompose1D(signal: Float64Array | number[]): WaveletDecompositionResult {
  let s = signal instanceof Float64Array ? signal : new Float64Array(signal);
  let n = s.length;
  if (n < 8 || (n & (n - 1)) !== 0) {
    s = padToPowerOf2(s, 8);
    n = s.length;
  }

  const w = buildHaarOrthonormalMatrix(n);
  const coeffs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0.0;
    const row = w[i]!;
    for (let j = 0; j < n; j++) sum += row[j]! * s[j]!;
    coeffs[i] = sum;
  }

  const micro = new Float64Array(n);
  const meso = new Float64Array(n);
  const macro = new Float64Array(n);

  for (let j = 0; j < n; j++) {
    let sumMicro = 0.0;
    for (let i = 0; i < n / 2; i++) sumMicro += w[i]![j]! * coeffs[i]!;
    micro[j] = sumMicro;

    let sumMeso = 0.0;
    for (let i = n / 2; i < (3 * n) / 4; i++) sumMeso += w[i]![j]! * coeffs[i]!;
    meso[j] = sumMeso;

    let sumMacro = 0.0;
    for (let i = (3 * n) / 4; i < n; i++) sumMacro += w[i]![j]! * coeffs[i]!;
    macro[j] = sumMacro;
  }

  return { micro, meso, macro, coeffs };
}

// ============================================================================
// Morphological Component Analysis (MCA) via ISTA
// ============================================================================

export function mcaSeparate1D(
  signal: Float64Array | number[],
  nIter: number = 25,
  lambdaL1: number = 0.15
): MCAResult {
  const s = signal instanceof Float64Array ? signal : new Float64Array(signal);
  const n = s.length;
  const numHarmonics = 4;
  const numFCols = 2 * numHarmonics + 1;
  const dF: number[][] = Array.from({ length: numFCols }, () => new Array<number>(n).fill(0.0));

  // 1. Build Fourier Dictionary D_F
  const invSqrtN = 1.0 / Math.sqrt(n);
  for (let t = 0; t < n; t++) dF[0]![t] = invSqrtN;

  let fColIdx = 1;
  for (let k = 1; k <= numHarmonics; k++) {
    const cosCol = new Float64Array(n);
    const sinCol = new Float64Array(n);
    const freq = (2.0 * Math.PI * k) / n;
    for (let t = 0; t < n; t++) {
      cosCol[t] = Math.cos(freq * t);
      sinCol[t] = Math.sin(freq * t);
    }
    const cNorm = vecNorm(cosCol) || 1.0;
    const sNorm = vecNorm(sinCol) || 1.0;
    for (let t = 0; t < n; t++) {
      dF[fColIdx]![t] = cosCol[t]! / cNorm;
      dF[fColIdx + 1]![t] = sinCol[t]! / sNorm;
    }
    fColIdx += 2;
  }

  // 2. Build Soliton Kernel Dictionary D_S (N x 2N)
  const numSCols = n * 2;
  const dS: number[][] = Array.from({ length: numSCols }, () => new Array<number>(n).fill(0.0));
  let sColIdx = 0;
  for (let tau = 0; tau < n; tau++) {
    for (const omega of [1.5, 2.5]) {
      const col = new Float64Array(n);
      for (let t = 0; t < n; t++) {
        const arg = Math.min(20.0, Math.max(-20.0, omega * (t - tau)));
        col[t] = 1.0 / Math.cosh(arg);
      }
      const colNorm = vecNorm(col) || 1.0;
      for (let t = 0; t < n; t++) {
        dS[sColIdx]![t] = col[t]! / colNorm;
      }
      sColIdx++;
    }
  }

  // Power iteration for Lipschitz constant L_S = ||D_S^T D_S||_2
  let v = new Float64Array(numSCols).fill(1.0 / Math.sqrt(numSCols));
  let lS = 1.0;
  for (let iter = 0; iter < 8; iter++) {
    const u = new Float64Array(n);
    for (let c = 0; c < numSCols; c++) {
      const vc = v[c]!;
      const col = dS[c]!;
      for (let t = 0; t < n; t++) {
        u[t] = (u[t] ?? 0.0) + col[t]! * vc;
      }
    }
    const nextV = new Float64Array(numSCols);
    for (let c = 0; c < numSCols; c++) {
      let sum = 0.0;
      const col = dS[c]!;
      for (let t = 0; t < n; t++) sum += col[t]! * u[t]!;
      nextV[c] = sum;
    }
    lS = vecNorm(nextV);
    if (lS > 1e-12) {
      for (let c = 0; c < numSCols; c++) {
        nextV[c] = (nextV[c] ?? 0.0) / lS;
      }
    }
    v = nextV;
  }

  const etaF = 0.95; // Since D_F columns are orthonormal
  const etaS = 0.95 / Math.max(lS, 1e-5);

  const cF = new Float64Array(numFCols);
  const cS = new Float64Array(numSCols);

  // ISTA Iterations
  for (let it = 0; it < nIter; it++) {
    const res = new Float64Array(n);
    for (let t = 0; t < n; t++) {
      let pred = 0.0;
      for (let c = 0; c < numFCols; c++) pred += dF[c]![t]! * cF[c]!;
      for (let c = 0; c < numSCols; c++) pred += dS[c]![t]! * cS[c]!;
      res[t] = s[t]! - pred;
    }

    for (let c = 0; c < numFCols; c++) {
      let g = 0.0;
      const col = dF[c]!;
      for (let t = 0; t < n; t++) g += col[t]! * res[t]!;
      cF[c] = (cF[c] ?? 0.0) + etaF * g;
    }

    for (let t = 0; t < n; t++) {
      let pred = 0.0;
      for (let c = 0; c < numFCols; c++) pred += dF[c]![t]! * cF[c]!;
      for (let c = 0; c < numSCols; c++) pred += dS[c]![t]! * cS[c]!;
      res[t] = s[t]! - pred;
    }

    const thresh = etaS * lambdaL1;
    for (let c = 0; c < numSCols; c++) {
      let g = 0.0;
      const col = dS[c]!;
      for (let t = 0; t < n; t++) g += col[t]! * res[t]!;
      const gradS = cS[c]! + etaS * g;
      const absG = Math.abs(gradS);
      cS[c] = absG > thresh ? Math.sign(gradS) * (absG - thresh) : 0.0;
    }
  }

  const fourierComp = new Float64Array(n);
  const solitonComp = new Float64Array(n);
  for (let c = 0; c < numFCols; c++) {
    const vVal = cF[c]!;
    const col = dF[c]!;
    for (let t = 0; t < n; t++) {
      fourierComp[t] = (fourierComp[t] ?? 0.0) + col[t]! * vVal;
    }
  }
  for (let c = 0; c < numSCols; c++) {
    const vVal = cS[c]!;
    const col = dS[c]!;
    for (let t = 0; t < n; t++) {
      solitonComp[t] = (solitonComp[t] ?? 0.0) + col[t]! * vVal;
    }
  }
  const residual = new Float64Array(n);
  for (let t = 0; t < n; t++) residual[t] = s[t]! - fourierComp[t]! - solitonComp[t]!;

  return {
    fourierComp,
    solitonComp,
    cF,
    cS,
    residual,
    fourier_comp: fourierComp,
    soliton_comp: solitonComp,
    c_F: cF,
    c_S: cS,
  };
}

// ============================================================================
// Recurrence Quantification Analysis (RQA)
// ============================================================================

export function computeRqaMetrics(macroSignal: Float64Array | number[]): RQAResult {
  const s = macroSignal instanceof Float64Array ? macroSignal : new Float64Array(macroSignal);
  const n = s.length;
  const dists: number[] = [];
  const maxPairs = 5000;
  const step = Math.max(1, Math.floor(n / 100));
  for (let i = 0; i < n && dists.length < maxPairs; i += step) {
    for (let j = i + 1; j < n && dists.length < maxPairs; j += step) {
      dists.push(Math.abs(s[i]! - s[j]!));
    }
  }
  let eps = 1e-4;
  if (dists.length > 0) {
    dists.sort((a, b) => a - b);
    const p25Idx = Math.floor(dists.length * 0.25);
    eps = Math.max(dists[p25Idx] ?? 1e-4, 1e-4);
  }

  const recMat: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));
  let totalRecPoints = 0;

  for (let i = 0; i < n; i++) {
    const si = s[i]!;
    for (let j = 0; j < n; j++) {
      if (Math.abs(si - s[j]!) <= eps) {
        recMat[i]![j] = 1.0;
        if (i !== j) totalRecPoints++;
      }
    }
  }

  // Determinism: diagonal lines parallel to main diagonal of length >= 2
  let totalDiagPoints = 0;
  for (let k = 1; k < n - 1; k++) {
    let curLen = 0;
    for (let i = 0; i < n - k; i++) {
      if (recMat[i]![i + k] === 1.0) {
        curLen++;
      } else {
        if (curLen >= 2) totalDiagPoints += curLen;
        curLen = 0;
      }
    }
    if (curLen >= 2) totalDiagPoints += curLen;
  }

  const denom = Math.max(totalRecPoints, 1.0);
  const det = Math.min(1.0, Math.max(0.0, totalDiagPoints / denom));

  // Long-range return arch index: early quarter vs late quarter recurrence
  const q1 = Math.max(Math.floor(n / 4), 1);
  const q3 = Math.floor((3 * n) / 4);
  let archSum = 0;
  let archCount = 0;
  for (let i = 0; i < q1; i++) {
    for (let j = q3; j < n; j++) {
      archSum += recMat[i]![j]!;
      archCount++;
    }
  }
  const meanArch = archCount > 0 ? archSum / archCount : 0.0;
  const iArch = Math.min(1.0, Math.max(0.0, meanArch * 2.5));

  return {
    det,
    iArch,
    i_arch: iArch,
    lam: 0.0,
    recurrenceMatrix: recMat,
  };
}

// ============================================================================
// WaveletHermiteEncoder Class
// ============================================================================

export class WaveletHermiteEncoder {
  public readonly lexicon: HTHLexicon;

  constructor(lexicon?: HTHLexicon) {
    this.lexicon = lexicon ?? new HTHLexicon();
  }

  /**
   * Extract 4 base structural signals s_m = [s1, s2, s3, s4] for a single clause.
   */
  public extractClauseSignals(clauseText: string): number[] {
    const tokens = this.lexicon.segmentText(clauseText);
    if (!tokens || tokens.length === 0) {
      return [0.1, 0.5, 0.5, 0.0];
    }

    const nTok = tokens.length;
    const contentTokens = tokens.filter((t) => t.pos <= 3 && !CLOSED_LOGICAL.has(t.surface));

    // 1. s1 (Flux): Radial entropy + Transition surprise + Effective sensory density
    const lengths = tokens.map((t) => t.surface.length);
    const totChars = lengths.reduce((acc, l) => acc + l, 0);
    let hRadial = 0.2;
    if (totChars > 0 && nTok > 1) {
      let ent = 0.0;
      for (const l of lengths) {
        const p = l / totChars;
        ent -= p * Math.log2(p + 1e-12);
      }
      hRadial = ent / Math.log2(Math.max(nTok, 2));
    }

    const clean = clauseText.replace(/\s+/g, '');
    let surprise = 0.1;
    if (clean.length >= 3) {
      const counts = new Map<string, number>();
      const nBg = clean.length - 1;
      for (let i = 0; i < nBg; i++) {
        const bg = clean.slice(i, i + 2);
        counts.set(bg, (counts.get(bg) ?? 0) + 1);
      }
      let bgEnt = 0.0;
      for (const c of counts.values()) {
        const p = c / nBg;
        bgEnt -= p * Math.log2(p);
      }
      surprise = bgEnt / Math.log2(Math.max(nBg, 2));
    }

    let sensorySum = 0.0;
    let negMomentum = 0.0;
    if (contentTokens.length > 0) {
      for (const t of contentTokens) {
        const vecSum = t.effectiveSensory8d.reduce((a, b) => a + b, 0.0);
        if (t.isNegated) {
          negMomentum += 0.8 * vecNorm(t.effectiveSensory8d);
        } else {
          sensorySum += vecSum;
        }
      }
    }
    const sensoryDensity = contentTokens.length > 0 ? sensorySum / (contentTokens.length * 8.0) : 0.05;
    const s1 = Math.min(1.0, Math.max(0.0, 0.35 * hRadial + 0.35 * surprise + 0.30 * sensoryDensity));

    // 2. s2 (Lattice): Mora cadence + Token symmetry + Closed logical density
    let moraCount = 0;
    for (const c of clean) {
      if (!'ゃゅょぁぃぅぇぉャュョァィゥェォっッー'.includes(c)) moraCount++;
    }
    const cadenceDist = Math.min(
      Math.abs(moraCount - 5),
      Math.abs(moraCount - 7),
      Math.abs(moraCount - 12),
      Math.abs(moraCount - 17)
    );
    const moraResonance = Math.max(0.0, 1.0 - cadenceDist / 8.0);

    let lapSym = 0.5;
    if (nTok >= 3) {
      let diff = 0.0;
      for (let i = 0; i < Math.floor(nTok / 2); i++) {
        diff += Math.abs(lengths[i]! - lengths[nTok - 1 - i]!);
      }
      lapSym = Math.max(0.0, 1.0 - diff / (totChars || 1.0));
    }

    const logicalCount = tokens.filter((t) => CLOSED_LOGICAL.has(t.surface)).length;
    const logicalDensity = logicalCount / Math.max(nTok, 1);
    const s2 = Math.min(1.0, Math.max(0.0, 0.40 * moraResonance + 0.30 * lapSym + 0.30 * logicalDensity));

    // 3. s3 (Void [間]): Concreteness * Subtraction rate * Apophatic stillness
    let wagoConcreteness = 0.1;
    let apophaticStillness = 0.5;
    if (contentTokens.length > 0) {
      wagoConcreteness = vecMean(contentTokens.map((t) => t.entry.concreteness));
      apophaticStillness = vecMean(contentTokens.map((t) => t.effectiveSensory8d[7] ?? 0.5));
    }
    const nAdjAdv = tokens.filter((t) => (t.pos === 2 || t.pos === 3) && !t.mimeticImmunity).length;
    const subtractionRate = Math.exp(-0.5 * nAdjAdv);
    const s3 = Math.min(1.0, Math.max(0.0, wagoConcreteness * subtractionRate * (0.6 + 0.4 * apophaticStillness)));

    // 4. s4 (Shock): Octree discrete jump + Negation momentum + Phonological edge
    let jumpSum = 0.0;
    let jumpCount = 0;
    if (contentTokens.length >= 2) {
      for (let i = 0; i < contentTokens.length - 1; i++) {
        const u = contentTokens[i]!;
        const v = contentTokens[i + 1]!;
        const dj = discreteJumpDistance(u.entry.octree30, v.entry.octree30);
        jumpSum += dj;
        jumpCount++;
      }
    }
    const jumpVal = jumpCount > 0 ? jumpSum / jumpCount : 0.0;

    let voiced = 0;
    let geminate = 0;
    for (const c of clean) {
      if (VOICED_OBSTRUENTS.has(c)) voiced++;
      if (GEMINATES.has(c)) geminate++;
    }
    const phonologicalEdge = Math.min(1.0, voiced * 0.15 + geminate * 0.35);
    const hasAdversative = tokens.some((t) => CLOSED_ADVERSATIVE.has(t.surface));
    const adversativeImpulse = hasAdversative ? 0.6 : 0.0;
    const s4 = Math.min(1.0, Math.max(0.0, 0.35 * jumpVal + 0.25 * negMomentum + 0.20 * phonologicalEdge + 0.20 * adversativeImpulse));

    return [s1, s2, s3, s4];
  }

  /**
   * Extract sequence of base structural signals s_m across all clauses in narrative text.
   */
  public extractTextSignals(text: string): number[][] {
    const { narrativeText } = this.lexicon.separateNarrativeAndDialogue(text);
    const narrative = narrativeText.length >= 2 ? narrativeText : text;
    const clauseSplits = narrative
      .split(/[。！？!?\n、,]+/)
      .map((c) => c.trim())
      .filter((c) => c.length >= 2);
    const targetClauses = clauseSplits.length > 0 ? clauseSplits : [narrative];
    return targetClauses.map((c) => this.extractClauseSignals(c));
  }

  public static computeLogarithmicScales(m: number): LogarithmicScalesResult {
    return computeLogarithmicScales(m);
  }

  public computeLogarithmicScales(m: number): LogarithmicScalesResult {
    return computeLogarithmicScales(m);
  }

  public static renormalizeDecimateSignals(
    signals: Float64Array[] | number[][],
    maxLength: number = 1024
  ): Float64Array[] {
    return renormalizeDecimateSignals(signals, maxLength);
  }

  public renormalizeDecimateSignals(
    signals: Float64Array[] | number[][],
    maxLength: number = 1024
  ): Float64Array[] {
    return renormalizeDecimateSignals(signals, maxLength);
  }

  /**
   * OPEN-08: Ghost History Warm-Start with James-Stein shrinkage.
   */
  public static applyOpen08GhostPadding(
    rawSignals: number[][] | Float64Array[],
    minLen: number = 16,
    maxLen: number = 1024
  ): Float64Array[] {
    let signals: Float64Array[] | number[][] = rawSignals;
    let m = signals.length;
    if (m === 0) {
      const neutral = new Float64Array([0.1, 0.5, 0.5, 0.0]);
      return Array.from({ length: minLen }, () => new Float64Array(neutral));
    }

    // Gap 2: Renormalization group decimation if sequence exceeds maxLen
    if (m > maxLen) {
      signals = renormalizeDecimateSignals(signals, maxLen);
      m = signals.length;
    }

    let targetLen = minLen;
    while (targetLen < m && targetLen < maxLen) targetLen <<= 1;

    // Compute column-wise stationary mean
    const statMean = new Float64Array(4);
    for (let i = 0; i < m; i++) {
      const row = signals[i]!;
      for (let a = 0; a < 4; a++) {
        statMean[a] = (statMean[a] ?? 0.0) + row[a]!;
      }
    }
    const invM = 1.0 / Math.max(m, 1);
    for (let a = 0; a < 4; a++) {
      statMean[a] = (statMean[a] ?? 0.0) * invM;
    }

    // Reflection padding
    const padded: number[][] = [];
    for (let i = 0; i < m; i++) padded.push([...signals[i]!]);
    while (padded.length < targetLen) {
      const take = Math.min(targetLen - padded.length, m);
      for (let i = 0; i < take; i++) {
        padded.push([...signals[m - 1 - i]!]);
      }
    }

    // James-Stein shrinkage alpha_k = k / (k + 5)
    const shrunk: Float64Array[] = [];
    for (let k = 0; k < targetLen; k++) {
      const alpha = (k + 1.0) / (k + 6.0);
      const row = new Float64Array(4);
      for (let a = 0; a < 4; a++) {
        row[a] = statMean[a]! + alpha * (padded[k]![a]! - statMean[a]!);
      }
      shrunk.push(row);
    }
    return shrunk;
  }

  /**
   * Encode (M, 4) base structural signals into 24D symplectic state z = (q, p).
   */
  public encodeSignals(rawSignals: number[][], useMca: boolean = true): SymplecticStateResult {
    const paddedSignals = WaveletHermiteEncoder.applyOpen08GhostPadding(rawSignals, 16);
    const n = paddedSignals.length;

    // Decompose each of 4 channels
    const microArr: Float64Array[] = [];
    const mesoArr: Float64Array[] = [];
    const macroArr: Float64Array[] = [];

    for (let a = 0; a < 4; a++) {
      const channel = new Float64Array(n);
      for (let i = 0; i < n; i++) channel[i] = paddedSignals[i]![a]!;
      const dec = orthogonalWaveletDecompose1D(channel);
      microArr.push(dec.micro);
      mesoArr.push(dec.meso);
      macroArr.push(dec.macro);
    }

    const q = new Float64Array(12);

    // Micro (j=1): Elements 0..3
    for (let a = 0; a < 4; a++) {
      const col = microArr[a]!;
      const rms = Math.sqrt(vecMean(Array.from(col).map((x) => x * x)));
      const std = vecStd(col);
      const zNorm = Array.from(col).map((x) => x / (std + 1e-8));
      const meanHe3 = vecMean(zNorm.map((x) => x * x * x - 3.0 * x));
      const meanHe4 = vecMean(zNorm.map((x) => x * x * x * x - 6.0 * x * x + 3.0));
      const momentMod = 0.05 * meanHe3 + 0.02 * meanHe4;
      q[a] = Math.min(1.0, Math.max(0.0, rms * 2.5 + momentMod));
    }

    // Meso (j=2): Elements 4..7
    for (let a = 0; a < 4; a++) {
      const col = mesoArr[a]!;
      const rms = Math.sqrt(vecMean(Array.from(col).map((x) => x * x)));
      const std = vecStd(col);
      const zNorm = Array.from(col).map((x) => x / (std + 1e-8));
      const meanHe3 = vecMean(zNorm.map((x) => x * x * x - 3.0 * x));
      const momentMod = 0.05 * meanHe3;
      q[4 + a] = Math.min(1.0, Math.max(0.0, rms * 3.5 + momentMod));
    }

    // Macro (j=3): Elements 8..11
    q[8] = Math.min(1.0, Math.max(0.0, vecMean(macroArr[0]!))); // Flux Macro

    const rqa = computeRqaMetrics(macroArr[1]!); // Element 9: Macro Lattice RQA
    q[9] = Math.min(1.0, Math.max(0.0, 0.5 * rqa.det + 0.5 * rqa.iArch));

    q[10] = Math.min(1.0, Math.max(0.0, vecMean(macroArr[2]!))); // Element 10: Void Macro

    if (useMca) {
      const mca = mcaSeparate1D(macroArr[3]!, 25, 0.15); // Element 11: Shock Soliton MCA
      const fNorm = vecNorm(mca.fourierComp);
      let sL1 = 0.0;
      for (let i = 0; i < mca.cS.length; i++) sL1 += Math.abs(mca.cS[i]!);
      q[11] = Math.min(1.0, Math.max(0.0, sL1 / (fNorm + 1e-4)));
    } else {
      q[11] = Math.min(1.0, Math.max(0.0, vecMean(macroArr[3]!)));
    }

    // Momentum p in R_{>=0}^12
    const p = new Float64Array(12);
    for (let a = 0; a < 4; a++) {
      p[a] = Math.max(0.0, vecStd(microArr[a]!) + diffRms(microArr[a]!));
      p[4 + a] = Math.max(0.0, vecStd(mesoArr[a]!) + diffRms(mesoArr[a]!));
      p[8 + a] = Math.max(0.0, vecStd(macroArr[a]!) + diffRms(macroArr[a]!));
    }

    const z = new Float64Array(24);
    z.set(q, 0);
    z.set(p, 12);

    return { q, p, z };
  }

  /**
   * Encode text or raw signals or music note events into 24D symplectic state.
   */
  public encode(
    textOrSignals: string | number[][] | MusicNoteEvent[],
    useMca: boolean = true
  ): SymplecticStateResult {
    if (typeof textOrSignals === 'string') {
      const signals = this.extractTextSignals(textOrSignals);
      return this.encodeSignals(signals, useMca);
    }
    if (Array.isArray(textOrSignals)) {
      if (textOrSignals.length > 0 && typeof (textOrSignals[0] as MusicNoteEvent).pitch === 'number') {
        const signals = extractMusicFunctorSignals(textOrSignals as MusicNoteEvent[]);
        return this.encodeSignals(signals, useMca);
      }
      return this.encodeSignals(textOrSignals as number[][], useMca);
    }
    throw new TypeError(`Unsupported input type for encode: ${typeof textOrSignals}`);
  }
}

// ============================================================================
// Public API Convenience Function
// ============================================================================

let defaultEncoderInstance: WaveletHermiteEncoder | null = null;

export function extract24dV2(
  textOrSignals: string | number[][] | MusicNoteEvent[],
  useMca: boolean = true
): Float64Array {
  if (!defaultEncoderInstance) {
    defaultEncoderInstance = new WaveletHermiteEncoder();
  }
  return defaultEncoderInstance.encode(textOrSignals, useMca).z;
}
