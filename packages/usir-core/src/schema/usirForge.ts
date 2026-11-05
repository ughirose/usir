/**
 * OSWE Polar-Fourier Phase-Aligned usir-forge & 4KB 3-Mode GMM Binary Compiler.
 *
 * Phase D WS-D1 TypeScript Implementation based on Blueprint Section 3:
 * 1. alignPolarFourierPhase:
 *    Calculates climax peak phase theta_peak = arg(c_1) from fundamental Fourier
 *    coefficient c_1 = 1/M sum_{m=0}^{M-1} E(m) exp(-i 2pi m / M), and circularly
 *    aligns chunks to target phase theta_0 = 3pi/4.
 *    Prevents destructive interference / phase cancellation Deadzone.
 * 2. OSWEStreamingCompiler:
 *    - Welford online update for 3 modes (Mode 1: 地の文, Mode 2: 会話, Mode 3: クライマックス):
 *      means mu_k in R^{24}, covariance Sigma_k in R^{24x24}, and lower-triangular
 *      Cholesky precision factors L_k in R^{300} (Sigma_k^{-1} = L_k L_k^T).
 *    - K >= 5 cluster anonymity gate (prevents single-author de-anonymization).
 *    - compileTierMBinary() -> Uint8Array: strictly 4,096 bytes .usir binary.
 *    - loadTierMBinary(data: Uint8Array) -> DecodedTierM: validates CRC32 and unpacks 4,096 bytes binary.
 */

import {
  encodeTierM,
  decodeTierM,
  type TierMProfile,
  type DecodedTierM,
  USIR_MAGIC,
  TIER_M_BYTE_LENGTH,
} from './usirBinarySchema.js';
import {
  computePackedCholesky24d,
  unpackCholesky24d,
  precisionMatrixFromPackedCholesky24d,
} from '../algebra/genomeCompiler.js';
import { invertMatrixGaussJordan } from '../dynamics/landauPhase.js';

/** Weight of the sample covariance in the Tier-M precision estimate (rest: scaled identity). */
export const SHRINKAGE_ALPHA = 0.75;
import { extractUsir24d } from '../sensor/symplectic24d.js';
import { splitSentences } from '../sensor/dictionaries.js';

export {
  USIR_MAGIC,
  TIER_M_BYTE_LENGTH,
  computePackedCholesky24d,
  unpackCholesky24d,
  precisionMatrixFromPackedCholesky24d,
};

// =============================================================================
// 1. POLAR-FOURIER PHASE ALIGNMENT
// =============================================================================

/**
 * Computes peak phase angle arg(c_1) of fundamental Fourier component c_1.
 * c_1 = (1/M) sum_{m=0}^{M-1} E(m) exp(-i 2pi m / M)
 */
export function computePolarFourierPhase(energy: number[]): number {
  const M = energy.length;
  if (M <= 1) return 0.0;

  for (let m = 0; m < M; m++) {
    if (!Number.isFinite(energy[m])) return 0.0;
  }

  let re = 0.0;
  let im = 0.0;
  for (let m = 0; m < M; m++) {
    const angle = (-2.0 * Math.PI * m) / M;
    const val = energy[m]!;
    re += val * Math.cos(angle);
    im += val * Math.sin(angle);
  }
  re /= M;
  im /= M;

  const mag = Math.sqrt(re * re + im * im);
  if (!Number.isFinite(mag) || mag < 1e-12) return 0.0;

  return Math.atan2(im, re);
}

/**
 * Circularly aligns energy trajectory peak phase to targetPhase (default 3pi/4).
 * Prevents destructive interference / phase cancellation Deadzone when aggregating chunks.
 * Supports both 1D energy sequences and 2D vector trajectories (M x D).
 */
export function alignPolarFourierPhase<T extends number[] | number[][]>(
  energyTrajectory: T,
  targetPhase?: number
): T;
export function alignPolarFourierPhase<T extends number[] | number[][]>(
  energyTrajectory: T,
  targetPhase: number,
  returnShift: true
): { aligned: T; shift: number };
export function alignPolarFourierPhase<T extends number[] | number[][]>(
  energyTrajectory: T,
  targetPhase = (3.0 * Math.PI) / 4.0,
  returnShift = false
): T | { aligned: T; shift: number } {
  const is2D = Array.isArray(energyTrajectory[0]);
  const isFiniteSignal = is2D
    ? (energyTrajectory as number[][]).every((row) => row.every((v) => Number.isFinite(v)))
    : (energyTrajectory as number[]).every((v) => Number.isFinite(v));

  if (!isFiniteSignal || energyTrajectory.length <= 1) {
    const res = (Array.isArray(energyTrajectory[0])
      ? (energyTrajectory as number[][]).map((r) => [...r])
      : [...(energyTrajectory as number[])]) as T;
    return returnShift ? { aligned: res, shift: 0 } : res;
  }

  const M = energyTrajectory.length;

  let energy: number[];
  if (is2D) {
    const matrix = energyTrajectory as number[][];
    energy = matrix.map((vec) => {
      if (vec.length >= 24) {
        let p2 = 0.0;
        for (let i = 12; i < 24; i++) {
          p2 += (vec[i] ?? 0) * (vec[i] ?? 0);
        }
        const shock = ((vec[3] ?? 0) + (vec[7] ?? 0) + (vec[11] ?? 0)) / 3.0;
        return 0.5 * p2 + shock;
      } else {
        let sumSq = 0.0;
        for (const val of vec) sumSq += val * val;
        return Math.sqrt(sumSq);
      }
    });
  } else {
    energy = energyTrajectory as number[];
  }

  const thetaPeak = computePolarFourierPhase(energy);
  const diff = thetaPeak - targetPhase;
  const twoPi = 2.0 * Math.PI;
  const diffWrapped = ((diff + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
  // Deterministic half-up rounding matching Python floor(x + 0.5)
  let s = Math.floor((M * diffWrapped) / twoPi + 0.5) % M;
  if (s < 0) s += M;

  if (s === 0) {
    const res = (is2D
      ? (energyTrajectory as number[][]).map((r) => [...r])
      : [...(energyTrajectory as number[])]) as T;
    return returnShift ? { aligned: res, shift: 0 } : res;
  }

  let aligned: T;
  if (is2D) {
    const matrix = energyTrajectory as number[][];
    const rolled: number[][] = new Array(M);
    for (let i = 0; i < M; i++) {
      rolled[(i + s) % M] = [...matrix[i]!];
    }
    aligned = rolled as T;
  } else {
    const arr = energyTrajectory as number[];
    const rolled: number[] = new Array(M);
    for (let i = 0; i < M; i++) {
      rolled[(i + s) % M] = arr[i]!;
    }
    aligned = rolled as T;
  }

  return returnShift ? { aligned, shift: s } : aligned;
}

// =============================================================================
// 2. 4,096 BYTES TIER M BINARY PACK / UNPACK
// =============================================================================

export function compileTierMBinary(profile: TierMProfile): Uint8Array {
  return encodeTierM(profile);
}

export function loadTierMBinary(data: Uint8Array | ArrayBuffer): DecodedTierM {
  return decodeTierM(data);
}

// =============================================================================
// 3. MODE CLASSIFICATION
// =============================================================================

export function classifyMode24d(z: number[]): number {
  if (z.length < 24) return 0;

  const q = z.slice(0, 12);
  const p = z.slice(12, 24);

  const gFlux = ((q[0] ?? 0) + (q[4] ?? 0) + (q[8] ?? 0)) / 3.0;
  const gLattice = ((q[1] ?? 0) + (q[5] ?? 0) + (q[9] ?? 0)) / 3.0;
  const gVoid = ((q[2] ?? 0) + (q[6] ?? 0) + (q[10] ?? 0)) / 3.0;
  const gShock = ((q[3] ?? 0) + (q[7] ?? 0) + (q[11] ?? 0)) / 3.0;

  let pNormSq = 0.0;
  for (let i = 0; i < 12; i++) {
    pNormSq += (p[i] ?? 0) * (p[i] ?? 0);
  }
  const pNorm = Math.sqrt(pNormSq);

  const scoreNarration = gLattice * 1.3 + gVoid * 1.2;
  const scoreDialogue = gFlux * 1.5 + pNorm * 0.4;
  const scoreClimax = gShock * 1.8 + pNorm * 1.5;

  if (scoreClimax > scoreNarration && scoreClimax > scoreDialogue) {
    return 2; // Climax
  } else if (scoreDialogue > scoreNarration) {
    return 1; // Dialogue
  } else {
    return 0; // Narration
  }
}

// -----------------------------------------------------------------------------
// Quantile 3-mode assignment (Phase 26.1; parity with usir_forge.assign_modes_quantile)
// -----------------------------------------------------------------------------

export const QUANTILE_MIN_WINDOWS = 20;
export const QUANTILE_DIALOGUE_FRACTION = 0.25;
export const QUANTILE_CLIMAX_FRACTION = 0.25;

function zscoreVec(x: number[]): number[] {
  const n = x.length;
  let mean = 0;
  for (const v of x) mean += v;
  mean /= n;
  let varSum = 0;
  for (const v of x) varSum += (v - mean) * (v - mean);
  const sd = Math.sqrt(varSum / n);
  if (sd < 1e-12) return x.map(() => 0);
  return x.map((v) => (v - mean) / sd);
}

function topKIndices(values: number[], candidates: number[], k: number): number[] {
  return [...candidates].sort((a, b) => values[b]! - values[a]! || a - b).slice(0, k);
}

/** Python round(): half to even. */
function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Corpus-level quantile assignment: Dialogue (1) = top 25% by `dialogue`; of the rest Climax (2) =
 * top 25% by z(g_shock)+z(||p||); remaining Narration (0). Ties -> smaller index.
 * Returns null (caller falls back to the rule) for fewer than QUANTILE_MIN_WINDOWS windows.
 */
export function assignModesQuantile(Z: number[][], dialogue: number[]): number[] | null {
  const n = Z.length;
  if (dialogue.length !== n) throw new Error(`dialogue length ${dialogue.length} != windows ${n}`);
  if (n < QUANTILE_MIN_WINDOWS) return null;
  const gShock = Z.map((z) => ((z[3] ?? 0) + (z[7] ?? 0) + (z[11] ?? 0)) / 3.0);
  const pNorm = Z.map((z) => {
    let s = 0;
    for (let i = 12; i < 24; i++) s += (z[i] ?? 0) * (z[i] ?? 0);
    return Math.sqrt(s);
  });
  const zg = zscoreVec(gShock);
  const zp = zscoreVec(pNorm);
  const shockScore = zg.map((v, i) => v + zp[i]!);

  const modes = new Array<number>(n).fill(0);
  const all = Array.from({ length: n }, (_, i) => i);
  const k1 = roundHalfEven(QUANTILE_DIALOGUE_FRACTION * n);
  for (const i of topKIndices(dialogue, all, k1)) modes[i] = 1;
  const rest = all.filter((i) => modes[i] === 0);
  const k2 = roundHalfEven(QUANTILE_CLIMAX_FRACTION * rest.length);
  for (const i of topKIndices(shockScore, rest, k2)) modes[i] = 2;
  return modes;
}

// =============================================================================
// 4. OSWE STREAMING COMPILER
// =============================================================================

export interface OSWEStreamingCompilerOptions {
  minClusters?: number;
  targetPhase?: number;
}

export class OSWEStreamingCompiler {
  public readonly minClusters: number;
  public readonly targetPhase: number;
  public readonly clusters: Set<string>;

  public counts: [number, number, number];
  public means: number[][]; // 3 x 24
  public M2: number[][][]; // 3 x 24 x 24
  public transitionCounts: number[][]; // 3 x 3
  public prevMode: number | null;
  public lastCluster: string | null;
  public sampleCount: number;

  constructor(options?: OSWEStreamingCompilerOptions) {
    this.minClusters = options?.minClusters ?? 5;
    this.targetPhase = options?.targetPhase ?? (3.0 * Math.PI) / 4.0;
    this.clusters = new Set<string>();

    this.counts = [0, 0, 0];
    this.means = [
      new Array(24).fill(0.0),
      new Array(24).fill(0.0),
      new Array(24).fill(0.0),
    ];
    this.M2 = [
      new Array(24).fill(0).map(() => new Array(24).fill(0.0)),
      new Array(24).fill(0).map(() => new Array(24).fill(0.0)),
      new Array(24).fill(0).map(() => new Array(24).fill(0.0)),
    ];
    this.transitionCounts = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ];
    this.prevMode = null;
    this.lastCluster = null;
    this.sampleCount = 0;
  }

  get clusterCount(): number {
    return this.clusters.size;
  }

  public resetBoundary(): void {
    this.prevMode = null;
  }

  public registerCluster(clusterId: string | number): void {
    const cid = String(clusterId).trim();
    if (!cid || cid === 'undefined' || cid === 'null') {
      throw new Error('clusterId cannot be empty or whitespace');
    }
    this.clusters.add(cid);
  }

  public normalizeMode(mode: number | string): number {
    if (typeof mode === 'number') {
      if (mode === 0 || mode === 1 || mode === 2) return mode;
      if (mode === 3) return 2;
      throw new Error(`Invalid integer mode ${mode}; expected 0, 1, 2 or 3`);
    }

    const s = String(mode).trim().toLowerCase();
    if (['0', 'narration', 'narrative', '地の文', 'setup', 'mode 0', 'mode_0', 'mode0', 'mode 1', 'mode_1', 'mode1'].includes(s)) {
      return 0;
    }
    if (['1', 'dialogue', 'dialog', '会話', 'polyphony', 'mode 2', 'mode_2', 'mode2'].includes(s)) {
      return 1;
    }
    if (['2', '3', 'climax', 'クライマックス', 'shock', 'mode 3', 'mode_3', 'mode3'].includes(s)) {
      return 2;
    }
    throw new Error(`Unrecognized mode identifier: ${mode}`);
  }

  public addSample(
    vector: number[],
    mode?: number | string,
    clusterId?: string | number
  ): number {
    if (vector.length !== 24) {
      throw new Error(`Expected 24-dimensional vector, got ${vector.length}`);
    }

    if (clusterId !== undefined && clusterId !== null) {
      const cid = String(clusterId).trim();
      if (!cid || cid === 'undefined' || cid === 'null') {
        throw new Error('clusterId cannot be empty or whitespace');
      }
      this.clusters.add(cid);
      if (this.lastCluster !== null && cid !== this.lastCluster) {
        this.resetBoundary();
      }
      this.lastCluster = cid;
    }

    const modeIdx = mode !== undefined ? this.normalizeMode(mode) : classifyMode24d(vector);

    // Welford online update
    const prevCount = this.counts[modeIdx as 0 | 1 | 2] ?? 0;
    const n = prevCount + 1;
    this.counts[modeIdx as 0 | 1 | 2] = n;
    const mean = this.means[modeIdx]!;
    const m2 = this.M2[modeIdx]!;

    const delta: number[] = new Array(24);
    for (let d = 0; d < 24; d++) {
      delta[d] = (vector[d] ?? 0) - (mean[d] ?? 0);
      mean[d] = (mean[d] ?? 0) + delta[d]! / n;
    }

    const delta2: number[] = new Array(24);
    for (let d = 0; d < 24; d++) {
      delta2[d] = (vector[d] ?? 0) - (mean[d] ?? 0);
    }

    for (let r = 0; r < 24; r++) {
      for (let c = 0; c < 24; c++) {
        m2[r]![c] = (m2[r]![c] ?? 0) + delta[r]! * delta2[c]!;
      }
    }

    // Symmetrize against float drift
    for (let r = 0; r < 24; r++) {
      for (let c = r + 1; c < 24; c++) {
        const avg = 0.5 * ((m2[r]![c] ?? 0) + (m2[c]![r] ?? 0));
        m2[r]![c] = avg;
        m2[c]![r] = avg;
      }
    }

    // Transition tracking
    if (this.prevMode !== null) {
      this.transitionCounts[this.prevMode]![modeIdx] =
        (this.transitionCounts[this.prevMode]![modeIdx] ?? 0.0) + 1.0;
    }
    this.prevMode = modeIdx;
    this.sampleCount++;

    return modeIdx;
  }

  public addTrajectory(
    trajectory: number[][],
    modes?: (number | string | undefined)[],
    clusterId?: string | number,
    alignPhase = true
  ): number[] {
    if (trajectory.length === 0) return [];
    for (const vec of trajectory) {
      if (vec.length !== 24) {
        throw new Error(`Expected trajectory of 24D vectors, got length ${vec.length}`);
      }
    }

    let traj = trajectory;
    let activeModes = modes ? [...modes] : undefined;

    if (alignPhase && trajectory.length > 1) {
      const res = alignPolarFourierPhase(trajectory, this.targetPhase, true);
      traj = res.aligned;
      const s = res.shift;
      if (activeModes && activeModes.length === traj.length && s !== 0) {
        const M = traj.length;
        const rolled: (number | string | undefined)[] = new Array(M);
        for (let i = 0; i < M; i++) {
          rolled[(i + s) % M] = activeModes[i];
        }
        activeModes = rolled;
      }
    }

    const assignedModes: number[] = [];
    for (let i = 0; i < traj.length; i++) {
      const m = activeModes && i < activeModes.length ? activeModes[i] : undefined;
      const assigned = this.addSample(traj[i]!, m, clusterId);
      assignedModes.push(assigned);
    }

    return assignedModes;
  }

  public addChunk(
    text: string,
    clusterId?: string | number,
    alignPhase = true
  ): number[] {
    this.resetBoundary();
    const sentences = splitSentences(text);
    if (sentences.length === 0) return [];

    const vectors: number[][] = [];
    const modes: (number | undefined)[] = [];

    for (const sent of sentences) {
      const sClean = sent.trim();
      if (!sClean) continue;
      const ext = extractUsir24d(sClean);
      vectors.push([...ext.z_24d]);
      if (sClean.includes('「') || sClean.includes('『')) {
        modes.push(1); // Dialogue
      } else {
        modes.push(undefined); // Auto-classify
      }
    }

    if (vectors.length === 0) return [];
    return this.addTrajectory(vectors, modes, clusterId, alignPhase);
  }

  public compileTierMBinary(options?: { enforceAnonymity?: boolean }): Uint8Array {
    const enforce = options?.enforceAnonymity ?? true;
    if (enforce && this.clusters.size < this.minClusters) {
      throw new Error(
        `Cluster anonymity gate failed: K=${this.clusters.size} < ${this.minClusters}. ` +
          `At least ${this.minClusters} distinct clusters/authors required to prevent single-author de-anonymization.`
      );
    }

    // 1. Weights
    const totalSamples = this.counts[0] + this.counts[1] + this.counts[2];
    const weights: number[] = totalSamples > 0
      ? [
          this.counts[0] / totalSamples,
          this.counts[1] / totalSamples,
          this.counts[2] / totalSamples,
        ]
      : [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

    // 2. Centroids
    const centroids: number[][] = [];
    for (let k = 0; k < 3; k++) {
      const count = this.counts[k as 0 | 1 | 2] ?? 0;
      if (count > 0) {
        centroids.push([...(this.means[k] ?? new Array(24).fill(0.5))]);
      } else {
        centroids.push(new Array(24).fill(0.5));
      }
    }

    // 3. Transition Matrix
    const transitionMatrix: number[][] = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ];
    for (let r = 0; r < 3; r++) {
      let rowSum = 0.0;
      for (let c = 0; c < 3; c++) {
        const prior = r === c ? 3.0 : 1.0;
        const val = (this.transitionCounts[r]![c] ?? 0.0) + prior;
        transitionMatrix[r]![c] = val;
        rowSum += val;
      }
      for (let c = 0; c < 3; c++) {
        transitionMatrix[r]![c] = transitionMatrix[r]![c]! / rowSum;
      }
    }

    // 4. Covariance & Cholesky Factors
    const choleskyFactors: number[][] = [];
    for (let k = 0; k < 3; k++) {
      const Nk = this.counts[k as 0 | 1 | 2] ?? 0;
      const m2 = this.M2[k]!;
      const S_k: number[][] = new Array(24).fill(0).map(() => new Array(24).fill(0.0));

      if (Nk > 1) {
        for (let r = 0; r < 24; r++) {
          for (let c = 0; c < 24; c++) {
            S_k[r]![c] = (m2[r]![c] ?? 0.0) / (Nk - 1.0);
          }
        }
      }

      // Scaled-identity shrinkage: Sigma_k = a * S_k + (1 - a) * (tr(S_k)/24) * I, a = 0.75.
      // For PSD S_k, cond(Sigma_k) <= 73 < 500. Modes without usable variance keep the legacy 0.8 * I.
      let trace = 0.0;
      for (let r = 0; r < 24; r++) trace += S_k[r]![r]!;
      const meanVar = trace / 24.0;
      const Sigma_k: number[][] = new Array(24).fill(0).map((_, r) => {
        const row = new Array(24).fill(0.0);
        for (let c = 0; c < 24; c++) {
          if (meanVar < 1e-8) {
            row[c] = r === c ? 0.8 : 0.0;
          } else {
            row[c] = SHRINKAGE_ALPHA * S_k[r]![c]! + (r === c ? (1.0 - SHRINKAGE_ALPHA) * meanVar : 0.0);
          }
        }
        return row;
      });

      // Precision matrix Lambda_k = Sigma_k^-1
      const Lambda_k = invertMatrixGaussJordan(Sigma_k);

      // Cholesky factor L_k: Lambda_k = L_k L_k^T
      const packedL = computePackedCholesky24d(Lambda_k);
      choleskyFactors.push(packedL);
    }

    const profile: TierMProfile = {
      version: 1,
      flags: 0,
      transition_matrix: transitionMatrix,
      weights,
      centroids,
      cholesky_factors: choleskyFactors,
    };

    return compileTierMBinary(profile);
  }

  public static loadTierMBinary(data: Uint8Array | ArrayBuffer): DecodedTierM {
    return loadTierMBinary(data);
  }
}
