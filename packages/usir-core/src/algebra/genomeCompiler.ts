/**
 * @nishiki/usir-core: Personal Style Genome Compiler (TASK-364 / Step 3A)
 * Compiles corpus text samples into strict 4096-byte Tier M Style Genome binary profiles (.usir).
 * Enforces 3-Room Clean Room non-retention and zero-leakage invariant.
 */

import {
  dirtyExtract,
  auditCleanRoomBinary,
  type CleanRoomAuditResult,
} from '../cleanroom/threeRoomGate.js';
import {
  encodeTierM,
  type TierMProfile,
} from '../schema/usirBinarySchema.js';
import { splitSentences } from '../sensor/dictionaries.js';
import {
  getDefaultClassifier,
  predictPhase,
  invertMatrixGaussJordan,
  type UsirPhase,
  type PhaseCentroidsAndCov,
} from '../dynamics/landauPhase.js';
import type { CanonicalParticle } from '../presets/canonicalParticles.js';

export type UsirTierMProfile = TierMProfile;

export interface CompiledStyleGenome {
  binary: Uint8Array;
  profile: UsirTierMProfile;
  audit: CleanRoomAuditResult;
  dominantPhase: UsirPhase;
  centroid24d: number[];
}

export interface CompileGenomeOptions {
  genomeName?: string;
  minSubwindowChars?: number;
  classifier?: PhaseCentroidsAndCov;
}

/**
 * Loads or builds the 40-chunk empirical phase classifier (in-memory browser/worker safe).
 */
export function getReferenceClassifier(): PhaseCentroidsAndCov {
  return getDefaultClassifier();
}

/**
 * Computes the Cholesky decomposition of a symmetric positive-definite matrix:
 * A = L * L^T where L is lower-triangular.
 * Returns 300 lower-triangular elements packed row-major (j <= i).
 */
export function computePackedCholesky24d(A: number[][]): number[] {
  const D = 24;
  const L: number[][] = new Array(D).fill(0).map(() => new Array(D).fill(0.0));

  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0.0;
      for (let k = 0; k < j; k++) {
        sum += L[i]![k]! * L[j]![k]!;
      }
      if (i === j) {
        const val = A[i]![i]! - sum;
        L[i]![j] = Math.sqrt(Math.max(val, 1e-4));
      } else {
        const diag = L[j]![j]!;
        L[i]![j] = (A[i]![j]! - sum) / (Math.abs(diag) > 1e-12 ? diag : 1e-12);
      }
    }
  }

  const packed: number[] = [];
  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) {
      packed.push(L[i]![j]!);
    }
  }
  return packed;
}

/**
 * Unpacks 300 lower-triangular elements back into a 24x24 lower-triangular matrix L.
 */
export function unpackCholesky24d(packed: number[]): number[][] {
  const D = 24;
  const L: number[][] = new Array(D).fill(0).map(() => new Array(D).fill(0.0));
  let idx = 0;
  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) {
      L[i]![j] = packed[idx++] ?? 0.0;
    }
  }
  return L;
}

/**
 * Reconstructs a full 24x24 precision matrix Lambda = L * L^T from packed Cholesky factor.
 */
export function precisionMatrixFromPackedCholesky24d(packed: number[]): number[][] {
  const D = 24;
  const L = unpackCholesky24d(packed);
  const Lambda: number[][] = new Array(D).fill(0).map(() => new Array(D).fill(0.0));

  for (let r = 0; r < D; r++) {
    for (let c = 0; c < D; c++) {
      let dot = 0.0;
      const maxK = Math.min(r, c);
      for (let k = 0; k <= maxK; k++) {
        dot += L[r]![k]! * L[c]![k]!;
      }
      Lambda[r]![c] = dot;
    }
  }
  return Lambda;
}

/**
 * Compiles an author or corpus text array into a strict 4096-byte Tier M Style Genome.
 */
export function compileStyleGenomeToTierM(
  texts: string[],
  options?: CompileGenomeOptions
): CompiledStyleGenome {
  if (texts.length === 0) {
    throw new Error('compileStyleGenomeToTierM requires at least one text chunk.');
  }

  const minChars = options?.minSubwindowChars ?? 25;
  const D = 24;

  // =========================================================================
  // Room 1 (Dirty Room): Extract raw numerical symplectic vectors
  // =========================================================================
  const sampleVectors: number[][] = [];
  const sequenceModeIndices: number[] = [];

  for (const chunk of texts) {
    const trimmed = chunk.trim();
    if (trimmed.length === 0) continue;

    // Full chunk extraction
    const fullExt = dirtyExtract(trimmed);
    sampleVectors.push([...fullExt.z_24d]);

    // Sub-window extraction (sliding 3 sentences)
    const sents = splitSentences(trimmed);
    if (sents.length >= 3) {
      for (let i = 0; i <= sents.length - 3; i++) {
        const sub = sents.slice(i, i + 3).join('');
        if (sub.length >= minChars) {
          const subExt = dirtyExtract(sub);
          sampleVectors.push([...subExt.z_24d]);
        }
      }
    }
  }

  if (sampleVectors.length === 0) {
    throw new Error('Failed to extract valid symplectic samples from input texts.');
  }

  // =========================================================================
  // Mode Partitioning: 3 Dynamic Modes
  // Mode 0: Setup / Lattice-Flux
  // Mode 1: Rupture / Shock-Momentum
  // Mode 2: Stillness / Void-Caesura
  // =========================================================================
  const modeBuckets: number[][][] = [[], [], []];

  for (const z of sampleVectors) {
    const q = z.slice(0, 12);
    const p = z.slice(12, 24);

    const gFlux = ((q[0] ?? 0) + (q[4] ?? 0) + (q[8] ?? 0)) / 3.0;
    const gLattice = ((q[1] ?? 0) + (q[5] ?? 0) + (q[9] ?? 0)) / 3.0;
    const gVoid = ((q[2] ?? 0) + (q[6] ?? 0) + (q[10] ?? 0)) / 3.0;
    const gShock = ((q[3] ?? 0) + (q[7] ?? 0) + (q[11] ?? 0)) / 3.0;

    const pNorm = Math.sqrt(p.reduce((acc, v) => acc + v * v, 0));

    // Affinity scores
    const score0 = gLattice * 1.3 + gFlux * 1.0; // Setup / Lattice-Flux
    const score1 = gShock * 1.4 + pNorm * 1.2;    // Rupture / Shock-Momentum
    const score2 = gVoid * 1.5 - gShock * 0.5 - pNorm * 0.4; // Stillness / Void-Caesura

    let mode = 0;
    if (score1 > score0 && score1 > score2) {
      mode = 1;
    } else if (score2 > score0 && score2 > score1) {
      mode = 2;
    } else {
      mode = 0;
    }

    modeBuckets[mode]!.push(z);
    sequenceModeIndices.push(mode);
  }

  // Ensure no bucket is empty (fallback to overall mean with slight perturbations)
  const grandCentroid = new Array<number>(D).fill(0.0);
  for (const z of sampleVectors) {
    for (let d = 0; d < D; d++) {
      grandCentroid[d] = (grandCentroid[d] ?? 0.0) + z[d]! / sampleVectors.length;
    }
  }

  for (let k = 0; k < 3; k++) {
    if (modeBuckets[k]!.length === 0) {
      const synthetic = [...grandCentroid];
      if (k === 1) {
        synthetic[3] = Math.min(1.0, (synthetic[3] ?? 0) + 0.2); // Boost shock
      } else if (k === 2) {
        synthetic[2] = Math.min(1.0, (synthetic[2] ?? 0) + 0.2); // Boost void
      }
      modeBuckets[k]!.push(synthetic);
    }
  }

  // 1. Mode Weights (strictly normalized to sum to 1.0 even if empty-bucket synthetic samples were added)
  const totalBucketSamples = modeBuckets.reduce((acc, b) => acc + b.length, 0);
  const weights = modeBuckets.map((b) => b.length / totalBucketSamples);

  // 2. Mode Centroids
  const centroids: number[][] = [];
  for (let k = 0; k < 3; k++) {
    const bucket = modeBuckets[k]!;
    const mu_k = new Array<number>(D).fill(0.0);
    for (const z of bucket) {
      for (let d = 0; d < D; d++) {
        mu_k[d] = (mu_k[d] ?? 0.0) + z[d]! / bucket.length;
      }
    }
    centroids.push(mu_k);
  }

  // 3. 3x3 HSMM Transition Matrix
  const transitionMatrix: number[][] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
  ];

  // Sequence transitions
  for (let i = 0; i < sequenceModeIndices.length - 1; i++) {
    const from = sequenceModeIndices[i]!;
    const to = sequenceModeIndices[i + 1]!;
    transitionMatrix[from]![to] = (transitionMatrix[from]![to] ?? 0.0) + 1.0;
  }

  // Laplace smoothing and self-persistence prior (HSMM state duration)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const prior = r === c ? 3.0 : 1.0;
      transitionMatrix[r]![c] = (transitionMatrix[r]![c] ?? 0.0) + prior;
    }
    const rowSum = transitionMatrix[r]!.reduce((s, v) => s + v, 0);
    for (let c = 0; c < 3; c++) {
      transitionMatrix[r]![c] = transitionMatrix[r]![c]! / rowSum;
    }
  }

  // 4. Covariance with Ledoit-Wolf Shrinkage & Cholesky Precision Factors
  const choleskyFactors: number[][] = [];

  for (let k = 0; k < 3; k++) {
    const bucket = modeBuckets[k]!;
    const mu_k = centroids[k]!;
    const Nk = bucket.length;

    // Sample covariance
    const S_k: number[][] = new Array(D).fill(0).map(() => new Array(D).fill(0.0));
    if (Nk > 1) {
      for (const z of bucket) {
        for (let r = 0; r < D; r++) {
          const diffR = z[r]! - mu_k[r]!;
          for (let c = 0; c < D; c++) {
            const diffC = z[c]! - mu_k[c]!;
            S_k[r]![c] = (S_k[r]![c] ?? 0.0) + (diffR * diffC) / (Nk - 1);
          }
        }
      }
    }

    // Ledoit-Wolf shrinkage: Sigma_k = 0.20 * S_k + 0.80 * I
    const Sigma_k: number[][] = new Array(D).fill(0).map((_, r) => {
      const row = new Array(D).fill(0.0);
      for (let c = 0; c < D; c++) {
        row[c] = 0.20 * S_k[r]![c]! + (r === c ? 0.80 : 0.0);
      }
      return row;
    });

    // Invert to precision matrix Lambda_k = Sigma_k^-1
    const Lambda_k = invertMatrixGaussJordan(Sigma_k);

    // Cholesky decomposition of Lambda_k = L_k * L_k^T
    const packedL = computePackedCholesky24d(Lambda_k);
    choleskyFactors.push(packedL);
  }

  // Overall centroid
  const overallCentroid = new Array<number>(D).fill(0.0);
  for (let d = 0; d < D; d++) {
    let sum = 0.0;
    for (let k = 0; k < 3; k++) {
      sum += weights[k]! * centroids[k]![d]!;
    }
    overallCentroid[d] = sum;
  }

  // Dominant phase classification
  const classifier = options?.classifier ?? getReferenceClassifier();
  const phasePred = predictPhase(overallCentroid, classifier);
  const dominantPhase = phasePred.phase;

  // Build Tier M profile
  const profile: TierMProfile = {
    version: 1,
    flags: 0,
    transition_matrix: transitionMatrix,
    weights,
    centroids,
    cholesky_factors: choleskyFactors,
  };

  // =========================================================================
  // Room 2 & 3: Encode to strictly 4096 bytes and Audit Gate Verification
  // =========================================================================
  const binary = encodeTierM(profile);
  const fullRawText = texts.join('\n');
  const audit = auditCleanRoomBinary(binary, fullRawText);

  if (!audit.passed) {
    throw new Error(
      `Clean Room Audit failed for compiled Style Genome: leaked 7-grams=${audit.leaked7Grams.length}, banned keywords=${audit.bannedWordsFound.join(',')}`
    );
  }

  return {
    binary,
    profile,
    audit,
    dominantPhase,
    centroid24d: overallCentroid,
  };
}

/**
 * Converts a compiled 4096-byte Tier M profile into a CanonicalParticle
 * compatible with mixStyleParticles and style algebra.
 */
export function tierMToCanonicalParticle(
  name: string,
  phase: UsirPhase,
  profile: TierMProfile
): CanonicalParticle {
  const D = 24;
  const weights = profile.weights ?? [1 / 3, 1 / 3, 1 / 3];
  const sumW = weights.reduce((s, w) => s + w, 0) || 1.0;
  const normW = weights.map((w) => w / sumW);

  // Overall centroid mu = sum_k pi_k mu_k
  const mu = new Array<number>(D).fill(0.0);
  for (let k = 0; k < 3; k++) {
    const wk = normW[k]!;
    const centroid = profile.centroids[k] ?? new Array(D).fill(0.5);
    for (let d = 0; d < D; d++) {
      mu[d] = (mu[d] ?? 0.0) + wk * (centroid[d] ?? 0.0);
    }
  }

  // Combined precision matrix Lambda = sum_k pi_k Lambda_k
  const Lambda = new Array(D).fill(0).map(() => new Array(D).fill(0.0));
  for (let k = 0; k < 3; k++) {
    const wk = normW[k]!;
    const lFlat = profile.cholesky_factors[k] ?? new Array(300).fill(0.0);
    const Lambda_k = precisionMatrixFromPackedCholesky24d(lFlat);
    for (let r = 0; r < D; r++) {
      for (let c = 0; c < D; c++) {
        Lambda[r]![c] = (Lambda[r]![c] ?? 0.0) + wk * Lambda_k[r]![c]!;
      }
    }
  }

  // Packed Cholesky factor of combined precision
  const cholesky_flat = computePackedCholesky24d(Lambda);

  return {
    name,
    phase,
    description: `Tier M Compiled Genome Particle: ${name}`,
    mu,
    precision: Lambda,
    cholesky_flat,
  };
}
