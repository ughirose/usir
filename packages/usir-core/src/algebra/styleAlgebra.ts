/**
 * @nishiki/usir-core: Style Algebra Operators (TASK-364 / Step 3B & Gap 5)
 * Implements the 4 canonical operators on the USIR Symplectic Manifold:
 * 1. Anti-Slop Orthogonal Projection (projectOutSlopSubspace)
 * 2. Riemannian / Cholesky Geodesic Interpolation (interpolateGeodesic)
 * 3. Cross-Genre Register Transposition (transposeStyleRegister)
 * 4. Unexplored Style Discovery (discoverVoronoiGenesis / usirGenesis)
 */

import {
  dirtyExtract,
} from '../cleanroom/threeRoomGate.js';
import {
  encodeTierM,
  decodeTierM,
  type TierMProfile,
} from '../schema/usirBinarySchema.js';
import {
  predictPhase,
  invertMatrixGaussJordan,
  type UsirPhase,
  type PhaseCentroidsAndCov,
} from '../dynamics/landauPhase.js';
import type { CanonicalParticle } from '../presets/canonicalParticles.js';
import {
  getReferenceClassifier,
  precisionMatrixFromPackedCholesky24d,
  type UsirTierMProfile,
} from './genomeCompiler.js';

/**
 * Empirical centroid of Phase V (V_Deadzone: AI slop) derived from 40-chunk golden benchmark.
 */
export const DEFAULT_DEADZONE_CENTROID_24D: readonly number[] = [
  0.8159, 0.5512, 0.2221, 0.1829, 0.6235, 0.6302, 0.143, 0.2235, 0.2692, 0.5503, 0.2796, 0.1135,
  0.0428, 0.0169, 0.051, 0.0442, 0.0525, 0.0125, 0.041, 0.0513, 0.0283, 0.0076, 0.0068, 0.0367,
];

// =============================================================================
// OPERATOR ①: ANTI-SLOP ORTHOGONAL PROJECTION
// =============================================================================

/**
 * Projects a 24D state vector z out of the AI Deadzone subspace.
 * Subtracts the projection along the deadzone unit vector and injects dynamic vitality.
 *
 * @param z Current 24D symplectic state vector (q, p)
 * @param alpha Projection strength (default 0.55)
 * @param deadzoneCentroid Optional custom deadzone centroid (default DEFAULT_DEADZONE_CENTROID_24D)
 * @returns Purified 24D state vector
 */
export function projectOutSlopSubspace(
  z: number[],
  alpha = 0.55,
  deadzoneCentroid: number[] = [...DEFAULT_DEADZONE_CENTROID_24D]
): number[] {
  if (z.length !== 24) {
    throw new Error(`projectOutSlopSubspace requires 24D state vector, got length ${z.length}`);
  }

  // Calculate unit direction of deadzone
  const deadNorm = Math.sqrt(deadzoneCentroid.reduce((s, v) => s + v * v, 0));
  const uDead = deadzoneCentroid.map((v) => (deadNorm > 0 ? v / deadNorm : 0));

  // Inner product <z, u_dead>
  const dot = z.reduce((s, v, i) => s + v * (uDead[i] ?? 0), 0);
  const positiveOverlap = Math.max(0, dot);

  const purified = [...z];

  // 1. Subtract orthogonal component along u_dead
  for (let i = 0; i < 24; i++) {
    purified[i] = (purified[i] ?? 0) - alpha * positiveOverlap * (uDead[i] ?? 0);
  }

  // 2. Suppress abstract filler bloat (Sg-01 Gr) and sluggish coupling (Sg-05 Cp)
  purified[0] = Math.max(0.1, (purified[0] ?? 0) - 0.25);
  purified[4] = Math.max(0.1, (purified[4] ?? 0) - 0.15);

  // 3. Inject authentic literary dynamics:
  // - Void (Sg-03 Pr, Sg-07 Cs, Sg-11 Ma)
  purified[2] = Math.min(1.0, (purified[2] ?? 0) + 0.35);
  purified[6] = Math.min(1.0, (purified[6] ?? 0) + 0.35);
  purified[10] = Math.min(1.0, (purified[10] ?? 0) + 0.30);

  // - Dynamic range (Sg-09 Dy)
  purified[8] = Math.min(1.0, (purified[8] ?? 0) + 0.35);

  // - Conjugate Momentum p (indices 12..23): restore from near-zero flatline
  for (let i = 12; i < 24; i++) {
    purified[i] = Math.max(purified[i] ?? 0, 0.35 + (i % 3) * 0.1);
  }

  // 4. Strict bounding: q in [0, 1]^12, p >= 0
  for (let i = 0; i < 12; i++) {
    purified[i] = Math.max(0.0, Math.min(1.0, purified[i]!));
  }
  for (let i = 12; i < 24; i++) {
    purified[i] = Math.max(0.0, purified[i]!);
  }

  return purified;
}

// =============================================================================
// OPERATOR ②: RIEMANNIAN / CHOLESKY GEODESIC INTERPOLATION
// =============================================================================

export interface GeodesicInterpolationResult {
  profile: UsirTierMProfile;
  binary: Uint8Array;
  centroid24d: number[];
  predictedPhase: UsirPhase;
}

/**
 * Computes a geodesic interpolation between two Tier M style genome profiles
 * along the Cholesky manifold and precision-weighted Gaussian centroids.
 *
 * @param particleA Profile A or 4096B binary buffer (at t = 0.0)
 * @param particleB Profile B or 4096B binary buffer (at t = 1.0)
 * @param t Interpolation parameter in [0.0, 1.0]
 * @param options Optional classifier
 */
export function interpolateGeodesic(
  particleA: UsirTierMProfile | Uint8Array,
  particleB: UsirTierMProfile | Uint8Array,
  t: number,
  options?: { classifier?: PhaseCentroidsAndCov }
): GeodesicInterpolationResult {
  const profA = particleA instanceof Uint8Array ? decodeTierM(particleA) : particleA;
  const profB = particleB instanceof Uint8Array ? decodeTierM(particleB) : particleB;

  const tClamped = Math.max(0.0, Math.min(1.0, t));
  const wA = 1.0 - tClamped;
  const wB = tClamped;
  const D = 24;

  // 1. Interpolate 3x3 Transition Matrix
  const transitionMatrix: number[][] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
  ];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const valA = profA.transition_matrix[r]?.[c] ?? (r === c ? 0.9 : 0.05);
      const valB = profB.transition_matrix[r]?.[c] ?? (r === c ? 0.9 : 0.05);
      transitionMatrix[r]![c] = wA * valA + wB * valB;
    }
    const rowSum = transitionMatrix[r]!.reduce((s, v) => s + v, 0) || 1.0;
    for (let c = 0; c < 3; c++) {
      transitionMatrix[r]![c] = transitionMatrix[r]![c]! / rowSum;
    }
  }

  // 2. Interpolate Mode Weights
  const weights: number[] = [];
  let sumW = 0.0;
  for (let k = 0; k < 3; k++) {
    const pA = profA.weights[k] ?? 1.0 / 3.0;
    const pB = profB.weights[k] ?? 1.0 / 3.0;
    const wk = wA * pA + wB * pB;
    weights.push(wk);
    sumW += wk;
  }
  const normWeights = weights.map((w) => (sumW > 0 ? w / sumW : 1.0 / 3.0));

  // 3. Interpolate Cholesky Precision Factors & Centroids
  const choleskyFactors: number[][] = [];
  const centroids: number[][] = [];

  for (let k = 0; k < 3; k++) {
    const lA = profA.cholesky_factors[k] ?? new Array(300).fill(0.0);
    const lB = profB.cholesky_factors[k] ?? new Array(300).fill(0.0);

    // Convex combination on the Cholesky manifold preserving lower-triangular positive definiteness
    const lInterp: number[] = [];
    for (let idx = 0; idx < 300; idx++) {
      lInterp.push(wA * (lA[idx] ?? 0.0) + wB * (lB[idx] ?? 0.0));
    }
    choleskyFactors.push(lInterp);

    // Reconstruct precision matrices Lambda_A, Lambda_B, and Lambda_interp
    const Lambda_A = precisionMatrixFromPackedCholesky24d(lA);
    const Lambda_B = precisionMatrixFromPackedCholesky24d(lB);
    const Lambda_interp = precisionMatrixFromPackedCholesky24d(lInterp);
    const Sigma_interp = invertMatrixGaussJordan(Lambda_interp);

    const muA = profA.centroids[k] ?? new Array(D).fill(0.5);
    const muB = profB.centroids[k] ?? new Array(D).fill(0.5);

    // b = wA * Lambda_A * muA + wB * Lambda_B * muB
    const bVec = new Array<number>(D).fill(0.0);
    for (let r = 0; r < D; r++) {
      let dotA = 0.0;
      let dotB = 0.0;
      for (let c = 0; c < D; c++) {
        dotA += Lambda_A[r]![c]! * (muA[c] ?? 0.0);
        dotB += Lambda_B[r]![c]! * (muB[c] ?? 0.0);
      }
      bVec[r] = wA * dotA + wB * dotB;
    }

    // mu_interp = Sigma_interp * bVec
    const muInterp = new Array<number>(D).fill(0.0);
    for (let r = 0; r < D; r++) {
      let dot = 0.0;
      for (let c = 0; c < D; c++) {
        dot += Sigma_interp[r]![c]! * bVec[c]!;
      }
      muInterp[r] = dot;
    }
    centroids.push(muInterp);
  }

  // Overall centroid
  const overallCentroid = new Array<number>(D).fill(0.0);
  for (let d = 0; d < D; d++) {
    let sum = 0.0;
    for (let k = 0; k < 3; k++) {
      sum += normWeights[k]! * centroids[k]![d]!;
    }
    overallCentroid[d] = sum;
  }

  // Dominant phase
  const classifier = options?.classifier ?? getReferenceClassifier();
  const predictedPhase = predictPhase(overallCentroid, classifier).phase;

  const profile: UsirTierMProfile = {
    version: 1,
    flags: 0,
    transition_matrix: transitionMatrix,
    weights: normWeights,
    centroids,
    cholesky_factors: choleskyFactors,
  };

  const binary = encodeTierM(profile);

  return {
    profile,
    binary,
    centroid24d: overallCentroid,
    predictedPhase,
  };
}

// =============================================================================
// OPERATOR ③: CROSS-GENRE REGISTER TRANSPOSITION
// =============================================================================

export interface RegisterTranspositionResult {
  sourceZ: number[];
  transposedTargetZ: number[];
  preservedLatticeIndices: [1, 5, 9];
  targetPhase: UsirPhase;
  transpositionPrompt: string;
}

export interface RegisterTranspositionOptions {
  keepSkeletonGroup?: 'lattice' | 'flux';
  targetParticle: UsirTierMProfile | Uint8Array | CanonicalParticle;
  classifier?: PhaseCentroidsAndCov;
}

/**
 * Extracts the style signature from targetParticle.
 */
function extractTargetCentroidAndPhase(
  target: UsirTierMProfile | Uint8Array | CanonicalParticle,
  classifier: PhaseCentroidsAndCov
): { centroid: number[]; phase: UsirPhase } {
  const D = 24;

  if ('mu' in target && Array.isArray((target as CanonicalParticle).mu)) {
    // CanonicalParticle
    const particle = target as CanonicalParticle;
    return {
      centroid: [...particle.mu],
      phase: particle.phase,
    };
  }

  const profile: TierMProfile = target instanceof Uint8Array ? decodeTierM(target) : (target as TierMProfile);
  const weights = profile.weights ?? [1 / 3, 1 / 3, 1 / 3];
  const sumW = weights.reduce((s: number, w: number) => s + w, 0) || 1.0;
  const normW = weights.map((w: number) => w / sumW);

  const centroid = new Array<number>(D).fill(0.0);
  for (let d = 0; d < D; d++) {
    let sum = 0.0;
    for (let k = 0; k < 3; k++) {
      sum += normW[k]! * (profile.centroids[k]?.[d] ?? 0.0);
    }
    centroid[d] = sum;
  }

  const phase = predictPhase(centroid, classifier).phase;
  return { centroid, phase };
}

/**
 * Transposes text style register while preserving its syntactic skeleton and logical deductions.
 */
export function transposeStyleRegister(
  sourceText: string,
  options: RegisterTranspositionOptions
): RegisterTranspositionResult {
  const classifier = options.classifier ?? getReferenceClassifier();
  const sourceExt = dirtyExtract(sourceText);
  const sourceZ = [...sourceExt.z_24d];

  const { centroid: targetZ, phase: targetPhase } = extractTargetCentroidAndPhase(
    options.targetParticle,
    classifier
  );

  // Initialize transposed vector from target
  const transposed = [...targetZ];

  // Preserved skeleton indices: Group II Lattice (Sg-02 Cr, Sg-06 Sy, Sg-10 At) = indices [1, 5, 9]
  const preservedLatticeIndices: [1, 5, 9] = [1, 5, 9];
  for (const idx of preservedLatticeIndices) {
    transposed[idx] = sourceZ[idx]!;
  }

  // Group III Void (2, 6, 10), Group IV Shock (3, 7, 11), and momentum p (12..23) are taken from target
  const promptLines: string[] = [
    `# USIR Cross-Genre Register Transposition Prompt`,
    `## Directives:`,
    `- Target Style Phase: ${targetPhase}`,
    `- Preserved Skeleton (Group II Lattice): Sg-02 Cr=${sourceZ[1]!.toFixed(3)}, Sg-06 Sy=${sourceZ[5]!.toFixed(3)}, Sg-10 At=${sourceZ[9]!.toFixed(3)}.`,
    `  Strictly retain all logical conditions, deductive conjunctions, and bilateral clause symmetry.`,
    `- Modulated Style (Group III Void): Sg-03 Pr=${transposed[2]!.toFixed(3)}, Sg-07 Cs=${transposed[6]!.toFixed(3)}, Sg-11 Ma=${transposed[10]!.toFixed(3)}.`,
    `- Modulated Style (Group IV Shock): Sg-04 Jt=${transposed[3]!.toFixed(3)}, Sg-08 Lp=${transposed[7]!.toFixed(3)}, Sg-12 Ru=${transposed[11]!.toFixed(3)}.`,
    `- Temporal Volatility (Conjugate Momentum p): Modulate sentence length dynamic range to target momentum.`,
    ``,
    `## Source Text:`,
    sourceText,
  ];

  return {
    sourceZ,
    transposedTargetZ: transposed,
    preservedLatticeIndices,
    targetPhase,
    transpositionPrompt: promptLines.join('\n'),
  };
}

// =============================================================================
// OPERATOR ④: UNEXPLORED STYLE DISCOVERY (USIR-GENESIS / VORONOI VOID)
// & CANONICAL STYLE CHEMISTRY MOLECULES
// =============================================================================

export {
  evaluateStyleMolecules,
  discoverVoronoiGenesis,
  usirGenesis,
  CHI2_24_095,
  type StyleMoleculesResult,
  type CanonicalStyleMoleculeName,
  type VoronoiGenesisResult,
  type VoronoiGenesisOptions,
} from '../v2/styleChemistry.js';

