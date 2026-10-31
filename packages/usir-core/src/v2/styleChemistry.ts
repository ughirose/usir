/**
 * Universal Style Genome (USIR v2.0 / Nishiki)
 * Gap 5: Style Chemistry - The Four Canonical Style Molecules & Unexplored Style Discovery Operator (usir-genesis).
 *
 * Implements:
 * 1. The Four Canonical Style Molecules (evaluateStyleMolecules):
 *    - Mol_Yugen: emerges from Sg-03 Pr (idx 2) x Sg-08 Lp (idx 7) x Sg-11 Ma (idx 10),
 *      amplified by so(8) metric coupling between (2, 7, 10).
 *    - Mol_Wabi: emerges from Sg-04 Jt (idx 3) x Sg-07 Cs (idx 6) x Sg-11 Ma (idx 10),
 *      amplified by so(8) metric coupling between (3, 6, 10).
 *    - Mol_QiYun: emerges from Sg-01 Dn/Gr (idx 0) x Sg-06 Mp/Sy (idx 5) x Sg-10 At (idx 9),
 *      gated by active symplectic momentum circulation tanh(||p||_{G_style}) > 0.
 *    - Mol_Iki: emerges from Sg-02 Cr (idx 1) x Sg-07 Cs (idx 6) x Sg-12 Ru (idx 11),
 *      amplified by so(8) metric coupling between (1, 6, 11).
 *
 * 2. The 4th Style Algebra Operator: usirGenesis / discoverVoronoiGenesis:
 *    Discovers unexplored style coordinates in the Voronoi void of reference style particles
 *    inside the 95% Mahalanobis trust region ellipsoid d_M^2 <= chi^2_{24, 0.95} = 36.4150.
 */

import { SO8PullbackMetric } from './thermodynamicsSo8.js';
import { invertMatrixGaussJordan } from '../dynamics/landauPhase.js';
import { decodeTierM, type TierMProfile } from '../schema/usirBinarySchema.js';
import type { CanonicalParticle } from '../presets/canonicalParticles.js';
import type { UsirTierMProfile } from '../algebra/genomeCompiler.js';

export type CanonicalStyleMoleculeName = 'Mol_Yugen' | 'Mol_Wabi' | 'Mol_QiYun' | 'Mol_Iki';

export interface StyleMoleculesResult {
  Mol_Yugen: number;
  Mol_Wabi: number;
  Mol_QiYun: number;
  Mol_Iki: number;
  dominantMolecule: CanonicalStyleMoleculeName;
  synestheticCouplingEnergy: number;
}

export interface VoronoiGenesisOptions {
  lambdaShrinkage?: number;
  chi2Threshold?: number;
  nRestarts?: number;
  nSteps?: number;
  learningRate?: number;
  softminBeta?: number;
}

export interface VoronoiGenesisResult {
  zGenesis: number[];
  mahalanobisD2: number;
  minRiemannianDistance: number;
  insideTrustRegion: boolean;
  molecules: StyleMoleculesResult;
  initialCentroidMinDist: number;
}

export const CHI2_24_095 = 36.415028;

export const MOLECULE_TRIPLETS: Record<CanonicalStyleMoleculeName, readonly [number, number, number]> = {
  Mol_Yugen: [2, 7, 10], // Sg-03 Pr, Sg-08 Lp, Sg-11 Ma
  Mol_Wabi: [3, 6, 10],  // Sg-04 Jt, Sg-07 Cs, Sg-11 Ma
  Mol_QiYun: [0, 5, 9],  // Sg-01 Gr, Sg-06 Sy, Sg-10 At
  Mol_Iki: [1, 6, 11],   // Sg-02 Cr, Sg-07 Cs, Sg-12 Ru
};

/**
 * Computes normalized off-diagonal Riemannian metric coupling between indices (i, j, k).
 */
function computeTripletCoupling(g: number[][], i: number, j: number, k: number): number {
  const g_ii = g[i]?.[i] ?? 1.0;
  const g_jj = g[j]?.[j] ?? 1.0;
  const g_kk = g[k]?.[k] ?? 1.0;

  const denom_ij = Math.sqrt(Math.max(1e-12, g_ii * g_jj));
  const c_ij = Math.abs(g[i]?.[j] ?? 0.0) / denom_ij;

  const denom_jk = Math.sqrt(Math.max(1e-12, g_jj * g_kk));
  const c_jk = Math.abs(g[j]?.[k] ?? 0.0) / denom_jk;

  const denom_ki = Math.sqrt(Math.max(1e-12, g_kk * g_ii));
  const c_ki = Math.abs(g[k]?.[i] ?? 0.0) / denom_ki;

  return (c_ij + c_jk + c_ki) / 3.0;
}

/**
 * Evaluates the activation scores in [0, 1] of the 4 canonical aesthetic molecules.
 *
 * @param z 24D symplectic state vector (q, p) in R^24, or 12D configuration vector q in [0, 1]^12.
 * @param gStyle Optional 12x12 Riemannian metric G_style. Defaults to new SO8PullbackMetric().metricMatrix().
 */
export function evaluateStyleMolecules(z: number[], gStyle?: number[][]): StyleMoleculesResult {
  if (z.length !== 12 && z.length !== 24) {
    throw new Error(`evaluateStyleMolecules expects 12D or 24D vector, got length ${z.length}`);
  }

  const q = new Array<number>(12);
  const p = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    q[i] = Math.max(0.0, Math.min(1.0, z[i] ?? 0.0));
    p[i] = z.length === 24 ? (z[12 + i] ?? 0.0) : 0.0;
  }

  const g = gStyle ?? new SO8PullbackMetric().metricMatrix();
  if (g.length !== 12 || g[0]?.length !== 12) {
    throw new Error(`gStyle must have shape 12x12, got ${g.length}x${g[0]?.length}`);
  }

  // 1. Synesthetic coupling energy: E_coupling = sum_{i < j} |G_ij| * q_i * q_j
  let couplingEnergy = 0.0;
  for (let i = 0; i < 12; i++) {
    const qi = q[i]!;
    for (let j = i + 1; j < 12; j++) {
      couplingEnergy += Math.abs(g[i]![j]!) * qi * q[j]!;
    }
  }

  // 2. Mol_Yugen (幽玄): (2, 7, 10)
  const [iy, jy, ky] = MOLECULE_TRIPLETS.Mol_Yugen;
  const baseYugen = Math.cbrt(q[iy]! * q[jy]! * q[ky]!);
  const coupYugen = computeTripletCoupling(g, iy, jy, ky);
  const scoreYugen = Math.min(1.0, Math.max(0.0, baseYugen * (1.0 + coupYugen)));

  // 3. Mol_Wabi (侘び・数寄): (3, 6, 10)
  const [iw, jw, kw] = MOLECULE_TRIPLETS.Mol_Wabi;
  const baseWabi = Math.cbrt(q[iw]! * q[jw]! * q[kw]!);
  const coupWabi = computeTripletCoupling(g, iw, jw, kw);
  const scoreWabi = Math.min(1.0, Math.max(0.0, baseWabi * (1.0 + coupWabi)));

  // 4. Mol_QiYun (気韻生動): (0, 5, 9) gated by tanh(||p||_{G_style})
  const [iq, jq, kq] = MOLECULE_TRIPLETS.Mol_QiYun;
  const baseQiYun = Math.cbrt(q[iq]! * q[jq]! * q[kq]!);
  const coupQiYun = computeTripletCoupling(g, iq, jq, kq);

  // ||p||_{G_style} = sqrt(p^T G p)
  let pNormG2 = 0.0;
  for (let r = 0; r < 12; r++) {
    const pr = p[r]!;
    let rowSum = 0.0;
    for (let c = 0; c < 12; c++) {
      rowSum += g[r]![c]! * p[c]!;
    }
    pNormG2 += pr * rowSum;
  }
  const pNormG = Math.sqrt(Math.max(0.0, pNormG2));
  const momentumGate = Math.tanh(pNormG);
  const scoreQiYun = Math.min(1.0, Math.max(0.0, baseQiYun * (1.0 + coupQiYun) * momentumGate));

  // 5. Mol_Iki (粋・いなせ): (1, 6, 11)
  const [i_ik, j_ik, k_ik] = MOLECULE_TRIPLETS.Mol_Iki;
  const baseIki = Math.cbrt(q[i_ik]! * q[j_ik]! * q[k_ik]!);
  const coupIki = computeTripletCoupling(g, i_ik, j_ik, k_ik);
  const scoreIki = Math.min(1.0, Math.max(0.0, baseIki * (1.0 + coupIki)));

  const scores: Record<CanonicalStyleMoleculeName, number> = {
    Mol_Yugen: scoreYugen,
    Mol_Wabi: scoreWabi,
    Mol_QiYun: scoreQiYun,
    Mol_Iki: scoreIki,
  };

  let dominant: CanonicalStyleMoleculeName = 'Mol_Yugen';
  let maxScore = -1.0;
  for (const [name, val] of Object.entries(scores) as [CanonicalStyleMoleculeName, number][]) {
    if (val > maxScore) {
      maxScore = val;
      dominant = name;
    }
  }

  return {
    Mol_Yugen: scoreYugen,
    Mol_Wabi: scoreWabi,
    Mol_QiYun: scoreQiYun,
    Mol_Iki: scoreIki,
    dominantMolecule: dominant,
    synestheticCouplingEnergy: couplingEnergy,
  };
}

/**
 * Extracts a 24D vector from heterogeneous particle representations.
 */
function extract24dVector(
  particle: number[] | UsirTierMProfile | Uint8Array | CanonicalParticle
): number[] {
  if (Array.isArray(particle)) {
    if (particle.length === 12) {
      return [...particle, ...new Array(12).fill(0.0)];
    }
    if (particle.length === 24) {
      return [...particle];
    }
    throw new Error(`Array particle must be 12D or 24D, got ${particle.length}`);
  }

  if ('mu' in particle && Array.isArray((particle as CanonicalParticle).mu)) {
    const mu = (particle as CanonicalParticle).mu;
    return mu.length === 24 ? [...mu] : [...mu, ...new Array(12).fill(0.0)];
  }

  const profile: TierMProfile =
    particle instanceof Uint8Array ? decodeTierM(particle) : (particle as TierMProfile);
  const D = 24;
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
  return centroid;
}

/**
 * Builds 24D Riemannian metric G_24 = block_diag(G_style, I_12) and G_24^{-1}.
 */
function buildG24(gStyle: number[][]): { g24: number[][]; g24Inv: number[][] } {
  const gInv12 = invertMatrixGaussJordan(gStyle);
  const g24: number[][] = [];
  const g24Inv: number[][] = [];

  for (let r = 0; r < 24; r++) {
    const rowG = new Array<number>(24).fill(0.0);
    const rowGInv = new Array<number>(24).fill(0.0);
    if (r < 12) {
      for (let c = 0; c < 12; c++) {
        rowG[c] = gStyle[r]![c]!;
        rowGInv[c] = gInv12[r]![c]!;
      }
    } else {
      rowG[r] = 1.0;
      rowGInv[r] = 1.0;
    }
    g24.push(rowG);
    g24Inv.push(rowGInv);
  }

  return { g24, g24Inv };
}

/**
 * Computes Riemannian quadratic distance ||z1 - z2||_{G_24} = sqrt((z1 - z2)^T G_24 (z1 - z2)).
 */
function computeRiemannianDistance(diff: number[], g24: number[][]): number {
  let sum = 0.0;
  for (let r = 0; r < 24; r++) {
    const dr = diff[r]!;
    if (dr === 0.0) continue;
    let rowDot = 0.0;
    const row = g24[r]!;
    for (let c = 0; c < 24; c++) {
      rowDot += row[c]! * diff[c]!;
    }
    sum += dr * rowDot;
  }
  return Math.sqrt(Math.max(0.0, sum));
}

/**
 * Computes matrix-vector product M * v.
 */
function matVecMul(M: number[][], v: number[]): number[] {
  const n = M.length;
  const res = new Array<number>(n).fill(0.0);
  for (let r = 0; r < n; r++) {
    let dot = 0.0;
    const row = M[r]!;
    for (let c = 0; c < n; c++) {
      dot += row[c]! * (v[c] ?? 0.0);
    }
    res[r] = dot;
  }
  return res;
}

/**
 * Computes quadratic form v^T M v.
 */
function quadForm(v: number[], M: number[][]): number {
  let sum = 0.0;
  const n = v.length;
  for (let r = 0; r < n; r++) {
    const vr = v[r]!;
    let rowDot = 0.0;
    const row = M[r]!;
    for (let c = 0; c < n; c++) {
      rowDot += row[c]! * (v[c] ?? 0.0);
    }
    sum += vr * rowDot;
  }
  return sum;
}

/**
 * OPERATOR ④: UNEXPLORED STYLE DISCOVERY (USIR-GENESIS / VORONOI VOID)
 *
 * Discovers the Voronoi void style coordinate z_genesis^* maximizing Riemannian distance
 * to known reference particles inside the 95% Mahalanobis trust region ellipsoid:
 *   z_genesis^* = argmax_{z in E_{0.95}} min_k d_{G_24}(z, z_k)
 *
 * @param knownParticles Collection of K >= 2 reference style particles
 * @param gStyle Optional 12x12 Riemannian metric
 * @param options Optional tuning parameters (shrinkage, chi2Threshold, restarts, etc.)
 */
export function discoverVoronoiGenesis(
  knownParticles: (number[] | UsirTierMProfile | Uint8Array | CanonicalParticle)[],
  gStyle?: number[][],
  options?: VoronoiGenesisOptions
): VoronoiGenesisResult {
  if (knownParticles.length < 2) {
    throw new Error(`discoverVoronoiGenesis requires at least 2 reference particles, got ${knownParticles.length}`);
  }

  const lambdaShrinkage = options?.lambdaShrinkage ?? 0.15;
  const chi2Threshold = options?.chi2Threshold ?? CHI2_24_095;
  const nRestarts = options?.nRestarts ?? 8;
  const nSteps = options?.nSteps ?? 45;
  const learningRate = options?.learningRate ?? 0.08;
  const softminBeta = options?.softminBeta ?? 8.0;

  const particles: number[][] = knownParticles.map(extract24dVector);
  const K = particles.length;

  const g12 = gStyle ?? new SO8PullbackMetric().metricMatrix();
  const { g24, g24Inv } = buildG24(g12);

  // 1. Centroid z_bar (24D)
  const zBar = new Array<number>(24).fill(0.0);
  for (let d = 0; d < 24; d++) {
    let sum = 0.0;
    for (let k = 0; k < K; k++) {
      sum += particles[k]![d]!;
    }
    zBar[d] = sum / K;
  }

  // 2. Regularized Covariance Matrix \hat{\Sigma}_{24}
  const sigmaEmp: number[][] = [];
  for (let r = 0; r < 24; r++) {
    sigmaEmp.push(new Array<number>(24).fill(0.0));
  }
  for (let k = 0; k < K; k++) {
    const diff = particles[k]!.map((v, i) => v - zBar[i]!);
    for (let r = 0; r < 24; r++) {
      for (let c = 0; c < 24; c++) {
        sigmaEmp[r]![c] = (sigmaEmp[r]![c] ?? 0.0) + (diff[r]! * diff[c]!) / (K - 1.0);
      }
    }
  }

  // Shrinkage: \hat{\Sigma}_{24} = (1 - \lambda)\Sigma_{emp} + \lambda G_{24}^{-1}
  const sigmaHat: number[][] = [];
  for (let r = 0; r < 24; r++) {
    const row = new Array<number>(24).fill(0.0);
    for (let c = 0; c < 24; c++) {
      const sVal = (1.0 - lambdaShrinkage) * sigmaEmp[r]![c]! + lambdaShrinkage * g24Inv[r]![c]!;
      // Enforce strict symmetry and diagonal stabilization
      row[c] = sVal;
    }
    row[r] = (row[r] ?? 0.0) + 1e-6;
    sigmaHat.push(row);
  }

  const sigmaInv = invertMatrixGaussJordan(sigmaHat);

  // Initial centroid distance
  let d0 = Number.POSITIVE_INFINITY;
  for (let k = 0; k < K; k++) {
    const diff = zBar.map((v, i) => v - particles[k]![i]!);
    const d = computeRiemannianDistance(diff, g24);
    if (d < d0) d0 = d;
  }

  const projectOntoFeasibleSet = (zCand: number[]): number[] => {
    const zProj = [...zCand];
    for (let iter = 0; iter < 5; iter++) {
      // Box bounds: q in [0, 1], p in [0, 2.0]
      for (let i = 0; i < 12; i++) {
        zProj[i] = Math.max(0.0, Math.min(1.0, zProj[i]!));
      }
      for (let i = 12; i < 24; i++) {
        zProj[i] = Math.max(0.0, Math.min(2.0, zProj[i]!));
      }

      // Ellipsoid constraint
      const diff = zProj.map((v, i) => v - zBar[i]!);
      const d2 = quadForm(diff, sigmaInv);
      if (d2 > chi2Threshold) {
        const scale = Math.sqrt((chi2Threshold - 1e-4) / d2);
        for (let i = 0; i < 24; i++) {
          zProj[i] = zBar[i]! + diff[i]! * scale;
        }
      } else {
        break;
      }
    }
    return zProj;
  };

  // 3. Multi-Start Projected Gradient Ascent
  const seeds: number[][] = [[...zBar]];
  for (let k = 0; k < Math.min(K, nRestarts - 1); k++) {
    // Point pushed opposite to particle k
    const antiSeed = zBar.map((v, i) => v + 0.6 * (v - particles[k]![i]!));
    seeds.push(antiSeed);
  }

  // Deterministic quasi-random spread for remaining restarts
  let seedCounter = 1;
  while (seeds.length < nRestarts) {
    const randSeed = zBar.map((v, i) => {
      const pseudoNorm = Math.sin(seedCounter * 12.9898 + i * 78.233) * 0.25;
      return v + pseudoNorm;
    });
    seeds.push(randSeed);
    seedCounter++;
  }

  let bestZ = [...zBar];
  let bestMinDist = -1.0;

  for (let s = 0; s < Math.min(seeds.length, nRestarts); s++) {
    let zCurr = projectOntoFeasibleSet(seeds[s]!);

    for (let step = 0; step < nSteps; step++) {
      // Compute distances to all particles
      const dists = new Array<number>(K);
      const diffs: number[][] = [];
      let minD = Number.POSITIVE_INFINITY;

      for (let k = 0; k < K; k++) {
        const diff = zCurr.map((v, i) => v - particles[k]![i]!);
        diffs.push(diff);
        const d = Math.max(1e-12, computeRiemannianDistance(diff, g24));
        dists[k] = d;
        if (d < minD) minD = d;
      }

      // Softmin weights
      let weightSum = 0.0;
      const weights = new Array<number>(K);
      for (let k = 0; k < K; k++) {
        const w = Math.exp(-softminBeta * (dists[k]! - minD));
        weights[k] = w;
        weightSum += w;
      }
      for (let k = 0; k < K; k++) {
        weights[k] = weightSum > 0 ? weights[k]! / weightSum : 1.0 / K;
      }

      // Ascent gradient = sum_k w_k * G_24 * (z - z_k) / d_k
      const grad = new Array<number>(24).fill(0.0);
      for (let k = 0; k < K; k++) {
        const gDiff = matVecMul(g24, diffs[k]!);
        const factor = weights[k]! / dists[k]!;
        for (let i = 0; i < 24; i++) {
          grad[i] = (grad[i] ?? 0.0) + factor * gDiff[i]!;
        }
      }

      let gradNorm = Math.sqrt(grad.reduce((s, v) => s + v * v, 0));
      const stepVec = gradNorm > 1e-6
        ? grad.map((g) => (g / gradNorm) * learningRate)
        : grad.map((g) => g * learningRate);

      const zNext = zCurr.map((v, i) => v + stepVec[i]!);
      zCurr = projectOntoFeasibleSet(zNext);
    }

    // Evaluate min distance
    let curMinDist = Number.POSITIVE_INFINITY;
    for (let k = 0; k < K; k++) {
      const diff = zCurr.map((v, i) => v - particles[k]![i]!);
      const d = computeRiemannianDistance(diff, g24);
      if (d < curMinDist) curMinDist = d;
    }

    if (curMinDist > bestMinDist) {
      bestMinDist = curMinDist;
      bestZ = [...zCurr];
    }
  }

  bestZ = projectOntoFeasibleSet(bestZ);
  const finalDiff = bestZ.map((v, i) => v - zBar[i]!);
  const finalD2 = quadForm(finalDiff, sigmaInv);

  let finalMinDist = Number.POSITIVE_INFINITY;
  for (let k = 0; k < K; k++) {
    const diff = bestZ.map((v, i) => v - particles[k]![i]!);
    const d = computeRiemannianDistance(diff, g24);
    if (d < finalMinDist) finalMinDist = d;
  }

  const molecules = evaluateStyleMolecules(bestZ, g12);

  return {
    zGenesis: bestZ,
    mahalanobisD2: finalD2,
    minRiemannianDistance: finalMinDist,
    insideTrustRegion: finalD2 <= chi2Threshold + 1e-4,
    molecules,
    initialCentroidMinDist: d0,
  };
}

/**
 * The 4th Style Algebra Operator: usirGenesis (Alias for discoverVoronoiGenesis).
 */
export const usirGenesis = discoverVoronoiGenesis;
