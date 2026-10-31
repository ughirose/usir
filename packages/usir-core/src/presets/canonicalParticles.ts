/**
 * Canonical Style Particles and Product-of-Experts (PoE) Chemical Mixer for USIR (TASK-361).
 */

import { invertMatrixGaussJordan, type UsirPhase } from '../dynamics/landauPhase.js';

export interface CanonicalParticle {
  name: string;
  phase: UsirPhase;
  description: string;
  mu: number[]; // 24D centroid vector
  precision: number[][]; // 24x24 precision matrix Lambda
  cholesky_flat: number[]; // 300 lower-triangular elements
}

// Canonical Centroids derived from the 40-chunk empirical golden benchmark
export const PORCELAIN_CRYSTAL_MU: number[] = [
  0.6416, 0.7718, 0.0543, 0.1804, 0.6558, 0.7512, 0.0435, 0.2815, 0.8123, 0.8752, 0.0211, 0.2114,
  0.4215, 0.3541, 0.0812, 0.2145, 0.4512, 0.3812, 0.3125, 0.2514, 0.3142, 0.4815, 0.1124, 0.2815
];

export const AMBER_CRITICAL_MU: number[] = [
  0.7214, 0.4815, 0.4125, 0.4215, 0.5412, 0.4915, 0.3214, 0.4512, 0.7512, 0.5124, 0.4812, 0.4125,
  0.6812, 0.4215, 0.8512, 0.6214, 0.7125, 0.4512, 0.6124, 0.6812, 0.7215, 0.6512, 0.3214, 0.6812
];

export const OBSIDIAN_VOID_MU: number[] = [
  0.6812, 0.5124, 0.5812, 0.2815, 0.4912, 0.5214, 0.5412, 0.2115, 0.6125, 0.4812, 0.6812, 0.1512,
  0.5124, 0.3125, 0.9124, 0.3812, 0.5512, 0.3124, 0.5812, 0.4125, 0.6214, 0.6125, 0.4125, 0.4512
];

export const LIGHTNING_RUPTURE_MU: number[] = [
  0.7512, 0.2114, 0.1215, 0.8512, 0.6125, 0.1812, 0.1124, 0.8812, 0.8812, 0.3124, 0.0812, 0.8912,
  0.9214, 0.6512, 0.4125, 0.9512, 0.8124, 0.5812, 0.8812, 0.9124, 0.8512, 0.4125, 0.0912, 0.9412
];

export const DEADZONE_MU: number[] = [
  0.8159, 0.5512, 0.2221, 0.1829, 0.6235, 0.6302, 0.143, 0.2235, 0.2692, 0.5503, 0.2796, 0.1135,
  0.0428, 0.0169, 0.051, 0.0442, 0.0525, 0.0125, 0.041, 0.0513, 0.0283, 0.0076, 0.0068, 0.0367
];

export const CANONICAL_ORDER_PARAMS: Record<
  UsirPhase,
  { psi_order: number; phi_void: number; kappa_kin: number }
> = {
  I_Crystalline: { psi_order: 0.812, phi_void: 0.0, kappa_kin: 0.612 },
  II_Critical: { psi_order: 0.01, phi_void: 0.355, kappa_kin: 2.094 },
  III_Void: { psi_order: 0.496, phi_void: 0.677, kappa_kin: 1.369 },
  IV_Turbulent: { psi_order: -0.999, phi_void: 0.0, kappa_kin: 2.802 },
  V_Deadzone: { psi_order: 0.661, phi_void: 0.043, kappa_kin: 0.04 },
};

function createIdentityPrecision(diagScale = 1.0): number[][] {
  const Lambda: number[][] = new Array(24).fill(0).map((_, r) => {
    const row = new Array(24).fill(0.0);
    row[r] = diagScale;
    return row;
  });
  return Lambda;
}

function createCholeskyFlatFromPrecision(Lambda: number[][]): number[] {
  // For diagonal Lambda with elements d_i, L_ii = sqrt(d_i), off-diagonals = 0
  const flat: number[] = [];
  for (let i = 0; i < 24; i++) {
    for (let j = 0; j <= i; j++) {
      if (i === j) {
        flat.push(Math.sqrt(Math.max(Lambda[i]![i]!, 1e-4)));
      } else {
        flat.push(0.0);
      }
    }
  }
  return flat;
}

export const CANONICAL_PARTICLES: Record<string, CanonicalParticle> = {
  PORCELAIN_CRYSTAL: {
    name: 'PORCELAIN_CRYSTAL',
    phase: 'I_Crystalline',
    description: 'Phase I Crystalline Order (High formal structure, axiomatic symmetry, ambiguity-free)',
    mu: PORCELAIN_CRYSTAL_MU,
    precision: createIdentityPrecision(1.2),
    cholesky_flat: createCholeskyFlatFromPrecision(createIdentityPrecision(1.2)),
  },
  AMBER_CRITICAL: {
    name: 'AMBER_CRITICAL',
    phase: 'II_Critical',
    description: 'Phase II Self-Organized Criticality (1/f rhythm volatility, dynamic leap, literary balance)',
    mu: AMBER_CRITICAL_MU,
    precision: createIdentityPrecision(1.0),
    cholesky_flat: createCholeskyFlatFromPrecision(createIdentityPrecision(1.0)),
  },
  OBSIDIAN_VOID: {
    name: 'OBSIDIAN_VOID',
    phase: 'III_Void',
    description: 'Phase III Superfluid Void (Wago brevity, stillness operators, clean ending cut)',
    mu: OBSIDIAN_VOID_MU,
    precision: createIdentityPrecision(1.1),
    cholesky_flat: createCholeskyFlatFromPrecision(createIdentityPrecision(1.1)),
  },
  LIGHTNING_RUPTURE: {
    name: 'LIGHTNING_RUPTURE',
    phase: 'IV_Turbulent',
    description: 'High-Momentum Shock Catalyst (Purges deadzone sluggishness with high temporal impulse)',
    mu: LIGHTNING_RUPTURE_MU,
    precision: createIdentityPrecision(1.5),
    cholesky_flat: createCholeskyFlatFromPrecision(createIdentityPrecision(1.5)),
  },
};

/**
 * Mixes multiple canonical particles using Product-of-Experts (PoE) Gaussian precision weighting:
 * Lambda_mix = sum_k w_k Lambda_k
 * mu_mix = Lambda_mix^-1 (sum_k w_k Lambda_k mu_k)
 */
export function mixStyleParticles(
  particles: CanonicalParticle[],
  weights: number[]
): { mu_mix: number[]; precision_mix: number[][] } {
  if (particles.length === 0) {
    throw new Error('mixStyleParticles requires at least one particle.');
  }
  if (particles.length !== weights.length) {
    throw new Error('Number of particles must match number of weights.');
  }

  const D = 24;
  const sumWeights = weights.reduce((acc, w) => acc + w, 0);
  const normWeights = weights.map((w) => (sumWeights > 0 ? w / sumWeights : 1.0 / weights.length));

  // Initialize Lambda_mix = 0
  const Lambda_mix: number[][] = new Array(D).fill(0).map(() => new Array(D).fill(0.0));
  const weighted_info: number[] = new Array(D).fill(0.0);

  for (let k = 0; k < particles.length; k++) {
    const wk = normWeights[k]!;
    const p = particles[k]!;
    for (let r = 0; r < D; r++) {
      let precisionDotMu = 0.0;
      const rowMix = Lambda_mix[r]!;
      for (let c = 0; c < D; c++) {
        const val = p.precision[r]![c]!;
        rowMix[c] = (rowMix[c] ?? 0.0) + wk * val;
        precisionDotMu += val * (p.mu[c] ?? 0.0);
      }
      weighted_info[r] = (weighted_info[r] ?? 0.0) + wk * precisionDotMu;
    }
  }

  // Invert Lambda_mix
  const Sigma_mix = invertMatrixGaussJordan(Lambda_mix);

  // mu_mix = Sigma_mix * weighted_info
  const mu_mix: number[] = new Array(D).fill(0.0);
  for (let r = 0; r < D; r++) {
    let dot = 0.0;
    for (let c = 0; c < D; c++) {
      dot += Sigma_mix[r]![c]! * weighted_info[c]!;
    }
    mu_mix[r] = dot;
  }

  return { mu_mix, precision_mix: Lambda_mix };
}
