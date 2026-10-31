/**
 * Hamiltonian Dynamics and Symplectic Energy Calculations for USIR.
 */

export interface HamiltonianParams {
  massDiag?: number[]; // Diagonal mass matrix M in R^12 (default 1.0)
  targetQ?: number[]; // Equilibrium target position q* in R^12
  potentialStiffnessDiag?: number[]; // Diagonal spring constant K in R^12 (default 1.0)
}

export interface HamiltonianResult {
  kineticEnergy: number; // T(p) = 1/2 p^T M^-1 p
  potentialEnergy: number; // V(q) = 1/2 (q - q*)^T K (q - q*)
  totalHamiltonian: number; // H(q, p) = T(p) + V(q)
  symplecticGradient: {
    dq_dt: number[]; // dq/dt = dH/dp = M^-1 p
    dp_dt: number[]; // dp/dt = -dH/dq = -K (q - q*)
  };
}

export function computeHamiltonian(
  q: number[],
  p: number[],
  params?: HamiltonianParams
): HamiltonianResult {
  const M_diag = params?.massDiag ?? new Array(12).fill(1.0);
  const q_star = params?.targetQ ?? new Array(12).fill(0.5);
  const K_diag = params?.potentialStiffnessDiag ?? new Array(12).fill(1.0);

  let T = 0.0;
  const dq_dt = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    const invM = 1.0 / (M_diag[i] ?? 1.0);
    const pi = p[i] ?? 0.0;
    T += 0.5 * pi * invM * pi;
    dq_dt[i] = invM * pi;
  }

  let V = 0.0;
  const dp_dt = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    const ki = K_diag[i] ?? 1.0;
    const diff = (q[i] ?? 0.0) - (q_star[i] ?? 0.5);
    V += 0.5 * ki * diff * diff;
    dp_dt[i] = -ki * diff;
  }

  return {
    kineticEnergy: T,
    potentialEnergy: V,
    totalHamiltonian: T + V,
    symplecticGradient: {
      dq_dt,
      dp_dt,
    },
  };
}

/**
 * Computes Cholesky Mahalanobis energy 1/2 || L^T (z - mu) ||_2^2
 * where L is the lower triangular factor of precision matrix Lambda = L L^T.
 */
export function computeCholeskyEnergy(
  z: number[],
  mu: number[],
  L_flat: number[],
  dim = 24
): number {
  const diff = new Array<number>(dim);
  for (let i = 0; i < dim; i++) {
    diff[i] = (z[i] ?? 0.0) - (mu[i] ?? 0.0);
  }

  // Multiply L^T * diff
  // L is dim x dim lower triangular stored in row-major compact (i*(i+1)/2 + j)
  const y = new Array<number>(dim).fill(0.0);
  let idx = 0;
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j <= i; j++) {
      const Lij = L_flat[idx++] ?? 0.0;
      // y[j] += Lij * diff[i]
      y[j] = (y[j] ?? 0.0) + Lij * diff[i]!;
    }
  }

  let energy = 0.0;
  for (let j = 0; j < dim; j++) {
    energy += 0.5 * y[j]! * y[j]!;
  }
  return energy;
}
