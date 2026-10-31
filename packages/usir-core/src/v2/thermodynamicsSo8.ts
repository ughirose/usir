/**
 * Universal Style Genome (USIR v2.0)
 * Module 3: SO(8) Pullback Metric, Kac-Moreau-Yosida Thermodynamics & Hamilton-Friston Resonance
 */

export interface MetricParameters {
  theta28?: number[];
  lambda8?: number[];
  epsilon?: number;
}

/**
 * SO(8) Pullback Riemannian Metric G_style = P^T R^T diag(Lambda_+) R P + epsilon I_12.
 * Constructs an exact 12x12 positive-definite Riemannian metric on style state space R^12.
 */
export class SO8PullbackMetric {
  public readonly theta28: number[];
  public readonly lambda8: number[];
  public readonly epsilon: number;
  private readonly G: number[][]; // 12x12
  private readonly G_inv: number[][]; // 12x12
  private readonly R: number[][]; // 8x8 Cayley orthogonal rotation

  constructor(params?: MetricParameters) {
    this.theta28 = params?.theta28 ? [...params.theta28] : SO8PullbackMetric.defaultTheta28();
    if (this.theta28.length !== 28) {
      throw new Error(`SO8PullbackMetric requires 28 parameters for so(8), got ${this.theta28.length}`);
    }
    this.lambda8 = params?.lambda8 ? [...params.lambda8] : [1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6];
    if (this.lambda8.length !== 8) {
      throw new Error(`SO8PullbackMetric requires 8 positive eigenvalues, got ${this.lambda8.length}`);
    }
    this.epsilon = params?.epsilon ?? 1e-4;

    // 1. Build skew-symmetric generator A in so(8)
    const A = SO8PullbackMetric.buildSkewSymmetric(this.theta28);

    // 2. Cayley transform R = (I_8 - A)(I_8 + A)^(-1) in SO(8)
    this.R = SO8PullbackMetric.computeCayleyTransform(A);

    // 3. Construct Haar 2nd-difference projection P_(8x12) = U_(2x3) (x) I_4
    const P = SO8PullbackMetric.buildHaarEmbedding();

    // 4. M_8 = R^T diag(Lambda_+) R (8x8)
    const M8 = SO8PullbackMetric.buildInnerMetric(this.R, this.lambda8);

    // 5. G_style = P^T M_8 P + epsilon I_12 (12x12)
    this.G = SO8PullbackMetric.buildPullbackMetric(P, M8, this.epsilon);

    // 6. Precompute G^(-1) via Gauss-Jordan elimination
    this.G_inv = SO8PullbackMetric.invert12x12(this.G);
  }

  public static defaultTheta28(): number[] {
    // 28 parameters for Lie algebra so(8)
    return [
      0.08, -0.05, 0.12, -0.07, 0.04, -0.06, 0.09,
      0.11, -0.04, 0.05, -0.08, 0.06, -0.03,
      -0.09, 0.07, -0.05, 0.04, -0.06,
      0.05, -0.07, 0.06, -0.04,
      -0.08, 0.05, -0.06,
      0.07, -0.05,
      0.06,
    ];
  }

  public metricMatrix(): number[][] {
    return this.G.map((row) => [...row]);
  }

  public inverseMetricMatrix(): number[][] {
    return this.G_inv.map((row) => [...row]);
  }

  public rotationMatrix(): number[][] {
    return this.R.map((row) => [...row]);
  }

  public quadraticForm(v: number[]): number {
    if (v.length !== 12) {
      throw new Error(`quadraticForm expected 12D vector, got length ${v.length}`);
    }
    let sum = 0;
    for (let i = 0; i < 12; i++) {
      const vi = v[i] ?? 0;
      const row = this.G[i];
      if (!row) continue;
      let rowSum = 0;
      for (let j = 0; j < 12; j++) {
        rowSum += (row[j] ?? 0) * (v[j] ?? 0);
      }
      sum += vi * rowSum;
    }
    return sum;
  }

  public mahalanobisDistance(q1: number[], q2: number[]): number {
    if (q1.length !== 12 || q2.length !== 12) {
      throw new Error(`mahalanobisDistance expected 12D vectors, got ${q1.length} and ${q2.length}`);
    }
    const diff = new Array<number>(12);
    for (let i = 0; i < 12; i++) {
      diff[i] = (q1[i] ?? 0) - (q2[i] ?? 0);
    }
    const quad = this.quadraticForm(diff);
    return Math.sqrt(Math.max(0, quad));
  }

  public transformVector(v: number[]): number[] {
    const out = new Array<number>(12).fill(0);
    for (let i = 0; i < 12; i++) {
      const row = this.G[i];
      if (!row) continue;
      let s = 0;
      for (let j = 0; j < 12; j++) {
        s += (row[j] ?? 0) * (v[j] ?? 0);
      }
      out[i] = s;
    }
    return out;
  }

  public transformInverse(v: number[]): number[] {
    const out = new Array<number>(12).fill(0);
    for (let i = 0; i < 12; i++) {
      const row = this.G_inv[i];
      if (!row) continue;
      let s = 0;
      for (let j = 0; j < 12; j++) {
        s += (row[j] ?? 0) * (v[j] ?? 0);
      }
      out[i] = s;
    }
    return out;
  }

  // --- Internal Linear Algebra Primitives ---

  private static buildSkewSymmetric(theta: number[]): number[][] {
    const A: number[][] = Array.from({ length: 8 }, () => new Array<number>(8).fill(0));
    let idx = 0;
    for (let i = 0; i < 8; i++) {
      for (let j = i + 1; j < 8; j++) {
        const val = theta[idx++] ?? 0;
        const rowI = A[i];
        const rowJ = A[j];
        if (rowI && rowJ) {
          rowI[j] = val;
          rowJ[i] = -val;
        }
      }
    }
    return A;
  }

  public static computeCayleyTransform(A: number[][]): number[][] {
    const n = 8;
    // M = I_8 + A
    const M: number[][] = Array.from({ length: n }, (_, i) => {
      const row = new Array<number>(n).fill(0);
      const Arow = A[i];
      for (let j = 0; j < n; j++) {
        row[j] = (i === j ? 1.0 : 0.0) + (Arow ? (Arow[j] ?? 0) : 0);
      }
      return row;
    });

    // Invert M via Gauss-Jordan elimination with partial pivoting
    const invM: number[][] = Array.from({ length: n }, (_, i) => {
      const row = new Array<number>(n).fill(0);
      row[i] = 1.0;
      return row;
    });

    const aug = M.map((row) => [...row]);
    for (let k = 0; k < n; k++) {
      let maxRow = k;
      let maxVal = Math.abs(aug[k]?.[k] ?? 0);
      for (let r = k + 1; r < n; r++) {
        const v = Math.abs(aug[r]?.[k] ?? 0);
        if (v > maxVal) {
          maxVal = v;
          maxRow = r;
        }
      }
      if (maxRow !== k) {
        const tmpA = aug[k]!;
        aug[k] = aug[maxRow]!;
        aug[maxRow] = tmpA;
        const tmpI = invM[k]!;
        invM[k] = invM[maxRow]!;
        invM[maxRow] = tmpI;
      }
      const pivot = aug[k]?.[k] ?? 1.0;
      if (Math.abs(pivot) < 1e-12) {
        throw new Error('Singular matrix in Cayley transform computation');
      }
      for (let c = 0; c < n; c++) {
        aug[k]![c] = (aug[k]![c] ?? 0) / pivot;
        invM[k]![c] = (invM[k]![c] ?? 0) / pivot;
      }
      for (let r = 0; r < n; r++) {
        if (r === k) continue;
        const factor = aug[r]?.[k] ?? 0;
        if (factor === 0) continue;
        for (let c = 0; c < n; c++) {
          aug[r]![c] = (aug[r]![c] ?? 0) - factor * (aug[k]![c] ?? 0);
          invM[r]![c] = (invM[r]![c] ?? 0) - factor * (invM[k]![c] ?? 0);
        }
      }
    }

    // N = I_8 - A
    const N: number[][] = Array.from({ length: n }, (_, i) => {
      const row = new Array<number>(n).fill(0);
      const Arow = A[i];
      for (let j = 0; j < n; j++) {
        row[j] = (i === j ? 1.0 : 0.0) - (Arow ? (Arow[j] ?? 0) : 0);
      }
      return row;
    });

    // R = N * invM
    const R: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += (N[i]?.[k] ?? 0) * (invM[k]?.[j] ?? 0);
        }
        R[i]![j] = sum;
      }
    }
    return R;
  }

  private static buildHaarEmbedding(): number[][] {
    // U_(2x3) with orthonormal rows:
    // row 0: (1/sqrt(2), 0, -1/sqrt(2))
    // row 1: (1/sqrt(6), -2/sqrt(6), 1/sqrt(6))
    const invSqrt2 = 1.0 / Math.sqrt(2.0);
    const invSqrt6 = 1.0 / Math.sqrt(6.0);
    const U = [
      [invSqrt2, 0.0, -invSqrt2],
      [invSqrt6, -2.0 * invSqrt6, invSqrt6],
    ];

    // P_(8x12) = U_(2x3) (x) I_4
    const P: number[][] = Array.from({ length: 8 }, () => new Array<number>(12).fill(0));
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        const uVal = U[r]?.[c] ?? 0;
        for (let k = 0; k < 4; k++) {
          P[4 * r + k]![4 * c + k] = uVal;
        }
      }
    }
    return P;
  }

  private static buildInnerMetric(R: number[][], lambda: number[]): number[][] {
    // M8 = R^T diag(Lambda_+) R
    const n = 8;
    const diagR: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      const lam = lambda[i] ?? 1.0;
      for (let j = 0; j < n; j++) {
        diagR[i]![j] = lam * (R[i]?.[j] ?? 0);
      }
    }

    const M8: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += (R[k]?.[i] ?? 0) * (diagR[k]?.[j] ?? 0);
        }
        M8[i]![j] = sum;
      }
    }
    return M8;
  }

  private static buildPullbackMetric(P: number[][], M8: number[][], epsilon: number): number[][] {
    // G = P^T M8 P + epsilon I_12
    const rows = 8;
    const cols = 12;
    const M8P: number[][] = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        let sum = 0;
        for (let k = 0; k < rows; k++) {
          sum += (M8[i]?.[k] ?? 0) * (P[k]?.[j] ?? 0);
        }
        M8P[i]![j] = sum;
      }
    }

    const G: number[][] = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < cols; j++) {
        let sum = 0;
        for (let k = 0; k < rows; k++) {
          sum += (P[k]?.[i] ?? 0) * (M8P[k]?.[j] ?? 0);
        }
        if (i === j) {
          sum += epsilon;
        }
        G[i]![j] = sum;
      }
    }
    return G;
  }

  private static invert12x12(A: number[][]): number[][] {
    const n = 12;
    const aug = A.map((r) => [...r]);
    const inv: number[][] = Array.from({ length: n }, (_, i) => {
      const row = new Array<number>(n).fill(0);
      row[i] = 1.0;
      return row;
    });

    for (let k = 0; k < n; k++) {
      let maxRow = k;
      let maxVal = Math.abs(aug[k]?.[k] ?? 0);
      for (let r = k + 1; r < n; r++) {
        const v = Math.abs(aug[r]?.[k] ?? 0);
        if (v > maxVal) {
          maxVal = v;
          maxRow = r;
        }
      }
      if (maxRow !== k) {
        const tmpA = aug[k]!;
        aug[k] = aug[maxRow]!;
        aug[maxRow] = tmpA;
        const tmpI = inv[k]!;
        inv[k] = inv[maxRow]!;
        inv[maxRow] = tmpI;
      }
      const pivot = aug[k]?.[k] ?? 1.0;
      for (let c = 0; c < n; c++) {
        aug[k]![c] = (aug[k]![c] ?? 0) / pivot;
        inv[k]![c] = (inv[k]![c] ?? 0) / pivot;
      }
      for (let r = 0; r < n; r++) {
        if (r === k) continue;
        const factor = aug[r]?.[k] ?? 0;
        if (factor === 0) continue;
        for (let c = 0; c < n; c++) {
          aug[r]![c] = (aug[r]![c] ?? 0) - factor * (aug[k]![c] ?? 0);
          inv[r]![c] = (inv[r]![c] ?? 0) - factor * (inv[k]![c] ?? 0);
        }
      }
    }
    return inv;
  }
}

/**
 * Kac-Regularized Moreau-Yosida Thermodynamics.
 * Computes non-local power-law interaction energy, Tsallis entropy potential,
 * Lipschitz proximal gradient, and intensive free energy density O(1).
 */
export class KacMoreauYosidaThermodynamics {
  public readonly metric: SO8PullbackMetric;
  public readonly alpha: number;
  public readonly J0: number;
  public readonly qTsallis: number;
  public readonly lambdaMY: number;
  public readonly mu1: number;

  constructor(
    metric?: SO8PullbackMetric,
    params?: { alpha?: number; J0?: number; qTsallis?: number; lambdaMY?: number; mu1?: number }
  ) {
    this.metric = metric ?? new SO8PullbackMetric();
    this.alpha = params?.alpha ?? 0.8;
    this.J0 = params?.J0 ?? 1.0;
    this.qTsallis = params?.qTsallis ?? 0.5;
    this.lambdaMY = params?.lambdaMY ?? 0.1;
    this.mu1 = params?.mu1 ?? 0.05;
  }

  /**
   * Kac normalization factor Z_Kac(L, alpha) = sum_{k=1}^{L-1} k^(-alpha).
   */
  public static kacNormalizationFactor(L: number, alpha: number): number {
    if (L <= 1) return 1.0;
    let sum = 0.0;
    for (let k = 1; k < L; k++) {
      sum += Math.pow(k, -alpha);
    }
    return Math.max(1e-12, sum);
  }

  /**
   * Computes long-range Kac interaction energy U_Kac for a trajectory X in (R^12)^L.
   */
  public kacInteractionEnergy(trajectory: number[][]): number {
    const L = trajectory.length;
    if (L <= 1) return 0.0;
    const Z = KacMoreauYosidaThermodynamics.kacNormalizationFactor(L, this.alpha);
    let total = 0.0;
    for (let t = 1; t < L; t++) {
      const xt = trajectory[t]!;
      for (let k = 1; k <= t; k++) {
        const xtk = trajectory[t - k]!;
        const weight = Math.pow(k, -this.alpha);
        const distSq = this.metric.mahalanobisDistance(xt, xtk) ** 2;
        total += weight * distSq;
      }
    }
    return (this.J0 / Z) * total;
  }

  /**
   * Tsallis entropy H_q(p) = (sum p_i^q - 1) / (1 - q) for 0 < q < 1.
   */
  public tsallisEntropy(p: number[]): number {
    let sumPq = 0.0;
    for (const pi of p) {
      if (pi > 0) {
        sumPq += Math.pow(pi, this.qTsallis);
      }
    }
    return (sumPq - 1.0) / (1.0 - this.qTsallis);
  }

  /**
   * Non-smooth sparsity + negative Tsallis entropy potential Psi_q(x).
   */
  public psiQ(x: number[]): number {
    const eps = 1e-8;
    let absSum = 0.0;
    for (let i = 0; i < 12; i++) {
      absSum += Math.abs(x[i] ?? 0) + eps;
    }
    const p = new Array<number>(12);
    let l1Norm = 0.0;
    for (let i = 0; i < 12; i++) {
      const xi = x[i] ?? 0;
      p[i] = (Math.abs(xi) + eps) / absSum;
      l1Norm += Math.abs(xi);
    }
    const Hq = this.tsallisEntropy(p);
    return -Hq + this.mu1 * l1Norm;
  }

  /**
   * Analytical gradient of Psi_q(x).
   */
  public gradPsiQ(x: number[]): number[] {
    const eps = 1e-8;
    const grad = new Array<number>(12).fill(0);
    let S = 0.0;
    for (let i = 0; i < 12; i++) {
      S += Math.abs(x[i] ?? 0) + eps;
    }
    const p = new Array<number>(12);
    for (let i = 0; i < 12; i++) {
      p[i] = (Math.abs(x[i] ?? 0) + eps) / S;
    }

    const q = this.qTsallis;
    for (let k = 0; k < 12; k++) {
      const xk = x[k] ?? 0;
      const sgnK = Math.sign(xk);
      // dHq/dxk = sum_i (dHq/dpi * dpi/dxk)
      let dHq_dxk = 0.0;
      for (let i = 0; i < 12; i++) {
        const dHq_dpi = (q / (1.0 - q)) * Math.pow(p[i]!, q - 1.0);
        const delta = i === k ? 1.0 : 0.0;
        const dpi_dxk = (sgnK * (delta * S - (Math.abs(x[i] ?? 0) + eps))) / (S * S);
        dHq_dxk += dHq_dpi * dpi_dxk;
      }
      grad[k] = -dHq_dxk + this.mu1 * sgnK;
    }
    return grad;
  }

  /**
   * Moreau-Yosida Proximal Operator prox_{lambda Psi_q}^{G_style}(x).
   * Solves argmin_y { Psi_q(y) + 1/(2*lambda) ||y - x||_G^2 } via damped Newton-Raphson.
   */
  public proxPsiQ(x: number[], maxIter = 6): number[] {
    let y = [...x];
    const lambda = this.lambdaMY;

    for (let iter = 0; iter < maxIter; iter++) {
      const gradPsi = this.gradPsiQ(y);
      // diff = y - x
      const diff = y.map((v, i) => v - (x[i] ?? 0));
      // g = gradPsi + (1 / lambda) G * (y - x)
      const Gdiff = this.metric.transformVector(diff);
      const res = gradPsi.map((gp, i) => gp + (1.0 / lambda) * (Gdiff[i] ?? 0));

      // Newton step direction: Delta y = - ( (1/lambda) G )^(-1) res = - lambda G^(-1) res
      const step = this.metric.transformInverse(res).map((v) => -lambda * v);

      let stepNorm = 0.0;
      for (let i = 0; i < 12; i++) {
        y[i] = (y[i] ?? 0) + 0.5 * (step[i] ?? 0);
        stepNorm += (step[i] ?? 0) ** 2;
      }
      if (Math.sqrt(stepNorm) < 1e-6) break;
    }
    return y;
  }

  /**
   * Moreau-Yosida Smoothed Gradient nabla_x F_{q, lambda}(x) = (1/lambda) G_style (x - prox(x)).
   */
  public moreauYosidaGradient(x: number[]): number[] {
    const yStar = this.proxPsiQ(x);
    const diff = x.map((xi, i) => xi - (yStar[i] ?? 0));
    const Gdiff = this.metric.transformVector(diff);
    return Gdiff.map((v) => v / this.lambdaMY);
  }

  /**
   * Demonstrates Theorem 1.1: Free Energy Density remains O(1) intensive as L -> infinity.
   */
  public freeEnergyDensity(trajectory: number[][]): number {
    const L = trajectory.length;
    if (L === 0) return 0.0;
    const UKac = this.kacInteractionEnergy(trajectory);
    let totalPsiSmoothed = 0.0;
    for (let t = 0; t < L; t++) {
      const xt = trajectory[t]!;
      const yStar = this.proxPsiQ(xt);
      const distSq = this.metric.mahalanobisDistance(xt, yStar) ** 2;
      totalPsiSmoothed += this.psiQ(yStar) + (0.5 / this.lambdaMY) * distSq;
    }
    return (UKac + totalPsiSmoothed) / L;
  }
}

/**
 * Hamilton-Friston Resonance Engine.
 * Reader aesthetic valence spike V_t = -dF/dt and 1/f^beta cortical power equipartition.
 */
export class HamiltonFristonResonance {
  public static readerValenceSpike(
    gradFreeEnergy: number[],
    momentum: number[],
    invMassDiag?: number[],
    kappaReader = 1.0
  ): number {
    const invM = invMassDiag ?? new Array<number>(12).fill(1.0);
    let inner = 0.0;
    for (let i = 0; i < 12; i++) {
      inner += (gradFreeEnergy[i] ?? 0) * (invM[i] ?? 1.0) * (momentum[i] ?? 0);
    }
    // Tension-Resolution catharsis: negative free energy rate of change
    return -kappaReader * inner;
  }

  public static saturatedAestheticUtility(
    valence: number,
    freeEnergy: number,
    fatiguePenalty: number,
    fOverload = 15.0
  ): number {
    const overloadCutoff = Math.exp(-Math.max(0, freeEnergy) / fOverload);
    return valence * overloadCutoff - fatiguePenalty;
  }

  /**
   * Computes 1/f^beta spectral slope beta using discrete power spectral density.
   */
  public static computeSpectralSlope(signal: number[]): { beta: number; r2: number } {
    const N = signal.length;
    if (N < 8) return { beta: 0.0, r2: 0.0 };

    const halfN = Math.floor(N / 2);
    const freqs: number[] = [];
    const logP: number[] = [];

    for (let k = 1; k < halfN; k++) {
      let re = 0.0;
      let im = 0.0;
      for (let n = 0; n < N; n++) {
        const phi = (2.0 * Math.PI * k * n) / N;
        const val = signal[n] ?? 0;
        re += val * Math.cos(phi);
        im -= val * Math.sin(phi);
      }
      const power = (re * re + im * im) / N;
      if (power > 1e-12) {
        freqs.push(Math.log(k / N));
        logP.push(Math.log(power));
      }
    }

    if (freqs.length < 3) return { beta: 0.0, r2: 0.0 };

    const meanF = freqs.reduce((a, b) => a + b, 0) / freqs.length;
    const meanP = logP.reduce((a, b) => a + b, 0) / logP.length;

    let cov = 0.0;
    let varF = 0.0;
    let varP = 0.0;
    for (let i = 0; i < freqs.length; i++) {
      const df = freqs[i]! - meanF;
      const dp = logP[i]! - meanP;
      cov += df * dp;
      varF += df * df;
      varP += dp * dp;
    }

    const beta = varF > 1e-12 ? -cov / varF : 0.0;
    const r2 = varF > 1e-12 && varP > 1e-12 ? (cov * cov) / (varF * varP) : 0.0;
    return { beta, r2 };
  }

  /**
   * Measures 3-band cortical power equipartition (Macro, Meso, Micro).
   * When beta = 1.0 (pink noise), the 3 bands partition energy equally (P1 = P2 = P3).
   */
  public static corticalEquipartition(signal: number[]): {
    equipartitionScore: number;
    bandPowers: [number, number, number];
  } {
    const N = signal.length;
    if (N < 8) {
      return { equipartitionScore: 1.0, bandPowers: [1 / 3, 1 / 3, 1 / 3] };
    }
    const halfN = Math.floor(N / 2);
    const powers = new Array<number>(halfN).fill(0);

    for (let k = 1; k < halfN; k++) {
      let re = 0.0;
      let im = 0.0;
      for (let n = 0; n < N; n++) {
        const phi = (2.0 * Math.PI * k * n) / N;
        const val = signal[n] ?? 0;
        re += val * Math.cos(phi);
        im -= val * Math.sin(phi);
      }
      powers[k] = (re * re + im * im) / N;
    }

    const b1End = Math.max(2, Math.floor(halfN / 4));
    const b2End = Math.max(b1End + 1, Math.floor(halfN / 2));

    let p1 = 0.0;
    for (let k = 1; k < b1End; k++) p1 += powers[k] ?? 0;
    let p2 = 0.0;
    for (let k = b1End; k < b2End; k++) p2 += powers[k] ?? 0;
    let p3 = 0.0;
    for (let k = b2End; k < halfN; k++) p3 += powers[k] ?? 0;

    const total = p1 + p2 + p3 + 1e-12;
    const w1 = p1 / total;
    const w2 = p2 / total;
    const w3 = p3 / total;

    const deviation = 0.5 * (Math.abs(w1 - 1 / 3) + Math.abs(w2 - 1 / 3) + Math.abs(w3 - 1 / 3));
    const equipartitionScore = Math.max(0.0, 1.0 - deviation);

    return {
      equipartitionScore,
      bandPowers: [w1, w2, w3],
    };
  }
}
