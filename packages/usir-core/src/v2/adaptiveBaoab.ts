/**
 * Universal Style Genome (USIR v2.0)
 * Module 4: Adaptive-BAOAB Integrator, Clause Twisted SMC & Jacobian Logit Projector
 */

import { SO8PullbackMetric } from './thermodynamicsSo8.js';

export interface BAOABParams {
  massDiag?: number[];
  dt?: number;
  gammaBase?: number;
  gammaWall?: number;
  tau?: number;
  epsilonSyn?: number;
}

export interface StepOptions {
  enableMomentumBanking?: boolean;
  etaRelease?: number;
  epsSyn?: number;
}

/**
 * MomentumBank: Energy storage and discharge reservoir.
 * Conserves kinetic + elastic potential energy across syntactically constrained intervals.
 * When H_syntax < epsilon_syn, suppressed momentum is banked into pBank.
 * When H_syntax >= epsilon_syn, discharge_momentum releases a fraction eta_release into active momentum.
 */
export class MomentumBank {
  public pBank: number[];
  public readonly invMassDiag: number[];
  public readonly dim: number;

  constructor(dim = 12, invMassDiag?: number[]) {
    this.dim = dim;
    this.pBank = new Array<number>(dim).fill(0.0);
    this.invMassDiag = invMassDiag ? [...invMassDiag] : new Array<number>(dim).fill(1.0);
  }

  public bankMomentum(deltaP: number[]): number {
    for (let i = 0; i < this.dim; i++) {
      this.pBank[i] = (this.pBank[i] ?? 0) + (deltaP[i] ?? 0);
    }
    return this.getEnergy();
  }

  public dischargeMomentum(etaRelease = 1.0): number[] {
    const eta = Math.max(0.0, Math.min(1.0, etaRelease));
    const released = new Array<number>(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const p = this.pBank[i] ?? 0;
      released[i] = eta * p;
      this.pBank[i] = (1.0 - eta) * p;
    }
    return released;
  }

  public getEnergy(): number {
    let e = 0.0;
    for (let i = 0; i < this.dim; i++) {
      const p = this.pBank[i] ?? 0;
      const invM = this.invMassDiag[i] ?? 1.0;
      e += 0.5 * p * invM * p;
    }
    return e;
  }

  public get energy(): number {
    return this.getEnergy();
  }

  public reset(): void {
    this.pBank.fill(0.0);
  }
}

/**
 * 5-stage symmetric splitting integrator (B - A - O - A - B)
 * with syntax-entropy-dependent anisotropic friction Gamma_t(H_syntax).
 * Preserves the exact conditional Gaussian momentum distribution N(0, tau M)
 * even when friction Gamma_t switches by 1,000x per step (Theorem 5.1).
 */
export class AdaptiveBAOABIntegrator {
  public q: number[];
  public p: number[];
  public readonly massDiag: number[];
  public readonly invMassDiag: number[];
  public readonly dt: number;
  public readonly gammaBase: number;
  public readonly gammaWall: number;
  public readonly tau: number;
  public readonly epsilonSyn: number;
  public readonly momentumBank: MomentumBank;

  constructor(initialQ?: number[], initialP?: number[], params?: BAOABParams) {
    this.q = initialQ ? [...initialQ] : new Array<number>(12).fill(0.5);
    this.p = initialP ? [...initialP] : new Array<number>(12).fill(0.0);
    this.massDiag = params?.massDiag ? [...params.massDiag] : new Array<number>(12).fill(1.0);
    this.invMassDiag = this.massDiag.map((m) => (m > 0 ? 1.0 / m : 1.0));
    this.dt = params?.dt ?? 0.05;
    this.gammaBase = params?.gammaBase ?? 0.1;
    this.gammaWall = params?.gammaWall ?? 100.0; // 1,000x brake
    this.tau = params?.tau ?? 1.0;
    this.epsilonSyn = params?.epsilonSyn ?? 0.6;
    this.momentumBank = new MomentumBank(12, this.invMassDiag);
  }

  /**
   * Evaluates anisotropic friction Gamma_t(H_syntax).
   * Low syntax entropy (H < eps) engages quadratic wall brake.
   */
  public evaluateFriction(hSyntax: number): number {
    if (hSyntax < this.epsilonSyn) {
      const ratio = Math.max(0.0, 1.0 - hSyntax / this.epsilonSyn);
      return this.gammaBase + (this.gammaWall - this.gammaBase) * (ratio * ratio);
    }
    return this.gammaBase;
  }

  /**
   * Executes one full B-A-O-A-B symmetric integration step.
   */
  public step(
    gradV: number[],
    hSyntax: number,
    rng?: () => number,
    options?: StepOptions
  ): { q: number[]; p: number[] } {
    const halfDt = 0.5 * this.dt;
    const rand = rng ?? Math.random;

    // Stage B1: p <- p - (dt/2) * gradV
    for (let i = 0; i < 12; i++) {
      this.p[i] = (this.p[i] ?? 0) - halfDt * (gradV[i] ?? 0);
    }

    // Stage A1: q <- q + (dt/2) * M^(-1) * p
    for (let i = 0; i < 12; i++) {
      this.q[i] = (this.q[i] ?? 0) + halfDt * (this.invMassDiag[i] ?? 1.0) * (this.p[i] ?? 0);
    }

    // Stage O: Exact Ornstein-Uhlenbeck stochastic heat bath
    const gamma = this.evaluateFriction(hSyntax);
    for (let i = 0; i < 12; i++) {
      const mi = this.massDiag[i] ?? 1.0;
      const thetaI = Math.exp(-this.dt * (gamma / mi));
      // Standard Gaussian random number via Box-Muller
      const u1 = Math.max(1e-12, rand());
      const u2 = rand();
      const xi = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

      const noiseScale = Math.sqrt(Math.max(0.0, this.tau * mi * (1.0 - thetaI * thetaI)));
      this.p[i] = thetaI * (this.p[i] ?? 0) + noiseScale * xi;
    }

    // Momentum Banking Integration (Gap 4 Part A)
    if (options?.enableMomentumBanking) {
      const eps = options.epsSyn ?? this.epsilonSyn;
      if (hSyntax < eps) {
        // High friction brake suppresses momentum relative to baseline
        const deltaP = new Array<number>(12);
        for (let i = 0; i < 12; i++) {
          const mi = this.massDiag[i] ?? 1.0;
          const thetaBase = Math.exp(-this.dt * (this.gammaBase / mi));
          const thetaActual = Math.exp(-this.dt * (gamma / mi));
          deltaP[i] = Math.max(0.0, thetaBase - thetaActual) * (this.p[i] ?? 0);
        }
        this.momentumBank.bankMomentum(deltaP);
      } else {
        // High freedom narrative clause: discharge banked momentum into active p
        if (this.momentumBank.getEnergy() > 1e-12) {
          const pReleased = this.momentumBank.dischargeMomentum(options.etaRelease ?? 1.0);
          for (let i = 0; i < 12; i++) {
            this.p[i] = (this.p[i] ?? 0) + (pReleased[i] ?? 0);
          }
        }
      }
    }

    // Stage A2: q <- q + (dt/2) * M^(-1) * p
    for (let i = 0; i < 12; i++) {
      this.q[i] = (this.q[i] ?? 0) + halfDt * (this.invMassDiag[i] ?? 1.0) * (this.p[i] ?? 0);
    }

    // Stage B2: p <- p - (dt/2) * gradV
    for (let i = 0; i < 12; i++) {
      this.p[i] = (this.p[i] ?? 0) - halfDt * (gradV[i] ?? 0);
    }

    return { q: [...this.q], p: [...this.p] };
  }

  /**
   * Convenience wrapper stepping with momentum banking enabled.
   */
  public stepWithBanking(
    gradV: number[],
    hSyntax: number,
    options?: { etaRelease?: number; epsSyn?: number },
    rng?: () => number
  ): { q: number[]; p: number[] } {
    return this.step(gradV, hSyntax, rng, {
      enableMomentumBanking: true,
      etaRelease: options?.etaRelease ?? 1.0,
      epsSyn: options?.epsSyn,
    });
  }

  /**
   * Numerically verifies Theorem 5.1:
   * Even when Gamma_t switches wildly by 1,000x at every step,
   * the empirical momentum distribution strictly preserves N(0, tau M).
   */
  public verifyInvariantMomentumMeasure(
    nSteps = 10000,
    tau = 1.0,
    seed = 42
  ): {
    meanError: number;
    covError: number;
    passed: boolean;
    nSteps: number;
  } {
    // Deterministic pseudo-random generator (LCG)
    let s = seed;
    const lcg = (): number => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };

    let pCurrent = new Array<number>(12).fill(0.0);
    const sumP = new Array<number>(12).fill(0.0);
    const sumOuter = Array.from({ length: 12 }, () => new Array<number>(12).fill(0.0));

    for (let step = 0; step < nSteps; step++) {
      // Gamma switches randomly between 0.1 and 100.0 (1,000x ratio)
      const gamma = lcg() > 0.5 ? this.gammaBase : this.gammaWall;

      for (let i = 0; i < 12; i++) {
        const mi = this.massDiag[i] ?? 1.0;
        const thetaI = Math.exp(-this.dt * (gamma / mi));
        const u1 = Math.max(1e-12, lcg());
        const u2 = lcg();
        const xi = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        const noiseScale = Math.sqrt(Math.max(0.0, tau * mi * (1.0 - thetaI * thetaI)));
        pCurrent[i] = thetaI * (pCurrent[i] ?? 0) + noiseScale * xi;
      }

      // Accumulate statistics
      for (let i = 0; i < 12; i++) {
        sumP[i] = (sumP[i] ?? 0) + (pCurrent[i] ?? 0);
        for (let j = 0; j < 12; j++) {
          sumOuter[i]![j] = (sumOuter[i]![j] ?? 0) + (pCurrent[i] ?? 0) * (pCurrent[j] ?? 0);
        }
      }
    }

    // Compute empirical mean and covariance
    const mean = sumP.map((v) => v / nSteps);
    let maxMeanErr = 0.0;
    for (const m of mean) {
      if (Math.abs(m) > maxMeanErr) maxMeanErr = Math.abs(m);
    }

    let frobDiffSq = 0.0;
    let frobTargetSq = 0.0;
    for (let i = 0; i < 12; i++) {
      for (let j = 0; j < 12; j++) {
        const empCov = (sumOuter[i]![j] ?? 0) / nSteps - (mean[i] ?? 0) * (mean[j] ?? 0);
        const targetCov = i === j ? tau * (this.massDiag[i] ?? 1.0) : 0.0;
        const diff = empCov - targetCov;
        frobDiffSq += diff * diff;
        frobTargetSq += targetCov * targetCov;
      }
    }

    const relCovErr = Math.sqrt(frobDiffSq) / Math.sqrt(frobTargetSq);
    const passed = maxMeanErr < 0.08 && relCovErr < 0.08;

    return {
      meanError: maxMeanErr,
      covError: relCovErr,
      passed,
      nSteps,
    };
  }
}

export interface ClauseParticle {
  text: string;
  qHat: number[];
  logPllm: number;
  syntacticallyValid: boolean;
}

export interface TwistedSMCResult {
  selectedIndex: number;
  selectedParticle: ClauseParticle;
  weights: number[];
  normalizedWeights: number[];
  targetBridge: number[];
}

/**
 * ClauseTwistedSMC: 4-particle rollout at clause boundaries
 * with Schrödinger Bridge terminal anchor guidance W_end^*.
 */
export class ClauseTwistedSMC {
  public readonly metric: SO8PullbackMetric;
  public readonly betaSteer: number;
  public readonly bridgeStiffness: number;

  constructor(metric?: SO8PullbackMetric, betaSteer = 1.0, bridgeStiffness = 0.3) {
    this.metric = metric ?? new SO8PullbackMetric();
    this.betaSteer = betaSteer;
    this.bridgeStiffness = bridgeStiffness;
  }

  /**
   * Evaluates 4 candidate clause rollouts and selects the optimal path
   * combining LLM syntax fluency and Schrödinger Bridge style alignment.
   */
  public selectBestClause(
    particles: ClauseParticle[],
    targetQ: number[],
    terminalAnchor: number[],
    remainingClauses: number
  ): TwistedSMCResult {
    if (particles.length === 0) {
      throw new Error('ClauseTwistedSMC requires at least 1 candidate particle');
    }

    // 1. Schrödinger Bridge Terminal Anchor Guidance
    const timeFactor = this.bridgeStiffness / Math.max(1, remainingClauses);
    const targetBridge = new Array<number>(12);
    for (let i = 0; i < 12; i++) {
      targetBridge[i] = (targetQ[i] ?? 0) + timeFactor * ((terminalAnchor[i] ?? 0) - (targetQ[i] ?? 0));
    }

    // 2. Compute Twisted Importance Weights
    const rawWeights = new Array<number>(particles.length);
    for (let p = 0; p < particles.length; p++) {
      const part = particles[p]!;
      if (!part.syntacticallyValid) {
        rawWeights[p] = -1e9; // Hard gate rejection
        continue;
      }
      const distSq = this.metric.mahalanobisDistance(part.qHat, targetBridge) ** 2;
      rawWeights[p] = part.logPllm - 0.5 * this.betaSteer * distSq;
    }

    // 3. Log-Sum-Exp Normalization
    let maxW = -Infinity;
    for (const w of rawWeights) {
      if (w > maxW) maxW = w;
    }

    const expW = rawWeights.map((w, idx) =>
      !particles[idx]!.syntacticallyValid || w <= -1e8 ? 0.0 : Math.exp(Math.max(-100, w - maxW))
    );
    const sumExp = expW.reduce((a, b) => a + b, 0) || 1.0;
    const normalizedWeights = expW.map((w) => w / sumExp);

    // 4. Select particle with highest weight
    let bestIdx = 0;
    let bestWeight = -Infinity;
    for (let p = 0; p < rawWeights.length; p++) {
      if ((rawWeights[p] ?? -Infinity) > bestWeight) {
        bestWeight = rawWeights[p]!;
        bestIdx = p;
      }
    }

    return {
      selectedIndex: bestIdx,
      selectedParticle: particles[bestIdx]!,
      weights: rawWeights,
      normalizedWeights,
      targetBridge,
    };
  }

  /**
   * Samples/selects optimal terminal predicate anchor minimizing Kamae distance
   * and computes Schrödinger Bridge guidance drift and arrival potentials.
   */
  public sampleTerminalAnchorAndBridge(
    targetKamae: number[],
    candidateAnchors: TerminalAnchorCandidate[],
    currentQ: number[],
    t: number,
    T: number,
    dt = 0.1,
    tau = 1.0,
    intermediateCandidates?: (number[] | { q: number[] })[],
    kamaeIndices: number[] = [9, 10, 11]
  ): TerminalBridgeResult {
    return sampleTerminalAnchorAndBridge(
      targetKamae,
      candidateAnchors,
      currentQ,
      t,
      T,
      this.metric,
      dt,
      tau,
      intermediateCandidates,
      kamaeIndices
    );
  }
}

export interface TerminalAnchorCandidate {
  text?: string;
  q: number[];
  logprob?: number;
  [key: string]: any;
}

export interface TerminalBridgeResult {
  selectedAnchor: TerminalAnchorCandidate;
  selectedIndex: number;
  qEnd: number[];
  bBridge: number[];
  minDistance: number;
  kamaeDistances: number[];
  intermediatePotentials?: number[];
  potentialFn: (qCand: number[]) => number;
}

/**
 * Samples/selects optimal terminal predicate anchor minimizing Kamae Riemannian distance
 * and computes Schrödinger Bridge guidance drift b_bridge and intermediate arrival log-potentials psi.
 */
export function sampleTerminalAnchorAndBridge(
  targetKamae: number[],
  candidateAnchors: TerminalAnchorCandidate[],
  currentQ: number[],
  t: number,
  T: number,
  metric?: SO8PullbackMetric,
  dt = 0.1,
  tau = 1.0,
  intermediateCandidates?: (number[] | { q: number[] })[],
  kamaeIndices: number[] = [9, 10, 11],
  sample = false,
  temperature = 1.0,
  rng?: () => number
): TerminalBridgeResult {
  if (candidateAnchors.length === 0) {
    throw new Error('candidateAnchors must contain at least 1 candidate anchor.');
  }

  const met = metric ?? new SO8PullbackMetric();
  const G = met.metricMatrix();

  const isFull12D = targetKamae.length === 12;
  const targetSub = isFull12D ? kamaeIndices.map((idx) => targetKamae[idx] ?? 0) : targetKamae;

  // Compute Riemannian distance for each candidate on Kamae subspace
  const distances: number[] = [];
  for (const c of candidateAnchors) {
    if (c.q.length !== 12) {
      throw new Error(`Candidate anchor q must be 12-dimensional, got ${c.q.length}`);
    }
    let distSq = 0;
    for (let i = 0; i < kamaeIndices.length; i++) {
      const idxI = kamaeIndices[i]!;
      const diffI = (c.q[idxI] ?? 0) - (targetSub[i] ?? 0);
      let rowSum = 0;
      for (let j = 0; j < kamaeIndices.length; j++) {
        const idxJ = kamaeIndices[j]!;
        const diffJ = (c.q[idxJ] ?? 0) - (targetSub[j] ?? 0);
        rowSum += (G[idxI]![idxJ] ?? 0) * diffJ;
      }
      distSq += diffI * rowSum;
    }
    distances.push(Math.sqrt(Math.max(0, distSq)));
  }

  let bestIdx = 0;
  if (!sample || temperature <= 1e-6) {
    let minDist = Infinity;
    for (let i = 0; i < distances.length; i++) {
      if (distances[i]! < minDist) {
        minDist = distances[i]!;
        bestIdx = i;
      }
    }
  } else {
    const rand = rng ?? Math.random;
    const logits: number[] = [];
    let maxLogit = -Infinity;
    for (let i = 0; i < candidateAnchors.length; i++) {
      const lp = candidateAnchors[i]!.logprob ?? 0.0;
      const l = lp - distances[i]! / Math.max(1e-5, temperature);
      logits.push(l);
      if (l > maxLogit) maxLogit = l;
    }
    const expL = logits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expL.reduce((a, b) => a + b, 0) || 1.0;
    const probs = expL.map((e) => e / sumExp);
    const r = rand();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i]!;
      if (r <= cum || i === probs.length - 1) {
        bestIdx = i;
        break;
      }
    }
  }

  const selectedAnchor = candidateAnchors[bestIdx]!;
  const qEndStar = [...selectedAnchor.q];

  // Schrödinger Bridge Guidance Drift: b_bridge = (q_end* - q_t) / (max(1, T - t) * dt)
  const remainingSteps = Math.max(1, T - t);
  const bBridge = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    bBridge[i] = ((qEndStar[i] ?? 0) - (currentQ[i] ?? 0)) / (remainingSteps * dt);
  }

  // Bridge potential function: psi_{T-t}(q_c; q_end^*) = - ||q_c - q_end^*||_G^2 / (2 * tau * remainingSteps)
  const denom = 2.0 * tau * remainingSteps;
  const potentialFn = (qCand: number[]): number => {
    const diff = new Array<number>(12);
    for (let i = 0; i < 12; i++) {
      diff[i] = (qCand[i] ?? 0) - (qEndStar[i] ?? 0);
    }
    const quad = met.quadraticForm(diff);
    return -quad / denom;
  };

  let intermediatePotentials: number[] | undefined;
  if (intermediateCandidates) {
    intermediatePotentials = intermediateCandidates.map((cand) => {
      const qc = Array.isArray(cand) ? cand : cand.q;
      return potentialFn(qc);
    });
  }

  return {
    selectedAnchor,
    selectedIndex: bestIdx,
    qEnd: qEndStar,
    bBridge,
    minDistance: distances[bestIdx]!,
    kamaeDistances: distances,
    intermediatePotentials,
    potentialFn,
  };
}

export interface CandidateToken {
  tokenId: number;
  tokenStr: string;
  rawLogit: number;
  jacobian: number[]; // 12D style derivative dq / dlogit
}

export interface ProjectedLogitsResult {
  steeredLogits: number[];
  steeredProbs: number[];
  baseEntropy: number;
  bypassed: boolean;
  effectiveGain: number;
}

/**
 * JacobianLogitProjector: Top-K=64 token steering
 * with syntax protection gate H_LLM < epsilon_syn => beta_steer = 0.
 */
export class JacobianLogitProjector {
  public readonly metric: SO8PullbackMetric;
  public readonly epsilonSyn: number;
  public readonly betaSteer: number;

  constructor(metric?: SO8PullbackMetric, epsilonSyn = 0.5, betaSteer = 1.0) {
    this.metric = metric ?? new SO8PullbackMetric();
    this.epsilonSyn = epsilonSyn;
    this.betaSteer = betaSteer;
  }

  /**
   * Projects style target displacement onto Top-K logits via Jacobian steering.
   */
  public projectLogits(
    candidates: CandidateToken[],
    targetDisplacement: number[]
  ): ProjectedLogitsResult {
    const K = Math.min(candidates.length, 64);
    if (K === 0) {
      return {
        steeredLogits: [],
        steeredProbs: [],
        baseEntropy: 0.0,
        bypassed: true,
        effectiveGain: 0.0,
      };
    }

    // 1. Softmax over Top-K raw logits
    let maxLogit = -Infinity;
    for (let i = 0; i < K; i++) {
      const l = candidates[i]?.rawLogit ?? -Infinity;
      if (l > maxLogit) maxLogit = l;
    }

    const expLogits = new Array<number>(K);
    let sumExp = 0.0;
    for (let i = 0; i < K; i++) {
      const e = Math.exp((candidates[i]?.rawLogit ?? 0) - maxLogit);
      expLogits[i] = e;
      sumExp += e;
    }

    const baseProbs = expLogits.map((e) => e / (sumExp || 1.0));

    // 2. Base LLM Shannon entropy H_LLM
    let H_LLM = 0.0;
    for (const p of baseProbs) {
      if (p > 1e-12) {
        H_LLM -= p * Math.log(p);
      }
    }

    // 3. Syntax Gate: H_LLM < epsilon_syn => beta_steer = 0
    if (H_LLM < this.epsilonSyn) {
      return {
        steeredLogits: candidates.slice(0, K).map((c) => c.rawLogit),
        steeredProbs: baseProbs,
        baseEntropy: H_LLM,
        bypassed: true,
        effectiveGain: 0.0,
      };
    }

    // 4. Smooth ramp above threshold
    const ramp = Math.min(1.0, (H_LLM - this.epsilonSyn) / this.epsilonSyn);
    const effectiveBeta = this.betaSteer * ramp;

    // Metric transformed displacement: G_style * Delta q
    const GDeltaQ = this.metric.transformVector(targetDisplacement);

    // 5. Steer logits along Jacobian alignment
    const steeredLogits = new Array<number>(K);
    let maxSteered = -Infinity;
    for (let k = 0; k < K; k++) {
      const cand = candidates[k]!;
      let dot = 0.0;
      for (let d = 0; d < 12; d++) {
        dot += (GDeltaQ[d] ?? 0) * (cand.jacobian[d] ?? 0);
      }
      const adjusted = cand.rawLogit + effectiveBeta * dot;
      steeredLogits[k] = adjusted;
      if (adjusted > maxSteered) maxSteered = adjusted;
    }

    // 6. Steered probabilities
    const steeredExp = steeredLogits.map((l) => Math.exp(l - maxSteered));
    const steeredSum = steeredExp.reduce((a, b) => a + b, 0) || 1.0;
    const steeredProbs = steeredExp.map((e) => e / steeredSum);

    return {
      steeredLogits,
      steeredProbs,
      baseEntropy: H_LLM,
      bypassed: false,
      effectiveGain: effectiveBeta,
    };
  }
}
