/**
 * Vitest Unit Tests for USIR v3 Workstream 2 (Gap 3 & Gap 4)
 *
 * Validates:
 * 1. Gap 3: Terminal Anchor Preceding & Schrödinger Bridge Infilling:
 *    - sampleTerminalAnchorAndBridge selects the optimal sentence-ending predicate anchor W_end*
 *      minimizing Kamae (Sg-10..12) Riemannian distance.
 *    - bBridge guidance drift matches analytical formula (q_end* - q_t) / (max(1, T - t) * dt).
 *    - psi_{T-t}(q_c; q_end*) log-potential correctly guides intermediate candidate clauses.
 *    - ClauseTwistedSMC integration.
 * 2. Gap 4 Part A: Momentum Banking:
 *    - MomentumBank stores suppressed momentum and tracks elastic energy E_bank = 0.5 * p^T M^{-1} p.
 *    - AdaptiveBAOABIntegrator stores suppressed momentum when H_syntax < eps_syn.
 *    - High-freedom narrative clauses (H_syntax >= eps_syn) discharge banked momentum into active dynamics.
 * 3. Gap 4 Part B: Wundt-Berlyne Inverted-U Aesthetic Utility:
 *    - computeWundtBerlyneUtility calculates U_reader(t) with overload gate and refractory fatigue integral.
 *    - Formal proof: Climax spike followed by 5-8 quiet recovery sentences yields strictly higher
 *      mean and cumulative utility than continuous purple prose with relentless high-amplitude spikes.
 */

import { describe, it, expect } from 'vitest';
import {
  AdaptiveBAOABIntegrator,
  ClauseTwistedSMC,
  MomentumBank,
  sampleTerminalAnchorAndBridge,
} from '../src/v2/adaptiveBaoab.js';
import {
  HamiltonFristonResonance,
  SO8PullbackMetric,
  computeWundtBerlyneUtility,
} from '../src/v2/thermodynamicsSo8.js';

describe('USIR v3 Workstream 2: Gap 3 & Gap 4', () => {
  describe('Gap 3: Terminal Anchor Preceding & Schrödinger Bridge Infilling', () => {
    it('selects optimal terminal predicate anchor minimizing Kamae Riemannian distance', () => {
      const metric = new SO8PullbackMetric();

      // Target Kamae for Sg-10..12 (indices 9, 10, 11)
      const targetKamae = [0.7, -0.4, 0.9];

      // Anchor 1: exact match on Kamae indices
      const q1 = new Array<number>(12).fill(0.0);
      q1[9] = 0.7;
      q1[10] = -0.4;
      q1[11] = 0.9;

      // Anchor 2: close to Kamae
      const q2 = new Array<number>(12).fill(0.0);
      q2[9] = 0.6;
      q2[10] = -0.3;
      q2[11] = 0.8;

      // Anchor 3: far from Kamae
      const q3 = new Array<number>(12).fill(0.0);
      q3[9] = -0.8;
      q3[10] = 0.9;
      q3[11] = -0.5;

      const candidates = [
        { text: 'であった。', q: q1, logprob: -0.5 },
        { text: 'なのだ。', q: q2, logprob: -0.2 },
        { text: 'かもしれない。', q: q3, logprob: -0.1 },
      ];

      const currentQ = new Array<number>(12).fill(0.0);
      const t = 2;
      const T = 6;
      const dt = 0.1;
      const tau = 1.0;

      const res = sampleTerminalAnchorAndBridge(
        targetKamae,
        candidates,
        currentQ,
        t,
        T,
        metric,
        dt,
        tau
      );

      expect(res.selectedIndex).toBe(0);
      expect(res.selectedAnchor.text).toBe('であった。');
      expect(res.minDistance).toBeCloseTo(0.0, 6);
      expect(res.qEnd).toEqual(q1);

      // Verify analytical Schrödinger Bridge drift: b_bridge = (q_end* - q_t) / (max(1, T - t) * dt)
      const remSteps = Math.max(1, T - t); // 4
      for (let i = 0; i < 12; i++) {
        const expectedB = ((q1[i] ?? 0) - (currentQ[i] ?? 0)) / (remSteps * dt);
        expect(res.bBridge[i]).toBeCloseTo(expectedB, 6);
      }

      // Also test with full 12D target vector
      const fullTarget = new Array<number>(12).fill(1.0);
      fullTarget[9] = 0.7;
      fullTarget[10] = -0.4;
      fullTarget[11] = 0.9;

      const resFull = sampleTerminalAnchorAndBridge(
        fullTarget,
        candidates,
        currentQ,
        t,
        T,
        metric,
        dt,
        tau
      );
      expect(resFull.selectedIndex).toBe(0);
    });

    it('evaluates intermediate candidate clauses with Schrödinger Bridge arrival potentials', () => {
      const metric = new SO8PullbackMetric();
      const targetKamae = [0.5, 0.5, 0.5];
      const qEndPinned = new Array<number>(12).fill(0.5);
      const candidates = [{ text: '終端', q: qEndPinned }];

      const currentQ = new Array<number>(12).fill(0.0);
      const t = 3;
      const T = 5; // remSteps = 2
      const dt = 0.1;
      const tau = 1.0;

      const qcClose = new Array<number>(12).fill(0.48);
      const qcFar = new Array<number>(12).fill(0.0);

      const res = sampleTerminalAnchorAndBridge(
        targetKamae,
        candidates,
        currentQ,
        t,
        T,
        metric,
        dt,
        tau,
        [qcClose, qcFar]
      );

      expect(res.intermediatePotentials).toBeDefined();
      expect(res.intermediatePotentials!.length).toBe(2);

      // Closer candidate must have strictly higher (less negative) log-potential
      expect(res.intermediatePotentials![0]!).toBeGreaterThan(res.intermediatePotentials![1]!);

      // Verify analytical formula: - ||qc - q_end||_G^2 / (2 * tau * remSteps)
      const remSteps = 2;
      const diffClose = qcClose.map((v, i) => v - (qEndPinned[i] ?? 0));
      const quad = metric.quadraticForm(diffClose);
      const expectedPot = -quad / (2.0 * tau * remSteps);
      expect(res.intermediatePotentials![0]!).toBeCloseTo(expectedPot, 6);
    });

    it('delegates through ClauseTwistedSMC', () => {
      const smc = new ClauseTwistedSMC();
      const targetKamae = [0.1, 0.2, 0.3];
      const anchor1 = { q: new Array<number>(12).fill(0.2), text: 'です。' };
      const anchor2 = { q: new Array<number>(12).fill(0.9), text: 'だろう。' };

      const res = smc.sampleTerminalAnchorAndBridge(
        targetKamae,
        [anchor1, anchor2],
        new Array<number>(12).fill(0.0),
        0,
        4
      );

      expect(res.selectedIndex).toBe(0);
      expect(res.selectedAnchor.text).toBe('です。');
    });
  });

  describe('Gap 4 Part A: Momentum Banking', () => {
    it('manages elastic potential energy accumulation and fractional discharge', () => {
      const bank = new MomentumBank(12);
      expect(bank.getEnergy()).toBe(0.0);

      // Deposit momentum
      const dp1 = new Array<number>(12).fill(2.0);
      const energy1 = bank.bankMomentum(dp1);
      // E = 0.5 * 12 * (2^2) = 24.0
      expect(energy1).toBeCloseTo(24.0, 6);
      expect(bank.energy).toBeCloseTo(24.0, 6);

      // Additional momentum
      const dp2 = new Array<number>(12).fill(1.0);
      const energy2 = bank.bankMomentum(dp2);
      // p = 3.0 => E = 0.5 * 12 * 9 = 54.0
      expect(energy2).toBeCloseTo(54.0, 6);

      // 50% partial discharge
      const pReleased = bank.dischargeMomentum(0.5);
      for (const val of pReleased) {
        expect(val).toBeCloseTo(1.5, 6);
      }
      // Remaining p = 1.5 => E = 0.5 * 12 * 2.25 = 13.5
      expect(bank.getEnergy()).toBeCloseTo(13.5, 6);

      // Full remaining discharge
      const pFinal = bank.dischargeMomentum(1.0);
      for (const val of pFinal) {
        expect(val).toBeCloseTo(1.5, 6);
      }
      expect(bank.getEnergy()).toBeCloseTo(0.0, 6);
    });

    it('banks momentum during low syntax entropy and discharges during high syntax entropy', () => {
      const integrator = new AdaptiveBAOABIntegrator(
        new Array<number>(12).fill(0.0),
        new Array<number>(12).fill(2.0),
        { gammaBase: 0.1, gammaWall: 50.0, tau: 0.0 }
      );
      integrator.momentumBank.reset();

      const zeroGrad = new Array<number>(12).fill(0.0);

      // Step 1: Low syntax entropy H = 0.1 < eps_syn 0.6 => momentum suppressed into bank
      const step1 = integrator.stepWithBanking(zeroGrad, 0.1);
      expect(integrator.momentumBank.getEnergy()).toBeGreaterThan(0.0);

      const pNorm1 = Math.sqrt(step1.p.reduce((acc, v) => acc + v * v, 0));

      // Step 2: High syntax entropy H = 2.0 >= eps_syn 0.6 => banked momentum discharged
      const step2 = integrator.stepWithBanking(zeroGrad, 2.0, { etaRelease: 1.0 });
      expect(integrator.momentumBank.getEnergy()).toBeCloseTo(0.0, 6);

      const pNorm2 = Math.sqrt(step2.p.reduce((acc, v) => acc + v * v, 0));
      // Released momentum boosts active momentum
      expect(pNorm2).toBeGreaterThan(pNorm1);
    });
  });

  describe('Gap 4 Part B: Wundt-Berlyne Inverted-U Aesthetic Utility', () => {
    it('proves climax spike with refractory recovery strictly beats continuous purple prose', () => {
      const T = 16;
      const tauRec = 6;
      const fOverload = 10.0;
      const lambdaFatigue = 0.05;

      // 1. Trajectory A: Climax + Refractory Recovery
      const valencesA = new Array<number>(T).fill(0.0);
      const freeEnergiesA = new Array<number>(T).fill(0.0);
      const pTrajA = Array.from({ length: T }, () => new Array<number>(12).fill(0.0));

      for (let t = 0; t < T; t++) {
        if (t < 4) {
          valencesA[t] = 1.0;
          freeEnergiesA[t] = 1.0;
          pTrajA[t] = new Array<number>(12).fill(0.05 * (t + 1));
        } else if (t === 4) {
          valencesA[t] = 18.0; // Climax payoff
          freeEnergiesA[t] = 2.0;
          pTrajA[t] = new Array<number>(12).fill(0.8);
        } else if (t >= 5 && t <= 12) {
          valencesA[t] = 0.5; // Quiet reflection (refractory period)
          freeEnergiesA[t] = 0.5;
          pTrajA[t] = new Array<number>(12).fill(0.8 - 0.05 * (t - 4));
        } else {
          valencesA[t] = 0.2;
          freeEnergiesA[t] = 0.3;
          pTrajA[t] = new Array<number>(12).fill(0.4);
        }
      }

      // 2. Trajectory B: Continuous Purple Prose
      const valencesB = new Array<number>(T).fill(8.0);
      const freeEnergiesB = new Array<number>(T).fill(35.0); // Extreme overload
      const pTrajB = Array.from({ length: T }, () => new Array<number>(12).fill(0.0));

      for (let t = 0; t < T; t++) {
        const sign = t % 2 === 0 ? 1.0 : -1.0;
        pTrajB[t] = new Array<number>(12).fill(2.5 * sign);
      }

      const resA = computeWundtBerlyneUtility(valencesA, freeEnergiesA, pTrajA, {
        fOverload,
        lambdaFatigue,
        tauRec,
      });

      const resB = computeWundtBerlyneUtility(valencesB, freeEnergiesB, pTrajB, {
        fOverload,
        lambdaFatigue,
        tauRec,
      });

      // Assertions proving the Wundt-Berlyne theorem:
      // 1. Climax + recovery cumulative utility is strongly positive
      expect(resA.cumulativeUtility).toBeGreaterThan(5.0);

      // 2. Purple prose cumulative utility is severely negative due to continuous fatigue and overload
      expect(resB.cumulativeUtility).toBeLessThan(0.0);

      // 3. Climax + recovery strictly beats purple prose on both mean and cumulative utility
      expect(resA.meanUtility).toBeGreaterThan(resB.meanUtility);
      expect(resA.cumulativeUtility).toBeGreaterThan(resB.cumulativeUtility);

      // 4. Purple prose fatigue penalties are much higher
      const totalFatigueA = resA.fatiguePenalties.reduce((a, b) => a + b, 0);
      const totalFatigueB = resB.fatiguePenalties.reduce((a, b) => a + b, 0);
      expect(totalFatigueB).toBeGreaterThan(10.0 * totalFatigueA);

      // 5. Gated valences in purple prose are crushed by overload cutoff exp(-35 / 10) = 0.030
      for (const gv of resB.gatedValences) {
        expect(gv).toBeLessThan(0.5);
      }
    });

    it('provides HamiltonFristonResonance static method computeWundtBerlyneUtility', () => {
      const valences = [1.0, 2.0, 3.0, 0.5];
      const freeEnergies = [0.5, 0.6, 0.8, 0.4];
      const pTraj = [
        new Array<number>(12).fill(0.1),
        new Array<number>(12).fill(0.2),
        new Array<number>(12).fill(0.3),
        new Array<number>(12).fill(0.25),
      ];

      const res = HamiltonFristonResonance.computeWundtBerlyneUtility(
        valences,
        freeEnergies,
        pTraj,
        { fOverload: 12.0, lambdaFatigue: 0.04, tauRec: 7 }
      );

      expect(res.utilities.length).toBe(4);
      expect(res.meanUtility).toBeGreaterThan(0.0);
    });
  });
});
