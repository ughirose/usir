import { describe, it, expect } from 'vitest';
import {
  HSMMLongformDirector,
  createDefaultWells,
  computeCvL,
  computeWundtBerlyneUtility,
  DEFAULT_HSMM_TRANSITION_MATRIX,
  type PhaseWell,
} from '../src/dynamics/hsmmLongformDirector.js';

describe('Phase B WS-B2: 3-Mode HSMM Multi-Well Director & Longform Generation', () => {
  // ===========================================================================
  // 1. 3 Phase Wells & Transition Matrix Normalization
  // ===========================================================================
  describe('Wells & Transition Matrix Normalization', () => {
    it('initializes 3 canonical phase wells with exact targets', () => {
      const wells = createDefaultWells(12);

      // Well 0: Narrative
      expect(wells[0]!.name).toBe('Narrative');
      expect(wells[0]!.targetCvL).toBeCloseTo(0.48);
      expect(wells[0]!.targetPhiVoid).toBeCloseTo(0.65);
      expect(wells[0]!.targetKappaKin).toBeCloseTo(0.20);
      expect(wells[0]!.targetQ).toHaveLength(12);
      expect(wells[0]!.targetP).toHaveLength(12);

      // Well 1: Dialogue
      expect(wells[1]!.name).toBe('Dialogue');
      expect(wells[1]!.targetCvL).toBeCloseTo(0.55);
      expect(wells[1]!.deltaVoice).toBeGreaterThanOrEqual(0.25);

      // Well 2: Climax
      expect(wells[2]!.name).toBe('Climax');
      expect(wells[2]!.targetCvL).toBeCloseTo(0.65);
      expect(wells[2]!.targetKappaKin).toBeCloseTo(0.70);
    });

    it('enforces row-stochastic transition matrix and refractory transition', () => {
      const director = new HSMMLongformDirector({ dim: 12 });
      const T = director.transitionMatrix;

      expect(T).toHaveLength(3);
      for (let i = 0; i < 3; i++) {
        expect(T[i]).toHaveLength(3);
        const sum = T[i]!.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1.0);
        for (let j = 0; j < 3; j++) {
          expect(T[i]![j]).toBeGreaterThanOrEqual(0.0);
        }
      }

      // OPEN-10: Climax row strictly forbids staying in Climax
      expect(T[2]![2]).toBe(0.0);
      expect(T[2]![0]).toBe(1.0); // Transitions to Narrative / Void (残心)
    });
  });

  // ===========================================================================
  // 2. OPEN-10 Wundt-Berlyne Refractory Period Gate
  // ===========================================================================
  describe('Refractory Period Gate', () => {
    it('forces transition to Well 0 immediately after Climax and tracks refractory lifecycle', () => {
      const director = new HSMMLongformDirector({ dim: 12 });

      expect(director.isInRefractory()).toBe(false);
      const next = director.stepTransition(2);
      expect(next).toBe(0);
      expect(director.isInRefractory()).toBe(true);

      const after = director.stepTransition(0);
      expect(director.isInRefractory()).toBe(false);
    });

    it('enforces multi-step cooldown with adversarial transition matrix', () => {
      const adversarialT = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
      ];
      const director = new HSMMLongformDirector({
        dim: 12,
        transitionMatrix: adversarialT,
        refractoryCooldown: 2,
      });

      const w1 = director.stepTransition(2);
      expect(w1).toBe(0);
      expect(director.isInRefractory()).toBe(true);

      // Step 2: cooldown=2 blocks Climax even though T[0][2]=1.0
      const w2 = director.stepTransition(w1);
      expect(w2).not.toBe(2);
      expect(director.isInRefractory()).toBe(true);
    });

    it('prohibits consecutive Climaxes across random walk', () => {
      const director = new HSMMLongformDirector({ dim: 12 });
      let cur = 0;
      const path: number[] = [cur];

      for (let step = 0; step < 100; step++) {
        cur = director.stepTransition(cur);
        path.push(cur);
      }

      for (let i = 0; i < path.length - 1; i++) {
        if (path[i] === 2) {
          expect(path[i + 1]).not.toBe(2);
        }
      }
    });

    it('computes inverted-U aesthetic utility peaking at optimal arousal', () => {
      const uPeak = computeWundtBerlyneUtility(0.60, 0.60);
      expect(uPeak).toBeCloseTo(1.0, 2);

      const uDeadzone = computeWundtBerlyneUtility(0.12, 0.60);
      expect(uDeadzone).toBeLessThan(0.60);

      const uPurple = computeWundtBerlyneUtility(1.80, 0.60);
      expect(uPurple).toBeLessThan(0.60);
      expect(uPurple).toBeLessThan(uPeak);
    });
  });

  // ===========================================================================
  // 3. OPEN-09 Momentum Banking
  // ===========================================================================
  describe('Momentum Banking', () => {
    it('accumulates unreleased momentum in restrained paragraphs and discharges at Climax', () => {
      const director = new HSMMLongformDirector({ dim: 12 });
      expect(director.getBankedEnergy()).toBe(0.0);

      const targetP = new Array(12).fill(0.40);
      const measuredP = new Array(12).fill(0.15);

      const delta1 = director.accumulateUnreleasedMomentum(targetP, measuredP);
      expect(delta1.every((v) => Math.abs(v - 0.25) < 1e-6)).toBe(true);
      expect(director.getBankedEnergy()).toBeGreaterThan(0.0);

      // Accumulate second step
      director.accumulateUnreleasedMomentum(targetP, measuredP);
      expect(director.pBanked.every((v) => Math.abs(v - 0.50) < 1e-6)).toBe(true);

      // Discharge burst in Climax
      const baseClimaxP = new Array(12).fill(0.70);
      const { boostedP, dischargedDeltaP } = director.dischargeMomentum(baseClimaxP, false, false); // previous behaviour

      expect(dischargedDeltaP.every((v) => Math.abs(v - 0.50) < 1e-6)).toBe(true);
      expect(boostedP.every((v) => Math.abs(v - 1.20) < 1e-6)).toBe(true);

      // Bank must be reset
      expect(director.pBanked.every((v) => v === 0.0)).toBe(true);
      expect(director.getBankedEnergy()).toBe(0.0);
    });
  });

  // ===========================================================================
  // 4. CV_L Metric & Deadzone Defense
  // ===========================================================================
  describe('CV_L Metric Calculation', () => {
    it('calculates 0.0 for monotonous sentence lengths', () => {
      expect(computeCvL([30, 30, 30, 30])).toBeCloseTo(0.0);
    });

    it('calculates healthy variation for contrasting sentence lengths', () => {
      const cv = computeCvL([12, 45, 18, 52]);
      expect(cv).toBeGreaterThanOrEqual(0.40);
      expect(cv).toBeLessThanOrEqual(0.75);
    });

    it('handles edge cases gracefully', () => {
      expect(computeCvL([])).toBe(0.0);
      expect(computeCvL([42])).toBe(0.0);
      expect(computeCvL('')).toBe(0.0);
    });
  });

  // ===========================================================================
  // 5. 3,000-Char Longform Generation Simulation
  // ===========================================================================
  describe('3,000-Char Longform Generation', () => {
    it('generates 6-8 paragraphs totaling >= 2800 chars maintaining CV_L in [0.40, 0.70]', () => {
      const director = new HSMMLongformDirector({ dim: 12 });

      const res = director.generateLongform({
        prompt: '雨上がりの古寺と苔の庭',
        targetParagraphs: 7,
        targetTotalChars: 3000,
      });

      // Character count & structure
      expect(res.paragraphCount).toBe(7);
      expect(res.paragraphs).toHaveLength(7);
      expect(res.totalChars).toBeGreaterThanOrEqual(2800);

      // Deadzone defense: CV_L must not sink to < 0.28, must stay in [0.40, 0.70]
      expect(res.inDeadzone).toBe(false);
      expect(res.isHealthy).toBe(true);
      expect(res.overallCvL).toBeGreaterThanOrEqual(0.40);
      expect(res.overallCvL).toBeLessThanOrEqual(0.70);

      // Invariant: Climax well is followed by Well 0
      for (let i = 0; i < res.wells.length - 1; i++) {
        if (res.wells[i] === 2) {
          expect(res.wells[i + 1]).toBe(0);
        }
      }

      // Banked history tracking
      expect(res.bankedHistory).toHaveLength(7);
      for (const e of res.bankedHistory) {
        expect(e).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('generates distinct text across multiple paragraphs of the same well', () => {
      const director = new HSMMLongformDirector({ dim: 12, seed: 42 });
      const res = director.generateLongform({
        prompt: '秋の夕暮れの枯野',
        targetParagraphs: 8,
        targetTotalChars: 3200,
      });

      const textsByWell = new Map<number, string[]>();
      for (const p of res.paragraphs) {
        const list = textsByWell.get(p.wellIndex) ?? [];
        list.push(p.text);
        textsByWell.set(p.wellIndex, list);
      }

      for (const [wellIdx, list] of textsByWell.entries()) {
        if (list.length > 1) {
          for (let i = 0; i < list.length; i++) {
            for (let j = i + 1; j < list.length; j++) {
              expect(list[i]).not.toBe(list[j]);
            }
          }
        }
      }
    });
  });
});
