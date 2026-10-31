import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  dirtyExtract,
  auditCleanRoomBinary,
  emitCleanTierS,
  emitCleanTierM,
} from '../src/cleanroom/threeRoomGate.js';
import { steerPhaseRewrite } from '../src/dynamics/equalizer.js';
import { extractUsir24d } from '../src/sensor/symplectic24d.js';
import { fitPhaseClassifier, predictPhase } from '../src/dynamics/landauPhase.js';

describe('Gate 3: 3-Room Clean Room & 7-Gram Non-Retention Audit Gate', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: any[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

  it('scans all 40 golden corpus passages and guarantees 0 contiguous 7-gram matches in Tier S (128B)', () => {
    for (const record of fixtureData) {
      const state = dirtyExtract(record.text);
      const tierSBuffer = emitCleanTierS(state, record.text);

      expect(tierSBuffer.byteLength).toBe(128);

      const audit = auditCleanRoomBinary(tierSBuffer, record.text);
      expect(audit.passed, `Audit failed for ${record.id}`).toBe(true);
      expect(audit.leaked7Grams.length, `Leaked 7-grams in ${record.id}: ${audit.leaked7Grams.join(',')}`).toBe(0);
      expect(audit.bannedWordsFound.length).toBe(0);
    }
  });

  it('scans Tier M (4096B) style profile and guarantees 0 contiguous 7-gram matches', () => {
    const sampleRecord = fixtureData[0]!;
    const state = dirtyExtract(sampleRecord.text);

    const dummyTierM = {
      transition_matrix: [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
      ],
      weights: [0.5, 0.3, 0.2],
      centroids: [state.z_24d, state.z_24d, state.z_24d],
      cholesky_factors: [
        new Array(300).fill(0.1),
        new Array(300).fill(0.2),
        new Array(300).fill(0.3),
      ],
    };

    const tierMBuffer = emitCleanTierM(dummyTierM, sampleRecord.text);
    expect(tierMBuffer.byteLength).toBe(4096);

    const audit = auditCleanRoomBinary(tierMBuffer, sampleRecord.text);
    expect(audit.passed).toBe(true);
    expect(audit.leaked7Grams.length).toBe(0);
    expect(audit.bannedWordsFound.length).toBe(0);
  });
});

describe('Gate 4: Before / After Phase Transition Rewrite Verification', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: any[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

  // Fit reference classifier from 40 golden samples
  const classifier = fitPhaseClassifier(
    fixtureData.map((item) => ({
      phase: item.phase,
      features: item.z,
    }))
  );

  it('transforms DEAD-01 into III_Void adhering to Gate 4 exit criteria', () => {
    const dead01 = fixtureData.find((item) => item.id === 'DEAD-01')!;

    // Initial state verification: Phase V Deadzone
    const initialExtracted = extractUsir24d(dead01.text);
    const initialPred = predictPhase(initialExtracted.z_24d, classifier);
    expect(initialPred.phase).toBe('V_Deadzone');
    expect(initialExtracted.len_cv).toBeLessThan(0.28);
    expect(initialExtracted.order_params.phi_void).toBeLessThan(0.1);

    // Apply Hamiltonian Phase Steering to III_Void
    const rewrittenText = steerPhaseRewrite(dead01.text, 'III_Void');

    const rewrittenExtracted = extractUsir24d(rewrittenText);
    const rewrittenPred = predictPhase(rewrittenExtracted.z_24d, classifier);

    // Gate 4 Assertions:
    // 1. Len CV recovers from 0.13 to >= 0.40
    expect(rewrittenExtracted.len_cv).toBeGreaterThanOrEqual(0.40);

    // 2. Void condensation phi_void transitions to >= 0.55
    expect(rewrittenExtracted.order_params.phi_void).toBeGreaterThanOrEqual(0.55);

    // 3. Abstract filler bloat drops to <= 0.01
    expect(rewrittenExtracted.abstract_b_mean).toBeLessThanOrEqual(0.01);

    // 4. Thermodynamic phase transitions from V_Deadzone to III_Void
    expect(rewrittenPred.phase).toBe('III_Void');
  });

  it('guarantees zero topic/work-specific keywords inside equalizer.ts', () => {
    const equalizerSourcePath = path.resolve(__dirname, '../src/dynamics/equalizer.ts');
    const sourceCode = fs.readFileSync(equalizerSourcePath, 'utf-8');

    const forbiddenTopicWords = ['蜂', '温泉', '電車', '怪我', '城の崎', '志賀', '檸檬', 'メロス', '李徴', '羅生門'];
    for (const word of forbiddenTopicWords) {
      expect(sourceCode.includes(word), `Topic shortcut word "${word}" found in equalizer.ts`).toBe(false);
    }
  });

  it('universally transforms ALL 8 AI Deadzone chunks (DEAD-01..DEAD-08) into III_Void adhering to Gate 4', () => {
    const deadChunks = fixtureData.filter((item) => item.phase === 'V_Deadzone');
    expect(deadChunks.length).toBe(8);

    for (const chunk of deadChunks) {
      const initialExtracted = extractUsir24d(chunk.text);
      const initialPred = predictPhase(initialExtracted.z_24d, classifier);
      expect(initialPred.phase, `${chunk.id} should initially be V_Deadzone`).toBe('V_Deadzone');

      // Universal phase steering
      const rewrittenText = steerPhaseRewrite(chunk.text, 'III_Void');
      const rewrittenExtracted = extractUsir24d(rewrittenText);
      const rewrittenPred = predictPhase(rewrittenExtracted.z_24d, classifier);

      // Gate 4 Criteria across ALL 8 deadzone chunks:
      // 1. Len CV recovers to >= 0.40
      expect(
        rewrittenExtracted.len_cv,
        `${chunk.id} CV_L (${rewrittenExtracted.len_cv}) must be >= 0.40`
      ).toBeGreaterThanOrEqual(0.40);

      // 2. Void condensation phi_void transitions to >= 0.55
      expect(
        rewrittenExtracted.order_params.phi_void,
        `${chunk.id} phi_void (${rewrittenExtracted.order_params.phi_void}) must be >= 0.55`
      ).toBeGreaterThanOrEqual(0.55);

      // 3. Abstract filler bloat drops to <= 0.01
      expect(
        rewrittenExtracted.abstract_b_mean,
        `${chunk.id} abstract_b_mean (${rewrittenExtracted.abstract_b_mean}) must be <= 0.01`
      ).toBeLessThanOrEqual(0.01);

      // 4. Thermodynamic phase transitions to III_Void
      expect(
        rewrittenPred.phase,
        `${chunk.id} phase must transition to III_Void, got ${rewrittenPred.phase}`
      ).toBe('III_Void');
    }
  });
});
