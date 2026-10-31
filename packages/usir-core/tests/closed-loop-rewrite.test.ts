import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  CLOSED_LOOP_BENCHMARK_CORPUS,
  type ClosedLoopRewriteRecord,
} from '../src/dynamics/closedLoopCorpus.js';
import { extractUsir24d } from '../src/sensor/symplectic24d.js';
import { classifyPhase, fitPhaseClassifier, predictPhase } from '../src/dynamics/landauPhase.js';
import {
  generateRewriteDirectives,
  compileIterativeFeedbackPrompt,
} from '../src/dynamics/equalizer.js';

const BANNED_PROPER_NOUNS: readonly string[] = [
  'メロス',
  'セリヌンティウス',
  '李徴',
  '袁傪',
  '島村',
  '駒子',
  '下人',
  '羅生門',
  '城の崎',
  '檸檬',
  '丸善',
];

describe('Stage 2: Closed-Loop Generative Phase Transition & Hamiltonian Benchmark', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: { id: string; phase: string; text: string; z: number[] }[] = JSON.parse(
    fs.readFileSync(fixturePath, 'utf-8')
  );

  // Train reference classifier on the 40 golden corpus items
  const goldenClassifier = fitPhaseClassifier(
    fixtureData.map((item) => ({
      phase: item.phase,
      features: item.z,
    }))
  );

  it('contains exactly 16 benchmark records (8 Deadzone chunks x 2 target phases)', () => {
    expect(CLOSED_LOOP_BENCHMARK_CORPUS.length).toBe(16);

    const voidRecords = CLOSED_LOOP_BENCHMARK_CORPUS.filter(
      (r) => r.targetPhase === 'III_Void'
    );
    const criticalRecords = CLOSED_LOOP_BENCHMARK_CORPUS.filter(
      (r) => r.targetPhase === 'II_Critical'
    );

    expect(voidRecords.length).toBe(8);
    expect(criticalRecords.length).toBe(8);

    const expectedIds = [
      'DEAD-01',
      'DEAD-02',
      'DEAD-03',
      'DEAD-04',
      'DEAD-05',
      'DEAD-06',
      'DEAD-07',
      'DEAD-08',
    ];
    for (const id of expectedIds) {
      expect(voidRecords.some((r) => r.id === id)).toBe(true);
      expect(criticalRecords.some((r) => r.id === id)).toBe(true);
    }
  });

  it('guarantees zero banned proper nouns across all 16 generated rewrites', () => {
    for (const record of CLOSED_LOOP_BENCHMARK_CORPUS) {
      for (const banned of BANNED_PROPER_NOUNS) {
        expect(
          record.rewrittenText.includes(banned),
          `Banned proper noun '${banned}' detected in ${record.id} (${record.targetPhase})`
        ).toBe(false);
      }
    }
  });

  it('guarantees zero 7-gram retention against all 32 golden masterwork passages (Clean Room)', () => {
    // Collect all masterwork passages (exclude deadzone slop itself)
    const masterworks = fixtureData.filter((item) => item.phase !== 'V_Deadzone');
    const masterwork7Grams = new Set<string>();

    for (const mw of masterworks) {
      for (let i = 0; i <= mw.text.length - 7; i++) {
        masterwork7Grams.add(mw.text.slice(i, i + 7));
      }
    }

    for (const record of CLOSED_LOOP_BENCHMARK_CORPUS) {
      const text = record.rewrittenText;
      const leaked: string[] = [];

      for (let i = 0; i <= text.length - 7; i++) {
        const gram = text.slice(i, i + 7);
        if (masterwork7Grams.has(gram)) {
          leaked.push(gram);
        }
      }

      expect(
        leaked.length,
        `Leaked 7-grams detected in ${record.id} (${record.targetPhase}): ${leaked.join(', ')}`
      ).toBe(0);
    }
  });

  it('proves 8/8 III_Void rewrites achieve 100% Phase Transition and meet all physical bounds', () => {
    const voidRecords = CLOSED_LOOP_BENCHMARK_CORPUS.filter(
      (r) => r.targetPhase === 'III_Void'
    );

    for (const record of voidRecords) {
      const metrics = extractUsir24d(record.rewrittenText);
      const pred = predictPhase(metrics.z_24d, goldenClassifier);
      const defaultPred = classifyPhase(metrics.z_24d);

      // 1. Thermodynamic Phase must be III_Void
      expect(pred.phase, `Phase mismatch on ${record.id}`).toBe('III_Void');
      expect(defaultPred.phase, `Default classifier mismatch on ${record.id}`).toBe('III_Void');

      // 2. Length CV >= 0.40
      expect(
        metrics.len_cv,
        `CV_L below 0.40 on ${record.id}: got ${metrics.len_cv}`
      ).toBeGreaterThanOrEqual(0.4);

      // 3. Void order parameter phi_void >= 0.55
      expect(
        metrics.order_params.phi_void,
        `phi_void below 0.55 on ${record.id}: got ${metrics.order_params.phi_void}`
      ).toBeGreaterThanOrEqual(0.55);

      // 4. Abstract bloat density <= 0.01
      expect(
        metrics.abstract_b_mean,
        `abstract bloat above 0.01 on ${record.id}: got ${metrics.abstract_b_mean}`
      ).toBeLessThanOrEqual(0.01);
    }
  });

  it('proves 8/8 II_Critical rewrites achieve 100% Phase Transition and meet all physical bounds', () => {
    const criticalRecords = CLOSED_LOOP_BENCHMARK_CORPUS.filter(
      (r) => r.targetPhase === 'II_Critical'
    );

    for (const record of criticalRecords) {
      const metrics = extractUsir24d(record.rewrittenText);
      const pred = predictPhase(metrics.z_24d, goldenClassifier);
      const defaultPred = classifyPhase(metrics.z_24d);

      expect(pred.phase, `Golden classifier mismatch on ${record.id}`).toBe('II_Critical');
      expect(defaultPred.phase, `Default classifier mismatch on ${record.id}`).toBe('II_Critical');
      expect(metrics.len_cv, `CV_L below 0.40 on ${record.id}`).toBeGreaterThanOrEqual(0.4);
      expect(metrics.order_params.kappa_kin, `kappa_kin below 0.65 on ${record.id}`).toBeGreaterThanOrEqual(0.65);
    }
  });

  it('verifies generateRewriteDirectives and compileIterativeFeedbackPrompt on DEAD-01', () => {
    const dead01 = CLOSED_LOOP_BENCHMARK_CORPUS.find(
      (r) => r.id === 'DEAD-01' && r.targetPhase === 'III_Void'
    )!;

    const directives = generateRewriteDirectives(dead01.sourceText, 'III_Void');
    expect(directives.currentPhase).toBe('V_Deadzone');
    expect(directives.targetPhase).toBe('III_Void');
    expect(directives.deltaZ.length).toBe(24);
    expect(directives.topDeficitElements.length).toBe(4);
    expect(directives.rhythmPrescription.targetCv).toBe(0.45);
    expect(directives.compiledPrompt).toContain('USIR Phase Transition Steering Directive: III_Void');
    expect(directives.compiledPrompt).toContain('Critical Hamiltonian Deficits');

    // Test iteration feedback on converged rewrite
    const feedback = compileIterativeFeedbackPrompt(
      dead01.sourceText,
      dead01.rewrittenText,
      'III_Void',
      1
    );
    expect(feedback).toContain('CONVERGED (PASS)');
    expect(feedback).toContain('PASS');
  });
});
