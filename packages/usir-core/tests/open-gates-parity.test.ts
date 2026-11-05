import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  computeConcreteness,
  applyOpen01PolysemyDamping,
  evaluateModalityAlpha,
  computeDualSensoryTensor,
  extractSpeaker,
  extractDualChannelBakhtinian,
} from '../src/linguistics/openGates.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, 'fixtures', 'open_gates_parity.json'), 'utf-8'));
const TOL = 1e-9;

function near(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((a, i) => expect(Math.abs(a - expected[i])).toBeLessThanOrEqual(TOL));
}
function num(a: number, e: number): void {
  expect(Math.abs(a - e)).toBeLessThanOrEqual(TOL);
}

describe('OPEN-01 parity', () => {
  it('computeConcreteness', () => {
    expect(fx.concreteness.length).toBeGreaterThanOrEqual(5);
    for (const c of fx.concreteness) num(computeConcreteness(c.surface, c.lemma, c.sensory), c.expected);
  });
  it('applyOpen01PolysemyDamping', () => {
    expect(fx.polysemy.length).toBeGreaterThanOrEqual(8);
    const cases = new Set<string>();
    for (const c of fx.polysemy) {
      const r = applyOpen01PolysemyDamping(c.c_u, c.v_pri, c.v_sec, c.u_vec, c.mimetic);
      const e = c.expected;
      cases.add(r.case);
      expect(r.case).toBe(e.case);
      expect(r.slot).toBe(e.slot);
      near(r.effectiveSensory, e.effective_sensory);
      num(r.dampingFactor, e.damping_factor);
      num(r.metaphorBonus, e.metaphor_bonus);
      num(r.concretenessU, e.concreteness_u);
    }
    expect(cases.size).toBe(5);
  });
});

describe('OPEN-02 parity', () => {
  it('evaluateModalityAlpha and computeDualSensoryTensor', () => {
    for (const c of fx.modality) {
      const a = evaluateModalityAlpha(c.tail, c.neg, c.spec);
      num(a, c.expected_alpha);
      for (const key of ['red', 'cold']) {
        const vec = key === 'red' ? [0.9, 0.1, 0.2, 0.0, 0.1, 0.3, 0.2, 0.0] : [0.1, 0.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0];
        const t = computeDualSensoryTensor(vec, a);
        const e = c.tensor[key];
        num(t.alphaMod, e.alpha_mod);
        near(t.sPlus, e.s_plus);
        num(t.sMinus, e.s_minus);
        near(t.sMinusVec, e.s_minus_vec);
        expect(t.isNegated).toBe(e.is_negated);
        expect(t.isSpeculative).toBe(e.is_speculative);
      }
    }
  });
});

describe('OPEN-04 parity', () => {
  it('extractSpeaker', () => {
    for (const c of fx.speaker) expect(extractSpeaker(c.before, c.after)).toBe(c.expected);
  });
  it('extractDualChannelBakhtinian', () => {
    let collapses = 0;
    for (const c of fx.bakhtin) {
      const r = extractDualChannelBakhtinian(c.text, c.threshold);
      const e = c.expected;
      expect(r.narrationText).toBe(e.narration_text);
      expect(Array.from(r.speakerUtterances.entries())).toEqual(e.speaker_utterances);
      const vecs = Array.from(r.speakerVectors.entries());
      expect(vecs.map(([k]) => k)).toEqual(e.speaker_vectors.map((x: [string, number[]]) => x[0]));
      vecs.forEach(([, v], i) => near(v, e.speaker_vectors[i][1]));
      num(r.minVoiceDistance, e.min_voice_distance);
      expect(r.dialogicCollapseDetected).toBe(e.dialogic_collapse_detected);
      expect(r.pairwiseDistances.length).toBe(e.pairwise_distances.length);
      r.pairwiseDistances.forEach(([a, b, d], i) => {
        expect([a, b]).toEqual([e.pairwise_distances[i][0], e.pairwise_distances[i][1]]);
        num(d, e.pairwise_distances[i][2]);
      });
      if (r.dialogicCollapseDetected) collapses++;
    }
    expect(collapses).toBeGreaterThan(0);
  });
});
