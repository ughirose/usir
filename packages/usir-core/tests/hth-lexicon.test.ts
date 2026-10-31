import { describe, expect, it } from 'vitest';
import {
  clz30,
  lcaDepth30Bit,
  discreteJumpDistance,
  poincareHyperbolicDistance,
  buildDiffusedClusterTable,
  synthesizeMimeticHTH,
  HTHLexicon,
  extractMusicFunctorSignals,
} from '../src/v2/hthLexicon.js';

describe('USIR v2.0 Module 1: HTHLexicon & Operators', () => {
  it('implements clz30 and 30-bit LCA depth and discrete jump distance', () => {
    expect(clz30(0)).toBe(30);
    expect(clz30(1 << 29)).toBe(0);
    expect(clz30(1 << 28)).toBe(1);
    expect(clz30(1 << 27)).toBe(2);

    const hA = 0b001010011000000000000000000000;
    const hB = 0b001010100000000000000000000000;
    // Shared levels: level 1 (001), level 2 (010) match; level 3 differs (011 vs 100)
    expect(lcaDepth30Bit(hA, hB)).toBe(2);
    expect(discreteJumpDistance(hA, hB)).toBeCloseTo(0.8, 5);

    // Identical codes have LCA depth 10 and discrete jump 0
    expect(lcaDepth30Bit(hA, hA)).toBe(10);
    expect(discreteJumpDistance(hA, hA)).toBe(0.0);
  });

  it('computes Poincare hyperbolic heat-kernel distance', () => {
    const lex = new HTHLexicon();
    const e1 = lex.lookup('氷');
    const e2 = lex.lookup('水');
    const distExact = poincareHyperbolicDistance(e1, e2, 0.0);
    expect(distExact).toBeGreaterThanOrEqual(0.0);
    expect(distExact).toBeLessThanOrEqual(1.0);

    const distSmooth = poincareHyperbolicDistance(e1, e2, 0.5);
    expect(distSmooth).toBeGreaterThanOrEqual(0.0);
    expect(distSmooth).toBeLessThanOrEqual(1.0);
  });

  it('solves 256-node Laplacian diffusion matching cluster properties', () => {
    const table = buildDiffusedClusterTable(0.85);
    expect(table.diffused8d.length).toBe(256 * 8);
    expect(table.octantPaths.length).toBe(256);
    for (let i = 0; i < 256 * 8; i++) {
      const v = table.diffused8d[i]!;
      expect(v).toBeGreaterThanOrEqual(0.0);
      expect(v).toBeLessThanOrEqual(1.0);
    }
  });

  it('OPEN-03: synthesizes onomatopoeia HTH with mimetic immunity', () => {
    const mim = synthesizeMimeticHTH('ぎらぎら');
    expect(mim.isMimetic).toBe(true);
    expect(mim.mimeticImmunity).toBe(true);
    expect(mim.concreteness).toBeCloseTo(0.85, 2);
    expect(mim.sensory8d.length).toBe(8);

    const mimP = synthesizeMimeticHTH('ぱちぱち');
    expect(mimP.isMimetic).toBe(true);
    expect(mimP.sensory8d[0]).toBeGreaterThan(0.2); // Visual/luminance from 'p'
  });

  it('OPEN-01: applies contextual polysemy damping gate', () => {
    const lex = new HTHLexicon();
    const ice = lex.lookup('氷');
    const cold = lex.lookup('冷たい');
    const lit = lex.applyOpen01PolysemyDamping(ice, cold);
    expect(lit.state).toBe('literal');

    const abstractNoun: typeof ice = {
      ...ice,
      concreteness: 0.1,
      sensory8d: [0, 0, 0, 0, 0, 0, 0, 0],
    };
    const dead = lex.applyOpen01PolysemyDamping(abstractNoun, cold);
    expect(dead.state).toBe('dead_metaphor');
  });

  it('OPEN-02: handles dual negation scope and sensory modulation', () => {
    const lex = new HTHLexicon();
    const tokens = lex.segmentText('深紅の炎などどこにもなかった');
    const negToken = tokens.find((t) => t.isNegated);
    expect(negToken).toBeDefined();
    expect(negToken!.modalityAlpha).toBe(-0.5);
    // Stationary sensory is zeroed by ReLU gate max(0, alpha)
    for (const val of negToken!.effectiveSensory8d) {
      expect(val).toBe(0.0);
    }
  });

  it('OPEN-04: cleanly separates narrative and dialogue channels', () => {
    const lex = new HTHLexicon();
    const text = '彼が歩いた。「寒いね」と呟いた。「本当に」と風が吹いた。';
    const result = lex.separateNarrativeAndDialogue(text);
    expect(result.narrativeText).toContain('彼が歩いた。');
    expect(result.narrativeText).toContain('と呟いた。');
    expect(result.dialogueText).toContain('寒いね');
    expect(result.dialogueText).toContain('本当に');
    expect(result.dialogueRatio).toBeGreaterThan(0.0);
    expect(result.alternationCount).toBe(4);
  });

  it('M1: extracts music functor signals into Mx4 matrix', () => {
    const notes = [
      { pitch: 60, startTime: 0.0, duration: 0.5, velocity: 80 },
      { pitch: 64, startTime: 0.5, duration: 0.5, velocity: 85 },
      { pitch: 67, startTime: 1.0, duration: 1.0, velocity: 90 },
      { pitch: 72, startTime: 2.0, duration: 0.5, velocity: 100 },
    ];
    const signals = extractMusicFunctorSignals(notes);
    expect(signals.length).toBeGreaterThanOrEqual(8);
    for (const row of signals) {
      expect(row.length).toBe(4);
      for (const val of row) {
        expect(val).toBeGreaterThanOrEqual(0.0);
        expect(val).toBeLessThanOrEqual(1.0);
      }
    }
  });
});
