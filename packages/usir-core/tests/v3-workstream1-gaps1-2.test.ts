import { describe, expect, it } from 'vitest';
import {
  HTHLexicon,
  DUAL_SENSE_LEXICON,
  getDualSenseEntry,
  resolveDualSenseHth,
  pack30bitOctree,
  packLower32Sensory,
  unpackLower32Sensory,
  pack64bitHth,
  unpack64bitHth,
  type DualSenseHthEntry,
} from '../src/v2/hthLexicon.js';
import {
  WaveletHermiteEncoder,
  computeLogarithmicScales,
  renormalizeDecimateSignals,
  computeRqaMetrics,
  extract24dV2,
} from '../src/v2/waveletHermiteEncoder.js';

const POLYSEMOUS_20_WORDS = [
  '冷たい', '落とす', '走る', '深い', '高い',
  '重い', '光る', '闇', '切る', '立つ',
  '熱い', '甘い', '固い', '開く', '閉じる',
  '浮かぶ', '沈む', '揺れる', '響く', '掴む',
];

describe('USIR v3 Workstream 1: Gap 1 & Gap 2 Test Suite', () => {
  // ==========================================================================
  // Gap 1: 128-bit Dual-Sense HTH Lexicon & Dynamic Contextual Resolution
  // ==========================================================================
  describe('Gap 1: 128-bit Dual-Sense HTH Lexicon', () => {
    it('verifies all 20 specified polysemous words exist with 128-bit entries', () => {
      expect(Object.keys(DUAL_SENSE_LEXICON)).toHaveLength(20);

      for (const word of POLYSEMOUS_20_WORDS) {
        const entry = getDualSenseEntry(word);
        expect(entry, `Word '${word}' missing from DUAL_SENSE_LEXICON`).toBeDefined();
        expect(entry!.word).toBe(word);

        // 64-bit BigInt properties
        const priHth = entry!.primary_physical_hth;
        const secHth = entry!.secondary_abstract_hth;
        expect(typeof priHth).toBe('bigint');
        expect(typeof secHth).toBe('bigint');
        expect(priHth).toBeGreaterThan(0n);
        expect(secHth).toBeGreaterThan(0n);
        expect(priHth !== secHth).toBe(true);

        // Concreteness separation
        expect(entry!.primary_entry.concreteness).toBeGreaterThanOrEqual(0.70);
        expect(entry!.secondary_entry.concreteness).toBeLessThan(0.40);

        // Sensory vector dimension
        expect(entry!.primary_entry.sensory8d).toHaveLength(8);
        expect(entry!.secondary_entry.sensory8d).toHaveLength(8);
      }
    });

    it('verifies 64-bit pack and unpack roundtrip fidelity', () => {
      const pos = 2;
      const octree = 0x20049249;
      const vec = [0.3, 0.1, 0.9, 0.7, 0.2, 0.4, 0.0, 0.2];

      const lower32 = packLower32Sensory(vec);
      expect(lower32).toBeGreaterThanOrEqual(0);
      const vecUnpacked = unpackLower32Sensory(lower32);
      expect(vecUnpacked).toHaveLength(8);
      for (let d = 0; d < 8; d++) {
        expect(Math.abs(vec[d]! - vecUnpacked[d]!)).toBeLessThanOrEqual(1.0 / 15.0 + 1e-6);
      }

      const hth64 = pack64bitHth(pos, octree, lower32);
      expect(typeof hth64).toBe('bigint');
      const { pos: pOut, octree30: octOut, sensory8d: vecOut } = unpack64bitHth(hth64);
      expect(pOut).toBe(pos);
      expect(octOut).toBe(octree & 0x3fffffff);
      for (let d = 0; d < 8; d++) {
        expect(vecOut[d]).toBeCloseTo(vecUnpacked[d]!, 6);
      }
    });

    it('dynamically selects Primary Physical slot when context concreteness C(u) >= 0.52', () => {
      const lex = new HTHLexicon();
      const concretePairs = [
        ['冷たい', '氷'],
        ['落とす', '瓦'],
        ['走る', '馬'],
        ['高い', '山'],
        ['固い', '石'],
      ] as const;

      for (const [word, ctx] of concretePairs) {
        const res = resolveDualSenseHth(word, ctx, lex);
        expect(res.contextConcreteness).toBeGreaterThanOrEqual(0.52);
        expect(res.selectedSlot).toBe('primary_physical');
        expect(res.selected_slot).toBe('primary_physical');

        const entry = getDualSenseEntry(word)!;
        expect(res.selectedHth).toBe(entry.primary_physical_hth);
        expect(res.primary64bit).toBe(entry.primary_physical_hth);
        expect(res.secondary64bit).toBe(entry.secondary_abstract_hth);
        expect(res.full128bit).toEqual([entry.primary_physical_hth, entry.secondary_abstract_hth]);
      }
    });

    it('dynamically selects Secondary Abstract slot when context concreteness C(u) < 0.52', () => {
      const lex = new HTHLexicon();
      const abstractPairs = [
        ['冷たい', '制度'],
        ['落とす', '論理'],
        ['冷たい', '態度'],
        ['落とす', '単位'],
        ['走る', '計画'],
      ] as const;

      for (const [word, ctx] of abstractPairs) {
        const res = resolveDualSenseHth(word, ctx, lex);
        expect(res.contextConcreteness).toBeLessThan(0.52);
        expect(res.selectedSlot).toBe('secondary_abstract');
        expect(res.selected_slot).toBe('secondary_abstract');

        const entry = getDualSenseEntry(word)!;
        expect(res.selectedHth).toBe(entry.secondary_abstract_hth);
        expect(res.primary64bit).toBe(entry.primary_physical_hth);
        expect(res.secondary64bit).toBe(entry.secondary_abstract_hth);
        expect(res.full128bit).toEqual([entry.primary_physical_hth, entry.secondary_abstract_hth]);
      }
    });

    it('verifies resolveDualSenseHth and getConcreteness on HTHLexicon class', () => {
      const lex = new HTHLexicon();
      expect(lex.getConcreteness('氷')).toBeGreaterThanOrEqual(0.52);
      expect(lex.getConcreteness('制度')).toBeLessThan(0.52);

      const resIce = lex.resolveDualSenseHth('冷たい', '氷');
      expect(resIce.selectedSlot).toBe('primary_physical');

      const resInst = lex.resolveDualSenseHth('冷たい', '制度');
      expect(resInst.selectedSlot).toBe('secondary_abstract');
    });

    it('gracefully handles non-polysemous fallback words', () => {
      const lex = new HTHLexicon();
      const res = resolveDualSenseHth('木', '森', lex);
      expect(['primary_physical', 'secondary_abstract']).toContain(res.selectedSlot);
      expect(typeof res.selectedHth).toBe('bigint');
      expect(res.full128bit).toHaveLength(2);
      expect(res.effectiveSensory).toHaveLength(8);
    });
  });

  // ==========================================================================
  // Gap 2: Logarithmic Self-Similar Renormalization & Multiscale Wavelet Filters
  // ==========================================================================
  describe('Gap 2: Logarithmic Self-Similar Renormalization', () => {
    it('computes logarithmic tertile scales and filter levels across sequence lengths', () => {
      // 1. Short haiku M = 3
      const s3 = computeLogarithmicScales(3);
      expect(s3.m).toBe(3);
      expect(s3.logM).toBeCloseTo(Math.log2(3), 5);
      expect(s3.scaleMicro).toBeLessThan(s3.scaleMeso);
      expect(s3.scaleMeso).toBeLessThan(s3.scaleMacro);
      expect(s3.tertiles).toEqual([s3.scaleMicro, s3.scaleMeso, s3.scaleMacro]);
      expect(s3.microLevel).toBeGreaterThanOrEqual(1);
      expect(s3.mesoLevel).toBeGreaterThan(s3.microLevel);
      expect(s3.macroLevel).toBeGreaterThan(s3.mesoLevel);

      // 2. Medium text M = 64
      const s64 = computeLogarithmicScales(64);
      expect(s64.logM).toBe(6.0);
      expect(s64.scaleMicro).toBeCloseTo(2.0, 5);
      expect(s64.scaleMeso).toBeCloseTo(4.0, 5);
      expect(s64.scaleMacro).toBeCloseTo(6.0, 5);

      // 3. Long novel M = 2048
      const s2048 = computeLogarithmicScales(2048);
      expect(s2048.logM).toBe(11.0);
      expect(s2048.scaleMicro).toBeCloseTo(11.0 / 3.0, 5);
      expect(s2048.scaleMeso).toBeCloseTo(22.0 / 3.0, 5);
      expect(s2048.scaleMacro).toBeCloseTo(11.0, 5);
    });

    it('preserves signals of length <= 1024 unchanged', () => {
      const sig: Float64Array[] = Array.from({ length: 512 }, () =>
        new Float64Array([0.2, 0.4, 0.6, 0.8])
      );
      const dec = renormalizeDecimateSignals(sig, 1024);
      expect(dec).toHaveLength(512);
      expect(dec[0]![0]).toBeCloseTo(0.2);
    });

    it('verifies 2-clause coarse-graining: mean fields on 0..2 and peak envelope on 3', () => {
      const sig: Float64Array[] = [
        new Float64Array([0.2, 0.4, 0.6, 0.1]),
        new Float64Array([0.4, 0.6, 0.8, 0.9]), // Pair 1: mean=[0.3, 0.5, 0.7], peak=0.9
        new Float64Array([0.1, 0.3, 0.5, 0.8]),
        new Float64Array([0.3, 0.5, 0.7, 0.2]), // Pair 2: mean=[0.2, 0.4, 0.6], peak=0.8
      ];

      const dec = renormalizeDecimateSignals(sig, 2);
      expect(dec).toHaveLength(2);

      // Pair 1
      expect(dec[0]![0]).toBeCloseTo(0.3);
      expect(dec[0]![1]).toBeCloseTo(0.5);
      expect(dec[0]![2]).toBeCloseTo(0.7);
      expect(dec[0]![3]).toBeCloseTo(0.9); // Preserved peak!

      // Pair 2
      expect(dec[1]![0]).toBeCloseTo(0.2);
      expect(dec[1]![1]).toBeCloseTo(0.4);
      expect(dec[1]![2]).toBeCloseTo(0.6);
      expect(dec[1]![3]).toBeCloseTo(0.8); // Preserved peak!
    });

    it('decimates sequences of length 2048 and 3000 down to <= 1024', () => {
      const sig2048: Float64Array[] = Array.from({ length: 2048 }, () =>
        new Float64Array([0.1, 0.2, 0.3, 0.4])
      );
      const dec2048 = renormalizeDecimateSignals(sig2048, 1024);
      expect(dec2048).toHaveLength(1024);

      const sig3000: Float64Array[] = Array.from({ length: 3000 }, () =>
        new Float64Array([0.1, 0.2, 0.3, 0.4])
      );
      const dec3000 = renormalizeDecimateSignals(sig3000, 1024);
      expect(dec3000.length).toBeLessThanOrEqual(1024);
      // 3000 -> 1500 -> 750
      expect(dec3000).toHaveLength(750);
    });

    it('preserves Beginning-Middle-End return arch for M = 2048', () => {
      const m = 2048;
      const sig: Float64Array[] = [];

      for (let i = 0; i < m; i++) {
        let latVal: number;
        if (i < 512 || i >= 1536) {
          // Opening theme and late return arch
          latVal = 0.85 + 0.05 * Math.sin((2.0 * Math.PI * (i % 512)) / 64.0);
        } else {
          // Middle contrasting development
          latVal = 0.15 + 0.05 * Math.cos((2.0 * Math.PI * (i % 512)) / 32.0);
        }
        const shockVal = i === 1800 ? 0.95 : 0.1;
        sig.push(new Float64Array([0.5, latVal, 0.4, shockVal]));
      }

      // 1. Under naive truncation (first 1024 clauses)
      const sigNaive = sig.slice(0, 1024);
      const colNaive = new Float64Array(1024);
      for (let i = 0; i < 1024; i++) colNaive[i] = sigNaive[i]![1]!;
      const rqaNaive = computeRqaMetrics(colNaive);
      // Naive truncation misses the return arch in the second half
      expect(rqaNaive.iArch).toBeLessThan(0.15);

      // 2. Under renormalization decimation
      const sigDec = renormalizeDecimateSignals(sig, 1024);
      expect(sigDec).toHaveLength(1024);
      const colDec = new Float64Array(1024);
      for (let i = 0; i < 1024; i++) colDec[i] = sigDec[i]![1]!;
      const rqaRenorm = computeRqaMetrics(colDec);

      // Return arch is preserved through decimation!
      expect(rqaRenorm.iArch).toBeGreaterThanOrEqual(0.35);
      expect(rqaRenorm.iArch).toBeGreaterThan(rqaNaive.iArch * 2.0);

      // 3. Shock peak on channel 3 is preserved
      let maxShock = 0.0;
      for (let i = 0; i < 1024; i++) {
        if (sigDec[i]![3]! > maxShock) maxShock = sigDec[i]![3]!;
      }
      expect(maxShock).toBeCloseTo(0.95, 2);

      // 4. Verify 24D extraction operates on M = 2048 without error
      const z = extract24dV2(sig, false);
      expect(z).toHaveLength(24);
      for (let i = 0; i < 12; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
        expect(z[i]!).toBeLessThanOrEqual(1.0);
      }
      for (let i = 12; i < 24; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
      }
      // Coordinate 9 (Sg-10 At: 0.5 * det + 0.5 * i_arch)
      expect(z[9]!).toBeGreaterThan(0.15);
    });
  });
});
