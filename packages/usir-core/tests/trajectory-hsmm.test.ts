import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  extractTrajectoryWave,
  decodeHsmmModes,
  packTrajectoryToTierM,
  detectDeadzoneSinkholes,
  NARRATIVE_MODE_NAMES,
  type TrajectoryWindow,
  type HsmmDecodingResult,
} from '../src/dynamics/trajectoryHsmm.js';
import { decodeTierM } from '../src/schema/usirBinarySchema.js';

describe('Stage 3: 3-Mode HSMM Trajectory Engine & Deadzone Sinkhole Detector', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: { id: string; phase: string; text: string; z: number[] }[] = JSON.parse(
    fs.readFileSync(fixturePath, 'utf-8')
  );

  const void01 = fixtureData.find((x) => x.id === 'VOID-01')!.text;
  const void02 = fixtureData.find((x) => x.id === 'VOID-02')!.text;
  const void03 = fixtureData.find((x) => x.id === 'VOID-03')!.text;
  const crit01 = fixtureData.find((x) => x.id === 'CRIT-01')!.text;
  const crit02 = fixtureData.find((x) => x.id === 'CRIT-02')!.text;
  const dead01 = fixtureData.find((x) => x.id === 'DEAD-01')!.text;

  // ===========================================================================
  // 1. Sliding Window Trajectory Wave Extraction
  // ===========================================================================
  describe('extractTrajectoryWave', () => {
    it('extracts valid trajectory windows across multi-paragraph literary prose', () => {
      const multiParagraph = `${void01}\n\n${crit01}`;
      const wave = extractTrajectoryWave(multiParagraph, { windowSize: 3, stride: 1 });

      expect(wave.length).toBeGreaterThan(0);

      for (let i = 0; i < wave.length; i++) {
        const w = wave[i]!;
        expect(w.windowIndex).toBe(i);
        expect(w.sentenceRange[0]).toBeLessThanOrEqual(w.sentenceRange[1]);
        expect(w.charRange[0]).toBeLessThan(w.charRange[1]);
        expect(w.paragraphIndex).toBeGreaterThanOrEqual(0);
        expect(w.windowText.length).toBeGreaterThan(0);

        // 24D Symplectic structure
        expect(w.z_24d).toHaveLength(24);
        expect(w.q).toHaveLength(12);
        expect(w.p).toHaveLength(12);
        for (const val of w.z_24d) {
          expect(Number.isFinite(val)).toBe(true);
        }

        // Order parameters & groupings
        expect(w.order_params.psi_order).toBeGreaterThanOrEqual(-1.0);
        expect(w.order_params.psi_order).toBeLessThanOrEqual(1.0);
        expect(w.order_params.phi_void).toBeGreaterThanOrEqual(0);
        expect(w.order_params.kappa_kin).toBeGreaterThanOrEqual(0);
        expect(w.groups.I_Flux).toBeGreaterThanOrEqual(0);
        expect(w.groups.II_Lattice).toBeGreaterThanOrEqual(0);
        expect(w.groups.III_Void).toBeGreaterThanOrEqual(0);
        expect(w.groups.IV_Shock).toBeGreaterThanOrEqual(0);

        expect(w.len_cv).toBeGreaterThanOrEqual(0);
        expect(w.abstract_b_mean).toBeGreaterThanOrEqual(0);
        expect(['I_Crystalline', 'II_Critical', 'III_Void', 'IV_Superheated', 'V_Deadzone']).toContain(
          w.phase
        );
      }
    });

    it('honors windowSize and stride parameters', () => {
      const text = `${void01}\n\n${void02}`;
      const waveW2S1 = extractTrajectoryWave(text, { windowSize: 2, stride: 1 });
      const waveW2S2 = extractTrajectoryWave(text, { windowSize: 2, stride: 2 });

      expect(waveW2S1.length).toBeGreaterThan(waveW2S2.length);
    });

    it('gracefully handles edge cases: single sentence, short text, empty text', () => {
      // Empty string
      const emptyWave = extractTrajectoryWave('');
      expect(emptyWave).toEqual([]);

      // Whitespace only
      const wsWave = extractTrajectoryWave('   \n\n  \t  ');
      expect(wsWave).toEqual([]);

      // Single sentence
      const singleSentence = '山路を登りながら、こう考えた。';
      const singleWave = extractTrajectoryWave(singleSentence, { windowSize: 3, stride: 1 });
      expect(singleWave).toHaveLength(1);
      expect(singleWave[0]!.sentenceRange).toEqual([0, 0]);
      expect(singleWave[0]!.z_24d).toHaveLength(24);

      // 2 sentences with windowSize 5
      const twoSentences = '最初の文である。次の文である。';
      const fallbackWave = extractTrajectoryWave(twoSentences, { windowSize: 5 });
      expect(fallbackWave).toHaveLength(1);
      expect(fallbackWave[0]!.sentenceRange).toEqual([0, 1]);
    });
  });

  // ===========================================================================
  // 2. 3-Mode HSMM Viterbi Decoding
  // ===========================================================================
  describe('decodeHsmmModes', () => {
    it('decodes a dynamic narrative progression into Setup, Rupture, and Deep Void modes', () => {
      // Setup (quiet descriptive) -> Rupture (violent sudden shock) -> Deep Void (quietude)
      const setupPart = '門の下には、この男のほかに誰もいない。ただ所々に丹塗の剥げた太い円柱がある。';
      const rupturePart = '忽ち冷たい刃が閃光を放ち、激しく石畳を叩き割った！一瞬にして血潮が噴き飛ぶ！';
      const voidPart = 'ただ静まり返った闇の中に、冷たい夜風が吹き抜けていく。何の物音も残らない。';

      const sequence = `${setupPart}\n\n${rupturePart}\n\n${voidPart}`;
      const wave = extractTrajectoryWave(sequence, { windowSize: 1, stride: 1 });
      expect(wave.length).toBeGreaterThanOrEqual(3);

      const hsmm = decodeHsmmModes(wave);

      expect(hsmm.viterbiPath).toHaveLength(wave.length);
      expect(hsmm.modeNames).toHaveLength(wave.length);
      expect(hsmm.windowCount).toBe(wave.length);

      // Verify that all assigned modes are 0, 1, or 2
      for (const m of hsmm.viterbiPath) {
        expect([0, 1, 2]).toContain(m);
      }

      // Viterbi path must capture narrative differentiation (not a static single-mode flatline)
      const uniqueModes = new Set(hsmm.viterbiPath);
      expect(uniqueModes.size).toBeGreaterThanOrEqual(2);

      // Mode 1 (Rupture) should be triggered during the high-shock rupture window
      expect(uniqueModes.has(1)).toBe(true);

      // Check modeNames naming consistency
      for (let i = 0; i < hsmm.viterbiPath.length; i++) {
        expect(hsmm.modeNames[i]).toBe(NARRATIVE_MODE_NAMES[hsmm.viterbiPath[i]!]);
      }
    });

    it('produces a valid 3x3 stochastic Markov transition matrix and mode weights', () => {
      const sequence = `${void01}\n\n${crit01}\n\n${void02}`;
      const wave = extractTrajectoryWave(sequence, { windowSize: 2, stride: 1 });
      const hsmm = decodeHsmmModes(wave);

      const A = hsmm.transitionMatrix;
      expect(A).toHaveLength(3);

      // Stochastic matrix verification: each row must sum to 1.0, non-negative entries
      for (let r = 0; r < 3; r++) {
        expect(A[r]).toHaveLength(3);
        let rowSum = 0;
        for (let c = 0; c < 3; c++) {
          expect(A[r]![c]).toBeGreaterThan(0); // Laplace-smoothed > 0
          rowSum += A[r]![c]!;
        }
        expect(rowSum).toBeCloseTo(1.0, 5);
      }

      // Mode weights verification: sum to 1.0, non-negative entries
      expect(hsmm.modeWeights).toHaveLength(3);
      let weightSum = 0;
      for (let k = 0; k < 3; k++) {
        expect(hsmm.modeWeights[k]).toBeGreaterThan(0);
        weightSum += hsmm.modeWeights[k]!;
      }
      expect(weightSum).toBeCloseTo(1.0, 5);
    });

    it('computes 24D centroids and 300-float Cholesky factors with strictly positive diagonals', () => {
      const sequence = `${void01}\n\n${crit01}\n\n${void02}`;
      const wave = extractTrajectoryWave(sequence, { windowSize: 2, stride: 1 });
      const hsmm = decodeHsmmModes(wave);

      expect(hsmm.modeCentroids).toHaveLength(3);
      for (let k = 0; k < 3; k++) {
        expect(hsmm.modeCentroids[k]).toHaveLength(24);
        for (let d = 0; d < 24; d++) {
          expect(Number.isFinite(hsmm.modeCentroids[k]![d])).toBe(true);
        }
      }

      expect(hsmm.choleskyFactors).toHaveLength(3);
      for (let k = 0; k < 3; k++) {
        // Lower triangular Cholesky representation: 24 * (24 + 1) / 2 = 300 entries
        expect(hsmm.choleskyFactors[k]).toHaveLength(300);

        let factorIdx = 0;
        for (let i = 0; i < 24; i++) {
          for (let j = 0; j <= i; j++) {
            const val = hsmm.choleskyFactors[k]![factorIdx++]!;
            expect(Number.isFinite(val)).toBe(true);
            if (i === j) {
              // Diagonal precision elements must be positive
              expect(val).toBeGreaterThan(0);
            }
          }
        }
        expect(factorIdx).toBe(300);
      }
    });

    it('returns valid fallback profile on empty trajectory wave', () => {
      const emptyHsmm = decodeHsmmModes([]);
      expect(emptyHsmm.viterbiPath).toEqual([]);
      expect(emptyHsmm.modeNames).toEqual([]);
      expect(emptyHsmm.windowCount).toBe(0);
      expect(emptyHsmm.modeWeights).toEqual([1 / 3, 1 / 3, 1 / 3]);
      expect(emptyHsmm.transitionMatrix).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      expect(emptyHsmm.modeCentroids).toHaveLength(3);
      expect(emptyHsmm.choleskyFactors).toHaveLength(3);
      expect(emptyHsmm.choleskyFactors[0]).toHaveLength(300);
    });
  });

  // ===========================================================================
  // 3. Tier M 4096B Binary Roundtrip
  // ===========================================================================
  describe('packTrajectoryToTierM & decodeTierM', () => {
    it('encodes trajectory into exactly 4096-byte binary buffer and decodes with 100% roundtrip fidelity', () => {
      const sequence = `${void01}\n\n${crit01}\n\n${void02}`;
      const wave = extractTrajectoryWave(sequence, { windowSize: 2, stride: 1 });
      const rawHsmm = decodeHsmmModes(wave);

      const buffer = packTrajectoryToTierM(sequence, { windowSize: 2, stride: 1 });

      // Strict binary size check
      expect(buffer).toBeInstanceOf(Uint8Array);
      expect(buffer.byteLength).toBe(4096);

      // Roundtrip decode
      const decoded = decodeTierM(buffer);

      // Binary magic check
      const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
      expect(view.getUint32(0, true)).toBe(0x55534952); // "USIR"
      expect(decoded.tier).toBe(2); // Tier M
      expect(decoded.version).toBe(1);

      // Transition matrix fidelity
      expect(decoded.transition_matrix).toHaveLength(3);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          expect(decoded.transition_matrix[r]![c]).toBeCloseTo(
            rawHsmm.transitionMatrix[r]![c]!,
            4
          );
        }
      }

      // Weights fidelity
      expect(decoded.weights).toHaveLength(3);
      for (let k = 0; k < 3; k++) {
        expect(decoded.weights[k]).toBeCloseTo(rawHsmm.modeWeights[k]!, 4);
      }

      // Centroids fidelity (3 x 24 floats)
      expect(decoded.centroids).toHaveLength(3);
      for (let k = 0; k < 3; k++) {
        expect(decoded.centroids[k]).toHaveLength(24);
        for (let d = 0; d < 24; d++) {
          expect(decoded.centroids[k]![d]).toBeCloseTo(rawHsmm.modeCentroids[k]![d]!, 4);
        }
      }

      // Cholesky factors fidelity (3 x 300 floats)
      expect(decoded.cholesky_factors).toHaveLength(3);
      for (let k = 0; k < 3; k++) {
        expect(decoded.cholesky_factors[k]).toHaveLength(300);
        for (let f = 0; f < 300; f++) {
          expect(decoded.cholesky_factors[k]![f]).toBeCloseTo(
            rawHsmm.choleskyFactors[k]![f]!,
            4
          );
        }
      }
    });
  });

  // ===========================================================================
  // 4. Localized Deadzone Sinkhole Detection & Surgical Directives
  // ===========================================================================
  describe('detectDeadzoneSinkholes', () => {
    it('surgically pinpoints the deadzone sinkhole paragraph without false positives in literary sections', () => {
      // Paragraph 0: High quality Critical text (Akutagawa Rashomon)
      // Paragraph 1: High quality Void text (Soseki Garasudo)
      // Paragraph 2: AI Slop Deadzone text (DEAD-01)
      // Paragraph 3: High quality Void text (Soseki Kusamakura)
      const hybridDoc = [crit01, void02, dead01, void03].join('\n\n');

      const result = detectDeadzoneSinkholes(hybridDoc);

      expect(result.hasSinkholes).toBe(true);
      expect(result.sinkholeCount).toBe(1);
      expect(result.sinkholes).toHaveLength(1);

      const sinkhole = result.sinkholes[0]!;
      expect(sinkhole.paragraphIndex).toBe(2); // Exactly paragraph 2 (DEAD-01)
      expect(sinkhole.sentenceRange[0]).toBeLessThanOrEqual(sinkhole.sentenceRange[1]);
      expect(sinkhole.textExcerpt).toContain('電車の事故で怪我を負った私は');

      // Physical diagnostics of the sinkhole
      expect(sinkhole.localPhase).toBe('V_Deadzone');
      expect(sinkhole.localCvL).toBeLessThan(0.35);

      // Surgical rewrite directives
      expect(sinkhole.surgicalDirective).toBeDefined();
      expect(sinkhole.surgicalDirective.currentPhase).toBe('V_Deadzone');
      expect(sinkhole.surgicalDirective.targetPhase).toBe('III_Void');
      expect(sinkhole.surgicalDirective.phaseSpecificRules.length).toBeGreaterThan(0);
      expect(sinkhole.surgicalDirective.instructions.length).toBeGreaterThan(0);
    });

    it('reports 0 sinkholes on clean, high-quality literary texts', () => {
      const cleanDoc = [crit01, void01, crit02, void02].join('\n\n');
      const result = detectDeadzoneSinkholes(cleanDoc);

      expect(result.hasSinkholes).toBe(false);
      expect(result.sinkholeCount).toBe(0);
      expect(result.sinkholes).toEqual([]);
    });

    it('identifies multiple sinkholes when multiple deadzone paragraphs are present', () => {
      const dead02 = fixtureData.find((x) => x.id === 'DEAD-02')!.text;
      const multiSlopDoc = [crit01, dead01, void01, dead02].join('\n\n');

      const result = detectDeadzoneSinkholes(multiSlopDoc);

      expect(result.hasSinkholes).toBe(true);
      expect(result.sinkholeCount).toBe(2);
      expect(result.sinkholes.map((s) => s.paragraphIndex)).toEqual([1, 3]);
    });
  });
});
