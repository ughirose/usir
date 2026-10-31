import { describe, it, expect } from 'vitest';
import { extractUsir24d } from '../src/sensor/symplectic24d.js';
import {
  encodeTierS,
  decodeTierS,
  encodeTierM,
  decodeTierM,
  type TierMProfile,
  type UsirState,
} from '../src/schema/usirBinarySchema.js';
import {
  extract24dV2,
  type MusicNoteEvent,
  WaveletHermiteEncoder,
} from '../src/index.js';

describe('TASK-360: Boundary, Fuzzing & Edge Cases Test Suite', () => {
  const edgeInputs = [
    { name: 'empty string', text: '' },
    { name: 'whitespace and newlines only', text: '   \n\t\r   \n   ' },
    { name: 'single character without period', text: 'あ' },
    { name: 'single period only', text: '。' },
    {
      name: '1000 characters without punctuation',
      text: 'あ'.repeat(1000),
    },
    {
      name: 'exclamation and question marks only',
      text: '！？！？！？！？！？！？！？！？',
    },
    {
      name: 'emojis and surrogate pairs',
      text: '🌸🔥💀🎉🚀✨🍣🍵🍙🌸🔥💀🎉🚀✨🍣🍵🍙',
    },
    {
      name: '10 sentences with identical length (sigma_L = 0)',
      text: new Array(10).fill('これはちょうど十五文字の文。').join(''),
    },
    {
      name: 'extreme contrast (2 chars followed by 500 chars)',
      text: '雨。' + 'あ'.repeat(500) + '。',
    },
  ];

  for (const { name, text } of edgeInputs) {
    it(`handles ${name} without NaN, Infinity, or throw`, () => {
      expect(() => {
        const res = extractUsir24d(text);

        // Check q elements in [0, 1]
        for (let i = 0; i < 12; i++) {
          const qi = res.q[i]!;
          expect(Number.isFinite(qi), `q[${i}] is not finite in ${name}: ${qi}`).toBe(true);
          expect(Number.isNaN(qi), `q[${i}] is NaN in ${name}`).toBe(false);
          expect(qi).toBeGreaterThanOrEqual(0.0);
          expect(qi).toBeLessThanOrEqual(1.0);
        }

        // Check p elements finite and >= 0
        for (let i = 0; i < 12; i++) {
          const pi = res.p[i]!;
          expect(Number.isFinite(pi), `p[${i}] is not finite in ${name}: ${pi}`).toBe(true);
          expect(Number.isNaN(pi), `p[${i}] is NaN in ${name}`).toBe(false);
          expect(pi).toBeGreaterThanOrEqual(0.0);
        }

        // Check order parameters
        expect(Number.isFinite(res.order_params.psi_order)).toBe(true);
        expect(Number.isFinite(res.order_params.phi_void)).toBe(true);
        expect(Number.isFinite(res.order_params.kappa_kin)).toBe(true);
        expect(res.order_params.kappa_kin).toBeGreaterThanOrEqual(0.04);
      }).not.toThrow();
    });
  }

  describe('Tier S (128B) and Tier M (4096B) Round-Trip Numerical Invariance', () => {
    it('round-trips Tier S with precision error < 1e-5', () => {
      const state: UsirState = {
        q: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35],
        p: [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.5, 0.6, 0.7],
        order_params: {
          psi_order: 0.812,
          phi_void: 0.677,
          kappa_kin: 1.369,
        },
        len_cv: 0.542,
      };

      const buf = encodeTierS(state);
      expect(buf.byteLength).toBe(128);

      const decoded = decodeTierS(buf);

      for (let i = 0; i < 12; i++) {
        expect(Math.abs(decoded.q[i]! - state.q[i]!)).toBeLessThan(1e-5);
        expect(Math.abs(decoded.p[i]! - state.p[i]!)).toBeLessThan(1e-5);
      }

      expect(Math.abs(decoded.order_params.psi_order - state.order_params.psi_order)).toBeLessThan(1e-5);
      expect(Math.abs(decoded.order_params.phi_void - state.order_params.phi_void)).toBeLessThan(1e-5);
      expect(Math.abs(decoded.order_params.kappa_kin - state.order_params.kappa_kin)).toBeLessThan(1e-5);
      expect(Math.abs(decoded.len_cv - state.len_cv)).toBeLessThan(1e-5);
    });

    it('round-trips Tier M with precision error < 1e-5', () => {
      const profile: TierMProfile = {
        version: 1,
        flags: 0,
        transition_matrix: [
          [0.7, 0.2, 0.1],
          [0.15, 0.75, 0.1],
          [0.2, 0.2, 0.6],
        ],
        weights: [0.4, 0.35, 0.25],
        centroids: [
          new Array(24).fill(0).map((_, i) => i * 0.04),
          new Array(24).fill(0).map((_, i) => 1.0 - i * 0.03),
          new Array(24).fill(0).map((_, i) => (i % 2 === 0 ? 0.8 : 0.2)),
        ],
        cholesky_factors: [
          new Array(300).fill(0).map((_, i) => 0.01 * (i % 10)),
          new Array(300).fill(0).map((_, i) => 0.02 * (i % 10)),
          new Array(300).fill(0).map((_, i) => 0.03 * (i % 10)),
        ],
      };

      const buf = encodeTierM(profile);
      expect(buf.byteLength).toBe(4096);

      const decoded = decodeTierM(buf);

      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          expect(Math.abs(decoded.transition_matrix[r]![c]! - profile.transition_matrix[r]![c]!)).toBeLessThan(1e-5);
        }
      }

      for (let k = 0; k < 3; k++) {
        expect(Math.abs(decoded.weights[k]! - profile.weights[k]!)).toBeLessThan(1e-5);
        for (let d = 0; d < 24; d++) {
          expect(Math.abs(decoded.centroids[k]![d]! - profile.centroids[k]![d]!)).toBeLessThan(1e-5);
        }
        for (let idx = 0; idx < 300; idx++) {
          expect(Math.abs(decoded.cholesky_factors[k]![idx]! - profile.cholesky_factors[k]![idx]!)).toBeLessThan(1e-5);
        }
      }
    });
  });

  describe('USIR v2 Extreme Fuzzing & Robustness (TS / Wasm Target)', () => {
    function assertValidV2State(z: Float64Array, desc: string) {
      expect(z.length, `${desc}: length must be 24`).toBe(24);
      for (let i = 0; i < 24; i++) {
        const val = z[i]!;
        expect(Number.isFinite(val), `${desc}: z[${i}] must be finite, got ${val}`).toBe(true);
        expect(Number.isNaN(val), `${desc}: z[${i}] is NaN`).toBe(false);
      }
      for (let i = 0; i < 12; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
        expect(z[i]!).toBeLessThanOrEqual(1.0);
      }
      for (let i = 12; i < 24; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
      }
    }

    it('processes text of only punctuation/symbols without crash or NaN', () => {
      for (const text of ['「……！？……」', '！？！？！？！？！？', '…・、。！？「」『』（）―〜', '', '   \n\t\r   ']) {
        const z = extract24dV2(text);
        assertValidV2State(z, `Punctuation: "${text}"`);
      }
    });

    it('processes kanji-only classical Chinese without crash or NaN', () => {
      const classical = '子曰學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎巧言令色鮮矣仁'.repeat(10);
      const z = extract24dV2(classical);
      assertValidV2State(z, 'Kanji-only classical');
    });

    it('processes hiragana-only and katakana-only streams', () => {
      const hira = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'.repeat(10);
      const kata = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'.repeat(10);
      assertValidV2State(extract24dV2(hira), 'Hiragana stream');
      assertValidV2State(extract24dV2(kata), 'Katakana stream');
    });

    it('processes single very long text without newline/space (>100k chars)', () => {
      const longKana = 'あ'.repeat(100005);
      const zKana = extract24dV2(longKana);
      assertValidV2State(zKana, 'Single kana 100k');

      const longKanji = '龍'.repeat(100005);
      const zKanji = extract24dV2(longKanji);
      assertValidV2State(zKanji, 'Single kanji 100k');
    });

    it('processes >100k-char structured natural text', () => {
      const passage = '山路を登りながら、こう考えた。智に働けば角が立つ。情に棹させば流される。意地を通せば窮屈だ。とかくに人の世は住みにくい。';
      const text100k = passage.repeat(Math.ceil(100000 / passage.length) + 2);
      expect(text100k.length).toBeGreaterThan(100000);
      const z = extract24dV2(text100k);
      assertValidV2State(z, '>100k-char structured text');
    });

    it('processes MIDI sequences of single note, all rests, and empty array', () => {
      // Single note
      const singleNote: MusicNoteEvent[] = [
        { pitch: 60, duration: 1.0, velocity: 64, startTime: 0.0 },
      ];
      assertValidV2State(extract24dV2(singleNote), 'MIDI single note');

      // Rests via velocity = 0
      const restNotes: MusicNoteEvent[] = Array.from({ length: 8 }, (_, i) => ({
        pitch: 0,
        duration: 1.0,
        velocity: 0,
        startTime: i * 1.0,
      }));
      assertValidV2State(extract24dV2(restNotes), 'MIDI rest notes (vel=0)');

      // Empty notes / signals array (zero allocation guard)
      assertValidV2State(extract24dV2([]), 'Empty notes array');

      const padded = WaveletHermiteEncoder.applyOpen08GhostPadding([], 16);
      expect(padded.length).toBe(16);
    });
  });
});
