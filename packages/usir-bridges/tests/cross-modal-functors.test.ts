import { describe, it, expect } from 'vitest';
import {
  extractVisualFeaturesFromImage,
  visualFunctorForward,
  visualFunctorInverse,
  circleOfFifthsDistance,
  intervalDissonance,
  extractMusicFeaturesFromNotes,
  musicFunctorForward,
  musicFunctorInverse,
  type VisualFeatures2D,
  type MusicFeatures,
  type MusicNoteEvent,
  type ImageData2D,
} from '../src/crossModalFunctors.js';

describe('Phase E WS-E1: Cross-Modal Visual & Music Functors', () => {
  describe('Visual Functor F_visual (2D Features <-> 24D Symplectic State <-> Diffusion Params)', () => {
    it('extracts complete 2D visual features from raw pixel grid image', () => {
      const width = 32;
      const height = 32;
      const data = new Uint8Array(width * height);
      // Create high-contrast diagonal pattern
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          data[y * width + x] = (x + y) % 4 === 0 ? 255 : 20;
        }
      }

      const img: ImageData2D = { width, height, data };
      const feats = extractVisualFeaturesFromImage(img);

      expect(feats.gaborFrequencies).toHaveLength(4);
      expect(feats.gaborFrequencies.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 2);
      expect(feats.colorEntropy).toBeGreaterThanOrEqual(0.0);
      expect(feats.colorEntropy).toBeLessThanOrEqual(1.0);
      expect(feats.chiaroscuroGradient).toBeGreaterThan(0.0);
      expect(feats.quadtreeDepthEntropy).toBeGreaterThanOrEqual(0.0);
      expect(feats.saliencyScanpath.length).toBeGreaterThan(0);
    });

    it('maps visual features to 24D symplectic state and verifies coordinate bounds', () => {
      const feats: VisualFeatures2D = {
        gaborFrequencies: [0.35, 0.25, 0.25, 0.15],
        colorEntropy: 0.65,
        chiaroscuroGradient: 0.55,
        quadtreeDepthEntropy: 0.70,
        saliencyScanpath: [
          { x: 0.2, y: 0.2, intensity: 0.4 },
          { x: 0.5, y: 0.6, intensity: 0.8 },
          { x: 0.8, y: 0.3, intensity: 0.7 },
        ],
        compositionMarginRatio: 0.30,
      };

      const z24d = visualFunctorForward(feats);
      expect(z24d).toHaveLength(24);

      const q = z24d.slice(0, 12);
      const p = z24d.slice(12, 24);

      // Invariant: Position coordinates q in [0, 1]
      for (const val of q) {
        expect(val).toBeGreaterThanOrEqual(0.0);
        expect(val).toBeLessThanOrEqual(1.0);
      }
      // Invariant: Momentum coordinates p >= 0
      for (const val of p) {
        expect(val).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('verifies Void and Shock invariant preservation in inverse Diffusion mapping', () => {
      // Case A: Void-dominant scene (minimalist / high negative space margin)
      const voidFeats: VisualFeatures2D = {
        gaborFrequencies: [0.25, 0.25, 0.25, 0.25],
        colorEntropy: 0.15,
        chiaroscuroGradient: 0.10,
        quadtreeDepthEntropy: 0.20,
        saliencyScanpath: [{ x: 0.5, y: 0.5, intensity: 0.2 }],
        compositionMarginRatio: 0.88,
      };
      const zVoid = visualFunctorForward(voidFeats);
      const diffVoid = visualFunctorInverse(zVoid);

      expect(diffVoid.compositionMarginRatio).toBeGreaterThan(0.65);
      expect(diffVoid.highFreqNoiseGain).toBeLessThan(0.40);

      // Case B: Shock-dominant scene (dramatic chiaroscuro contrast / high rupture)
      const shockFeats: VisualFeatures2D = {
        gaborFrequencies: [0.60, 0.10, 0.20, 0.10],
        colorEntropy: 0.85,
        chiaroscuroGradient: 0.92,
        quadtreeDepthEntropy: 0.80,
        saliencyScanpath: [
          { x: 0.1, y: 0.1, intensity: 0.95 },
          { x: 0.9, y: 0.9, intensity: 0.90 },
        ],
        compositionMarginRatio: 0.10,
      };
      const zShock = visualFunctorForward(shockFeats);
      const diffShock = visualFunctorInverse(zShock);

      expect(diffShock.highFreqNoiseGain).toBeGreaterThan(0.70);
      expect(diffShock.compositionMarginRatio).toBeLessThan(0.35);

      // Monotonic relationship: higher chiaroscuro -> higher noise gain
      expect(diffShock.highFreqNoiseGain).toBeGreaterThan(diffVoid.highFreqNoiseGain);
    });
  });

  describe('Music Functor F_music (Note Sequences <-> 24D Symplectic State <-> WebAudio / MIDI)', () => {
    it('calculates Circle of Fifths distance and interval dissonance accurately', () => {
      // Unison: dist = 0
      expect(circleOfFifthsDistance(0, 0)).toBe(0.0);
      // Perfect 5th (C to G): step = 7 -> distance = 1/6
      expect(circleOfFifthsDistance(0, 7)).toBeCloseTo(1 / 6, 4);
      // Tritone (C to F#): distance is maximal = 6/6 = 1.0
      expect(circleOfFifthsDistance(0, 6)).toBeCloseTo(1.0, 4);

      // Dissonance
      expect(intervalDissonance(60, 60)).toBe(0.0);  // Unison
      expect(intervalDissonance(60, 67)).toBe(0.05); // Perfect 5th (very consonant)
      expect(intervalDissonance(60, 61)).toBe(1.0);  // Minor 2nd (harsh)
      expect(intervalDissonance(60, 66)).toBe(0.85); // Tritone (dissonant)
    });

    it('extracts musical features from structured note sequence', () => {
      const notes: MusicNoteEvent[] = [
        { pitch: 60, startTime: 0.0, duration: 0.5, velocity: 80 },
        { pitch: 64, startTime: 0.5, duration: 0.5, velocity: 75 },
        { pitch: 67, startTime: 1.0, duration: 0.5, velocity: 85 },
        { pitch: 72, startTime: 2.0, duration: 0.5, velocity: 90 }, // 0.5s pause before this note
      ];

      const feats = extractMusicFeaturesFromNotes(notes);
      expect(feats.torusDistance).toBeGreaterThanOrEqual(0.0);
      expect(feats.torusDistance).toBeLessThanOrEqual(1.0);
      expect(feats.dissonance).toBeGreaterThanOrEqual(0.0);
      expect(feats.dissonance).toBeLessThanOrEqual(1.0);
      expect(feats.restRatio).toBeGreaterThan(0.0);
      expect(feats.syncopation).toBeGreaterThanOrEqual(0.0);
    });

    it('maps musical features to 24D symplectic state and verifies coordinate invariants', () => {
      const feats: MusicFeatures = {
        torusDistance: 0.35,
        dissonance: 0.20,
        syncopation: 0.30,
        restRatio: 0.40,
        bpm: 110,
      };

      const z24d = musicFunctorForward(feats);
      expect(z24d).toHaveLength(24);

      const q = z24d.slice(0, 12);
      const p = z24d.slice(12, 24);

      for (const val of q) {
        expect(val).toBeGreaterThanOrEqual(0.0);
        expect(val).toBeLessThanOrEqual(1.0);
      }
      for (const val of p) {
        expect(val).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('verifies Rest and Tension invariants in inverse WebAudio sequence synthesis', () => {
      // Case A: Serene meditation sequence (high rest ratio, low dissonance, low BPM)
      const calmFeats: MusicFeatures = {
        torusDistance: 0.10,
        dissonance: 0.05,
        syncopation: 0.10,
        restRatio: 0.70,
        bpm: 65,
      };
      const zCalm = musicFunctorForward(calmFeats);
      const audioCalm = musicFunctorInverse(zCalm);

      expect(audioCalm.restInterval).toBeGreaterThan(0.50);
      expect(audioCalm.harmonicTension).toBeLessThan(0.30);
      expect(audioCalm.reverbDecay).toBeGreaterThan(2.5);
      expect(audioCalm.bpm).toBeLessThan(100.0);
      expect(audioCalm.webAudioSequence.length).toBeGreaterThanOrEqual(1);

      // Case B: Climax battle sequence (high dissonance, high syncopation, high BPM)
      const tenseFeats: MusicFeatures = {
        torusDistance: 0.85,
        dissonance: 0.90,
        syncopation: 0.80,
        restRatio: 0.05,
        bpm: 165,
      };
      const zTense = musicFunctorForward(tenseFeats);
      const audioTense = musicFunctorInverse(zTense);

      expect(audioTense.bpm).toBeGreaterThan(120.0);
      expect(audioTense.harmonicTension).toBeGreaterThan(0.55);
      expect(audioTense.restInterval).toBeLessThan(0.30);
      expect(audioTense.solitonAccentGain).toBeGreaterThan(0.70);

      // Monotonic invariant: tense sequence has significantly higher harmonic tension than calm
      expect(audioTense.harmonicTension).toBeGreaterThan(audioCalm.harmonicTension);
      // Calm sequence has significantly longer reverb decay than tense
      expect(audioCalm.reverbDecay).toBeGreaterThan(audioTense.reverbDecay);
    });

    it('handles degenerate single-pitch music sequence stably without NaN or division by zero', () => {
      const singlePitchNotes: MusicNoteEvent[] = [
        { pitch: 60, startTime: 0.0, duration: 0.5, velocity: 80 },
        { pitch: 60, startTime: 0.5, duration: 0.5, velocity: 80 },
        { pitch: 60, startTime: 1.0, duration: 0.5, velocity: 80 },
      ];
      const feats = extractMusicFeaturesFromNotes(singlePitchNotes);
      expect(feats.torusDistance).toBe(0.0);
      expect(feats.dissonance).toBe(0.0);
      expect(feats.bpm).toBe(120);

      const z = musicFunctorForward(feats);
      expect(z).toHaveLength(24);
      for (const val of z.slice(0, 12)) {
        expect(val).toBeGreaterThanOrEqual(0.0);
        expect(val).toBeLessThanOrEqual(1.0);
      }
    });

    it('correctly calculates totalDuration when an earlier note is sustained past later notes', () => {
      const sustainedNotes: MusicNoteEvent[] = [
        { pitch: 48, startTime: 0.0, duration: 8.0, velocity: 70 }, // Bass pedal held 8s
        { pitch: 60, startTime: 1.0, duration: 0.5, velocity: 80 },
        { pitch: 64, startTime: 2.0, duration: 0.5, velocity: 80 },
      ];
      const feats = extractMusicFeaturesFromNotes(sustainedNotes);
      expect(feats.restRatio).toBeLessThan(0.1); // 9.0 active / 8.0 total -> clamped to 0
    });
  });

  describe('Edge Cases & Type Array Invariance', () => {
    it('handles uniform flat pixel buffer preserving Gabor normalization', () => {
      const flatImg: ImageData2D = {
        width: 16,
        height: 16,
        data: new Uint8Array(16 * 16).fill(255), // All white
      };
      const feats = extractVisualFeaturesFromImage(flatImg);
      expect(feats.gaborFrequencies).toEqual([0.25, 0.25, 0.25, 0.25]);
      expect(feats.gaborFrequencies.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 4);
    });

    it('accepts Float32Array in visual and music inverse functors', () => {
      const zF32 = new Float32Array(24).fill(0.3);
      const vParams = visualFunctorInverse(zF32);
      expect(vParams.cfgScale).toBeGreaterThan(0);
      const mParams = musicFunctorInverse(zF32);
      expect(mParams.bpm).toBeGreaterThan(0);
    });
  });

  describe('Conjugate momentum from window series and 24D-driven music sequence', () => {
    const mkPatch = (chiaro: number, color: number, quad: number, gabor: [number, number, number, number]): VisualFeatures2D => ({
      gaborFrequencies: gabor,
      colorEntropy: color,
      chiaroscuroGradient: chiaro,
      quadtreeDepthEntropy: quad,
      saliencyScanpath: [
        { x: 0.1, y: 0.1, intensity: 0.5 },
        { x: 0.1 + chiaro * 0.5, y: 0.4, intensity: 0.5 },
      ],
    });

    it('measures all 12 momentum coordinates from patch series (non-negative, not a copy of q)', () => {
      const patches = [
        mkPatch(0.1, 0.2, 0.3, [0.1, 0.2, 0.3, 0.4]),
        mkPatch(0.7, 0.6, 0.8, [0.5, 0.1, 0.2, 0.9]),
        mkPatch(0.3, 0.9, 0.1, [0.2, 0.8, 0.1, 0.3]),
      ];
      const z = visualFunctorForward({ ...patches[0]!, patchFeatures: patches });
      const p = z.slice(12);
      expect(p.every((v) => v >= 0)).toBe(true);
      expect(p.filter((v) => v !== 0).length).toBe(12);
    });

    it('measures all 12 momentum coordinates from bar series and stays 0 for constant bars', () => {
      const mk = (t: number, d: number, s: number, r: number, bpm: number): MusicFeatures => ({
        torusDistance: t, dissonance: d, syncopation: s, restRatio: r, bpm,
      });
      const bars = [mk(0.1, 0.1, 0.2, 0.1, 80), mk(0.6, 0.5, 0.7, 0.4, 120), mk(0.3, 0.9, 0.1, 0.8, 100)];
      const z = musicFunctorForward({ ...bars[0]!, barFeatures: bars });
      expect(z.slice(12).filter((v) => v !== 0).length).toBe(12);
      const flat = mk(0.3, 0.2, 0.2, 0.2, 100);
      const zFlat = musicFunctorForward({ ...flat, barFeatures: [flat, flat, flat] });
      expect(zFlat.slice(12).every((v) => v === 0)).toBe(true);
    });

    it('image extractor yields patchFeatures and non-zero momentum on a textured image', () => {
      const w = 48, h = 48;
      const data = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          data[y * w + x] = Math.floor(((x * 7 + y * 13 + (x >> 4) * 90 + (y >> 4) * 40) % 256));
        }
      }
      const feats = extractVisualFeaturesFromImage({ width: w, height: h, data });
      expect(feats.patchFeatures?.length).toBe(9);
      const z = visualFunctorForward(feats);
      expect(z.slice(12).filter((v) => v > 0).length).toBeGreaterThanOrEqual(8);
    });

    it('note extractor yields per-bar features', () => {
      const notes: MusicNoteEvent[] = [];
      for (let i = 0; i < 24; i++) {
        notes.push({ pitch: 60 + ((i * 7) % 12), duration: i < 12 ? 0.5 : 0.1, velocity: 80, startTime: i * 0.6 });
      }
      const feats = extractMusicFeaturesFromNotes(notes);
      expect((feats.barFeatures?.length ?? 0)).toBeGreaterThanOrEqual(2);
    });

    it('music sequence: Void removes notes, deterministic, matches the Python implementation', () => {
      const mk = (rest: number, diss: number, synco = 0.3, bpm = 110): MusicFeatures => ({
        torusDistance: 0.4, dissonance: diss, syncopation: synco, restRatio: rest, bpm,
      });
      const dense = musicFunctorInverse(musicFunctorForward(mk(0.05, 0.3))).webAudioSequence;
      const sparse = musicFunctorInverse(musicFunctorForward(mk(0.9, 0.3))).webAudioSequence;
      expect(sparse.length).toBeLessThan(dense.length);
      expect(sparse.length).toBeGreaterThanOrEqual(1);

      // Reference values produced by python/usir_py/nishiki_unified_field.py for the same state.
      const z = musicFunctorForward(mk(0.05, 0.5, 0.8, 170));
      const seq = musicFunctorInverse(z).webAudioSequence;
      expect(seq.map((n) => n.note)).toEqual([65, 64, 65, 60, 60, 59, 57, 62, 64, 66, 64, 69]);
      expect(seq[0]!.time).toBeCloseTo(0.09627942044378301, 9);
      expect(seq[0]!.gain).toBeCloseTo(0.806258473896996, 9);
    });
  });
});
