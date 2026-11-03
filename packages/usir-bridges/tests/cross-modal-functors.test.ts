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
      expect(audioCalm.webAudioSequence.length).toBe(8);

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
  });
});
