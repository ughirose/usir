/**
 * Workstream 4 (Gap 6 & Gap 7) Vitest Verification Suite.
 *
 * Validates:
 * 1. Gap 6:
 *    - extractUiDom24d: UI/DOM component descriptors & HTML -> 24D symplectic state.
 *    - extractVisualSaccade24d: Visual fixations / saccade trajectory -> 24D symplectic state.
 *    - transpileCrossModal: 24D symplectic state -> music / diffusion / ui_css.
 *    - Invariant: III_Void maps to high rest_ratio / negative_space_ratio / whitespace_multiplier;
 *      IV_Turbulent / II_Critical maps to higher soliton_accent_gain / high_freq_noise_gain.
 * 2. Gap 7:
 *    - countSyllables: English Syllable Counter & Cadence.
 *    - 1-to-1 Structural Isomorphism:
 *      1. Hemingway minimalist prose -> III_Void (high Sg-03, Sg-07, Sg-11).
 *      2. Faulkner/Poe literary climax prose -> II_Critical (balanced high Sg-06, Sg-08, Sg-10, Sg-12).
 *      3. English legal statutes / mathematical axioms -> I_Crystalline (high Sg-02, Sg-05, low Sg-04, Sg-12).
 *      4. Purple-prose English -> IV_Turbulent.
 *      5. English AI Slop -> V_Deadzone.
 */

import { describe, expect, it } from 'vitest';
import {
  extractUiDom24d,
  extractVisualSaccade24d,
  transpileCrossModal,
  type UiDomNode,
  type VisualSaccadePatch,
  type MusicTranspiledParams,
  type DiffusionTranspiledParams,
  type UiCssTranspiledParams,
} from '../src/v2/crossModalFunctors.js';
import {
  classifyEnglishPhase,
  countSyllables,
  extractUsir24dEnglish,
} from '../src/v2/englishDriver.js';

describe('Workstream 4: Gap 6 & Gap 7 Tests', () => {
  // ==========================================================================
  // GAP 6: Universal Cross-Modal Functors & Synesthetic Transpiler Tests
  // ==========================================================================

  describe('Gap 6: Cross-Modal Functors & Synesthetic Transpiler', () => {
    it('verifies Functor F_ui extracts 24D symplectic state from UiDomNode descriptors', () => {
      const nodes: UiDomNode[] = [
        { depth: 1.0, whitespace_ratio: 0.4, font_scale_ratio: 2.5, card_uniformity: 0.9, contrast_jump: 0.1, padding_rhythm: 0.9 },
        { depth: 2.0, whitespace_ratio: 0.3, font_scale_ratio: 1.5, card_uniformity: 0.8, contrast_jump: 0.2, padding_rhythm: 0.8 },
        { depth: 2.0, whitespace_ratio: 0.5, font_scale_ratio: 1.0, card_uniformity: 0.9, contrast_jump: 0.1, padding_rhythm: 0.9 },
        { depth: 3.0, whitespace_ratio: 0.2, font_scale_ratio: 1.0, card_uniformity: 0.7, contrast_jump: 0.7, padding_rhythm: 0.7 },
        { depth: 3.0, whitespace_ratio: 0.6, font_scale_ratio: 0.9, card_uniformity: 0.9, contrast_jump: 0.1, padding_rhythm: 0.9 },
      ];

      const z = extractUiDom24d(nodes, true);
      expect(z.length).toBe(24);
      for (let i = 0; i < 12; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
        expect(z[i]!).toBeLessThanOrEqual(1.0);
      }
      for (let i = 12; i < 24; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('verifies Functor F_ui extracts 24D state directly from raw HTML layout string', () => {
      const html = `
      <div class="card p-4 m-2 shadow rounded grid">
        <h1 class="text-2xl font-bold mb-4">Dashboard Overview</h1>
        <p class="text-base text-gray-700 whitespace-pre">System status is operational.</p>
        <div class="card p-2 gap-4 border">
          <h2 class="text-xl">Metrics</h2>
          <span class="empty p-4"></span>
          <button class="btn bg-primary text-white p-2 rounded">Refresh</button>
        </div>
      </div>
      `;

      const z = extractUiDom24d(html, true);
      expect(z.length).toBe(24);
      for (let i = 0; i < 12; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
        expect(z[i]!).toBeLessThanOrEqual(1.0);
      }
      for (let i = 12; i < 24; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('verifies Functor F_visual extracts 24D symplectic state from visual fixations', () => {
      const patches: VisualSaccadePatch[] = [
        { luminance_contrast: 0.1, edge_density: 0.2, negative_space_ratio: 0.7, chromatic_jump: 0.05, x: 0.2, y: 0.2 },
        { luminance_contrast: 0.3, edge_density: 0.4, negative_space_ratio: 0.5, chromatic_jump: 0.10, x: 0.3, y: 0.4 },
        { luminance_contrast: 0.8, edge_density: 0.9, negative_space_ratio: 0.1, chromatic_jump: 0.75, x: 0.5, y: 0.5 },
        { luminance_contrast: 0.2, edge_density: 0.3, negative_space_ratio: 0.6, chromatic_jump: 0.10, x: 0.7, y: 0.6 },
        { luminance_contrast: 0.1, edge_density: 0.1, negative_space_ratio: 0.8, chromatic_jump: 0.05, x: 0.8, y: 0.8 },
      ];

      const z = extractVisualSaccade24d(patches, true);
      expect(z.length).toBe(24);
      for (let i = 0; i < 12; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
        expect(z[i]!).toBeLessThanOrEqual(1.0);
      }
      for (let i = 12; i < 24; i++) {
        expect(z[i]!).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('verifies Synesthetic Transpiler cross-modal parameter mappings and physical invariants', () => {
      // Prototypical III_Void state
      const zVoid = new Float64Array(24);
      zVoid[2] = 0.85;  // Sg-03 Pr
      zVoid[6] = 0.80;  // Sg-07 Cs
      zVoid[10] = 0.90; // Sg-11 Ma
      zVoid[0] = 0.15;
      zVoid[1] = 0.20;
      zVoid[3] = 0.05;
      zVoid[7] = 0.05;
      zVoid[11] = 0.05;

      // Prototypical II_Critical / IV_Turbulent state
      const zShock = new Float64Array(24);
      zShock[3] = 0.75;  // Sg-04 Jt
      zShock[7] = 0.80;  // Sg-08 Lp
      zShock[11] = 0.90; // Sg-12 Ru
      zShock[2] = 0.10;
      zShock[6] = 0.10;
      zShock[10] = 0.10;

      // 1. Music Modality
      const musicVoid = transpileCrossModal(zVoid, 'music') as MusicTranspiledParams;
      const musicShock = transpileCrossModal(zShock, 'music') as MusicTranspiledParams;

      expect(musicVoid.rest_ratio).toBeGreaterThan(0.60);
      expect(musicShock.rest_ratio).toBeLessThan(0.25);
      expect(musicShock.soliton_accent_gain).toBeGreaterThan(musicVoid.soliton_accent_gain);
      expect(musicShock.soliton_accent_gain).toBeGreaterThan(0.65);

      // 2. Diffusion Modality
      const diffVoid = transpileCrossModal(zVoid, 'diffusion') as DiffusionTranspiledParams;
      const diffShock = transpileCrossModal(zShock, 'diffusion') as DiffusionTranspiledParams;

      expect(diffVoid.negative_space_ratio).toBeGreaterThan(0.60);
      expect(diffShock.negative_space_ratio).toBeLessThan(0.30);
      expect(diffShock.high_freq_noise_gain).toBeGreaterThan(diffVoid.high_freq_noise_gain);
      expect(diffShock.high_freq_noise_gain).toBeGreaterThan(0.60);

      // 3. UI/CSS Modality
      const uiVoid = transpileCrossModal(zVoid, 'ui_css') as UiCssTranspiledParams;
      const uiShock = transpileCrossModal(zShock, 'ui_css') as UiCssTranspiledParams;

      expect(uiVoid.whitespace_multiplier).toBeGreaterThan(2.2);
      expect(uiShock.whitespace_multiplier).toBeLessThan(1.3);
      expect(uiShock.grid_irregularity).toBeGreaterThan(uiVoid.grid_irregularity);
    });
  });

  // ==========================================================================
  // GAP 7: English Isomorphic Driver (usir-driver-en) Tests
  // ==========================================================================

  describe('Gap 7: English Cross-Linguistic Isomorphic Driver', () => {
    it('verifies deterministic English syllable counter across various morphological patterns', () => {
      expect(countSyllables('stone')).toBe(1);
      expect(countSyllables('rain')).toBe(1);
      expect(countSyllables('cold')).toBe(1);
      expect(countSyllables('water')).toBe(2);
      expect(countSyllables('shadow')).toBe(2);
      expect(countSyllables('silence')).toBe(2);
      expect(countSyllables('pursuant')).toBe(3);
      expect(countSyllables('homomorphism')).toBe(5);
      expect(countSyllables('multifaceted')).toBe(5);
    });

    it('proves Hemingway-style minimalist English prose is classified into III_Void', () => {
      const text =
        'The river was green. Cold water ran over the white stones. ' +
        'Nick looked down through the dark pool and saw the trout. He did not move. ' +
        'The sun was hot on his back. There was no sound in the trees. ' +
        'The wind blew from the hills, cold and clean.';

      const z = extractUsir24dEnglish(text, false);
      const phase = classifyEnglishPhase(text);

      expect(phase).toBe('III_Void');
      // Sg-11 (Ma) dominant, Sg-03 (Pr), Sg-07 (Cs) high, exceeding shock
      expect(z[10]!).toBeGreaterThan(0.50);
      expect(z[2]!).toBeGreaterThan(0.08);
      expect(z[6]!).toBeGreaterThan(0.05);
      expect(z[10]!).toBeGreaterThan(z[11]!);
      expect(z[10]!).toBeGreaterThan(z[3]!);
    });

    it('proves Faulkner/Poe-style literary climax prose is classified into II_Critical', () => {
      const text =
        'He ran through the weeping darkness, the old house trembling as the storm broke with sudden fury. ' +
        'The shutter swung wide; a shriek ripped the silence of the night! ' +
        'Blood beat in his temples. ' +
        'He saw the black abyss yawning before him, and in that single terrifying instant of horror, ' +
        'the bell tolling above the ruins shattered his soul.';

      const z = extractUsir24dEnglish(text, false);
      const phase = classifyEnglishPhase(text);

      expect(phase).toBe('II_Critical');
      // Sg-10 (At), Sg-12 (Ru), Sg-04 (Jt) balanced high
      expect(z[9]!).toBeGreaterThan(0.30);
      expect(z[11]!).toBeGreaterThan(0.20);
      expect(z[7]!).toBeGreaterThan(0.08);
      expect(z[5]!).toBeGreaterThan(0.03);
      expect(z[3]!).toBeGreaterThan(0.15);
    });

    it('proves English legal statutes / mathematical axioms are classified into I_Crystalline', () => {
      const text =
        'Pursuant to Section 4.2 hereof, the Licensee shall indemnify and hold harmless the Licensor against any liabilities arising thereunder. ' +
        'Notwithstanding the foregoing, if two topological spaces are homeomorphic, ' +
        'then every compact subspace of the domain is mapped to a compact subspace of the codomain under the continuous homomorphism.';

      const z = extractUsir24dEnglish(text, false);
      const phase = classifyEnglishPhase(text);

      expect(phase).toBe('I_Crystalline');
      expect(z[1]!).toBeGreaterThan(0.25); // Sg-02 Cr
      expect(z[4]!).toBeGreaterThan(0.10); // Sg-05 Cp
      expect(z[3]!).toBeLessThan(0.15);    // Sg-04 Jt
      expect(z[11]!).toBeLessThan(0.10);   // Sg-12 Ru
    });

    it('proves purple-prose English with excessive modifiers is classified into IV_Turbulent', () => {
      const text =
        'The exceedingly iridescent, gossamer-draped twilight descended exceedingly sluggishly ' +
        'upon the bewilderingly ornate, staggeringly grandiose architecture of the profoundly ostentatious palace, ' +
        'casting bizarrely flickering, nauseatingly garish luminescence across every excessively carved, monstrously gilded gargoyle.';

      const phase = classifyEnglishPhase(text);
      expect(phase).toBe('IV_Turbulent');
    });

    it('proves English AI slop loaded with clichés and buzzwords is classified into V_Deadzone', () => {
      const text =
        'Moreover, it is crucial to delve into the rich tapestry of multifaceted considerations ' +
        'that underscores the pivotal importance of this comprehensive framework. ' +
        'In conclusion, by fostering seamless integration across diverse ecosystems, ' +
        'organizations can navigate the complexities of modern paradigms and elevate operational efficacy.';

      const phase = classifyEnglishPhase(text);
      expect(phase).toBe('V_Deadzone');
    });
  });
});
