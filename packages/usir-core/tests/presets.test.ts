import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  CANONICAL_PARTICLES,
  mixStyleParticles,
} from '../src/presets/canonicalParticles.js';
import { fitPhaseClassifier, predictPhase } from '../src/dynamics/landauPhase.js';

describe('TASK-361: Canonical Style Particles & Product-of-Experts Mixer', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: any[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

  const classifier = fitPhaseClassifier(
    fixtureData.map((item) => ({
      phase: item.phase,
      features: item.z,
    }))
  );

  it('classifies all 4 canonical particles into their designated thermodynamic phases', () => {
    // 1. PORCELAIN_CRYSTAL -> I_Crystalline
    const crystal = CANONICAL_PARTICLES['PORCELAIN_CRYSTAL']!;
    const predCrystal = predictPhase(crystal.mu, classifier);
    expect(predCrystal.phase).toBe('I_Crystalline');

    // 2. AMBER_CRITICAL -> II_Critical
    const amber = CANONICAL_PARTICLES['AMBER_CRITICAL']!;
    const predAmber = predictPhase(amber.mu, classifier);
    expect(predAmber.phase).toBe('II_Critical');

    // 3. OBSIDIAN_VOID -> III_Void
    const voidP = CANONICAL_PARTICLES['OBSIDIAN_VOID']!;
    const predVoid = predictPhase(voidP.mu, classifier);
    expect(predVoid.phase).toBe('III_Void');

    // 4. LIGHTNING_RUPTURE -> IV_Turbulent
    const rupture = CANONICAL_PARTICLES['LIGHTNING_RUPTURE']!;
    const predRupture = predictPhase(rupture.mu, classifier);
    expect(predRupture.phase).toBe('IV_Turbulent');
  });

  it('mixes PORCELAIN_CRYSTAL and OBSIDIAN_VOID smoothly via PoE Gaussian precision weighting', () => {
    const p1 = CANONICAL_PARTICLES['PORCELAIN_CRYSTAL']!;
    const p2 = CANONICAL_PARTICLES['OBSIDIAN_VOID']!;

    const mixed50 = mixStyleParticles([p1, p2], [0.5, 0.5]);
    expect(mixed50.mu_mix.length).toBe(24);
    expect(mixed50.precision_mix.length).toBe(24);

    // Each element of mu_mix must lie between or near p1 and p2 centroids
    for (let d = 0; d < 24; d++) {
      const minVal = Math.min(p1.mu[d]!, p2.mu[d]!);
      const maxVal = Math.max(p1.mu[d]!, p2.mu[d]!);
      expect(mixed50.mu_mix[d]!).toBeGreaterThanOrEqual(minVal - 1e-4);
      expect(mixed50.mu_mix[d]!).toBeLessThanOrEqual(maxVal + 1e-4);
    }

    // Skewed mixture: 90% Void, 10% Crystal -> should classify as III_Void
    const mixedVoid = mixStyleParticles([p1, p2], [0.1, 0.9]);
    const predMixedVoid = predictPhase(mixedVoid.mu_mix, classifier);
    expect(predMixedVoid.phase).toBe('III_Void');

    // Skewed mixture: 90% Crystal, 10% Void -> should classify as I_Crystalline
    const mixedCrystal = mixStyleParticles([p1, p2], [0.9, 0.1]);
    const predMixedCrystal = predictPhase(mixedCrystal.mu_mix, classifier);
    expect(predMixedCrystal.phase).toBe('I_Crystalline');
  });
});
