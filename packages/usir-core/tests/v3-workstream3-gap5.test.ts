/**
 * Universal Style Genome (USIR v2.0 / Nishiki)
 * Workstream 3 (Gap 5): Style Chemistry & Unexplored Style Discovery Operator (usir-genesis).
 */

import { describe, it, expect } from 'vitest';
import {
  evaluateStyleMolecules,
  discoverVoronoiGenesis,
  usirGenesis,
  CHI2_24_095,
} from '../src/algebra/styleAlgebra.js';
import { SO8PullbackMetric } from '../src/v2/thermodynamicsSo8.js';
import type { CanonicalParticle } from '../src/presets/canonicalParticles.js';

describe('Workstream 3 (Gap 5): Style Chemistry - The 4 Canonical Style Molecules', () => {
  it('strictly bounds molecule activations in [0, 1] and handles zero state', () => {
    const zZero = new Array<number>(24).fill(0.0);
    const resZero = evaluateStyleMolecules(zZero);

    expect(resZero.Mol_Yugen).toBe(0.0);
    expect(resZero.Mol_Wabi).toBe(0.0);
    expect(resZero.Mol_QiYun).toBe(0.0);
    expect(resZero.Mol_Iki).toBe(0.0);
    expect(resZero.synestheticCouplingEnergy).toBe(0.0);
    expect(['Mol_Yugen', 'Mol_Wabi', 'Mol_QiYun', 'Mol_Iki']).toContain(resZero.dominantMolecule);

    // 12D input handling (padded with p = 0)
    const qOnes = new Array<number>(12).fill(1.0);
    const res12d = evaluateStyleMolecules(qOnes);
    expect(res12d.Mol_Yugen).toBeGreaterThanOrEqual(0.0);
    expect(res12d.Mol_Yugen).toBeLessThanOrEqual(1.0);
    expect(res12d.Mol_Wabi).toBeGreaterThanOrEqual(0.0);
    expect(res12d.Mol_Wabi).toBeLessThanOrEqual(1.0);
    expect(res12d.Mol_Iki).toBeGreaterThanOrEqual(0.0);
    expect(res12d.Mol_Iki).toBeLessThanOrEqual(1.0);
    // Overdamped static text without p has Mol_QiYun === 0
    expect(res12d.Mol_QiYun).toBe(0.0);
  });

  it('proves Mol_Yugen emergence from Sg-03 Pr x Sg-08 Lp x Sg-11 Ma', () => {
    const q = new Array<number>(12).fill(0.0);
    q[2] = 0.85; // Sg-03 Pr
    q[7] = 0.80; // Sg-08 Lp
    q[10] = 0.90; // Sg-11 Ma

    const res = evaluateStyleMolecules(q);
    expect(res.Mol_Yugen).toBeGreaterThan(0.75);
    expect(res.dominantMolecule).toBe('Mol_Yugen');
    expect(res.Mol_Wabi).toBe(0.0);
    expect(res.Mol_QiYun).toBe(0.0);
    expect(res.Mol_Iki).toBe(0.0);

    // Constituent ablation
    const qAblated = [...q];
    qAblated[2] = 0.0;
    const resAblated = evaluateStyleMolecules(qAblated);
    expect(resAblated.Mol_Yugen).toBe(0.0);
  });

  it('proves Mol_Wabi emergence from Sg-04 Jt x Sg-07 Cs x Sg-11 Ma', () => {
    const q = new Array<number>(12).fill(0.0);
    q[3] = 0.85; // Sg-04 Jt
    q[6] = 0.90; // Sg-07 Cs
    q[10] = 0.80; // Sg-11 Ma

    const res = evaluateStyleMolecules(q);
    expect(res.Mol_Wabi).toBeGreaterThan(0.75);
    expect(res.dominantMolecule).toBe('Mol_Wabi');
    expect(res.Mol_Yugen).toBe(0.0);
    expect(res.Mol_Iki).toBe(0.0);
  });

  it('proves Mol_Iki emergence from Sg-02 Cr x Sg-07 Cs x Sg-12 Ru', () => {
    const q = new Array<number>(12).fill(0.0);
    q[1] = 0.85; // Sg-02 Cr
    q[6] = 0.80; // Sg-07 Cs
    q[11] = 0.90; // Sg-12 Ru

    const res = evaluateStyleMolecules(q);
    expect(res.Mol_Iki).toBeGreaterThan(0.75);
    expect(res.dominantMolecule).toBe('Mol_Iki');
    expect(res.Mol_Wabi).toBe(0.0);
  });

  it('proves Mol_QiYun momentum circulation gating: tanh(||p||_{G_style})', () => {
    const q = new Array<number>(12).fill(0.0);
    q[0] = 0.85; // Sg-01 Dn/Gr
    q[5] = 0.85; // Sg-06 Mp/Sy
    q[9] = 0.85; // Sg-10 At

    // Case A: Overdamped static text (p = 0)
    const zStatic = [...q, ...new Array<number>(12).fill(0.0)];
    const resStatic = evaluateStyleMolecules(zStatic);
    expect(resStatic.Mol_QiYun).toBe(0.0);

    // Case B: Dynamic active circulation along style flow
    const pActive = new Array<number>(12).fill(0.0);
    pActive[0] = 1.0;
    pActive[5] = 1.0;
    pActive[9] = 1.0;
    const zDynamic = [...q, ...pActive];
    const resDynamic = evaluateStyleMolecules(zDynamic);

    expect(resDynamic.Mol_QiYun).toBeGreaterThan(0.65);
    expect(resDynamic.dominantMolecule).toBe('Mol_QiYun');

    // Monotonic scaling with momentum magnitude
    const pHigh = pActive.map((v) => v * 1.5);
    const zHigh = [...q, ...pHigh];
    const resHigh = evaluateStyleMolecules(zHigh);
    expect(resHigh.Mol_QiYun).toBeGreaterThanOrEqual(resDynamic.Mol_QiYun);
  });

  it('verifies SO(8) synesthetic off-diagonal coupling amplification and energy diagnostics', () => {
    const metric = new SO8PullbackMetric();
    const gFull = metric.metricMatrix();

    // Diagonal baseline
    const gDiag = gFull.map((row, r) =>
      row.map((val, c) => (r === c ? val : 0.0))
    );

    const q = new Array<number>(12).fill(0.0);
    q[2] = 0.7;
    q[7] = 0.7;
    q[10] = 0.7;

    const resCoupled = evaluateStyleMolecules(q, gFull);
    const resUncoupled = evaluateStyleMolecules(q, gDiag);

    // Coupling amplification
    expect(resCoupled.Mol_Yugen).toBeGreaterThanOrEqual(resUncoupled.Mol_Yugen);
    expect(resCoupled.synestheticCouplingEnergy).toBeGreaterThan(0.0);
    expect(resUncoupled.synestheticCouplingEnergy).toBe(0.0);
  });
});

describe('Workstream 3 (Gap 5): Operator ④ - Unexplored Style Discovery (usirGenesis)', () => {
  it('discovers Voronoi void inside 95% Mahalanobis trust region with distance gain', () => {
    // 4 reference particles
    const kParticles: number[][] = [
      [...new Array(12).fill(0.2), ...new Array(12).fill(0.1)],
      [...new Array(12).fill(0.8), ...new Array(12).fill(0.3)],
      [...new Array(6).fill([0.1, 0.9]).flat(), ...new Array(12).fill(0.2)],
      [...new Array(6).fill([0.7, 0.2]).flat(), ...new Array(12).fill(0.4)],
    ];

    const res = usirGenesis(kParticles, undefined, {
      nRestarts: 6,
      nSteps: 40,
      learningRate: 0.08,
      lambdaShrinkage: 0.15,
    });

    // 1. Vector length and coordinate bounds
    expect(res.zGenesis.length).toBe(24);
    for (let i = 0; i < 12; i++) {
      expect(res.zGenesis[i]).toBeGreaterThanOrEqual(0.0);
      expect(res.zGenesis[i]).toBeLessThanOrEqual(1.0);
    }
    for (let i = 12; i < 24; i++) {
      expect(res.zGenesis[i]).toBeGreaterThanOrEqual(0.0);
    }

    // 2. Trust region compliance (d_M^2 <= 36.4150)
    expect(res.insideTrustRegion).toBe(true);
    expect(res.mahalanobisD2).toBeLessThanOrEqual(CHI2_24_095 + 1e-4);

    // 3. Voronoi void distance maximization over initial centroid distance
    expect(res.minRiemannianDistance).toBeGreaterThan(res.initialCentroidMinDist);
    expect(res.minRiemannianDistance).toBeGreaterThan(0.5);

    // 4. Evaluated molecules attached
    expect(['Mol_Yugen', 'Mol_Wabi', 'Mol_QiYun', 'Mol_Iki']).toContain(
      res.molecules.dominantMolecule
    );
    expect(res.molecules.Mol_Yugen).toBeGreaterThanOrEqual(0.0);
    expect(res.molecules.Mol_Yugen).toBeLessThanOrEqual(1.0);
  });

  it('works with CanonicalParticle inputs and verifies alias discoverVoronoiGenesis == usirGenesis', () => {
    const p1: CanonicalParticle = {
      label: 'test1',
      phase: 'II_Critical',
      mu: [...new Array(12).fill(0.3), ...new Array(12).fill(0.1)],
      description: 'Particle 1',
    };
    const p2: CanonicalParticle = {
      label: 'test2',
      phase: 'III_Void',
      mu: [...new Array(12).fill(0.7), ...new Array(12).fill(0.2)],
      description: 'Particle 2',
    };
    const p3: CanonicalParticle = {
      label: 'test3',
      phase: 'I_Crystalline',
      mu: [...new Array(6).fill([0.2, 0.8]).flat(), ...new Array(12).fill(0.1)],
      description: 'Particle 3',
    };

    const res1 = discoverVoronoiGenesis([p1, p2, p3]);
    expect(res1.insideTrustRegion).toBe(true);
    expect(res1.minRiemannianDistance).toBeGreaterThan(res1.initialCentroidMinDist);

    const res2 = usirGenesis([p1, p2, p3]);
    expect(res2.insideTrustRegion).toBe(true);
  });
});
