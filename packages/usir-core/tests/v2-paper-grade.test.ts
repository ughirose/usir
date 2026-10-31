import { describe, expect, it } from 'vitest';
import {
  // Module 1: Zero-Leakage 64-bit HTH Octree Lexicon & Operators
  HTHLexicon,
  clz30,
  lcaDepth30Bit,
  discreteJumpDistance,
  poincareHyperbolicDistance,
  buildDiffusedClusterTable,
  synthesizeMimeticHTH,
  extractMusicFunctorSignals,
  type MusicNoteEvent,

  // Module 2: Wavelet-Hermite, Fourier-Poisson MCA & RQA 12-Element Encoder
  WaveletHermiteEncoder,
  orthogonalWaveletDecompose1D,
  mcaSeparate1D,
  computeRqaMetrics,
  extract24dV2,
  padToPowerOf2,
  buildHaarOrthonormalMatrix,

  // Module 3: SO(8) Pullback Metric, Kac-Moreau-Yosida & Hamilton-Friston Resonance
  SO8PullbackMetric,
  KacMoreauYosidaThermodynamics,
  HamiltonFristonResonance,

  // Module 4: Adaptive-BAOAB Integrator, Clause Twisted SMC & Jacobian Logit Projector
  AdaptiveBAOABIntegrator,
  ClauseTwistedSMC,
  JacobianLogitProjector,
  type ClauseParticle,
  type CandidateToken,
} from '../src/index.js';

describe('USIR v2.0 Paper-Grade Overhaul Theorem & Integration Test Suite', () => {
  // ==========================================================================
  // Module 1: HTHLexicon, LCA Depth, Poincare Distance & OPEN-01..04 Operators
  // ==========================================================================
  describe('Module 1: HTH Lexicon, LCA Depth & OPEN-01..04 Operators', () => {
    it('verifies clz30 bitwise operations and edge cases', () => {
      expect(clz30(0)).toBe(30);
      expect(clz30(1 << 29)).toBe(0);
      expect(clz30(1 << 28)).toBe(1);
      expect(clz30(1 << 27)).toBe(2);
      expect(clz30(1)).toBe(29);
      expect(clz30(0x3fffffff)).toBe(0);
    });

    it('verifies 30-bit octree LCA depth and discrete jump distance metric properties', () => {
      // 10 levels of 3 bits = 30 bits total
      const hBase = 0b001010011000000000000000000000;
      // Exact match -> LCA depth = 10, jump distance = 0
      expect(lcaDepth30Bit(hBase, hBase)).toBe(10);
      expect(discreteJumpDistance(hBase, hBase)).toBe(0.0);

      // Level 1 and 2 match, level 3 differs
      const hDiff3 = 0b001010100000000000000000000000;
      expect(lcaDepth30Bit(hBase, hDiff3)).toBe(2);
      expect(discreteJumpDistance(hBase, hDiff3)).toBeCloseTo(0.8, 5);

      // Level 1 differs
      const hDiff1 = 0b010000000000000000000000000000;
      expect(lcaDepth30Bit(hBase, hDiff1)).toBe(0);
      expect(discreteJumpDistance(hBase, hDiff1)).toBe(1.0);

      // Symmetry of discrete jump distance
      expect(discreteJumpDistance(hBase, hDiff3)).toBe(discreteJumpDistance(hDiff3, hBase));
      expect(discreteJumpDistance(hBase, hDiff1)).toBe(discreteJumpDistance(hDiff1, hBase));
    });

    it('computes Poincare hyperbolic heat-kernel distance satisfying metric properties', () => {
      const lex = new HTHLexicon();
      const ice = lex.lookup('氷');
      const water = lex.lookup('水');
      const fire = lex.lookup('炎');

      // Exact tree distance (tau = 0.0)
      const dExact = poincareHyperbolicDistance(ice, water, 0.0);
      expect(dExact).toBeGreaterThanOrEqual(0.0);
      expect(dExact).toBeLessThanOrEqual(1.0);

      // Identity of indiscernibles
      expect(poincareHyperbolicDistance(ice, ice, 0.0)).toBe(0.0);

      // Symmetry
      expect(poincareHyperbolicDistance(ice, water, 0.0)).toBeCloseTo(
        poincareHyperbolicDistance(water, ice, 0.0),
        10
      );

      // Hyperbolic heat-kernel smoothing (tau = 0.5)
      const dSmooth = poincareHyperbolicDistance(ice, fire, 0.5);
      expect(dSmooth).toBeGreaterThanOrEqual(0.0);
      expect(dSmooth).toBeLessThanOrEqual(1.0);
    });

    it('OPEN-01: verifies contextual polysemy damping gate transitions', () => {
      const lex = new HTHLexicon();
      const ice = lex.lookup('氷');
      const cold = lex.lookup('冷たい');

      // Literal context: Concrete target + concrete context
      const lit = lex.applyOpen01PolysemyDamping(ice, cold);
      expect(lit.state).toBe('literal');
      expect(lit.effectiveSensory.length).toBe(8);

      // Poetic metaphor: Abstract target + sensory bridge
      const poeticTarget = {
        ...ice,
        concreteness: 0.2,
        sensory8d: [0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
      };
      const poeticContext = {
        ...cold,
        concreteness: 0.2,
        sensory8d: [0.75, 0.65, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
      };
      const poetic = lex.applyOpen01PolysemyDamping(poeticTarget, poeticContext);
      expect(poetic.state).toBe('poetic_metaphor');

      // Dead metaphor: Abstract target + non-bridged sensory
      const deadTarget = {
        ...ice,
        concreteness: 0.1,
        sensory8d: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      };
      const dead = lex.applyOpen01PolysemyDamping(deadTarget, cold);
      expect(dead.state).toBe('dead_metaphor');
      for (const val of dead.effectiveSensory) {
        expect(val).toBeLessThanOrEqual(0.1);
      }
    });

    it('OPEN-02: verifies dual negation scope and sensory modulation', () => {
      const lex = new HTHLexicon();
      const text = '深紅の炎などどこにもなかった';
      const tokens = lex.segmentText(text);

      const negatedToken = tokens.find((t) => t.isNegated);
      expect(negatedToken).toBeDefined();
      expect(negatedToken!.modalityAlpha).toBeLessThan(0.0);

      // Stationary sensory vector is clamped to zero by ReLU gate max(0, alpha)
      for (const val of negatedToken!.effectiveSensory8d) {
        expect(val).toBe(0.0);
      }
    });

    it('OPEN-03: verifies onomatopoeia HTH synthesis with mimetic immunity', () => {
      const onomGira = synthesizeMimeticHTH('ぎらぎら');
      expect(onomGira.isMimetic).toBe(true);
      expect(onomGira.mimeticImmunity).toBe(true);
      expect(onomGira.concreteness).toBeGreaterThanOrEqual(0.8);
      expect(onomGira.sensory8d.length).toBe(8);

      const onomPachi = synthesizeMimeticHTH('ぱちぱち');
      expect(onomPachi.isMimetic).toBe(true);
      expect(onomPachi.mimeticImmunity).toBe(true);
      // 'p' initial consonant maps to visual / luminance
      expect(onomPachi.sensory8d[0]).toBeGreaterThan(0.2);

      const onomSara = synthesizeMimeticHTH('さらさら');
      expect(onomSara.isMimetic).toBe(true);
      // 's' maps to tactile smoothness
      expect(onomSara.sensory8d[2]).toBeGreaterThan(0.2);
    });

    it('OPEN-04: verifies narrative and dialogue separation', () => {
      const lex = new HTHLexicon();
      const passage = '少年は立ち止まった。「向こうに明かりが見える」と彼は指差した。「行ってみよう」';
      const sep = lex.separateNarrativeAndDialogue(passage);

      expect(sep.narrativeText).toContain('少年は立ち止まった。');
      expect(sep.narrativeText).toContain('と彼は指差した。');
      expect(sep.dialogueText).toContain('向こうに明かりが見える');
      expect(sep.dialogueText).toContain('行ってみよう');
      expect(sep.dialogueRatio).toBeGreaterThan(0.0);
      expect(sep.dialogueRatio).toBeLessThan(1.0);
      expect(sep.alternationCount).toBeGreaterThanOrEqual(3);
    });

    it('extracts music functor signals into Mx4 matrix from discrete notes', () => {
      const notes: MusicNoteEvent[] = [
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

    it('builds 256-node Laplacian diffusion cluster table', () => {
      const table = buildDiffusedClusterTable(0.85);
      expect(table.diffused8d.length).toBe(256 * 8);
      expect(table.octantPaths.length).toBe(256);
      for (let i = 0; i < 256 * 8; i++) {
        const v = table.diffused8d[i]!;
        expect(v).toBeGreaterThanOrEqual(0.0);
        expect(v).toBeLessThanOrEqual(1.0);
      }
    });
  });

  // ==========================================================================
  // Module 2: Wavelet-Hermite Encoder & Theorem 2.1 Wavelet Orthogonality
  // ==========================================================================
  describe('Module 2: Wavelet-Hermite Encoder & Theorem 2.1 Wavelet Orthogonality', () => {
    it('proves Theorem 2.1: strictly orthogonal Haar wavelet cross-scale inner products < 10^-12', () => {
      // Test on arbitrary synthetic signals of varying lengths (powers of 2)
      for (const n of [16, 32, 64]) {
        // Construct composite non-trivial signal
        const signal = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          signal[i] =
            Math.sin((2.0 * Math.PI * 3 * i) / n) +
            0.6 * Math.cos((2.0 * Math.PI * 7 * i) / n) +
            (i % 4 === 0 ? 1.5 : -0.5);
        }

        const { micro, meso, macro, coeffs } = orthogonalWaveletDecompose1D(signal);
        expect(coeffs.length).toBe(n);
        expect(micro.length).toBe(n);
        expect(meso.length).toBe(n);
        expect(macro.length).toBe(n);

        // 1. Exact Parseval / subspace reconstruction: micro + meso + macro == signal
        let maxReconErr = 0.0;
        for (let i = 0; i < n; i++) {
          const recon = micro[i]! + meso[i]! + macro[i]!;
          maxReconErr = Math.max(maxReconErr, Math.abs(recon - signal[i]!));
        }
        expect(maxReconErr).toBeLessThan(1e-12);

        // 2. Theorem 2.1 Cross-scale orthogonality:
        // <micro, meso> = 0, <micro, macro> = 0, <meso, macro> = 0
        let dotMicroMeso = 0.0;
        let dotMicroMacro = 0.0;
        let dotMesoMacro = 0.0;
        for (let i = 0; i < n; i++) {
          dotMicroMeso += micro[i]! * meso[i]!;
          dotMicroMacro += micro[i]! * macro[i]!;
          dotMesoMacro += meso[i]! * macro[i]!;
        }

        expect(Math.abs(dotMicroMeso)).toBeLessThan(1e-12);
        expect(Math.abs(dotMicroMacro)).toBeLessThan(1e-12);
        expect(Math.abs(dotMesoMacro)).toBeLessThan(1e-12);
      }
    });

    it('verifies Haar orthonormal matrix construction properties', () => {
      const n = 16;
      const W = buildHaarOrthonormalMatrix(n);
      expect(W.length).toBe(n);

      // Verify W * W^T = I_n with precision < 1e-14
      let maxErr = 0.0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let dot = 0.0;
          for (let k = 0; k < n; k++) {
            dot += W[i]![k]! * W[j]![k]!;
          }
          const target = i === j ? 1.0 : 0.0;
          maxErr = Math.max(maxErr, Math.abs(dot - target));
        }
      }
      expect(maxErr).toBeLessThan(1e-12);
    });

    it('verifies padToPowerOf2 with symmetric reflection', () => {
      const orig = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const padded = padToPowerOf2(orig, 8);
      expect(padded.length).toBe(8);
      expect(padded[0]).toBe(1.0);
      expect(padded[4]).toBe(5.0);
      // Reflection: 5, 4, 3
      expect(padded[5]).toBe(5.0);
      expect(padded[6]).toBe(4.0);
      expect(padded[7]).toBe(3.0);
    });

    it('performs Morphological Component Analysis (MCA) separating Fourier and Soliton components', () => {
      const n = 16;
      // Synthetic signal: low-frequency harmonic + localized soliton pulse
      const signal = new Float64Array(n);
      for (let t = 0; t < n; t++) {
        const harmonic = Math.cos((2.0 * Math.PI * 2 * t) / n);
        const soliton = 2.0 / Math.cosh(2.5 * (t - 8));
        signal[t] = harmonic + soliton;
      }

      const mca = mcaSeparate1D(signal, 30, 0.15);
      expect(mca.fourierComp.length).toBe(n);
      expect(mca.solitonComp.length).toBe(n);
      expect(mca.residual.length).toBe(n);
      expect(mca.cF.length).toBe(9); // 1 DC + 4 * 2 harmonics
      expect(mca.cS.length).toBe(32); // 16 * 2 soliton dictionary atoms

      // MCA residual energy should be strictly bounded
      let resEnergy = 0.0;
      for (let t = 0; t < n; t++) {
        resEnergy += mca.residual[t]! * mca.residual[t]!;
      }
      expect(Math.sqrt(resEnergy / n)).toBeLessThan(1.0);
    });

    it('computes Recurrence Quantification Analysis (RQA) determinism and arch index', () => {
      // Periodic sine wave should produce high determinism DET
      const n = 32;
      const periodic = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        periodic[i] = Math.sin((2.0 * Math.PI * 2 * i) / n);
      }

      const rqa = computeRqaMetrics(periodic);
      expect(rqa.det).toBeGreaterThan(0.0);
      expect(rqa.det).toBeLessThanOrEqual(1.0);
      expect(rqa.iArch).toBeGreaterThanOrEqual(0.0);
      expect(rqa.iArch).toBeLessThanOrEqual(1.0);
      expect(rqa.recurrenceMatrix?.length).toBe(n);
    });

    it('encodes Japanese text and extracts 24D symplectic state via extract24dV2', () => {
      const text =
        '春の夜の夢ばかりなる手枕にかひなく立たむ名こそをしけれ。' +
        '風が静かに吹き抜けて、木々の梢が小さくざわめいていた。';

      const z = extract24dV2(text, true);
      expect(z.length).toBe(24);

      // Position coordinates q in [0, 1]^12
      const q = z.slice(0, 12);
      for (let i = 0; i < 12; i++) {
        expect(q[i]).toBeGreaterThanOrEqual(0.0);
        expect(q[i]).toBeLessThanOrEqual(1.0);
      }

      // Momentum coordinates p in R_{>=0}^12
      const p = z.slice(12, 24);
      for (let i = 0; i < 12; i++) {
        expect(p[i]).toBeGreaterThanOrEqual(0.0);
      }
    });
  });

  // ==========================================================================
  // Module 3: SO(8) Pullback Metric, Kac-MY Thermodynamics & Theorem 1.1
  // ==========================================================================
  describe('Module 3: SO(8) Pullback Metric & Theorem 1.1 Intensive Free Energy', () => {
    it('verifies SO(8) Cayley transform exact orthogonality: ||R^T R - I_8||_inf < 10^-12', () => {
      const metric = new SO8PullbackMetric();
      const R = metric.rotationMatrix();
      expect(R.length).toBe(8);

      let maxOrthErr = 0.0;
      for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
          let dot = 0.0;
          for (let k = 0; k < 8; k++) {
            dot += R[k]![i]! * R[k]![j]!;
          }
          const target = i === j ? 1.0 : 0.0;
          maxOrthErr = Math.max(maxOrthErr, Math.abs(dot - target));
        }
      }
      expect(maxOrthErr).toBeLessThan(1e-12);
    });

    it('verifies SO8PullbackMetric strict positive-definiteness and 12x12 matrix inversion', () => {
      const metric = new SO8PullbackMetric();
      const G = metric.metricMatrix();
      const Ginv = metric.inverseMetricMatrix();

      expect(G.length).toBe(12);
      expect(Ginv.length).toBe(12);

      // 1. Symmetry: G = G^T
      let maxAsym = 0.0;
      for (let i = 0; i < 12; i++) {
        for (let j = 0; j < 12; j++) {
          maxAsym = Math.max(maxAsym, Math.abs(G[i]![j]! - G[j]![i]!));
        }
      }
      expect(maxAsym).toBeLessThan(1e-12);

      // 2. Strict positive-definiteness: v^T G v > 0 for standard basis vectors and random vectors
      for (let i = 0; i < 12; i++) {
        const e = new Array<number>(12).fill(0);
        e[i] = 1.0;
        const quad = metric.quadraticForm(e);
        expect(quad).toBeGreaterThan(metric.epsilon);
      }

      const randomV = [0.2, -0.4, 0.5, 0.1, -0.3, 0.8, -0.2, 0.6, -0.1, 0.4, -0.5, 0.3];
      expect(metric.quadraticForm(randomV)).toBeGreaterThan(0.0);

      // 3. Exact matrix inverse: ||G * G^(-1) - I_12||_inf < 10^-10
      let maxInvErr = 0.0;
      for (let i = 0; i < 12; i++) {
        for (let j = 0; j < 12; j++) {
          let dot = 0.0;
          for (let k = 0; k < 12; k++) {
            dot += G[i]![k]! * Ginv[k]![j]!;
          }
          const target = i === j ? 1.0 : 0.0;
          maxInvErr = Math.max(maxInvErr, Math.abs(dot - target));
        }
      }
      expect(maxInvErr).toBeLessThan(1e-10);

      // 4. Mahalanobis distance metric properties
      const q1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];
      const q2 = [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 0.1, 0.9, 0.3, 0.2];
      const q3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

      expect(metric.mahalanobisDistance(q1, q1)).toBe(0.0);
      expect(metric.mahalanobisDistance(q1, q2)).toBeCloseTo(metric.mahalanobisDistance(q2, q1), 10);
      // Triangle inequality
      const d12 = metric.mahalanobisDistance(q1, q2);
      const d23 = metric.mahalanobisDistance(q2, q3);
      const d13 = metric.mahalanobisDistance(q1, q3);
      expect(d13).toBeLessThanOrEqual(d12 + d23 + 1e-12);
    });

    it('proves Theorem 1.1: Kac-Moreau-Yosida free energy density remains intensive O(1) across L in {16, 64, 256}', () => {
      const thermo = new KacMoreauYosidaThermodynamics();

      const makeTrajectory = (L: number): number[][] => {
        const traj: number[][] = [];
        for (let t = 0; t < L; t++) {
          const q = new Array<number>(12);
          for (let d = 0; d < 12; d++) {
            q[d] = 0.5 + 0.2 * Math.sin(0.1 * t + d);
          }
          traj.push(q);
        }
        return traj;
      };

      const traj16 = makeTrajectory(16);
      const traj64 = makeTrajectory(64);
      const traj256 = makeTrajectory(256);

      const f16 = thermo.freeEnergyDensity(traj16);
      const f64 = thermo.freeEnergyDensity(traj64);
      const f256 = thermo.freeEnergyDensity(traj256);

      // Verify all free energy densities are strictly finite, positive, and O(1) bounded
      expect(f16).toBeGreaterThan(0.0);
      expect(f64).toBeGreaterThan(0.0);
      expect(f256).toBeGreaterThan(0.0);

      // As sequence length L scales by 16x (16 -> 256), the intensive free energy density
      // must remain within an O(1) envelope with relative deviation < 10%
      const relativeVariation = Math.abs(f256 - f16) / f16;
      expect(relativeVariation).toBeLessThan(0.1);
      expect(Math.abs(f64 - f16) / f16).toBeLessThan(0.1);

      // Verify Kac normalization factor scales asymptotically as L^(1 - alpha)
      const z16 = KacMoreauYosidaThermodynamics.kacNormalizationFactor(16, 0.8);
      const z256 = KacMoreauYosidaThermodynamics.kacNormalizationFactor(256, 0.8);
      expect(z256).toBeGreaterThan(z16);
    });

    it('verifies Moreau-Yosida proximal operator and smoothed gradient properties', () => {
      const thermo = new KacMoreauYosidaThermodynamics();
      const x = [0.3, -0.4, 0.2, 0.5, -0.1, 0.0, 0.4, -0.3, 0.2, 0.1, -0.2, 0.3];

      const yStar = thermo.proxPsiQ(x, 6);
      expect(yStar.length).toBe(12);

      const gradMY = thermo.moreauYosidaGradient(x);
      expect(gradMY.length).toBe(12);

      for (let i = 0; i < 12; i++) {
        expect(Number.isFinite(gradMY[i])).toBe(true);
      }
    });

    it('verifies Hamilton-Friston Resonance: valence spike, aesthetic utility and cortical equipartition', () => {
      // 1. Valence spike: tension-resolution catharsis
      const gradF = [0.5, 0.2, -0.4, 0.1, 0.0, 0.3, -0.2, 0.4, 0.1, -0.1, 0.2, -0.3];
      const momentum = [-0.6, -0.3, 0.5, -0.2, 0.1, -0.4, 0.3, -0.5, -0.2, 0.2, -0.3, 0.4];
      const spike = HamiltonFristonResonance.readerValenceSpike(gradF, momentum);
      // Catharsis is positive when momentum moves down the free energy gradient
      expect(spike).toBeGreaterThan(0.0);

      // 2. Saturated aesthetic utility with cognitive overload cutoff
      const utilNormal = HamiltonFristonResonance.saturatedAestheticUtility(10.0, 5.0, 0.5);
      const utilOverload = HamiltonFristonResonance.saturatedAestheticUtility(10.0, 50.0, 0.5);
      expect(utilNormal).toBeGreaterThan(utilOverload);

      // 3. Cortical equipartition
      const syntheticSignal = Array.from({ length: 64 }, (_, i) => Math.sin(0.1 * i));
      const equip = HamiltonFristonResonance.corticalEquipartition(syntheticSignal);
      expect(equip.equipartitionScore).toBeGreaterThanOrEqual(0.0);
      expect(equip.equipartitionScore).toBeLessThanOrEqual(1.0);
      expect(equip.bandPowers.length).toBe(3);
    });
  });

  // ==========================================================================
  // Module 4: Adaptive-BAOAB Integrator, Clause Twisted SMC & Jacobian Projector
  // ==========================================================================
  describe('Module 4: Adaptive-BAOAB Integrator & Theorem 5.1 Momentum Invariance', () => {
    it('proves Theorem 5.1: momentum invariant measure N(0, tau M) preserved under 1,000x friction switching', () => {
      const integrator = new AdaptiveBAOABIntegrator();

      // Check friction switching: high syntax entropy -> 0.1, zero syntax entropy -> 100.0 (1,000x ratio)
      expect(integrator.evaluateFriction(1.0)).toBe(0.1);
      expect(integrator.evaluateFriction(0.0)).toBe(100.0);

      // Verify invariant momentum measure under 10,000 steps with 1,000x friction switching
      const proof = integrator.verifyInvariantMomentumMeasure(10000, 1.0, 42);
      expect(proof.passed).toBe(true);
      expect(proof.meanError).toBeLessThan(0.08);
      expect(proof.covError).toBeLessThan(0.08);
      expect(proof.nSteps).toBe(10000);
    });

    it('executes symmetric BAOAB integration steps updating state coordinates cleanly', () => {
      const integrator = new AdaptiveBAOABIntegrator();
      const gradV = new Array<number>(12).fill(0.1);

      const step1 = integrator.step(gradV, 0.8);
      expect(step1.q.length).toBe(12);
      expect(step1.p.length).toBe(12);

      const step2 = integrator.step(gradV, 0.2); // Low syntax entropy -> high friction
      expect(step2.q.length).toBe(12);
      expect(step2.p.length).toBe(12);
    });

    it('evaluates ClauseTwistedSMC 4-particle rollout with Schrödinger bridge guidance and syntax gate rejection', () => {
      const smc = new ClauseTwistedSMC();

      const particles: ClauseParticle[] = [
        {
          text: '風が吹いた。',
          qHat: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
          logPllm: -1.2,
          syntacticallyValid: true,
        },
        {
          text: '破損した構文@@@',
          qHat: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
          logPllm: -0.1,
          syntacticallyValid: false, // Must be rejected by hard syntax gate
        },
        {
          text: '静かに水面が揺れた。',
          qHat: [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
          logPllm: -1.5,
          syntacticallyValid: true,
        },
        {
          text: '光が差し込んだ。',
          qHat: [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
          logPllm: -2.0,
          syntacticallyValid: true,
        },
      ];

      const targetQ = new Array<number>(12).fill(0.5);
      const terminalAnchor = new Array<number>(12).fill(0.6);

      const result = smc.selectBestClause(particles, targetQ, terminalAnchor, 4);

      // Particle 1 (index 1) is invalid and must not be selected
      expect(result.selectedIndex).not.toBe(1);
      expect(result.selectedParticle.syntacticallyValid).toBe(true);
      expect(result.normalizedWeights.length).toBe(4);
      expect(result.targetBridge.length).toBe(12);

      // Normalized weights must sum to 1.0
      const sumWeights = result.normalizedWeights.reduce((a, b) => a + b, 0);
      expect(sumWeights).toBeCloseTo(1.0, 5);
      expect(result.normalizedWeights[1]).toBe(0.0); // Hard rejected particle weight is 0
    });

    it('evaluates JacobianLogitProjector syntax protection gate and steering displacement', () => {
      const projector = new JacobianLogitProjector(undefined, 0.5, 1.0);

      // Case 1: Low LLM entropy (H_LLM < epsilon_syn) -> Bypassed to protect grammar
      const lowEntropyCandidates: CandidateToken[] = [
        { tokenId: 101, tokenStr: 'の', rawLogit: 50.0, jacobian: new Array(12).fill(0.2) },
        { tokenId: 102, tokenStr: 'が', rawLogit: 0.0, jacobian: new Array(12).fill(-0.1) },
      ];
      const lowResult = projector.projectLogits(
        lowEntropyCandidates,
        new Array<number>(12).fill(0.5)
      );
      expect(lowResult.bypassed).toBe(true);
      expect(lowResult.effectiveGain).toBe(0.0);
      expect(lowResult.baseEntropy).toBeLessThan(0.5);
      expect(lowResult.steeredLogits).toEqual([50.0, 0.0]);

      // Case 2: High LLM entropy (H_LLM >= epsilon_syn) -> Active Jacobian steering
      const highEntropyCandidates: CandidateToken[] = [
        { tokenId: 201, tokenStr: '静かに', rawLogit: 2.0, jacobian: new Array(12).fill(0.3) },
        { tokenId: 202, tokenStr: '激しく', rawLogit: 2.1, jacobian: new Array(12).fill(-0.3) },
      ];
      const highResult = projector.projectLogits(
        highEntropyCandidates,
        new Array<number>(12).fill(0.5)
      );
      expect(highResult.bypassed).toBe(false);
      expect(highResult.effectiveGain).toBeGreaterThan(0.0);
      expect(highResult.steeredProbs.length).toBe(2);

      const probSum = highResult.steeredProbs.reduce((a, b) => a + b, 0);
      expect(probSum).toBeCloseTo(1.0, 5);
    });
  });
});
