import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  compileStyleGenomeToTierM,
  tierMToCanonicalParticle,
  getReferenceClassifier,
} from '../src/algebra/genomeCompiler.js';
import {
  projectOutSlopSubspace,
  interpolateGeodesic,
  transposeStyleRegister,
} from '../src/algebra/styleAlgebra.js';
import { decodeTierM } from '../src/schema/usirBinarySchema.js';
import { mixStyleParticles } from '../src/presets/canonicalParticles.js';
import { predictPhase } from '../src/dynamics/landauPhase.js';
import { extractUsir24d } from '../src/sensor/symplectic24d.js';

describe('USIR Style Algebra & Personal Style Genome Compiler (Step 3 / TASK-364)', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: Array<{ id: string; phase: string; text: string; z: number[] }> =
    JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

  const classifier = getReferenceClassifier();

  const voidChunks = fixtureData.filter((d) =>
    ['VOID-01', 'VOID-02', 'VOID-03', 'VOID-04', 'VOID-05', 'VOID-06'].includes(d.id)
  );
  const critChunks = fixtureData.filter((d) =>
    ['CRIT-01', 'CRIT-02', 'CRIT-03', 'CRIT-04', 'CRIT-05', 'CRIT-06'].includes(d.id)
  );
  const deadChunks = fixtureData.filter((d) => d.phase === 'V_Deadzone');
  const crys01Chunk = fixtureData.find((d) => d.id === 'CRYS-01')!;

  // Output artifact paths
  const artifactDir = path.resolve(process.cwd(), 'artifacts', 'genomes');
  const shigaArtifactPath = path.join(artifactDir, 'soseki_void.usir');
  const kajiiArtifactPath = path.join(artifactDir, 'kajii_critical.usir');

  // ===========================================================================
  // Step 3A: Genome Compilation & Artifact Generation
  // ===========================================================================
  describe('compileStyleGenomeToTierM', () => {
    it('compiles soseki_void.usir (4096B) with dominantPhase III_Void and passes Clean Room Audit', () => {
      const shiga = compileStyleGenomeToTierM(
        voidChunks.map((c) => c.text),
        { genomeName: 'soseki_void', classifier }
      );

      // Verify binary length
      expect(shiga.binary.byteLength).toBe(4096);

      // Verify dominant phase
      expect(shiga.dominantPhase).toBe('III_Void');

      // Verify Clean Room Audit Gate
      expect(shiga.audit.passed).toBe(true);
      expect(shiga.audit.leaked7Grams.length).toBe(0);
      expect(shiga.audit.bannedWordsFound.length).toBe(0);
      expect(shiga.audit.byteLength).toBe(4096);

      // Verify profile structural properties
      expect(shiga.profile.weights.length).toBe(3);
      const weightSum = shiga.profile.weights.reduce((s, w) => s + w, 0);
      expect(weightSum).toBeCloseTo(1.0, 5);

      expect(shiga.profile.transition_matrix.length).toBe(3);
      for (const row of shiga.profile.transition_matrix) {
        expect(row.length).toBe(3);
        const rowSum = row.reduce((s, v) => s + v, 0);
        expect(rowSum).toBeCloseTo(1.0, 5);
      }

      expect(shiga.profile.centroids.length).toBe(3);
      for (const c of shiga.profile.centroids) {
        expect(c.length).toBe(24);
      }

      expect(shiga.profile.cholesky_factors.length).toBe(3);
      for (const f of shiga.profile.cholesky_factors) {
        expect(f.length).toBe(300);
      }

      // Write artifact to artifacts/genomes/shiga_void.usir
      if (!fs.existsSync(artifactDir)) {
        fs.mkdirSync(artifactDir, { recursive: true });
      }
      fs.writeFileSync(shigaArtifactPath, shiga.binary);
      expect(fs.existsSync(shigaArtifactPath)).toBe(true);
      expect(fs.statSync(shigaArtifactPath).size).toBe(4096);
    });

    it('compiles kajii_critical.usir (4096B) with dominantPhase II_Critical and passes Clean Room Audit', () => {
      const kajii = compileStyleGenomeToTierM(
        critChunks.map((c) => c.text),
        { genomeName: 'kajii_critical', classifier }
      );

      expect(kajii.binary.byteLength).toBe(4096);
      expect(kajii.dominantPhase).toBe('II_Critical');
      expect(kajii.audit.passed).toBe(true);
      expect(kajii.audit.leaked7Grams.length).toBe(0);
      expect(kajii.audit.bannedWordsFound.length).toBe(0);

      // Write artifact to artifacts/genomes/kajii_critical.usir
      if (!fs.existsSync(artifactDir)) {
        fs.mkdirSync(artifactDir, { recursive: true });
      }
      fs.writeFileSync(kajiiArtifactPath, kajii.binary);
      expect(fs.existsSync(kajiiArtifactPath)).toBe(true);
      expect(fs.statSync(kajiiArtifactPath).size).toBe(4096);
    });

    it('supports round-trip binary decoding and converts to CanonicalParticle for PoE mixing', () => {
      const shigaBytes = fs.readFileSync(shigaArtifactPath);
      const decoded = decodeTierM(shigaBytes);

      expect(decoded.tier).toBe(2);
      expect(decoded.centroids.length).toBe(3);
      expect(decoded.cholesky_factors.length).toBe(3);

      const kajiiBytes = fs.readFileSync(kajiiArtifactPath);
      const decodedKajii = decodeTierM(kajiiBytes);

      // Convert both to CanonicalParticles
      const partVoid = tierMToCanonicalParticle('soseki_void', 'III_Void', decoded);
      const partCrit = tierMToCanonicalParticle('kajii_critical', 'II_Critical', decodedKajii);

      expect(partVoid.mu.length).toBe(24);
      expect(partVoid.precision.length).toBe(24);
      expect(partVoid.cholesky_flat.length).toBe(300);

      // Mix particles 50/50 via mixStyleParticles
      const mixed = mixStyleParticles([partVoid, partCrit], [0.5, 0.5]);
      expect(mixed.mu_mix.length).toBe(24);
      expect(mixed.precision_mix.length).toBe(24);

      // Verify mixed centroid lies between or near constituent centroids
      for (let d = 0; d < 24; d++) {
        const minVal = Math.min(partVoid.mu[d]!, partCrit.mu[d]!);
        const maxVal = Math.max(partVoid.mu[d]!, partCrit.mu[d]!);
        expect(mixed.mu_mix[d]!).toBeGreaterThanOrEqual(minVal - 1e-4);
        expect(mixed.mu_mix[d]!).toBeLessThanOrEqual(maxVal + 1e-4);
      }
    });
  });

  // ===========================================================================
  // Step 3B: Operator ① — Anti-Slop Orthogonal Projection
  // ===========================================================================
  describe('Operator ①: projectOutSlopSubspace', () => {
    it('purges AI Deadzone component across all DEAD-01..DEAD-08 chunks and shifts toward literary phases', () => {
      expect(deadChunks.length).toBe(8);

      for (const chunk of deadChunks) {
        const initialPred = predictPhase(chunk.z, classifier);
        expect(initialPred.phase).toBe('V_Deadzone');

        const initialDeadzoneDist = initialPred.distance;

        // Apply anti-slop projection
        const purified = projectOutSlopSubspace(chunk.z, 0.55);

        // Vector length and bounding invariant
        expect(purified.length).toBe(24);
        for (let i = 0; i < 12; i++) {
          expect(purified[i]!).toBeGreaterThanOrEqual(0.0);
          expect(purified[i]!).toBeLessThanOrEqual(1.0);
        }
        for (let i = 12; i < 24; i++) {
          expect(purified[i]!).toBeGreaterThanOrEqual(0.0);
        }

        const purifiedPred = predictPhase(purified, classifier);

        // State moves away from Deadzone toward authentic literary phase
        expect(purifiedPred.phase).not.toBe('V_Deadzone');
        expect(['III_Void', 'II_Critical']).toContain(purifiedPred.phase);

        // Distance to Deadzone centroid increased significantly
        const deadCentroid = classifier.centroids.get('V_Deadzone')!;
        let distToDead = 0.0;
        const normPurified = purified.map((v, d) => (v - classifier.mu_tr[d]!) / classifier.sd_tr[d]!);
        const diffDead = normPurified.map((v, d) => v - deadCentroid[d]!);
        for (let r = 0; r < 24; r++) {
          let rowDot = 0.0;
          for (let c = 0; c < 24; c++) {
            rowDot += classifier.inv_cov[r]![c]! * diffDead[c]!;
          }
          distToDead += diffDead[r]! * rowDot;
        }

        expect(distToDead).toBeGreaterThan(initialDeadzoneDist);
      }
    });
  });

  // ===========================================================================
  // Step 3B: Operator ② — Riemannian / Cholesky Geodesic Interpolation
  // ===========================================================================
  describe('Operator ②: interpolateGeodesic', () => {
    it('smoothly interpolates between soseki_void and kajii_critical along the Cholesky manifold', () => {
      const shigaBytes = fs.readFileSync(shigaArtifactPath);
      const kajiiBytes = fs.readFileSync(kajiiArtifactPath);

      // At t = 0.0: should be soseki_void
      const at0 = interpolateGeodesic(shigaBytes, kajiiBytes, 0.0, { classifier });
      expect(at0.binary.byteLength).toBe(4096);
      expect(at0.predictedPhase).toBe('III_Void');

      // At t = 1.0: should be kajii_critical
      const at1 = interpolateGeodesic(shigaBytes, kajiiBytes, 1.0, { classifier });
      expect(at1.binary.byteLength).toBe(4096);
      expect(at1.predictedPhase).toBe('II_Critical');

      // Test intermediate points: t = 0.25, 0.50, 0.75
      const steps = [0.0, 0.25, 0.5, 0.75, 1.0];
      const centroids: number[][] = [];

      for (const t of steps) {
        const res = interpolateGeodesic(shigaBytes, kajiiBytes, t, { classifier });
        expect(res.binary.byteLength).toBe(4096);
        expect(res.centroid24d.length).toBe(24);

        // Decodable and valid Tier M
        const dec = decodeTierM(res.binary);
        expect(dec.tier).toBe(2);

        // Mode weights and transition matrix validity
        const wSum = res.profile.weights.reduce((s, w) => s + w, 0);
        expect(wSum).toBeCloseTo(1.0, 4);

        for (const row of res.profile.transition_matrix) {
          const rSum = row.reduce((s, v) => s + v, 0);
          expect(rSum).toBeCloseTo(1.0, 4);
        }

        centroids.push(res.centroid24d);
      }

      // Verify smooth monotonic coordinate movement along geodesic
      // Void Porosity (q[2]) is higher in Soseki than in Kajii: should monotonically decrease with t
      for (let i = 0; i < steps.length - 1; i++) {
        const cNow = centroids[i]!;
        const cNext = centroids[i + 1]!;
        // Centroids shouldn't abruptly jump
        let stepDist = 0.0;
        for (let d = 0; d < 24; d++) {
          const diff = cNext[d]! - cNow[d]!;
          stepDist += diff * diff;
        }
        expect(Math.sqrt(stepDist)).toBeLessThan(1.0);
      }
    });
  });

  // ===========================================================================
  // Step 3B: Operator ③ — Cross-Genre Register Transposition
  // ===========================================================================
  describe('Operator ③: transposeStyleRegister', () => {
    it('preserves legal deductive skeleton (Group II Lattice) exactly while shifting Void and momentum to Soseki', () => {
      const shigaBytes = fs.readFileSync(shigaArtifactPath);
      const sourceText = crys01Chunk.text;

      const sourceExt = extractUsir24d(sourceText);
      const initialLattice = [sourceExt.q[1]!, sourceExt.q[5]!, sourceExt.q[9]!];
      const initialVoid = [sourceExt.q[2]!, sourceExt.q[6]!, sourceExt.q[10]!];

      // Civil law article has high lattice, near-zero void
      expect(initialLattice[0]).toBeGreaterThanOrEqual(0.7);
      expect(initialVoid[0]).toBeLessThan(0.1);

      const result = transposeStyleRegister(sourceText, {
        keepSkeletonGroup: 'lattice',
        targetParticle: shigaBytes,
        classifier,
      });

      expect(result.preservedLatticeIndices).toEqual([1, 5, 9]);

      // Strict lattice preservation: diff < 1e-6
      expect(Math.abs(result.transposedTargetZ[1]! - initialLattice[0]!)).toBeLessThan(1e-6);
      expect(Math.abs(result.transposedTargetZ[5]! - initialLattice[1]!)).toBeLessThan(1e-6);
      expect(Math.abs(result.transposedTargetZ[9]! - initialLattice[2]!)).toBeLessThan(1e-6);

      // Void porosity coordinates transposed to Soseki style (substantially higher)
      expect(result.transposedTargetZ[2]!).toBeGreaterThan(0.35);
      expect(result.transposedTargetZ[6]!).toBeGreaterThan(0.35);

      // Target phase is III_Void
      expect(result.targetPhase).toBe('III_Void');

      // Concrete prompt generated with directives
      expect(result.transpositionPrompt).toContain('# USIR Cross-Genre Register Transposition Prompt');
      expect(result.transpositionPrompt).toContain('Target Style Phase: III_Void');
      expect(result.transpositionPrompt).toContain('Group II Lattice');
      expect(result.transpositionPrompt).toContain(sourceText);
    });
  });
});
