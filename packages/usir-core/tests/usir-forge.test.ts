import { describe, it, expect } from 'vitest';
import { execSync, execFileSync } from 'child_process';
import {
  alignPolarFourierPhase,
  computePolarFourierPhase,
  computePackedCholesky24d,
  unpackCholesky24d,
  precisionMatrixFromPackedCholesky24d,
  compileTierMBinary,
  loadTierMBinary,
  classifyMode24d,
  OSWEStreamingCompiler,
  USIR_MAGIC,
  TIER_M_BYTE_LENGTH,
} from '../src/schema/usirForge.js';

describe('Phase D WS-D1: OSWE Polar-Fourier Phase-Aligned usir-forge & 4KB Binary Compiler', () => {
  // ===========================================================================
  // 1. POLAR-FOURIER PHASE ALIGNMENT & DESTRUCTIVE CANCELLATION PREVENTION
  // ===========================================================================
  describe('Polar-Fourier Phase Alignment', () => {
    it('prevents destructive wave cancellation between anti-phase signals', () => {
      const M = 64;
      const wave1 = new Array<number>(M);
      const wave2 = new Array<number>(M);

      for (let m = 0; m < M; m++) {
        // Wave 1 peak at m = 0
        wave1[m] = 1.0 + Math.cos((2.0 * Math.PI * m) / M);
        // Wave 2 peak at m = M / 2 (anti-phase, delta_theta = pi)
        wave2[m] = 1.0 + Math.cos((2.0 * Math.PI * m) / M - Math.PI);
      }

      // 1. Unaligned average: destructive cancellation into flat Deadzone
      const unalignedAvg = wave1.map((v, i) => (v + wave2[i]!) / 2.0);
      const minUnaligned = Math.min(...unalignedAvg);
      const maxUnaligned = Math.max(...unalignedAvg);
      const ptpUnaligned = maxUnaligned - minUnaligned;
      expect(ptpUnaligned).toBeLessThan(1e-12);

      // 2. Aligned average to targetPhase = 3*pi/4
      const targetPhase = (3.0 * Math.PI) / 4.0;
      const aligned1 = alignPolarFourierPhase(wave1, targetPhase);
      const aligned2 = alignPolarFourierPhase(wave2, targetPhase);

      const alignedAvg = aligned1.map((v, i) => (v + aligned2[i]!) / 2.0);
      const minAligned = Math.min(...alignedAvg);
      const maxAligned = Math.max(...alignedAvg);
      const ptpAligned = maxAligned - minAligned;

      // Peak amplitude is strongly preserved (>= 1.90, near 2.0)
      expect(ptpAligned).toBeGreaterThanOrEqual(1.90);

      // Phases match target phase within discretization step
      const phase1 = computePolarFourierPhase(aligned1);
      const phase2 = computePolarFourierPhase(aligned2);
      const phaseAvg = computePolarFourierPhase(alignedAvg);

      const maxDiscretizationErr = (2.0 * Math.PI) / M + 1e-4;
      expect(Math.abs(phase1 - targetPhase)).toBeLessThan(maxDiscretizationErr);
      expect(Math.abs(phase2 - targetPhase)).toBeLessThan(maxDiscretizationErr);
      expect(Math.abs(phaseAvg - targetPhase)).toBeLessThan(maxDiscretizationErr);
    });

    it('aligns 2D vector trajectories along time axis 0', () => {
      const M = 32;
      const traj: number[][] = [];
      for (let t = 0; t < M; t++) {
        const vec = new Array(24).fill(0.2);
        vec[3] = 0.5 + 0.5 * Math.cos((2.0 * Math.PI * t) / M); // Shock coordinate
        traj.push(vec);
      }

      const targetPhase = (3.0 * Math.PI) / 4.0;
      const aligned = alignPolarFourierPhase(traj, targetPhase);
      expect(aligned.length).toBe(M);
      expect(aligned[0]?.length).toBe(24);

      // Check aligned energy phase
      const energyAligned = aligned.map((v) => {
        let p2 = 0.0;
        for (let i = 12; i < 24; i++) p2 += (v[i] ?? 0) * (v[i] ?? 0);
        return 0.5 * p2 + (v[3] ?? 0);
      });
      const phase = computePolarFourierPhase(energyAligned);
      const maxDiscretizationErr = (2.0 * Math.PI) / M + 1e-4;
      expect(Math.abs(phase - targetPhase)).toBeLessThan(maxDiscretizationErr);
    });
  });

  // ===========================================================================
  // 2. STRICT 4,096 BYTES TIER M BINARY COMPILATION
  // ===========================================================================
  describe('Tier M 4KB Binary Compilation', () => {
    it('compiles strictly 4,096-byte binary with valid magic and CRC32', () => {
      const compiler = new OSWEStreamingCompiler({ minClusters: 5 });

      for (let k = 0; k < 5; k++) {
        const sample = new Array(24).fill(0.4 + 0.1 * k);
        compiler.addSample(sample, k % 3, `author_cluster_${k}`);
      }

      const binary = compiler.compileTierMBinary();
      expect(binary.byteLength).toBe(TIER_M_BYTE_LENGTH);
      expect(binary.byteLength).toBe(4096);

      // View magic at offset 0
      const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
      expect(view.getUint32(0, true)).toBe(USIR_MAGIC);

      // Decode with loadTierMBinary
      const decoded = loadTierMBinary(binary);
      expect(decoded.version).toBe(1);
      expect(decoded.tier).toBe(2);
      expect(decoded.flags).toBe(0);
      expect(decoded.transition_matrix.length).toBe(3);
      expect(decoded.weights.length).toBe(3);
      expect(decoded.centroids.length).toBe(3);
      expect(decoded.centroids[0]?.length).toBe(24);
      expect(decoded.cholesky_factors.length).toBe(3);
      expect(decoded.cholesky_factors[0]?.length).toBe(300);
      expect(typeof decoded.crc32).toBe('number');
    });
  });

  // ===========================================================================
  // 3. K >= 5 CLUSTER ANONYMITY GATE
  // ===========================================================================
  describe('Cluster Anonymity Gate (K >= 5)', () => {
    it('rejects compilation when K < 5 and passes when K >= 5', () => {
      const compiler = new OSWEStreamingCompiler({ minClusters: 5 });

      for (let i = 0; i < 3; i++) {
        compiler.addSample(new Array(24).fill(0.5), 0, `author_${i}`);
      }

      expect(compiler.clusterCount).toBe(3);

      // Throws error for K < 5
      expect(() => {
        compiler.compileTierMBinary({ enforceAnonymity: true });
      }).toThrow(/Cluster anonymity gate failed: K=3 < 5/);

      // Add 4th cluster: still fails
      compiler.registerCluster('author_3');
      expect(compiler.clusterCount).toBe(4);
      expect(() => {
        compiler.compileTierMBinary({ enforceAnonymity: true });
      }).toThrow(/Cluster anonymity gate failed: K=4 < 5/);

      // Add 5th cluster: passes
      compiler.registerCluster('author_4');
      expect(compiler.clusterCount).toBe(5);
      const binary = compiler.compileTierMBinary({ enforceAnonymity: true });
      expect(binary.byteLength).toBe(4096);
    });
  });

  // ===========================================================================
  // 4. PYTHON <-> TYPESCRIPT ROUNDTRIP INTEROPERABILITY
  // ===========================================================================
  describe('Cross-Language Round-Trip Interoperability', () => {
    it('successfully interoperates with Python implementation', () => {
      const compiler = new OSWEStreamingCompiler({ minClusters: 5 });
      for (let i = 0; i < 15; i++) {
        const vec = new Array(24).fill(0).map((_, d) => 0.05 * ((i * 3 + d) % 17 + 1));
        compiler.addSample(vec, i % 3, `cluster_${i % 5}`);
      }

      const tsBinary = compiler.compileTierMBinary();
      expect(tsBinary.byteLength).toBe(4096);

      const pyScript = [
        'import sys, json',
        'from usir_py.usir_forge import load_tier_m_binary',
        'data = sys.stdin.buffer.read()',
        'decoded = load_tier_m_binary(data)',
        'res = {',
        '    "version": decoded["version"],',
        '    "tier": decoded["tier"],',
        '    "crc32": decoded["crc32"],',
        '    "weights": decoded["weights"]',
        '}',
        'sys.stdout.write(json.dumps(res))',
      ].join('\n');

      const stdout = execFileSync(
        'c:\\dev\\usir\\python\\.venv\\Scripts\\python.exe',
        ['-c', pyScript],
        {
          input: Buffer.from(tsBinary),
          cwd: 'c:\\dev\\usir',
        }
      ).toString();

      const pyResult = JSON.parse(stdout);
      expect(pyResult.version).toBe(1);
      expect(pyResult.tier).toBe(2);

      const tsDecoded = loadTierMBinary(tsBinary);
      expect(pyResult.crc32).toBe(tsDecoded.crc32);
      expect(pyResult.weights[0]).toBeCloseTo(tsDecoded.weights[0]!, 5);
      expect(pyResult.weights[1]).toBeCloseTo(tsDecoded.weights[1]!, 5);
      expect(pyResult.weights[2]).toBeCloseTo(tsDecoded.weights[2]!, 5);
    });
  });

  // ===========================================================================
  // 5. CHOLESKY FACTOR RECONSTRUCTION
  // ===========================================================================
  describe('Cholesky Precision Reconstruction', () => {
    it('reconstructs symmetric positive-definite precision matrix from 300 elements', () => {
      const compiler = new OSWEStreamingCompiler({ minClusters: 5 });
      for (let i = 0; i < 20; i++) {
        const vec = new Array(24).fill(0).map((_, d) => 0.3 + 0.02 * ((i + d) % 10));
        compiler.addSample(vec, 0, `author_${i % 5}`);
      }

      const binary = compiler.compileTierMBinary();
      const decoded = loadTierMBinary(binary);

      const packedL0 = decoded.cholesky_factors[0]!;
      expect(packedL0.length).toBe(300);

      const Lambda = precisionMatrixFromPackedCholesky24d(packedL0);
      expect(Lambda.length).toBe(24);
      expect(Lambda[0]?.length).toBe(24);

      // Verify symmetry Lambda[r][c] === Lambda[c][r]
      for (let r = 0; r < 24; r++) {
        for (let c = 0; c < 24; c++) {
          expect(Math.abs(Lambda[r]![c]! - Lambda[c]![r]!)).toBeLessThan(1e-6);
        }
      }

      // Verify diagonal elements are positive
      for (let i = 0; i < 24; i++) {
        expect(Lambda[i]![i]!).toBeGreaterThan(0.0);
      }
    });
  });
});
