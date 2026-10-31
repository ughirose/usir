import { describe, expect, it } from 'vitest';
import {
  computeCholeskyEnergy,
  computeHamiltonian,
  extractUsir24d,
  CANONICAL_PARTICLES,
} from '@nishiki/usir-core';
import {
  benchmarkInstrumentationLatency,
  choleskySqDist24d,
  computeHamiltonianSimd,
  loadUsirSimdKernel,
  poeStep24d,
  DEFAULT_BENCHMARK_400_CHAR_TEXT,
} from '../src/index.js';

describe('@nishiki/usir-wasm Kernel & Latency Suite (TASK-363)', () => {
  it('loads the USIR SIMD kernel successfully', async () => {
    const kernel = await loadUsirSimdKernel();
    expect(kernel).toBeDefined();
    expect(typeof kernel.choleskySqDist24d).toBe('function');
    expect(typeof kernel.computeHamiltonianSimd).toBe('function');
    expect(typeof kernel.poeStep24d).toBe('function');
  });

  it('guarantees strict numerical equivalence (< 1e-5) for 24D Cholesky squared distance', () => {
    // Generate synthetic 24D vectors and packed L
    const z = new Float32Array(24);
    const mu = new Float32Array(24);
    const LPacked = new Float32Array(300);

    for (let i = 0; i < 24; i++) {
      z[i] = 0.2 + (i % 7) * 0.11;
      mu[i] = 0.15 + (i % 5) * 0.13;
    }

    let idx = 0;
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j <= i; j++) {
        if (i === j) {
          LPacked[idx++] = 1.0 + (i % 4) * 0.2;
        } else {
          LPacked[idx++] = 0.01 * ((i + j) % 3);
        }
      }
    }

    const wasmDist = choleskySqDist24d(z, mu, LPacked);

    // Compute via usir-core
    const zArr = Array.from(z);
    const muArr = Array.from(mu);
    const LArr = Array.from(LPacked);
    const coreEnergy = computeCholeskyEnergy(zArr, muArr, LArr, 24);
    const coreSqDist = 2.0 * coreEnergy;

    expect(Math.abs(wasmDist - coreSqDist)).toBeLessThan(1e-5);
  });

  it('guarantees numerical equivalence (< 1e-5) with Canonical Particles from usir-core', () => {
    const amber = CANONICAL_PARTICLES.AMBER_CRITICAL!;
    const z = new Float32Array(amber.mu.map((v, i) => v + (i % 2 === 0 ? 0.05 : -0.05)));
    const mu = new Float32Array(amber.mu);
    const LPacked = new Float32Array(amber.cholesky_flat);

    const wasmDist = choleskySqDist24d(z, mu, LPacked);
    const coreDist = 2.0 * computeCholeskyEnergy(Array.from(z), amber.mu, amber.cholesky_flat, 24);

    expect(Math.abs(wasmDist - coreDist)).toBeLessThan(1e-5);
  });

  it('guarantees strict numerical equivalence (< 1e-5) for 12D Hamiltonian Dynamics', () => {
    const q = new Float32Array(12);
    const p = new Float32Array(12);
    const massInv = new Float32Array(12);
    const stiffness = new Float32Array(12);

    for (let i = 0; i < 12; i++) {
      q[i] = 0.35 + i * 0.04;
      p[i] = 0.1 + i * 0.05;
      massInv[i] = 1.0 / (0.8 + (i % 3) * 0.2);
      stiffness[i] = 1.2 + (i % 4) * 0.15;
    }

    const wasmH = computeHamiltonianSimd(q, p, massInv, stiffness);

    const coreResult = computeHamiltonian(Array.from(q), Array.from(p), {
      massDiag: Array.from(massInv).map((inv) => 1.0 / inv),
      potentialStiffnessDiag: Array.from(stiffness),
      targetQ: new Array(12).fill(0.5),
    });

    expect(Math.abs(wasmH - coreResult.totalHamiltonian)).toBeLessThan(1e-5);
  });

  it('computes PoE precision-weighted gradient step correctly', () => {
    const currentZ = new Float32Array(24).fill(0.2);
    const targetMu = new Float32Array(24).fill(0.8);
    const lambdaDiag = new Float32Array(24).fill(1.5);
    const stepSize = 0.1;

    const outGrad = new Float32Array(24);
    poeStep24d(currentZ, targetMu, lambdaDiag, stepSize, outGrad);

    // Expected: 0.1 * 1.5 * (0.8 - 0.2) = 0.15 * 0.6 = 0.09
    for (let i = 0; i < 24; i++) {
      expect(outGrad[i]).toBeCloseTo(0.09, 5);
    }
  });

  it('validates buffer length requirements', () => {
    const shortBuf = new Float32Array(10);
    const normalBuf = new Float32Array(24);
    const lBuf = new Float32Array(300);

    expect(() => choleskySqDist24d(shortBuf, normalBuf, lBuf)).toThrow();
    expect(() => computeHamiltonianSimd(shortBuf, normalBuf)).toThrow();
  });

  it('satisfies latency requirement: 400-char instrumentation < 1.0 ms and kernel distance < 10 us', () => {
    const bench = benchmarkInstrumentationLatency(DEFAULT_BENCHMARK_400_CHAR_TEXT, 30, 2000);

    expect(bench.textLength).toBeGreaterThanOrEqual(400);
    expect(bench.instrumentationLatencyMs).toBeLessThan(1.0);
    expect(bench.kernelDistanceLatencyUs).toBeLessThan(10.0);
    expect(bench.isInstrumentationUnder1ms).toBe(true);
    expect(bench.isKernelUnder10us).toBe(true);
    expect(bench.passed).toBe(true);
  });
});
