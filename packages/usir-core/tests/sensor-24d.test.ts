import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { extractUsir24d } from '../src/sensor/symplectic24d.js';

interface GoldenRecord {
  id: string;
  phase: string;
  author: string;
  title: string;
  text: string;
  q: number[];
  p: number[];
  z: number[];
  groups: {
    I_Flux: number;
    II_Lattice: number;
    III_Void: number;
    IV_Shock: number;
  };
  order_params: {
    psi_order: number;
    phi_void: number;
    kappa_kin: number;
  };
  len_cv: number;
  cadence_alive: number;
}

describe('Gate 1: Sensor 24D Numerical Equivalence (TS vs Python)', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: GoldenRecord[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

  it('loads all 40 golden corpus records', () => {
    expect(fixtureData.length).toBe(40);
  });

  for (const record of fixtureData) {
    it(`matches Python golden reference for ${record.id} (${record.title}) within 1e-5`, () => {
      const res = extractUsir24d(record.text);

      // Check 12D Position q
      for (let i = 0; i < 12; i++) {
        const diffQ = Math.abs(res.q[i]! - record.q[i]!);
        expect(diffQ, `q[${i}] mismatch in ${record.id}: TS=${res.q[i]}, Py=${record.q[i]}`).toBeLessThan(1e-5);
      }

      // Check 12D Momentum p
      for (let i = 0; i < 12; i++) {
        const diffP = Math.abs(res.p[i]! - record.p[i]!);
        expect(diffP, `p[${i}] mismatch in ${record.id}: TS=${res.p[i]}, Py=${record.p[i]}`).toBeLessThan(1e-5);
      }

      // Check full 24D state
      for (let i = 0; i < 24; i++) {
        const diffZ = Math.abs(res.z_24d[i]! - record.z[i]!);
        expect(diffZ, `z[${i}] mismatch in ${record.id}`).toBeLessThan(1e-5);
      }

      // Check Order Parameters
      expect(Math.abs(res.order_params.psi_order - record.order_params.psi_order)).toBeLessThan(1e-5);
      expect(Math.abs(res.order_params.phi_void - record.order_params.phi_void)).toBeLessThan(1e-5);
      expect(Math.abs(res.order_params.kappa_kin - record.order_params.kappa_kin)).toBeLessThan(1e-5);

      // Check Len CV and Cadence Alive
      expect(Math.abs(res.len_cv - record.len_cv)).toBeLessThan(1e-5);
      expect(Math.abs(res.cadence_alive - record.cadence_alive)).toBeLessThan(1e-5);
    });
  }
});
