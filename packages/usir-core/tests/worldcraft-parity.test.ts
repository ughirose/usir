import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  WorldCraftArbitrator,
  buildGeneralJacobian,
  computeNullSpaceProjectorSvd,
  computeNullSpaceProjectorTikhonov,
  compileConstraintJacobian,
  matMul,
  identity,
  type Matrix,
} from '../src/dynamics/worldcraftArbitrator.js';
import { HSMMLongformDirector } from '../src/dynamics/hsmmLongformDirector.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const FX = JSON.parse(fs.readFileSync(path.join(here, 'fixtures', 'worldcraft_arbitrator_parity.json'), 'utf-8'));
const TOL = 1e-6;

function flat(a: number[] | number[][]): number[] {
  return (a as unknown[]).flat() as number[];
}
function maxAbsDiff(a: number[] | number[][], b: number[] | number[][]): number {
  const fa = flat(a);
  const fb = flat(b);
  expect(fa.length).toBe(fb.length);
  return fa.reduce((m, x, i) => Math.max(m, Math.abs(x - fb[i]!)), 0);
}
const transpose = (m: Matrix): Matrix => m[0]!.map((_, c) => m.map((r) => r[c]!));

describe('M17 WorldCraftArbitrator SVD parity (Python fixture)', () => {
  for (const name of Object.keys(FX.constraint_sets)) {
    const cs = FX.constraint_sets[name];
    it(`projector ${name}`, () => {
      const j = buildGeneralJacobian(cs.rows);
      const p = computeNullSpaceProjectorSvd(j);
      expect(maxAbsDiff(p, cs.p_null_svd)).toBeLessThan(TOL);
      expect(maxAbsDiff(computeNullSpaceProjectorTikhonov(j), cs.p_null_tikhonov)).toBeLessThan(TOL);
      expect(maxAbsDiff(p, transpose(p))).toBeLessThan(1e-12);
      expect(maxAbsDiff(matMul(p, p), p)).toBeLessThan(1e-12);
      if (j.length > 0) {
        const jp = matMul(j, p);
        expect(maxAbsDiff(jp, jp.map((r) => r.map(() => 0)))).toBeLessThan(1e-12);
      }
      const trace = p.reduce((a, r, i) => a + r[i]!, 0);
      expect(Math.round(12 - trace)).toBe(cs.rank);
    });
    it(`arbitration ${name}`, () => {
      const j = buildGeneralJacobian(cs.rows);
      const arb = new WorldCraftArbitrator({ projectorMethod: 'svd' });
      FX.momenta.forEach((m: number[], idx: number) => {
        const exp = cs.arbitration[idx];
        const { safe, cut } = arb.arbitrateMomentum(m, j);
        expect(maxAbsDiff(safe, exp.svd_safe)).toBeLessThan(TOL);
        expect(maxAbsDiff(cut, exp.svd_cut)).toBeLessThan(TOL);
        expect(new WorldCraftArbitrator().projectorMethod).toBe('svd');
        expect(maxAbsDiff(new WorldCraftArbitrator().arbitrateMomentum(m, j).safe, exp.svd_safe)).toBeLessThan(TOL);
        const tik = new WorldCraftArbitrator({ projectorMethod: 'tikhonov' }).arbitrateMomentum(m, j);
        expect(maxAbsDiff(tik.safe, exp.tik_safe)).toBeLessThan(TOL);
      });
    });
  }

  it('unit rows: svd projector is the exact coordinate projector', () => {
    const p = computeNullSpaceProjectorSvd(compileConstraintJacobian('', ['evidence']));
    const exact = identity();
    exact[7]![7] = 0;
    exact[11]![11] = 0;
    expect(maxAbsDiff(p, exact)).toBeLessThan(1e-12);
  });

  it('rejects k > 12 and wrong width', () => {
    expect(() => computeNullSpaceProjectorSvd(Array.from({ length: 13 }, () => new Array(12).fill(1)))).toThrow();
    expect(() => computeNullSpaceProjectorSvd([[1, 2, 3]])).toThrow();
  });

  it('director step sequence matches Python fixture', () => {
    const cfg = FX.director.config;
    const d = new HSMMLongformDirector({
      transitionMatrix: cfg.transition_matrix,
      refractoryCooldown: cfg.refractory_cooldown,
      seed: 1,
      arbitrator: new WorldCraftArbitrator({ projectorMethod: cfg.projector_method }),
    });
    for (const rec of FX.director.steps) {
      const out = rec.out;
      if (rec.op === 'arbitrate_rows') {
        const r = d.arbitrateWithJacobian(rec.target_p, buildGeneralJacobian(rec.rows));
        expect(maxAbsDiff(r.safeP, out.safe_p)).toBeLessThan(TOL);
        expect(maxAbsDiff(r.cutP, out.cut_p)).toBeLessThan(TOL);
      } else if (rec.op === 'accumulate') {
        expect(maxAbsDiff(d.accumulateUnreleasedMomentum(rec.target_p, rec.measured_p), out.delta)).toBeLessThan(TOL);
      } else if (rec.op === 'step') {
        expect(d.stepTransition(rec.current_well)).toBe(out.next_well);
      } else if (rec.op === 'discharge') {
        const r = d.dischargeMomentum(rec.base_climax_p, rec.later_climax ?? false);
        expect(maxAbsDiff(r.boostedP, out.boosted_p)).toBeLessThan(TOL);
        expect(maxAbsDiff(r.dischargedDeltaP, out.discharged_p)).toBeLessThan(TOL);
      }
      expect(maxAbsDiff(d.pBanked, rec.state.p_banked)).toBeLessThan(TOL);
      expect(d.refractoryCounter).toBe(rec.state.refractory_counter);
      expect(Math.abs(d.getBankedEnergy() - rec.state.banked_energy)).toBeLessThan(TOL);
    }
  });
});
