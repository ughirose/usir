import { describe, expect, it } from 'vitest';
import { HSMMLongformDirector } from '../src/dynamics/hsmmLongformDirector.js';

const TM = [
  [0, 0, 1],
  [0, 0, 1],
  [1, 0, 0],
];
const BANK = Array.from({ length: 12 }, (_, i) => 0.1 + 0.1 * i);

function baseVec(): number[] {
  const v = new Array(12).fill(0);
  for (const i of [3, 7, 11]) v[i] = 0.25;
  for (const i of [0, 4, 8]) v[i] = -0.2;
  return v;
}

describe('Climax base momentum and multi-Climax carry', () => {
  it('releases the Climax base with an empty bank', () => {
    const d = new HSMMLongformDirector({ seed: 1 });
    const r = d.dischargeMomentum(new Array(12).fill(0.5));
    r.dischargedDeltaP.forEach((v, i) => expect(v).toBeCloseTo(baseVec()[i]!, 12));
    r.boostedP.forEach((v, i) => expect(v).toBeCloseTo(0.5 + baseVec()[i]!, 12));
    expect(d.pBanked.every((v) => v === 0)).toBe(true);
  });

  it('carries (1 - releaseFrac) of the bank when another Climax follows', () => {
    const d = new HSMMLongformDirector({ seed: 1 });
    d.pBanked = [...BANK];
    const mid = d.dischargeMomentum(new Array(12).fill(0), true);
    mid.dischargedDeltaP.forEach((v, i) => expect(v).toBeCloseTo(0.7 * BANK[i]! + baseVec()[i]!, 12));
    d.pBanked.forEach((v, i) => expect(v).toBeCloseTo(0.3 * BANK[i]!, 12));
    const last = d.dischargeMomentum(new Array(12).fill(0), false);
    last.dischargedDeltaP.forEach((v, i) => expect(v).toBeCloseTo(0.3 * BANK[i]! + baseVec()[i]!, 12));
    expect(d.pBanked.every((v) => v === 0)).toBe(true);
  });

  it('applyClimaxBase = false restores the previous full release', () => {
    const d = new HSMMLongformDirector({ seed: 1 });
    d.pBanked = [...BANK];
    const r = d.dischargeMomentum(new Array(12).fill(0.5), true, false);
    r.dischargedDeltaP.forEach((v, i) => expect(v).toBeCloseTo(BANK[i]!, 12));
    expect(d.pBanked.every((v) => v === 0)).toBe(true);
  });

  it('honours constructor parameters', () => {
    const d = new HSMMLongformDirector({ seed: 1, climaxBaseShock: 0.5, climaxBaseFlux: 0.1, climaxReleaseFrac: 0.5 });
    d.pBanked = [...BANK];
    const r = d.dischargeMomentum(new Array(12).fill(0), true);
    expect(r.dischargedDeltaP[3]).toBeCloseTo(0.5 * BANK[3]! + 0.5, 12);
    expect(r.dischargedDeltaP[0]).toBeCloseTo(0.5 * BANK[0]! - 0.1, 12);
    expect(d.pBanked[5]).toBeCloseTo(0.5 * BANK[5]!, 12);
  });

  it('planWells has no side effects and matches generateLongform', () => {
    const d = new HSMMLongformDirector({ seed: 7, transitionMatrix: TM });
    const planned = d.planWells(6);
    expect(planned).toEqual([0, 2, 0, 0, 2, 0]);
    expect(d.refractoryCounter).toBe(0);
    expect(d.planWells(6)).toEqual(planned);
    const res = d.generateLongform({ targetParagraphs: 6, climaxBaseMomentum: true });
    expect(res.wells ?? planned).toEqual(planned);
    const climaxNorms = res.paragraphs.filter((p) => p.wellIndex === 2).map((p) => p.dischargedMomentumNorm);
    expect(climaxNorms.length).toBe(2);
    expect(climaxNorms.every((n) => n > 0)).toBe(true);
  });
});
