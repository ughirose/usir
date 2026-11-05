import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeTierM } from '@nishiki/usir-core';
import { momentumFromSummary } from '../src/crossModalFunctors.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const FX = JSON.parse(
  fs.readFileSync(path.join(here, '..', '..', 'usir-core', 'tests', 'fixtures', 'momentum_genome_parity.json'), 'utf-8'),
) as { tier_m_b64: string; cases: { q: number[]; p: number[] }[] };
const blob = new Uint8Array(Buffer.from(FX.tier_m_b64, 'base64'));
const RTOL = 1e-5;
const EQ_FLOOR = 0.02;

function expectClose(a: number[], b: number[]) {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) => expect(Math.abs(v - b[i]!)).toBeLessThanOrEqual(RTOL * Math.abs(b[i]!) + 1e-9));
}

describe('M18.3 momentumFromSummary genome path (Python fixture parity)', () => {
  FX.cases.forEach((c, i) => {
    it(`case ${i} matches fixture (bytes and decoded)`, () => {
      expectClose(momentumFromSummary(c.q, undefined, blob), c.p);
      expectClose(momentumFromSummary(c.q, undefined, decodeTierM(blob)), c.p);
    });
  });

  it('responsibility selects the nearer mode', () => {
    const g = decodeTierM(blob);
    const lo = momentumFromSummary(g.centroids[0]!.slice(0, 12), undefined, g);
    const hi = momentumFromSummary(g.centroids[2]!.slice(0, 12), undefined, g);
    lo.forEach((v, i) => expect(Math.abs(v - Math.max(g.centroids[0]![12 + i]!, EQ_FLOOR))).toBeLessThan(1e-3));
    hi.forEach((v, i) => expect(Math.abs(v - Math.max(g.centroids[2]![12 + i]!, EQ_FLOOR))).toBeLessThan(1e-3));
    expect(lo.some((v, i) => Math.abs(v - hi[i]!) > 0.1)).toBe(true);
  });

  it('invalid modes are skipped; all-invalid falls back to fixed Lambda', () => {
    const g = decodeTierM(blob);
    const q = FX.cases[0]!.q;
    const dead = { ...g, weights: [0, 0, 0] };
    expect(momentumFromSummary(q, undefined, dead)).toEqual(momentumFromSummary(q));
    const some = momentumFromSummary(q, undefined, { ...g, weights: [g.weights[0]!, 0, NaN] });
    expect(some.every(Number.isFinite)).toBe(true);
  });

  it('genome undefined is unchanged (legacy proxy hypot kept)', () => {
    const q = Array.from({ length: 12 }, (_, i) => 0.1 + (0.8 * i) / 11);
    expect(momentumFromSummary(q, undefined, undefined)).toEqual(momentumFromSummary(q));
    const proxy = new Array<number>(12).fill(0);
    proxy[0] = 0.4;
    const base = momentumFromSummary(q);
    expect(momentumFromSummary(q, proxy)[0]).toBeCloseTo(Math.hypot(0.4, base[0]!), 12);
  });
});
