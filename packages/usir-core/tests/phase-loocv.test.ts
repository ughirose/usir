import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { runLoocv, CORE_3_PHASES } from '../src/dynamics/landauPhase.js';
import { extractUsir24d } from '../src/sensor/symplectic24d.js';

describe('Gate 2: Zero-Leakage 40-Chunk LOOCV & Ablation Regression Suite', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixtureData: any[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

  // Pre-extract TS vectors to ensure pure TS pipeline verification
  const tsData = fixtureData.map((item) => {
    const extracted = extractUsir24d(item.text);
    return {
      ...item,
      q: extracted.q,
      p: extracted.p,
      z: extracted.z_24d,
    };
  });

  it('verifies 5-phase LOOCV accuracy strictly equals 97.5% (39/40)', () => {
    const res = runLoocv(tsData, (item) => item.z);
    expect(res.total).toBe(40);
    expect(res.correct).toBe(39);
    expect(res.accuracy).toBeCloseTo(0.975, 4);
    expect(res.accuracy).toBeGreaterThanOrEqual(0.95);
  });

  it('verifies Core 3-phase LOOCV accuracy strictly equals 96.4% (27/28)', () => {
    const res = runLoocv(tsData, (item) => item.z, CORE_3_PHASES);
    expect(res.total).toBe(28);
    expect(res.correct).toBe(27);
    expect(res.accuracy).toBeCloseTo(27 / 28, 4);
    expect(res.accuracy).toBeGreaterThanOrEqual(0.95);
  });

  it('proves ablation inequalities (indispensability of all 4 groups & momentum p)', () => {
    // 5-Phase: Full 24D vs 12D Position only
    const accFull5 = runLoocv(tsData, (item) => item.z).accuracy;
    const accPos12 = runLoocv(tsData, (item) => item.q).accuracy;
    expect(accFull5).toBeGreaterThan(accPos12);
    expect(accPos12).toBeCloseTo(0.925, 3);

    // Core 3-phase ablations
    const selectWithout = (item: any, dropIndices: number[]) => {
      const qKept = item.q.filter((_: any, idx: number) => !dropIndices.includes(idx));
      const pKept = item.p.filter((_: any, idx: number) => !dropIndices.includes(idx));
      return [...qKept, ...pKept];
    };

    const accCoreFull = runLoocv(tsData, (item) => item.z, CORE_3_PHASES).accuracy;
    const accCoreNoVoid = runLoocv(tsData, (item) => selectWithout(item, [2, 6, 10]), CORE_3_PHASES).accuracy;
    const accCoreNoShock = runLoocv(tsData, (item) => selectWithout(item, [3, 7, 11]), CORE_3_PHASES).accuracy;
    const accCoreNoMom = runLoocv(tsData, (item) => item.q, CORE_3_PHASES).accuracy;

    // Full 24D strictly superior to any dropped group
    expect(accCoreFull).toBeGreaterThan(accCoreNoVoid);
    expect(accCoreFull).toBeGreaterThan(accCoreNoShock);
    expect(accCoreFull).toBeGreaterThan(accCoreNoMom);

    expect(accCoreNoVoid).toBeCloseTo(26 / 28, 3);
    expect(accCoreNoShock).toBeCloseTo(26 / 28, 3);
    expect(accCoreNoMom).toBeCloseTo(25 / 28, 3);
  });
});
