/**
 * Landau 5-Phase Thermodynamic Classifier and LOOCV Engine for USIR.
 */

import { stdPop } from '../sensor/dictionaries.js';
import { GOLDEN_CORPUS_FEATURES } from './goldenCorpusFeatures.js';

export type UsirPhase =
  | 'I_Crystalline'
  | 'II_Critical'
  | 'III_Void'
  | 'IV_Turbulent'
  | 'V_Deadzone';

export const ALL_PHASES: readonly UsirPhase[] = [
  'I_Crystalline',
  'II_Critical',
  'III_Void',
  'IV_Turbulent',
  'V_Deadzone',
];

export const CORE_3_PHASES: readonly UsirPhase[] = [
  'II_Critical',
  'III_Void',
  'V_Deadzone',
];

export interface PhaseCentroidsAndCov {
  classes: UsirPhase[];
  mu_tr: number[];
  sd_tr: number[];
  inv_cov: number[][];
  centroids: Map<UsirPhase, number[]>;
}

export function invertMatrixGaussJordan(A: number[][]): number[][] {
  const n = A.length;
  if (n === 1) {
    return [[1.0 / A[0]![0]!]];
  }

  const M: number[][] = A.map((row, i) => {
    const r = new Array<number>(2 * n);
    for (let j = 0; j < n; j++) {
      r[j] = row[j]!;
      r[n + j] = i === j ? 1.0 : 0.0;
    }
    return r;
  });

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    let maxVal = Math.abs(M[i]![i]!);
    for (let k = i + 1; k < n; k++) {
      const val = Math.abs(M[k]![i]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = k;
      }
    }
    if (maxRow !== i) {
      const tmp = M[i]!;
      M[i] = M[maxRow]!;
      M[maxRow] = tmp;
    }

    const pivot = M[i]![i]!;
    if (Math.abs(pivot) < 1e-12) {
      // Degenerate fallback: regularized diagonal inverse
      const I_fallback = new Array(n).fill(0).map((_, r) => {
        const row = new Array(n).fill(0.0);
        row[r] = 1.0 / (A[r]![r]! + 1e-4);
        return row;
      });
      return I_fallback;
    }

    const rowI = M[i]!;
    for (let j = 0; j < 2 * n; j++) {
      rowI[j] = (rowI[j] ?? 0.0) / pivot;
    }

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const rowK = M[k]!;
        const factor = rowK[i] ?? 0.0;
        for (let j = 0; j < 2 * n; j++) {
          rowK[j] = (rowK[j] ?? 0.0) - factor * (rowI[j] ?? 0.0);
        }
      }
    }
  }

  return M.map((row) => row.slice(n));
}

export function fitPhaseClassifier(
  data: { phase: string; features: number[] }[]
): PhaseCentroidsAndCov {
  const N = data.length;
  const D = data[0]!.features.length;
  const classes = Array.from(new Set(data.map((d) => d.phase as UsirPhase))).sort();

  // mu_tr and sd_tr with ddof=0 matching numpy.std
  const mu_tr = new Array<number>(D).fill(0.0);
  for (let d = 0; d < D; d++) {
    let sum = 0.0;
    for (let i = 0; i < N; i++) {
      sum += data[i]!.features[d]!;
    }
    mu_tr[d] = sum / N;
  }

  const sd_tr = new Array<number>(D).fill(0.0);
  for (let d = 0; d < D; d++) {
    const vals = data.map((item) => item.features[d]!);
    sd_tr[d] = stdPop(vals) + 1e-6;
  }

  // Normalized train matrix
  const X_tr_norm: number[][] = data.map((item) => {
    return item.features.map((val, d) => (val - mu_tr[d]!) / sd_tr[d]!);
  });

  // Sample covariance with ddof=1 (numpy.cov default)
  const cov: number[][] = new Array(D).fill(0).map(() => new Array(D).fill(0.0));
  const denom = Math.max(N - 1, 1);
  for (let r = 0; r < D; r++) {
    for (let c = 0; c <= r; c++) {
      let sum = 0.0;
      for (let i = 0; i < N; i++) {
        sum += X_tr_norm[i]![r]! * X_tr_norm[i]![c]!;
      }
      const val = sum / denom;
      cov[r]![c] = val;
      cov[c]![r] = val;
    }
  }

  // Ledoit-Wolf style shrinkage: reg_cov = 0.20 * cov + 0.80 * I
  const reg_cov: number[][] = new Array(D).fill(0).map((_, r) => {
    const row = new Array(D).fill(0.0);
    for (let c = 0; c < D; c++) {
      row[c] = 0.20 * cov[r]![c]! + (r === c ? 0.80 : 0.0);
    }
    return row;
  });

  const inv_cov = invertMatrixGaussJordan(reg_cov);

  // Compute class centroids in normalized space
  const centroids = new Map<UsirPhase, number[]>();
  for (const c of classes) {
    const subset = X_tr_norm.filter((_, idx) => data[idx]!.phase === c);
    const centroid = new Array<number>(D).fill(0.0);
    for (let d = 0; d < D; d++) {
      let sum = 0.0;
      for (const row of subset) {
        sum += row[d]!;
      }
      centroid[d] = sum / Math.max(subset.length, 1);
    }
    centroids.set(c, centroid);
  }

  return { classes, mu_tr, sd_tr, inv_cov, centroids };
}

export function predictPhase(
  x: number[],
  classifier: PhaseCentroidsAndCov
): { phase: UsirPhase; distance: number } {
  const { classes, mu_tr, sd_tr, inv_cov, centroids } = classifier;
  const D = x.length;
  const x_norm = x.map((v, d) => (v - mu_tr[d]!) / sd_tr[d]!);

  let bestPhase: UsirPhase = classes[0]!;
  let bestDist = 1e18;

  for (const c of classes) {
    const centroid = centroids.get(c)!;
    const diff = x_norm.map((v, d) => v - centroid[d]!);

    // diff^T * inv_cov * diff
    let dist = 0.0;
    for (let r = 0; r < D; r++) {
      let rowDot = 0.0;
      const row = inv_cov[r]!;
      for (let j = 0; j < D; j++) {
        rowDot += row[j]! * diff[j]!;
      }
      dist += diff[r]! * rowDot;
    }

    if (dist < bestDist) {
      bestDist = dist;
      bestPhase = c;
    }
  }

  return { phase: bestPhase, distance: bestDist };
}

export interface LoocvResult {
  accuracy: number;
  correct: number;
  total: number;
  predictions: UsirPhase[];
}

export function runLoocv(
  corpus: { id?: string; phase: string; [key: string]: unknown }[],
  featureFn: (item: any) => number[],
  targetFilter?: readonly string[]
): LoocvResult {
  const filtered = corpus.filter(
    (item) => !targetFilter || targetFilter.includes(item.phase)
  );
  const N = filtered.length;
  const items = filtered.map((item) => ({
    phase: item.phase,
    features: featureFn(item),
  }));

  let correct = 0;
  const predictions: UsirPhase[] = [];

  for (let i = 0; i < N; i++) {
    const trainData = items.filter((_, idx) => idx !== i);
    const testItem = items[i]!;

    const classifier = fitPhaseClassifier(trainData);
    const pred = predictPhase(testItem.features, classifier);
    predictions.push(pred.phase);

    if (pred.phase === testItem.phase) {
      correct++;
    }
  }

  return {
    accuracy: N > 0 ? correct / N : 0.0,
    correct,
    total: N,
    predictions,
  };
}

let defaultClassifier: PhaseCentroidsAndCov | null = null;

export function getDefaultClassifier(): PhaseCentroidsAndCov {
  if (!defaultClassifier) {
    defaultClassifier = fitPhaseClassifier(
      GOLDEN_CORPUS_FEATURES.map((item) => ({
        phase: item.phase,
        features: item.z,
      }))
    );
  }
  return defaultClassifier;
}

export function classifyPhase(z: number[]): { phase: UsirPhase; distance: number } {
  return predictPhase(z, getDefaultClassifier());
}
