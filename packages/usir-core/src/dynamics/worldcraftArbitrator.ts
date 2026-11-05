/**
 * WorldCraft null-space arbitrator (numeric part only; pure TS, no DOM / no deps).
 *
 * Port of python/usir_py/nishiki_unified_field.py WorldCraftArbitrator:
 *   compile_constraint_jacobian / compute_null_space_projector(_svd) / arbitrate_momentum.
 * The Tier-1 text-ontology mask (check_tier1_violations, apply_tier1_hard_mask) is NOT ported.
 *
 * General constraint Jacobian J_C in R^{k x 12} (k <= 12, rank-deficient allowed):
 *   P_null = I - J_C^+ J_C,  J_C^+ from a one-sided Jacobi (Hestenes) SVD,
 *   singular values s_i <= max(k, 12) * EPS_MACHINE * s_max are treated as zero.
 */

export const WORLDCRAFT_DIM = 12;
export const MACHINE_EPS = 2.220446049250313e-16;

export const CONSTRAINT_MODE_ELEMENTS: Readonly<Record<string, readonly number[]>> = {
  evidence: [7, 11],
  logic_voice: [1, 5],
  stillness: [2, 6, 10],
};

export const CONSTRAINT_MODE_CUES: Readonly<Record<string, readonly string[]>> = {
  evidence: ['証拠', 'データ', '事実', '記録', '報告', '検証', '調査', '計測', '実測'],
  logic_voice: ['論理', '証明', 'したがって', 'ゆえに', '仮定', '定理', '前提', '結論'],
  stillness: ['静寂', '沈黙', '息をひそ', '潜', '身を隠', '無音', '闇', 'しじま'],
};

export type Matrix = number[][];
export type ProjectorMethod = 'svd' | 'tikhonov';

export function identity(n = WORLDCRAFT_DIM): Matrix {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

export function matVec(m: Matrix, v: readonly number[]): number[] {
  return m.map((row) => row.reduce((acc, x, j) => acc + x * (v[j] ?? 0), 0));
}

export function matMul(a: Matrix, b: Matrix): Matrix {
  const n = a.length;
  const m = b[0]?.length ?? 0;
  const inner = b.length;
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i]![k]!;
      if (aik === 0) continue;
      for (let j = 0; j < m; j++) out[i]![j]! += aik * b[k]![j]!;
    }
  }
  return out;
}

/** Rank tolerance: max(k, n) * machine_eps * s_max. */
export function svdRankTolerance(k: number, n: number, sMax: number): number {
  return Math.max(k, n) * MACHINE_EPS * sMax;
}

export interface JacobiSvd {
  /** Singular values (column norms of J V), unsorted, length n. */
  singularValues: number[];
  /** Right singular vectors V (n x n), column i pairs with singularValues[i]. */
  v: Matrix;
  /** Columns of J V (k x n); column i = s_i * u_i. */
  jv: Matrix;
  sweeps: number;
}

/**
 * One-sided Jacobi (Hestenes) SVD of J (k x n). Deterministic: fixed sweep order (p<q),
 * a column pair is skipped when |a_p . a_q| <= 1e-15 sqrt(|a_p|^2 |a_q|^2).
 */
export function jacobiSvd(j: Matrix, n = WORLDCRAFT_DIM, maxSweeps = 60): JacobiSvd {
  const k = j.length;
  const a: Matrix = j.map((row) => row.slice(0, n));
  const v = identity(n);
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let i = 0; i < k; i++) {
          const ap = a[i]![p]!;
          const aq = a[i]![q]!;
          alpha += ap * ap;
          beta += aq * aq;
          gamma += ap * aq;
        }
        if (gamma === 0 || Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let i = 0; i < k; i++) {
          const ap = a[i]![p]!;
          const aq = a[i]![q]!;
          a[i]![p] = c * ap - s * aq;
          a[i]![q] = s * ap + c * aq;
        }
        for (let i = 0; i < n; i++) {
          const vp = v[i]![p]!;
          const vq = v[i]![q]!;
          v[i]![p] = c * vp - s * vq;
          v[i]![q] = s * vp + c * vq;
        }
      }
    }
    if (!rotated) break;
  }
  const singularValues = new Array<number>(n).fill(0);
  for (let col = 0; col < n; col++) {
    let ss = 0;
    for (let i = 0; i < k; i++) ss += a[i]![col]! * a[i]![col]!;
    singularValues[col] = Math.sqrt(ss);
  }
  return { singularValues, v, jv: a, sweeps };
}

function normaliseJacobian(jacobianC: Matrix | null | undefined): Matrix | null {
  if (!jacobianC || jacobianC.length === 0) return null;
  for (const row of jacobianC) {
    if (row.length !== WORLDCRAFT_DIM) {
      throw new Error(`Constraint Jacobian must have 12 columns, got ${row.length}`);
    }
  }
  return jacobianC;
}

/** Exact projector P_null = I - J^+ J (SVD pseudo-inverse, rank tolerance max(k,12)*eps*s_max). */
export function computeNullSpaceProjectorSvd(jacobianC: Matrix | null | undefined, rtol?: number): Matrix {
  const j = normaliseJacobian(jacobianC);
  const eye = identity();
  if (!j) return eye;
  const k = j.length;
  if (k > WORLDCRAFT_DIM) throw new Error(`Constraint Jacobian must have k <= 12 rows, got ${k}`);
  const { singularValues: s, v, jv } = jacobiSvd(j);
  const sMax = Math.max(...s);
  if (sMax === 0) return eye;
  const tol = rtol === undefined ? svdRankTolerance(k, WORLDCRAFT_DIM, sMax) : rtol * sMax;

  // J^+ (12 x k) = sum_{s_i > tol} v_i (J v_i)^T / s_i^2
  const jPinv: Matrix = Array.from({ length: WORLDCRAFT_DIM }, () => new Array<number>(k).fill(0));
  for (let c = 0; c < WORLDCRAFT_DIM; c++) {
    const si = s[c]!;
    if (!(si > tol)) continue;
    const inv2 = 1 / (si * si);
    for (let r = 0; r < WORLDCRAFT_DIM; r++) {
      const vr = v[r]![c]!;
      for (let i = 0; i < k; i++) jPinv[r]![i]! += vr * jv[i]![c]! * inv2;
    }
  }
  const pinvJ = matMul(jPinv, j);
  return eye.map((row, r) => row.map((x, c) => x - pinvJ[r]![c]!));
}

/** Gauss-Jordan inverse with partial pivoting (for SPD Gram matrices, k <= 12). */
function invertSpd(m: Matrix): Matrix {
  const n = m.length;
  const aug: Matrix = m.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r]![col]!) > Math.abs(aug[piv]![col]!)) piv = r;
    if (aug[piv]![col] === 0) throw new Error('Singular Gram matrix');
    [aug[col], aug[piv]] = [aug[piv]!, aug[col]!];
    const d = aug[col]![col]!;
    for (let c = 0; c < 2 * n; c++) aug[col]![c]! /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r]![col]!;
      if (f === 0) continue;
      for (let c = 0; c < 2 * n; c++) aug[r]![c]! -= f * aug[col]![c]!;
    }
  }
  return aug.map((row) => row.slice(n));
}

/** Legacy regularised projector P = I - J^T (J J^T + eps I)^{-1} J (Python default; eps-biased). */
export function computeNullSpaceProjectorTikhonov(jacobianC: Matrix | null | undefined, eps = 1e-6): Matrix {
  const j = normaliseJacobian(jacobianC);
  const eye = identity();
  if (!j) return eye;
  const k = j.length;
  const gram: Matrix = Array.from({ length: k }, (_, a) =>
    Array.from({ length: k }, (_, b) => {
      let acc = a === b ? eps : 0;
      for (let c = 0; c < WORLDCRAFT_DIM; c++) acc += j[a]![c]! * j[b]![c]!;
      return acc;
    }),
  );
  const gInv = invertSpd(gram);
  const jt: Matrix = Array.from({ length: WORLDCRAFT_DIM }, (_, r) => j.map((row) => row[r]!));
  const corr = matMul(matMul(jt, gInv), j);
  return eye.map((row, r) => row.map((x, c) => x - corr[r]![c]!));
}

export function buildGeneralJacobian(rows: readonly (readonly number[])[]): Matrix {
  if (rows.length > WORLDCRAFT_DIM) throw new Error(`k must be <= 12, got ${rows.length}`);
  for (const r of rows) if (r.length !== WORLDCRAFT_DIM) throw new Error('rows must form a k x 12 matrix');
  return rows.map((r) => [...r]);
}

export function compileConstraintJacobian(contextText = '', activeConstraints?: readonly string[] | null): Matrix {
  let modes: string[];
  if (activeConstraints === undefined || activeConstraints === null) {
    modes = Object.entries(CONSTRAINT_MODE_CUES)
      .filter(([, cues]) => cues.some((c) => contextText.includes(c)))
      .map(([m]) => m);
  } else {
    modes = [...activeConstraints];
    const unknown = modes.filter((m) => !(m in CONSTRAINT_MODE_ELEMENTS));
    if (unknown.length > 0) {
      throw new Error(
        `Unknown constraint mode(s): ${unknown.join(',')}; known: ${Object.keys(CONSTRAINT_MODE_ELEMENTS).sort().join(',')}`,
      );
    }
  }
  const indices = [...new Set(modes.flatMap((m) => [...CONSTRAINT_MODE_ELEMENTS[m]!]))].sort((a, b) => a - b);
  return indices.map((idx) => {
    const row = new Array<number>(WORLDCRAFT_DIM).fill(0);
    row[idx] = 1;
    return row;
  });
}

export interface ArbitrationResult {
  safe: number[];
  cut: number[];
  pNull: Matrix;
}

export class WorldCraftArbitrator {
  public readonly projectorMethod: ProjectorMethod;

  constructor(options?: { projectorMethod?: ProjectorMethod }) {
    this.projectorMethod = options?.projectorMethod ?? 'svd';
  }

  compileConstraintJacobian(contextText = '', activeConstraints?: readonly string[] | null): Matrix {
    return compileConstraintJacobian(contextText, activeConstraints);
  }

  computeNullSpaceProjector(jacobianC: Matrix | null | undefined, method?: ProjectorMethod, eps = 1e-6): Matrix {
    const use = method ?? this.projectorMethod;
    return use === 'svd' ? computeNullSpaceProjectorSvd(jacobianC) : computeNullSpaceProjectorTikhonov(jacobianC, eps);
  }

  arbitrateMomentum(
    targetDeltaQ: readonly number[],
    jacobianC: Matrix | null | undefined,
    options?: { method?: ProjectorMethod; eps?: number },
  ): ArbitrationResult {
    if (targetDeltaQ.length !== WORLDCRAFT_DIM) {
      throw new Error(`target_delta_q must have 12 elements, got ${targetDeltaQ.length}`);
    }
    const pNull = this.computeNullSpaceProjector(jacobianC, options?.method, options?.eps ?? 1e-6);
    const safe = matVec(pNull, targetDeltaQ);
    const cut = targetDeltaQ.map((x, i) => x - safe[i]!);
    return { safe, cut, pNull };
  }
}
