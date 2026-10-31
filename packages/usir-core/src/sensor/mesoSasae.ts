import { clip } from './dictionaries.js';
import type { IntermediateSensorData } from './types.js';

export interface MesoSasaeOutput {
  sg05: number[]; // Cp (層: Flux)
  sg06: number[]; // Sy (対: Lattice)
  sg07: number[]; // Cs (切: Void)
  sg08: number[]; // Lp (跳: Shock)
}

export function computeMesoSasae(data: IntermediateSensorData): MesoSasaeOutput {
  const {
    M,
    clause_counts,
    kango_compound_r,
    formal_rate,
    kanji_r,
    exclam_rate,
    brevity,
    sluggish,
    kango_light,
    quietude,
    slop_suppress,
    void_op_rate,
    stem_sets,
    fig_rate,
    rhet_rate,
    abstract_b,
  } = data;

  const sg05: number[] = new Array(M);
  const sg06: number[] = new Array(M);
  const sg07: number[] = new Array(M);
  const sg08: number[] = new Array(M);

  for (let i = 0; i < M; i++) {
    // Sg-05 Cp (層)
    sg05[i] = clip(
      (clause_counts[i]! - 1.0) / 3.8 + kango_compound_r[i]! * 0.85,
      0.0,
      1.0
    );

    // Sg-06 Sy (対)
    sg06[i] = clip(
      0.48 +
        0.45 * formal_rate +
        0.28 * kanji_r[i]! -
        clip(exclam_rate * 0.65, 0.0, 0.80),
      0.0,
      1.0
    );

    // Sg-07 Cs (切)
    const clean_cut = clip(
      brevity[i]! * 0.45 + (1.0 - sluggish[i]!) * 0.55,
      0.0,
      1.0
    );
    sg07[i] = clip(
      clean_cut *
        kango_light[i]! *
        quietude *
        slop_suppress[i]! *
        (0.65 + 0.55 * void_op_rate[i]!),
      0.0,
      1.0
    );

    // Sg-08 Lp (跳)
    const s_curr = stem_sets[i]!;
    const s_prev = i > 0 ? stem_sets[i - 1]! : new Set<string>();

    let intersectionCount = 0;
    for (const stem of s_curr) {
      if (s_prev.has(stem)) {
        intersectionCount++;
      }
    }
    const unionCount = s_curr.size + s_prev.size - intersectionCount;
    const j_dist =
      unionCount > 0
        ? 1.0 - intersectionCount / unionCount
        : 0.2;

    const fig_boost =
      0.45 * fig_rate[i]! +
      0.38 * rhet_rate[i]! +
      0.30 * kango_compound_r[i]!;

    sg08[i] = clip(
      (j_dist * 0.20 + fig_boost) *
        (1.0 - 0.80 * Math.min(1.0, formal_rate)) -
        abstract_b[i]! * 1.6,
      0.0,
      1.0
    );
  }

  return { sg05, sg06, sg07, sg08 };
}
