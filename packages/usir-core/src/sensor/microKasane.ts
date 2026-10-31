import { clip } from './dictionaries.js';
import type { IntermediateSensorData } from './types.js';

export interface MicroKasaneOutput {
  sg01: number[]; // Gr (密: Flux)
  sg02: number[]; // Cr (晶: Lattice)
  sg03: number[]; // Pr (孔: Void)
  sg04: number[]; // Jt (揺: Shock)
}

export function computeMicroKasane(data: IntermediateSensorData): MicroKasaneOutput {
  const {
    M,
    concrete_d,
    kango_compound_r,
    ent,
    c_cv,
    formal_rate,
    kanji_r,
    exclam_rate,
    wago_simplex_r,
    brevity,
    kango_light,
    quietude,
    slop_suppress,
    voiced_r,
    rhet_rate,
    fig_rate,
    abstract_b,
  } = data;

  const sg01: number[] = new Array(M);
  const sg02: number[] = new Array(M);
  const sg03: number[] = new Array(M);
  const sg04: number[] = new Array(M);

  for (let i = 0; i < M; i++) {
    // Sg-01 Gr (密)
    sg01[i] = clip(
      concrete_d[i]! * 1.90 + kango_compound_r[i]! * 0.95 + ent * 0.20,
      0.0,
      1.0
    );

    // Sg-02 Cr (晶)
    sg02[i] = clip(
      0.55 - 0.35 * c_cv + 0.48 * formal_rate + 0.20 * kanji_r[i]! - 0.35 * exclam_rate,
      0.0,
      1.0
    );

    // Sg-03 Pr (孔)
    sg03[i] = clip(
      (0.56 * wago_simplex_r[i]! * 2.5 + 0.44 * brevity[i]!) *
        kango_light[i]! *
        quietude *
        slop_suppress[i]!,
      0.0,
      1.0
    );

    // Sg-04 Jt (揺)
    sg04[i] = clip(
      (voiced_r[i]! * 2.4 +
        0.36 * rhet_rate[i]! +
        0.25 * fig_rate[i]! +
        0.30 * Math.min(exclam_rate, 1.5)) *
        (1.0 - 0.70 * Math.min(1.0, formal_rate)) -
        abstract_b[i]! * 1.5,
      0.0,
      1.0
    );
  }

  return { sg01, sg02, sg03, sg04 };
}
