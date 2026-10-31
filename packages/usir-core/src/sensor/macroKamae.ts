import { clip, mean } from './dictionaries.js';
import type { IntermediateSensorData } from './types.js';

export interface MacroKamaeOutput {
  sg09: number[]; // Dy (幅: Flux)
  sg10: number[]; // At (帰: Lattice)
  sg11: number[]; // Ma (間: Void)
  sg12: number[]; // Ru (破: Shock)
}

export function computeMacroKamae(
  data: IntermediateSensorData,
  sg03: number[]
): MacroKamaeOutput {
  const {
    M,
    lengths,
    mean_len,
    max_min_r,
    concrete_d,
    stem_sets,
    formal_rate,
    exclam_rate,
    wago_simplex_r,
    void_op_rate,
    fig_rate,
    rhet_rate,
    kango_light,
    quietude,
    slop_suppress,
    kango_compound_r,
    abstract_b,
  } = data;

  const sg09: number[] = new Array(M);
  const sg10: number[] = new Array(M);
  const sg11: number[] = new Array(M);
  const sg12: number[] = new Array(M);

  // Sg-10 (At: 帰) calculation across first and last thirds
  const third = Math.max(Math.floor(M / 3), 1);
  const open_s = new Set<string>();
  for (let i = 0; i < third; i++) {
    for (const s of stem_sets[i]!) {
      open_s.add(s);
    }
  }

  const close_s = new Set<string>();
  for (let i = Math.max(0, M - third); i < M; i++) {
    for (const s of stem_sets[i]!) {
      close_s.add(s);
    }
  }

  let inter_s = 0;
  for (const s of open_s) {
    if (close_s.has(s)) {
      inter_s++;
    }
  }
  const union_s = open_s.size + close_s.size - inter_s;
  const rec = union_s > 0 ? inter_s / union_s : 0.0;

  const sg10_val = clip(
    0.42 +
      0.45 * Math.min(1.0, rec * 3.2) +
      0.35 * Math.min(1.0, formal_rate) -
      0.40 * Math.min(exclam_rate, 1.0),
    0.0,
    1.0
  );

  // Sg-11 (Ma: 間) base
  const wago_purity = Math.min(
    1.0,
    mean(wago_simplex_r) * 2.35 + mean(void_op_rate) * 0.65
  );
  const macro_subtraction = Math.exp(-0.65 * (mean(fig_rate) + mean(rhet_rate)));
  const sg11_base = clip(
    mean(kango_light) *
      quietude *
      macro_subtraction *
      wago_purity *
      mean(slop_suppress) *
      1.55,
    0.0,
    1.0
  );

  for (let i = 0; i < M; i++) {
    // Sg-09 Dy (幅)
    sg09[i] = clip(
      0.32 * max_min_r +
        (0.45 * Math.abs(lengths[i]! - mean_len)) / (mean_len + 1e-5) +
        0.25 * concrete_d[i]!,
      0.0,
      1.0
    );

    // Sg-10 At (帰)
    sg10[i] = sg10_val;

    // Sg-11 Ma (間)
    sg11[i] = clip(sg11_base * (0.72 + 0.28 * sg03[i]!), 0.0, 1.0);

    // Sg-12 Ru (破)
    sg12[i] = clip(
      (0.42 * rhet_rate[i]! +
        0.38 * fig_rate[i]! +
        0.32 * kango_compound_r[i]! +
        0.38 * Math.min(exclam_rate, 2.0)) *
        (1.0 - 0.80 * Math.min(1.0, formal_rate)) -
        abstract_b[i]! * 1.6,
      0.0,
      1.0
    );
  }

  return { sg09, sg10, sg11, sg12 };
}
