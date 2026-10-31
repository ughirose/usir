/**
 * 24-Dimensional Symplectic Feature Extractor and Order Parameter Engine for USIR.
 * Fully reproduces the mathematical invariants of Python extractor_24d.py.
 */

import {
  ABSTRACT_FILLERS,
  APOPHATIC_VOID_OPS,
  FIGURATIVE_MARKERS,
  FORMAL_LOGIC_TERMS,
  PHYSICAL_SENSE_CHARS,
  RHETORICAL_RUPTURE,
  VOICED_CHARS,
  charBigramEntropy,
  clip,
  countSubstr,
  extractKanjiStems,
  isHiragana,
  isKanji,
  mean,
  splitSentences,
  stdPop,
} from './dictionaries.js';
import { computeMicroKasane } from './microKasane.js';
import { computeMesoSasae } from './mesoSasae.js';
import { computeMacroKamae } from './macroKamae.js';
import type { IntermediateSensorData } from './types.js';

export interface Usir24dResult {
  q: number[];
  p: number[];
  z_24d: number[];
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
  abst_bloat: number;
  concrete_dens: number;
  abstract_b_mean: number;
  sluggish_mean: number;
  cadence_alive: number;
}

export function extractUsir24d(text: string): Usir24dResult {
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    const qZeros = new Array(12).fill(0.0);
    const pZeros = new Array(12).fill(0.0);
    return {
      q: qZeros,
      p: pZeros,
      z_24d: new Array(24).fill(0.0),
      groups: { I_Flux: 0.0, II_Lattice: 0.0, III_Void: 0.0, IV_Shock: 0.0 },
      order_params: { psi_order: 0.0, phi_void: 0.0, kappa_kin: 0.04 },
      len_cv: 0.0,
      abst_bloat: 0.0,
      concrete_dens: 0.0,
      abstract_b_mean: 0.0,
      sluggish_mean: 0.0,
      cadence_alive: 1.0,
    };
  }

  let sents = splitSentences(text);
  if (sents.length < 3) {
    sents = [];
    for (let i = 0; i < text.length; i += 30) {
      const chunk = text.slice(i, i + 30);
      if (chunk.length >= 5) {
        sents.push(chunk);
      }
    }
    if (sents.length === 0) {
      sents = [text];
    }
  }

  const M = sents.length;
  const lengths = sents.map((s) => s.length);
  const clause_counts = sents.map((s) => countSubstr(s, '、') + 1);

  const clauses_all: string[] = [];
  for (const s of sents) {
    for (const c of s.split('、')) {
      const ct = c.trim();
      if (ct.length > 0) clauses_all.push(ct);
    }
  }
  const c_lens = clauses_all.length > 0 ? clauses_all.map((c) => c.length) : [12.0];

  const exclam_rate =
    (countSubstr(text, '！') + countSubstr(text, '？') + countSubstr(text, '!')) /
    Math.max(M, 1);

  let formal_count = 0;
  for (const w of FORMAL_LOGIC_TERMS) {
    formal_count += countSubstr(text, w);
  }
  const formal_rate = formal_count / Math.max(M, 1);

  const mean_len = mean(lengths);
  const len_cv = stdPop(lengths) / (mean_len + 1e-5);
  const c_cv = stdPop(c_lens) / (mean(c_lens) + 1e-5);

  const kanji_r = new Array<number>(M);
  const hira_r = new Array<number>(M);
  const wago_simplex_r = new Array<number>(M);
  const kango_compound_r = new Array<number>(M);
  const concrete_d = new Array<number>(M);
  const abstract_b = new Array<number>(M);
  const voiced_r = new Array<number>(M);
  const fig_rate = new Array<number>(M);
  const rhet_rate = new Array<number>(M);
  const void_op_rate = new Array<number>(M);
  const stem_sets: Set<string>[] = [];

  for (let i = 0; i < M; i++) {
    const s = sents[i]!;
    const L = Math.max(s.length, 1);

    let k_cnt = 0;
    let h_cnt = 0;
    let phys_cnt = 0;
    let v_cnt = 0;

    for (let j = 0; j < s.length; j++) {
      const ch = s[j]!;
      if (isKanji(ch)) k_cnt++;
      if (isHiragana(ch)) h_cnt++;
      if (PHYSICAL_SENSE_CHARS.has(ch)) phys_cnt++;
      if (VOICED_CHARS.has(ch)) v_cnt++;
    }

    v_cnt += (countSubstr(s, 'っ') + countSubstr(s, 'ッ')) * 1.5;

    kanji_r[i] = k_cnt / L;
    hira_r[i] = h_cnt / L;

    const runs = extractKanjiStems(s);
    const n_runs = Math.max(runs.length, 1);
    let one_char_runs = 0;
    let multi_char_chars = 0;
    let long_compounds = 0;

    for (const r of runs) {
      if (r.length === 1) one_char_runs++;
      if (r.length >= 2) multi_char_chars += r.length;
      if (r.length >= 3) long_compounds++;
    }

    wago_simplex_r[i] = (one_char_runs / n_runs) * (h_cnt / L);
    kango_compound_r[i] = multi_char_chars / L + 0.16 * long_compounds;

    let abst_cnt = 0;
    for (const w of ABSTRACT_FILLERS) {
      abst_cnt += countSubstr(s, w);
    }

    concrete_d[i] = (phys_cnt * 3.5 + k_cnt * 0.35) / L;
    abstract_b[i] = (abst_cnt * 8.5) / L;
    voiced_r[i] = v_cnt / L;

    let fig_cnt = 0;
    for (const w of FIGURATIVE_MARKERS) {
      fig_cnt += countSubstr(s, w);
    }
    fig_rate[i] = fig_cnt;

    let rhet_cnt = 0;
    for (const w of RHETORICAL_RUPTURE) {
      rhet_cnt += countSubstr(s, w);
    }
    rhet_rate[i] = rhet_cnt + 0.35 * long_compounds;

    let void_cnt = 0;
    for (const w of APOPHATIC_VOID_OPS) {
      void_cnt += countSubstr(s, w);
    }
    void_op_rate[i] = void_cnt;

    stem_sets.push(new Set(runs));
  }

  // rhet_rate = rhet_rate + exclam_rate * 0.85
  for (let i = 0; i < M; i++) {
    rhet_rate[i] = (rhet_rate[i] ?? 0.0) + exclam_rate * 0.85;
  }

  const sluggish = sents.map((s) => {
    return s.endsWith('のだった') ||
      s.endsWith('であった') ||
      s.endsWith('思われた') ||
      s.endsWith('感じられた') ||
      s.endsWith('ていた')
      ? 1.0
      : 0.0;
  });

  const abstract_b_mean = mean(abstract_b);
  const sluggish_mean = mean(sluggish);

  // Alive cadence gate
  const is_flat_slop =
    mean_len > 40.0 &&
    len_cv < 0.28 &&
    (abstract_b_mean > 0.015 || sluggish_mean > 0.15)
      ? 1.0
      : 0.0;
  const slop_deadness = is_flat_slop * 0.68 + abstract_b_mean * 2.6;
  const cadence_alive = clip(1.0 - slop_deadness, 0.08, 1.0);

  // Shared variables for groups
  const brevity = lengths.map((l) => 1.0 / (1.0 + Math.pow(l / 31.0, 2.2)));
  const kango_light = kango_compound_r.map((kc) => Math.exp(-2.4 * kc));
  const quietude =
    Math.exp(-1.5 * exclam_rate) * (1.0 - 0.85 * Math.min(1.0, formal_rate));
  const slop_suppress = abstract_b.map((ab) => Math.exp(-4.5 * ab));
  const ent = charBigramEntropy(text);

  const max_len = Math.max(...lengths);
  const min_len = Math.min(...lengths);
  const max_min_r = (max_len - min_len) / (mean_len + 1e-5);

  const interData: IntermediateSensorData = {
    text,
    sents,
    M,
    lengths,
    mean_len,
    len_cv,
    clause_counts,
    c_lens,
    c_cv,
    exclam_rate,
    formal_rate,
    kanji_r,
    hira_r,
    wago_simplex_r,
    kango_compound_r,
    concrete_d,
    abstract_b,
    voiced_r,
    fig_rate,
    rhet_rate,
    void_op_rate,
    stem_sets,
    sluggish,
    brevity,
    kango_light,
    quietude,
    slop_suppress,
    ent,
    max_min_r,
    cadence_alive,
    abstract_b_mean,
    sluggish_mean,
    concrete_d_mean: mean(concrete_d),
  };

  const { sg01, sg02, sg03, sg04 } = computeMicroKasane(interData);
  const { sg05, sg06, sg07, sg08 } = computeMesoSasae(interData);
  const { sg09, sg10, sg11, sg12 } = computeMacroKamae(interData, sg03);

  const elem_matrix: number[][] = [
    sg01, sg02, sg03, sg04,
    sg05, sg06, sg07, sg08,
    sg09, sg10, sg11, sg12,
  ];

  const q = elem_matrix.map((row) => mean(row));

  // Conjugate momentum p in R^12
  const adj_diff = new Array<number>(M);
  adj_diff[0] = 0.0;
  for (let i = 1; i < M; i++) {
    adj_diff[i] = Math.abs(lengths[i]! - lengths[i - 1]!) / (mean_len + 1e-5);
  }

  const rms_vel = new Array<number>(12).fill(0.0);
  if (M > 1) {
    for (let k = 0; k < 12; k++) {
      let sumSq = 0.0;
      const row = elem_matrix[k]!;
      for (let j = 0; j < M - 1; j++) {
        const diff = row[j + 1]! - row[j]!;
        sumSq += diff * diff;
      }
      rms_vel[k] = Math.sqrt(sumSq / (M - 1));
    }
  }

  const fig_plus_rhet = new Array<number>(M);
  for (let i = 0; i < M; i++) {
    fig_plus_rhet[i] = fig_rate[i]! + rhet_rate[i]!;
  }

  const p = new Array<number>(12);
  p[0] = (rms_vel[0]! * 1.4 + stdPop(concrete_d) * 2.0) * cadence_alive;
  p[1] = (rms_vel[1]! * 1.4 + c_cv * 0.45) * cadence_alive;
  p[2] = (rms_vel[2]! * 1.4 + stdPop(wago_simplex_r) * 1.8) * cadence_alive;
  p[3] = (rms_vel[3]! * 1.4 + stdPop(voiced_r) * 2.8) * cadence_alive;
  p[4] = (rms_vel[4]! * 1.4 + stdPop(clause_counts) * 0.40) * cadence_alive;
  p[5] = (rms_vel[5]! * 1.4 + stdPop(kanji_r) * 2.2) * cadence_alive;
  p[6] = (rms_vel[6]! * 1.4 + mean(adj_diff) * 0.75) * cadence_alive;
  p[7] = (rms_vel[7]! * 1.4 + stdPop(fig_plus_rhet) * 0.55) * cadence_alive;
  p[8] = (len_cv * 1.05 + max_min_r * 0.25) * cadence_alive;
  p[9] =
    Math.exp(-Math.pow((len_cv - 0.52) / 0.22, 2)) * 0.90 * cadence_alive;
  p[10] = len_cv * 0.80 * mean(kango_light) * cadence_alive;
  p[11] =
    (Math.max(...adj_diff) * 0.55 + stdPop(fig_plus_rhet) * 0.45) * cadence_alive;

  const z_24d = [...q, ...p];

  const G_flux = mean([q[0]!, q[4]!, q[8]!]);
  const G_lattice = mean([q[1]!, q[5]!, q[9]!]);
  const G_void = mean([q[2]!, q[6]!, q[10]!]);
  const G_shock = mean([q[3]!, q[7]!, q[11]!]);

  const psi_order = Math.tanh(
    2.4 * (G_lattice + 0.35 * formal_rate - 1.35 * G_shock - 0.45 * exclam_rate)
  );
  const phi_void = clip(
    (G_void * 1.48 - 0.68 * G_shock) *
      (1.0 - 0.85 * Math.min(1.0, formal_rate)) -
      0.50 * abstract_b_mean * 3.0,
    0.0,
    1.0
  );

  const p_norm = Math.sqrt(p.reduce((acc, v) => acc + v * v, 0));
  const kappa_kin = Math.max(
    0.04,
    (p_norm * 0.62 + 1.10 * G_shock + 0.30 * exclam_rate) * cadence_alive -
      0.50 * abstract_b_mean * 2.0
  );

  return {
    q,
    p,
    z_24d,
    groups: {
      I_Flux: G_flux,
      II_Lattice: G_lattice,
      III_Void: G_void,
      IV_Shock: G_shock,
    },
    order_params: {
      psi_order,
      phi_void,
      kappa_kin,
    },
    len_cv,
    abst_bloat: abstract_b_mean,
    concrete_dens: mean(concrete_d),
    abstract_b_mean,
    sluggish_mean,
    cadence_alive,
  };
}
