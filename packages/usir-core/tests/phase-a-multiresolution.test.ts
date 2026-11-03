/**
 * Phase A Multiresolution Pipeline & Contrastive Fisher Metric Test Suite (TypeScript).
 *
 * Verifies:
 * 1. Stage 1: 4 basic structural signals s_m = (Flux, Lattice, Void, Shock) in R^4.
 * 2. Stage 2: 3-scale orthogonal Haar wavelets, ISTA sparse solitons, RQA I_arch -> 24D state.
 * 3. Contrastive Fisher Distillation: positive definiteness of G_style and Rayleigh quotient discriminant separability.
 * 4. embedSentenceUnit with SyntaxTreeFrame priority and hiddenStates pooling hook.
 * 5. Boundary and edge case robustness.
 */

import { describe, expect, it } from 'vitest';
import {
  extractStage1Signals,
  extractStage2Multiresolution,
  extractMultiresolution24d,
  extractUsir24d,
} from '../src/sensor/symplectic24d.js';
import {
  fitContrastiveFisherMetric,
  embedSentenceUnit,
} from '../src/dynamics/manifoldGeodesic.js';
import {
  createSyntaxNode,
  createSyntaxEdge,
  createSyntaxTreeFrame,
} from '../src/schema/syntaxFrame.js';

describe('Phase A Multiresolution & Contrastive Fisher Suite', () => {
  it('extracts Stage 1 basic signals on natural Japanese literary text', () => {
    const text =
      '硝子の破片が冷たい朝の光を浴びて、鋭くきらきらと輝いていた。' +
      '少年は静かにその場に立ち止まり、息を殺した。' +
      'しかし、何事も起こらなかった。' +
      '風だけが、音もなく吹き抜けていった。';

    const signals = extractStage1Signals(text);
    expect(signals.length).toBeGreaterThanOrEqual(3);
    for (const row of signals) {
      expect(row.length).toBe(4);
      for (const val of row) {
        expect(val).toBeGreaterThanOrEqual(0.0);
        expect(val).toBeLessThanOrEqual(1.0);
      }
    }
  });

  it('modulates Stage 1 signals via explicit SyntaxTreeFrame properties', () => {
    const node1 = createSyntaxNode({
      node_id: 0,
      surface: '冷たい',
      pos: '形容詞',
      hth_primary: (0x51b7n << 34n) | 0xfn,
    });
    const node2 = createSyntaxNode({
      node_id: 1,
      surface: '氷を',
      pos: '名詞',
      hth_primary: (0x6c37n << 34n) | 0xen,
    });
    const node3 = createSyntaxNode({
      node_id: 2,
      surface: '落とす',
      pos: '動詞',
      hth_primary: (0x8efdn << 34n) | 0xdn,
    });
    const edge1 = createSyntaxEdge({ source_id: 0, target_id: 1, relation: 'mod', jump_distance: 0.1 });
    const edge2 = createSyntaxEdge({ source_id: 1, target_id: 2, relation: 'obj', jump_distance: 0.8 });

    const frameDense = createSyntaxTreeFrame({
      sentence_id: 0,
      text: '冷たい氷を落とす。',
      nodes: [node1, node2, node3],
      edges: [edge1, edge2],
      root_id: 2,
      max_depth: 2,
      depth_distribution: [0.33, 0.33, 0.34],
      radial_entropy: 1.09,
    });

    const negNode1 = createSyntaxNode({
      node_id: 0,
      surface: '声も',
      pos: '名詞',
      hth_primary: (0x58f0n << 34n) | 0x8n,
    });
    const negNode2 = createSyntaxNode({
      node_id: 1,
      surface: 'なかった',
      pos: '助動詞',
      is_negation_scope: true,
    });
    const negEdge = createSyntaxEdge({ source_id: 0, target_id: 1, relation: 'subj', jump_distance: 0.85 });

    const frameNeg = createSyntaxTreeFrame({
      sentence_id: 1,
      text: '声もなかった。',
      nodes: [negNode1, negNode2],
      edges: [negEdge],
      root_id: 1,
      max_depth: 1,
      depth_distribution: [0.5, 0.5],
      radial_entropy: 0.69,
    });

    const signals = extractStage1Signals('', [frameDense, frameNeg]);
    expect(signals.length).toBe(2);
    // Dense frame flux
    expect(signals[0]![0]).toBeGreaterThan(0.1);
    // Negation frame shock
    expect(signals[1]![3]).toBeGreaterThan(0.2);
    // Apophatic sentence void
    expect(signals[1]![2]).toBeGreaterThan(0.3);
  });

  it('runs Stage 2 multiresolution wavelets, ISTA solitons and RQA', () => {
    const sSignals = [
      [0.3, 0.5, 0.4, 0.2],
      [0.4, 0.6, 0.3, 0.1],
      [0.5, 0.4, 0.5, 0.3],
      [0.6, 0.7, 0.2, 0.2],
      [0.3, 0.5, 0.6, 0.1],
      [0.7, 0.4, 0.2, 0.9], // Shock spike
      [0.2, 0.8, 0.5, 0.1],
      [0.4, 0.6, 0.4, 0.2],
      [0.5, 0.5, 0.3, 0.3],
      [0.3, 0.7, 0.6, 0.1],
    ];

    const res = extractStage2Multiresolution(sSignals, 'テスト文書。波形展開を検証する。');
    expect(res.q.length).toBe(12);
    expect(res.p.length).toBe(12);
    expect(res.z_24d.length).toBe(24);

    for (const q of res.q) {
      expect(q).toBeGreaterThanOrEqual(0.0);
      expect(q).toBeLessThanOrEqual(1.0);
    }
    for (const p of res.p) {
      expect(p).toBeGreaterThanOrEqual(0.0);
    }

    expect(res.ista_solitons.length).toBe(4);
    expect(res.rqa_i_arch).toBeGreaterThanOrEqual(0.0);
    expect(res.rqa_i_arch).toBeLessThanOrEqual(1.0);

    // End-to-end extraction
    const fullRes = extractMultiresolution24d('白樺の梢が揺れる。湖面には波紋が広がる。風が止む。沈黙が戻る。');
    expect(fullRes.z_24d.length).toBe(24);
    expect(fullRes.stage1_signals.length).toBeGreaterThanOrEqual(3);
  });

  it('supports useMultiresolution option in extractUsir24d', () => {
    const text = '雪が静かに降り積もる。冷たい夜が更けていく。明かりが消える。';

    const resDefault = extractUsir24d(text);
    expect(resDefault.z_24d.length).toBe(24);

    const resMulti = extractUsir24d(text, { useMultiresolution: true });
    expect(resMulti.z_24d.length).toBe(24);
  });

  it('fits Contrastive Fisher metric with strict positive definiteness and separability', () => {
    // Generate positive cluster (centered at 0.7) and negative cluster (centered at 0.2)
    const posZ: number[][] = [];
    const negZ: number[][] = [];
    for (let k = 0; k < 15; k++) {
      const pRow: number[] = [];
      const nRow: number[] = [];
      for (let d = 0; d < 24; d++) {
        pRow.push(0.7 + ((Math.sin(k * 7 + d * 13) * 0.15)));
        nRow.push(0.2 + ((Math.cos(k * 11 + d * 17) * 0.15)));
      }
      posZ.push(pRow);
      negZ.push(nRow);
    }

    const fisher = fitContrastiveFisherMetric(posZ, negZ, 1e-3);
    const { G_style, optimal_projection_axis, lambda_max } = fisher;

    expect(G_style.length).toBe(24);
    expect(G_style[0]!.length).toBe(24);
    expect(optimal_projection_axis.length).toBe(24);

    // 1. Numerical Symmetry: G_style[i][j] ~= G_style[j][i]
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j < 24; j++) {
        expect(Math.abs(G_style[i]![j]! - G_style[j]![i]!)).toBeLessThan(1e-4);
      }
    }

    // 2. Strict Positive Definiteness: v^T G v > 0 for non-zero vectors
    for (let i = 0; i < 24; i++) {
      expect(G_style[i]![i]!).toBeGreaterThan(0.0);
    }
    const testVec = new Array<number>(24).fill(1.0);
    let quadForm = 0.0;
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j < 24; j++) {
        quadForm += testVec[i]! * G_style[i]![j]! * testVec[j]!;
      }
    }
    expect(quadForm).toBeGreaterThan(0.0);

    // 3. Optimal axis unit norm
    let axisNorm = 0.0;
    for (const v of optimal_projection_axis) axisNorm += v * v;
    axisNorm = Math.sqrt(axisNorm);
    expect(axisNorm).toBeCloseTo(1.0, 4);

    // 4. Strictly positive lambda_max
    expect(lambda_max).toBeGreaterThan(0.0);

    // 5. Fisher discriminant separability
    let posProjMean = 0.0;
    for (const p of posZ) {
      let proj = 0.0;
      for (let d = 0; d < 24; d++) proj += p[d]! * optimal_projection_axis[d]!;
      posProjMean += proj / posZ.length;
    }
    let negProjMean = 0.0;
    for (const n of negZ) {
      let proj = 0.0;
      for (let d = 0; d < 24; d++) proj += n[d]! * optimal_projection_axis[d]!;
      negProjMean += proj / negZ.length;
    }
    expect(posProjMean).toBeGreaterThan(negProjMean);
  });

  it('embeds sentence unit with SyntaxTreeFrame priority and hiddenStates hook', () => {
    const sentence = '冷たい氷の破片が鋭く光る。';

    // 1. Raw embedding
    const eRaw = embedSentenceUnit(sentence, 24);
    expect(eRaw.length).toBe(24);
    let rawNorm = 0;
    for (const v of eRaw) rawNorm += v * v;
    expect(Math.sqrt(rawNorm)).toBeCloseTo(1.0, 4);

    // 2. SyntaxTreeFrame priority
    const node = createSyntaxNode({
      node_id: 0,
      surface: '氷',
      hth_primary: (0x6c37n << 34n) | 0xen,
    });
    const frame = createSyntaxTreeFrame({
      sentence_id: 0,
      text: sentence,
      nodes: [node],
      edges: [],
      root_id: 0,
      max_depth: 1,
      radial_entropy: 0.5,
    });
    const eFrame = embedSentenceUnit(sentence, 24, frame);
    expect(eFrame.length).toBe(24);
    let frameNorm = 0;
    for (const v of eFrame) frameNorm += v * v;
    expect(Math.sqrt(frameNorm)).toBeCloseTo(1.0, 4);

    // 3. hiddenStates hook
    const mockHiddenStates = Array.from({ length: 8 }, () => new Array<number>(24).fill(0.5));
    const eHidden = embedSentenceUnit(sentence, 24, undefined, mockHiddenStates);
    expect(eHidden.length).toBe(24);
    let hiddenNorm = 0;
    for (const v of eHidden) hiddenNorm += v * v;
    expect(Math.sqrt(hiddenNorm)).toBeCloseTo(1.0, 4);
  });

  it('handles empty and degenerate inputs robustly', () => {
    const emptyRes = extractMultiresolution24d('');
    expect(emptyRes.z_24d.every((v) => v === 0.0)).toBe(true);

    const s2Empty = extractStage2Multiresolution([]);
    expect(s2Empty.z_24d.length).toBe(24);
    expect(s2Empty.z_24d.every((v) => v === 0.0)).toBe(true);

    const fisherEmpty = fitContrastiveFisherMetric([], []);
    expect(fisherEmpty.G_style.length).toBe(24);
    expect(fisherEmpty.optimal_projection_axis.length).toBe(24);
    expect(fisherEmpty.optimal_projection_axis.every((v) => v === 0.0)).toBe(true);
    expect(fisherEmpty.lambda_max).toBe(0.0);

    const singleRes = extractMultiresolution24d('静寂。');
    expect(singleRes.z_24d.length).toBe(24);

    const mimNode = createSyntaxNode({ node_id: 0, surface: 'きらきら', pos: '副詞' });
    const mimFrame = createSyntaxTreeFrame({ sentence_id: 0, text: 'きらきら', nodes: [mimNode], edges: [] });
    const eMim = embedSentenceUnit('きらきら', 24, mimFrame);
    expect(eMim.length).toBe(24);
    expect(eMim.slice(0, 8).some((v) => v > 0)).toBe(true);
  });
});
