import { describe, it, expect } from 'vitest';
import {
  pinBoundaryAnchors,
  solveBidirectionalCollisionBridge,
  solveTensorCollisionBridge,
  deriveEndogenousGoalHypotheses,
  executeClosedLoopRewrite,
} from '../src/dynamics/bidirectionalBridge.js';

describe('USIR v4.0 Bidirectional Schrödinger Bridge (TS)', () => {
  it('pins boundary anchors across novel, article, and cot_reasoning domains', () => {
    for (const domain of ['novel', 'article', 'cot_reasoning'] as const) {
      const anchors = pinBoundaryAnchors('夜明けの雪山', domain);
      expect(anchors.domain).toBe(domain);
      expect(anchors.x_0.length).toBeGreaterThan(10);
      expect(anchors.x_T.length).toBeGreaterThan(10);
      expect(anchors.q_start.length).toBe(24);
      expect(anchors.q_end.length).toBe(24);
      expect(anchors.kamae_coords.length).toBe(3);
    }
  });

  it('allows custom boundary overrides', () => {
    const customStart = '凍てつく湖の岸辺に、旅人は馬を止めた。';
    const customEnd = '薄氷が静かに割れ、水底へと青い光が吸い込まれていった。';
    const anchors = pinBoundaryAnchors('氷湖', 'novel', {
      custom_start: customStart,
      custom_end: customEnd,
    });
    expect(anchors.x_0).toBe(customStart);
    expect(anchors.x_T).toBe(customEnd);
  });

  it('solves bidirectional collision on a novel theme and welds midpoint', () => {
    const anchors = pinBoundaryAnchors('雨夜の湯治場と老陶工', 'novel');
    const res = solveBidirectionalCollisionBridge(anchors, { k_candidates: 4 });

    expect(res.domain).toBe('novel');
    expect(res.forward_waves.length).toBe(4);
    expect(res.backward_waves.length).toBe(4);
    expect(res.collision_matrix.length).toBe(4);
    expect(res.collision_matrix[0]!.length).toBe(4);
    expect(res.min_action).toBeGreaterThan(0.0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.m_bridge).toBeLessThanOrEqual(1.0);

    const [optI, optJ] = res.optimal_pair;
    expect(res.collision_matrix[optI]![optJ]).toBe(res.min_action);

    expect(res.welded_trajectory.length).toBeGreaterThanOrEqual(6);
    expect(res.welded_text.length).toBeGreaterThanOrEqual(400);
  });

  it('solves collision on analytical article domain', () => {
    const anchors = pinBoundaryAnchors('組織の非同期コミュニケーションと生産性', 'article');
    const res = solveBidirectionalCollisionBridge(anchors, { k_candidates: 4 });

    expect(res.domain).toBe('article');
    expect(res.optimal_pair[0]).toBeGreaterThanOrEqual(0);
    expect(res.optimal_pair[1]).toBeGreaterThanOrEqual(0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.welded_text).toContain('コミュニケーション');
    expect(res.welded_text.length).toBeGreaterThanOrEqual(400);
  });

  it('solves collision on CoT reasoning domain', () => {
    const anchors = pinBoundaryAnchors('コンパクト群上の不変測度の存在証明', 'cot_reasoning');
    const res = solveBidirectionalCollisionBridge(anchors, { k_candidates: 4 });

    expect(res.domain).toBe('cot_reasoning');
    expect(res.min_action).toBeLessThan(10.0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.welded_trajectory.some((s) => s.includes('ステップ'))).toBe(true);
    expect(res.welded_text.length).toBeGreaterThanOrEqual(400);
  });

  it('throws on empty or invalid candidate arrays', () => {
    const anchors = pinBoundaryAnchors('テスト', 'novel');
    expect(() =>
      solveBidirectionalCollisionBridge(anchors, { forward_candidates: [], backward_candidates: [] })
    ).toThrow(/non-empty arrays/);
    expect(() =>
      solveBidirectionalCollisionBridge(anchors, { forward_candidates: [[]], backward_candidates: [[]] })
    ).toThrow(/at least one non-empty candidate/);
  });

  it('penalizes topic-disconnected waves with higher action', () => {
    const anchors = pinBoundaryAnchors('テスト', 'novel');

    const fwdCoherent = [
      ['山道を登る。', '杉の木立から木漏れ日が差した。'],
    ];
    const bwdCoherent = [
      ['木漏れ日が土を温める。', '頂上の社にたどり着いた。'],
    ];
    const bwdIncoherent = [
      ['宇宙ステーションの軌道が修正された。', '火星の探査機が着陸した。'],
    ];

    const resGood = solveBidirectionalCollisionBridge(anchors, {
      forward_candidates: fwdCoherent,
      backward_candidates: bwdCoherent,
    });
    const resBad = solveBidirectionalCollisionBridge(anchors, {
      forward_candidates: fwdCoherent,
      backward_candidates: bwdIncoherent,
    });

    expect(resGood.min_action).toBeLessThan(resBad.min_action);
    expect(resGood.m_bridge).toBeGreaterThan(resBad.m_bridge);
  });

  it('solves abstract tensor collision bridge verifying pure Riemannian metric properties', () => {
    const qStart = new Array(24).fill(0.1);
    const qEnd = new Array(24).fill(0.9);

    // Trajectory 1: smoothly progresses toward midpoint
    const fwdAligned = [
      [new Array(24).fill(0.1), new Array(24).fill(0.3), new Array(24).fill(0.5)],
    ];
    const bwdAligned = [
      [new Array(24).fill(0.52), new Array(24).fill(0.7), new Array(24).fill(0.9)],
    ];

    // Trajectory 2: divergent/orthogonal in latent space
    const bwdDivergent = [
      [new Array(24).fill(-0.8), new Array(24).fill(0.0), new Array(24).fill(0.9)],
    ];

    const resAligned = solveTensorCollisionBridge(qStart, qEnd, fwdAligned, bwdAligned);
    const resDivergent = solveTensorCollisionBridge(qStart, qEnd, fwdAligned, bwdDivergent);

    expect(resAligned.min_action).toBeGreaterThan(0.0);
    expect(resAligned.min_action).toBeLessThan(resDivergent.min_action);
    expect(resAligned.m_bridge).toBeGreaterThan(resDivergent.m_bridge);
    expect(resAligned.action_matrix.length).toBe(1);
    expect(resAligned.action_matrix[0]!.length).toBe(1);
    expect(resAligned.welded_tensor.length).toBe(6);
  });

  it('derives endogenous goal hypothesis (Mode B) without manual end pinning', () => {
    const x0 = '未明の静寂のなかで、谷あいの水車が微かに軋む音を立てていた。';
    const { optimal_x_T, scored_candidates } = deriveEndogenousGoalHypotheses(x0, 'novel', { k_candidates: 3 });

    expect(typeof optimal_x_T).toBe('string');
    expect(optimal_x_T.length).toBeGreaterThan(10);
    expect(scored_candidates.length).toBe(3);
    expect(scored_candidates[0]!.score).toBeLessThanOrEqual(scored_candidates[1]!.score);
  });

  it('executes closed-loop rewrite (Mode A) on a drifting draft to weld intermediate drift', () => {
    const draft =
      '電車の事故で怪我を負った私は、温泉宿で療養することにした。' +
      '部屋の窓から外を眺めると、雲がゆっくりと流れていた。' +
      '夕方になって鳥が鳴き、やがて空は暗くなっていった。' +
      '微かな軋みとともに輪郭が沈み、あとに冷たい沈黙だけが残った。';

    const res = executeClosedLoopRewrite(draft, 'novel');
    expect(res.anchors.x_0).toContain('怪我を負った');
    expect(res.anchors.x_T).toContain('冷たい沈黙だけが残った');
    expect(res.min_action).toBeGreaterThan(0.0);
    expect(res.welded_text.length).toBeGreaterThan(50);
  });

  it('executes closed-loop rewrite on single sentence draft avoiding self-loop', () => {
    const singleSentence = '冷たい雨が古びた瓦屋根を激しく打ち鳴らしていた。';
    const res = executeClosedLoopRewrite(singleSentence, 'novel');
    expect(res.anchors.x_0).toBe(singleSentence);
    expect(res.anchors.x_T).not.toBe(singleSentence);
    expect(res.anchors.x_T.length).toBeGreaterThan(10);
    expect(res.welded_text.length).toBeGreaterThanOrEqual(300);
  });
});
