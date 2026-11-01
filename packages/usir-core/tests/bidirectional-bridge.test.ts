import { describe, it, expect } from 'vitest';
import {
  pinBoundaryAnchors,
  solveBidirectionalCollisionBridge,
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

    expect(res.welded_trajectory.length).toBeGreaterThanOrEqual(4);
    expect(res.welded_text.length).toBeGreaterThan(50);
  });

  it('solves collision on analytical article domain', () => {
    const anchors = pinBoundaryAnchors('組織の非同期コミュニケーションと生産性', 'article');
    const res = solveBidirectionalCollisionBridge(anchors, { k_candidates: 4 });

    expect(res.domain).toBe('article');
    expect(res.optimal_pair[0]).toBeGreaterThanOrEqual(0);
    expect(res.optimal_pair[1]).toBeGreaterThanOrEqual(0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.welded_text).toContain('コミュニケーション');
  });

  it('solves collision on CoT reasoning domain', () => {
    const anchors = pinBoundaryAnchors('コンパクト群上の不変測度の存在証明', 'cot_reasoning');
    const res = solveBidirectionalCollisionBridge(anchors, { k_candidates: 4 });

    expect(res.domain).toBe('cot_reasoning');
    expect(res.min_action).toBeLessThan(10.0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.welded_trajectory.some((s) => s.includes('ステップ'))).toBe(true);
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
});
