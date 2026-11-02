import { describe, it, expect } from 'vitest';
import {
  pinBoundaryAnchors,
  solveBidirectionalCollisionBridge,
  solveTensorCollisionBridge,
  deriveEndogenousGoalHypotheses,
  executeClosedLoopRewrite,
  computeTrajectoryRepetitionPenalty,
  sanitizeWaveSteps,
} from '../src/dynamics/bidirectionalBridge.js';


const TEST_FWD_NOVEL = [
  ['雨夜の湯治場と老陶工の気配が漂う。', '冷え切った空気が作業場を満たしていた。', '古い轆轤が低く軋み始める。'],
  ['雨夜の湯治場と老陶工の気配が漂う。', '軒先からの水滴が踏み石を打つ。', '静まり返った谷底に火の色が差す。'],
  ['雨夜の湯治場と老陶工の気配が漂う。', '湿った土の匂いが立ち込めていた。', '老人は無言で粘土の重みを測る。'],
  ['雨夜の湯治場と老陶工の気配が漂う。', '遠い夜風が木立を揺らしていた。', '削り屑が床に静かに積もる。'],
];
const TEST_BWD_NOVEL = [
  ['窯の焚口から微かな熱気が漏れる。', '言葉のないまま時間は過ぎていった。', '微かな軋みとともにすべての輪郭が沈む。'],
  ['指先が土の冷たさに慣れていく。', '暗がりのなかに炎の残像が浮かぶ。', '微かな軋みとともにすべての輪郭が沈む。'],
  ['夜雨の音がすべてを包み隠していく。', '静寂がふたたび作業場を満たした。', '微かな軋みとともにすべての輪郭が沈む。'],
  ['形を得た器が静かに冷えていく。', '暗闇の底で沈黙が定着していった。', '微かな軋みとともにすべての輪郭が沈む。'],
];

const TEST_FWD_ARTICLE = [
  ['組織の非同期コミュニケーションと生産性に関する観測。', '即時チャットは認知的過負荷を招く。', '文脈の永続化が意思決定を加速する。'],
  ['組織の非同期コミュニケーションと生産性に関する観測。', '同期会議の削減が集中時間を生む。', '情報ハブの分散がボトルネックを防ぐ。'],
  ['組織の非同期コミュニケーションと生産性に関する観測。', '非対称なコミュニケーションを許容する。', 'ドキュメントによる合意形成を徹底する。'],
  ['組織の非同期コミュニケーションと生産性に関する観測。', '業務ログの透明性が信頼を担保する。', '心理的安全性と実行速度を両立する。'],
];
const TEST_BWD_ARTICLE = [
  ['自律的な意思決定フレームワークが浸透する。', '組織スケールに伴う情報劣化を遮断する。', 'したがって本質的な競争優位は非同期プロトコルに帰着する。'],
  ['知識資産が複利的にレバレッジされる。', '属人性に依存しない組織基盤が完成する。', 'したがって本質的な競争優位は非同期プロトコルに帰着する。'],
  ['調整コストの線形爆発を未然に防止する。', '高密度な知識創造へ集中可能となる。', 'したがって本質的な競争優位は非同期プロトコルに帰着する。'],
  ['オープンな文脈共有が格差を解消する。', '強靭な自律分散エコシステムが確立される。', 'したがって本質的な競争優位は非同期プロトコルに帰着する。'],
];

const TEST_FWD_COT = [
  ['前提確定：コンパクト群上の不変測度の存在証明。', 'ステップ1：ハール測度の左不変性を仮定する。', 'ステップ2：畳み込み代数の性質を精査する。'],
  ['前提確定：コンパクト群上の不変測度の存在証明。', 'ステップ1：可換代数の極大イデアルを構成する。', 'ステップ2：ゲルファント変換を適用する。'],
  ['前提確定：コンパクト群上の不変測度の存在証明。', 'ステップ1：一様連続関数族の等程度連続性を示す。', 'ステップ2：アスコリ・アルツェラの定理を準備する。'],
  ['前提確定：コンパクト群上の不変測度の存在証明。', 'ステップ1：ヒルベルト空間の凸閉集合を定義する。', 'ステップ2：直交射影定理から最小元を特定する。'],
];
const TEST_BWD_COT = [
  ['ステップN-2：核空間が自明に縮退する。', 'ステップN-1：全単射性と等長性が示される。', '結論：不変測度の一意存在が証明された Q.E.D. 。'],
  ['ステップN-2：背理法仮定の矛盾が導かれる。', 'ステップN-1：境界条件の整合性が完結する。', '結論：不変測度の一意存在が証明された Q.E.D. 。'],
  ['ステップN-2：ルベーグ収束定理が適用される。', 'ステップN-1：極限の一意性が確定する。', '結論：不変測度の一意存在が証明された Q.E.D. 。'],
  ['ステップN-2：位相幾何学的不変量が保存される。', 'ステップN-1：証明の導出鎖が完全に閉じる。', '結論：不変測度の一意存在が証明された Q.E.D. 。'],
];

const TEST_GOALS = [
  'やがて微細な熱量の移動も完全に途絶え、空間のすべての輪郭が冷たく沈み込み、あとに不可逆な物理的平衡と静寂だけが残された。',
  '夜の底へと光の粒子の最後の余韻が吸い込まれ、空間を満たしていた微かな揺らぎは消滅し、世界は確固たる静止へと溶け落ちていった。',
  '遠くの物音も大気と同化して完全に消え去り、閉ざされた視界の奥には冷えた空気の重みと不動の輪郭だけが静かに定着していた。',
  '吹き抜けていた風も完全に凪ぎ、冷え切った大地と影の交錯する地点に、外界から隔絶された絶対的な沈黙が満ちていた。',
];

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
    const res = solveBidirectionalCollisionBridge(anchors, { forward_candidates: TEST_FWD_NOVEL, backward_candidates: TEST_BWD_NOVEL, k_candidates: 4 });

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
    expect(res.welded_text.length).toBeGreaterThanOrEqual(80);
  });

  it('solves collision on analytical article domain', () => {
    const anchors = pinBoundaryAnchors('組織の非同期コミュニケーションと生産性', 'article');
    const res = solveBidirectionalCollisionBridge(anchors, { forward_candidates: TEST_FWD_ARTICLE, backward_candidates: TEST_BWD_ARTICLE, k_candidates: 4 });

    expect(res.domain).toBe('article');
    expect(res.optimal_pair[0]).toBeGreaterThanOrEqual(0);
    expect(res.optimal_pair[1]).toBeGreaterThanOrEqual(0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.welded_text).toContain('コミュニケーション');
    expect(res.welded_text.length).toBeGreaterThanOrEqual(80);
  });

  it('solves collision on CoT reasoning domain', () => {
    const anchors = pinBoundaryAnchors('コンパクト群上の不変測度の存在証明', 'cot_reasoning');
    const res = solveBidirectionalCollisionBridge(anchors, { forward_candidates: TEST_FWD_COT, backward_candidates: TEST_BWD_COT, k_candidates: 4 });

    expect(res.domain).toBe('cot_reasoning');
    expect(res.min_action).toBeLessThan(10.0);
    expect(res.m_bridge).toBeGreaterThan(0.0);
    expect(res.welded_trajectory.some((s) => s.includes('ステップ'))).toBe(true);
    expect(res.welded_text.length).toBeGreaterThanOrEqual(80);
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
    const { optimal_x_T, scored_candidates } = deriveEndogenousGoalHypotheses(x0, 'novel', { candidate_goals: TEST_GOALS, k_candidates: 3 });

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
    const res = executeClosedLoopRewrite(singleSentence, 'novel', { candidate_goals: TEST_GOALS });
    expect(res.anchors.x_0).toBe(singleSentence);
    expect(res.anchors.x_T).not.toBe(singleSentence);
    expect(res.anchors.x_T.length).toBeGreaterThan(10);
    expect(res.welded_text.length).toBeGreaterThanOrEqual(80);
  });

  it('deduplicates parrot infill echo sentences', () => {
    const anchors = pinBoundaryAnchors('重複遮断検証', 'article');
    const fwd = [['都市空間の歩行速度は過去五十年で一割上昇した。', 'スマートフォンの普及により視線は下方に固定され、受動的な群衆流動が常態化している。']];
    const bwd = [['身体的な余白を取り戻す建築設計が求められている。', '広場における偶発的な滞留こそが都市の呼吸を再生する。']];

    const parrotInfill = (fLast: string, bFirst: string) => `${fLast}${bFirst}`;
    const res = solveBidirectionalCollisionBridge(anchors, {
      forward_candidates: fwd,
      backward_candidates: bwd,
      infill_generator: parrotInfill,
    });
    expect(res.infilling_step).toBeUndefined();
    expect(res.welded_trajectory.length).toBe(4);
    expect(res.welded_trajectory.filter((s) => s === fwd[0]![1]).length).toBe(1);
  });

  it('penalizes prefix repetition and 5-gram overlaps', () => {
    const stepsClean = [
      '夜明け前の山道に白い霧が立ち込めていた。',
      '遠くの沢から冷たい水音が微かに響いてくる。',
      '石畳の階段を踏みしめるたびに草の匂いが立った。',
    ];
    const stepsPrefixEcho = [
      '朝日が昇る頃には、山道に白い霧が立ち込めていた。',
      '遠くの沢から冷たい水音が微かに響いてくる。',
      '朝日が昇る頃には、谷底の集落に淡い光が差し込んだ。',
    ];
    const penClean = computeTrajectoryRepetitionPenalty(stepsClean);
    const penEcho = computeTrajectoryRepetitionPenalty(stepsPrefixEcho);
    expect(penClean).toBe(0.0);
    expect(penEcho).toBeGreaterThanOrEqual(4.0);
  });

  it('rejects CoT meta poem conclusions in favor of mathematical derivations', () => {
    const x0 = '実数列 $a_n$ が有界かつ単調増加であるとき、極限 $\\lim_{n\\to\\infty} a_n = \\sup_n a_n$ の存在を示せ。';
    const candidates = [
      'この定理は解析学における重要な基盤であり、社会や科学の様々な応用モデル化において不可欠な役割を果たしている。',
      '上限 $M = \\sup_n a_n$ に対し $\\forall \\varepsilon > 0, \\exists N \\in \\mathbb{N}$ で $M - \\varepsilon < a_n \\le M$ となり収束が一意に示される。Q.E.D.',
    ];
    const { optimal_x_T, scored_candidates } = deriveEndogenousGoalHypotheses(x0, 'cot_reasoning', {
      candidate_goals: candidates,
      k_candidates: 2,
    });
    expect(optimal_x_T).toContain('Q.E.D.');
    expect(optimal_x_T).not.toContain('重要な基盤');
    expect(scored_candidates[0]!.score).toBeLessThan(scored_candidates[1]!.score);
  });

  it('sanitizes wave steps by stripping kanji stem paraphrases (Jaccard >= 0.50), meta-leaks, and CJK tokens', () => {
    const x0 = '風雪の峠道と廃屋の灯りの気配が漂い、張り詰めた冷気が指先をかすかに凍らせていた。';
    const xT = '微かな軋みとともにすべての輪郭が沈み、あとに冷たい沈黙だけが残った。';
    const rawSteps = [
      x0,
      'それは雪国の伝統であり、またそれは現代の世界でも、我々の生活を支える力となる。',
      'それらは、雪国の伝統であり、現代の世界にもその力を支えている。',
      'この展開文は、冒頭文で示された状況を補強する。',
      '函数列是恒正且有界である。',
      '囲炉裏の残り火が暗がりで静かに揺れていた。',
      xT,
    ];
    const cleaned = sanitizeWaveSteps(rawSteps, x0, xT);
    expect(cleaned.length).toBe(4);
    expect(cleaned[0]).toBe(x0);
    expect(cleaned[1]).toContain('それは雪国の伝統であり');
    expect(cleaned[2]).toContain('囲炉裏の残り火');
    expect(cleaned[3]).toBe(xT);

    const penParaphrase = computeTrajectoryRepetitionPenalty([
      'それは雪国の伝統であり、またそれは現代の世界でも、我々の生活を支える力となる。',
      'それらは、雪国の伝統であり、現代の世界にもその力を支えている。',
    ]);
    expect(penParaphrase).toBeGreaterThanOrEqual(6.0);
  });

  it('deduplicates non-kanji sentences and strips zero-kana Chinese sentences in sanitizeWaveSteps', () => {
    const x0 = 'あかりがしずかにゆれていた。';
    const xT = 'そしてすべてがしずまりかえった。';
    const rawSteps = [
      x0,
      'そよかぜがまどをかすかにゆらしていく。',
      'そよかぜがまどをかすかにゆらしていく。',
      '假设在希尔伯特空间 (H) 中，(M) 是一个闭部分空间。',
      '晨光漸漸穿透，濡濕石板映射黎明前寂寥。',
      x0,
      xT,
    ];
    const cleaned = sanitizeWaveSteps(rawSteps, x0, xT);
    expect(cleaned).toEqual([x0, 'そよかぜがまどをかすかにゆらしていく。', xT]);
  });

  it('preserves valid Japanese with 若者, 若い, 若干, 寄与, 与える while pruning Chinese grammar leaks', () => {
    const x0 = '夜明け前の空が静かに白んでいた。';
    const xT = '新しい一日が始まった。';

    const rawSteps = [
      x0,
      '若者たちは雪の中を静かに歩き始めた。',
      '若い頃の記憶が鮮やかに蘇る。',
      'この研究の成果は今後の科学技術の発展に大きく寄与する。',
      '彼に新しい役割を与えることで組織の活性化を図る。',
      '若 $x \\in S$ 则 $f(x) > 0$ である。', // Chinese grammar leak
      '空間 $V$ 与 $W$ の直和分解を考える。', // Chinese grammar leak
      xT,
    ];
    const cleaned = sanitizeWaveSteps(rawSteps, x0, xT);
    expect(cleaned).toContain('若者たちは雪の中を静かに歩き始めた。');
    expect(cleaned).toContain('この研究の成果は今後の科学技術の発展に大きく寄与する。');
    expect(cleaned).toContain('彼に新しい役割を与えることで組織の活性化を図る。');
    expect(cleaned.some(s => s.includes('若 $x'))).toBe(false);
    expect(cleaned.some(s => s.includes('与 $W'))).toBe(false);
  });

  it('handles boundary whitespace resiliently and prunes intermediate steps echoing x_0', () => {
    const x0 = '国境の長いトンネルを抜けると雪国であった。';
    const xT = '夜の底が白くなった。';

    // 1. Whitespace resilience
    const stepsWithWs = ['  ', x0, '鳥が静かに歌っていた。', xT, '   '];
    const cleanedWs = sanitizeWaveSteps(stepsWithWs, x0, xT);
    expect(cleanedWs).toEqual([x0, '鳥が静かに歌っていた。', xT]);

    // 2. Intermediate step echoing x_0 must be pruned
    const trajectoryWithX0Echo = [
      x0,
      '信号所の手前で列車が止まった。',
      '長いトンネルを抜けると雪国であった。', // 5-gram and stem match with x0 at idx > 0
      xT,
    ];
    const cleanedEcho = sanitizeWaveSteps(trajectoryWithX0Echo, x0, xT);
    expect(cleanedEcho.length).toBe(3);
    expect(cleanedEcho[0]).toBe(x0);
    expect(cleanedEcho[1]).toBe('信号所の手前で列車が止まった。');
    expect(cleanedEcho[2]).toBe(xT);
  });
});
