/**
 * USIR v4.0 Bidirectional Schrödinger Bridge (Meet-in-the-Middle Collision Engine).
 *
 * Implements:
 * 1. Boundary Pinning (pinBoundaryAnchors):
 *    Pins x_0 (Left-to-Right introduction anchor) and x_T (Right-to-Left W_end* landing anchor)
 *    parameterized by 24D Kamae (Sg-10..12) macro constraints.
 * 2. Bidirectional Rollout & Midpoint Collision Solver (solveBidirectionalCollisionBridge):
 *    - Forward Wave psi: Left-to-Right rollout {tau_i} from x_0
 *    - Backward Wave phi: Right-to-Left adjoint rollout {tau_j} from x_T
 *    - Collision Action Integral:
 *      S_collision(tau_i, tau_j) = D_geodesic(e_tip_i, e_tip_j)
 *                                + D_spatial_causal(tau_i, tau_j)
 *                                + (beta/2) * D_3phase(z(tau_i o tau_j))
 *    - Optimal Midpoint Collision & Infilling Welding
 * 3. Universal 3-Domain Support:
 *    - "novel": Sensory perception -> Physical state change
 *    - "article": Empirical observation/data -> Structural insight
 *    - "cot_reasoning": Premise forward deduction x Goal backward abduction collision
 */

import { embedSentenceUnit, computeThreePhaseQdaDistance } from './manifoldGeodesic.js';

export type BridgeDomain = 'novel' | 'article' | 'cot_reasoning';

export interface BoundaryAnchors {
  domain: BridgeDomain;
  theme: string;
  x_0: string;
  x_T: string;
  q_start: number[];
  q_end: number[];
  kamae_coords: number[];
  metadata: Record<string, any>;
}

export interface TrajectoryWave {
  wave_id: number;
  direction: 'forward' | 'backward';
  steps: string[];
  tip_vector: number[];
  direction_vector: number[];
  cumulative_energy?: number;
}

export interface CollisionBridgeResult {
  domain: BridgeDomain;
  anchors: BoundaryAnchors;
  optimal_pair: [number, number]; // [forward_idx, backward_idx]
  min_action: number;
  welded_trajectory: string[];
  welded_text: string;
  m_bridge: number;
  d_geodesic: number;
  d_causal: number;
  d_3phase: number;
  forward_waves: TrajectoryWave[];
  backward_waves: TrajectoryWave[];
  collision_matrix: number[][];
}

export interface TensorBridgeResult {
  optimal_pair: [number, number];
  min_action: number;
  action_matrix: number[][];
  d_geodesic: number;
  d_direction: number;
  welded_tensor: number[][];
  m_bridge: number;
}

const DOMAIN_PRESETS: Record<BridgeDomain, { kamae_target: number[] }> = {
  novel: { kamae_target: [0.55, 0.60, 0.45] },
  article: { kamae_target: [0.70, 0.40, 0.35] },
  cot_reasoning: { kamae_target: [0.85, 0.20, 0.25] },
};

// ==============================================================================
// 1. Boundary Pinning (pinBoundaryAnchors)
// ==============================================================================

export function pinBoundaryAnchors(
  theme: string,
  domain: BridgeDomain = 'novel',
  options?: {
    target_state?: number[];
    custom_start?: string;
    custom_end?: string;
  }
): BoundaryAnchors {
  const preset = DOMAIN_PRESETS[domain] ?? DOMAIN_PRESETS.novel;

  let x_0: string;
  if (options?.custom_start) {
    x_0 = options.custom_start;
  } else {
    if (domain === 'novel') {
      x_0 = `${theme}の気配が濃く漂い、張り詰めた冷気が指先と肌の表面をかすかに凍らせていた。`;
    } else if (domain === 'article') {
      x_0 = `${theme}に関する最新の観測データおよび現場実証は、従来の想定モデルと明確に乖離している。`;
    } else {
      x_0 = `前提確定：問題設定「${theme}」における初期公理系、境界条件、および状態空間の基礎構造を確定する。`;
    }
  }

  let x_T: string;
  if (options?.custom_end) {
    x_T = options.custom_end;
  } else {
    if (domain === 'novel') {
      x_T = `微かな軋みとともにすべての輪郭が沈み、あとに冷たい沈黙と不動の平衡だけが残った。`;
    } else if (domain === 'article') {
      x_T = `したがって、本質的な競争優位は構造の非対称性と非同期プロトコルによる同期コスト削減に帰着する。`;
    } else {
      x_T = `結論：すべての背理条件が完全に排除され、導出鎖の完全性によって目標命題の一意存在が証明された Q.E.D.。`;
    }
  }

  const q_start = embedSentenceUnit(x_0);
  const q_end = embedSentenceUnit(x_T);
  const kamae_coords = options?.target_state ? options.target_state.slice(9, 12) : preset.kamae_target;

  return {
    domain,
    theme,
    x_0,
    x_T,
    q_start,
    q_end,
    kamae_coords,
    metadata: { preset: domain },
  };
}

// ==============================================================================
// 2. Candidate Wave Generation (Forward & Backward Proposers)
// ==============================================================================

// ==============================================================================
// 2. Abstract Tensor Collision Solver & Endogenous Boundary Engines
// ==============================================================================

export function solveTensorCollisionBridge(
  qStart: number[],
  qEnd: number[],
  fwdTrajectories: number[][][],
  bwdTrajectories: number[][][],
  options?: { beta?: number; lambda_bridge?: number }
): TensorBridgeResult {
  const beta = options?.beta ?? 0.5;
  const lambdaBridge = options?.lambda_bridge ?? 0.8;
  const Kfwd = fwdTrajectories.length;
  const Kbwd = bwdTrajectories.length;

  if (Kfwd === 0 || Kbwd === 0) {
    throw new Error('fwdTrajectories and bwdTrajectories must be non-empty.');
  }

  const actionMatrix: number[][] = [];
  let bestAction = Infinity;
  let bestPair: [number, number] = [0, 0];
  let bestComponents: [number, number] = [0, 0];
  let bestWelded: number[][] = [];

  for (let i = 0; i < Kfwd; i++) {
    const tauF = fwdTrajectories[i]!;
    const tipF = tauF[tauF.length - 1]!;
    const prevF = tauF.length >= 2 ? tauF[tauF.length - 2]! : qStart;
    const vF = tipF.map((v, idx) => v - (prevF[idx] ?? 0));
    let normF = 0;
    for (const v of vF) normF += v * v;
    normF = Math.sqrt(normF);
    const normVf = normF > 1e-7 ? vF.map((v) => v / normF) : vF;

    const row: number[] = [];
    for (let j = 0; j < Kbwd; j++) {
      const tauB = bwdTrajectories[j]!;
      const tipB = tauB[0]!;
      const nextB = tauB.length >= 2 ? tauB[1]! : qEnd;
      const vB = nextB.map((v, idx) => v - (tipB[idx] ?? 0));
      let normB = 0;
      for (const v of vB) normB += v * v;
      normB = Math.sqrt(normB);
      const normVb = normB > 1e-7 ? vB.map((v) => v / normB) : vB;

      // 1. Geodesic distance in manifold
      let distL2Sq = 0;
      let dotDir = 0;
      for (let d = 0; d < tipF.length; d++) {
        const diff = (tipF[d] ?? 0) - (tipB[d] ?? 0);
        distL2Sq += diff * diff;
        dotDir += (normVf[d] ?? 0) * (normVb[d] ?? 0);
      }
      const dGeo = Math.sqrt(distL2Sq);
      const dDir = 1.0 - Math.max(-1.0, Math.min(1.0, dotDir));

      // 3. Welded trajectory curvature
      const welded = [...tauF, ...tauB];
      let dCurve = 0;
      if (welded.length >= 3) {
        let curveSum = 0;
        for (let k = 0; k < welded.length - 2; k++) {
          const w0 = welded[k]!;
          const w1 = welded[k + 1]!;
          const w2 = welded[k + 2]!;
          let dot = 0;
          let n1Sq = 0;
          let n2Sq = 0;
          for (let d = 0; d < w0.length; d++) {
            const d1 = (w1[d] ?? 0) - (w0[d] ?? 0);
            const d2 = (w2[d] ?? 0) - (w1[d] ?? 0);
            dot += d1 * d2;
            n1Sq += d1 * d1;
            n2Sq += d2 * d2;
          }
          const denom = Math.sqrt(n1Sq * n2Sq) + 1e-7;
          curveSum += 1.0 - Math.max(-1.0, Math.min(1.0, dot / denom));
        }
        dCurve = curveSum / (welded.length - 2);
      }

      const sTot = 1.5 * dGeo + 0.8 * dDir + 0.5 * beta * dCurve;
      row.push(sTot);

      if (sTot < bestAction) {
        bestAction = sTot;
        bestPair = [i, j];
        bestComponents = [dGeo, dDir];
        bestWelded = welded;
      }
    }
    actionMatrix.push(row);
  }

  const mBridge = Math.max(0.0, Math.min(1.0, Math.exp(-lambdaBridge * bestAction)));
  return {
    optimal_pair: bestPair,
    min_action: bestAction,
    action_matrix: actionMatrix,
    d_geodesic: bestComponents[0],
    d_direction: bestComponents[1],
    welded_tensor: bestWelded,
    m_bridge: mBridge,
  };
}

// ------------------------------------------------------------------------------
// Mode B: Endogenous Goal Hypothesis
// ------------------------------------------------------------------------------

export const DOMAIN_GOAL_PROPOSALS: Record<BridgeDomain, string[]> = {
  novel: [
    'やがて微細な熱量の移動も完全に途絶え、空間のすべての輪郭が冷たく沈み込み、あとに不可逆な物理的平衡と静寂だけが残された。',
    '夜の底へと光の粒子の最後の余韻が吸い込まれ、空間を満たしていた微かな揺らぎは消滅し、世界は確固たる静止へと溶け落ちていった。',
    '遠くの物音も大気と同化して完全に消え去り、閉ざされた視界の奥には冷えた空気の重みと不動の輪郭だけが静かに定着していた。',
    '吹き抜けていた風も完全に凪ぎ、冷え切った大地と影の交錯する地点に、外界から隔絶された絶対的な沈黙が満ちていた。',
  ],
  article: [
    'したがって本質的な競争優位の源泉は、局所的な即時対応の強制を排し、構造の非対称性と非同期プロトコルによる同期コストの極小化に帰着する。',
    '分散台帳に基づく監査可能性と自律的コンテキスト同期の徹底こそが、組織スケールに伴う情報劣化と認知的負債を根本から遮断する。',
    '以上の定量的構造分析から、偶発的な場当たり的調整を排し、情報の永続性と疎結合アーキテクチャへの制度的移行が不可避の結論となる。',
    '局所最適化の罠を脱却し、二重ループの非同期フィードバック基盤を確立することこそが、長期的な技術的持続可能性を担保する唯一の道である。',
  ],
  cot_reasoning: [
    '結論：背理法におけるすべての反例仮定が直交補空間の閉包性により矛盾へと導かれ、目標命題の一意存在と正則性が完全に証明された Q.E.D.。',
    'したがって有界閉集合における完全連続性と自己随伴作用素のスペクトル分解定理の成立により、主定理の主張が境界も含めて確定する Q.E.D.。',
    '以上の代数的同型写像の一対一対応とコーシー列の完備性議論により、全領域における位相幾何学的不変量の保存が厳密に証明された Q.E.D.。',
    'これにより局所的な微小変動はイプシロン論法で完全に抑え込まれ、基礎微分方程式の解の大域的安定性と一意性が矛盾なく導出された Q.E.D.。',
  ],
};

export function deriveEndogenousGoalHypotheses(
  x_0: string,
  domain: BridgeDomain = 'novel',
  options?: { candidate_goals?: string[]; k_candidates?: number }
): { optimal_x_T: string; scored_candidates: Array<{ text: string; score: number }> } {
  const preset = DOMAIN_PRESETS[domain] ?? DOMAIN_PRESETS.novel;
  const kamaeTarget = preset.kamae_target;
  const kCandidates = options?.k_candidates ?? 4;
  const safeX0 = x_0 && x_0.trim() ? x_0.trim() : `${domain}の導入境界を確定する。`;
  const candidates = options?.candidate_goals?.slice(0, kCandidates) ?? DOMAIN_GOAL_PROPOSALS[domain]!.slice(0, kCandidates);

  const q0 = embedSentenceUnit(safeX0);
  const scored: Array<{ text: string; score: number }> = [];

  for (const c of candidates) {
    const qc = embedSentenceUnit(c);
    let kamaeDiffSq = 0;
    for (let i = 0; i < 3; i++) {
      const diff = (qc[9 + i] ?? 0) - (kamaeTarget[i] ?? 0);
      kamaeDiffSq += diff * diff;
    }
    const kamaeDiff = Math.sqrt(kamaeDiffSq);

    const jointText = `${safeX0} ${c}`;
    const qda = computeThreePhaseQdaDistance(jointText);

    let reachDistSq = 0;
    for (let i = 0; i < qc.length; i++) {
      const diff = (qc[i] ?? 0) - (q0[i] ?? 0);
      reachDistSq += diff * diff;
    }
    const reachDist = Math.sqrt(reachDistSq);
    let reachPenalty = 0;
    if (reachDist < 0.25) reachPenalty = 2.0;
    else if (reachDist > 2.8) reachPenalty = 1.5;

    const actionScore = kamaeDiff + 0.35 * qda.d_3phase + reachPenalty;
    scored.push({ text: c, score: actionScore });
  }

  scored.sort((a, b) => a.score - b.score);
  return { optimal_x_T: scored[0]?.text ?? candidates[0]!, scored_candidates: scored };
}

// ------------------------------------------------------------------------------
// Mode A: Closed-Loop Rewrite
// ------------------------------------------------------------------------------

export function executeClosedLoopRewrite(
  initialRolloutText: string,
  domain: BridgeDomain = 'novel',
  options?: { theme?: string; beta?: number; forward_candidates?: string[][]; backward_candidates?: string[][] }
): CollisionBridgeResult {
  const cleaned = initialRolloutText.trim();
  const rawParts = cleaned.split(/[。！？\n]+/).map((s) => s.trim()).filter((s) => s.length >= 5);
  const sentences = rawParts.map((s) => s + '。');

  let x_0: string;
  let x_T: string;

  if (sentences.length === 0) {
    x_0 = `${options?.theme || 'closed_loop_rewrite'}の導入境界を確定する。`;
    const res = deriveEndogenousGoalHypotheses(x_0, domain);
    x_T = res.optimal_x_T;
  } else if (sentences.length === 1) {
    x_0 = sentences[0]!;
    const res = deriveEndogenousGoalHypotheses(x_0, domain);
    x_T = res.optimal_x_T;
  } else {
    x_0 = sentences[0]!;
    x_T = sentences[sentences.length - 1]!;
    if (x_0 === x_T || x_T.length < 8) {
      const res = deriveEndogenousGoalHypotheses(x_0, domain);
      x_T = res.optimal_x_T;
    }
  }

  const anchors = pinBoundaryAnchors(options?.theme || 'closed_loop_rewrite', domain, {
    custom_start: x_0,
    custom_end: x_T,
  });

  return solveBidirectionalCollisionBridge(anchors, {
    forward_candidates: options?.forward_candidates,
    backward_candidates: options?.backward_candidates,
    beta: options?.beta ?? 0.5,
  });
}

// ------------------------------------------------------------------------------
// Contextual Candidate Wave Generator
// ------------------------------------------------------------------------------

export function generateContextualCandidateWaves(
  anchors: BoundaryAnchors,
  kCandidates = 4
): [TrajectoryWave[], TrajectoryWave[]] {
  const domain = anchors.domain;
  const theme = anchors.theme;

  let fwdVariations: string[][];
  let bwdVariations: string[][];

  if (domain === 'novel') {
    fwdVariations = [
      [
        `${theme}の気配のなかで、張り詰めた氷のような空気が肌の表面をかすかに撫で、足元からは白い地霧が静かに立ち昇り始めていた。`,
        '周囲の物音がことごとく遠のき、視界の奥でかすかに明滅する光の粒子だけが、闇のなかに微小な境界線を引いていた。',
        '冷気は呼吸を重ねるたびに肺の奥を鋭く満たし、外界と自己の境界が曖昧に融解していく奇妙な透明感が広がっていく。',
      ],
      [
        `暮れゆく薄暮の陰影が${theme}の空間の輪郭を奪い去り、凍てつく静けさが意識のリズムを少しずつ内側へと研ぎ澄ませていく。`,
        '乾いた小枝が微かな風に軋んで乾いた余韻を残し、冷え切った湿り気が土の奥底から静かな手応えとなって立ち現れる。',
        '視覚が遮断されたことで聴覚と触覚が過敏になり、目に見えぬ冷気の流れが周囲の空間密度を均等に塗り替えていた。',
      ],
      [
        `吹き抜ける冷風が${theme}の静けさを揺らし、遠くの木立が重い波音を立て、影の境界は不確かに足元の地面へと溶け込んでいく。`,
        '時の流れが極度に緩慢になり、大気の底に沈殿する冷気の重みだけが、立ち止まった旅人の身体に確かな存在感を与えていた。',
        '暗がりの中に潜む微かな温度の痕跡を手探りで求めようとするものの、世界はすでに完全な冷却の過程へと移行している。',
      ],
      [
        `立ち込める濃密な霧が${theme}の光彩を均一に拡散させ、辿るべき道筋の行く手を静謐な白のグラデーションで覆い隠した。`,
        '肌に触れる大気の緊張は極限まで高まり、微かな外因的変化をも拒絶するかのように、風景全体が緩やかに凝固しつつある。',
        'すべての色彩がモノクロームの陰影へと収束し、世界は自らの存在を忘却したかのような圧倒的な静寂に包まれていた。',
      ],
    ];
    bwdVariations = [
      [
        'やがて空間を震わせていた微細な熱量が完全に放散し、あらゆる動的な兆候が静かに秩序ある配置へと収斂していく。',
        '残された物理的な輪郭が夜の帳のなかに不動の形態として定着し、冷たく澄んだ沈黙が全空間を隙間なく満たしていた。',
        '外部からの干渉の余波は完全に拭い去られ、暗がりの奥底にただ純粋な冷却の平衡状態だけが横たわっている。',
      ],
      [
        'すべての運動が不可逆な安定点へと引き寄せられ、揺らぎの止まった大気が冷たい静寂を確固たるものにしていった。',
        '外的な干渉の痕跡は完全に拭い去られ、暗がりの奥深くにただ均質で純粋な物理的平衡状態だけが横たわっている。',
        '冷気はもはや刺すような鋭さを失い、風景と一体化した絶対的な静止の媒質としてその場に留まり続けていた。',
      ],
      [
        '細い光の余韻もまた漆黒の底へと溶け去り、空間の幾何学的な広がりは静止した意識のなかに静かに回収されていく。',
        '冷気はもはや刺すような鋭さを失い、風景と一体化した絶対的な静止の媒質として、その場に留まり続けていた。',
        '時間の不可逆な矢が静止点に到達し、凍てついた大気は微かな揺らぎさえ許さぬ確固たる静謐を保っていた。',
      ],
      [
        'あらゆる偶発的な気配が排された極限において、冷えた大気の質量が確かな密度をもって空間を均等に満たしていく。',
        '境界を失った暗がりの中で、微かな軋みさえも永久に封じ込められ、完成された沈黙の構造だけがそこに残されていた。',
        'やがて視界のすべてが闇と同化し、外界から遮断された空間には絶対的な無為と静止の律動だけが息づいている。',
      ],
    ];
  } else if (domain === 'article') {
    fwdVariations = [
      [
        `課題設定「${theme}」における実証データは、従来の同期型コミュニケーションモデルの幾何学的限界を明示している。`,
        '組織規模の拡大に伴い、暗黙知や即時応答に依存した同期プロトコルは急激な認知的過負荷と情報局所滞留を招く。',
        '知識伝達のボトルネックを構造的に解析すると、情報の非対称性と同期待ちレイテンシが意思決定速度を致命的に低下させている。',
      ],
      [
        `分散組織における「${theme}」の実践においては、情報の局所滞留を排除し、自律的な情報アクセス権を設計することが中核となる。`,
        '非同期な文書化とプロトコル化を推進することで、暗黙知に依存していた文脈の再現性と再利用性が組織全体で飛躍的に担保される。',
        '即時応答の社会的プレッシャーを意図的に排除する制度設計こそが、深い思考時間の確保と高品質な設計レビューに直結する。',
      ],
      [
        `「${theme}」に関する定量的観測から明らかなように、コミュニケーションの非対称性を工学的に受容する視点が不可欠である。`,
        'テキストを主軸とした非同期レビューサイクルへの転換は、偶発的な場当たり合意を排し、合意形成プロセスの監査可能性を高める。',
        '分散環境下での意思決定スループットは、ハブノードの負荷集中を回避し、疎結合な自律チームを編制することによってのみ維持される。',
      ],
      [
        `現代のソフトウェア組織における「${theme}」の課題は、単なるツール導入の問題ではなく、情報流通の物理的トポロジーに起因する。`,
        '同期型会議の頻発は作業記憶の細片化を引き起こし、エンジニアリング組織が本来発揮すべき創造的複利効果を根本から阻害する。',
        'したがって、情報の永続化と非同期フィードバックループの確立を開発文化の根幹に据える構造改革が強く要請される。',
      ],
    ];
    bwdVariations = [
      [
        'この構造的転換を実践することで、局所的な認識の齟齬は最小化され、自律的な意思決定フレームワークが組織全体へと浸透する。',
        '結果として情報の散逸が完全に防止され、組織スケールに伴う調整コストの線形爆発を根本から遮断することが可能となる。',
        '組織の認知リソースは真に不可逆な戦略課題へと再配分され、持続的なイノベーション創出の推進力となる。',
      ],
      [
        '透明性の高い台帳管理と非同期フィードバックの結合により、再現性の高いガバナンス基盤が現場レベルで確立される。',
        'これらの一貫したプロトコル遵守の積み重ねによって、長期的なエンジニアリング品質と事業推進力の高度な両立が具現化する。',
        '制度化された非同期レビューと台帳同期は、属人性を完全に排した堅牢なナレッジベースを自律的に生成し続ける。',
      ],
      [
        '非同期プロトコルの定着は心理的安全性を損なうことなく、成果物に対する客観的かつ建設的な批判的吟味を日常化させる。',
        '最終的に組織の認知リソースは真に不可逆な戦略的課題へと再配分され、持続的なイノベーション創出の推進力となる。',
        'こうして確立された疎結合アーキテクチャこそが、将来の不確実な環境変動に対する最大の組織的耐障害性を提供する。',
      ],
      [
        '制度化された非同期レビューと台帳同期は、属人性を完全に排した堅牢なナレッジベースを自律的に生成し続ける。',
        'こうして確立された疎結合アーキテクチャこそが、将来の不確実な環境変動に対する最大の組織的耐障害性を提供する。',
        '同期コストを構造的に極小化し得た組織のみが、規模の拡大に伴う情報劣化の壁を不可逆的に突破できる。',
      ],
    ];
  } else {
    fwdVariations = [
      [
        `ステップ1（公理確定）：問題「${theme}」における基本仮定、状態空間の位相構造、および境界束縛条件を厳密に同定する。`,
        'ステップ2（補題展開）：有界閉集合上の連続作用素列に対してコーシーの判定条件を適用し、ノルム空間における完備性を保証する。',
        'ステップ3（正則性評価）：レゾルベント作用素の解析接続とスペクトル分解定理を用い、極限候補関数の連続性と局所有界性を導出する。',
      ],
      [
        `ステップ1（初期束縛）：ヒルベルト空間の幾何学的性質に基づき、問題「${theme}」の解空間を凸閉部分集合として定式化する。`,
        'ステップ2（直交射影）：射影定理より差ベクトルのノルム最小化問題へと帰着させ、極限操作の一様収束性を厳密に検証する。',
        'ステップ3（測度論的展開）：測度空間の正則性と可算加法性を用いて特異集合の測度を評価し、ルベーグ可測関数の弱収束を確立する。',
      ],
      [
        `ステップ1（背理前提）：命題「${theme}」の結論の否定を仮定し、不連続点または保存則を破る特異点の存在を仮定する。`,
        'ステップ2（エネルギー不等式）：ポアンカレ不等式およびソボレフ埋め込み定理を適用し、微小領域におけるエネルギー積分の下限を評価する。',
        'ステップ3（微小摂動解析）：境界点近傍における微小変動をイプシロン論法で抑え込み、解の存在領域を特定する。',
      ],
      [
        `ステップ1（代数構造同定）：問題「${theme}」に付随する群作用の不変性と同型写像の線形独立性を同定する。`,
        'ステップ2（積分表示の確立）：コーシー・リーマン関係式を満たす正則関数族に対して周回積分表示を確立し、留数定理を適用する。',
        'ステップ3（一様収束性）：極限操作と微分演算の可換性を保証するため、優収束定理により積分記号下での微分可能性を示す。',
      ],
    ];
    bwdVariations = [
      [
        'ステップN-2（矛盾の導出）：これにより仮定された特異点の存在がハウスドルフ分離公理およびノルムの狭義凸性と矛盾する。',
        'ステップN-1（一意性の確定）：背理法より反例は存在せず、補題の成立から主命題の成立が論理的必然として確定する。',
        'ステップN（大域的一意拡張）：したがって主定理の成立条件は局所閉近傍から大域多様体全体へと矛盾なく拡張され、証明が完結する。',
      ],
      [
        'ステップN-2（直交補空間の閉包）：直交分解定理より全空間が核空間と値域の閉包の直和として表現可能であることを確認する。',
        'ステップN-1（境界条件の充足）：随伴作用素の自己共役性からスペクトル測度の一意性が担保され、境界値問題の解が固定される。',
        'ステップN（正則性の完備性）：これにより境界作用素のスペクトル半径は単位開円板内に収縮し、極限解の正則性と連続性が完全に担保される。',
      ],
      [
        'ステップN-2（極限の交換可能性）：ルベーグの優収束定理より積分と極限の可換性が成立し、目的関数の強連続性が保存される。',
        'ステップN-1（一様収束の保持）：連続性の遺伝が全区間で保証され、極限関数が元の微分方程式の古典解であることが示される。',
        'ステップN（解の一意存在性）：連続性の遺伝と微分可能性の保存が全区間で同時に成立し、要求された関数空間における解の一意存在が確定する。',
      ],
      [
        'ステップN-2（同型対応の完結）：基底変換行列の正則性から核空間が自明なゼロ部分空間に縮退し、単射性が確立される。',
        'ステップN-1（論理的帰着）：すべての不変量条件が満たされ、前提公理から結論に至る導出鎖の完全性が閉じられる。',
        'ステップN（全単射の証明）：以上の同型写像の一対一対応と保存則の充足により、目標命題の数学的正当性が何らの飛躍もなく保証される。',
      ],
    ];
  }

  const fwdWaves: TrajectoryWave[] = [];
  const bwdWaves: TrajectoryWave[] = [];
  const K = Math.max(1, kCandidates);

  for (let i = 0; i < K; i++) {
    const steps = [anchors.x_0, ...fwdVariations[i % fwdVariations.length]!];
    const tipVec = embedSentenceUnit(steps[steps.length - 1]!);
    const prevVec = embedSentenceUnit(steps[steps.length - 2]!);
    const dirVec = tipVec.map((v, idx) => v - (prevVec[idx] ?? 0));
    let norm = 0;
    for (const v of dirVec) norm += v * v;
    norm = Math.sqrt(norm);
    const normalizedDir = norm > 1e-7 ? dirVec.map((v) => v / norm) : dirVec;
    fwdWaves.push({ wave_id: i, direction: 'forward', steps, tip_vector: tipVec, direction_vector: normalizedDir });
  }

  for (let j = 0; j < K; j++) {
    const steps = [...bwdVariations[j % bwdVariations.length]!, anchors.x_T];
    const tipVec = embedSentenceUnit(steps[0]!);
    const nextVec = embedSentenceUnit(steps[1]!);
    const dirVec = nextVec.map((v, idx) => v - (tipVec[idx] ?? 0));
    let norm = 0;
    for (const v of dirVec) norm += v * v;
    norm = Math.sqrt(norm);
    const normalizedDir = norm > 1e-7 ? dirVec.map((v) => v / norm) : dirVec;
    bwdWaves.push({ wave_id: j, direction: 'backward', steps, tip_vector: tipVec, direction_vector: normalizedDir });
  }

  return [fwdWaves, bwdWaves];
}

export function getCorpusLatticeNodes(domain: BridgeDomain): string[] {
  return [...(DOMAIN_GOAL_PROPOSALS[domain] ?? DOMAIN_GOAL_PROPOSALS.novel)];
}

export function generateUnconstrainedLatticeWalk(anchors: BoundaryAnchors, stepsCount = 5): string {
  const domain = anchors.domain;
  const theme = anchors.theme;
  const clauses = domain === 'novel'
    ? [
        anchors.x_0,
        `足元から立ち昇る白い霧が、${theme}の古びた木造の壁面をぼんやりと包み込んでいく。`,
        'どこか遠くで乾いた風が吹き抜け、錆びついた看板がかすかな音を立てて小さく揺れていた。',
        '旅人は歩みを止め、暗がりの中にわずかに残る温もりの痕跡を探すように息を潜めた。',
        '周囲の景色は変わり映えせず、同じような小径が幾重にも交差して続いているようだった。',
        '冷え切った空気のなかで、遠くの街灯の光がちらちらと瞬いているのがかろうじて見えた。',
        'しかし冷たい沈黙は深まるばかりで、夜の帳は容赦なく周囲の視界をすべて奪い去っていった。',
      ]
    : domain === 'article'
    ? [
        anchors.x_0,
        '多くの組織現場では即時的なチャット対応に追われ、本来優先すべき長期的思考が著しく圧迫されている。',
        'この現象は個人の努力不足ではなく、同期型プロトコルそのものが内包する構造的認知的負債に起因する。',
        '情報の局所化が進むことで、サイロ化されたチーム間でのコンテキスト同期コストは指数関数的に増加していく。',
        '現場からは業務効率化を求める声が上がっているものの、具体的な解決策はまだ見出されていないのが実情である。',
        '定例会議の数も以前と変わらず、日々のタスク消化に追われる状況が依然として惰性で継続している。',
        'したがって、場当たり的なツールの導入ではなく、情報の永続性と非同期性を中核に据えた組織設計が求められる。',
      ]
    : [
        anchors.x_0,
        'ステップ1：まず位相空間上の連続関数列に対して点ごとの収束性を確認し、近傍系を構成する。',
        'ステップ2：任意の点xにおいて局所的な有界性を検証し、積分区間の可測性を慎重に保証する。',
        'ステップ3：有界変動関数の性質から部分列の収束を議論し、極限候補関数の連続性を導出する。',
        'ステップ4：ただし、一部の境界点において積分の順序交換可能性に関する疑義が依然として残存する。',
        'ステップ5：先行研究における類似の議論を参照しながら、特異点近傍の振る舞いをさらに詳しく検討する。',
        'ステップ6：境界点近傍における微小変動をイプシロン論法で抑え込み、解の存在領域を特定する。',
      ];
  return clauses.slice(0, stepsCount + 1).join('');
}

// ==============================================================================
// 3. Collision Action Integral & Midpoint Infilling Solver
// ==============================================================================

export function computeCollisionAction(
  fwdWave: TrajectoryWave,
  bwdWave: TrajectoryWave,
  beta = 0.5
): [number, number, number, number] {
  const tipFwd = fwdWave.tip_vector;
  const tipBwd = bwdWave.tip_vector;

  // 1. Geodesic distance in semantic manifold
  let distL2Sq = 0;
  let dotDir = 0;
  for (let i = 0; i < tipFwd.length; i++) {
    const d = (tipFwd[i] ?? 0) - (tipBwd[i] ?? 0);
    distL2Sq += d * d;
    dotDir += (fwdWave.direction_vector[i] ?? 0) * (bwdWave.direction_vector[i] ?? 0);
  }
  const distL2 = Math.sqrt(distL2Sq);
  const dirCurvature = 1.0 - Math.max(-1.0, Math.min(1.0, dotDir));
  const dGeodesic = distL2 * 1.5 + dirCurvature * 0.8;

  // 2. Spatial / Causal consistency distance with repetition penalty
  const stepFwdLast = fwdWave.steps[fwdWave.steps.length - 1] ?? '';
  const stepBwdFirst = bwdWave.steps[0] ?? '';

  const eLast = embedSentenceUnit(stepFwdLast);
  const eFirst = embedSentenceUnit(stepBwdFirst);
  let dotE = 0;
  let nL = 0;
  let nF = 0;
  for (let i = 0; i < eLast.length; i++) {
    dotE += (eLast[i] ?? 0) * (eFirst[i] ?? 0);
    nL += (eLast[i] ?? 0) * (eLast[i] ?? 0);
    nF += (eFirst[i] ?? 0) * (eFirst[i] ?? 0);
  }
  const cosSim = dotE / (Math.sqrt(nL * nF) + 1e-7);
  const transitionDist = 1.0 - Math.max(-1.0, Math.min(1.0, cosSim));

  const stemsFwd = new Set(stepFwdLast.match(/[\u4e00-\u9fff]+/g) || []);
  const stemsBwd = new Set(stepBwdFirst.match(/[\u4e00-\u9fff]+/g) || []);

  let intersect = 0;
  for (const s of stemsFwd) {
    if (stemsBwd.has(s)) intersect++;
  }
  const union = new Set([...stemsFwd, ...stemsBwd]).size;
  const jaccard = union > 0 ? intersect / union : 0.2;
  let repPenalty = 0.0;
  if (jaccard > 0.40) repPenalty += 3.0 * (jaccard - 0.40);
  for (let i = 0; i < Math.max(0, stepFwdLast.length - 6); i++) {
    const gram6 = stepFwdLast.slice(i, i + 6);
    if (stepBwdFirst.includes(gram6)) {
      repPenalty += 2.5;
      break;
    }
  }

  const dCausal = transitionDist * 1.2 + repPenalty;

  // 3. 3-Phase QDA of welded sequence
  const weldedText = fwdWave.steps.join('') + bwdWave.steps.join('');
  const qdaRes = computeThreePhaseQdaDistance(weldedText);
  const d3Phase = qdaRes.d_3phase;

  const sTotal = dGeodesic + dCausal + 0.5 * beta * d3Phase;
  return [sTotal, dGeodesic, dCausal, d3Phase];
}

export function solveBidirectionalCollisionBridge(
  anchors: BoundaryAnchors,
  options?: {
    forward_candidates?: string[][];
    backward_candidates?: string[][];
    beta?: number;
    lambda_bridge?: number;
    k_candidates?: number;
  }
): CollisionBridgeResult {
  const beta = options?.beta ?? 0.5;
  const lambdaBridge = options?.lambda_bridge ?? 0.8;
  const kCandidates = options?.k_candidates ?? 4;

  let fwdWaves: TrajectoryWave[];
  let bwdWaves: TrajectoryWave[];

  if (options?.forward_candidates !== undefined && options?.backward_candidates !== undefined) {
    if (options.forward_candidates.length === 0 || options.backward_candidates.length === 0) {
      throw new Error('forward_candidates and backward_candidates must be non-empty arrays.');
    }

    const validFwd = options.forward_candidates.filter((c) => c.length > 0 && c.some((s) => s.trim().length > 0));
    const validBwd = options.backward_candidates.filter((c) => c.length > 0 && c.some((s) => s.trim().length > 0));
    if (validFwd.length === 0 || validBwd.length === 0) {
      throw new Error('forward_candidates and backward_candidates must contain at least one non-empty candidate trajectory.');
    }

    fwdWaves = [];
    bwdWaves = [];
    for (let i = 0; i < Math.min(kCandidates, validFwd.length); i++) {
      const steps = validFwd[i]!;
      const tipVec = embedSentenceUnit(steps[steps.length - 1]!);
      const prevVec = steps.length >= 2 ? embedSentenceUnit(steps[steps.length - 2]!) : tipVec;
      const dirVec = tipVec.map((v, idx) => v - (prevVec[idx] ?? 0));
      let norm = 0;
      for (const v of dirVec) norm += v * v;
      norm = Math.sqrt(norm);
      const normalizedDir = norm > 1e-7 ? dirVec.map((v) => v / norm) : dirVec;
      fwdWaves.push({ wave_id: i, direction: 'forward', steps, tip_vector: tipVec, direction_vector: normalizedDir });
    }
    for (let j = 0; j < Math.min(kCandidates, validBwd.length); j++) {
      const steps = validBwd[j]!;
      const tipVec = embedSentenceUnit(steps[0]!);
      const nextVec = steps.length >= 2 ? embedSentenceUnit(steps[1]!) : tipVec;
      const dirVec = nextVec.map((v, idx) => v - (tipVec[idx] ?? 0));
      let norm = 0;
      for (const v of dirVec) norm += v * v;
      norm = Math.sqrt(norm);
      const normalizedDir = norm > 1e-7 ? dirVec.map((v) => v / norm) : dirVec;
      bwdWaves.push({ wave_id: j, direction: 'backward', steps, tip_vector: tipVec, direction_vector: normalizedDir });
    }
  } else {
    [fwdWaves, bwdWaves] = generateContextualCandidateWaves(anchors, kCandidates);
  }

  const Kfwd = fwdWaves.length;
  const Kbwd = bwdWaves.length;
  const actionMatrix: number[][] = [];

  let bestAction = Infinity;
  let bestPair: [number, number] = [0, 0];
  let bestComponents: [number, number, number] = [0, 0, 0];

  for (let i = 0; i < Kfwd; i++) {
    const row: number[] = [];
    for (let j = 0; j < Kbwd; j++) {
      const [sTot, dGeo, dCau, d3ph] = computeCollisionAction(fwdWaves[i]!, bwdWaves[j]!, beta);
      row.push(sTot);
      if (sTot < bestAction) {
        bestAction = sTot;
        bestPair = [i, j];
        bestComponents = [dGeo, dCau, d3ph];
      }
    }
    actionMatrix.push(row);
  }

  const optFwd = fwdWaves[bestPair[0]]!;
  const optBwd = bwdWaves[bestPair[1]]!;
  const weldedTrajectory = [...optFwd.steps, ...optBwd.steps];
  const weldedText = weldedTrajectory.join('');
  const mBridge = Math.max(0.0, Math.min(1.0, Math.exp(-lambdaBridge * bestAction)));

  return {
    domain: anchors.domain,
    anchors,
    optimal_pair: bestPair,
    min_action: bestAction,
    welded_trajectory: weldedTrajectory,
    welded_text: weldedText,
    m_bridge: mBridge,
    d_geodesic: bestComponents[0],
    d_causal: bestComponents[1],
    d_3phase: bestComponents[2],
    forward_waves: fwdWaves,
    backward_waves: bwdWaves,
    collision_matrix: actionMatrix,
  };
}
