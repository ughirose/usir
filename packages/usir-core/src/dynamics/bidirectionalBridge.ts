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
      x_0 = `${theme}の気配が漂い、張り詰めた冷気が指先をかすかに凍らせていた。`;
    } else if (domain === 'article') {
      x_0 = `${theme}に関する最新の観測データは、従来の想定と明確に乖離している。`;
    } else {
      x_0 = `前提：問題設定「${theme}」における初期条件および束縛条件を確定する。`;
    }
  }

  let x_T: string;
  if (options?.custom_end) {
    x_T = options.custom_end;
  } else {
    if (domain === 'novel') {
      x_T = `微かな軋みとともに輪郭が沈み、あとに冷たい沈黙だけが残った。`;
    } else if (domain === 'article') {
      x_T = `したがって、本質的な競争優位は構造の非対称性と同期コストの削減に帰着する。`;
    } else {
      x_T = `結論：すべての背理条件が排除され、目標命題が唯一無二に証明された。`;
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

function generateDeterministicRollouts(
  anchors: BoundaryAnchors,
  kCandidates = 4
): [TrajectoryWave[], TrajectoryWave[]] {
  const domain = anchors.domain;
  const fwdWaves: TrajectoryWave[] = [];
  const bwdWaves: TrajectoryWave[] = [];

  let fwdVariations: string[][];
  let bwdVariations: string[][];

  if (domain === 'novel') {
    fwdVariations = [
      [
        anchors.x_0,
        '土間の踏み石に冷たい雨滴が落ち、微かな飛沫を上げた。',
        '薄暗い作業台の奥で、使い込まれた鉄の道具が鈍い光を返している。',
      ],
      [
        anchors.x_0,
        '板壁の隙間から吹き込む風が、吊るされた裸電球の傘をかすかに揺らした。',
        '老人は無言のまま指先を動かし、濡れた粘土の固さを慎重に確かめている。',
      ],
      [
        anchors.x_0,
        '遠い谷底の川音が闇を抜けて響き、作業場の静寂を際立たせていた。',
        '削り屑の積もった轆轤の軸に手を添え、低く軋む感触を掌で測る。',
      ],
      [
        anchors.x_0,
        '湿った杉の香りが満ちる中、小さな薪ストーブが赤い火の粉を散らした。',
        '乾いた土壁に老人の影が大きく揺れ、作業の緊張が高まっていく。',
      ],
    ];

    bwdVariations = [
      [
        '窯の扉が重く閉じられ、焚口から漏れる余熱が徐々に鎮まっていく。',
        '手入れを終えた道具が元の木箱へ静かに収められた。',
        anchors.x_T,
      ],
      [
        '指先に伝わる土の抵抗が均一になり、轆轤の回転が緩やかに速度を落とした。',
        '研ぎ澄まされた刃先が最後の余分な削り屑を削ぎ落とす。',
        anchors.x_T,
      ],
      [
        '夜雨の音が遠ざかり、山裾の夜霧が建物の屋根を白く包み込んでいた。',
        '男は革手袋を外し、冷え切った指先を息で温めた。',
        anchors.x_T,
      ],
      [
        '路地の外灯が一瞬瞬き、闇の奥へと長い影を引いて消えた。',
        '床に落ちた金属片を靴底で掃き集め、作業の完了を確かめる。',
        anchors.x_T,
      ],
    ];
  } else if (domain === 'article') {
    fwdVariations = [
      [
        anchors.x_0,
        '現場のヒアリング調査では、78%の回答者が同一の情報伝達経路における滞留を指摘している。',
        'この数値は単なるコミュニケーション不足ではなく、業務プロセスの構造的不全を示唆する。',
      ],
      [
        anchors.x_0,
        '過去3年間の時系列推移を精査すると、組織規模の拡大に反比例して意思決定速度が低下していた。',
        '階層型承認フローの各ノードにおいて、情報の非対称性が指数関数的に増大している。',
      ],
      [
        anchors.x_0,
        '初期投資と運用保守コストの対比分析から、既存システムのサンクコストが変革を阻んでいる実態が浮き彫りとなった。',
        '現場の暗黙知がシステムへ還元されず、属人的な負荷の偏在が常態化している。',
      ],
      [
        anchors.x_0,
        '先行事例のベンチマーク比較によれば、部分最適を繰り返した組織ほど全体最適化の障壁が高い。',
        '局所的な自動化ツール導入が、かえって部署間の断絶を深刻化させている。',
      ],
    ];

    bwdVariations = [
      [
        'この制度設計により、各ステークホルダーの自律的インセンティブが完全に合致する。',
        '中間調整コストを物理的に排除した非同期情報流通基盤が不可欠となる。',
        anchors.x_T,
      ],
      [
        '実証実験の結果は、プロセスの脱中心化が平均リードタイムを42%短縮することを実証した。',
        'したがって、評価指標を活動量からアウトカムの再現性へと転換しなければならない。',
        anchors.x_T,
      ],
      [
        '局所的な対症療法を排し、アーキテクチャレベルでの責任分離を断行することが唯一の解である。',
        'データの単一信頼源（SSOT）を確立し、意思決定の検証可能性を担保する。',
        anchors.x_T,
      ],
      [
        '以上の比較検証から導かれる結論は、スケーラビリティの限界を突破する新標準を提示している。',
        '持続可能な運用体制への移行ロードマップを策定し、段階的移行を完了する customer。',
        anchors.x_T,
      ],
    ];
  } else {
    // cot_reasoning
    fwdVariations = [
      [
        anchors.x_0,
        'ステップ1（順方向演繹）：初期状態S_0に対して不変量保存則を適用し、状態空間の有界性を導出する。',
        'ステップ2（境界条件）：変数xの定義域における連続性により、特異点近傍での発散が排除される。',
      ],
      [
        anchors.x_0,
        'ステップ1（演繹展開）：補題1.1により、群Gの正規部分群Hに対する準同型写像Phiの核Ker(Phi)を決定する。',
        'ステップ2（同型対応）：準同型定理を適用すると、剰余群G/Ker(Phi)と像Im(Phi)の間に自然な同型が成立する。',
      ],
      [
        anchors.x_0,
        'ステップ1（帰納的基底）：基底ケースn=1において不等式が自明に成立することを確認する。',
        'ステップ2（帰納的仮定）：n=kでの成立を仮定し、n=k+1での展開項の差分評価式を導出する。',
      ],
      [
        anchors.x_0,
        'ステップ1（対称性解析）：ハミルトン系の時間反転対称性により、位相空間の体積保存則（リウヴィルの定理）が成り立つ。',
        'ステップ2（エネルギー面）：運動方程式の第一積分より、解軌道はコンパクトなエネルギー曲面上に拘束される。',
      ],
    ];

    bwdVariations = [
      [
        'ステップN-1（逆方向逆算）：目標結論が成り立つための必要十分条件として、中間補題L_midの成立が要求される。',
        'ステップN（整合溶接）：順方向の有界性評価と逆方向の必要条件が中間点において完全に一致する。',
        anchors.x_T,
      ],
      [
        'ステップN-1（ゴール逆算）：最終主張の対偶を仮定した際、極小反例の存在が直前の同型構造と矛盾する。',
        'ステップN（背理完了）：よって反例は存在せず、中間状態での整合性が数学的に保証される。',
        anchors.x_T,
      ],
      [
        'ステップN-1（目標到達条件）：k+1での差分上限が正であることを示すため、中間不等式の縮小写像性を検証する。',
        'ステップN（統合証明）：両方向の境界が一点に収束し、すべての自然数nに対する命題が完結する。',
        anchors.x_T,
      ],
      [
        'ステップN-1（終端アトラクター）：目標終端状態への漸近安定性は、リアプノフ関数の負定値性によって定まる。',
        'ステップN（不変多様体結合）：順方向軌道と逆方向随伴軌道が中間超曲面上で横断的に衝突・接続する。',
        anchors.x_T,
      ],
    ];
  }

  const K = Math.min(kCandidates, fwdVariations.length);
  for (let i = 0; i < K; i++) {
    const steps = fwdVariations[i]!;
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
    const steps = bwdVariations[j]!;
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

  // 2. Spatial / Causal consistency distance
  const stepFwdLast = fwdWave.steps[fwdWave.steps.length - 1] ?? '';
  const stepBwdFirst = bwdWave.steps[0] ?? '';

  const stemsFwd = new Set(stepFwdLast.match(/[\u4e00-\u9fff]+/g) || []);
  const stemsBwd = new Set(stepBwdFirst.match(/[\u4e00-\u9fff]+/g) || []);

  let intersect = 0;
  for (const s of stemsFwd) {
    if (stemsBwd.has(s)) intersect++;
  }
  const union = new Set([...stemsFwd, ...stemsBwd]).size;
  const jaccard = union > 0 ? intersect / union : 0.2;
  const dCausal = (1.0 - jaccard) * 1.2;

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

  if (options?.forward_candidates && options?.backward_candidates) {
    fwdWaves = [];
    bwdWaves = [];
    for (let i = 0; i < Math.min(kCandidates, options.forward_candidates.length); i++) {
      const steps = options.forward_candidates[i]!;
      const tipVec = embedSentenceUnit(steps[steps.length - 1]!);
      const prevVec = steps.length >= 2 ? embedSentenceUnit(steps[steps.length - 2]!) : tipVec;
      const dirVec = tipVec.map((v, idx) => v - (prevVec[idx] ?? 0));
      let norm = 0;
      for (const v of dirVec) norm += v * v;
      norm = Math.sqrt(norm);
      const normalizedDir = norm > 1e-7 ? dirVec.map((v) => v / norm) : dirVec;
      fwdWaves.push({ wave_id: i, direction: 'forward', steps, tip_vector: tipVec, direction_vector: normalizedDir });
    }
    for (let j = 0; j < Math.min(kCandidates, options.backward_candidates.length); j++) {
      const steps = options.backward_candidates[j]!;
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
    [fwdWaves, bwdWaves] = generateDeterministicRollouts(anchors, kCandidates);
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
