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
import { CLOSED_LOOP_BENCHMARK_CORPUS } from './closedLoopCorpus.js';

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

const DOMAIN_LATTICE_POOLS: Record<BridgeDomain, string[]> = {
  novel: [
    '冷え切った空気のなかに微かな足音が吸い込まれていく',
    '雪明かりが遠くの木々の輪郭を淡く照らし出す',
    '凍りついた小枝が風に揺れて細い乾いた音を立てた',
    '吐く息が白く霧となって消え去るのを見つめる',
    '暗がりの中で古い建物の羽目板がきしむ',
    '遠くの谷底から風の呻き声が低く響いてきた',
    '足元の雪を踏みしめる鈍い音が静寂を破る',
    '閉ざされた扉の隙間からわずかな温もりが漏れていた',
  ],
  article: [
    '同期コミュニケーションは即時応答の錯覚を生み出す一方で思考の深化を阻害する',
    '非同期な文書化の徹底により文脈の永続性と再利用性が飛躍的に向上する',
    '組織内の知識伝達におけるレイテンシを削減するには情報ハブの分散が不可欠である',
    '不要な定例会議を廃止し非同期のレビューサイクルへ移行することで集中時間を確保する',
    'コミュニケーションの非対称性を認めることが認知的負荷の最小化に直結する',
    'テキストベースの明確な合意形成が心理的安全性と実行速度を両立させる',
    '暗黙知を形式知へ転換するプロセスがスケーラブルな組織基盤を構築する',
    '各メンバーが自律的にコンテキストを同期できる情報台帳の整備が求められる',
  ],
  cot_reasoning: [
    '初期公理に基づき有界閉区間上の連続関数列の収束特性を精査する',
    'コーシーの判定条件を適用して一様収束極限関数の存在と連続性を担保する',
    '任意の正数イプシロンに対して全区間で成立する自然数Nを一様に確定する',
    '平均値の定理を差分商に適用し導関数列の一様有界性を導出する',
    '極限操作と微分演算の交換可能性に関する積分表示を確立する',
    '背理法により不連続点または特異点の存在仮定が矛盾を生じることを示す',
    '各点収束と一様収束の位相的乖離を排除して微分方程式の解の正則性を証明する',
    '境界条件の整合性を確認し補題の成立から主定理の結論を確定する',
  ],
};

export function getCorpusLatticeNodes(domain: BridgeDomain): string[] {
  return [...(DOMAIN_LATTICE_POOLS[domain] ?? DOMAIN_LATTICE_POOLS.novel)];
}

function generateDeterministicRollouts(
  anchors: BoundaryAnchors,
  kCandidates = 4
): [TrajectoryWave[], TrajectoryWave[]] {
  const domain = anchors.domain;
  const nodes = getCorpusLatticeNodes(domain);
  const q0 = anchors.q_start;
  const qT = anchors.q_end;

  const vFwd: number[] = [];
  let sumSq = 0;
  for (let i = 0; i < 24; i++) {
    const d = (qT[i] ?? 0) - (q0[i] ?? 0);
    vFwd.push(d);
    sumSq += d * d;
  }
  const normFwd = Math.sqrt(sumSq);
  if (normFwd > 1e-6) {
    for (let i = 0; i < 24; i++) vFwd[i] = (vFwd[i] ?? 0) / normFwd;
  }

  const embs = nodes.map((c) => ({ emb: embedSentenceUnit(c), text: c }));

  const dot = (a: number[], b: number[]) => {
    let s = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) s += (a[i] ?? 0) * (b[i] ?? 0);
    return s;
  };

  const fwdScored = embs
    .map((item) => ({ ...item, score: dot(item.emb, vFwd) + 0.3 * dot(item.emb, q0) }))
    .sort((a, b) => b.score - a.score);

  const bwdScored = embs
    .map((item) => ({ ...item, score: -dot(item.emb, vFwd) + 0.3 * dot(item.emb, qT) }))
    .sort((a, b) => b.score - a.score);

  const fwdWaves: TrajectoryWave[] = [];
  const bwdWaves: TrajectoryWave[] = [];
  const K = Math.max(1, kCandidates);

  for (let i = 0; i < K; i++) {
    const c1 = fwdScored[i % fwdScored.length]?.text ?? '静かに雨が降り続く';
    const c2 = fwdScored[(i + K) % fwdScored.length]?.text ?? '水滴が土間を濡らす';
    const steps =
      domain === 'cot_reasoning'
        ? [anchors.x_0, `ステップ1（順方向演繹）：${c1}。`, `ステップ2（境界展開）：${c2}。`]
        : [anchors.x_0, `${c1}。`, `${c2}。`];

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
    const p1 = bwdScored[(j + K) % bwdScored.length]?.text ?? '作業の完了を確かめる';
    const p2 = bwdScored[j % bwdScored.length]?.text ?? '道具を箱へと収めた';
    const steps =
      domain === 'cot_reasoning'
        ? [`ステップN-1（逆方向逆算）：${p1}。`, `ステップN（整合結合）：${p2}。`, anchors.x_T]
        : [`${p1}。`, `${p2}。`, anchors.x_T];

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

export function generateUnconstrainedLatticeWalk(anchors: BoundaryAnchors, stepsCount = 3): string {
  const nodes = getCorpusLatticeNodes(anchors.domain);
  let qCurr = [...anchors.q_start];
  const walkSteps: string[] = [anchors.x_0];

  const embs = nodes.map((c) => ({ emb: embedSentenceUnit(c), text: c }));
  const dot = (a: number[], b: number[]) => {
    let s = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) s += (a[i] ?? 0) * (b[i] ?? 0);
    return s;
  };

  for (let stepI = 0; stepI < stepsCount; stepI++) {
    const scored = embs
      .map((item) => ({ ...item, sim: dot(item.emb, qCurr) }))
      .sort((a, b) => b.sim - a.sim);
    const pickIdx = Math.min(stepI * 3 + 2, scored.length - 1);
    const picked = scored[pickIdx] ?? scored[0];
    if (picked) {
      walkSteps.push(`${picked.text}。`);
      qCurr = qCurr.map((v, i) => 0.7 * v + 0.3 * (picked.emb[i] ?? 0));
    }
  }

  return walkSteps.join('');
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
