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

export function deriveEndogenousGoalHypotheses(
  x_0: string,
  domain: BridgeDomain = 'novel',
  options?: { candidate_goals?: string[]; k_candidates?: number }
): { optimal_x_T: string; scored_candidates: Array<{ text: string; score: number }> } {
  const preset = DOMAIN_PRESETS[domain] ?? DOMAIN_PRESETS.novel;
  const kamaeTarget = preset.kamae_target;
  const kCandidates = options?.k_candidates ?? 4;
  const safeX0 = x_0 && x_0.trim() ? x_0.trim() : `${domain}の導入境界を確定する。`;
  const candidates = options?.candidate_goals?.filter(c => c && c.trim().length > 0).slice(0, kCandidates);
  if (!candidates || candidates.length === 0) {
    throw new Error('candidate_goals must be explicitly provided to deriveEndogenousGoalHypotheses. Hardcoded text generation is strictly banned.');
  }

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
  options?: { theme?: string; beta?: number; forward_candidates?: string[][]; backward_candidates?: string[][]; candidate_goals?: string[] }
): CollisionBridgeResult {
  const cleaned = initialRolloutText.trim();
  const rawParts = cleaned.split(/[。！？\n]+/).map((s) => s.trim()).filter((s) => s.length >= 5);
  const sentences = rawParts.map((s) => s + '。');

  let x_0: string;
  let x_T: string;

  if (sentences.length === 0) {
    x_0 = `${options?.theme || 'closed_loop_rewrite'}の導入境界を確定する。`;
    const res = deriveEndogenousGoalHypotheses(x_0, domain, { candidate_goals: options?.candidate_goals });
    x_T = res.optimal_x_T;
  } else if (sentences.length === 1) {
    x_0 = sentences[0]!;
    const res = deriveEndogenousGoalHypotheses(x_0, domain, { candidate_goals: options?.candidate_goals });
    x_T = res.optimal_x_T;
  } else {
    x_0 = sentences[0]!;
    x_T = sentences[sentences.length - 1]!;
    if (x_0 === x_T || x_T.length < 8) {
      const res = deriveEndogenousGoalHypotheses(x_0, domain, { candidate_goals: options?.candidate_goals });
      x_T = res.optimal_x_T;
    }
  }

  const anchors = pinBoundaryAnchors(options?.theme || 'closed_loop_rewrite', domain, {
    custom_start: x_0,
    custom_end: x_T,
  });

  const fwdCandidates = options?.forward_candidates ?? (
    sentences.length >= 3 ? sentences.slice(1, -1).map(s => [sentences[0]!, s]) : [[x_0, '事態の推移を観測する。']]
  );
  const bwdCandidates = options?.backward_candidates ?? (
    sentences.length >= 3 ? sentences.slice(1, -1).map(s => [s, x_T]) : [['境界条件の整合性を確認し、', x_T]]
  );

  return solveBidirectionalCollisionBridge(anchors, {
    forward_candidates: fwdCandidates,
    backward_candidates: bwdCandidates,
    beta: options?.beta ?? 0.5,
  });
}

// ------------------------------------------------------------------------------
// Contextual Candidate Wave Generator
// ------------------------------------------------------------------------------

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
    throw new Error('forward_candidates and backward_candidates must be explicitly provided. Hardcoded text generation is strictly banned.');
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
