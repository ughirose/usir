"""USIR v4.0 Bidirectional Schrödinger Bridge (Meet-in-the-Middle Collision Engine).

Implements:
1. Boundary Pinning (pin_boundary_anchors):
   Pins x_0 (Left-to-Right introduction anchor) and x_T (Right-to-Left W_end* landing anchor)
   parameterized by 24D Kamae (Sg-10..12) macro constraints.
2. Bidirectional Rollout & Midpoint Collision Solver (solve_bidirectional_collision_bridge):
   - Forward Wave psi: Left-to-Right rollout {tau_i} from x_0
   - Backward Wave phi: Right-to-Left adjoint rollout {tau_j} from x_T
   - Collision Action Integral:
     S_collision(tau_i, tau_j) = D_geodesic(e_tip_i, e_tip_j)
                               + D_spatial_causal(tau_i, tau_j)
                               + (beta/2) * D_3phase(z(tau_i o tau_j))
   - Optimal Midpoint Collision & Infilling Welding
3. Universal 3-Domain Support:
   - "novel": Sensory perception -> Physical state change
   - "article": Empirical observation/data -> Structural insight
   - "cot_reasoning": Premise forward deduction x Goal backward abduction collision
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .extractor_24d import extract_usir_24d
from .manifold_geodesic import (
    _embed_sentence_unit,
    compute_semantic_geodesic_curvature,
    compute_three_phase_qda_distance,
)


@dataclass
class BoundaryAnchors:
    domain: str  # "novel" | "article" | "cot_reasoning"
    theme: str
    x_0: str  # Left boundary (premise / start observation)
    x_T: str  # Right boundary (conclusion / landing predicate W_end*)
    q_start: np.ndarray
    q_end: np.ndarray
    kamae_coords: np.ndarray  # Sg-10, Sg-11, Sg-12
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryWave:
    wave_id: int
    direction: str  # "forward" | "backward"
    steps: List[str]
    tip_vector: np.ndarray
    direction_vector: np.ndarray
    cumulative_energy: float = 0.0


@dataclass
class CollisionBridgeResult:
    domain: str
    anchors: BoundaryAnchors
    optimal_pair: Tuple[int, int]  # (forward_idx, backward_idx)
    min_action: float
    welded_trajectory: List[str]
    welded_text: str
    m_bridge: float
    d_geodesic: float
    d_causal: float
    d_3phase: float
    forward_waves: List[TrajectoryWave]
    backward_waves: List[TrajectoryWave]
    collision_matrix: np.ndarray


# ==============================================================================
# 1. Boundary Pinning (pin_boundary_anchors)
# ==============================================================================

DOMAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "novel": {
        "kamae_target": np.array([0.55, 0.60, 0.45]),  # High Sg-11 (余白) & balanced tension
        "start_template": "観測者の身体知覚と空間の物理輪郭：{theme}",
        "end_template": "不可逆な物理変化と静寂の着地点：{theme}",
    },
    "article": {
        "kamae_target": np.array([0.70, 0.40, 0.35]),  # Structured clarity & low shock
        "start_template": "具体的データと現場の課題提起：{theme}",
        "end_template": "構造的アナロジーに基づく普遍的提言：{theme}",
    },
    "cot_reasoning": {
        "kamae_target": np.array([0.85, 0.20, 0.25]),  # Strict logical lattice & low void
        "start_template": "前提条件と初期公理の確定：{theme}",
        "end_template": "論理的必然性による最終証明目標 Q.E.D.：{theme}",
    },
}


def pin_boundary_anchors(
    theme: str,
    domain: str = "novel",
    target_state: Optional[np.ndarray] = None,
    custom_start: Optional[str] = None,
    custom_end: Optional[str] = None,
) -> BoundaryAnchors:
    """Pins boundary anchors x_0 and x_T with 24D Kamae (Sg-10..12) alignment."""
    if domain not in DOMAIN_PRESETS:
        domain = "novel"
    preset = DOMAIN_PRESETS[domain]

    if custom_start:
        x_0 = custom_start
    else:
        if domain == "novel":
            x_0 = f"{theme}の気配が漂い、張り詰めた冷気が指先をかすかに凍らせていた。"
        elif domain == "article":
            x_0 = f"{theme}に関する最新の観測データは、従来の想定と明確に乖離している。"
        else:  # cot_reasoning
            x_0 = f"前提：問題設定「{theme}」における初期条件および束縛条件を確定する。"

    if custom_end:
        x_T = custom_end
    else:
        if domain == "novel":
            x_T = f"微かな軋みとともに輪郭が沈み、あとに冷たい沈黙だけが残った。"
        elif domain == "article":
            x_T = f"したがって、本質的な競争優位は構造の非対称性と同期コストの削減に帰着する。"
        else:  # cot_reasoning
            x_T = f"結論：すべての背理条件が排除され、目標命題が唯一無二に証明された。"

    # Compute boundary vectors
    q_start = _embed_sentence_unit(x_0)
    q_end = _embed_sentence_unit(x_T)

    kamae_coords = preset["kamae_target"] if target_state is None else np.asarray(target_state[9:12], dtype=float)

    return BoundaryAnchors(
        domain=domain,
        theme=theme,
        x_0=x_0,
        x_T=x_T,
        q_start=q_start,
        q_end=q_end,
        kamae_coords=kamae_coords,
        metadata={"preset": domain},
    )


# ==============================================================================
# 2. Candidate Wave Generation (Forward & Backward Proposers)
# ==============================================================================

def _generate_deterministic_rollouts(
    anchors: BoundaryAnchors,
    k_candidates: int = 4,
) -> Tuple[List[TrajectoryWave], List[TrajectoryWave]]:
    """Generates K forward and K backward trajectory rollout candidates.

    In production, this is driven by LLM logits sampling or Twisted SMC.
    In testing/CI, deterministic multi-step candidate trajectories are generated.
    """
    domain = anchors.domain
    theme = anchors.theme

    fwd_waves: List[TrajectoryWave] = []
    bwd_waves: List[TrajectoryWave] = []

    if domain == "novel":
        fwd_variations = [
            [
                anchors.x_0,
                "土間の踏み石に冷たい雨滴が落ち、微かな飛沫を上げた。",
                "薄暗い作業台の奥で、使い込まれた鉄の道具が鈍い光を返している。",
            ],
            [
                anchors.x_0,
                "板壁の隙間から吹き込む風が、吊るされた裸電球の傘をかすかに揺らした。",
                "老人は無言のまま指先を動かし、濡れた粘土の固さを慎重に確かめている。",
            ],
            [
                anchors.x_0,
                "遠い谷底の川音が闇を抜けて響き、作業場の静寂を際立たせていた。",
                "削り屑の積もった轆轤の軸に手を添え、低く軋む感触を掌で測る。",
            ],
            [
                anchors.x_0,
                "湿った杉の香りが満ちる中、小さな薪ストーブが赤い火の粉を散らした。",
                "乾いた土壁に老人の影が大きく揺れ、作業の緊張が高まっていく。",
            ],
        ]

        bwd_variations = [
            [
                "窯の扉が重く閉じられ、焚口から漏れる余熱が徐々に鎮まっていく。",
                "手入れを終えた道具が元の木箱へ静かに収められた。",
                anchors.x_T,
            ],
            [
                "指先に伝わる土の抵抗が均一になり、轆轤の回転が緩やかに速度を落とした。",
                "研ぎ澄まされた刃先が最後の余分な削り屑を削ぎ落とす。",
                anchors.x_T,
            ],
            [
                "夜雨の音が遠ざかり、山裾の夜霧が建物の屋根を白く包み込んでいた。",
                "男は革手袋を外し、冷え切った指先を息で温めた。",
                anchors.x_T,
            ],
            [
                "路地の外灯が一瞬瞬き、闇の奥へと長い影を引いて消えた。",
                "床に落ちた金属片を靴底で掃き集め、作業の完了を確かめる。",
                anchors.x_T,
            ],
        ]

    elif domain == "article":
        fwd_variations = [
            [
                anchors.x_0,
                "現場のヒアリング調査では、78%の回答者が同一の情報伝達経路における滞留を指摘している。",
                "この数値は単なるコミュニケーション不足ではなく、業務プロセスの構造的不全を示唆する。",
            ],
            [
                anchors.x_0,
                "過去3年間の時系列推移を精査すると、組織規模の拡大に反比例して意思決定速度が低下していた。",
                "階層型承認フローの各ノードにおいて、情報の非対称性が指数関数的に増大している。",
            ],
            [
                anchors.x_0,
                "初期投資と運用保守コストの対比分析から、既存システムのサンクコストが変革を阻んでいる実態が浮き彫りとなった。",
                "現場の暗黙知がシステムへ還元されず、属人的な負荷の偏在が常態化している。",
            ],
            [
                anchors.x_0,
                "先行事例のベンチマーク比較によれば、部分最適を繰り返した組織ほど全体最適化の障壁が高い。",
                "局所的な自動化ツール導入が、かえって部署間の断絶を深刻化させている。",
            ],
        ]

        bwd_variations = [
            [
                "この制度設計により、各ステークホルダーの自律的インセンティブが完全に合致する。",
                "中間調整コストを物理的に排除した非同期情報流通基盤が不可欠となる。",
                anchors.x_T,
            ],
            [
                "実証実験の結果は、プロセスの脱中心化が平均リードタイムを42%短縮することを実証した。",
                "したがって、評価指標を活動量からアウトカムの再現性へと転換しなければならない。",
                anchors.x_T,
            ],
            [
                "局所的な対症療法を排し、アーキテクチャレベルでの責任分離を断行することが唯一の解である。",
                "データの単一信頼源（SSOT）を確立し、意思決定の検証可能性を担保する。",
                anchors.x_T,
            ],
            [
                "以上の比較検証から導かれる結論は、スケーラビリティの限界を突破する新標準を提示している。",
                "持続可能な運用体制への移行ロードマップを策定し、段階的移行を完了する。",
                anchors.x_T,
            ],
        ]

    else:  # cot_reasoning
        fwd_variations = [
            [
                anchors.x_0,
                "ステップ1（順方向演繹）：初期状態S_0に対して不変量保存則を適用し、状態空間の有界性を導出する。",
                "ステップ2（境界条件）：変数xの定義域における連続性により、特異点近傍での発散が排除される。",
            ],
            [
                anchors.x_0,
                "ステップ1（演繹展開）：補題1.1により、群Gの正規部分群Hに対する準同型写像Phiの核Ker(Phi)を決定する。",
                "ステップ2（同型対応）：準同型定理を適用すると、剰余群G/Ker(Phi)と像Im(Phi)の間に自然な同型が成立する。",
            ],
            [
                anchors.x_0,
                "ステップ1（帰納的基底）：基底ケースn=1において不等式が自明に成立することを確認する。",
                "ステップ2（帰納的仮定）：n=kでの成立を仮定し、n=k+1での展開項の差分評価式を導出する。",
            ],
            [
                anchors.x_0,
                "ステップ1（対称性解析）：ハミルトン系の時間反転対称性により、位相空間の体積保存則（リウヴィルの定理）が成り立つ。",
                "ステップ2（エネルギー面）：運動方程式の第一積分より、解軌道はコンパクトなエネルギー曲面上に拘束される。",
            ],
        ]

        bwd_variations = [
            [
                "ステップN-1（逆方向逆算）：目標結論が成り立つための必要十分条件として、中間補題L_midの成立が要求される。",
                "ステップN（整合溶接）：順方向の有界性評価と逆方向の必要条件が中間点において完全に一致する。",
                anchors.x_T,
            ],
            [
                "ステップN-1（ゴール逆算）：最終主張の対偶を仮定した際、極小反例の存在が直前の同型構造と矛盾する。",
                "ステップN（背理完了）：よって反例は存在せず、中間状態での整合性が数学的に保証される。",
                anchors.x_T,
            ],
            [
                "ステップN-1（目標到達条件）：k+1での差分上限が正であることを示すため、中間不等式の縮小写像性を検証する。",
                "ステップN（統合証明）：両方向の境界が一点に収束し、すべての自然数nに対する命題が完結する。",
                anchors.x_T,
            ],
            [
                "ステップN-1（終端アトラクター）：目標終端状態への漸近安定性は、リアプノフ関数の負定値性によって定まる。",
                "ステップN（不変多様体結合）：順方向軌道と逆方向随伴軌道が中間超曲面上で横断的に衝突・接続する。",
                anchors.x_T,
            ],
        ]

    for i in range(min(k_candidates, len(fwd_variations))):
        steps = fwd_variations[i]
        tip_vec = _embed_sentence_unit(steps[-1])
        dir_vec = tip_vec - _embed_sentence_unit(steps[-2])
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-7:
            dir_vec = dir_vec / norm
        fwd_waves.append(TrajectoryWave(wave_id=i, direction="forward", steps=steps, tip_vector=tip_vec, direction_vector=dir_vec))

    for j in range(min(k_candidates, len(bwd_variations))):
        steps = bwd_variations[j]
        tip_vec = _embed_sentence_unit(steps[0])
        dir_vec = _embed_sentence_unit(steps[1]) - tip_vec
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-7:
            dir_vec = dir_vec / norm
        bwd_waves.append(TrajectoryWave(wave_id=j, direction="backward", steps=steps, tip_vector=tip_vec, direction_vector=dir_vec))

    return fwd_waves, bwd_waves


# ==============================================================================
# 3. Collision Action Integral & Midpoint Infilling Solver
# ==============================================================================

def compute_collision_action(
    fwd_wave: TrajectoryWave,
    bwd_wave: TrajectoryWave,
    beta: float = 0.5,
) -> Tuple[float, float, float, float]:
    r"""Computes the collision action integral:

    \mathcal{S}_{\text{collision}}(\vec{\tau}_i, \overleftarrow{\tau}_j) =
        \mathcal{D}_{\text{geodesic}}(\vec{\mathbf{e}}_{\text{tip}}, \overleftarrow{\mathbf{e}}_{\text{tip}})
        + \mathcal{D}_{\text{spatial\_causal}}(\vec{\tau}_i, \overleftarrow{\tau}_j)
        + \frac{\beta}{2} \mathcal{D}_{\text{3-phase}}(\mathbf{z}(\vec{\tau}_i \circ \overleftarrow{\tau}_j))

    Returns:
        (s_total, d_geodesic, d_causal, d_3phase)
    """
    tip_fwd = fwd_wave.tip_vector
    tip_bwd = bwd_wave.tip_vector

    # 1. Geodesic distance in semantic manifold
    diff_tip = tip_fwd - tip_bwd
    dist_l2 = float(np.linalg.norm(diff_tip))
    cos_dir = float(np.dot(fwd_wave.direction_vector, bwd_wave.direction_vector))
    # Desired: directions are smooth and aligned (cos_dir > 0 => low curvature)
    dir_curvature = 1.0 - np.clip(cos_dir, -1.0, 1.0)
    d_geodesic = dist_l2 * 1.5 + dir_curvature * 0.8

    # 2. Spatial / Causal consistency distance
    # Evaluates transition continuity between last step of forward and first step of backward
    step_fwd_last = fwd_wave.steps[-1]
    step_bwd_first = bwd_wave.steps[0]

    stems_fwd = set(re.findall(r'[\u4e00-\u9fff]+', step_fwd_last))
    stems_bwd = set(re.findall(r'[\u4e00-\u9fff]+', step_bwd_first))

    # Jaccard overlap of content stems
    if stems_fwd or stems_bwd:
        jaccard = len(stems_fwd & stems_bwd) / float(len(stems_fwd | stems_bwd))
    else:
        jaccard = 0.2
    d_causal = float(1.0 - jaccard) * 1.2

    # 3. 3-Phase QDA of welded sequence
    welded_text = "".join(fwd_wave.steps) + "".join(bwd_wave.steps)
    qda_res = compute_three_phase_qda_distance(welded_text)
    d_3phase = qda_res["d_3phase"]

    # Total Action
    s_total = float(d_geodesic + d_causal + 0.5 * beta * d_3phase)

    return s_total, d_geodesic, d_causal, d_3phase


def solve_bidirectional_collision_bridge(
    anchors: BoundaryAnchors,
    forward_candidates: Optional[List[List[str]]] = None,
    backward_candidates: Optional[List[List[str]]] = None,
    beta: float = 0.5,
    lambda_bridge: float = 0.8,
    k_candidates: int = 4,
) -> CollisionBridgeResult:
    """Solves the meet-in-the-middle Schrödinger bridge collision.

    Rolls out Forward Wave psi (Left-to-Right) and Backward Wave phi (Right-to-Left),
    computes S_collision across all K x K pairs, selects the minimal action trajectory,
    and welds the midpoint into a single cohesive narrative / reasoning chain.
    """
    if forward_candidates is not None and backward_candidates is not None:
        fwd_waves: List[TrajectoryWave] = []
        bwd_waves: List[TrajectoryWave] = []
        for i, steps in enumerate(forward_candidates[:k_candidates]):
            tip_vec = _embed_sentence_unit(steps[-1])
            prev_vec = _embed_sentence_unit(steps[-2]) if len(steps) >= 2 else tip_vec
            dir_vec = tip_vec - prev_vec
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-7:
                dir_vec = dir_vec / norm
            fwd_waves.append(TrajectoryWave(wave_id=i, direction="forward", steps=steps, tip_vector=tip_vec, direction_vector=dir_vec))

        for j, steps in enumerate(backward_candidates[:k_candidates]):
            tip_vec = _embed_sentence_unit(steps[0])
            next_vec = _embed_sentence_unit(steps[1]) if len(steps) >= 2 else tip_vec
            dir_vec = next_vec - tip_vec
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-7:
                dir_vec = dir_vec / norm
            bwd_waves.append(TrajectoryWave(wave_id=j, direction="backward", steps=steps, tip_vector=tip_vec, direction_vector=dir_vec))
    else:
        fwd_waves, bwd_waves = _generate_deterministic_rollouts(anchors, k_candidates=k_candidates)

    K_fwd = len(fwd_waves)
    K_bwd = len(bwd_waves)
    action_matrix = np.zeros((K_fwd, K_bwd), dtype=float)

    best_action = float("inf")
    best_pair = (0, 0)
    best_components = (0.0, 0.0, 0.0)

    for i in range(K_fwd):
        for j in range(K_bwd):
            s_tot, d_geo, d_cau, d_3ph = compute_collision_action(fwd_waves[i], bwd_waves[j], beta=beta)
            action_matrix[i, j] = s_tot
            if s_tot < best_action:
                best_action = s_tot
                best_pair = (i, j)
                best_components = (d_geo, d_cau, d_3ph)

    # Weld the optimal forward and backward waves
    opt_fwd = fwd_waves[best_pair[0]]
    opt_bwd = bwd_waves[best_pair[1]]

    welded_trajectory: List[str] = list(opt_fwd.steps) + list(opt_bwd.steps)
    welded_text = "".join(welded_trajectory)

    # Bridge multiplier M_bridge in [0, 1]
    m_bridge = float(np.clip(math.exp(-lambda_bridge * best_action), 0.0, 1.0))

    return CollisionBridgeResult(
        domain=anchors.domain,
        anchors=anchors,
        optimal_pair=best_pair,
        min_action=best_action,
        welded_trajectory=welded_trajectory,
        welded_text=welded_text,
        m_bridge=m_bridge,
        d_geodesic=best_components[0],
        d_causal=best_components[1],
        d_3phase=best_components[2],
        forward_waves=fwd_waves,
        backward_waves=bwd_waves,
        collision_matrix=action_matrix,
    )
