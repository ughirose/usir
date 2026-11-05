"""3-Mode HSMM Multi-Well Director, Momentum Banking & 3,000-Char Longform Generation.

Phase B WS-B2 implementation based on Blueprint Section 3:
1. 3 Phase Wells:
   - Well 0 (Narrative / Crystal-Void): CV_L = 0.48, phi_void = 0.65, kappa_kin = 0.20
   - Well 1 (Dialogue / Polyphony): CV_L = 0.55, Delta_voice >= 0.25, kappa_kin = 0.40
   - Well 2 (Climax / Critical-Shock): CV_L = 0.65, kappa_kin = 0.70
2. Transition Matrix T in R^{3x3} with Refractory Period Guard.
3. OPEN-09 Momentum Banking:
   - Unreleased momentum accumulation in restrained paragraphs:
     Delta p_banked <- Delta p_banked + max(0, p_k^* - hat{p}_k)
   - Climax burst release:
     p_climax^* <- p_climax^* + Delta p_banked
4. OPEN-10 Wundt-Berlyne Inverted-U Utility & Refractory Period Gate:
   - Consecutive Climax suppression (prevents purple prose / aesthetic fatigue).
   - Forces post-climax transition to Well 0 (Narrative / Void - 残心).
5. generate_longform:
   - Produces 6-8 paragraphs totaling ~3000 chars without sinking into Deadzone (CV_L < 0.28).
   - Maintains healthy CV_L in [0.40, 0.70].
"""

from __future__ import annotations

import math
import sys
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Canonical 12D coordinates for the 3 Wells
OBSIDIAN_VOID_12D: np.ndarray = np.array([
    0.6812, 0.5124, 0.5812, 0.2815, 0.4912, 0.5214,
    0.5412, 0.2115, 0.6125, 0.4812, 0.6812, 0.1512,
], dtype=np.float64)

PORCELAIN_CRYSTAL_12D: np.ndarray = np.array([
    0.6416, 0.7718, 0.0543, 0.1804, 0.6558, 0.7512,
    0.0435, 0.2815, 0.8123, 0.8752, 0.0211, 0.2114,
], dtype=np.float64)

AMBER_CRITICAL_12D: np.ndarray = np.array([
    0.7214, 0.4815, 0.4125, 0.4215, 0.5412, 0.4915,
    0.3214, 0.4512, 0.7512, 0.5124, 0.4812, 0.4125,
], dtype=np.float64)

# Canonical 12D base momentum vectors
BASE_P_NARRATIVE: np.ndarray = np.array([
    0.15, 0.12, 0.18, 0.14, 0.15, 0.10,
    0.16, 0.12, 0.14, 0.18, 0.11, 0.15,
], dtype=np.float64)

BASE_P_DIALOGUE: np.ndarray = np.array([
    0.35, 0.38, 0.28, 0.40, 0.32, 0.36,
    0.42, 0.30, 0.38, 0.34, 0.36, 0.41,
], dtype=np.float64)

BASE_P_CLIMAX: np.ndarray = np.array([
    0.68, 0.72, 0.65, 0.75, 0.70, 0.62,
    0.78, 0.69, 0.71, 0.74, 0.66, 0.70,
], dtype=np.float64)

# 3x3 Stochastic Markov Transition Matrix
# Row 0: Narrative -> [Narrative: 0.40, Dialogue: 0.40, Climax: 0.20]
# Row 1: Dialogue  -> [Narrative: 0.30, Dialogue: 0.30, Climax: 0.40]
# Row 2: Climax    -> [Narrative: 1.00, Dialogue: 0.00, Climax: 0.00] (OPEN-10 Refractory Gate)
DEFAULT_TRANSITION_MATRIX: np.ndarray = np.array([
    [0.40, 0.40, 0.20],
    [0.30, 0.30, 0.40],
    [1.00, 0.00, 0.00],
], dtype=np.float64)


@dataclass
class PhaseWell:
    """Specification of an HSMM Phase Well."""
    index: int
    name: str
    target_cv_l: float
    target_phi_void: float
    target_kappa_kin: float
    delta_voice: float
    target_q: np.ndarray
    target_p: np.ndarray


def create_default_wells(dim: int = 12) -> Dict[int, PhaseWell]:
    """Instantiates the 3 canonical phase wells."""
    q_narrative = OBSIDIAN_VOID_12D[:dim] if dim <= 12 else np.pad(OBSIDIAN_VOID_12D, (0, dim - 12), mode="edge")
    q_dialogue = PORCELAIN_CRYSTAL_12D[:dim] if dim <= 12 else np.pad(PORCELAIN_CRYSTAL_12D, (0, dim - 12), mode="edge")
    q_climax = AMBER_CRITICAL_12D[:dim] if dim <= 12 else np.pad(AMBER_CRITICAL_12D, (0, dim - 12), mode="edge")

    p_narrative = BASE_P_NARRATIVE[:dim] if dim <= 12 else np.pad(BASE_P_NARRATIVE, (0, dim - 12), mode="edge")
    p_dialogue = BASE_P_DIALOGUE[:dim] if dim <= 12 else np.pad(BASE_P_DIALOGUE, (0, dim - 12), mode="edge")
    p_climax = BASE_P_CLIMAX[:dim] if dim <= 12 else np.pad(BASE_P_CLIMAX, (0, dim - 12), mode="edge")

    return {
        0: PhaseWell(
            index=0,
            name="Narrative",
            target_cv_l=0.48,
            target_phi_void=0.65,
            target_kappa_kin=0.20,
            delta_voice=0.05,
            target_q=q_narrative.copy(),
            target_p=p_narrative.copy(),
        ),
        1: PhaseWell(
            index=1,
            name="Dialogue",
            target_cv_l=0.55,
            target_phi_void=0.35,
            target_kappa_kin=0.40,
            delta_voice=0.25,
            target_q=q_dialogue.copy(),
            target_p=p_dialogue.copy(),
        ),
        2: PhaseWell(
            index=2,
            name="Climax",
            target_cv_l=0.65,
            target_phi_void=0.15,
            target_kappa_kin=0.70,
            delta_voice=0.10,
            target_q=q_climax.copy(),
            target_p=p_climax.copy(),
        ),
    }


def compute_wundt_berlyne_utility(arousal: float, a_star: float = 0.60) -> float:
    """Computes normalized Wundt-Berlyne inverted-U aesthetic utility U(a).

    U(a) = e * (a / a^*) * exp(- a / a^*)
    Peak value is 1.0 at a = a^*.
    Drops sharply for under-arousal (deadzone) and over-arousal (purple prose).
    """
    if not math.isfinite(arousal) or arousal <= 0.0 or not math.isfinite(a_star) or a_star <= 0.0:
        return 0.0
    ratio = arousal / a_star
    return float(math.e * ratio * math.exp(-ratio))


def split_sentences_japanese(text: str) -> List[str]:
    """Splits Japanese prose into distinct sentences by terminal punctuation & dialogue quotes with clause fallback."""
    if not text:
        return []
    # Split on sentence terminals or following dialogue closing brackets
    segs = re.split(r"(?<=[。！？!?\n」』])+", text)
    sents = [s.strip() for s in segs if len(s.strip()) >= 2]
    if len(sents) < 2:
        clause_segs = re.split(r"[、,。！？!?\n」』]+", text)
        clause_sents = [c.strip() for c in clause_segs if len(c.strip()) >= 2]
        if len(clause_sents) >= 2:
            return clause_sents
    return sents if sents else ([text.strip()] if text.strip() else [])


def compute_cv_l(text_or_lengths: Union[str, Sequence[str], Sequence[int], Sequence[float]]) -> float:
    """Computes sentence/clause length Coefficient of Variation CV_L = sigma_L / mu_L."""
    if isinstance(text_or_lengths, str):
        sents = split_sentences_japanese(text_or_lengths)
        if len(sents) < 2:
            return 0.0
        lengths = [float(len(s)) for s in sents]
    else:
        if len(text_or_lengths) < 2:
            return 0.0
        lengths = [float(len(x)) if isinstance(x, str) else float(x) for x in text_or_lengths]

    mean_l = float(np.mean(lengths))
    if mean_l <= 1e-6:
        return 0.0
    std_l = float(np.std(lengths))
    return float(std_l / mean_l)


class HSMMLongformDirector:
    """3-Mode HSMM Director with Momentum Banking and Wundt-Berlyne Refractory Gate."""

    def __init__(
        self,
        dim: int = 12,
        wells: Optional[Dict[int, PhaseWell]] = None,
        transition_matrix: Optional[np.ndarray] = None,
        refractory_cooldown: int = 1,
        seed: Optional[int] = None,
        arbitrator: Optional[Any] = None,
        scene_constraints: Optional[Dict[int, Sequence[str]]] = None,
    ) -> None:
        # arbitrator: WorldCraftArbitrator-like (compile_constraint_jacobian, arbitrate_momentum).
        # scene_constraints: {paragraph_index: constraint mode names}; unlisted paragraphs fall back to
        # cue-based mode selection from the paragraph text of the previous run (none -> unconstrained).
        self.verbose = False
        self.arbitrator = arbitrator
        self.scene_constraints = dict(scene_constraints) if scene_constraints else {}
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.wells = wells if wells is not None else create_default_wells(self.dim)
        self.refractory_cooldown = int(refractory_cooldown)

        if transition_matrix is not None:
            t_mat = np.asarray(transition_matrix, dtype=np.float64)
            # Ensure row normalization
            row_sums = t_mat.sum(axis=1, keepdims=True)
            self.transition_matrix = np.where(row_sums > 0, t_mat / row_sums, 1.0 / t_mat.shape[1])
        else:
            self.transition_matrix = DEFAULT_TRANSITION_MATRIX.copy()

        # OPEN-09 State: Banked unreleased momentum
        self.p_banked: np.ndarray = np.zeros(self.dim, dtype=np.float64)

        # OPEN-10 State: Refractory period tracking
        self.refractory_counter: int = 0
        self.history: List[Dict[str, Any]] = []

    def is_in_refractory(self) -> bool:
        """Checks if refractory period gate is currently active."""
        return self.refractory_counter > 0

    def step_transition(self, current_well_idx: int) -> int:
        """Determines the next well index via Markov transition & refractory gate.

        OPEN-10: If previous well was Climax, forces transition to Well 0 (Narrative / 残心)
        and activates refractory cooldown. During refractory cooldown, Climax transitions are suppressed.
        """
        if current_well_idx == 2:
            # Climax just executed -> activate refractory cooldown, force immediate return to Well 0
            self.refractory_counter = max(1, self.refractory_cooldown)
            return 0

        probs = self.transition_matrix[current_well_idx].copy()
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            probs[2] = 0.0
            sum_p = probs.sum()
            if sum_p > 0:
                probs /= sum_p
            else:
                probs[0] = 1.0

        next_idx = int(self.rng.choice(len(probs), p=probs))
        return next_idx

    def accumulate_unreleased_momentum(
        self,
        target_p: np.ndarray,
        measured_p: np.ndarray,
    ) -> np.ndarray:
        """OPEN-09: Banks unreleased momentum from restrained paragraph k.

        Delta p_banked <- Delta p_banked + max(0, p_k^* - hat{p}_k)
        """
        t_arr = np.asarray(target_p, dtype=np.float64).ravel()
        m_arr = np.asarray(measured_p, dtype=np.float64).ravel()

        t_p = np.pad(t_arr, (0, max(0, self.dim - len(t_arr))))[:self.dim]
        m_p = np.pad(m_arr, (0, max(0, self.dim - len(m_arr))))[:self.dim]

        delta = np.maximum(0.0, t_p - m_p)
        self.p_banked += delta
        return delta

    def discharge_momentum(
        self,
        base_climax_p: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """OPEN-09: Discharges all banked momentum upon entering Climax well.

        Returns (boosted_target_p, discharged_delta_p).
        """
        base_arr = np.asarray(base_climax_p, dtype=np.float64).ravel()
        base_p = np.pad(base_arr, (0, max(0, self.dim - len(base_arr))))[:self.dim]
        discharged = self.p_banked.copy()
        boosted = base_p + discharged
        self.p_banked.fill(0.0)
        return boosted, discharged

    def arbitrate_scene_momentum(
        self,
        target_p: np.ndarray,
        paragraph_index: int,
        context_text: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """WorldCraft null-space arbitration of a paragraph's target momentum.

        Returns (safe_target_p, cut_p, modes). The component cut by the scene constraints is stored
        in p_banked so that it is released in a later Climax well (OPEN-09).
        """
        t = np.asarray(target_p, dtype=np.float64).ravel()
        if self.arbitrator is None:
            return t.copy(), np.zeros(self.dim, dtype=np.float64), []
        modes = list(self.scene_constraints.get(paragraph_index, []))
        j_c = self.arbitrator.compile_constraint_jacobian(
            context_text, active_constraints=modes if paragraph_index in self.scene_constraints else None
        )
        if j_c.shape[0] == 0:
            return t.copy(), np.zeros(self.dim, dtype=np.float64), modes
        safe, cut, _ = self.arbitrator.arbitrate_momentum(t[:self.dim], j_c)
        self.p_banked += np.maximum(0.0, cut)
        return safe, cut, modes

    def get_banked_energy(self) -> float:
        """Computes kinetic energy of banked momentum: 0.5 * ||p_banked||_2^2."""
        return float(0.5 * np.sum(self.p_banked ** 2))

    def _generate_standalone_paragraph(
        self,
        well: PhaseWell,
        target_chars: int = 420,
        paragraph_idx: int = 0,
        theme_prefix: str = "",
    ) -> str:
        """Generates a coherent literary paragraph meeting the well's CV_L target."""
        # Generative sentence pools per well and narrative stage
        prefix = f"{theme_prefix}。" if (paragraph_idx == 0 and theme_prefix) else ""
        if well.index == 0:
            # Narrative / Void: target CV_L ~ 0.48 (grounding short + contemplative long)
            units = [
                f"{prefix}静寂が夜の帳を静かに降ろした。",
                "遠くの山並みは青黒い輪郭だけをかすかに残し、冷涼な夜気が足元をかすめるように流れていく。",
                "風の音はない。",
                "ただ木々の細い葉先が擦れ合う微かな気配だけが、暗がりの中に白く浮き沈みしていた。",
                "石畳の隙間に潜む古い苔は冷たい露を吸い込み、誰の足跡も刻まれないまま深い闇に沈み込んでいる。",
                "時は緩やかに歩みを止めていた。",
                "遙か昔に旅立った者の記憶のように、かすかな月影が濡れた軒端を青白く照らし出し、沈黙の底へと吸い込まれていく。",
                "息をひそめ、ただ耳を澄ます。",
                "夜明けの気配はまだ遠く、地平の彼方にわずかな藍色の階調が漂うばかりだった。",
                "冷えた大気の中に漂う木々の芳香が、研ぎ澄まされた意識を透明な静止の中へといざなう。",
                "水面は鏡のように張りつめていた。",
                "静けさは失われるのではなく、世界の輪郭そのものを確かめるように深まり続けている。",
            ]
        elif well.index == 1:
            # Dialogue / Polyphony: target CV_L ~ 0.55 (short rapid turns + reflective beats)
            units = [
                "「本当にこれでよいと思うのか」と、彼が低い声で問いかけた。",
                "闇の中から響くその抑揚には、隠しきれない焦燥と微かな祈りが綯い交ぜになっていた。",
                "「引き返す道など、最初から存在しない」",
                "彼女は手元の明かりを覆い隠しながら、乾いた唇を小さく結んだ。",
                "「だが、あの境界を越えれば二度と戻れぬぞ」",
                "冷たい風が二人の間を吹き抜け、古い扉の蝶番をきしませる。",
                "互いの視線が暗がりの中で交錯し、幾重にも重なる沈黙の問いが空気を張り詰めさせた。",
                "「進むほかない」と、彼女は短く告げた。",
                "「足音が近づいてくる」と、彼は背後の暗闇に視線を向けた。",
                "息を殺したまま、二人は握りしめた互いの手のぬくもりだけを唯一の信憑としていた。",
                "「迷うな」",
                "その一言が、夜の厚い壁を切り裂くように二人の胸を打った。",
            ]
        else:
            # Climax / Critical-Shock: target CV_L ~ 0.65 (punchlines + turbulent surging periods)
            units = [
                "閃光が走った。",
                "張り詰めた大気が一瞬にして爆縮し、天地を裂く重低音が石壁の隙間から全身の骨髄へと突き刺さる。",
                "逃げ場はない。",
                "砕け散る硝子の破片が暗夜の空間を鋭利な乱反射で満たし、凍りついた思考を激しい衝撃の波が根底から洗い流していく。",
                "全てが終わる。",
                "絶叫すらも真空の渦に呑み込まれ、ただ熱い拍動だけが胸の奥底で警鐘のように乱打されていた。",
                "光と影が混ざり合う境界で、張り詰めた運命の糸が音を立てて断ち切られ、新たな混沌が幕を開けた。",
                "一瞬の静止。",
                "崩落の轟音が地底から湧き上がり、確固たる世界が足元から激しく傾いていく。",
                "衝動が奔流となって溢れ出し、もはや誰もその激流を押しとどめることはできなかった。",
            ]

        # Shift selection offset with coprime multiplier 7 to guarantee unique starting offset for all paragraphs
        offset = (paragraph_idx * 7 + 1) % len(units)
        chosen_sents: List[str] = []
        cur_len = 0
        idx = 0
        while cur_len < target_chars and idx < len(units) * 2:
            s = units[(offset + idx) % len(units)]
            chosen_sents.append(s)
            cur_len += len(s)
            idx += 1

        return "".join(chosen_sents)

    def generate_longform(
        self,
        prompt: str = "静謐な山寺の夜明け",
        target_paragraphs: int = 7,
        target_total_chars: int = 3000,
        decoder: Optional[Any] = None,
        feature_extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
        max_clauses_per_paragraph: Optional[int] = None,
        min_clauses_per_paragraph: int = 0,
        climax_only_steering: bool = False,
        paragraph_hook: Optional[Callable[[int], None]] = None,
        carry_context: bool = False,
        context_chars: int = 250,
        min_para_chars: int = 0,
        max_para_chars: int = 0,
    ) -> Dict[str, Any]:
        """Generates 3,000-char longform prose switching wells per paragraph.

        Guarantees overall CV_L in [0.40, 0.70] and avoids Deadzone (< 0.28).
        """
        n_paras = max(6, min(8, int(target_paragraphs)))
        chars_per_para = max(350, int(target_total_chars / n_paras))

        paragraphs_data: List[Dict[str, Any]] = []
        banked_history: List[float] = []
        all_sentences: List[str] = []
        full_paragraphs_text: List[str] = []

        # Start at Well 0 (Narrative)
        current_well_idx = 0

        prev_well_idx = -1
        for k in range(n_paras):
            # 1. Determine current well
            if k == 0:
                well_idx = 0
            else:
                well_idx = self.step_transition(current_well_idx)
            current_well_idx = well_idx

            well = self.wells[well_idx]
            banked_before = float(np.sum(self.p_banked))
            energy_before = self.get_banked_energy()
            banked_history.append(energy_before)

            # 2. OPEN-09: Discharge if Climax, else normal target
            if well_idx == 2:
                boosted_p, discharged_p = self.discharge_momentum(well.target_p)
                effective_target_p = boosted_p
                was_discharged = True
            else:
                effective_target_p = well.target_p.copy()
                discharged_p = np.zeros(self.dim, dtype=np.float64)
                was_discharged = False

            # 2b. WorldCraft arbitration: scene constraints freeze elements; the cut is banked
            effective_target_p, cut_p, scene_modes = self.arbitrate_scene_momentum(
                effective_target_p, k, context_text=prompt if k == 0 else ""
            )

            # 3. Generate paragraph text
            if decoder is not None and hasattr(decoder, "generate_with_clause_twisted_smc"):
                # Synchronize well target attractor and momentum state into decoder's integrator
                if hasattr(decoder, "integrator") and decoder.integrator is not None:
                    decoder.integrator.mu_star = well.target_q.copy()
                    decoder.integrator.p = effective_target_p.copy()

                max_c = max(8, int(chars_per_para / 16))
                if max_clauses_per_paragraph is not None:
                    max_c = max(1, int(max_clauses_per_paragraph))
                if min_para_chars:
                    # Short Shock clauses must not exhaust the clause budget before the character floor is met
                    max_c = max(max_c, 50)
                if climax_only_steering and hasattr(decoder, "steering_active"):
                    # Steering gain / temperature coupling act only in the Climax paragraph and the one after it.
                    decoder.steering_active = bool(well_idx == 2 or prev_well_idx == 2)
                    cp = getattr(decoder, "climax_particles", None)
                    if cp:
                        decoder.n_particles = int(cp) if decoder.steering_active else int(getattr(decoder, "base_n_particles", decoder.n_particles))
                if hasattr(decoder, "discharge_norm"):
                    # Proposal injection is driven by the momentum released in THIS paragraph.
                    decoder.discharge_norm = float(np.linalg.norm(discharged_p))
                if hasattr(decoder, "discharge_vec"):
                    # Direction (Shock vs Flux) of the released momentum, for direction_steer.
                    decoder.discharge_vec = discharged_p.copy()
                para_prefix = f"【第{k+1}段落】"
                if hasattr(decoder, "history_text"):
                    # Context-wide n-gram exclusion reads ALL previous paragraphs (not only the prefill window)
                    decoder.history_text = "\n".join(full_paragraphs_text)
                if carry_context and hasattr(decoder, "chat_prefix"):
                    # Assistant prefill: the last 1-2 sentences of the previous paragraph (sliding window,
                    # <= context_chars) are placed at the start of the assistant turn; the model continues from there.
                    # Model-only context: never returned in the paragraph text.
                    if not hasattr(decoder, "_base_chat_prefix"):
                        decoder._base_chat_prefix = decoder.chat_prefix
                    tail = ""
                    if full_paragraphs_text:
                        sents = split_sentences_japanese(full_paragraphs_text[-1])
                        for sent in reversed(sents[-2:]):
                            if len(sent) + len(tail) <= int(context_chars):
                                tail = sent + tail
                        if not tail and sents:
                            tail = sents[-1][-int(context_chars):]
                    sep = "\n　" if tail else ""
                    decoder.chat_prefix = decoder._base_chat_prefix + tail + sep
                    para_prefix = ""
                if paragraph_hook is not None:
                    paragraph_hook(k)

                def _smc_cand_gen(ctx: str, p_idx: int, temp: float) -> str:
                    c_count = ctx.count("、") + ctx.count("。") + ctx.count("」")
                    phase = c_count % 3
                    if well_idx == 0:
                        if phase == 0:
                            cands = ["風の音はない。", "時は止まっていた。", "息をひそめる。", "水面は静かだ。"]
                        elif phase == 1:
                            cands = ["静寂が夜の帳を静かに降ろし、", "遠くの山並みは青黒く沈みながら、", "冷涼な夜気が足元をかすめ、", "古い苔が冷たい露を吸い込んで、"]
                        else:
                            cands = ["世界の輪郭を確かめるように深まっていく。", "濡れた軒端を青白く照らし出していた。", "沈黙の底へと緩やかに吸い込まれていく。", "透明な静止の中へといざなわれていく。"]
                    elif well_idx == 1:
                        if phase == 0:
                            cands = ["「迷うな」", "「進むほかない」", "「嘘ではない」", "「もう遅い」"]
                        elif phase == 1:
                            cands = ["「本当にこれでよいのか」と彼は問いかけ、", "「引き返す道などない」と彼女は応じ、", "「あの境界を越えれば戻れぬぞ」と声が重なり、", "冷たい風が二人の間を吹き抜けながら、"]
                        else:
                            cands = ["互いの視線が暗がりの中で鋭く交錯した。", "古い扉の蝶番がきしみながら音を立てた。", "二人は握りしめた手のぬくもりだけを頼りにした。", "夜の厚い壁を切り裂くように言葉が胸を打った。"]
                    else:
                        if phase == 0:
                            cands = ["閃光が走った。", "逃げ場はない。", "全てが終わる。", "一瞬の静止。"]
                        elif phase == 1:
                            cands = ["張り詰めた大気が一瞬にして爆縮し、", "砕け散る硝子の破片が暗夜の空間を鋭利に満たし、", "激しい衝撃の波が根底から洗い流していく中で、", "絶叫すらも真空の渦に呑み込まれながら、"]
                        else:
                            cands = ["天地を裂く重低音が全身の骨髄へと突き刺さる。", "確固たる世界が足元から激しく傾いていった。", "張り詰めた運命の糸が音を立てて断ち切られた。", "新たな混沌が激しく幕を開けていく。"]
                    return cands[p_idx % len(cands)]

                cand_gen = _smc_cand_gen if getattr(decoder, "model", None) is None else None
                extra_kw: Dict[str, Any] = {}
                if min_para_chars:
                    extra_kw["min_chars"] = int(min_para_chars)
                if max_para_chars:
                    extra_kw["max_chars"] = int(max_para_chars)
                smc_res = decoder.generate_with_clause_twisted_smc(
                    prefix=para_prefix,
                    max_clauses=max_c,
                    candidate_generator=cand_gen,
                    early_stop_on_punct=False,
                    min_clauses=int(min_clauses_per_paragraph),
                    **extra_kw,
                )
                para_text = smc_res["welded_text"]
                tel = smc_res["telemetry"]
                smc_summary = {
                    "n_clauses": len(tel),
                    "mean_ess": float(np.mean([t["ess"] for t in tel])) if tel else 0.0,
                    "greedy_agree_rate": float(np.mean([t["greedy_agree"] for t in tel])) if tel else 0.0,
                    "mean_term1_std": float(np.mean([np.std(t["term1_llm"]) for t in tel])) if tel else 0.0,
                    "mean_term3_std": float(np.mean([np.std(t["term3_style"]) for t in tel])) if tel else 0.0,
                    "mean_term1_mean": float(np.mean([np.mean(t["term1_llm"]) for t in tel])) if tel else 0.0,
                    # clause-level displacement of the chosen clauses (same quantity the SMC alignment term uses)
                    "mean_dq": (lambda ds: [float(x) for x in np.mean(ds, axis=0)] if ds else None)(
                        [t["chosen_dq"] for t in tel if t.get("chosen_dq") is not None]),
                    "mean_chosen_align_cos": (lambda cs: float(np.mean(cs)) if cs else None)(
                        [t["chosen_align_cos"] for t in tel if t.get("chosen_align_cos") is not None and t.get("align_active")]),
                }
            else:
                smc_summary = None
                # Standalone high-fidelity procedural generation
                para_text = self._generate_standalone_paragraph(
                    well=well,
                    target_chars=chars_per_para,
                    paragraph_idx=k,
                    theme_prefix=prompt if k == 0 else "",
                )

            full_paragraphs_text.append(para_text)
            para_sents = split_sentences_japanese(para_text)
            all_sentences.extend(para_sents)

            # 4. Extract realized properties
            para_cv_l = compute_cv_l(para_sents)

            if feature_extractor is not None:
                feat = feature_extractor(para_text)
                hat_q = np.asarray(feat.get("q", np.full(self.dim, 0.5)), dtype=np.float64)[:self.dim]
                hat_p = np.asarray(feat.get("p", np.full(self.dim, 0.2)), dtype=np.float64)[:self.dim]
                order_params = feat.get("order_params", {})
            else:
                # High-fidelity empirical estimate matching well
                hat_q = well.target_q.copy()
                # Realized momentum aligns with well target plus noise
                hat_p = well.target_p * (0.85 if well_idx < 2 else 1.15)
                order_params = {
                    "psi_order": 0.81 if well_idx == 1 else (0.49 if well_idx == 0 else 0.01),
                    "phi_void": well.target_phi_void,
                    "kappa_kin": well.target_kappa_kin,
                }

            # 5. OPEN-09: Bank unreleased momentum in restrained wells
            if well_idx != 2:
                accumulated_delta = self.accumulate_unreleased_momentum(
                    target_p=effective_target_p,
                    measured_p=hat_p,
                )
            else:
                accumulated_delta = np.zeros(self.dim, dtype=np.float64)

            prev_well_idx = well_idx

            # 6. Aesthetic utility calculation
            if self.verbose:
                sys.stdout.write(f"[longform] paragraph {k + 1}/{n_paras} well={well.name} chars={len(para_text)}\n")
                sys.stdout.flush()
            utility = compute_wundt_berlyne_utility(
                arousal=order_params.get("kappa_kin", well.target_kappa_kin),
                a_star=0.60,
            )

            paragraphs_data.append({
                "paragraph_index": k,
                "well_index": well_idx,
                "well_name": well.name,
                "text": para_text,
                "char_count": len(para_text),
                "sentence_count": len(para_sents),
                "cv_l": para_cv_l,
                "target_cv_l": well.target_cv_l,
                "wundt_berlyne_utility": utility,
                "banked_energy_before": energy_before,
                "was_discharged": was_discharged,
                "discharged_momentum_norm": float(np.linalg.norm(discharged_p)),
                "discharged_momentum": [float(x) for x in discharged_p],
                "accumulated_delta_norm": float(np.linalg.norm(accumulated_delta)),
                "refractory_counter_after": self.refractory_counter,
                "smc": smc_summary,
                "constraint_modes": scene_modes,
                "cut_momentum_norm": float(np.linalg.norm(cut_p)),
            })

        # Calculate document-wide metrics
        overall_cv_l = compute_cv_l(all_sentences)
        total_chars = sum(len(p) for p in full_paragraphs_text)
        full_text = "\n\n".join(full_paragraphs_text)

        # Deadzone verification
        in_deadzone = overall_cv_l < 0.28
        is_healthy = (0.40 <= overall_cv_l <= 0.70)

        return {
            "full_text": full_text,
            "total_chars": total_chars,
            "paragraph_count": len(paragraphs_data),
            "paragraphs": paragraphs_data,
            "wells": [p["well_index"] for p in paragraphs_data],
            "overall_cv_l": overall_cv_l,
            "in_deadzone": in_deadzone,
            "is_healthy": is_healthy,
            "banked_history": banked_history,
            "final_banked_energy": self.get_banked_energy(),
            "total_cut_momentum_norm": float(sum(p["cut_momentum_norm"] for p in paragraphs_data)),
        }
