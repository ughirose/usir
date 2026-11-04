"""Nishiki Genre Temperature Unified Field, WorldCraft Null-Space Arbitration, and Cross-Modal Functors.

Phase E Implementation (WS-E1):
1. OPEN-06: Toulmin Logic Audit Potential U_Toulmin(x):
   - Claim (assertive ending), Data (numbers/evidence/citations), Warrant (causal connectors) triad.
   - Max dependency depth penalty (D_max > 6).
   - Subject-predicate twist penalty (主述ねじれ).
2. Genre Temperature tau_genre in [0, 1] Unified Field Energy:
   - F_total(x) = (1 - tau_genre) * U_Toulmin(x) + tau_genre * F_USIR(x)
   - Continuous control: Specification (tau=0.1), Article/Note (tau=0.5), Literary Fiction (tau=0.9).
3. OPEN-09: WorldCraft Ontology Constraints and Lexicographic Null-Space Arbitration:
   - Tier 1 Hard Mask: Naming matrix, deceased character actions, unreleased knowledge.
   - Constraint Jacobian J_C in R^{k x 12} and Null-Space Projector:
     P_null = I_12 - J_C^T (J_C J_C^T + eps * I)^{-1} J_C
   - Safe momentum Delta q_safe = P_null * Delta q* (zero worldview infringement: ||J_C Delta q_safe|| < 1e-5).
   - Orthogonal cut momentum Delta q_cut banked in MomentumBank.
4. Universal Cross-Modal Visual and Music Functors:
   - F_visual: 2D image features (Gabor, color entropy, chiaroscuro, quadtree depth, saliency) <-> 24D <-> Diffusion.
   - F_music: Note/chord sequences (Circle of Fifths torus distance, dissonance, syncopation, rest ratio) <-> 24D <-> WebAudio/MIDI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from usir_py.drivers.syntax_driver import SyntaxDriver
from usir_py.extractor_24d import extract_usir_24d
from usir_py.manifold_geodesic import compute_three_phase_qda_distance


# ==============================================================================
# 1. OPEN-06: Toulmin Logic Audit Potential U_Toulmin(x)
# ==============================================================================

# Claim (主張文末) assertive suffixes and copulas
ASSERTIVE_CLAIM_PATTERNS = [
    r'(?:である|だ|べきである|べきだ|となる|を示す|と結論づけられる|が成立する|と確認された|が必要である|とする|に違いない|と考えられる|と推定される|と判定される|が示された|と立証された|を意味する|と規定する|実証された|立証された|確認された|証明された|成立する|からである|ためである|となるのである)[。！？!?]?$',
    r'(?:必須である|自明である|証明される|反証される|成立する|確認された|定まる|結論される)[。！？!?]?$',
]

# Vague, evasive, or trailing non-claim patterns (penalized in logic mode)
VAGUE_ENDING_PATTERNS = [
    r'(?:かも|かもしれない|ような気がする|気がした|思われる|感じられた|のではないか|のではないだろうか|など|…|、、|とか)[。！？!?]?$',
]

# Data (数値/根拠/引用) indicators
DATA_NUMERIC_RE = re.compile(r'\d+(?:\.\d+)?(?:%|px|ms|dB|Hz|B|KB|MB|GB|次元|個|件|名|万|億|割|倍)?')
DATA_KEYWORDS = [
    "データ", "数値", "実測", "実験", "根拠", "引用", "統計", "事実", "結果", "観察",
    "文献", "図", "表", "計測", "検証", "サンプル", "定理", "公理", "定義", "式", "ログ",
    "基準", "測度", "記録", "指標", "スコア", "標本", "測定値"
]
CITATION_PATTERNS = re.compile(r'\[\d+\]|第\d+(?:節|章|条|項)|表\d+|図\d+|式\s*\(\d+\)|参考文献|出典')

# Warrant (因果接続詞 / 論理推論ブリッジ)
WARRANT_CONNECTORS = [
    "なぜなら", "したがって", "ゆえに", "よって", "それゆえ", "これにより", "であるから",
    "のため", "理由として", "根拠として", "結果として", "ゆえ", "に基づき", "に従い",
    "より導かれる", "からである", "ためである", "ゆえに導かれる", "以上の理由から"
]
DATA_KW_RE = re.compile('|'.join(re.escape(kw) for kw in sorted(DATA_KEYWORDS, key=len, reverse=True)))
WARRANT_RE = re.compile('|'.join(re.escape(w) for w in sorted(WARRANT_CONNECTORS, key=len, reverse=True)))

# Subject-Predicate Twist (主述ねじれ) detection indicators
TWIST_REASON_HEAD_RE = re.compile(r'(?:理由は|原因は|目的は|課題は|方針は|契機は)')
TWIST_REASON_VALID_TAIL_RE = re.compile(r'(?:からだ|ためだ|ことだ|である|にある|にあった|ためである|からである)[。！？!?]?$')
DANGLING_TOPIC_RE = re.compile(r'^[^\s、,]+は[、,\s].+?[^\s、,]+は[、,\s]')

# Backward warrant markers: the marker sentence is the premise, the previous sentence the conclusion.
_BACKWARD_WARRANT_RE = re.compile(r'^(?:なぜなら|というのも|理由として|根拠として)')
# Forward warrant markers: the marker sentence is the conclusion, the previous sentence the premise.
_FORWARD_WARRANT_RE = re.compile(r'^(?:したがって|ゆえに|よって|それゆえ|だから|このため|これにより|以上の理由から|結果として)')
# Function material that carries no new concept (copulas, meta-conclusion phrases, particles).
_NOISE_RE = re.compile(
    r'(?:なぜなら|というのも|したがって|ゆえに|よって|それゆえ|だから|このため|これにより|以上の理由から|結果として|理由として|根拠として'
    r'|と結論づけられる|と結論される|と言える|といえる|と考えられる|と判断される|ことになる|のである|からである|ためである|である|であり|です|だからである|だから|だ)'
)
_PARTICLE_RE = re.compile(r'[はがをにでとのもへやかてで、，,。！？!?\s]')


def _content_bigrams(sentence: str) -> set:
    """Character bigrams of the content material of a sentence (function words removed)."""
    core = _NOISE_RE.sub('', sentence)
    core = _PARTICLE_RE.sub('', core)
    if len(core) < 2:
        return {core} if core else set()
    return {core[i:i + 2] for i in range(len(core) - 1)}


def _is_restatement(premise: str, conclusion: str, jaccard_min: float = 0.70, novelty_max: float = 0.10) -> bool:
    """True when the conclusion repeats the premise: high bigram Jaccard and no new concept.

    novelty = share of the conclusion's bigrams absent from the premise.
    """
    a, b = _content_bigrams(premise), _content_bigrams(conclusion)
    if not a or not b:
        return False
    jac = len(a & b) / len(a | b)
    novelty = len(b - a) / len(b)
    return jac > jaccard_min and novelty <= novelty_max


def _count_tautological_warrants(sents: List[str]) -> int:
    """Counts adjacent (premise, conclusion) pairs joined by a causal marker that restate each other."""
    count = 0
    for i in range(1, len(sents)):
        cur, prev = sents[i], sents[i - 1]
        if _BACKWARD_WARRANT_RE.match(cur):
            prem, conc = cur, prev
        elif _FORWARD_WARRANT_RE.match(cur):
            prem, conc = prev, cur
        else:
            continue
        # Restatement in either direction: conclusion adds nothing beyond the premise, or vice versa.
        if _is_restatement(prem, conc) or _is_restatement(conc, prem):
            count += 1
    return count

@dataclass
class ToulminAuditResult:
    """Detailed audit metrics of the Toulmin logic potential."""
    claim_score: float         # S_claim in [0, 1]
    data_score: float          # S_data in [0, 1]
    warrant_score: float       # S_warrant in [0, 1]
    triad_score: float         # S_triad in [0, 1]
    triad_deficiency: float    # U_triad >= 0
    max_depth_found: int       # Maximum syntactic dependency depth observed
    depth_penalty: float       # Penalty for D_max > 6
    twist_count: int           # Number of subject-predicate twists detected
    twist_penalty: float       # Twist penalty
    u_toulmin: float           # Total potential U_Toulmin(x) >= 0
    tautology_count: int = 0       # Warrant pairs whose premise and conclusion are restatements
    tautology_penalty: float = 0.0  # tautology_count / n_sents


def evaluate_toulmin_components(text: str) -> ToulminAuditResult:
    """Evaluates Toulmin argumentative triad (Claim, Data, Warrant) and syntactic penalties."""
    clean = text.strip()
    if not clean:
        return ToulminAuditResult(
            claim_score=0.0,
            data_score=0.0,
            warrant_score=0.0,
            triad_score=0.0,
            triad_deficiency=3.0,
            max_depth_found=0,
            depth_penalty=0.0,
            twist_count=0,
            twist_penalty=0.0,
            u_toulmin=3.0,
        )

    # Sentence segmentation
    raw_sents = re.split(r'(?<=[。！？!?\n])', clean)
    sents = [s.strip() for s in raw_sents if s.strip()]
    if not sents:
        sents = [clean]
    n_sents = len(sents)

    # 1. Claim evaluation (主張文末)
    claim_hits = 0
    vague_hits = 0
    for s in sents:
        is_assertive = any(re.search(pat, s) for pat in ASSERTIVE_CLAIM_PATTERNS)
        is_vague = any(re.search(pat, s) for pat in VAGUE_ENDING_PATTERNS)
        if is_assertive and not is_vague:
            claim_hits += 1
        elif is_vague:
            vague_hits += 1

    claim_score = max(0.0, min(1.0, (claim_hits - 0.5 * vague_hits) / n_sents))

    # 2. Data evaluation (数値/根拠/引用)
    num_matches = len(DATA_NUMERIC_RE.findall(clean))
    data_kw_matches = len(DATA_KW_RE.findall(clean))
    citation_matches = len(CITATION_PATTERNS.findall(clean))
    raw_data_density = (num_matches * 0.25 + data_kw_matches * 0.20 + citation_matches * 0.35) / max(1, n_sents)
    data_score = max(0.0, min(1.0, raw_data_density))

    # 3. Warrant evaluation (因果接続詞)
    tautology_count = _count_tautological_warrants(sents)
    # A causal marker that only restates the premise is not a warrant.
    warrant_hits = max(0, len(WARRANT_RE.findall(clean)) - tautology_count * 2)
    raw_warrant_density = warrant_hits / max(1, n_sents * 0.5)
    warrant_score = max(0.0, min(1.0, raw_warrant_density))

    triad_score = float((claim_score + data_score + warrant_score) / 3.0)
    triad_deficiency = float(
        (1.0 - claim_score) ** 2 + (1.0 - data_score) ** 2 + (1.0 - warrant_score) ** 2
    )

    # 4. Dependency depth penalty (D_max > 6)
    driver = SyntaxDriver(force_cleanroom=True)
    max_d_overall = 0
    depth_excess_sum = 0.0
    for s in sents:
        frame = driver.parse_sentence(s)
        d_max = frame.max_depth
        if d_max > max_d_overall:
            max_d_overall = d_max
        if d_max > 6:
            depth_excess_sum += float((d_max - 6) ** 2)

    depth_penalty = float(depth_excess_sum / max(1, n_sents))

    # 5. Subject-predicate twist penalty (主述ねじれ)
    twist_count = 0
    for s in sents:
        # Check topic mismatch: starts with 理由は/原因は but tail is not からだ/ためだ/ことだ/である
        if TWIST_REASON_HEAD_RE.search(s) and not TWIST_REASON_VALID_TAIL_RE.search(s):
            twist_count += 1
        # Check dangling topic: double topic marker 'は' without conjunction
        if DANGLING_TOPIC_RE.search(s) and "なら" not in s and "対して" not in s:
            twist_count += 1
        # Check sentence ending with conjunctive particle (unclosed sentence)
        if re.search(r'(?:て|で|が|のに|けれど|ながら)[、,\s]*$', s):
            twist_count += 1

    twist_penalty = float(twist_count / max(1, n_sents))

    # Total Toulmin Potential U_Toulmin(x)
    tautology_penalty = float(tautology_count / max(1, n_sents))
    u_toulmin = float(triad_deficiency + 0.5 * depth_penalty + 1.2 * twist_penalty + 1.5 * tautology_penalty)

    return ToulminAuditResult(
        claim_score=claim_score,
        data_score=data_score,
        warrant_score=warrant_score,
        triad_score=triad_score,
        triad_deficiency=triad_deficiency,
        max_depth_found=max_d_overall,
        depth_penalty=depth_penalty,
        twist_count=twist_count,
        twist_penalty=twist_penalty,
        u_toulmin=u_toulmin,
        tautology_count=tautology_count,
        tautology_penalty=tautology_penalty,
    )


def compute_toulmin_potential(text: str) -> float:
    """Computes U_Toulmin(x) potential in [0, inf)."""
    return evaluate_toulmin_components(text).u_toulmin


# ==============================================================================
# 2. Genre Temperature tau_genre in [0, 1] Unified Field Energy
# ==============================================================================

def compute_usir_style_energy(text: str) -> float:
    """Computes literary style free energy F_USIR(x) using 3-Phase QDA double-well barrier."""
    extracted = extract_usir_24d(text)
    z_24d = extracted["z_24d"]
    # Double-well barrier to literary critical manifold
    # Novel profile is calibrated to the corpus momentum scale (see DOMAIN_QDA_PROFILES)
    d_3phase = compute_three_phase_qda_distance(z_24d)["d_3phase"]
    # Void defect and cadence deadness penalties
    phi_void = float(extracted["order_params"]["phi_void"])
    cadence_alive = float(extracted["cadence_alive"])
    void_defect = float((1.0 - phi_void) ** 2)
    slop_deadness = float((1.0 - cadence_alive) ** 2)
    return float(d_3phase + 1.2 * void_defect + 1.5 * slop_deadness)


def compute_unified_field_energy(
    text: str,
    tau_genre: float = 0.5,
) -> Dict[str, float]:
    r"""Computes total unified field energy:

    F_total(x) = (1 - tau_genre) * U_Toulmin(x) + tau_genre * F_USIR(x)

    Args:
        text: Input text candidate.
        tau_genre: Genre temperature in [0.0, 1.0].
            tau = 0.1: Specification mode (dominated by Toulmin logic, terms unambiguous).
            tau = 0.5: Article / Note mode (balanced argumentative and stylistic rhythm).
            tau = 0.9: Literary Fiction mode (dominated by USIR symplectic trajectory).

    Returns:
        Dict with 'f_total', 'u_toulmin', 'f_usir', 'tau_genre', 'claim_score', 'data_score', 'warrant_score'.
    """
    tau = float(np.clip(tau_genre, 0.0, 1.0))
    toulmin_res = evaluate_toulmin_components(text)
    u_toulmin = toulmin_res.u_toulmin
    f_usir = compute_usir_style_energy(text)

    f_total = float((1.0 - tau) * u_toulmin + tau * f_usir)

    return {
        "f_total": f_total,
        "u_toulmin": u_toulmin,
        "f_usir": f_usir,
        "tau_genre": tau,
        "claim_score": toulmin_res.claim_score,
        "data_score": toulmin_res.data_score,
        "warrant_score": toulmin_res.warrant_score,
        "depth_penalty": toulmin_res.depth_penalty,
        "twist_penalty": toulmin_res.twist_penalty,
    }


# ==============================================================================
# 3. OPEN-09: WorldCraft Ontology Constraints and Null-Space Arbitration
# ==============================================================================

@dataclass
class WorldCraftOntologyRule:
    """Specifies Tier 1 hard constraints on characters, facts, and spoiler boundaries."""
    # Mapping of character ID to valid pronouns / names, e.g. {"Alice": ["私", "アリス"]}
    allowed_naming: Dict[str, List[str]] = field(default_factory=dict)
    # Forbidden pronouns / names for character ID, e.g. {"Alice": ["俺", "妾"]}
    forbidden_naming: Dict[str, List[str]] = field(default_factory=dict)
    # Deceased characters who cannot perform active agent verbs in present tense
    deceased_characters: Set[str] = field(default_factory=set)
    # Unreleased knowledge / spoilers that cannot appear
    unreleased_knowledge: Set[str] = field(default_factory=set)
    # Worldview forbidden terms (e.g., modern tech in medieval fantasy)
    forbidden_terms: Set[str] = field(default_factory=set)


class WorldCraftArbitrator:
    """Tier 1 Hard Masking and Null-Space Projector for WorldCraft Ontology."""

    def __init__(self, rule: Optional[WorldCraftOntologyRule] = None) -> None:
        self.rule = rule if rule is not None else WorldCraftOntologyRule()

    def check_tier1_violations(self, candidate_text: str) -> List[str]:
        """Audits candidate text against Tier 1 hard constraints."""
        violations: List[str] = []

        # 1. Forbidden terms check
        for term in self.rule.forbidden_terms:
            if term in candidate_text:
                violations.append(f"Forbidden world term present: '{term}'")

        # 2. Unreleased knowledge / spoiler check
        for spoiler in self.rule.unreleased_knowledge:
            if spoiler in candidate_text:
                violations.append(f"Unreleased knowledge leak: '{spoiler}'")

        # 3. Forbidden naming check
        for char_id, forbidden_list in self.rule.forbidden_naming.items():
            for forbidden_name in forbidden_list:
                if forbidden_name in candidate_text:
                    violations.append(
                        f"Character '{char_id}' uses forbidden naming/pronoun: '{forbidden_name}'"
                    )

        # 4. Deceased character action check
        for deceased in self.rule.deceased_characters:
            # Check if deceased character acts as active live agent
            # Avoid matching passive voice or burial/death descriptions (e.g. 埋葬された, 見送られた)
            pattern = rf'{re.escape(deceased)}(?:が|は)(?:[^\s。！？]*?)(?:走|言|語|叫|立|笑|歩|向|動|見|来|行|聞|答|問|思|考|入|出)(?:[^\s。！？]*?)(?<!られた)(?<!れた)(?<!される)(?<!された)'
            if re.search(pattern, candidate_text):
                violations.append(f"Deceased character '{deceased}' acts as live agent")

        return violations

    def apply_tier1_hard_mask(self, candidates: Sequence[str]) -> Tuple[List[str], List[bool]]:
        """Applies Tier 1 hard mask. Returns (valid_candidates, mask_flags)."""
        valid_candidates: List[str] = []
        mask_flags: List[bool] = []
        for cand in candidates:
            viols = self.check_tier1_violations(cand)
            if not viols:
                valid_candidates.append(cand)
                mask_flags.append(True)
            else:
                mask_flags.append(False)
        return valid_candidates, mask_flags

    @staticmethod
    def compute_null_space_projector(
        jacobian_c: Optional[np.ndarray],
        eps: float = 1e-6,
    ) -> np.ndarray:
        r"""Computes null-space projection operator for constraint Jacobian J_C in R^{k x 12}:

        P_null = I_12 - J_C^T (J_C J_C^T + eps * I)^{-1} J_C

        Guarantees J_C * P_null \approx 0.
        """
        i_12 = np.eye(12, dtype=np.float64)
        if jacobian_c is None:
            return i_12

        j_c = np.asarray(jacobian_c, dtype=np.float64)
        if j_c.size == 0 or (j_c.ndim == 2 and j_c.shape[0] == 0):
            return i_12

        if j_c.ndim == 1:
            j_c = j_c.reshape(1, -1)

        k, n = j_c.shape
        if n != 12:
            raise ValueError(f"Constraint Jacobian must have 12 columns, got {n}")

        if k == 0:
            return i_12

        # Gram matrix: J_C J_C^T in R^{k x k}
        gram = j_c @ j_c.T + eps * np.eye(k, dtype=np.float64)
        gram_inv = np.linalg.pinv(gram)

        # Null-space projector P_null in R^{12 x 12}
        p_null = i_12 - (j_c.T @ gram_inv @ j_c)
        return p_null

    def arbitrate_momentum(
        self,
        target_delta_q: np.ndarray,
        jacobian_c: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Arbitrates stylistic momentum Delta q* via null space projection.

        Args:
            target_delta_q: Desired 12D style momentum Delta q* in R^12.
            jacobian_c: Constraint Jacobian J_C in R^{k x 12}.
            eps: Regularization parameter.

        Returns:
            Tuple of:
            - delta_q_safe: P_null * Delta q* in R^12 (worldview infringement is 0).
            - delta_q_cut: (I - P_null) * Delta q* in R^12 (orthogonal cut component).
            - p_null: Null-space projector matrix P_null in R^{12 x 12}.
        """
        dq = np.asarray(target_delta_q, dtype=np.float64).ravel()
        if dq.shape[0] != 12:
            raise ValueError(f"target_delta_q must have 12 elements, got {dq.shape[0]}")

        p_null = self.compute_null_space_projector(jacobian_c, eps=eps)
        dq_safe = p_null @ dq
        dq_cut = dq - dq_safe

        return dq_safe, dq_cut, p_null


@dataclass
class MomentumBank:
    """Stores unreleased momentum from constrained paragraphs to discharge in climax."""

    banked_momentum: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.float64)
    )

    def accumulate(self, delta_q_cut: np.ndarray) -> None:
        """Banks unreleased orthogonal cut momentum."""
        dq = np.asarray(delta_q_cut, dtype=np.float64).ravel()
        if dq.shape[0] < 12:
            raise ValueError(f"delta_q_cut must have at least 12 elements, got {dq.shape[0]}")
        self.banked_momentum = self.banked_momentum + dq[:12]

    def discharge(self) -> np.ndarray:
        """Discharges all stored momentum and resets bank to zero."""
        discharged = self.banked_momentum.copy()
        self.banked_momentum = np.zeros(12, dtype=np.float64)
        return discharged

    def get_energy(self) -> float:
        """Returns banked kinetic potential E_bank = 0.5 * ||p_bank||^2."""
        return float(0.5 * np.sum(self.banked_momentum ** 2))


# ==============================================================================
# 4. Universal Cross-Modal Visual and Music Functors (Python Interop)
# ==============================================================================

def visual_functor_forward_24d(visual_features: Dict[str, Any]) -> np.ndarray:
    r"""Maps 2D image features to 24D USIR symplectic state z = (q, p).

    Expected features:
    - gabor_frequencies: list of 4 floats [E_0, E_45, E_90, E_135]
    - color_entropy: float in [0, 1]
    - chiaroscuro_gradient: float in [0, 1]
    - quadtree_depth_entropy: float in [0, 1]
    - saliency_scanpath: list of dicts with 'x', 'y', 'intensity'
    """
    gabor = np.asarray(visual_features.get("gabor_frequencies", [0.25, 0.25, 0.25, 0.25]), dtype=np.float64)
    color_ent = float(visual_features.get("color_entropy", 0.5))
    chiaro = float(visual_features.get("chiaroscuro_gradient", 0.4))
    quad_ent = float(visual_features.get("quadtree_depth_entropy", 0.5))
    scanpath = visual_features.get("saliency_scanpath", [])

    # Saliency path motion dynamics
    if len(scanpath) >= 2:
        dists = [
            math.hypot(scanpath[i]["x"] - scanpath[i - 1]["x"], scanpath[i]["y"] - scanpath[i - 1]["y"])
            for i in range(1, len(scanpath))
        ]
        saccade_mean = float(np.mean(dists))
    else:
        saccade_mean = 0.2

    gabor_var = float(np.var(gabor))

    # 12 Position coordinates q in [0, 1]
    q = np.zeros(12, dtype=np.float64)
    # Group I: Flux (Sg-01, Sg-04, Sg-08) -> spatial frequency dynamics and saccade pace
    q[0] = np.clip(0.5 * saccade_mean + 0.5 * gabor_var * 4.0, 0.0, 1.0)
    q[3] = np.clip(0.4 * gabor[0] + 0.6 * saccade_mean, 0.0, 1.0)
    q[7] = np.clip(0.6 * gabor_var + 0.4 * color_ent, 0.0, 1.0)

    # Group II: Lattice (Sg-02, Sg-05, Sg-09) -> Quadtree depth entropy and structural symmetry
    q[1] = np.clip(quad_ent, 0.0, 1.0)
    q[4] = np.clip(0.5 * (1.0 - gabor_var * 2.0) + 0.5 * quad_ent, 0.0, 1.0)
    q[8] = np.clip(0.7 * (1.0 - color_ent) + 0.3 * quad_ent, 0.0, 1.0)

    # Group III: Void (Sg-03, Sg-06, Sg-10) -> Composition margin & low density
    margin_ratio = float(visual_features.get("composition_margin_ratio", 1.0 - chiaro))
    q[2] = np.clip(margin_ratio, 0.0, 1.0)
    q[5] = np.clip(0.6 * margin_ratio + 0.4 * (1.0 - quad_ent), 0.0, 1.0)
    q[9] = np.clip(0.8 * margin_ratio + 0.2 * (1.0 - chiaro), 0.0, 1.0)

    # Group IV: Shock (Sg-04/07/11/12) -> Chiaroscuro gradient and high-frequency rupture
    q[6] = np.clip(chiaro, 0.0, 1.0)
    q[10] = np.clip(0.7 * chiaro + 0.3 * color_ent, 0.0, 1.0)
    q[11] = np.clip(0.8 * chiaro + 0.2 * gabor_var * 5.0, 0.0, 1.0)

    # 12 Momentum coordinates p >= 0
    p = np.zeros(12, dtype=np.float64)
    p[0] = float(saccade_mean * 1.5)
    p[2] = float(margin_ratio * 0.8)
    p[6] = float(chiaro * 1.8)
    p[11] = float(chiaro * 2.0 + gabor_var)

    return np.concatenate([q, p])


def visual_functor_inverse(z_24d: np.ndarray) -> Dict[str, float]:
    r"""Inverse visual functor F_visual^{-1}: Maps 24D state to Diffusion generative parameters."""
    z = np.asarray(z_24d, dtype=np.float64).ravel()
    q = z[:12]
    p = z[12:24]

    # CFG scale: driven by Lattice stability and clarity, restrained by Void
    cfg_scale = float(np.clip(3.5 + 8.5 * (0.4 * q[1] + 0.4 * q[4] + 0.2 * q[8]) - 2.5 * q[9], 1.5, 14.0))

    # High frequency noise gain: driven by Shock / chiaroscuro and momentum
    high_freq_noise_gain = float(np.clip(0.1 + 0.7 * q[11] + 0.3 * q[6] + 0.15 * math.tanh(p[6] + p[11]), 0.0, 2.0))

    # Composition margin ratio: driven by Void coordinates (Sg-03, Sg-06, Sg-10)
    composition_margin_ratio = float(np.clip(0.1 + 0.3 * q[2] + 0.3 * q[5] + 0.4 * q[9], 0.0, 1.0))

    return {
        "cfg_scale": cfg_scale,
        "high_freq_noise_gain": high_freq_noise_gain,
        "composition_margin_ratio": composition_margin_ratio,
        "color_contrast_temperature": float(np.clip(0.2 + 0.5 * q[10] + 0.3 * q[7], 0.0, 1.0)),
        "edge_sharpness": float(np.clip(0.4 * q[1] + 0.4 * q[6] + 0.2 * (1.0 - q[2]), 0.0, 1.0)),
    }


def music_functor_forward_24d(music_features: Dict[str, Any]) -> np.ndarray:
    r"""Maps note/chord sequence features to 24D USIR symplectic state z = (q, p).

    Expected features:
    - torus_distance: Circle of Fifths torus distance in [0, 1]
    - dissonance: Sensory/interval dissonance in [0, 1]
    - syncopation: Metric syncopation degree in [0, 1]
    - rest_ratio: Void silence ratio in [0, 1]
    - bpm: Base tempo in BPM (e.g. 60 - 180)
    """
    torus_dist = float(music_features.get("torus_distance", 0.3))
    diss = float(music_features.get("dissonance", 0.2))
    synco = float(music_features.get("syncopation", 0.25))
    rest = float(music_features.get("rest_ratio", 0.2))
    bpm = float(music_features.get("bpm", 100.0))

    norm_bpm = float(np.clip((bpm - 60.0) / 120.0, 0.0, 1.0))

    q = np.zeros(12, dtype=np.float64)
    # Group I: Flux -> syncopation and tempo momentum
    q[0] = np.clip(0.6 * norm_bpm + 0.4 * synco, 0.0, 1.0)
    q[3] = np.clip(0.5 * synco + 0.5 * torus_dist, 0.0, 1.0)
    q[7] = np.clip(0.7 * synco + 0.3 * norm_bpm, 0.0, 1.0)

    # Group II: Lattice -> harmonic consonance (1 - dissonance) and key stability
    consonance = 1.0 - diss
    q[1] = np.clip(consonance, 0.0, 1.0)
    q[4] = np.clip(0.6 * consonance + 0.4 * (1.0 - torus_dist), 0.0, 1.0)
    q[8] = np.clip(0.8 * consonance + 0.2 * (1.0 - synco), 0.0, 1.0)

    # Group III: Void -> rest ratio & pause spacing
    q[2] = np.clip(rest, 0.0, 1.0)
    q[5] = np.clip(0.7 * rest + 0.3 * (1.0 - norm_bpm), 0.0, 1.0)
    q[9] = np.clip(0.9 * rest + 0.1 * (1.0 - diss), 0.0, 1.0)

    # Group IV: Shock -> dissonance and Circle of Fifths rupture
    q[6] = np.clip(diss, 0.0, 1.0)
    q[10] = np.clip(0.6 * diss + 0.4 * torus_dist, 0.0, 1.0)
    q[11] = np.clip(0.7 * diss + 0.3 * synco, 0.0, 1.0)

    # Momentum p >= 0
    p = np.zeros(12, dtype=np.float64)
    p[0] = float(norm_bpm * 1.5)
    p[2] = float(rest * 0.9)
    p[6] = float(diss * 1.6)
    p[11] = float(diss * 1.8 + torus_dist)

    return np.concatenate([q, p])


def music_functor_inverse(z_24d: np.ndarray) -> Dict[str, Any]:
    r"""Inverse music functor F_music^{-1}: Maps 24D state to WebAudio/MIDI generation controls."""
    z = np.asarray(z_24d, dtype=np.float64).ravel()
    q = z[:12]
    p = z[12:24]

    # BPM: driven by Flux (q0, q3, q7) and momentum p0
    bpm = float(np.clip(60.0 + 100.0 * (0.5 * q[0] + 0.3 * q[3] + 0.2 * q[7]) + 20.0 * math.tanh(p[0]), 50.0, 190.0))

    # Harmonic tension: driven by Shock (q6, q10, q11) and dissonance
    harmonic_tension = float(np.clip(0.4 * q[6] + 0.3 * q[10] + 0.3 * q[11] - 0.2 * q[1], 0.0, 1.0))

    # Rest interval: driven by Void (q2, q5, q9)
    rest_interval = float(np.clip(0.3 * q[2] + 0.3 * q[5] + 0.4 * q[9], 0.0, 1.0))

    # Reverb decay in seconds: driven by Void depth (obsidian stillness)
    reverb_decay = float(np.clip(0.5 + 4.5 * q[9] + 1.5 * q[2], 0.4, 6.0))
    rubato_depth = float(np.clip(0.1 + 0.5 * q[7] + 0.2 * math.tanh(p[7]), 0.0, 1.0))
    soliton_accent_gain = float(np.clip(0.6 * q[11] + 0.4 * q[6] + 0.3 * math.tanh(p[11]), 0.0, 2.0))

    # WebAudio sequence synthesis
    base_scale = [60, 62, 64, 67, 69]
    web_audio_seq = []
    beat_sec = 60.0 / bpm
    curr_time = 0.0
    for step in range(8):
        scale_idx = step % len(base_scale)
        pitch = base_scale[scale_idx]
        if harmonic_tension > 0.5 and step % 3 == 2:
            pitch += 1
        duration = beat_sec * (1.0 - rest_interval * 0.6)
        gain = float(np.clip(0.4 + 0.3 * soliton_accent_gain, 0.1, 1.0))
        web_audio_seq.append({
            "note": int(pitch),
            "time": float(curr_time),
            "duration": float(duration),
            "gain": float(gain),
        })
        curr_time += beat_sec

    return {
        "bpm": bpm,
        "harmonic_tension": harmonic_tension,
        "rest_interval": rest_interval,
        "reverb_decay": reverb_decay,
        "rubato_depth": rubato_depth,
        "soliton_accent_gain": soliton_accent_gain,
        "web_audio_sequence": web_audio_seq,
    }
