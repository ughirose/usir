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

from usir_py.clause_twist import count_clause_twists

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
    clause_twist_count: int = 0     # Clause-local subject/predicate selectional twists (included in twist_count)


def evaluate_toulmin_components(text: str, clause_twist: bool = True) -> ToulminAuditResult:
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

    clause_twist_count = count_clause_twists(sents) if clause_twist else 0
    twist_count += clause_twist_count
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
        clause_twist_count=clause_twist_count,
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

# WorldCraft constraint modes -> frozen style elements (0-based index = Sg number - 1).
# Each mode contributes unit rows e_i to the constraint Jacobian J_C in R^{k x 12}.
CONSTRAINT_MODE_ELEMENTS: Dict[str, Tuple[int, ...]] = {
    # Fact / evidence-conveying scene: no figurative leap (Sg-08 Lp) and no rupture (Sg-12 Ru).
    "evidence": (7, 11),
    # Hard-logic character speech: keep Lattice regularity (Sg-02 Cr, Sg-06 Sy).
    "logic_voice": (1, 5),
    # Stillness / concealment scene: hold Void (Sg-03 Pr, Sg-07 Cs, Sg-11 Ma).
    "stillness": (2, 6, 10),
}

# Surface cues used only to pick a constraint mode when none is given explicitly.
_CONSTRAINT_MODE_CUES: Dict[str, Tuple[str, ...]] = {
    "evidence": ("証拠", "データ", "事実", "記録", "報告", "検証", "調査", "計測", "実測"),
    "logic_voice": ("論理", "証明", "したがって", "ゆえに", "仮定", "定理", "前提", "結論"),
    "stillness": ("静寂", "沈黙", "息をひそ", "潜", "身を隠", "無音", "闇", "しじま"),
}


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

    def __init__(
        self,
        rule: Optional[WorldCraftOntologyRule] = None,
        projector_method: str = "svd",
    ) -> None:
        if projector_method not in ("tikhonov", "svd"):
            raise ValueError(f"projector_method must be 'tikhonov' or 'svd', got {projector_method!r}")
        self.rule = rule if rule is not None else WorldCraftOntologyRule()
        # "svd" (default): exact projector I - J^+ J via SVD pseudo-inverse (rank-deficient / general rows).
        # "tikhonov": legacy regularised projector (eps-biased by ~eps for unit rows); explicit opt-in.
        self.projector_method = projector_method

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
    def compile_constraint_jacobian(
        context_text: str = "",
        active_constraints: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        r"""Builds the constraint Jacobian J_C in R^{k x 12} for a scene.

        Each active mode freezes a set of style elements (see CONSTRAINT_MODE_ELEMENTS), adding one
        unit row e_i per element, so that J_C P_null = 0 and the frozen elements receive no momentum.
        When active_constraints is None the modes are chosen from surface cues in context_text; when
        no cue matches, J_C has k = 0 rows (unconstrained scene, P_null = I).
        """
        if active_constraints is None:
            modes = [m for m, cues in _CONSTRAINT_MODE_CUES.items() if any(c in context_text for c in cues)]
        else:
            modes = list(active_constraints)
            unknown = [m for m in modes if m not in CONSTRAINT_MODE_ELEMENTS]
            if unknown:
                raise ValueError(f"Unknown constraint mode(s): {unknown}; known: {sorted(CONSTRAINT_MODE_ELEMENTS)}")

        indices = sorted({i for m in modes for i in CONSTRAINT_MODE_ELEMENTS[m]})
        j_c = np.zeros((len(indices), 12), dtype=np.float64)
        for row, idx in enumerate(indices):
            j_c[row, idx] = 1.0
        return j_c

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

    @staticmethod
    def svd_rank_tolerance(k: int, n: int, s_max: float) -> float:
        """Rank tolerance for the SVD pseudo-inverse: max(k, n) * machine_eps * s_max."""
        return float(max(k, n) * np.finfo(np.float64).eps * s_max)

    @staticmethod
    def compute_null_space_projector_svd(
        jacobian_c: Optional[np.ndarray],
        rtol: Optional[float] = None,
    ) -> np.ndarray:
        r"""Exact null-space projector P_null = I_12 - J_C^+ J_C for general J_C in R^{k x 12}.

        J_C = U S V^T (thin SVD); J_C^+ = V S^{-1} U^T over singular values s_i > tol, where
        tol = max(k, 12) * eps_machine * s_max (or rtol * s_max when rtol is given). Singular
        values <= tol are treated as zero, so rank-deficient / linearly dependent rows are allowed
        (k <= 12). k = 0 or all-zero J_C gives P_null = I. No eps bias: J_C P_null = 0 to
        round-off, P_null is symmetric and idempotent.
        """
        i_12 = np.eye(12, dtype=np.float64)
        if jacobian_c is None:
            return i_12
        j_c = np.asarray(jacobian_c, dtype=np.float64)
        if j_c.size == 0:
            return i_12
        if j_c.ndim == 1:
            j_c = j_c.reshape(1, -1)
        k, n = j_c.shape
        if n != 12:
            raise ValueError(f"Constraint Jacobian must have 12 columns, got {n}")
        if k > 12:
            raise ValueError(f"Constraint Jacobian must have k <= 12 rows, got {k}")

        u, s, vt = np.linalg.svd(j_c, full_matrices=False)
        s_max = float(s[0]) if s.size else 0.0
        if s_max == 0.0:
            return i_12
        tol = (
            WorldCraftArbitrator.svd_rank_tolerance(k, n, s_max)
            if rtol is None
            else float(rtol) * s_max
        )
        keep = s > tol
        s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
        j_pinv = (vt.T * s_inv) @ u.T  # J^+ in R^{12 x k}
        return i_12 - j_pinv @ j_c

    @staticmethod
    def build_general_jacobian(rows: Sequence[Sequence[float]]) -> np.ndarray:
        """Builds J_C in R^{k x 12} from arbitrary linear-combination rows (k <= 12)."""
        j_c = np.asarray(rows, dtype=np.float64)
        if j_c.size == 0:
            return np.zeros((0, 12), dtype=np.float64)
        if j_c.ndim != 2 or j_c.shape[1] != 12:
            raise ValueError(f"rows must form a k x 12 matrix, got shape {j_c.shape}")
        if j_c.shape[0] > 12:
            raise ValueError(f"k must be <= 12, got {j_c.shape[0]}")
        return j_c

    def arbitrate_momentum(
        self,
        target_delta_q: np.ndarray,
        jacobian_c: np.ndarray,
        eps: float = 1e-6,
        method: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Arbitrates stylistic momentum Delta q* via null space projection.

        Args:
            target_delta_q: Desired 12D style momentum Delta q* in R^12.
            jacobian_c: Constraint Jacobian J_C in R^{k x 12}.
            eps: Regularization parameter (tikhonov method only).
            method: "tikhonov" or "svd"; None uses self.projector_method.

        Returns:
            Tuple of:
            - delta_q_safe: P_null * Delta q* in R^12 (worldview infringement is 0).
            - delta_q_cut: (I - P_null) * Delta q* in R^12 (orthogonal cut component).
            - p_null: Null-space projector matrix P_null in R^{12 x 12}.
        """
        dq = np.asarray(target_delta_q, dtype=np.float64).ravel()
        if dq.shape[0] != 12:
            raise ValueError(f"target_delta_q must have 12 elements, got {dq.shape[0]}")

        use = self.projector_method if method is None else method
        if use == "svd":
            p_null = self.compute_null_space_projector_svd(jacobian_c)
        elif use == "tikhonov":
            p_null = self.compute_null_space_projector(jacobian_c, eps=eps)
        else:
            raise ValueError(f"method must be 'tikhonov' or 'svd', got {use!r}")
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

def _visual_q_and_terms(visual_features: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
    """12-element position q for one image (or one local patch) plus the terms used by the legacy p proxy."""
    gabor = np.asarray(visual_features.get("gabor_frequencies", [0.25, 0.25, 0.25, 0.25]), dtype=np.float64)
    color_ent = float(visual_features.get("color_entropy", 0.5))
    chiaro = float(visual_features.get("chiaroscuro_gradient", 0.4))
    quad_ent = float(visual_features.get("quadtree_depth_entropy", 0.5))
    scanpath = visual_features.get("saliency_scanpath", [])

    if len(scanpath) >= 2:
        dists = [
            math.hypot(scanpath[i]["x"] - scanpath[i - 1]["x"], scanpath[i]["y"] - scanpath[i - 1]["y"])
            for i in range(1, len(scanpath))
        ]
        saccade_mean = float(np.mean(dists))
    else:
        saccade_mean = 0.2

    gabor_var = float(np.var(gabor))
    margin_ratio = float(visual_features.get("composition_margin_ratio", 1.0 - chiaro))

    q = np.zeros(12, dtype=np.float64)
    # Group I: Flux
    q[0] = np.clip(0.5 * saccade_mean + 0.5 * gabor_var * 4.0, 0.0, 1.0)
    q[3] = np.clip(0.4 * gabor[0] + 0.6 * saccade_mean, 0.0, 1.0)
    q[7] = np.clip(0.6 * gabor_var + 0.4 * color_ent, 0.0, 1.0)
    # Group II: Lattice
    q[1] = np.clip(quad_ent, 0.0, 1.0)
    q[4] = np.clip(0.5 * (1.0 - gabor_var * 2.0) + 0.5 * quad_ent, 0.0, 1.0)
    q[8] = np.clip(0.7 * (1.0 - color_ent) + 0.3 * quad_ent, 0.0, 1.0)
    # Group III: Void
    q[2] = np.clip(margin_ratio, 0.0, 1.0)
    q[5] = np.clip(0.6 * margin_ratio + 0.4 * (1.0 - quad_ent), 0.0, 1.0)
    q[9] = np.clip(0.8 * margin_ratio + 0.2 * (1.0 - chiaro), 0.0, 1.0)
    # Group IV: Shock
    q[6] = np.clip(chiaro, 0.0, 1.0)
    q[10] = np.clip(0.7 * chiaro + 0.3 * color_ent, 0.0, 1.0)
    q[11] = np.clip(0.8 * chiaro + 0.2 * gabor_var * 5.0, 0.0, 1.0)
    return q, {"saccade_mean": saccade_mean, "gabor_var": gabor_var, "chiaro": chiaro, "margin_ratio": margin_ratio}


def momentum_from_window_series(q_series: Sequence[np.ndarray]) -> np.ndarray:
    r"""p_i = sqrt(1/(M-1) sum_m (q_{m+1,i} - q_{m,i})^2): window-to-window RMS velocity of each element.

    Needs M >= 2 windows; returns zeros otherwise. p >= 0 by construction and is not a copy of static q.
    """
    if len(q_series) < 2:
        return np.zeros(12, dtype=np.float64)
    arr = np.asarray(q_series, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return np.sqrt(np.mean(diffs ** 2, axis=0))


# --- Equilibrium Momentum Reconstruction (summary-only path) -------------------------------------
# Quadratic style potential U(q) = 1/2 (q-mu)^T Lambda (q-mu). The trained Forge/GMM precision is not
# available to the cross-modal functors (no fitted model at this layer), so Lambda is a fixed
# 4-group coupling matrix: unit diagonal, EQ_W_WITHIN between elements of the same group
# (Flux {0,3,7}, Lattice {1,4,8}, Void {2,5,9}, Shock {6,10,11}) and EQ_W_CROSS across groups.
# Eigenvalues: 0.7 (x8), 1.3 (x3), 2.5 (x1) -> positive definite. mu = 0.5 matches the Hamiltonian
# default q* in dynamics/hamiltonian.ts. Mirrored 1:1 in packages/usir-bridges/src/crossModalFunctors.ts.
EQ_GROUPS = ((0, 3, 7), (1, 4, 8), (2, 5, 9), (6, 10, 11))
EQ_W_WITHIN = 0.3
EQ_W_CROSS = 0.1
EQ_MU = 0.5
EQ_FLOOR = 0.02   # equipartition floor: keeps every coordinate strictly positive when grad U_i = 0
EQ_SCALE = 2.2    # calibrated: mean windowed RMS p / mean reconstruction = 2.19 (51 bar series, LCG seed 12345)


def _equilibrium_coupling() -> np.ndarray:
    group_of = {i: gi for gi, g in enumerate(EQ_GROUPS) for i in g}
    lam = np.empty((12, 12), dtype=np.float64)
    for i in range(12):
        for j in range(12):
            lam[i, j] = 1.0 if i == j else (EQ_W_WITHIN if group_of[i] == group_of[j] else EQ_W_CROSS)
    return lam


_EQ_LAMBDA = _equilibrium_coupling()


def _genome_conditional_momentum(q: np.ndarray, genome: Any) -> Optional[np.ndarray]:
    r"""Conditional-Gaussian momentum from a Tier-M genome; None if no mode is usable.

    Mode k has precision Lambda_k = L_k L_k^T over z = (q, p) (dims 0-11 = q, 12-23 = p).
    m_k(q) = mu_p - Lpp^-1 Lpq (q - mu_q); q-marginal precision S_k = Lqq - Lqp Lpp^-1 Lpq,
    so N(q; mu_q, Sigma_qq) has log-density -1/2 d^T S d + 1/2 logdet S - 6 log(2 pi).
    r_k ~ w_k N_k via log-sum-exp; p = max(sum_k r_k m_k, EQ_FLOOR).
    """
    from usir_py.usir_forge import load_tier_m_binary, unpack_cholesky_24d

    if isinstance(genome, (bytes, bytearray, memoryview)):
        genome = load_tier_m_binary(bytes(genome))
    logs: List[float] = []
    means: List[np.ndarray] = []
    for k in range(len(genome["weights"])):
        w = float(genome["weights"][k])
        if not math.isfinite(w) or w <= 0.0:
            continue
        mu = np.asarray(genome["centroids"][k], dtype=np.float64)
        packed = np.asarray(genome["cholesky_factors"][k], dtype=np.float64)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(packed))):
            continue
        L = unpack_cholesky_24d(packed)
        lam = L @ L.T
        l_qq, l_qp = lam[:12, :12], lam[:12, 12:]
        l_pq, l_pp = lam[12:, :12], lam[12:, 12:]
        try:
            x = np.linalg.solve(l_pp, l_pq)
        except np.linalg.LinAlgError:
            continue
        s_mat = l_qq - l_qp @ x
        sign, logdet = np.linalg.slogdet(s_mat)
        if sign <= 0.0 or not math.isfinite(logdet) or not np.all(np.isfinite(x)):
            continue
        d = q - mu[:12]
        logn = -0.5 * float(d @ s_mat @ d) + 0.5 * logdet - 6.0 * math.log(2.0 * math.pi)
        m = mu[12:] - x @ d
        if not (math.isfinite(logn) and np.all(np.isfinite(m))):
            continue
        logs.append(math.log(w) + logn)
        means.append(m)
    if not logs:
        return None
    lg = np.asarray(logs)
    r = np.exp(lg - lg.max())
    r /= r.sum()
    return np.maximum(np.sum(r[:, None] * np.asarray(means), axis=0), EQ_FLOOR)


def momentum_from_summary(q: np.ndarray, legacy_proxy: Optional[np.ndarray] = None, genome: Any = None) -> np.ndarray:
    r"""Equilibrium Momentum Reconstruction: 12-D p from a single summary q (no window series).

    g = grad U(q) = Lambda (q - mu);  r_i = EQ_SCALE * sqrt(g_i^2 + EQ_FLOOR^2)  (> 0 for all i).
    A summary displaced from the style equilibrium would relax with force -g, so |g_i| is the
    momentum scale it carries; EQ_FLOOR is the equipartition floor. The legacy 4 proxy dims
    (p[0,2,6,11], summary statistics, not temporal momenta) combine as hypot(proxy_i, r_i).
    Deterministic; not a copy of q (depends on the coupled displacement from mu).

    genome (optional): 4096-byte Tier-M binary or the dict from usir_forge.load_tier_m_binary. When
    given, p is the responsibility-weighted conditional mean of the 3-mode Gaussian over z = (q, p)
    (see _genome_conditional_momentum); the fixed-Lambda path is the fallback if no mode is valid.
    """
    q = np.asarray(q, dtype=np.float64)
    p = _genome_conditional_momentum(q, genome) if genome is not None else None
    if p is None:
        g = _EQ_LAMBDA @ (q - EQ_MU)
        p = EQ_SCALE * np.sqrt(g * g + EQ_FLOOR * EQ_FLOOR)
    if legacy_proxy is not None:
        proxy = np.asarray(legacy_proxy, dtype=np.float64)
        p = np.where(proxy != 0.0, np.hypot(proxy, p), p)
    return p


def visual_functor_forward_24d(visual_features: Dict[str, Any]) -> np.ndarray:
    r"""Maps 2D image features to 24D USIR symplectic state z = (q, p).

    Expected features: gabor_frequencies, color_entropy, chiaroscuro_gradient, quadtree_depth_entropy,
    saliency_scanpath, composition_margin_ratio (summary of the whole image), and optionally
    patch_features: ordered list of the same feature dicts for local blocks / fixation patches.
    With >= 2 patches all 12 momentum coordinates are the RMS step-to-step change of the per-patch
    elements. Without patches the input has no temporal information: p is the Equilibrium Momentum
    Reconstruction `momentum_from_summary` (all 12 dims > 0; legacy proxy dims p[0,2,6,11] combined).
    """
    q, t = _visual_q_and_terms(visual_features)
    patches = visual_features.get("patch_features") or []
    if len(patches) >= 2:
        p = momentum_from_window_series([_visual_q_and_terms(pf)[0] for pf in patches])
    else:
        p = np.zeros(12, dtype=np.float64)
        p[0] = float(t["saccade_mean"] * 1.5)
        p[2] = float(t["margin_ratio"] * 0.8)
        p[6] = float(t["chiaro"] * 1.8)
        p[11] = float(t["chiaro"] * 2.0 + t["gabor_var"])
        p = momentum_from_summary(q, p)
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


def _music_q_and_terms(music_features: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
    torus_dist = float(music_features.get("torus_distance", 0.3))
    diss = float(music_features.get("dissonance", 0.2))
    synco = float(music_features.get("syncopation", 0.25))
    rest = float(music_features.get("rest_ratio", 0.2))
    bpm = float(music_features.get("bpm", 100.0))
    norm_bpm = float(np.clip((bpm - 60.0) / 120.0, 0.0, 1.0))

    q = np.zeros(12, dtype=np.float64)
    q[0] = np.clip(0.6 * norm_bpm + 0.4 * synco, 0.0, 1.0)
    q[3] = np.clip(0.5 * synco + 0.5 * torus_dist, 0.0, 1.0)
    q[7] = np.clip(0.7 * synco + 0.3 * norm_bpm, 0.0, 1.0)
    consonance = 1.0 - diss
    q[1] = np.clip(consonance, 0.0, 1.0)
    q[4] = np.clip(0.6 * consonance + 0.4 * (1.0 - torus_dist), 0.0, 1.0)
    q[8] = np.clip(0.8 * consonance + 0.2 * (1.0 - synco), 0.0, 1.0)
    q[2] = np.clip(rest, 0.0, 1.0)
    q[5] = np.clip(0.7 * rest + 0.3 * (1.0 - norm_bpm), 0.0, 1.0)
    q[9] = np.clip(0.9 * rest + 0.1 * (1.0 - diss), 0.0, 1.0)
    q[6] = np.clip(diss, 0.0, 1.0)
    q[10] = np.clip(0.6 * diss + 0.4 * torus_dist, 0.0, 1.0)
    q[11] = np.clip(0.7 * diss + 0.3 * synco, 0.0, 1.0)
    return q, {"norm_bpm": norm_bpm, "rest": rest, "diss": diss, "torus_dist": torus_dist}


def music_functor_forward_24d(music_features: Dict[str, Any]) -> np.ndarray:
    r"""Maps note/chord sequence features to 24D USIR symplectic state z = (q, p).

    Expected features: torus_distance, dissonance, syncopation, rest_ratio, bpm (whole piece), and
    optionally bar_features: ordered per-bar (time-window) feature dicts of the same keys. With >= 2
    bars all 12 momentum coordinates are the RMS bar-to-bar change of the per-bar elements. Without
    bars p is the Equilibrium Momentum Reconstruction `momentum_from_summary` (all 12 dims > 0).
    """
    q, t = _music_q_and_terms(music_features)
    bars = music_features.get("bar_features") or []
    if len(bars) >= 2:
        p = momentum_from_window_series([_music_q_and_terms(bf)[0] for bf in bars])
    else:
        p = np.zeros(12, dtype=np.float64)
        p[0] = float(t["norm_bpm"] * 1.5)
        p[2] = float(t["rest"] * 0.9)
        p[6] = float(t["diss"] * 1.6)
        p[11] = float(t["diss"] * 1.8 + t["torus_dist"])
        p = momentum_from_summary(q, p)
    return np.concatenate([q, p])


_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# Low-discrepancy (golden-ratio family) sequences: deterministic, monotone in their threshold,
# and bit-identical between the Python and TypeScript implementations.
_PHI = 0.6180339887498949
_PLASTIC = 0.7548776662466927
_PLASTIC2 = 0.5698402909980532
_SILVER = 0.3819660112501051


def _frac(x: float) -> float:
    return x - math.floor(x)


def _synthesize_music_sequence(
    q: np.ndarray,
    p: np.ndarray,
    bpm: float,
    harmonic_tension: float,
    rest_interval: float,
    soliton_accent_gain: float,
    n_slots: int = 16,
) -> List[Dict[str, float]]:
    """24D-driven note sequence on an eighth-note grid of n_slots slots.

    Flux (q0,q3,q7)      -> onset density: a slot sounds when a golden-ratio sequence value is below it.
    Void (rest_interval) -> lowers density; skipped slots are real rests (no note is emitted).
    Lattice (q1,q4,q8)   -> voice-leading: stepwise motion, damps leaps.
    Shock (tension,q10)  -> leap probability/size, circle-of-fifths tonic motion every 4 notes,
                            tritone (+6) on the third note of a group resolved up a semitone to the fifth.
    Syncopation (q7,p7)  -> onsets displaced by a quarter beat.
    """
    beat_sec = 60.0 / bpm
    slot_sec = beat_sec / 2.0
    flux = 0.5 * q[0] + 0.3 * q[3] + 0.2 * q[7]
    lattice = (q[1] + q[4] + q[8]) / 3.0
    density = float(np.clip(0.35 + 0.6 * flux - 0.7 * rest_interval, 0.1, 1.0))
    leap_prob = float(np.clip(0.1 + 0.8 * harmonic_tension * (1.0 - 0.5 * lattice) + 0.3 * math.tanh(p[11]), 0.0, 0.95))
    leap_size = 2 + int(math.floor(harmonic_tension * 4.0))
    fifths_steps = int(math.floor(q[10] * 3.0))
    synco_prob = float(np.clip(0.8 * q[7] + 0.2 * math.tanh(p[7]), 0.0, 0.9))
    gain_base = float(np.clip(0.4 + 0.3 * soliton_accent_gain, 0.1, 1.0))

    onsets = [sl for sl in range(n_slots) if _frac(sl * _PHI + 0.5) < density]
    if not onsets:
        onsets = [0]

    seq: List[Dict[str, float]] = []
    degree = 0
    for k, slot in enumerate(onsets):
        phase = k // 4
        tonic = 60 + ((7 * fifths_steps * phase) % 12)
        if tonic > 66:
            tonic -= 12
        direction = 1 if _frac(k * _PLASTIC2 + 0.1) < 0.5 else -1
        if _frac(k * _PLASTIC + 0.2) < leap_prob:
            degree += direction * leap_size
        else:
            degree += direction
        degree = max(-7, min(14, degree))
        octave, idx = divmod(degree, 7)
        pitch = tonic + 12 * octave + _MAJOR_SCALE[idx]
        if harmonic_tension > 0.6 and k % 4 == 2:
            pitch = tonic + 6
        elif harmonic_tension > 0.6 and k % 4 == 3:
            pitch = tonic + 7
        pitch = max(36, min(96, pitch))

        onset_offset = beat_sec / 4.0 if _frac(k * _SILVER + 0.3) < synco_prob else 0.0
        duration = slot_sec * (1.0 - rest_interval * 0.5)
        gain = float(np.clip(gain_base + (0.2 * soliton_accent_gain if k % 4 == 0 else 0.0), 0.1, 1.0))
        seq.append({
            "note": int(pitch),
            "time": float(slot * slot_sec + onset_offset),
            "duration": float(duration),
            "gain": gain,
        })
    return seq


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

    web_audio_seq = _synthesize_music_sequence(
        q, p, bpm=bpm, harmonic_tension=harmonic_tension, rest_interval=rest_interval,
        soliton_accent_gain=soliton_accent_gain,
    )

    return {
        "bpm": bpm,
        "harmonic_tension": harmonic_tension,
        "rest_interval": rest_interval,
        "reverb_decay": reverb_decay,
        "rubato_depth": rubato_depth,
        "soliton_accent_gain": soliton_accent_gain,
        "web_audio_sequence": web_audio_seq,
    }
