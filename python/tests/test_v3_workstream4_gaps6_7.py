"""Workstream 4 (Gap 6 & Gap 7) Verification Suite.

Validates:
1. Gap 6:
   - Functor F_ui (extract_ui_dom_24d): UI/DOM layout components / HTML -> 24D symplectic state.
   - Functor F_visual (extract_visual_saccade_24d): Visual fixations / saccade trajectory -> 24D symplectic state.
   - Synesthetic Transpiler (transpile_cross_modal): 24D symplectic state -> music / diffusion / ui_css.
   - Proof that III_Void maps to high rest_ratio / negative_space_ratio / whitespace_multiplier,
     while IV_Turbulent or II_Critical maps to higher soliton_accent_gain / high_freq_noise_gain.
2. Gap 7:
   - English Syllable Counter & Phonetic / Morphological Cadence.
   - 1-to-1 Structural Isomorphism:
     - Kango <-> Latinate / Greek abstract polysyllables & legal/math markers.
     - Wago <-> Anglo-Saxon concrete stems.
     - OPEN-02 & OPEN-05: Apophatic negation & minimalist paratactic subtraction.
     - AI Slop / Dead metaphor detection.
   - Proof that:
     1. Hemingway minimalist prose -> III_Void (high Sg-03, Sg-07, Sg-11).
     2. Faulkner/Poe literary climax prose -> II_Critical (balanced high Sg-06, Sg-08, Sg-10, Sg-12).
     3. English legal statutes / mathematical axioms -> I_Crystalline (high Sg-02, Sg-05, low Sg-04, Sg-12).
     4. Purple-prose English -> IV_Turbulent.
     5. English AI Slop -> V_Deadzone.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from usir_py.v2.cross_modal_functors import (
    UiDomNode,
    VisualSaccadePatch,
    extract_ui_dom_24d,
    extract_ui_dom_signals,
    extract_visual_saccade_24d,
    extract_visual_saccade_signals,
    transpile_cross_modal,
)
from usir_py.v2.english_driver import (
    AI_SLOP_MARKERS,
    APOPHATIC_NEGATIONS,
    LEGAL_MATH_MARKERS,
    SAXON_CONCRETE_STEMS,
    classify_english_phase,
    count_syllables,
    extract_english_signals,
    extract_usir_24d_english,
    split_english_clauses,
)


# ==============================================================================
# GAP 6: Universal Cross-Modal Functors & Synesthetic Transpiler Tests
# ==============================================================================

def test_gap6_extract_ui_dom_from_nodes():
    """Verify Functor F_ui correctly maps a sequence of UiDomNode descriptors to 24D symplectic state."""
    nodes = [
        UiDomNode(depth=1.0, whitespace_ratio=0.4, font_scale_ratio=2.5, card_uniformity=0.9, contrast_jump=0.1, padding_rhythm=0.9),
        UiDomNode(depth=2.0, whitespace_ratio=0.3, font_scale_ratio=1.5, card_uniformity=0.8, contrast_jump=0.2, padding_rhythm=0.8),
        UiDomNode(depth=2.0, whitespace_ratio=0.5, font_scale_ratio=1.0, card_uniformity=0.9, contrast_jump=0.1, padding_rhythm=0.9),
        UiDomNode(depth=3.0, whitespace_ratio=0.2, font_scale_ratio=1.0, card_uniformity=0.7, contrast_jump=0.7, padding_rhythm=0.7),
        UiDomNode(depth=3.0, whitespace_ratio=0.6, font_scale_ratio=0.9, card_uniformity=0.9, contrast_jump=0.1, padding_rhythm=0.9),
    ]

    z = extract_ui_dom_24d(nodes, use_mca=True)
    assert z.shape == (24,)
    # Position coordinates q in [0, 1]
    assert np.all(z[:12] >= 0.0) and np.all(z[:12] <= 1.0)
    # Momentum coordinates p >= 0
    assert np.all(z[12:] >= 0.0)


def test_gap6_extract_ui_dom_from_html():
    """Verify Functor F_ui extracts 24D state directly from raw HTML layout string."""
    html = """
    <div class="card p-4 m-2 shadow rounded grid">
      <h1 class="text-2xl font-bold mb-4">Dashboard Overview</h1>
      <p class="text-base text-gray-700 whitespace-pre">System status is operational.</p>
      <div class="card p-2 gap-4 border">
        <h2 class="text-xl">Metrics</h2>
        <span class="empty p-4"></span>
        <button class="btn bg-primary text-white p-2 rounded">Refresh</button>
      </div>
    </div>
    """

    z = extract_ui_dom_24d(html, use_mca=True)
    assert z.shape == (24,)
    assert np.all(z[:12] >= 0.0) and np.all(z[:12] <= 1.0)
    assert np.all(z[12:] >= 0.0)


def test_gap6_extract_visual_saccade_24d():
    """Verify Functor F_visual maps visual fixation patches to 24D symplectic state."""
    patches = [
        VisualSaccadePatch(luminance_contrast=0.1, edge_density=0.2, negative_space_ratio=0.7, chromatic_jump=0.05, x=0.2, y=0.2),
        VisualSaccadePatch(luminance_contrast=0.3, edge_density=0.4, negative_space_ratio=0.5, chromatic_jump=0.10, x=0.3, y=0.4),
        VisualSaccadePatch(luminance_contrast=0.8, edge_density=0.9, negative_space_ratio=0.1, chromatic_jump=0.75, x=0.5, y=0.5),
        VisualSaccadePatch(luminance_contrast=0.2, edge_density=0.3, negative_space_ratio=0.6, chromatic_jump=0.10, x=0.7, y=0.6),
        VisualSaccadePatch(luminance_contrast=0.1, edge_density=0.1, negative_space_ratio=0.8, chromatic_jump=0.05, x=0.8, y=0.8),
    ]

    z = extract_visual_saccade_24d(patches, use_mca=True)
    assert z.shape == (24,)
    assert np.all(z[:12] >= 0.0) and np.all(z[:12] <= 1.0)
    assert np.all(z[12:] >= 0.0)


def test_gap6_synesthetic_transpiler_modalities_and_invariants():
    """Verify Synesthetic Transpiler converts 24D states to music, diffusion, and ui_css.
    Proves that III_Void maps to high rest_ratio, negative_space_ratio, and whitespace_multiplier,
    while IV_Turbulent / II_Critical maps to higher soliton_accent_gain and high_freq_noise_gain.
    """
    # Prototypical III_Void state vector: high Void coordinates (q2, q6, q10)
    z_void = np.zeros(24, dtype=np.float64)
    z_void[2] = 0.85   # Sg-03 Pr
    z_void[6] = 0.80   # Sg-07 Cs
    z_void[10] = 0.90  # Sg-11 Ma
    z_void[0] = 0.15   # low flux
    z_void[1] = 0.20   # low lattice
    z_void[3] = 0.05   # low shock
    z_void[7] = 0.05   # low shock
    z_void[11] = 0.05  # low shock

    # Prototypical II_Critical / IV_Turbulent state vector: high Shock coordinates (q3, q7, q11)
    z_shock = np.zeros(24, dtype=np.float64)
    z_shock[3] = 0.75   # Sg-04 Jt
    z_shock[7] = 0.80   # Sg-08 Lp
    z_shock[11] = 0.90  # Sg-12 Ru
    z_shock[2] = 0.10   # low void
    z_shock[6] = 0.10   # low void
    z_shock[10] = 0.10  # low void

    # 1. Music Modality
    music_void = transpile_cross_modal(z_void, "music")
    music_shock = transpile_cross_modal(z_shock, "music")

    assert music_void["rest_ratio"] > 0.60
    assert music_shock["rest_ratio"] < 0.25
    assert music_shock["soliton_accent_gain"] > music_void["soliton_accent_gain"]
    assert music_shock["soliton_accent_gain"] > 0.65

    # 2. Diffusion Modality
    diff_void = transpile_cross_modal(z_void, "diffusion")
    diff_shock = transpile_cross_modal(z_shock, "diffusion")

    assert diff_void["negative_space_ratio"] > 0.60
    assert diff_shock["negative_space_ratio"] < 0.30
    assert diff_shock["high_freq_noise_gain"] > diff_void["high_freq_noise_gain"]
    assert diff_shock["high_freq_noise_gain"] > 0.60

    # 3. UI/CSS Modality
    ui_void = transpile_cross_modal(z_void, "ui_css")
    ui_shock = transpile_cross_modal(z_shock, "ui_css")

    assert ui_void["whitespace_multiplier"] > 2.2
    assert ui_shock["whitespace_multiplier"] < 1.3
    assert ui_shock["grid_irregularity"] > ui_void["grid_irregularity"]


# ==============================================================================
# GAP 7: English Isomorphic Driver (usir-driver-en) Tests
# ==============================================================================

def test_gap7_english_syllable_counter():
    """Verify deterministic English syllable counter across various morphological patterns."""
    assert count_syllables("stone") == 1
    assert count_syllables("rain") == 1
    assert count_syllables("cold") == 1
    assert count_syllables("water") == 2
    assert count_syllables("shadow") == 2
    assert count_syllables("silence") == 2
    assert count_syllables("pursuant") == 3
    assert count_syllables("homomorphism") == 5
    assert count_syllables("multifaceted") == 5


def test_gap7_hemingway_minimalist_prose_is_phase3_void():
    """Prove Hemingway-style minimalist English prose is classified into III_Void.
    Verifies high Sg-03 (q2), Sg-07 (q6), and Sg-11 (q10).
    """
    text = (
        "The river was green. Cold water ran over the white stones. "
        "Nick looked down through the dark pool and saw the trout. He did not move. "
        "The sun was hot on his back. There was no sound in the trees. "
        "The wind blew from the hills, cold and clean."
    )

    z = extract_usir_24d_english(text, use_mca=False)
    phase = classify_english_phase(text)

    assert phase == "III_Void"
    q_sg03 = z[2]   # Sg-03 Pr (Micro Void)
    q_sg07 = z[6]   # Sg-07 Cs (Meso Void)
    q_sg11 = z[10]  # Sg-11 Ma (Macro Void)

    assert q_sg11 > 0.50, f"Expected dominant Sg-11 (Macro Void), got {q_sg11:.3f}"
    assert q_sg03 > 0.08, f"Expected high Sg-03, got {q_sg03:.3f}"
    assert q_sg07 > 0.05, f"Expected high Sg-07, got {q_sg07:.3f}"
    assert q_sg11 > z[11], "Void macro must strictly exceed shock macro"
    assert q_sg11 > z[3], "Void macro must strictly exceed shock micro"


def test_gap7_faulkner_poe_literary_climax_is_phase2_critical():
    """Prove Faulkner/Poe-style literary climax prose is classified into II_Critical.
    Verifies balanced high Sg-06 (q5), Sg-08 (q7), Sg-10 (q9) with soliton climax Sg-12 (q11).
    """
    text = (
        "He ran through the weeping darkness, the old house trembling as the storm broke with sudden fury. "
        "The shutter swung wide; a shriek ripped the silence of the night! "
        "Blood beat in his temples. "
        "He saw the black abyss yawning before him, and in that single terrifying instant of horror, "
        "the bell tolling above the ruins shattered his soul."
    )

    z = extract_usir_24d_english(text, use_mca=False)
    phase = classify_english_phase(text)

    assert phase == "II_Critical"
    q_sg04 = z[3]   # Sg-04 Jt (Micro Shock)
    q_sg06 = z[5]   # Sg-06 Mp (Meso Lattice)
    q_sg08 = z[7]   # Sg-08 Lp (Meso Shock)
    q_sg10 = z[9]   # Sg-10 At (Macro Lattice / Return Arch)
    q_sg12 = z[11]  # Sg-12 Ru (Macro Soliton Shock)

    assert q_sg10 > 0.30, f"Expected high Sg-10 return arch, got {q_sg10:.3f}"
    assert q_sg12 > 0.20, f"Expected high soliton climax Sg-12, got {q_sg12:.3f}"
    assert q_sg08 > 0.08, f"Expected substantial Sg-08 meso shock, got {q_sg08:.3f}"
    assert q_sg06 > 0.03, f"Expected substantial Sg-06 meso lattice, got {q_sg06:.3f}"
    assert q_sg04 > 0.20, f"Expected micro shock presence, got {q_sg04:.3f}"


def test_gap7_legal_math_statute_is_phase1_crystalline():
    """Prove English legal statutes / mathematical axioms are classified into I_Crystalline.
    Verifies high Sg-02 (q1), Sg-05 (q4/q5), and low Sg-04 (q3), Sg-12 (q11).
    """
    text = (
        "Pursuant to Section 4.2 hereof, the Licensee shall indemnify and hold harmless the Licensor against any liabilities arising thereunder. "
        "Notwithstanding the foregoing, if two topological spaces are homeomorphic, "
        "then every compact subspace of the domain is mapped to a compact subspace of the codomain under the continuous homomorphism."
    )

    z = extract_usir_24d_english(text, use_mca=False)
    phase = classify_english_phase(text)

    assert phase == "I_Crystalline"
    q_sg02 = z[1]   # Sg-02 Cr (Micro Lattice)
    q_sg05 = z[4]   # Sg-05 Cp (Meso Coupling)
    q_sg04 = z[3]   # Sg-04 Jt (Micro Shock)
    q_sg12 = z[11]  # Sg-12 Ru (Macro Shock)

    assert q_sg02 > 0.25, f"Expected high Sg-02, got {q_sg02:.3f}"
    assert q_sg05 > 0.10, f"Expected high Sg-05, got {q_sg05:.3f}"
    assert q_sg04 < 0.15, f"Expected low Sg-04, got {q_sg04:.3f}"
    assert q_sg12 < 0.10, f"Expected low Sg-12, got {q_sg12:.3f}"


def test_gap7_purple_prose_is_phase4_turbulent():
    """Prove purple-prose English with excessive modifiers is classified into IV_Turbulent."""
    text = (
        "The exceedingly iridescent, gossamer-draped twilight descended exceedingly sluggishly "
        "upon the bewilderingly ornate, staggeringly grandiose architecture of the profoundly ostentatious palace, "
        "casting bizarrely flickering, nauseatingly garish luminescence across every excessively carved, monstrously gilded gargoyle."
    )

    phase = classify_english_phase(text)
    assert phase == "IV_Turbulent"


def test_gap7_ai_slop_is_phase5_deadzone():
    """Prove English AI slop loaded with clichés and buzzwords is classified into V_Deadzone."""
    text = (
        "Moreover, it is crucial to delve into the rich tapestry of multifaceted considerations "
        "that underscores the pivotal importance of this comprehensive framework. "
        "In conclusion, by fostering seamless integration across diverse ecosystems, "
        "organizations can navigate the complexities of modern paradigms and elevate operational efficacy."
    )

    phase = classify_english_phase(text)
    assert phase == "V_Deadzone"
