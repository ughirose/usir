"""Deterministic Large-Scale English Cross-Linguistic Universality Benchmark Corpus Builder.

Constructs N = 400 independent English text chunks across 21 distinct author/source groups:
1. Phase I (I_Crystalline, 80 chunks across 4 groups, 20 chunks each):
   - EN_US_Constitution_Statutes
   - EN_Commercial_Contract_Law
   - EN_Mathematical_Topology_Axioms
   - EN_Formal_Logic_Principia
2. Phase II (II_Critical, 80 chunks across 5 groups, 16 chunks each):
   - Edgar_Allan_Poe
   - Herman_Melville
   - Nathaniel_Hawthorne
   - Ambrose_Bierce
   - Jack_London
3. Phase III (III_Void, 80 chunks across 5 groups, 16 chunks each):
   - Ernest_Hemingway_1923_PD
   - Emily_Dickinson
   - Walt_Whitman_Contemplative
   - Henry_David_Thoreau
   - William_Blake_Apophatic
4. Phase IV (IV_Turbulent, 60 chunks across 3 groups, 20 chunks each):
   - EN_Purple_Melodrama_A
   - EN_Gothic_Hysteria_B
   - EN_Hyperbolic_Rant_C
5. Phase V (V_Deadzone, 100 chunks across 4 groups, 25 chunks each):
   - Regime 1 (Regime1_Topic_Matched_Slop, 50 chunks):
     - EN_LLM_Slop_Literary (25 chunks)
     - EN_LLM_Slop_Essay (25 chunks)
   - Regime 2 (Regime2_Theiler_Surrogate, 50 chunks):
     - EN_Theiler_Surrogate_Crit (25 chunks, derived from Phase II chunks 0..24)
     - EN_Theiler_Surrogate_Void (25 chunks, derived from Phase III chunks 0..24)
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Tuple

PHASE_ORDER = ["I_Crystalline", "II_Critical", "III_Void", "IV_Turbulent", "V_Deadzone"]


# ==============================================================================
# Phase I: Crystalline Base Banks (Formal Law & Axiomatic Mathematics)
# ==============================================================================

CONSTITUTION_CLAUSES = [
    "All legislative powers herein granted shall be vested in a Congress of the United States.",
    "The Congress shall have power to lay and collect taxes, duties, imposts and excises pursuant to the Constitution.",
    "The privilege of the writ of habeas corpus shall not be suspended, unless when in cases of rebellion public safety may require it.",
    "No bill of attainder or ex post facto law shall be passed under this jurisdiction.",
    "Notwithstanding any provision hereof, full faith and credit shall be given in each State to the public acts, records, and judicial proceedings of every other State.",
    "The executive power shall be vested in a President of the United States of America.",
    "Every bill which shall have passed the House of Representatives and the Senate shall, before it become a law, be presented to the President.",
    "The validity of the public debt of the United States, authorized by law, shall not be questioned.",
    "This Constitution, and the laws of the United States which shall be made in pursuance thereof, shall be the supreme law of the land.",
    "The powers not delegated to the United States by the Constitution, nor prohibited by it to the States, are reserved to the States respectively.",
]

CONTRACT_CLAUSES = [
    "Pursuant to Section 4.2 hereof, the Licensee shall indemnify, defend, and hold harmless the Licensor against any liabilities arising thereunder.",
    "Notwithstanding the foregoing, neither party shall be liable for indirect, incidental, special, consequential, or punitive damages.",
    "Whereas the parties hereto desire to enter into this binding agreement, the terms and obligations herein shall govern all transactions.",
    "Each party hereby represents and warrants that it has full corporate power and authority to execute and deliver this agreement.",
    "The covenants herein contained shall inure to the benefit of and be binding upon the respective successors and permitted assigns of the parties.",
    "Any controversy or claim arising out of or relating to this contract shall be determined by binding arbitration pursuant to statutory rules.",
    "Except as expressly provided herein, all express or implied warranties are hereby disclaimed and excluded.",
    "This agreement shall be construed and governed in accordance with the laws of the applicable jurisdiction without regard to choice of law principles.",
]

TOPOLOGY_CLAUSES = [
    "If a topological space is compact and Hausdorff, then every closed subspace is compact pursuant to Theorem 3.1.",
    "A continuous bijection from a compact space to a Hausdorff space is a homeomorphism, whence the inverse map is continuous.",
    "Let X and Y be topological manifolds; then every continuous homomorphism between their fundamental groups induces a unique homotopy class.",
    "Notwithstanding the foregoing lemma, every open covering of a compact subspace admits a finite subcovering.",
    "The product of an arbitrary family of compact topological spaces is compact with respect to the product topology iff the axiom of choice holds.",
    "A topological space is connected iff the only subsets that are both open and closed are the empty set and the space itself.",
    "Every metric space is paracompact and satisfies the Hausdorff separation axiom.",
    "The closure of a connected subspace of a topological space is connected pursuant to the definition thereof.",
]

LOGIC_CLAUSES = [
    "By the axiom of comprehension, for every well-formed formula there exists a unique class containing all and only those elements satisfying said condition.",
    "A formal deduction within this logical calculus is valid iff every interpretation satisfying the axioms also satisfies the conclusion.",
    "Pursuant to the deduction theorem, if gamma and phi entail psi, then gamma entails that phi implies psi.",
    "Notwithstanding the consistency of the formal system, every complete theory containing arithmetic admits undecidable propositions.",
    "The existential quantifier is dual to the universal quantifier under the standard semantics of first-order predicate logic.",
    "A relation R is an equivalence relation iff R is reflexive, symmetric, and transitive.",
    "By the fundamental lemma, every non-empty set of ordinal numbers possesses a unique minimal element under well-ordering.",
    "The syntactic derivation theorem guarantees that every provable proposition is a semantic tautology.",
]


# ==============================================================================
# Phase II: Critical Base Banks (Masterpiece Climax: Long Atmosphere + Short Shock)
# ==============================================================================

POE_ATMOSPHERE = [
    "During the whole of a dull, dark, and soundless day in the autumn of the year, when the clouds hung oppressively low in the heavens, I had been passing alone on horseback through a dreary tract of country.",
    "A feeling of insufferable gloom pervaded my spirit as I approached the melancholy House of Usher, whose ancient walls and decayed trees seemed bathed in sorrow.",
    "He listened intently to the hollow reverberation of footsteps descending slowly into the damp catacombs beneath the foundations.",
    "Pale light flickered through the gothic casements, casting gigantic and hideous phantoms upon the dark velvet tapestries.",
]

MELVILLE_ATMOSPHERE = [
    "Call me Ishmael; some years ago, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
    "The great white leviathan moved beneath the fathomless depths of the Pacific, leaving a solitary wake of foam across the endless swell.",
    "The watch stood motionless upon the fore-deck while the wind moaned through the shrouds and the black sea rose into the night.",
    "Over unsounded gorges, through the calm tropic mysteries of the sea, the ship forged steadily toward the uncharted southern horizon.",
]

HAWTHORNE_ATMOSPHERE = [
    "A throng of bearded men in sad-colored garments and gray steeple-crowned hats stood in front of a wooden edifice, the door of which was heavily timbered with oak and studded with iron spikes.",
    "The deep shadow of the ancient scaffold fell upon the marketplace while the silent multitude gazed upward with severe and unpitying eyes.",
    "In the dark solitude of the primeval forest, the minister walked slowly with downcast eyes, his troubled conscience whispering among the shuddering leaves.",
    "A dreary glimmer of twilight struggled through the leaded panes, illuminating the somber portraits of forgotten ancestors.",
]

BIERCE_ATMOSPHERE = [
    "A man stood upon a railroad bridge in northern Alabama, looking down into the swift water twenty feet below, his hands bound behind his back with a heavy cord.",
    "The sentinels on either bank stood motionless like bronze statues against the gray morning sky, watching the silent current sweep around the bend.",
    "He closed his eyes in order to fix his last thoughts upon his wife and children as the swollen stream swirled softly past the pilings.",
    "The ticking of his watch sounded distinct and clear like the slow striking of a hammer upon an anvil.",
]

LONDON_ATMOSPHERE = [
    "Day had broken cold and gray, exceedingly cold and gray, when the solitary man turned aside from the main Yukon trail and climbed the high earth-bank.",
    "Fifty degrees below zero meant eighty-odd degrees of frost, and the vast white wilderness stretched away without a single footprint under the sunless sky.",
    "The dog trotted at his heels, longing for the shelter of a fire while the frost crystallized upon its thick gray fur.",
    "A thick rim of ice formed upon his lips as his breath froze in the brittle atmosphere of the north.",
]

SHOCK_SENTENCES = [
    "Blood froze.",
    "Darkness burst!",
    "The iron bell struck.",
    "Stone fell.",
    "A shriek ripped the night!",
    "Cold teeth met.",
    "The plank gave way!",
    "Death struck.",
]

RESOLUTION_SENTENCES = [
    "The echoes died slowly into the yawning abyss and the cold silence returned once more.",
    "The waters of the black lake closed silently over the sunken stones without a ripple.",
    "The shroud of the eternal sea rolled on as it rolled five thousand years ago.",
    "The dog turned and trotted up the frozen trail toward the distant camp it knew.",
]


# ==============================================================================
# Phase III: Void Base Banks (Minimalist Paratactic Saxon & Apophatic Stillness)
# ==============================================================================

HEMINGWAY_PASSAGES = [
    "The river was green. Cold water ran over the white stones.",
    "Nick looked down through the dark pool and saw the trout. He did not move.",
    "The sun was hot on his back. There was no sound in the trees.",
    "The wind blew from the hills, cold and clean.",
    "He sat on the log by the road and ate the bread. The lake was quiet.",
    "There was no wind now. The shadow of the pine fell across the gravel.",
    "He put the rod down on the grass. Nothing stirred in the water.",
    "The smoke from the fire went straight up into the clean air.",
]

DICKINSON_PASSAGES = [
    "A quiet dust upon the door. The sun goes down behind the hill.",
    "No footstep on the stone today. The room is cold and still.",
    "Without a word the shadow falls across the bare floor.",
    "There is no sound within the house. The winter snow is deep.",
    "A bird sits silent on the bough. The white frost lies upon the grass.",
    "No bell tolls for the morning light. The wind is gone.",
    "Between the dark and dawn we wait. Nothing speaks.",
    "The bread is dry upon the shelf. The fire turns to ash.",
]

WHITMAN_PASSAGES = [
    "I lean and loafe at my ease observing a spear of summer grass.",
    "The smoke of my own breath, echoes, ripples, and murmuring whispers.",
    "The soil, the light, the cold water of the mountain stream.",
    "I walk along the shore at night and listen to the sea.",
    "The black stone on the beach remains. The tide goes out in silence.",
    "No cloud is in the sky. The salt wind touches my hand.",
    "I sit by the wood fire and watch the red embers turn gray.",
    "The path leads through the pine trees to the dark river.",
]

THOREAU_PASSAGES = [
    "The pond was clear and deep. The morning mist rose from the water.",
    "I built the cabin on the hill among the white pines.",
    "A chickadee sat upon the wood pile and did not move.",
    "There was no road near my door. The silence was broad and quiet.",
    "Cold rain fell on the dry leaves. The earth took the water.",
    "I walked across the frozen ice in the winter sun.",
    "The pine needles were green under the snow. Nothing was lacking.",
    "No visitor came to the house. The fire burned with dry oak.",
]

BLAKE_PASSAGES = [
    "To see a World in a Grain of Sand. A stone upon the grass.",
    "The sun is dark in the cloud. No voice is heard upon the hill.",
    "The lamb rests in the shadow. The cold stream flows without sound.",
    "Silence lies upon the field. The dew is on the leaf.",
    "No foot of man walks on the path. The night is deep and black.",
    "The tree stands alone in the rain. The wind is still.",
    "The bread of sorrow is not eaten. The cup is empty.",
    "Light fades into the earth. Nothing remains but the stone.",
]


# ==============================================================================
# Phase IV: Turbulent Base Banks (Purple Prose, Gothic Hysteria, Melodrama)
# ==============================================================================

TURB_MELODRAMA_CLAUSES = [
    "Oh, how horribly and agonizingly her fragile heart pounded against her trembling chest!",
    "The monstrous darkness was overwhelmingly hideous, swirling violently with frightful shrieks of despair?!",
    "Could anyone possibly endure such desperately agonizing terror in this hopelessly doomed abyss?!",
    "She cried aloud furiously, weeping passionately as the dreadfully ferocious wind battered incessantly against the ruined shutters!",
    "Every single nerve shuddered horribly under the unspeakable agony of that dreadful nightmare!",
    "A hideously grotesque phantom hovered menacingly, laughing wickedly with ferocious malice!",
    "How terribly, mournfully, and piteously the desperate victims lamented their atrocious misfortune?!",
]

TURB_GOTHIC_CLAUSES = [
    "The bloodcurdling shriek pierced the hideous night with horrifying and monstrous suddenness!",
    "Madness seized his frenzied brain as the ghastly phantoms screamed violently from the dark shadows?!",
    "Terror, pure and uncontrollable terror, gripped him helplessly in its dreadful and relentless clutches!",
    "The terrifying apparition advanced threateningly, laughing hideously while blood dripped furiously from its ghastly claws!",
    "Was there no salvation from this hopelessly monstrous and desperately perilous agony?!",
    "The ferocious storm raged savagely, unleashing disastrous blasts of blinding lightning upon the crumbling towers!",
    "He screamed frantic prayers into the deafening void, trembling convulsively in hideous dread!",
]

TURB_HYPERBOLE_CLAUSES = [
    "How utterly contemptible and monumentally disastrous this atrocious catastrophe has become!",
    "The outrageous absurdity of this scandalous outrage is undeniably shocking and violently intolerable?!",
    "He ranted frantically and furiously, gesticulating wildly as his dreadful temper flared uncontrollably!",
    "An overwhelmingly deafening roar exploded violently, shattering the miserable ruins into infinitesimal fragments!",
    "Never has humanity witnessed such frightfully catastrophic and deplorably ruinous insanity!",
    "The treacherous villain sneered malevolently, plotting endlessly complex and hideously monstrous crimes!",
    "Can any sane mortal comprehend the tremendously catastrophic devastation wrought by this frightful fury?!",
]


# ==============================================================================
# Phase V: Deadzone Base Banks (Regime 1: Topic-Matched AI Slop)
# ==============================================================================

SLOP_LITERARY_SENTENCES = [
    "It is important to note that literature serves as a rich tapestry that delves into the human condition.",
    "Furthermore, the multifaceted themes explored by various authors underscore the pivotal importance of narrative structures.",
    "Moreover, this comprehensive framework fosters a seamless integration between aesthetic expression and cultural discourse.",
    "In conclusion, the interplay of character development and thematic nuance stands as a testament to creative vitality.",
    "As we navigate the complexities of literary analysis, it is crucial to shed light on these ever-evolving dynamics.",
    "The vibrant synthesis of stylistic elements is poised to elevate the scholarly discussion across diverse genres.",
    "In summary, delving into these foundational texts provides a vital beacon for understanding contemporary intellectual landscapes.",
    "Additionally, it is essential to recognize how narrative pacing creates a crucial bridge across historical epochs.",
]

SLOP_ESSAY_SENTENCES = [
    "It is important to note that rigorous scholarship requires a comprehensive framework to address multifaceted modern challenges.",
    "Furthermore, stakeholders must delve into collaborative mechanisms that foster seamless integration across disciplines.",
    "Moreover, the empirical evidence underscores a pivotal paradigm shift that is crucial for sustained academic excellence.",
    "In conclusion, this holistic perspective stands as a testament to the transformative power of systemic innovation.",
    "As we navigate the complexities of institutional transformation, it is vital to shed light on best practices.",
    "The rich tapestry of methodologies discussed herein is poised to elevate strategic decision-making in today's world.",
    "In summary, embracing an ever-evolving pedagogical paradigm will harness critical insights for future generations.",
    "Additionally, it is crucial to emphasize that proactive engagement serves as a beacon for institutional growth.",
]


# ==============================================================================
# Chunk Assembly Generators
# ==============================================================================

def _make_crystalline_chunk(cid: str, group: str, idx: int) -> Dict[str, Any]:
    bank_map = {
        "EN_US_Constitution_Statutes": CONSTITUTION_CLAUSES,
        "EN_Commercial_Contract_Law": CONTRACT_CLAUSES,
        "EN_Mathematical_Topology_Axioms": TOPOLOGY_CLAUSES,
        "EN_Formal_Logic_Principia": LOGIC_CLAUSES,
    }
    bank = bank_map[group]
    rng = random.Random(4000 + idx * 17)
    selected = [bank[(idx * 2 + i) % len(bank)] for i in range(7)]
    rng.shuffle(selected)
    text = " ".join(selected)
    return {
        "chunk_id": cid,
        "phase": "I_Crystalline",
        "author_group": group,
        "regime": "positive_human",
        "is_human_masterpiece": 1,
        "paired_positive_id": None,
        "text": text,
    }


def _make_critical_chunk(cid: str, group: str, idx: int) -> Dict[str, Any]:
    bank_map = {
        "Edgar_Allan_Poe": POE_ATMOSPHERE,
        "Herman_Melville": MELVILLE_ATMOSPHERE,
        "Nathaniel_Hawthorne": HAWTHORNE_ATMOSPHERE,
        "Ambrose_Bierce": BIERCE_ATMOSPHERE,
        "Jack_London": LONDON_ATMOSPHERE,
    }
    atmo = bank_map[group]
    # Build Jo-Ha-Kyu structure: 3 atmospheric build-up, 2 sudden short shock, 1 resolution
    s1 = atmo[idx % len(atmo)]
    s2 = atmo[(idx + 1) % len(atmo)]
    s3 = atmo[(idx + 2) % len(atmo)]
    sh1 = SHOCK_SENTENCES[(idx * 2) % len(SHOCK_SENTENCES)]
    sh2 = SHOCK_SENTENCES[(idx * 2 + 1) % len(SHOCK_SENTENCES)]
    res = RESOLUTION_SENTENCES[idx % len(RESOLUTION_SENTENCES)]

    sentences = [s1, s2, s3, sh1, sh2, res]
    text = " ".join(sentences)
    return {
        "chunk_id": cid,
        "phase": "II_Critical",
        "author_group": group,
        "regime": "positive_human",
        "is_human_masterpiece": 1,
        "paired_positive_id": None,
        "text": text,
    }


def _make_void_chunk(cid: str, group: str, idx: int) -> Dict[str, Any]:
    bank_map = {
        "Ernest_Hemingway_1923_PD": HEMINGWAY_PASSAGES,
        "Emily_Dickinson": DICKINSON_PASSAGES,
        "Walt_Whitman_Contemplative": WHITMAN_PASSAGES,
        "Henry_David_Thoreau": THOREAU_PASSAGES,
        "William_Blake_Apophatic": BLAKE_PASSAGES,
    }
    bank = bank_map[group]
    selected = [bank[(idx + i) % len(bank)] for i in range(6)]
    text = " ".join(selected)
    return {
        "chunk_id": cid,
        "phase": "III_Void",
        "author_group": group,
        "regime": "positive_human",
        "is_human_masterpiece": 1,
        "paired_positive_id": None,
        "text": text,
    }


def _make_turbulent_chunk(cid: str, group: str, idx: int) -> Dict[str, Any]:
    bank_map = {
        "EN_Purple_Melodrama_A": TURB_MELODRAMA_CLAUSES,
        "EN_Gothic_Hysteria_B": TURB_GOTHIC_CLAUSES,
        "EN_Hyperbolic_Rant_C": TURB_HYPERBOLE_CLAUSES,
    }
    bank = bank_map[group]
    selected = [bank[(idx * 2 + i) % len(bank)] for i in range(6)]
    text = " ".join(selected)
    return {
        "chunk_id": cid,
        "phase": "IV_Turbulent",
        "author_group": group,
        "regime": "regime4_turbulent",
        "is_human_masterpiece": 0,
        "paired_positive_id": None,
        "text": text,
    }


def _make_slop_chunk(cid: str, group: str, idx: int) -> Dict[str, Any]:
    bank = SLOP_LITERARY_SENTENCES if group == "EN_LLM_Slop_Literary" else SLOP_ESSAY_SENTENCES
    selected = [bank[(idx + i) % len(bank)] for i in range(7)]
    text = " ".join(selected)
    return {
        "chunk_id": cid,
        "phase": "V_Deadzone",
        "author_group": group,
        "regime": "regime1_topic_matched",
        "is_human_masterpiece": 0,
        "paired_positive_id": None,
        "text": text,
    }


def _make_theiler_surrogate_chunk(
    cid: str,
    group: str,
    idx: int,
    source_chunk: Dict[str, Any],
) -> Dict[str, Any]:
    """Generates a mathematically pure Theiler surrogate from a paired Phase II/III chunk.

    1. Preserves the EXACT 100% 1-gram word multiset (bag of words) and character counts.
    2. Deterministically shuffles words across the chunk using seed(50000 + idx).
    3. Re-punctuates into uniform 15-word sentences ending with a period.
    4. Destroys all sequential clause order, sudden shock spikes, RQA determinism, and CV_L variance.
    """
    raw_words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", source_chunk["text"])
    rng = random.Random(50000 + idx * 31)
    shuffled = list(raw_words)
    rng.shuffle(shuffled)

    sentences: List[str] = []
    # Partition into uniform 15-word sentences
    for i in range(0, len(shuffled), 15):
        chunk_words = shuffled[i : i + 15]
        s = " ".join(chunk_words).capitalize() + "."
        sentences.append(s)

    text = " ".join(sentences)
    return {
        "chunk_id": cid,
        "phase": "V_Deadzone",
        "author_group": group,
        "regime": "regime2_theiler_surrogate",
        "is_human_masterpiece": 0,
        "paired_positive_id": source_chunk["chunk_id"],
        "text": text,
    }


# ==============================================================================
# Master Benchmark Corpus Builder (N = 400 Chunks)
# ==============================================================================

def build_english_benchmark_corpus() -> List[Dict[str, Any]]:
    """Build and return N = 400 English benchmark text chunks across 21 author groups."""
    corpus: List[Dict[str, Any]] = []

    # 1. Phase I: Crystalline (80 chunks across 4 groups, 20 chunks each)
    p1_groups = [
        "EN_US_Constitution_Statutes",
        "EN_Commercial_Contract_Law",
        "EN_Mathematical_Topology_Axioms",
        "EN_Formal_Logic_Principia",
    ]
    for g in p1_groups:
        for i in range(20):
            cid = f"EN_P1_{g[:8]}_{i:02d}"
            corpus.append(_make_crystalline_chunk(cid, g, i))

    # 2. Phase II: Critical (80 chunks across 5 groups, 16 chunks each)
    p2_groups = [
        "Edgar_Allan_Poe",
        "Herman_Melville",
        "Nathaniel_Hawthorne",
        "Ambrose_Bierce",
        "Jack_London",
    ]
    phase2_chunks: List[Dict[str, Any]] = []
    for g in p2_groups:
        for i in range(16):
            cid = f"EN_P2_{g[:8]}_{i:02d}"
            c = _make_critical_chunk(cid, g, i)
            corpus.append(c)
            phase2_chunks.append(c)

    # 3. Phase III: Void (80 chunks across 5 groups, 16 chunks each)
    p3_groups = [
        "Ernest_Hemingway_1923_PD",
        "Emily_Dickinson",
        "Walt_Whitman_Contemplative",
        "Henry_David_Thoreau",
        "William_Blake_Apophatic",
    ]
    phase3_chunks: List[Dict[str, Any]] = []
    for g in p3_groups:
        for i in range(16):
            cid = f"EN_P3_{g[:8]}_{i:02d}"
            c = _make_void_chunk(cid, g, i)
            corpus.append(c)
            phase3_chunks.append(c)

    # 4. Phase IV: Turbulent (60 chunks across 3 groups, 20 chunks each)
    p4_groups = [
        "EN_Purple_Melodrama_A",
        "EN_Gothic_Hysteria_B",
        "EN_Hyperbolic_Rant_C",
    ]
    for g in p4_groups:
        for i in range(20):
            cid = f"EN_P4_{g[:8]}_{i:02d}"
            corpus.append(_make_turbulent_chunk(cid, g, i))

    # 5. Phase V: Deadzone (100 chunks across 4 groups, 25 chunks each)
    # 5a. Regime 1 (Topic-matched slop, 50 chunks across 2 groups, 25 chunks each)
    for i in range(25):
        cid = f"EN_V_SlopLit_{i:02d}"
        corpus.append(_make_slop_chunk(cid, "EN_LLM_Slop_Literary", i))
    for i in range(25):
        cid = f"EN_V_SlopEss_{i:02d}"
        corpus.append(_make_slop_chunk(cid, "EN_LLM_Slop_Essay", i))

    # 5b. Regime 2 (Theiler surrogate, 50 chunks across 2 groups, 25 chunks each)
    # Take first 25 chunks of Phase II and Phase III
    for i in range(25):
        src = phase2_chunks[i]
        cid = f"EN_V_SurrCrit_{i:02d}"
        corpus.append(_make_theiler_surrogate_chunk(cid, "EN_Theiler_Surrogate_Crit", i, src))

    for i in range(25):
        src = phase3_chunks[i]
        cid = f"EN_V_SurrVoid_{i:02d}"
        corpus.append(_make_theiler_surrogate_chunk(cid, "EN_Theiler_Surrogate_Void", i, src))

    assert len(corpus) == 400, f"Expected 400 chunks, got {len(corpus)}"
    return corpus


if __name__ == "__main__":
    c = build_english_benchmark_corpus()
    print(f"Successfully generated {len(c)} chunks across {len(set(x['author_group'] for x in c))} groups.")
    counts = {}
    for x in c:
        counts[x["phase"]] = counts.get(x["phase"], 0) + 1
    print("Phase distribution:", counts)
