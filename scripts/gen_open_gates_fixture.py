"""Generates packages/usir-core/tests/fixtures/open_gates_parity.json (Python reference).

Run: cd python; uv run python ../scripts/gen_open_gates_fixture.py
packages/usir-core/tests/open-gates-parity.test.ts asserts the TS port of OPEN-01/02/04
(packages/usir-core/src/linguistics/openGates.ts) against this file within 1e-9.
All inputs are plain strings / numeric arrays (no parser involved).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from usir_py.linguistic_gates import (
    apply_open01_polysemy_damping,
    compute_concreteness,
    compute_dual_sensory_tensor,
    evaluate_modality_alpha,
    extract_dual_channel_bakhtinian,
    extract_speaker,
)

OUT = (
    Path(__file__).resolve().parents[1]
    / "packages" / "usir-core" / "tests" / "fixtures" / "open_gates_parity.json"
)


def L(a):
    return [float(x) for x in np.asarray(a, dtype=float)]


# --- OPEN-01 -------------------------------------------------------------
V_RED = [0.9, 0.1, 0.2, 0.0, 0.1, 0.3, 0.2, 0.0]
V_SEC = [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0]
V_COLD = [0.1, 0.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0]

concreteness_cases = [
    {"surface": "石", "lemma": "", "sensory": [0.2, 0.7, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]},
    {"surface": "概念", "lemma": "", "sensory": [0.2, 0.7, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]},
    {"surface": "構造的", "lemma": "", "sensory": None},
    {"surface": "走った", "lemma": "走る", "sensory": []},
    {"surface": "光", "lemma": "", "sensory": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    {"surface": "状態", "lemma": "", "sensory": [1.0] * 8},
    {"surface": "冷たさ", "lemma": "冷たい", "sensory": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.8, 0.0]},
]
for c in concreteness_cases:
    sv = None if c["sensory"] is None else np.array(c["sensory"], dtype=float)
    c["expected"] = compute_concreteness(c["surface"], c["lemma"], sv)

# (c_u, v_pri, v_sec, u_vec, mimetic)
polysemy_cases = [
    {"c_u": 0.8, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": None, "mimetic": False},   # case1
    {"c_u": 0.2, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": None, "mimetic": False},   # case3
    {"c_u": 0.2, "v_pri": V_RED, "v_sec": None, "u_vec": None, "mimetic": False},    # case3 default sec
    {"c_u": 0.2, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": V_RED, "mimetic": False},  # case2
    {"c_u": 0.2, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": V_COLD, "mimetic": False}, # case3 (low cosine)
    {"c_u": 0.35, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": None, "mimetic": False},  # transition, secondary
    {"c_u": 0.40, "v_pri": V_COLD, "v_sec": None, "u_vec": [], "mimetic": False},    # transition, primary
    {"c_u": 0.1, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": None, "mimetic": True},    # mimetic immunity
    {"c_u": 1.7, "v_pri": V_COLD, "v_sec": V_SEC, "u_vec": None, "mimetic": False},  # clipped
    {"c_u": 0.45, "v_pri": V_RED, "v_sec": V_SEC, "u_vec": V_RED, "mimetic": False}, # boundary
]
for c in polysemy_cases:
    r = apply_open01_polysemy_damping(
        c["c_u"],
        np.array(c["v_pri"]),
        None if c["v_sec"] is None else np.array(c["v_sec"]),
        None if c["u_vec"] is None else np.array(c["u_vec"], dtype=float),
        c["mimetic"],
    )
    c["expected"] = {
        "case": r.case,
        "slot": r.slot,
        "effective_sensory": L(r.effective_sensory),
        "damping_factor": float(r.damping_factor),
        "metaphor_bonus": float(r.metaphor_bonus),
        "concreteness_u": float(r.concreteness_u),
    }

# --- OPEN-02 -------------------------------------------------------------
modality_cases = [
    {"tail": "赤かった", "neg": False, "spec": False},
    {"tail": "見えなかった", "neg": False, "spec": False},
    {"tail": "雨が降るだろう", "neg": False, "spec": False},
    {"tail": "雨の気配がした", "neg": False, "spec": False},
    {"tail": "静かだった", "neg": True, "spec": False},
    {"tail": "静かだった", "neg": False, "spec": True},
    {"tail": "ではないかもしれない", "neg": False, "spec": False},
    {"tail": "", "neg": False, "spec": False},
]
for c in modality_cases:
    c["expected_alpha"] = evaluate_modality_alpha(c["tail"], c["neg"], c["spec"])
    c["tensor"] = {}
    for key, vec in (("red", V_RED), ("cold", V_COLD)):
        t = compute_dual_sensory_tensor(np.array(vec), c["expected_alpha"])
        c["tensor"][key] = {
            "alpha_mod": float(t.alpha_mod),
            "s_plus": L(t.s_plus),
            "s_minus": float(t.s_minus),
            "s_minus_vec": L(t.s_minus_vec),
            "is_negated": bool(t.is_negated),
            "is_speculative": bool(t.is_speculative),
        }

# --- OPEN-04 -------------------------------------------------------------
speaker_cases = [
    {"before": "彼女は静かに頷いた。田中は", "after": "と言った"},
    {"before": "そのとき佐藤が笑って", "after": ""},
    {"before": "。鈴木って", "after": ""},
    {"before": "沈黙。", "after": "と山本は叫んだ。"},
    {"before": "沈黙。", "after": "、高橋が答えた。"},
    {"before": "しかしは", "after": "声がした"},
    {"before": "", "after": ""},
]
for c in speaker_cases:
    c["expected"] = extract_speaker(c["before"], c["after"])

bakhtin_texts = [
    # distinct voices: attributed speakers
    "田中は「俺は行くぜ！てめえも来いよ！」と叫んだ。佐藤は「わたしは参りますが、よろしいでしょうか。」と静かに言った。",
    # identical voices -> collapse
    "田中は「そうですね。行きましょう。」と言った。佐藤は「そうですね。行きましょう。」と言った。",
    # unattributed alternating default speakers
    "雨が降っていた。「寒いな」「そうだね」「帰ろうか」「うん」風が止んだ。",
    # single speaker, nested quote type
    "鈴木って『おい、何だよ！』と怒鳴った。夜は更けた。",
    # no dialogue
    "海は凪いでいた。雲が低く垂れ込めていた。",
    "",
    # three speakers with colloquial attribution and newline whitespace
    "山本は「あっ、えっ、うーん」\n高橋が「私は僕ではなくおれだ」と言った。 中村は「ゼロ」と答えた。",
    # astral-plane char before dialogue (code point offset check)
    "𠮷野は「ありがとうございます。」と言った。𠮷田は「いたします。」と述べた。",
]
bakhtin_cases = []
for t in bakhtin_texts:
    for thr in (0.15, 5.0):
        r = extract_dual_channel_bakhtinian(t, thr)
        bakhtin_cases.append({
            "text": t,
            "threshold": thr,
            "expected": {
                "narration_text": r.narration_text,
                "speaker_utterances": [[k, v] for k, v in r.speaker_utterances.items()],
                "speaker_vectors": [[k, L(v)] for k, v in r.speaker_vectors.items()],
                "min_voice_distance": float(r.min_voice_distance),
                "dialogic_collapse_detected": bool(r.dialogic_collapse_detected),
                "pairwise_distances": [[a, b, float(d)] for (a, b), d in r.pairwise_distances.items()],
            },
        })

OUT.write_text(
    json.dumps(
        {
            "concreteness": concreteness_cases,
            "polysemy": polysemy_cases,
            "modality": modality_cases,
            "speaker": speaker_cases,
            "bakhtin": bakhtin_cases,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print(f"wrote {OUT}")
