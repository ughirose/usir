"""Phase 21: scene_beats per-paragraph prompting, style_genome steering target, genome ΔLL metric."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from usir_py.genome_fit import delta_ll, genome_centroid, genome_loglik
from usir_py.grpo_reward import PHASE_CENTROIDS_24D
from usir_py.hsmm_longform_director import HSMMLongformDirector, beat_for_paragraph
from usir_py.realtime_decoder import AdaptiveBAOABIntegrator, ClauseTwistedSMCDecoder

ROOT = Path(__file__).resolve().parents[2]
GENOME_A = ROOT / "artifacts" / "style_atlas" / "000879.usir"
GENOME_B = ROOT / "artifacts" / "style_atlas" / "000035.usir"

BEATS = ["発端", "展開", "緊迫"]


def _load_runner():
    path = ROOT / "python" / "colab" / "run_v4_bidirectional_experiment.py"
    spec = importlib.util.spec_from_file_location("run_v4_p21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spy_decoder(seen):
    dec = ClauseTwistedSMCDecoder(integrator=AdaptiveBAOABIntegrator(dim=12, seed=0), seed=0, chat_prefix="BASE")
    orig = dec.generate_with_clause_twisted_smc

    def spy(*a, **k):
        seen.append({"chat_prefix": dec.chat_prefix, "mu": np.array(dec.integrator.mu_star, dtype=float)})
        return orig(*a, **{**k, "max_clauses": 2})

    dec.generate_with_clause_twisted_smc = spy
    return dec


def test_beat_for_paragraph_clamps():
    assert beat_for_paragraph(None, 3) is None
    assert beat_for_paragraph([], 0) is None
    assert [beat_for_paragraph(BEATS, k) for k in range(5)] == ["発端", "展開", "緊迫", "緊迫", "緊迫"]


@pytest.mark.parametrize("carry", [False, True])
def test_director_beat_k_reaches_paragraph_k(carry):
    seen = []
    res = HSMMLongformDirector(seed=0).generate_longform(
        target_paragraphs=6, decoder=_spy_decoder(seen), carry_context=carry,
        scene_beats=BEATS, roles=["青年"], paragraph_prompt_fn=lambda k: f"PROMPT{k}|{BEATS[min(k, 2)]}|",
    )
    assert len(seen) == 6
    for k, s in enumerate(seen):
        assert s["chat_prefix"].startswith(f"PROMPT{k}|{BEATS[min(k, 2)]}|")
        if not carry:
            assert s["chat_prefix"] == f"PROMPT{k}|{BEATS[min(k, 2)]}|"
    assert [p["scene_beat"] for p in res["paragraphs"]] == ["発端", "展開", "緊迫", "緊迫", "緊迫", "緊迫"]
    assert res["roles"] == ["青年"] and res["scene_beats"] == BEATS
    # beat never leaks into the narrated text
    assert all("発端" not in p["text"] and "PROMPT" not in p["text"] for p in res["paragraphs"])


def test_director_default_unchanged():
    seen = []
    res = HSMMLongformDirector(seed=0).generate_longform(target_paragraphs=6, decoder=_spy_decoder(seen))
    assert all(s["chat_prefix"] == "BASE" for s in seen)
    assert all(p["scene_beat"] is None for p in res["paragraphs"])
    assert res["scene_beats"] is None and res["roles"] is None
    plain = HSMMLongformDirector(seed=3).generate_longform(target_paragraphs=6)
    with_beats = HSMMLongformDirector(seed=3).generate_longform(target_paragraphs=6, scene_beats=BEATS)
    assert plain["full_text"] == with_beats["full_text"]  # procedural path: beats are metadata only
    assert [p["scene_beat"] for p in with_beats["paragraphs"]][:3] == BEATS


class _FakeTok:
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"


def _run_runner(monkeypatch, **kw):
    mod = _load_runner()
    import usir_py.realtime_decoder as rd

    seen = []
    real = rd.ClauseTwistedSMCDecoder

    class NoModelDecoder(real):
        def __init__(self, *a, **k):
            k["model"] = None
            super().__init__(*a, **k)

            orig = self.generate_with_clause_twisted_smc

            def spy(*aa, **kk):
                seen.append({"chat_prefix": self.chat_prefix, "mu": np.array(self.integrator.mu_star, dtype=float)})
                return orig(*aa, **{**kk, "max_clauses": 2})

            self.generate_with_clause_twisted_smc = spy

    monkeypatch.setattr(rd, "ClauseTwistedSMCDecoder", NoModelDecoder)
    res = mod.run_longform_director("止まった懐中時計をめぐる短編", model=object(), tokenizer=_FakeTok(), seed=3,
                                    target_paragraphs=6, carry_context=False, **kw)
    return res, seen


def test_runner_scene_beats_and_roles(monkeypatch):
    beats = ["発端：A", "展開：B", "緊迫：C"]
    res, seen = _run_runner(monkeypatch, scene_beats=beats, roles=["青年", "老職人"])
    assert len(seen) == 6
    for k, s in enumerate(seen):
        b = beats[min(k, 2)]
        assert f"第{k + 1}段落の筋: {b}。" in s["chat_prefix"]
        assert "登場人物は役割名（青年、老職人）で呼び、固有名詞を作らない。" in s["chat_prefix"]
    assert [p["beat"] for p in res["paragraphs"]] == [beats[min(k, 2)] for k in range(6)]
    assert all("第1段落の筋" not in p["text"] for p in res["paragraphs"])

    base, seen0 = _run_runner(monkeypatch)
    assert all("段落の筋" not in s["chat_prefix"] and "役割名" not in s["chat_prefix"] for s in seen0)
    assert len({s["chat_prefix"] for s in seen0}) == 1
    assert all(p["beat"] is None for p in base["paragraphs"])


def test_runner_style_genome_translates_wells(monkeypatch):
    base, seen0 = _run_runner(monkeypatch)
    res, seen1 = _run_runner(monkeypatch, style_genome=GENOME_A)
    assert res["style_genome"] is True and base["style_genome"] is False
    shift = (genome_centroid(GENOME_A) - np.asarray(PHASE_CENTROIDS_24D["II_Critical"], dtype=float))[:12]
    assert res["genome_shift_norm"] == pytest.approx(float(np.linalg.norm(shift)))
    assert res["genome_shift_norm"] > 0
    assert res["wells"] == base["wells"]  # same seed -> same well sequence
    for a, b in zip(seen0, seen1):
        assert np.allclose(b["mu"] - a["mu"], shift)
    # bytes input is accepted as well
    res_b, _ = _run_runner(monkeypatch, style_genome=GENOME_A.read_bytes(), scene_beats=BEATS, roles=["青年"])
    assert res_b["style_genome"] is True


def test_genome_centroid_is_weighted_mean():
    from usir_py.usir_forge import load_tier_m_binary

    d = load_tier_m_binary(GENOME_A.read_bytes())
    w = np.array(d["weights"], dtype=float)
    c = np.array(d["centroids"], dtype=float)
    m = genome_centroid(GENOME_A.read_bytes())
    assert m.shape == (24,)
    assert np.allclose(m, (w[w > 1e-9, None] * c[w > 1e-9]).sum(0) / w[w > 1e-9].sum())


_LONG = "夜の工房に雨が降っていた。青年は黙って歯車を磨いた。" * 60
_SHORT = "青年は黙って歯車を磨いた。雨が降っていた。"


def test_genome_loglik_windows_and_short_text():
    a = GENOME_A.read_bytes()
    long_res = genome_loglik(_LONG, a)
    assert long_res["n_windows"] >= 2 and np.isfinite(long_res["mean_ll"])
    short_res = genome_loglik(_SHORT, a)
    assert short_res["n_windows"] == 1 and np.isfinite(short_res["mean_ll"])
    empty = genome_loglik("", a)
    assert empty["n_windows"] == 0 and np.isnan(empty["mean_ll"])


def test_delta_ll_antisymmetric_and_path_input():
    a, b = GENOME_A.read_bytes(), GENOME_B.read_bytes()
    d_ab = delta_ll(_LONG, a, b)
    assert d_ab == pytest.approx(genome_loglik(_LONG, a)["mean_ll"] - genome_loglik(_LONG, b)["mean_ll"])
    assert delta_ll(_LONG, b, a) == pytest.approx(-d_ab)
    assert delta_ll(_LONG, a, a) == pytest.approx(0.0)
    assert delta_ll(_LONG, GENOME_A, GENOME_B) == pytest.approx(d_ab)
