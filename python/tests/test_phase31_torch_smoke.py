"""Phase 31 smoke test on a tiny random Qwen2: the HF generate path with forbidden words, year ban, kanji bias,
genome proposal and the Fisher term runs end to end.  Skipped without torch (runs in a CPU torch venv / Colab)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from usir_py.fisher_atlas import fit_fisher  # noqa: E402
from usir_py.realtime_decoder import ClauseTwistedSMCDecoder  # noqa: E402

CHARS = "。、少年女青工房時計雨の夜にはがをでたる一九六八古い針音"


class CharTok:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def __len__(self):
        return 2 + len(CHARS)

    def encode(self, text, add_special_tokens=False):
        return [2 + CHARS.index(c) for c in text if c in CHARS]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(CHARS[int(i) - 2] for i in ids if int(i) >= 2)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [self.decode(s) for s in seqs]

    def convert_ids_to_tokens(self, i):
        return self.decode([i])


def test_generate_with_phase31_flags():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(CharTok()), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    rng = np.random.default_rng(0)
    X = {"a": rng.normal(0, 1, (100, 51)), "b": rng.normal(0, 1, (100, 51))}
    X["a"][:, 24] += 2.0
    fm = fit_fisher(X)
    d = ClauseTwistedSMCDecoder(
        dim=12, seed=0, model=model, tokenizer=CharTok(), chat_prefix="序", rhythm=True, rhythm_profile="steady",
        kv_share=True, forbidden_words=["少年"], forbid_years=True,
        genome_proposal={"steady_min_chars": 8, "steady_max_chars": 30, "steady_max_commas": 3, "comma_target": 2.0,
                         "kanji_bias": 0.4},
        fisher_metric=fm, fisher_author="a", fisher_gamma=1.0)
    out = d.generate_with_clause_twisted_smc("雨の夜に少", max_clauses=4)
    text = out.get("text") or out.get("generated_text") or ""
    full = "雨の夜に少" + text
    assert "少年" not in full
    assert "一九六八" not in full
    assert d.stats.get("fisher_steps", 0) >= 1
