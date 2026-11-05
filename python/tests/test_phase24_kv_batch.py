"""Phase 24: batched candidate scoring and shared prefix KV cache must match the one-by-one log-likelihood.

Uses a tiny randomly initialised Qwen2 model (no download). Skipped when torch / transformers are not installed
(the local dev env has no torch; these run in the Colab job together with the timing run)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from usir_py.realtime_decoder import ClauseTwistedSMCDecoder  # noqa: E402


class CharTok:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 60) for c in text]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(0x3041 + int(i)) for i in ids)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [self.decode(s) for s in seqs]


def _tiny(dtype):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return transformers.Qwen2ForCausalLM(cfg).eval().to(dtype)


def _dec(model, **kw):
    return ClauseTwistedSMCDecoder(dim=12, seed=0, model=model, tokenizer=CharTok(), chat_prefix="序章：", **kw)


CTX = "工房の奥で青年は時計を見つめていた。"
TARGETS = ["針が止まった。", "雨が降る、", "老職人は黙っていた。", "光"]


# bf16: the one-by-one reference takes log_softmax of bf16 logits, so its values are quantised to the bf16 spacing
# (0.03125 at |x| ~ 4); the batched path computes log_softmax in fp32. Measured max diff on Colab: 0.016.
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.bfloat16, 0.035)])
@pytest.mark.parametrize("kv", [False, True])
def test_batched_logprob_matches_one_by_one(dtype, atol, kv):
    m = _tiny(dtype)
    single = _dec(m)
    ref = [single._compute_hf_logprob(CTX, t) for t in TARGETS]
    batched = _dec(m, kv_share=kv, batch_score=True)._compute_hf_logprob_batch(CTX, TARGETS)
    assert np.allclose(batched, ref, atol=atol), (batched, ref)


def test_kv_path_is_used_and_memoised():
    m = _tiny(torch.float32)
    d = _dec(m, kv_share=True)
    d._compute_hf_logprob_batch(CTX, TARGETS)
    key1 = d._pkv[0]
    d._compute_hf_logprob_batch(CTX, TARGETS[:2])
    assert d._pkv[0] == key1                       # same context -> cache reused, not recomputed
    assert d.stats.get("kv_fallbacks", 0) == 0


def test_generate_with_cropped_repeated_cache_matches_plain_generate():
    m = _tiny(torch.float32)
    d = _dec(m, kv_share=True)
    ids = torch.tensor([CharTok().encode("序章：" + CTX)], dtype=torch.long)
    P, L = 4, ids.shape[1]
    pk = d._prefix_kv(ids[0].tolist())
    cache = d._clone_kv(pk[0], P, crop_to=L - 1)
    assert cache is not None, d.stats.get("kv_error")
    inp = ids.expand(P, -1).contiguous()
    kw = dict(max_new_tokens=5, do_sample=False, attention_mask=torch.ones_like(inp), pad_token_id=0)
    with torch.no_grad():
        plain = m.generate(inp, **kw)
        cached = m.generate(inp, past_key_values=cache, **kw)
    assert torch.equal(plain, cached)
    assert d.stats.get("kv_fallbacks", 0) == 0


def test_clone_kv_falls_back_on_unsupported_cache():
    d = _dec(_tiny(torch.float32), kv_share=True)
    assert d._clone_kv(object(), 4, crop_to=3) is None
    assert d.stats["kv_fallbacks"] == 1


def test_particle_weights_identical_with_batch_scoring():
    m = _tiny(torch.float32)
    a = _dec(m).compute_particle_weights(CTX, TARGETS)
    b = _dec(m, kv_share=True).compute_particle_weights(CTX, TARGETS)
    assert np.allclose(np.asarray(a["term1_llm"]), np.asarray(b["term1_llm"]), atol=1e-5)
    assert np.allclose(np.asarray(a["weights"]), np.asarray(b["weights"]), atol=1e-5)
