"""Builds a self-contained Colab benchmark script (zip of usir_py embedded as base64).

Usage: python build_colab_bench.py <vanilla|smc>
Written via UTF-8-safe tooling; never pipe Japanese source through a Windows console.
Each mode must finish well under the ~160 s connection window observed for colab-exec.
"""
import base64, io, pathlib, sys, zipfile

MODE = sys.argv[1] if len(sys.argv) > 1 else "vanilla"
ROOT = pathlib.Path(__file__).resolve().parents[2] / "python"
OUT = pathlib.Path(__file__).resolve().parent / f"colab_bench_{MODE}.py"

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in (ROOT / "usir_py").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".bin", ".json"):
            z.write(p, p.relative_to(ROOT).as_posix())
b64 = base64.b64encode(buf.getvalue()).decode()

COMMON = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, re
import numpy as np
os.makedirs("/content/usir/python", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/usir/python")
sys.path.insert(0, "/content/usir/python")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate", "janome", "scipy"], check=True)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from usir_py.extractor_24d import extract_usir_24d

PROMPT = "静かな港町の夜明けを、約600字の短い小説の一節として書いてください。"
assert "港町" in PROMPT and "?" not in PROMPT, "prompt corrupted"
PUNCT = re.compile("[、。！？!?]")
assert PUNCT.search("あ。"), "regex corrupted"

MID = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="cuda")
chat = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)

def metrics(txt):
    t0 = time.time()
    ex = extract_usir_24d(txt)
    sents = [s for s in re.split("[。！？]", txt) if s.strip()]
    return dict(chars=len(txt), n_sent=len(sents), len_cv=round(float(ex["len_cv"]), 3),
                extract_sec=round(time.time() - t0, 2),
                order={k: round(float(v), 3) for k, v in ex["order_params"].items()},
                head=txt[:80], tail=txt[-40:])

def vanilla(seed, **kw):
    torch.manual_seed(seed)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).to("cuda")
    t = time.time()
    out = model.generate(**ids, max_new_tokens=700, pad_token_id=tok.eos_token_id, **kw)
    txt = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    d = metrics(txt)
    d["gen_sec"] = round(time.time() - t, 1)
    return d
'''

VANILLA = r'''
res = {"vanilla_greedy": vanilla(0, do_sample=False),
       "vanilla_sample_s0": vanilla(0, do_sample=True, temperature=0.8, top_p=0.95),
       "vanilla_sample_s1": vanilla(1, do_sample=True, temperature=0.8, top_p=0.95)}
print("RESULT_JSON", json.dumps(res, ensure_ascii=True, default=float))
'''

SMC = r'''
from usir_py.grpo_reward import PHASE_CENTROIDS_24D
from usir_py.realtime_decoder import ClauseTwistedSMCDecoder, AdaptiveBAOABIntegrator
NCL = 8
STAT = {"zero_logprob": 0, "logprob_calls": 0, "gen_calls": 0, "forced_comma": 0, "gen_sec": 0.0, "score_sec": 0.0, "feat_sec": 0.0}

def gen_clause(prefix, p, temp):
    t = time.time(); STAT["gen_calls"] += 1
    ids = tok(chat + prefix, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**ids, max_new_tokens=20, do_sample=True, temperature=temp, top_p=0.9, pad_token_id=tok.eos_token_id)
    s = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    STAT["gen_sec"] += time.time() - t
    m = PUNCT.search(s)
    if m:
        return s[:m.end()]
    STAT["forced_comma"] += 1
    return s.strip() + "、"

class Dec(ClauseTwistedSMCDecoder):
    def _compute_hf_logprob(self, context, target):
        t = time.time(); STAT["logprob_calls"] += 1
        v = super()._compute_hf_logprob(chat + context, target)
        STAT["score_sec"] += time.time() - t
        if v == 0.0:
            STAT["zero_logprob"] += 1
        return v

def timed_feat(text):
    t = time.time()
    from usir_py.extractor_24d import extract_usir_24d as e
    q = np.asarray(e(text)["q"], dtype=float)
    STAT["feat_sec"] += time.time() - t
    return q

mu = np.asarray(PHASE_CENTROIDS_24D["II_Critical"][:12], dtype=float)
integ = AdaptiveBAOABIntegrator(dim=12, mu_star=mu, seed=0)
dec = Dec(integrator=integ, model=model, tokenizer=tok, seed=0, feature_extractor=timed_feat)
t = time.time()
r = dec.generate_with_clause_twisted_smc("", max_clauses=NCL, candidate_generator=gen_clause, early_stop_on_punct=False)
wall = round(time.time() - t, 1)
d = metrics(r["text"])
d["wall_sec"] = wall
d["n_clauses"] = r["n_clauses"]
d["stat"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in STAT.items()}
d["mean_ess"] = round(float(np.mean([float(x["ess"]) for x in r["telemetry"]])), 2)
d["dup_clause_ratio"] = round(1 - len(set(r["clauses"])) / max(1, len(r["clauses"])), 3)
print("RESULT_JSON", json.dumps({"usir_smc": d}, ensure_ascii=True, default=float))
'''

body = COMMON + (VANILLA if MODE == "vanilla" else SMC)
OUT.write_text("B64 = " + repr(b64) + "\n" + body, encoding="utf-8")
assert "港町" in OUT.read_text(encoding="utf-8"), "Japanese prompt missing from generated script"
print("built", OUT.name, OUT.stat().st_size, "bytes; japanese ok")
