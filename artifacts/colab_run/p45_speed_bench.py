"""Phase 45 speed benchmark (Colab L4): decode throughput vs batch size, model size and HF decoding options.

Measures wall time of model.generate with a fixed number of new tokens (min_new_tokens == max_new_tokens), sampled,
on a rendered literary scene prompt.  No quality metric is computed here.  Prints one ARCHIVE_B64 line.
Run: python scripts/run_colab_file.py artifacts/colab_run/p45_speed_bench.py docs/handoff/v10 --gpu L4 --timeout 3600
"""
import base64, gc, gzip, json, subprocess, sys, time, traceback, zlib

T0 = time.time()
NEW = 256
PROMPT = "夏の午後、川辺で釣りをする老人を描いてください。地の文を中心に、会話は少なめにして、800字程度の小説の一場面として書いてください。"
PLAN = [
    ("Qwen/Qwen3-8B", [1, 12, 24, 48], True),
    ("Qwen/Qwen3-4B", [1, 12, 48], False),
    ("Qwen/Qwen3-1.7B", [1, 12, 48], False),
]
res = {"schema": "p45_speed_bench_v1", "new_tokens": NEW, "runs": [], "errors": []}


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

res["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__}


def bench(model, tok, B, label, **extra):
    msgs = [{"role": "user", "content": PROMPT}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tok.padding_side = "left"
    enc = tok([text] * B, return_tensors="pt", padding=True).to("cuda")
    kw = dict(max_new_tokens=NEW, min_new_tokens=NEW, do_sample=True, temperature=0.8, top_p=0.95,
              pad_token_id=tok.pad_token_id or tok.eos_token_id, **extra)
    torch.manual_seed(0)
    with torch.no_grad():
        model.generate(**enc, **dict(kw, max_new_tokens=8, min_new_tokens=8))  # warm-up
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.no_grad():
        model.generate(**enc, **kw)
    torch.cuda.synchronize()
    w = time.perf_counter() - t
    r = {"label": label, "batch": B, "wall_s": w, "ms_per_step": 1000 * w / NEW, "tok_per_s": B * NEW / w,
         "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    log(json.dumps(r))
    return r


for mid, batches, try_compile in PLAN:
    t = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
        load_s = time.time() - t
        log(f"loaded {mid} in {load_s:.0f}s")
        for B in batches:
            try:
                r = bench(model, tok, B, "eager")
                r.update(model=mid, load_s=load_s)
                res["runs"].append(r)
            except Exception as e:
                res["errors"].append({"model": mid, "batch": B, "label": "eager", "error": repr(e)[:400]})
                log(f"ERROR {mid} B={B}: {e!r}"[:300]); torch.cuda.empty_cache()
        if try_compile:
            for B in (1, 12):
                try:
                    t_c = time.time()
                    r = bench(model, tok, B, "static_cache", cache_implementation="static")
                    r.update(model=mid, setup_s=time.time() - t_c)
                    res["runs"].append(r)
                except Exception as e:
                    res["errors"].append({"model": mid, "batch": B, "label": "static_cache", "error": repr(e)[:400]})
                    log(f"ERROR static {mid} B={B}: {e!r}"[:300]); torch.cuda.empty_cache()
            try:
                model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
                t_c = time.time()
                r = bench(model, tok, 12, "static_cache+compile", cache_implementation="static")
                r.update(model=mid, setup_s=time.time() - t_c)
                res["runs"].append(r)
            except Exception as e:
                res["errors"].append({"model": mid, "batch": 12, "label": "static_cache+compile", "error": repr(e)[:400]})
                log(f"ERROR compile {mid}: {e!r}"[:300])
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        res["errors"].append({"model": mid, "stage": "load", "error": repr(e)[:400], "tb": traceback.format_exc()[-800:]})
        log(f"ERROR load {mid}: {e!r}"[:300])

res["total_s"] = time.time() - T0
raw = json.dumps(res, ensure_ascii=False).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw)).decode())
