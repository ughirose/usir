"""Phase 28.1 Colab job generator: blind pairwise judge (order swap) over Phase 27 outputs.

Cross-judging: Qwen3-Swallow-8B judges the Qwen3-8B outputs and Qwen3-8B judges the Swallow outputs, so no model
scores its own text. Both judges share the Qwen3 base, so a family-level preference is not excluded.
Pairs per (plot, seed): new_ctrl vs min, gA_scaled vs new_ctrl, gB_scaled vs new_ctrl, gA_scaled vs gA_legacy.
Each pair is judged in both orders (python/usir_py/pairwise_judge.py), greedy decoding, thinking disabled.

Run: python scripts/run_colab_file.py artifacts/colab_run/p28j_cross.py docs/handoff/v9 --gpu L4 --timeout 5400
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "python"
V9 = ROOT / "docs" / "handoff" / "v9"

MODELS = {"qwen3_8b": "Qwen/Qwen3-8B", "swallow_8b": "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2"}
# generator tag -> (result json, judge tag)
GENERATED = {"qwen3_8b": ("p27_q3_8b.json", "swallow_8b"), "swallow_8b": ("p27_swallow_8b.json", "qwen3_8b")}
PAIRS = [("new_ctrl", "min"), ("gA_scaled", "new_ctrl"), ("gB_scaled", "new_ctrl"), ("gA_scaled", "gA_legacy")]


def collect() -> dict:
    out = {}
    for gen, (fname, judge) in GENERATED.items():
        p = V9 / fname
        if not p.exists():
            print("missing", p)
            continue
        runs = json.loads(p.read_text(encoding="utf-8"))["runs"]
        groups = {}
        for r in runs.values():
            if r.get("error"):
                continue
            g = groups.setdefault(f"{r['plot']}_s{r['seed']}", {})
            g[r["condition"]] = r.get("formatted_text") or r["full_text"]
        out[gen] = {"judge": judge, "groups": groups}
    return out


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(PY / "usir_py" / "pairwise_judge.py", "usir_py/pairwise_judge.py")
        z.writestr("usir_py/__init__.py", "")
    return base64.b64encode(buf.getvalue()).decode()


BODY = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, threading, re, gc
os.makedirs("/content/j/python", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/j/python")
sys.path.insert(0, "/content/j/python")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=False)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from usir_py.pairwise_judge import tournament, DEFAULT_AXES

_alive = True
def _hb():
    t0 = time.time()
    while _alive:
        time.sleep(15); sys.stdout.write(f"hb {time.time()-t0:.0f}s\n"); sys.stdout.flush()
threading.Thread(target=_hb, daemon=True).start()
def log(m):
    sys.stdout.write(m + "\n"); sys.stdout.flush()

THINK = re.compile(r"<think>.*?</think>", re.S)
out = {"gpu": torch.cuda.get_device_name(0), "pairs": PAIRS, "axes": list(DEFAULT_AXES), "results": {}}
t_all = time.time()
for gen, spec in DATA.items():
    jtag = spec["judge"]; mid = MODELS[jtag]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    log(f"judge {mid} loaded in {time.time()-t0:.0f}s for generator {gen}")
    n_calls = [0]; t_gen = [0.0]
    def judge_fn(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).to("cuda")
        t1 = time.time()
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        t_gen[0] += time.time() - t1; n_calls[0] += 1
        return THINK.sub("", tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)).strip()
    res = {}
    for gkey, texts in spec["groups"].items():
        pairs = [(a, b) for a, b in PAIRS if a in texts and b in texts]
        res[gkey] = tournament(texts, pairs, judge_fn)
        log(f"{gen} {gkey}: {json.dumps(res[gkey]['summary'], ensure_ascii=False)} t={time.time()-t_all:.0f}s")
    out["results"][gen] = {"judge": mid, "groups": res, "n_calls": n_calls[0], "judge_gen_sec": round(t_gen[0], 1)}
    del model; gc.collect(); torch.cuda.empty_cache()
out["wall_sec"] = round(time.time() - t_all, 1)
_alive = False
import gzip, zlib
raw = json.dumps(out, ensure_ascii=False, default=float).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
'''


def main() -> None:
    data = collect()
    header = ("B64 = " + repr(build_zip_b64()) + "\n" + f"MODELS = {MODELS!r}\nPAIRS = {PAIRS!r}\n"
              + "DATA = " + repr(data) + "\n")
    out = HERE / "p28j_cross.py"
    out.write_text(header + BODY, encoding="utf-8")
    print(out, out.stat().st_size, {k: len(v["groups"]) for k, v in data.items()})


if __name__ == "__main__":
    main()
