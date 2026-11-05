"""Phase 32 Colab job generator: logit-debiased pairwise judge over the Phase 31 outputs (+ Phase 27 "min" texts).

Cross-judging (no model scores its own text): Swallow-8B judges Qwen3-8B outputs, Qwen3-8B judges Swallow outputs.
Per (plot, seed) group, pairs: new_ctrl vs min (seeds 3 and 42 only; min comes from Phase 27, same plots and seeds),
gA vs new_ctrl, gB vs new_ctrl, gC vs new_ctrl, gA vs gB. Each pair: whole text (both orders) and the six
paragraph-aligned pairs (both orders). One forward pass per prompt; ln P("1") - ln P("2") at the answer slot.

Run: python scripts/run_colab_file.py artifacts/colab_run/p32j_cross.py docs/handoff/v10 --gpu A100 --timeout 5400
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
V9, V10 = ROOT / "docs" / "handoff" / "v9", ROOT / "docs" / "handoff" / "v10"

MODELS = {"qwen3_8b": "Qwen/Qwen3-8B", "swallow_8b": "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2"}
GENERATED = {"qwen3_8b": ("p31_q3_8b.json", "p27_q3_8b.json", "swallow_8b"),
             "swallow_8b": ("p31_swallow_8b.json", "p27_swallow_8b.json", "qwen3_8b")}
PAIRS = [("new_ctrl", "min"), ("gA", "new_ctrl"), ("gB", "new_ctrl"), ("gC", "new_ctrl"), ("gA", "gB")]


def collect() -> dict:
    out = {}
    for gen, (f31, f27, judge) in GENERATED.items():
        p31, p27 = V10 / f31, V9 / f27
        if not p31.exists():
            print("missing", p31)
            continue
        groups = {}
        for r in json.loads(p31.read_text(encoding="utf-8"))["runs"].values():
            if r.get("error"):
                continue
            g = groups.setdefault(f"{r['plot']}_s{r['seed']}", {"text": {}, "paras": {}})
            g["text"][r["condition"]] = r.get("formatted_text") or r["full_text"]
            g["paras"][r["condition"]] = r["paragraph_texts"]
        for r in json.loads(p27.read_text(encoding="utf-8"))["runs"].values():
            k = f"{r['plot']}_s{r['seed']}"
            if r["condition"] == "min" and not r.get("error") and k in groups:
                groups[k]["text"]["min"] = r.get("formatted_text") or r["full_text"]
                groups[k]["paras"]["min"] = r["paragraph_texts"]
        out[gen] = {"judge": judge, "groups": groups}
    return out


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(PY / "usir_py" / "pairwise_judge.py", "usir_py/pairwise_judge.py")
        z.writestr("usir_py/__init__.py", "")
    return base64.b64encode(buf.getvalue()).decode()


BODY = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, threading, gc
os.makedirs("/content/j/python", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/j/python")
sys.path.insert(0, "/content/j/python")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=False)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from usir_py.pairwise_judge import logit_tournament, LOGIT_ANSWER_STEM, DEFAULT_AXES

_alive = True
def _hb():
    t0 = time.time()
    while _alive:
        time.sleep(15); sys.stdout.write(f"hb {time.time()-t0:.0f}s\n"); sys.stdout.flush()
threading.Thread(target=_hb, daemon=True).start()
def log(m):
    sys.stdout.write(m + "\n"); sys.stdout.flush()

out = {"gpu": torch.cuda.get_device_name(0), "pairs": PAIRS, "criteria": list(DEFAULT_AXES), "results": {}}
t_all = time.time()
for gen, spec in DATA.items():
    mid = MODELS[spec["judge"]]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    ids12 = [tok.encode(x, add_special_tokens=False) for x in ("1", "2")]
    assert all(len(i) == 1 for i in ids12), ids12
    i1, i2 = ids12[0][0], ids12[1][0]
    log(f"judge {mid} loaded in {time.time()-t0:.0f}s; ids 1/2 = {i1}/{i2}")
    n_calls = [0]; t_fwd = [0.0]; mass = []
    def score_fn(user_prompt):
        msgs = [{"role": "user", "content": user_prompt}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(s + LOGIT_ANSWER_STEM, return_tensors="pt", add_special_tokens=False).to("cuda")
        t1 = time.time()
        with torch.no_grad():
            lg = model(**enc).logits[0, -1].float()
        t_fwd[0] += time.time() - t1; n_calls[0] += 1
        lp = torch.log_softmax(lg, -1)
        mass.append(float(lp[i1].exp() + lp[i2].exp()))
        return float(lp[i1]), float(lp[i2])
    res = {}
    for gkey, g in spec["groups"].items():
        pairs = [(a, b) for a, b in PAIRS if a in g["text"] and b in g["text"]]
        res[gkey] = logit_tournament(g["text"], pairs, score_fn, paragraphs=g["paras"])
        log(f"{gen} {gkey}: " + " ".join(f"{m['a']}>{m['b']}={m['full']['preference']:+.2f}/{m['paragraph']['preference_mean']:+.2f}" for m in res[gkey]["matches"]))
    out["results"][gen] = {"judge": mid, "groups": res, "n_calls": n_calls[0], "fwd_sec": round(t_fwd[0], 1),
                           "answer_mass_12": {"mean": sum(mass) / len(mass), "min": min(mass)}}
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
    out = HERE / "p32j_cross.py"
    out.write_text(header + BODY, encoding="utf-8")
    print(out, out.stat().st_size, {k: len(v["groups"]) for k, v in data.items()})


if __name__ == "__main__":
    main()
