"""Phase 23: vanilla (no USIR processors) Japanese short-story bake-off across candidate models on Colab L4.

Same 6 scene beats / roles as Phase 21. Paragraph-by-paragraph generation, previous text carried in the prompt.
Sampling temperature 0.7, top_p 0.9, seeds from SEEDS. After generation, every text is scored by two scorers
from different families (mean NLL per character, lower = more fluent to that scorer) to reduce family bias.
MODELS is set by the launcher (two scripts split the list to stay under the runtime limit).

Run: python scripts/run_colab_file.py <generated script> docs/handoff/v8 --gpu L4 --timeout 4500
"""
import base64, gc, gzip, json, re, sys, time, zlib, subprocess, traceback

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = __MODELS__
SCORERS = ["Qwen/Qwen3-8B", "llm-jp/llm-jp-4-8b-instruct"]
SEEDS = [3, 42]
THEME = "止まった懐中時計をめぐる短編"
ROLES = "青年、老職人"
BEATS = [
    "発端：時計職人の青年が、工房に持ち込まれた止まった懐中時計の不可解な傷に気づく",
    "展開：青年は修理の手を止め、かつて工房を去った師である老職人の記憶をたどり、静かな葛藤を抱く",
    "緊迫：嵐が近づく夜、封じられていた工房の奥の引き出しに異変が起こる",
    "転換：歯車が噛み合った瞬間の閃光の中で、青年は時計が誰のために作られたかを悟る",
    "余韻：夜明けの光が工房に差し、見慣れた道具や街並みが違って見える",
    "結末：再び刻み始めた針の音を聞きながら、青年は時計を届けるために旅立つ",
]
SYS = ("あなたは小説家です。三人称・常体の小説本文だけを書きます。解説、要約、見出し、括弧書き、箇条書きは書きません。"
       "登場人物は指示された役割名でのみ呼び、人名や固有名詞を作りません。")


def _load(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()
    return tok, model


def _prompt(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _clean(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"^\s*(第.段落|段落.)[：:]?.*\n", "", t)
    return t.strip()


def generate_story(tok, model, seed):
    torch.manual_seed(seed)
    paras = []
    for k, beat in enumerate(BEATS):
        prev = "\n".join(paras)[-600:]
        user = (f"題材：{THEME}\n登場人物（役割名）：{ROLES}\n第{k + 1}段落（全6段落）の筋：{beat}\n"
                + (f"直前までの本文：\n{prev}\n" if prev else "")
                + "この筋に沿って、続きの段落の本文だけを350字程度で書いてください。")
        p = _prompt(tok, [{"role": "system", "content": SYS}, {"role": "user", "content": user}])
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=700, do_sample=True, temperature=0.7, top_p=0.9,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        paras.append(_clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)))
    return paras


def stats(paras):
    t = "".join(paras)
    sents = [s for s in re.split("(?<=[。！？])", t) if s.strip()]
    L = [len(s) for s in sents]
    mu = sum(L) / max(len(L), 1)
    sd = (sum((x - mu) ** 2 for x in L) / max(len(L), 1)) ** 0.5
    bg = [t[i:i + 2] for i in range(len(t) - 1)]
    return {"chars": len(t), "sentences": len(L), "mean_sent_len": round(mu, 1), "cv_l": round(sd / mu, 3) if mu else 0,
            "comma_per_100": round(100 * t.count("、") / max(len(t), 1), 2),
            "distinct_bigram": round(len(set(bg)) / max(len(bg), 1), 3),
            "polite": len(re.findall("(です|ます|でした|ました)[。、]", t)),
            "latin_or_hanzi_suspect": len(re.findall(r"[A-Za-z]{3,}", t))}


res = {"models": MODELS, "scorers": SCORERS, "seeds": SEEDS, "beats": BEATS, "runs": [], "errors": []}
t0 = time.time()
for name in MODELS:
    try:
        tl = time.time()
        tok, model = _load(name)
        load_s = time.time() - tl
        for s in SEEDS:
            tg = time.time()
            paras = generate_story(tok, model, s)
            res["runs"].append({"model": name, "seed": s, "paragraphs": paras, "stats": stats(paras),
                                "gen_sec": round(time.time() - tg, 1), "load_sec": round(load_s, 1)})
            print(f"[gen] {name} s{s} {res['runs'][-1]['stats']} {time.time() - tg:.0f}s", flush=True)
    except Exception as e:
        res["errors"].append({"model": name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]})
        print(f"[err] {name} {e}", flush=True)
    finally:
        model = None; tok = None; gc.collect(); torch.cuda.empty_cache()

for sname in SCORERS:
    try:
        tok, model = _load(sname)
        for r in res["runs"]:
            text = "\n".join(r["paragraphs"])
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :4000].to(model.device)
            with torch.no_grad():
                out = model(ids, labels=ids)
            n_tok = ids.shape[1] - 1
            ntext = len(tok.decode(ids[0], skip_special_tokens=True))
            r.setdefault("nll_per_char", {})[sname] = round(float(out.loss) * n_tok / max(ntext, 1), 4)
        print(f"[score] {sname} done", flush=True)
    except Exception as e:
        res["errors"].append({"scorer": sname, "error": f"{type(e).__name__}: {e}"})
    finally:
        model = None; tok = None; gc.collect(); torch.cuda.empty_cache()
res["total_sec"] = round(time.time() - t0, 1)
raw = json.dumps(res, ensure_ascii=False).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw)).decode())
