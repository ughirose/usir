"""Phase 23: builds Colab scripts comparing min / new_ctrl / new_B decoder configs (derived from build_colab_p21.py).

Conditions
  min      : no steering, no rhythm, no repetition/lex penalties; only register/meta/narration guards + finality gate
  new_ctrl : Phase 23 config, no steering: rhythm_profile=steady (climax paragraph: pulse), adnominal_guard, cv band
             [floor, 0.75], lex_compound_exempt, collapse_guard, demo_decay, short_prev_gate, line formatting
  new_B    : new_ctrl + steering + style_genome B (000035)
No LLM judge (Phase 21 judge saturated); texts are read and compared with docs/handoff/v8 vanilla outputs.

Previous docstring (Phase 21):
Builds the self-contained Phase 21 Colab scripts (usir_py + runner + 3 genome binaries embedded as base64).

Usage: python artifacts/colab_run/build_colab_p21.py

Writes (generated payloads, gitignored):
  p21_3b_a.py  Qwen2.5-3B-Instruct bf16, conditions ctrl/A/B/A_legacy, seeds 3,42
  p21_3b_b.py  Qwen2.5-3B-Instruct bf16, conditions ctrl/A/B/A_legacy, seed 137
  p21_7b.py    Qwen2.5-7B-Instruct bf16, conditions ctrl/A/B, seeds 3,42
(p21_3b_a + p21_3b_b together are the 3B plan of 12 runs; one script would be ~60 min on an L4, so it is split.)

Conditions
  ctrl     : steering off, proposal injection off, direction steer off, no scene constraints, no genome
  A / B    : steering on (new_c flag set of the v6 scripts) + style_genome = hard (000879) / fluid (000035)
  A_legacy : steering on + style_genome = legacy-shrinkage genome of 000879
All conditions use the same scene_beats / roles / theme. Generation runs first for every (seed, condition);
then the generator is freed and Qwen2.5-7B-Instruct is the judge (the 7B script reuses the loaded generator).
Run with: python scripts/run_colab_file.py artifacts/colab_run/<name>.py <out_dir> --gpu L4 --timeout 4500
Output: one line 'ARCHIVE_B64 <crc32> <base64(gzip(json))>'.
"""
import base64
import io
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = ROOT / "python"
HERE = pathlib.Path(__file__).resolve().parent
GENOMES = {
    "A": ROOT / "artifacts" / "style_atlas" / "000879.usir",
    "B": ROOT / "artifacts" / "style_atlas" / "000035.usir",
    "A_legacy": ROOT / "artifacts" / "style_atlas" / "legacy" / "000879.usir",
}

THEME = "止まった懐中時計をめぐる短編"
ROLES = ["青年", "老職人"]
SCENE_BEATS = [
    "発端：時計職人の青年が、工房に持ち込まれた止まった懐中時計の不可解な傷に気づく",
    "展開：青年は修理の手を止め、かつて工房を去った師である老職人の記憶をたどり、静かな葛藤を抱く",
    "緊迫：嵐が近づく夜、封じられていた工房の奥の引き出しに異変が起こる",
    "転換：歯車が噛み合った瞬間の閃光の中で、青年は時計が誰のために作られたかを悟る",
    "余韻：夜明けの光が工房に差し、見慣れた道具や街並みが違って見える",
    "結末：再び刻み始めた針の音を聞きながら、青年は時計を届けるために旅立つ",
]

# name -> (generator model id, seeds, conditions)
SCRIPTS = {
    # p23s: n-gram ban paragraph scope; min is seed-deterministic and unaffected, so only new_* is re-run
    "p23s_q3_8b": ("Qwen/Qwen3-8B", [3, 42], ["new_ctrl", "new_B"]),
    "p23s_swallow_8b": ("tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", [3, 42], ["new_ctrl", "new_B"]),
}
JUDGE_MID = "Qwen/Qwen2.5-7B-Instruct"


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (PY / "usir_py").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".bin", ".json"):
                z.write(p, p.relative_to(PY).as_posix())
        z.write(PY / "colab" / "run_v4_bidirectional_experiment.py", "colab/run_v4_bidirectional_experiment.py")
        for k, p in GENOMES.items():
            z.write(p, f"genomes/{k}.usir")
    return base64.b64encode(buf.getvalue()).decode()


BODY = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, importlib.util, re, threading, gc, traceback
import numpy as np
os.makedirs("/content/usir/python", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/usir/python")
sys.path.insert(0, "/content/usir/python")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "janome"], check=True)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_alive = True
def _hb():
    t0 = time.time()
    while _alive:
        time.sleep(15)
        sys.stdout.write(f"hb {time.time()-t0:.0f}s\n"); sys.stdout.flush()
threading.Thread(target=_hb, daemon=True).start()

def log(msg):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()

GDIR = "/content/usir/python/genomes"
GENOME_PATH = {k: f"{GDIR}/{k}.usir" for k in ("A", "B", "A_legacy")}
GBYTES = {k: open(v, "rb").read() for k, v in GENOME_PATH.items()}

spec = importlib.util.spec_from_file_location("run_v4", "/content/usir/python/colab/run_v4_bidirectional_experiment.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from usir_py.realtime_decoder import count_narration_meta, _FRAGMENT_RE, _I_ADJ_STEMS
from usir_py.genome_fit import genome_loglik, delta_ll
from usir_py import text_metrics as TM

NEW = dict(completion_mode=True, carry_context=True, meta_guard=True, polite_guard=True, narration_mask=True, anti_repeat=True, direction_steer=True)
_IADJ_END = re.compile("[" + "".join(sorted(_I_ADJ_STEMS)) + "]く。")
_DEMO = ["それ", "その", "そこ", "あれ", "あの", "これ", "この"]

def metrics(res, wall):
    full = res["full_text"]
    paras = res["paragraphs"]
    body = re.sub(r"[\s、。！？!?「」『』（）()]", "", full)
    bi = [body[i:i + 2] for i in range(len(body) - 1)]
    return {
        "full_text": full,
        "paragraph_texts": [p["text"] for p in paras],
        "paragraph_beats": [p.get("beat") for p in paras],
        "wells": res["wells"],
        "chars": res["total_chars"],
        "para_chars": [p["chars"] for p in paras],
        "cv_l": round(res["overall_cv_l"], 4),
        "distinct_char_bigram": round(len(set(bi)) / max(1, len(bi)), 4),
        "register": count_narration_meta(full),
        "fragment_hits": len(_FRAGMENT_RE.findall(full)),
        "kanji_iadj_renyoutome": len(_IADJ_END.findall(full)),
        "demonstratives_per_paragraph": [{d: p["text"].count(d) for d in _DEMO} for p in paras],
        "demonstratives_total_per_paragraph": [sum(p["text"].count(d) for d in _DEMO) for p in paras],
        "delta_ll_AB": round(float(delta_ll(full, GBYTES["A"], GBYTES["B"])), 4),
        "ll_A": round(genome_loglik(full, GBYTES["A"])["mean_ll"], 4),
        "ll_B": round(genome_loglik(full, GBYTES["B"])["mean_ll"], 4),
        "qda": res["qda"]["phase_regime"],
        "decoder_stats": res["decoder_stats"],
        "genome_shift_norm": res.get("genome_shift_norm"),
        "runner_wall_sec": res["wall_sec"],
        "formatted_text": res.get("formatted_text"),
        "text_metrics": {n: getattr(TM, n)(full) for n in dir(TM) if not n.startswith("_") and callable(getattr(TM, n)) and n in TM_NAMES},
    }
TM_NAMES = ("mean_sentence_len", "comma_per_100", "max_kanji_repeat_in_sentence", "repeat_2gram_in_sentence_max", "adnominal_split_suspects")

t_all = time.time()
gpu = torch.cuda.get_device_name(0)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=False)
tok = AutoTokenizer.from_pretrained(GEN_MID)
t_load = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(GEN_MID, dtype=torch.bfloat16, device_map="cuda")
except Exception as exc:   # Qwen3.5 ships as ImageTextToText; its text path is used
    log(f"CausalLM load failed ({exc!r}); trying AutoModelForImageTextToText")
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(GEN_MID, dtype=torch.bfloat16, device_map="cuda")
log(f"loaded {GEN_MID} bf16 on {gpu} in {time.time()-t_load:.0f}s vram={torch.cuda.memory_allocated()/2**30:.2f}GiB")

RUNS = {}
for SEED in SEEDS:
    plan = mod.plan_longform_constraints(SEED, 6)
    for COND in CONDS:
        key = f"{COND}_s{SEED}"
        t0 = time.time()
        log(f"start {key} t={time.time()-t_all:.0f}s")
        try:
            steered = COND == "new_B"
            if COND == "min":
                kw = dict(steering=False, proposal_injection=False, use_arbitrator=False, scene_constraints={}, direction_steer=False,
                          anti_repeat=False, rhythm=False, rep_penalty=1.0, lex_decay=0.0, cv_gain=0.0, case_frame_gate=False)
            else:
                kw = dict(steering=steered, proposal_injection=steered, use_arbitrator=steered,
                          scene_constraints=(plan["constraints"] if steered else {}), direction_steer=steered, anti_repeat=True,
                          rhythm=True, rhythm_profile="steady", climax_rhythm_profile="pulse", adnominal_guard=True, cv_ceil=0.75,
                          lex_compound_exempt=True, collapse_guard=True, rep_penalty=1.10, lex_decay=2.5,
                          demo_decay=True, short_prev_gate=True, cr_paragraph_scope=True,
                          ngram_paragraph_scope=True)
            res = mod.run_longform_director(
                THEME, model=model, tokenizer=tok, seed=SEED, target_paragraphs=6, target_total_chars=2400,
                min_clauses_per_paragraph=6, reseed_per_paragraph=True, verbose=False,
                min_para_chars=340, max_para_chars=385, finality_gate=True,
                completion_mode=True, carry_context=True, meta_guard=True, polite_guard=True, narration_mask=True,
                scene_beats=SCENE_BEATS, roles=ROLES, target_line_chars=120,
                style_genome=(GENOME_PATH["B"] if COND == "new_B" else None), **kw)
            rec = metrics(res, time.time() - t0)
            rec.update({"seed": SEED, "condition": COND, "model": GEN_MID, "elapsed_sec": round(time.time() - t0, 1), "error": None})
            RUNS[key] = rec
            log(f"done {key} elapsed={rec['elapsed_sec']}s chars={rec['chars']} cv_l={rec['cv_l']} bigram={rec['distinct_char_bigram']} "
                f"register={rec['register']} frag={rec['fragment_hits']} iadj={rec['kanji_iadj_renyoutome']} dLL_AB={rec['delta_ll_AB']}")
        except Exception as exc:
            traceback.print_exc()
            RUNS[key] = {"seed": SEED, "condition": COND, "model": GEN_MID, "error": repr(exc), "elapsed_sec": round(time.time() - t0, 1)}
            log(f"FAILED {key}: {exc!r}")
gen_sec = time.time() - t_all
log(f"generation finished {gen_sec:.0f}s")

out = {"generator": GEN_MID, "dtype": "bfloat16", "gpu": gpu, "theme": THEME, "roles": ROLES,
       "scene_beats": SCENE_BEATS, "seeds": SEEDS, "conditions": CONDS, "generation_sec": round(gen_sec, 1),
       "total_sec": round(time.time() - t_all, 1), "runs": RUNS}
_alive = False
import gzip, zlib
raw = json.dumps(out, ensure_ascii=False, default=float).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
'''


def main() -> None:
    b64 = build_zip_b64()
    for name, (mid, seeds, conds) in SCRIPTS.items():
        header = (
            "B64 = " + repr(b64) + "\n"
            + f"GEN_MID = {mid!r}\nJUDGE_MID = {JUDGE_MID!r}\nSEEDS = {seeds!r}\nCONDS = {conds!r}\n"
            + f"THEME = {THEME!r}\nROLES = {ROLES!r}\nSCENE_BEATS = {SCENE_BEATS!r}\n"
        )
        out = HERE / f"{name}.py"
        out.write_text(header + BODY, encoding="utf-8")
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
