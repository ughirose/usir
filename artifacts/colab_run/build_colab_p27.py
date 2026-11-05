"""Phase 27 Colab job generator: flagship models x 2 scripts x {min, new_ctrl, gA_scaled, gA_legacy, gB_scaled} x seeds.

Genomes (quantile 3-mode forge, Phase 26): A = artifacts/style_atlas/quantile/000879.usir (scaled) and legacy/000879,
B = quantile/001383.usir (scaled). Author ids are atlas labels only; nothing author-specific enters the decoder.
New-control flags: Phase 23 set + kv_share, simile_decay, body_compound_protect, grace_close; genome conditions add
steering + genome_modes (per-well centroid and precision block as G_style).

Run: python scripts/run_colab_file.py artifacts/colab_run/p27_<name>.py docs/handoff/v9 --gpu L4 --timeout 9000
"""
from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "python"
ATLAS = ROOT / "artifacts" / "style_atlas" / "quantile"
GENOMES = {"A": ATLAS / "000879.usir", "A_legacy": ATLAS / "legacy" / "000879.usir", "B": ATLAS / "001383.usir"}

PLOTS = {
    "watch": {
        "theme": "止まった懐中時計と嵐の工房をめぐる短編",
        "roles": ["青年", "老職人"],
        "beats": [
            "発端：時計職人の青年が一人で営む工房に、止まった懐中時計が持ち込まれる。ケースの裏に不可解な傷がある。場にいるのは青年だけで、老職人は十年前に工房を去り不在",
            "展開：青年は修理の手を止め、かつて自分を残して工房を去った師である老職人の記憶をたどる。場にいるのは青年だけで、老職人は回想の中にしか現れない",
            "緊迫：嵐の夜、工房の扉を叩く音がして、ずぶ濡れの老職人が十年ぶりに戻ってくる。ここから二人が同じ場にいる",
            "転換：二人で時計を開け、歯車が噛み合った瞬間、青年は時計が誰のために作られたかを悟る。場にいるのは青年と老職人",
            "余韻：夜明け、老職人は短い言葉を残して再び工房を出ていく。場に残るのは青年だけ",
            "結末：青年は直した懐中時計を手に、それを届けるために旅立つ。場にいるのは青年だけ",
        ],
    },
    "station": {
        "theme": "霧の廃駅と琥珀の切符をめぐる短編",
        "roles": ["少女", "老駅員"],
        "beats": [
            "発端：霧の朝、少女が廃線になった山あいの駅に一人で降り立つ。手には祖母の遺品の古い琥珀色の切符がある。場にいるのは少女だけ",
            "展開：待合室で少女は、切符の裏に書かれた日付と、祖母が語らなかった若い頃の話を思い返す。場にいるのは少女だけ",
            "緊迫：霧の奥から灯りが近づき、もう誰もいないはずの駅舎から老駅員が現れて切符を見せるよう言う。ここから二人が同じ場にいる",
            "転換：老駅員は切符を見て、五十年前にその切符を売った相手と、来なかった最終列車のことを語る。少女は祖母の秘密を知る。場にいるのは少女と老駅員",
            "余韻：霧が晴れ、老駅員の姿はなく、ホームには錆びた線路と朝の光だけが残る。場にいるのは少女だけ",
            "結末：少女は切符を胸にしまい、祖母が乗れなかった道のりを自分の足で歩き始める。場にいるのは少女だけ",
        ],
    },
}

# name -> (model id, plots, seeds, conditions)
SCRIPTS = {
    "p27_swallow_8b": ("tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", ["watch", "station"], [3, 42],
                       ["min", "new_ctrl", "gA_scaled", "gA_legacy", "gB_scaled"]),
    "p27_q3_8b": ("Qwen/Qwen3-8B", ["watch", "station"], [3, 42],
                  ["min", "new_ctrl", "gA_scaled", "gA_legacy", "gB_scaled"]),
    "p27_q35_9b": ("Qwen/Qwen3.5-9B", ["watch"], [3], ["min", "new_ctrl", "gA_scaled"]),
}


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
import base64, io, zipfile, sys, os, time, json, subprocess, importlib.util, re, threading, traceback
import numpy as np
os.makedirs("/content/usir/python", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/usir/python")
sys.path.insert(0, "/content/usir/python")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "janome"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=False)
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
GP = {k: f"{GDIR}/{k}.usir" for k in ("A", "A_legacy", "B")}
GB = {k: open(v, "rb").read() for k, v in GP.items()}
spec = importlib.util.spec_from_file_location("run_v4", "/content/usir/python/colab/run_v4_bidirectional_experiment.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from usir_py.genome_fit import genome_loglik, delta_ll
from usir_py.realtime_decoder import sentence_finished, count_similes
from usir_py import text_metrics as TM

COIN = re.compile(r"懐中時[^計]|青年人|歯輪|時鐘|時刻器|職人的|琥珀[^色のにがをはとで、。]")
TM_NAMES = ("mean_sentence_len", "comma_per_100", "adnominal_split_suspects")

def metrics(res):
    full = res["full_text"]; paras = [p["text"] for p in res["paragraphs"]]
    body = re.sub(r"[\s、。！？!?「」『』（）()]", "", full)
    bi = [body[i:i + 2] for i in range(len(body) - 1)]
    return {
        "full_text": full, "formatted_text": res.get("formatted_text"), "paragraph_texts": paras,
        "wells": res["wells"], "chars": res["total_chars"], "cv_l": round(res["overall_cv_l"], 4),
        "distinct_char_bigram": round(len(set(bi)) / max(1, len(bi)), 4),
        "coinages": COIN.findall(full), "unfinished_paragraphs": sum(0 if sentence_finished(p) else 1 for p in paras),
        "similes": count_similes(full), "marude": full.count("まるで"),
        "ll_A": round(genome_loglik(full, GB["A"])["mean_ll"], 4), "ll_B": round(genome_loglik(full, GB["B"])["mean_ll"], 4),
        "delta_ll_AB": round(float(delta_ll(full, GB["A"], GB["B"])), 4),
        "decoder_stats": res["decoder_stats"],
        "text_metrics": {n: getattr(TM, n)(full) for n in TM_NAMES},
    }

gpu = torch.cuda.get_device_name(0)
tok = AutoTokenizer.from_pretrained(GEN_MID)
t_load = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(GEN_MID, dtype=torch.bfloat16, device_map="cuda")
except Exception as exc:   # Qwen3.5 ships as ImageTextToText; text-only input uses its language model path
    log(f"CausalLM load failed ({exc!r}); trying AutoModelForImageTextToText")
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(GEN_MID, dtype=torch.bfloat16, device_map="cuda")
log(f"loaded {GEN_MID} bf16 on {gpu} in {time.time()-t_load:.0f}s vram={torch.cuda.memory_allocated()/2**30:.2f}GiB")

NEWC = dict(anti_repeat=True, rhythm=True, rhythm_profile="steady", climax_rhythm_profile="pulse", adnominal_guard=True,
            cv_ceil=0.75, lex_compound_exempt=True, collapse_guard=True, rep_penalty=1.10, lex_decay=2.5, demo_decay=True,
            short_prev_gate=True, cr_paragraph_scope=True, ngram_paragraph_scope=True, kv_share=True, simile_decay=True,
            body_compound_protect=True, grace_close=True)
t_all = time.time()
RUNS = {}
for PLOT in PLOTS:
    P = PLOT_DEFS[PLOT]
    for SEED in SEEDS:
        plan = mod.plan_longform_constraints(SEED, 6)
        for COND in CONDS:
            key = f"{PLOT}_{COND}_s{SEED}"
            t0 = time.time(); log(f"start {key} t={time.time()-t_all:.0f}s")
            try:
                if COND == "min":
                    kw = dict(steering=False, proposal_injection=False, use_arbitrator=False, scene_constraints={},
                              direction_steer=False, anti_repeat=False, rhythm=False, rep_penalty=1.0, lex_decay=0.0,
                              cv_gain=0.0, case_frame_gate=False)
                    genome = None
                else:
                    steered = COND.startswith("g")
                    kw = dict(NEWC, steering=steered, proposal_injection=steered, use_arbitrator=steered,
                              scene_constraints=(plan["constraints"] if steered else {}), direction_steer=steered,
                              genome_modes=steered)
                    genome = {"gA_scaled": GP["A"], "gA_legacy": GP["A_legacy"], "gB_scaled": GP["B"]}.get(COND)
                res = mod.run_longform_director(
                    P["theme"], model=model, tokenizer=tok, seed=SEED, target_paragraphs=6, target_total_chars=2400,
                    min_clauses_per_paragraph=6, reseed_per_paragraph=True, verbose=False,
                    min_para_chars=340, max_para_chars=385, finality_gate=True,
                    completion_mode=True, carry_context=True, meta_guard=True, polite_guard=True, narration_mask=True,
                    scene_beats=P["beats"], roles=P["roles"], target_line_chars=120, style_genome=genome, **kw)
                rec = metrics(res)
                rec.update({"plot": PLOT, "seed": SEED, "condition": COND, "model": GEN_MID,
                            "elapsed_sec": round(time.time() - t0, 1), "error": None})
                RUNS[key] = rec
                log(f"done {key} {rec['elapsed_sec']}s chars={rec['chars']} coin={len(rec['coinages'])} "
                    f"unfin={rec['unfinished_paragraphs']} marude={rec['marude']} dLL={rec['delta_ll_AB']}")
            except Exception as exc:
                traceback.print_exc()
                RUNS[key] = {"plot": PLOT, "seed": SEED, "condition": COND, "model": GEN_MID, "error": repr(exc),
                             "elapsed_sec": round(time.time() - t0, 1)}
                log(f"FAILED {key}: {exc!r}")
out = {"generator": GEN_MID, "gpu": gpu, "plots": PLOT_DEFS, "seeds": SEEDS, "conditions": CONDS,
       "generation_sec": round(time.time() - t_all, 1), "runs": RUNS}
_alive = False
import gzip, zlib
raw = json.dumps(out, ensure_ascii=False, default=float).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
'''


def main() -> None:
    b64 = build_zip_b64()
    for name, (mid, plots, seeds, conds) in SCRIPTS.items():
        header = (
            "B64 = " + repr(b64) + "\n"
            + f"GEN_MID = {mid!r}\nPLOTS = {plots!r}\nSEEDS = {seeds!r}\nCONDS = {conds!r}\n"
            + "PLOT_DEFS = " + repr({k: PLOTS[k] for k in plots}) + "\n"
        )
        out = HERE / f"{name}.py"
        out.write_text(header + BODY, encoding="utf-8")
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
