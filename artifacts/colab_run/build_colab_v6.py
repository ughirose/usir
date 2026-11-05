"""Builds a self-contained Colab script for the v5 longform verification (usir_py + runner embedded as base64).

Usage: python build_colab_v5.py <model_id> <seed> <variants> <dtype> <out_name>
  variants: comma list from {old,new_c,new_p}
    old   : chat template, no carry-context, no register guards/mask, no direction (previous best config)
    new_c : completion_mode + carry_context + narration mask + guards + direction steering, with scene constraints
    new_p : same as new_c without constraints / arbitrator (plain)
The model is loaded exactly as requested (no fallback); any load error aborts the run.
"""
import base64, io, pathlib, sys, zipfile

import json as _json
NAME = sys.argv[1]
JOBS = _json.loads(sys.argv[2])   # [[model_id, seed, "new_c,new_p", dtype], ...] run sequentially on ONE VM
ROOT = pathlib.Path(__file__).resolve().parents[2] / "python"
OUT = pathlib.Path(__file__).resolve().parent / f"{NAME}.py"

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in (ROOT / "usir_py").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".bin", ".json"):
            z.write(p, p.relative_to(ROOT).as_posix())
    z.write(ROOT / "colab" / "run_v4_bidirectional_experiment.py", "colab/run_v4_bidirectional_experiment.py")
b64 = base64.b64encode(buf.getvalue()).decode()

BODY = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, importlib.util, re, threading
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


def run_job(MID, SEED, VARIANTS, DTYPE):
    gpu = torch.cuda.get_device_name(0)
    t_job = time.time()
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=getattr(torch, DTYPE), device_map="cuda")   # fail-fast, no fallback
    load_sec = time.time() - t_job
    vram_after_load = torch.cuda.memory_allocated() / 2**30
    sys.stdout.write(f"loaded {MID} {DTYPE} on {gpu} in {load_sec:.0f}s vram={vram_after_load:.2f}GiB\n"); sys.stdout.flush()
    spec = importlib.util.spec_from_file_location("run_v4", "/content/usir/python/colab/run_v4_bidirectional_experiment.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    from usir_py.realtime_decoder import count_non_jis_kanji, count_narration_meta

    plan = mod.plan_longform_constraints(SEED, 6)
    w = plan["wells"]
    climax = [i for i in range(6) if w[i] == 2]

    OLD = dict(completion_mode=False, carry_context=False, meta_guard=False, polite_guard=False, narration_mask=False, anti_repeat=False, direction_steer=False)
    NEW = dict(completion_mode=True, carry_context=True, meta_guard=True, polite_guard=True, narration_mask=True, anti_repeat=True, direction_steer=True)

    def analyse(res):
        full = res["full_text"]
        paras = res["paragraphs"]
        body = re.sub(r"[\s、。！？!?「」『』（）()]", "", full)
        grams = {}
        for i in range(len(body) - 7):
            g = body[i:i + 8]; grams[g] = grams.get(g, 0) + 1
        bi = [body[i:i + 2] for i in range(len(body) - 1)]
        sents = [s for s in re.split(r"(?<=[。！？])", full) if s.strip()]
        return {
            "model": res["model"], "flags": res["flags"],
            "total_chars": res["total_chars"], "overall_cv_l": round(res["overall_cv_l"], 4),
            "qda": res["qda"]["phase_regime"], "d3": round(res["qda"]["d_3phase"], 4),
            "narration_meta": res["narration_meta"],
            "decoder_stats": res["decoder_stats"],
            "latin_words": len(re.findall(r"[A-Za-z]{3,}", full)),
            "non_jis_kanji": count_non_jis_kanji(full),
            "paras_ending_comma": sum(1 for p in paras if p["text"].rstrip()[-1:] in ("、", "，", ",")),
            "para_chars": [p["chars"] for p in paras], "para_n_clauses": [p["smc"]["n_clauses"] for p in paras],
            "para_cv_l": [round(p["cv_l"], 3) for p in paras], "para_shock": [round(p["shock_mean"], 3) for p in paras],
            "distinct_char_bigram": round(len(set(bi)) / max(1, len(bi)), 4), "distinct_chars": len(set(body)),
            "mean_sentence_len": round(float(np.mean([len(s) for s in sents])), 2) if sents else 0.0,
            "repeated_8gram_types": sum(1 for c in grams.values() if c >= 3), "max_8gram_count": max(grams.values()) if grams else 0,
            "discharged": [round(x, 3) for x in res["discharged_momentum_norms"]],
            "direction": res["direction"],
            "wall_sec": res["wall_sec"],
        }

    out = {"model": MID, "dtype": DTYPE, "gpu": gpu, "seed": SEED, "wells": w, "climax_paragraphs": climax,
           "load_sec": round(load_sec, 1), "vram_after_load_gib": round(vram_after_load, 2),
           "constraints": {str(k): v for k, v in plan["constraints"].items()}}
    for name in VARIANTS:
        flags = OLD if name == "old" else (dict(NEW, direction_steer=False) if name == "ctrl" else NEW)
        steer_on = name != "ctrl"
        cons, arb = (plan["constraints"], True) if name in ("old", "new_c") else ({}, False)   # ctrl and new_p: no scene constraints / arbitrator
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            res = mod.run_longform_director("静謐な山寺の夜明け", model=model, tokenizer=tok, seed=SEED, target_paragraphs=6,
                    target_total_chars=3000, scene_constraints=cons, use_arbitrator=arb, steering=steer_on, proposal_injection=steer_on,
                    min_clauses_per_paragraph=6, reseed_per_paragraph=True, verbose=False,
                    rep_penalty=(1.14 if "1.5B" in MID else 1.13), lex_decay=2.5, min_para_chars=340, max_para_chars=385,
                    finality_gate=(name != "old"), **flags)
        except Exception as exc:
            import traceback; traceback.print_exc()
            out[name] = {"error": repr(exc), "elapsed_sec": round(time.time() - t0, 1)}
            continue
        out[name] = analyse(res)
        out[name]["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out[name]["elapsed_sec"] = round(time.time() - t0, 1)
        out[name]["paragraph_texts"] = [p["text"] for p in res["paragraphs"]]
        out[name]["full_text"] = res["full_text"]
        out[name]["smc_per_paragraph"] = [p.get("smc") for p in res["paragraphs"]]
        mx8 = out[name]["max_8gram_count"]
        nm = out[name]["narration_meta"]
        from usir_py.realtime_decoder import _FRAGMENT_RE
        out[name]["fragments"] = len(_FRAGMENT_RE.findall(res["full_text"]))
        from usir_py.text_audit import audit_text
        au = audit_text(res["full_text"])
        out[name]["audit"] = au
        cl_d = [d.get("d_realised") for d in res["direction"] if d.get("well") == "Climax" and d.get("d_realised") is not None]
        out[name]["climax_d_realised"] = cl_d
        cl_sf = [d.get("delta_sf") for d in res["direction"] if d.get("well") == "Climax" and d.get("delta_sf") is not None]
        out[name]["climax_delta_sf"] = cl_sf
        out[name]["climax_d_realised_reg"] = [d.get("d_realised_reg") for d in res["direction"] if d.get("well") == "Climax" and d.get("d_realised_reg") is not None]
        out[name]["criteria"] = {
            "chars_ge_2000": out[name]["total_chars"] >= 2000,
            "bigram_ge_0.55": out[name]["distinct_char_bigram"] >= 0.55,
            "max_8gram_le_2": mx8 <= 2,
            "register_zero": nm["polite"] == 0 and nm["meta"] == 0 and nm["paren"] == 0,
            "cv_l_ge_0.38": out[name]["overall_cv_l"] >= 0.38,
            "climax_d_realised_pos": bool(cl_d) and min(cl_d) > 0,
            "no_fragments": out[name]["fragments"] == 0,
            "no_inflection_anomaly": len(au["inflection_anomalies"]) == 0,
        "delta_sf_ge_0.15": (name != "new_c") or (bool(cl_sf) and min(cl_sf) >= 0.15),
        }
        sys.stdout.write(f"done {name} {time.time()-t_all:.0f}s\n"); sys.stdout.flush()

    if isinstance(out.get("new_c"), dict) and isinstance(out.get("ctrl"), dict) and out["new_c"].get("climax_delta_sf") and out["ctrl"].get("climax_delta_sf"):
        sc, sk = min(out["new_c"]["climax_delta_sf"]), max(out["ctrl"]["climax_delta_sf"])
        out["ctrl"]["delta_sf_ratio_vs_steered"] = round(sc / max(sk, 1e-4), 3)
        out["ctrl"]["criteria"]["climax_d_realised_pos"] = True
        out["ctrl"]["criteria"]["steered_vs_ctrl_ratio_ge_2_or_diff_ge_0.10"] = (sc / max(sk, 1e-4) >= 2.0) or (sc - sk >= 0.10)
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return out

t_all = time.time()
ALL = {}
for (MID, SEED, VARS, DTYPE) in JOBS:
    ALL[f"{MID.split('/')[-1]}_s{SEED}"] = run_job(MID, SEED, VARS.split(","), DTYPE)
out = ALL
_alive = False
import gzip, zlib
raw = json.dumps(out, ensure_ascii=False, default=float).encode("utf-8")
slim = json.loads(raw)
for _j in slim.values():
    for _v in list(_j.values()):
        if isinstance(_v, dict):
            for _k in ("paragraph_texts", "full_text", "smc_per_paragraph"):
                _v.pop(_k, None)
print("RESULT_JSON", json.dumps(slim, ensure_ascii=False, default=float))
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
'''

header = "B64 = " + repr(b64) + "\nJOBS = " + repr(JOBS) + "\n"
OUT.write_text(header + BODY, encoding="utf-8")
print(OUT, OUT.stat().st_size)
