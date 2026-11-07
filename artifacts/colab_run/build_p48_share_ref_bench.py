"""Phase 48 Colab job generator: GRPO step time and numerics with / without the shared reference forward (L4).

Writes artifacts/colab_run/p48_share_ref_bench.py (gitignored, contains the base64 code bundle).
Run: python scripts/run_colab_file.py artifacts/colab_run/p48_share_ref_bench.py docs/handoff/v10 --gpu L4 --timeout 3600

Same settings as the Phase 42/47 GRPO stage (Qwen3-8B bf16, LoRA q/v, 2 prompts x 4 rollouts per step, 256-token
rollouts, shaped reward on complete sentences, completion-only reference PPL), except: 6 steps per arm and a fixed
PPL threshold 1.702 (the Phase 47 calibration value) instead of re-calibrating.  One discarded warm-up step, then
arm share_ref_forward=False and arm True, each from the same initial LoRA weights and the same torch seed.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_colab_p33 import build_zip_b64  # noqa: E402

BODY = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, importlib.util, threading, traceback, gzip, zlib
os.makedirs("/content/usir/python", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/usir/python")
sys.path.insert(0, "/content/usir/python")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "janome", "scikit-learn"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=False)
_alive = True
def _hb():
    t0 = time.time()
    while _alive:
        time.sleep(15)
        sys.stdout.write(f"hb {time.time()-t0:.0f}s\n"); sys.stdout.flush()
threading.Thread(target=_hb, daemon=True).start()
T0 = time.time()
res = {"schema": "p48_share_ref_bench_v1", "arms": [], "errors": []}
try:
    spec = importlib.util.spec_from_file_location("run_phase33_phase_c", "/content/usir/python/colab/run_phase33_phase_c.py")
    drv = importlib.util.module_from_spec(spec); sys.modules["run_phase33_phase_c"] = drv; spec.loader.exec_module(drv)
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from usir_py.lora_grpo_forge import attach_lora_to_model, run_authentic_grpo_lora_forge
    MID = "Qwen/Qwen3-8B"; SEED = 20261106; THR = 1.702
    cfg = drv.Cfg()
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to("cuda").eval()
    res["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
                  "load_s": time.time() - T0}
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(SEED)
    attach_lora_to_model(model, target_modules=("q_proj", "v_proj"), r=cfg.lora_r, alpha=cfg.lora_alpha)
    init = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k}
    items = [{"prompt": drv.render_chat(tok, u), "x_0": u} for u in drv.TRAIN_PROMPTS]
    reward = drv.make_shaped_reward(THR, cfg.alpha_ppl, strip_tail=True)
    def arm(name, share, steps):
        model.load_state_dict(init, strict=False)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(SEED)
        t = time.perf_counter()
        r = run_authentic_grpo_lora_forge(
            model, tokenizer=tok, prompts=items, num_rollouts=cfg.grpo_rollouts, beta_kl=cfg.grpo_beta_kl,
            learning_rate=cfg.grpo_lr, num_steps=steps, device="cuda", domain="novel", ppl_thresh=THR,
            alpha_ppl=cfg.alpha_ppl, ppl_completion_only=True, max_new_tokens=cfg.grpo_max_new_tokens,
            batched_rollouts=True, per_rollout_backward=True, prompts_per_step=cfg.grpo_prompts_per_step,
            store_rollouts=True, reward_fn=reward, share_ref_forward=share,
            progress_fn=lambda e: print(f"[{name}] step {e['step']} reward={e['mean_reward']:.4f} kl={e['mean_kl']:.6f} {e['sec']:.1f}s", flush=True))
        torch.cuda.synchronize()
        out = {"name": name, "share_ref_forward": share, "wall_s": time.perf_counter() - t,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "steps": r["steps"], "mode": r.get("mode")}
        print(f"[{name}] wall {out['wall_s']:.1f}s peak {out['peak_mem_gb']:.2f}GB", flush=True)
        return out
    arm("warmup", False, 1)
    for name, share in (("unshared", False), ("shared", True)):
        try:
            res["arms"].append(arm(name, share, 6))
        except Exception as e:
            res["errors"].append({"arm": name, "error": repr(e)[:400], "tb": traceback.format_exc()[-1500:]})
            torch.cuda.empty_cache()
except Exception as e:
    res["errors"].append({"stage": "setup", "error": repr(e)[:400], "tb": traceback.format_exc()[-1500:]})
res["total_s"] = time.time() - T0
raw = json.dumps(res, ensure_ascii=False).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
_alive = False
'''


def main() -> None:
    out = HERE / "p48_share_ref_bench.py"
    out.write_text("B64 = " + repr(build_zip_b64()) + "\n" + BODY, encoding="utf-8")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
