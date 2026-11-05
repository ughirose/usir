"""Phase 23 timing profile: one run (Qwen3-8B, new_ctrl, seed 3) under cProfile + synchronized model.forward timing.

Reuses the p23 payload body; only adds (1) a model.forward wrapper with torch.cuda.synchronize() that splits prefill
(input length > 1) from decode steps, and (2) cProfile around run_longform_director. cProfile inflates Python-heavy
code, so the unprofiled elapsed time of the same config (p23s new_ctrl_s3: 164.6 s on A100) is the reference.

Run: python scripts/run_colab_file.py artifacts/colab_run/p23p_q3_8b.py docs/handoff/v8 --gpu A100 --timeout 3000
"""
from __future__ import annotations

import build_colab_p23 as B

FWD_HOOK = '''
_FW = {"prefill_n": 0, "prefill_t": 0.0, "decode_n": 0, "decode_t": 0.0, "prefill_tokens": 0, "decode_rows": 0}
_orig_forward = model.forward
def _timed_forward(*a, **k):
    ii = k.get("input_ids", a[0] if a else None)
    torch.cuda.synchronize(); _t = time.perf_counter()
    o = _orig_forward(*a, **k)
    torch.cuda.synchronize(); _dt = time.perf_counter() - _t
    L = int(ii.shape[1]) if ii is not None else 1
    if L > 1:
        _FW["prefill_n"] += 1; _FW["prefill_t"] += _dt; _FW["prefill_tokens"] += L * int(ii.shape[0])
    else:
        _FW["decode_n"] += 1; _FW["decode_t"] += _dt; _FW["decode_rows"] += int(ii.shape[0]) if ii is not None else 1
    return o
model.forward = _timed_forward
import cProfile, pstats
PR = cProfile.Profile()
'''

ANCHOR_LOAD = 'log(f"loaded {GEN_MID} bf16 on {gpu} in {time.time()-t_load:.0f}s vram={torch.cuda.memory_allocated()/2**30:.2f}GiB")\n'
ANCHOR_CALL = "            res = mod.run_longform_director(\n"
ANCHOR_CALL_END = '                style_genome=(GENOME_PATH["B"] if COND == "new_B" else None), **kw)\n'
ANCHOR_OUT = '       "total_sec": round(time.time() - t_all, 1), "runs": RUNS}\n'

PROFILE_OUT = '''
_s = io.StringIO(); pstats.Stats(PR, stream=_s).sort_stats("tottime").print_stats(45)
_c = io.StringIO(); pstats.Stats(PR, stream=_c).sort_stats("cumulative").print_stats(60)
out["profile"] = {"forward": _FW, "tottime_top": _s.getvalue(), "cumtime_top": _c.getvalue()}
log(f"forward timing {_FW}")
'''


def main() -> None:
    body = B.BODY
    for a in (ANCHOR_LOAD, ANCHOR_CALL, ANCHOR_CALL_END, ANCHOR_OUT):
        assert body.count(a) == 1, a
    body = body.replace(ANCHOR_LOAD, ANCHOR_LOAD + FWD_HOOK)
    body = body.replace(ANCHOR_CALL, "            PR.enable()\n" + ANCHOR_CALL)
    body = body.replace(ANCHOR_CALL_END, ANCHOR_CALL_END + "            PR.disable()\n")
    body = body.replace(ANCHOR_OUT, ANCHOR_OUT + PROFILE_OUT)
    b64 = B.build_zip_b64()
    header = (
        "B64 = " + repr(b64) + "\n"
        + f"GEN_MID = 'Qwen/Qwen3-8B'\nJUDGE_MID = {B.JUDGE_MID!r}\nSEEDS = [3]\nCONDS = ['new_ctrl']\n"
        + f"THEME = {B.THEME!r}\nROLES = {B.ROLES!r}\nSCENE_BEATS = {B.SCENE_BEATS!r}\n"
    )
    out = B.HERE / "p23p_q3_8b.py"
    out.write_text(header + body, encoding="utf-8")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
