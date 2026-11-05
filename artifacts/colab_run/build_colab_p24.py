"""Phase 24 Colab job: (1) torch parity tests for batched scoring / shared prefix KV cache, (2) timing of one
Qwen3-8B new_ctrl run (seed 3) without and with kv_share on the same GPU in the same session.

Run: python scripts/run_colab_file.py artifacts/colab_run/p24_q3_8b.py docs/handoff/v9 --gpu A100 --timeout 3000
"""
from __future__ import annotations

import base64
import io
import zipfile

import build_colab_p23 as B

TEST_FILE = B.PY / "tests" / "test_phase24_kv_batch.py"

PYTEST = '''
_pt = subprocess.run([sys.executable, "-m", "pytest", "-q", "/content/usir/python/tests/test_phase24_kv_batch.py"],
                     cwd="/content/usir/python", capture_output=True, text=True)
PYTEST_OUT = (_pt.stdout or "")[-3000:] + (_pt.stderr or "")[-1500:]
log("PYTEST rc=%d\\n%s" % (_pt.returncode, PYTEST_OUT[-1500:]))
'''
ANCHOR_LOAD = 'log(f"loaded {GEN_MID} bf16 on {gpu} in {time.time()-t_load:.0f}s vram={torch.cuda.memory_allocated()/2**30:.2f}GiB")\n'
ANCHOR_KW = "                style_genome=(GENOME_PATH[\"B\"] if COND == \"new_B\" else None), **kw)\n"
ANCHOR_OUT = '       "total_sec": round(time.time() - t_all, 1), "runs": RUNS}\n'


def build_zip_b64() -> str:
    raw = base64.b64decode(B.build_zip_b64())
    buf = io.BytesIO(raw)
    with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as z:
        z.write(TEST_FILE, "tests/test_phase24_kv_batch.py")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    body = B.BODY
    for a in (ANCHOR_LOAD, ANCHOR_KW, ANCHOR_OUT):
        assert body.count(a) == 1, a
    body = body.replace(ANCHOR_LOAD, ANCHOR_LOAD + PYTEST)
    body = body.replace(ANCHOR_KW, "                kv_share=COND.endswith(\"_kv\"),\n" + ANCHOR_KW)
    body = body.replace(ANCHOR_OUT, ANCHOR_OUT + 'out["pytest"] = PYTEST_OUT\n')
    header = (
        "B64 = " + repr(build_zip_b64()) + "\n"
        + f"GEN_MID = 'Qwen/Qwen3-8B'\nJUDGE_MID = {B.JUDGE_MID!r}\nSEEDS = [3]\nCONDS = ['new_ctrl', 'new_ctrl_kv']\n"
        + f"THEME = {B.THEME!r}\nROLES = {B.ROLES!r}\nSCENE_BEATS = {B.SCENE_BEATS!r}\n"
    )
    out = B.HERE / "p24_q3_8b.py"
    out.write_text(header + body, encoding="utf-8")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
