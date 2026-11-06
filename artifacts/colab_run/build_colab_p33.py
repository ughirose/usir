"""Phase 33 Colab job generator: Phase C on a Deadzone-inducing instruct evaluation set (steering + GRPO-LoRA).

Packages python/usir_py, python/colab/run_phase28_phase_c_real.py (helpers imported by the Phase 33 driver) and
python/colab/run_phase33_phase_c.py as a base64 zip header (same approach as build_colab_p28c.py) followed by a body
that unpacks it on the Colab runtime and runs the driver for MODEL_ID.
The body prints one line 'ARCHIVE_B64 <crc32 hex> <base64(gzip(json))>' that scripts/run_colab_file.py collects.

Usage:
    python artifacts/colab_run/build_colab_p33.py
    python scripts/run_colab_file.py artifacts/colab_run/p33_q3_8b.py docs/handoff/v9 --gpu A100 --timeout 10800

Output (generated payload, gitignored):
    p33_q3_8b.py   Qwen/Qwen3-8B, bf16, A100 40GB
Driver options can be added through the EXTRA_ARGS list (for example ["--inject-positions", "last"]).
"""
from __future__ import annotations

import base64
import io
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "python"
DRIVER28 = PY / "colab" / "run_phase28_phase_c_real.py"
DRIVER = PY / "colab" / "run_phase33_phase_c.py"

# name -> (model id, extra driver args)
SCRIPTS = {
    "p33_q3_8b": ("Qwen/Qwen3-8B", []),
    # Phase 36-3: unit-normalised steering direction at last position, alpha 0.08/0.12; shaped GRPO reward
    "p36c_q3_8b": ("Qwen/Qwen3-8B", ["--alphas", "0.08", "0.12", "--inject-positions", "last", "--steer-unit-dir",
                                     "--grpo-reward", "shaped"]),
}
# Phase 38: steering-only re-evaluation without truncation (max_new_tokens 1024), three seed series, no GRPO
for _i, _seed in enumerate((20261106, 20261206, 20261306), start=1):
    SCRIPTS[f"p38c_q3_8b_s{_i}"] = ("Qwen/Qwen3-8B", ["--alphas", "0.08", "0.12", "--inject-positions", "last",
                                                    "--steer-unit-dir", "--no-grpo", "--max-new-tokens", "1024",
                                                    "--seed", str(_seed)])


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((PY / "usir_py").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".bin", ".json"):
                z.write(p, p.relative_to(PY).as_posix())
        z.write(DRIVER28, "colab/run_phase28_phase_c_real.py")
        z.write(DRIVER, "colab/run_phase33_phase_c.py")
    return base64.b64encode(buf.getvalue()).decode()


BODY = r'''
import base64, io, zipfile, sys, os, time, json, subprocess, importlib.util, threading, traceback
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

spec = importlib.util.spec_from_file_location("run_phase33_phase_c", "/content/usir/python/colab/run_phase33_phase_c.py")
drv = importlib.util.module_from_spec(spec)
sys.modules["run_phase33_phase_c"] = drv
spec.loader.exec_module(drv)
try:
    # main() prints the 'ARCHIVE_B64 <crc32> <base64(gzip(json))>' line itself (also when stages recorded errors)
    drv.main(argv=["--model-id", MODEL_ID] + list(EXTRA_ARGS), model_id=MODEL_ID)
except SystemExit:
    raise
except Exception as exc:
    # model load failure etc.: still emit an archive so the failure is collected, never a substituted result
    traceback.print_exc()
    import gzip, zlib
    raw = json.dumps({"schema": "phase33_phase_c_v1", "model_id": MODEL_ID, "fatal_error": repr(exc),
                      "traceback": traceback.format_exc()[-4000:]}, ensure_ascii=False).encode("utf-8")
    print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
_alive = False
'''


def main() -> None:
    b64 = build_zip_b64()
    only = set(sys.argv[1:])
    for name, (mid, extra) in SCRIPTS.items():
        if only and name not in only:
            continue
        header = "B64 = " + repr(b64) + "\n" + f"MODEL_ID = {mid!r}\nEXTRA_ARGS = {extra!r}\n"
        out = HERE / f"{name}.py"
        out.write_text(header + BODY, encoding="utf-8")
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
