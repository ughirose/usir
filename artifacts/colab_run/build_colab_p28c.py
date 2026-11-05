"""Phase 28.2 Colab job generator: Phase C real-model measurement (activation steering + GRPO-LoRA).

Packages python/usir_py and python/colab/run_phase28_phase_c_real.py as a base64 zip header (same approach as
build_colab_p27.py) followed by a body that unpacks it on the Colab runtime and runs the driver for MODEL_ID.
The body prints one line 'ARCHIVE_B64 <crc32 hex> <base64(gzip(json))>' that scripts/run_colab_file.py collects.

Usage:
    python artifacts/colab_run/build_colab_p28c.py
    python scripts/run_colab_file.py artifacts/colab_run/p28c_q25_3b.py docs/handoff/v9 --gpu L4 --timeout 5400

Outputs (generated payloads, gitignored):
    p28c_q25_3b.py   Qwen/Qwen2.5-3B-Instruct, bf16 (L4 24GB or A100)
    p28c_q3_8b.py    Qwen/Qwen3-8B, bf16 (L4 24GB is tight for the GRPO stage; A100 40GB recommended)
Driver options can be added per payload through the EXTRA_ARGS list (for example ["--grpo-steps", "60"]).
"""
from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "python"
DRIVER = PY / "colab" / "run_phase28_phase_c_real.py"

# name -> (model id, extra driver args)
SCRIPTS = {
    "p28c_q25_3b": ("Qwen/Qwen2.5-3B-Instruct", []),
    "p28c_q3_8b": ("Qwen/Qwen3-8B", []),
}


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((PY / "usir_py").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".bin", ".json"):
                z.write(p, p.relative_to(PY).as_posix())
        z.write(DRIVER, "colab/run_phase28_phase_c_real.py")
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

spec = importlib.util.spec_from_file_location("run_phase28_phase_c_real", "/content/usir/python/colab/run_phase28_phase_c_real.py")
drv = importlib.util.module_from_spec(spec)
sys.modules["run_phase28_phase_c_real"] = drv
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
    raw = json.dumps({"schema": "phase28_phase_c_real_v1", "model_id": MODEL_ID, "fatal_error": repr(exc),
                      "traceback": traceback.format_exc()[-4000:]}, ensure_ascii=False).encode("utf-8")
    print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw, 9)).decode())
_alive = False
'''


def main() -> None:
    b64 = build_zip_b64()
    for name, (mid, extra) in SCRIPTS.items():
        header = "B64 = " + repr(b64) + "\n" + f"MODEL_ID = {mid!r}\nEXTRA_ARGS = {extra!r}\n"
        out = HERE / f"{name}.py"
        out.write_text(header + BODY, encoding="utf-8")
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
