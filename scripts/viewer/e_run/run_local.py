"""Launcher: run local_from_step1_nogold.py against the sandbox works dir (e_run/works), so the committed work registry stays untouched.
usage: python run_local.py <work_id> [args of local_from_step1_nogold.py]"""
import runpy, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "sensor_runner"))
import common
common.WORKS_DIR = HERE / "works"
sys.argv = ["local_from_step1_nogold.py"] + sys.argv[1:] + ["--out-dir", str(HERE / "local_step1"), "--local-out", str(HERE / "local" / "000119_001738.utterances.json")]
(HERE / "local").mkdir(exist_ok=True)
runpy.run_path(str(HERE / "local_from_step1_nogold.py"), run_name="__main__")
