"""Phase 1.5 Stage 2 Step 9: the committed seed cache must hit with SENSOR_ANNOTATOR_MODEL_ID unset.

The seed L2 keys contain the annotator model_id (Step 8). The code default is that declared value, so a plain checkout hits the seed.
The key specification is unchanged: an explicit env value still overrides the default (and then misses, because model_id is a key part)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SR = Path(__file__).resolve().parents[2] / "scripts" / "sensor_runner"
sys.path.insert(0, str(SR))

import common  # noqa: E402

try:
    NIS = common.find_nishiki()
except FileNotFoundError:
    NIS = None

PROBE = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import common
from cache import Cache
from resolve import resolve_work
from prepare import prepare_work
nis = common.find_nishiki()
cache = Cache(Path(sys.argv[3]), legacy_adopt=False)  # a COPY of the seed cache: prepare writes manifests
out = {"model_id": common.ANNOTATION_MODEL_ID, "hits": 0, "scenes": 0, "pending": 0, "packs": 0}
for w in common.all_work_ids():
    work = common.load_work(w)
    seg = common.segment(nis, work["text_abs"])
    res = resolve_work(w, cache, seg["scenes"], work["characters"])
    out["scenes"] += len(res)
    out["hits"] += sum(r.how == "hit" for r in res)
    r = prepare_work(w, cache, seg=seg, packs_dir=sys.argv[2], defer_blocked=True)
    out["pending"] += r["pending"]
    out["packs"] += len(r["packs_written"])
print(json.dumps(out))
"""


def _probe(tmp_path: Path, env_value: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "SENSOR_ANNOTATOR_MODEL_ID"}
    if env_value is not None:
        env["SENSOR_ANNOTATOR_MODEL_ID"] = env_value
    work_cache = tmp_path / "cache"
    shutil.copytree(SR / "cache", work_cache)  # never touch the committed seed
    p = subprocess.run([sys.executable, "-c", PROBE, str(SR), str(tmp_path / "packs"), str(work_cache)], capture_output=True, env=env, cwd=str(SR))
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    return json.loads(p.stdout.decode("utf-8").strip().splitlines()[-1])


def test_default_model_id_is_the_seed_annotator_and_not_unspecified():
    assert common.DEFAULT_ANNOTATOR_MODEL_ID != "unspecified"
    # the seed L2 entries were keyed with this id (key part, not a label)
    ids = set()
    for f in (SR / "cache" / "l2").rglob("*.json"):
        ids.add(json.loads(f.read_text(encoding="utf-8"))["parts"].get("model_id"))
    assert ids == {common.DEFAULT_ANNOTATOR_MODEL_ID}


def test_step2_tool_declares_the_same_model_id():
    src = (SR / "step2.py").read_text(encoding="utf-8")
    assert f'MODEL_ID = "{common.DEFAULT_ANNOTATOR_MODEL_ID}"' in src


@pytest.mark.skipif(NIS is None, reason="nishiki binary missing")
def test_seed_hits_with_env_unset_and_prepare_is_empty(tmp_path):
    r = _probe(tmp_path, None)
    assert r["model_id"] == common.DEFAULT_ANNOTATOR_MODEL_ID
    assert r["scenes"] == 35 and r["hits"] == 35
    assert r["pending"] == 0 and r["packs"] == 0


@pytest.mark.skipif(NIS is None, reason="nishiki binary missing")
def test_explicit_env_still_overrides_and_then_misses(tmp_path):
    r = _probe(tmp_path, "unspecified")
    assert r["model_id"] == "unspecified"
    assert r["hits"] == 0  # key spec unchanged: model_id is part of the key
