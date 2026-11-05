"""Phase 34: paper-number verifier."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vpn", ROOT / "scripts" / "verify_paper_numbers.py")
vpn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vpn)


def _setup(tmp_path, data, paper_text):
    (tmp_path / "r.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "p.md").write_text(paper_text, encoding="utf-8")


def test_match_mismatch_mock_and_forbidden(tmp_path):
    _setup(tmp_path, {"provenance": {"mock_used": False}, "x": 0.12345}, "latency 0.12 ms; old 4.17 ms")
    ok = {"numbers": [{"id": "a", "file": "p.md", "text": "latency 0.12 ms", "source": "r.json", "expr": "d['x']",
                       "fmt": "{:.2f}"}]}
    assert vpn.check(ok, tmp_path) == []
    bad = {"numbers": [dict(ok["numbers"][0], fmt="{:.3f}")]}
    assert any("JSON gives" in e for e in vpn.check(bad, tmp_path))
    forb = dict(ok, forbidden=["4.17 ms"], paper_files=["p.md"])
    assert any("forbidden" in e for e in vpn.check(forb, tmp_path))
    _setup(tmp_path, {"metadata": {"mock_mode": True}, "x": 0.12345}, "latency 0.12 ms")
    assert any("mock" in e for e in vpn.check(ok, tmp_path))


def test_repository_registry_passes_when_present():
    reg = ROOT / "scripts" / "paper_numbers.json"
    if not reg.exists():
        return
    errs = vpn.check(json.loads(reg.read_text(encoding="utf-8")))
    assert errs == [], errs
