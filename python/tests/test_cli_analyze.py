"""cli_analyze / style_atlas tests on a small synthetic atlas built in tmp_path (no network, no data/ cache)."""

import json
import random

import numpy as np
import pytest

from usir_py import cli_analyze, style_atlas as sa
from usir_py.usir_forge import OSWEStreamingCompiler, load_tier_m_binary, TIER_M_BYTE_LENGTH

SIMPLE = ["かぜがふく", "ひとりでみちをあるく", "つきがでた", "あめはやんだ", "いぬがなく", "みずをのむ", "ゆめをみた", "そらはあかい"]
FORMAL = [
    "社会制度の構造的変革は経済的基盤の再編成を必然的に要請する",
    "したがって形式的合理性の原理は普遍的規範の体系的整合を前提とする",
    "歴史的条件の相互作用が政治的秩序の均衡を規定するのである",
    "認識論的批判は方法論的反省の一貫した展開を必要とする",
]


def _text(pool, seed, n=220, sep="、"):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.choice([1, 2]) if pool is SIMPLE else 1
        out.append(sep.join(rng.choice(pool) for _ in range(k)) + "。")
    return "".join(out)


@pytest.fixture(scope="module")
def atlas(tmp_path_factory):
    d = tmp_path_factory.mktemp("atlas")
    works = {
        "A": [("a1", _text(SIMPLE, 1)), ("a2", _text(SIMPLE, 2)), ("a3", _text(SIMPLE, 3))],
        "B": [("b1", _text(FORMAL, 4)), ("b2", _text(FORMAL, 5)), ("b3", _text(FORMAL, 6))],
    }
    built = sa.build_atlas(works, d)
    (d / "atlas.json").write_text(json.dumps({"authors": {"A": {"label_ja": "甲"}, "B": {"label_ja": "乙"}}}, ensure_ascii=False), encoding="utf-8")
    return d, built


def test_atlas_binaries_are_valid_tier_m_and_pass_gate(atlas):
    d, built = atlas
    for aid in ("A", "B"):
        for p in (d / f"{aid}.usir", d / "legacy" / f"{aid}.usir"):
            data = p.read_bytes()
            assert len(data) == TIER_M_BYTE_LENGTH
            load_tier_m_binary(data)
        assert built[aid]["audit"]["scaled"]["passed"] and built[aid]["audit"]["legacy"]["passed"]


def test_cli_ranks_matching_style_first(atlas, tmp_path, capsys):
    d, _ = atlas
    f = tmp_path / "in.txt"
    f.write_text(_text(SIMPLE, 99, n=120), encoding="utf-8")
    assert cli_analyze.main([str(f), "--atlas", str(d), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["ranking"][0]["author_id"] == "A"
    assert res["ranking"][0]["label"] == "甲"
    assert res["n_windows"] >= 3
    assert abs(sum(res["phase_occupancy"].values()) - 1.0) < 1e-9
    assert len(res["q"]) == 12 and len(res["p"]) == 12

    f.write_text(_text(FORMAL, 98, n=60), encoding="utf-8")
    assert cli_analyze.main([str(f), "--atlas", str(d), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ranking"][0]["author_id"] == "B"


def test_cli_text_output_and_out_file(atlas, tmp_path, capsys):
    d, _ = atlas
    f = tmp_path / "in.txt"
    f.write_text(_text(SIMPLE, 7, n=100), encoding="utf-8")
    out = tmp_path / "x.usir"
    assert cli_analyze.main([str(f), "--atlas", str(d), "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "5相 占有率" in txt and "甲" in txt
    blob = out.read_bytes()
    assert len(blob) == TIER_M_BYTE_LENGTH
    load_tier_m_binary(blob)


def test_missing_atlas_still_profiles(tmp_path, capsys):
    f = tmp_path / "in.txt"
    f.write_text(_text(SIMPLE, 8, n=60), encoding="utf-8")
    assert cli_analyze.main([str(f), "--atlas", str(tmp_path / "nope"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ranking"] == []


def test_gate_detects_embedded_source_ngram():
    src = "吾輩はここで始めて人間というものを見た。"
    leaky = bytearray(TIER_M_BYTE_LENGTH)
    leaky[100:100 + len(src.encode("utf-8"))] = src.encode("utf-8")
    a = sa.audit_cleanroom_binary(bytes(_with_crc(leaky)), src)
    assert not a["passed"] and a["leaked_7grams"]


def _with_crc(buf):
    import struct, zlib
    struct.pack_into("<I", buf, 0, 0x55534952)
    struct.pack_into("<I", buf, 4092, zlib.crc32(bytes(buf[:4092])) & 0xFFFFFFFF)
    return buf


def test_mw2_properties(atlas):
    d, _ = atlas
    g = sa.load_atlas_gmms(d)
    assert sa.mixture_w2_sq(g["A"], g["A"])[0] == pytest.approx(0.0, abs=1e-6)
    ab = sa.mixture_w2_sq(g["A"], g["B"])[0]
    ba = sa.mixture_w2_sq(g["B"], g["A"])[0]
    assert ab > 0 and ab == pytest.approx(ba, rel=1e-6)
    # mixture transport cost is bounded above by the moment-matched single-Gaussian cost? not guaranteed;
    # only check both are finite and positive.
    assert np.isfinite(sa.single_gaussian_w2_sq(g["A"], g["B"])) and sa.single_gaussian_w2_sq(g["A"], g["B"]) > 0


def test_gaussian_w2_closed_form():
    m1, m2 = np.zeros(24), np.ones(24)
    S = 2.0 * np.eye(24)
    assert sa.gaussian_w2_sq(m1, S, m2, S) == pytest.approx(24.0)
    assert sa.gaussian_w2_sq(m1, S, m1, 8.0 * np.eye(24)) == pytest.approx(24 * (np.sqrt(8) - np.sqrt(2)) ** 2)


def test_forge_default_shrinkage_unchanged_and_legacy_differs():
    rng = np.random.default_rng(0)
    comp = OSWEStreamingCompiler()
    for z in rng.normal(0.4, 0.1, size=(40, 24)):
        comp.add_sample(z, mode=0)
    default = comp.compile_tier_m_binary(enforce_anonymity=False)
    assert default == comp.compile_tier_m_binary(enforce_anonymity=False, shrinkage="scaled")
    assert default != comp.compile_tier_m_binary(enforce_anonymity=False, shrinkage="legacy")
    with pytest.raises(ValueError):
        comp.compile_tier_m_binary(enforce_anonymity=False, shrinkage="bogus")
