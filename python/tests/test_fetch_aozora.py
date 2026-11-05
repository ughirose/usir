"""Aozora cleaner: ruby / annotation / header / footer stripping on an inline synthetic string."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = _load("fetch_aozora")

SYNTHETIC = (
    "テスト作品\n"
    "架空 作者\n"
    "\n"
    "-------------------------------------------------------\n"
    "【テキスト中に現れる記号について】\n"
    "\n"
    "《》：ルビ\n"
    "（例）漢字《かんじ》\n"
    "-------------------------------------------------------\n"
    "\n"
    "［＃ここから２字下げ］\n"
    "　｜青空文庫《あおぞらぶんこ》の海《うみ》を見た［＃「海」に傍点］。\n"
    "※［＃「木＋世」、第3水準1-85-54］の枝が揺れた。\n"
    "\n"
    "\n"
    "二つ目の段落である。［＃改ページ］\n"
    "［＃ここで字下げ終わり］\n"
    "\n"
    "\n"
    "底本：「架空全集 第一巻」架空出版\n"
    "　　1999（平成11）年1月1日初版発行\n"
    "入力：誰か\n"
    "青空文庫作成ファイル：\n"
)


def test_header_footer_removed():
    out = fetch.clean_aozora_text(SYNTHETIC)
    assert "テスト作品" not in out and "架空" not in out
    assert "記号について" not in out and "-----" not in out
    assert "底本" not in out and "入力" not in out and "青空文庫作成" not in out


def test_ruby_and_bar_and_annotations_removed():
    out = fetch.clean_aozora_text(SYNTHETIC)
    for ch in "《》｜［］＃":
        assert ch not in out
    assert "青空文庫の海を見た。" in out
    assert "※" not in out
    assert "の枝が揺れた。" in out


def test_paragraphs_and_indent_normalised():
    out = fetch.clean_aozora_text(SYNTHETIC)
    assert out.endswith("二つ目の段落である。\n")
    assert "\n\n\n" not in out
    assert not any(ln.startswith("　") for ln in out.split("\n"))


def test_text_without_header_block_is_kept():
    out = fetch.clean_aozora_text("吾輩《わがはい》は猫である。\n名前はまだ無い。\n")
    assert out == "吾輩は猫である。\n名前はまだ無い。\n"


def test_chunking_sizes():
    measure = _load("measure_qda_domain")
    text = "".join(f"これは{i}番目の文であって、長さを揃えるための句を含んでいる。" for i in range(120))
    chunks = measure.chunk_text(text)
    assert chunks
    assert all(160 <= len(c) <= 440 for c in chunks)
