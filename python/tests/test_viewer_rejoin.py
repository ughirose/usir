"""Pack6 S1: display-side re-join check of quote_bracket flags (scripts/viewer/build_viewer.py). No diagnose logic involved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "viewer"))
import build_viewer as bv  # noqa: E402


def _scenes(text, cuts):
    b = [0, *cuts, len(text)]
    return [{"span": {"start": b[i], "end": b[i + 1]}} for i in range(len(b) - 1)]


def _span(pos, kind):
    return {"start": pos, "basis": {"kind": kind}}


def test_quote_cut_by_scene_boundary_is_hidden():
    t = "a「bbbb\ncccc」d"  # quote opens in scene 0, closes in scene 1
    sc = _scenes(t, [7])
    o2c, c2o = bv.pair_brackets(t)
    assert bv.rejoin_closed(t, sc, o2c, c2o, _span(1, "unclosed_at_eof"))
    assert bv.rejoin_closed(t, sc, o2c, c2o, _span(t.index("」"), "stray_closer"))


def test_unmatched_inside_one_scene_is_kept():
    t = "a「bb」c」d\n\neee"  # extra closer inside scene 0
    sc = _scenes(t, [t.index("eee")])
    o2c, c2o = bv.pair_brackets(t)
    assert not bv.rejoin_closed(t, sc, o2c, c2o, _span(t.rindex("」"), "stray_closer"))
    t2 = "a「bbbb\ncccc"  # never closed
    o2c, c2o = bv.pair_brackets(t2)
    assert not bv.rejoin_closed(t2, _scenes(t2, [7]), o2c, c2o, _span(1, "unclosed_at_eof"))


def test_pair_two_scenes_apart_is_kept():
    t = "a「b\nc\nd」e"
    sc = _scenes(t, [4, 6])  # opener scene 0, closer scene 2
    o2c, c2o = bv.pair_brackets(t)
    assert not bv.rejoin_closed(t, sc, o2c, c2o, _span(1, "unclosed_at_eof"))
