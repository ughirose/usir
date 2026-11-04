"""Regression tests: nested adnominal clauses and lexicon caching in SyntaxDriver."""

from usir_py.drivers import syntax_driver
from usir_py.drivers.syntax_driver import SyntaxDriver

NESTED = "私が昨日見た映画の中で彼が演じていた役が着ていた服の色が赤かったことについて彼女が言及した。"


def test_nested_adnominal_clauses_exceed_depth_6():
    frame = SyntaxDriver(force_cleanroom=True).parse_text(NESTED)[0]
    assert frame.max_depth > 6


def test_flat_sentences_stay_shallow():
    driver = SyntaxDriver(force_cleanroom=True)
    assert driver.parse_text("雨が降った。")[0].max_depth <= 2
    assert driver.parse_text("静かな朝に、彼は窓を開けた。")[0].max_depth <= 2


def test_default_lexicon_is_shared_between_drivers():
    a = SyntaxDriver(force_cleanroom=True)
    b = SyntaxDriver(force_cleanroom=True)
    assert a._get_lexicon() is b._get_lexicon()
    assert syntax_driver._DEFAULT_LEXICON is a._get_lexicon()
