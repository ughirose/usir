"""Line-break formatter for generated novel text (Phase 23). Pure function."""

from __future__ import annotations

import random
import re
from typing import List, Tuple

_INDENT = "　"
# a 「…」 dialogue, optionally with an attached quotation tail ("と彼は言った。" up to the first sentence end)
_DIALOGUE_RE = re.compile(r"「[^」]*」(?:(?:と|って)[^。！？!?「」]*[。！？!?]?)?")
_SENT_RE = re.compile(r"[^。！？!?]+[。！？!?]*|[。！？!?]+")


_SENT_END = "。！？!?"


def _scan_quote(s: str, i: int, closers: str, lookahead: bool) -> Tuple[str, int]:
    """Scan a quote opened at s[i]; return (content, end index). Unclosed quotes end at the sentence end."""
    n = len(s)
    j = i + 1
    while j < n:
        c = s[j]
        if c in closers:
            return s[i + 1:j], j + 1
        if c in _SENT_END:
            k = j
            while k < n and s[k] in _SENT_END:
                k += 1
            if k < n and s[k] in closers:
                return s[i + 1:k], k + 1
            if lookahead and any(x in s[k:] for x in closers):
                j = k
                continue
            return s[i + 1:k], k
        j += 1
    return s[i + 1:], n


def _normalize_quotes(par: str) -> str:
    """Normalise "…" “…” and dialogue-position 『…』 to 「…」; auto-close unclosed quotes at the sentence end.

    Text inside 「…」 is copied untouched (nested 『』 stays). A top-level 『 counts as dialogue only at the paragraph
    start or right after a sentence end / 」.
    """
    if not any(c in par for c in '"“『'):
        return par
    out: List[str] = []
    i, n = 0, len(par)
    while i < n:
        c = par[i]
        if c == "「":
            j = par.find("」", i)
            j = n - 1 if j < 0 else j
            out.append(par[i:j + 1])
            i = j + 1
        elif c in '"“':
            content, i = _scan_quote(par, i, '"”' if c == "“" else '"', c == "“")
            out.append("「" + content + "」")
        elif c == "『" and (i == 0 or par[i - 1] in _SENT_END + "」"):
            content, i = _scan_quote(par, i, "』", True)
            out.append("「" + content + "」")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _units(par: str) -> List[Tuple[str, bool]]:
    """Paragraph -> [(text, is_dialogue)], narration split into sentences."""
    out: List[Tuple[str, bool]] = []
    pos = 0
    for m in _DIALOGUE_RE.finditer(par):
        for s in _SENT_RE.findall(par[pos:m.start()]):
            if s.strip():
                out.append((s.strip(), False))
        out.append((m.group(0).strip(), True))
        pos = m.end()
    for s in _SENT_RE.findall(par[pos:]):
        if s.strip():
            out.append((s.strip(), False))
    return out


def format_novel(paragraphs: List[str], target_line_chars: int = 120, seed: int = 0) -> str:
    """Insert line breaks at sentence boundaries inside each director paragraph.

    - a 「…」 dialogue (with a directly attached と/って quotation tail) always stands alone on its line, without indent
    - "…" / “…” / dialogue-position 『…』 are normalised to 「…」 first; an unclosed quote is closed at its sentence end
    - narration lines are broken once the line reaches a target drawn from uniform[0.6, 1.4] * target_line_chars
      (new draw per line, deterministic for a given seed)
    - every non-dialogue line starts with a full-width space; director paragraphs are separated by a blank line
    Sentences are never split, so a line can exceed its target by up to one sentence.
    """
    rng = random.Random(seed)

    def draw() -> float:
        return rng.uniform(0.6, 1.4) * target_line_chars

    blocks: List[str] = []
    for par in paragraphs:
        par = re.sub(r"\s+", "", par or "")
        if not par:
            continue
        par = _normalize_quotes(par)
        lines: List[str] = []
        cur = ""
        target = draw()
        for text, is_dlg in _units(par):
            if is_dlg:
                if cur:
                    lines.append(_INDENT + cur)
                    cur = ""
                lines.append(text)
                target = draw()
                continue
            cur += text
            if len(cur) >= target:
                lines.append(_INDENT + cur)
                cur = ""
                target = draw()
        if cur:
            lines.append(_INDENT + cur)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
