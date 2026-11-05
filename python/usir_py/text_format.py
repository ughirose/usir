"""Line-break formatter for generated novel text (Phase 23). Pure function."""

from __future__ import annotations

import random
import re
from typing import List, Tuple

_INDENT = "　"
# a 「…」 dialogue, optionally with an attached quotation tail ("と彼は言った。" up to the first sentence end)
_DIALOGUE_RE = re.compile(r"「[^」]*」(?:(?:と|って)[^。！？!?「」]*[。！？!?]?)?")
_SENT_RE = re.compile(r"[^。！？!?]+[。！？!?]*|[。！？!?]+")


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
