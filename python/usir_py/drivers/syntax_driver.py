"""External Syntax Driver and Cleanroom Dependency Automaton for USIR.

Constructs `SyntaxTreeFrame` instances from input Japanese text.
Decouples USIR core from third-party morphological analyzers (Janome, SudachiPy)
by providing a 100% deterministic, zero-dependency cleanroom fallback parser.

References:
- 10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md Section 2.2
- 10_SPEC_PHASE_A_PARSER_FREE_AND_MULTIRESOLUTION.md Section 1
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from usir_py.hth_lexicon import HTHPolyLexicon, jump_distance_64bit, pack_poly_64bit, pack_sensory_32bit
from usir_py.linguistic_gates import (
    NEGATION_PATTERNS,
    SPECULATIVE_PATTERNS,
    evaluate_modality_alpha,
    is_mimetic_word,
    synthesize_mimetic_8d,
)
from usir_py.syntax_frame import SyntaxEdge, SyntaxNode, SyntaxTreeFrame

# Cleanroom Bunketsu tokenization regex (Section 1.2)
BUNKETSU_CHUNK_RE = re.compile(
    r'(?:[一-龯々ヶ]+|[ァ-ヴー]+|[a-zA-Z0-9]+|[ぁ-ん]{1,6}?)'
    r'(?:ないで|なかった|なければ|ない|ぬ|ず|まい|だろう|でしょう|かもしれない|かのよう|みたい|らしい|べき|ため|ように|から|より|けれど|のに|ながら|たり|ても|では|には|とは|へは|まで|だけ|ほど|くらい|など|って|という|として|[がをにへでとはもやの、，]+|[aiueoぁ-ん]{0,4}(?:[、，]|$))'
)

CASE_PARTICLES: Set[str] = {"が", "を", "に", "へ", "で", "と", "から", "より"}
TOPIC_PARTICLES: Set[str] = {"は", "も"}
CONJUNCTION_MARKERS: Set[str] = {"て", "で", "ば", "ながら", "けれど", "のに", "たり", "ても"}


class SyntaxDriver:
    """Hexagonal Adapter: converts natural text into pure SyntaxTreeFrame."""

    def __init__(
        self,
        lexicon: Optional[HTHPolyLexicon] = None,
        force_cleanroom: bool = False,
    ) -> None:
        self.lexicon = lexicon
        self.force_cleanroom = force_cleanroom
        self._janome_tokenizer = None

        if not force_cleanroom:
            try:
                from janome.tokenizer import Tokenizer
                self._janome_tokenizer = Tokenizer()
            except ImportError:
                self._janome_tokenizer = None

    def _get_lexicon(self) -> HTHPolyLexicon:
        if self.lexicon is None:
            self.lexicon = HTHPolyLexicon()
        return self.lexicon

    def parse_sentence(self, text: str, sentence_id: int = 0) -> SyntaxTreeFrame:
        """Parses a single Japanese sentence into a SyntaxTreeFrame."""
        clean_text = text.strip()
        if not clean_text:
            return SyntaxTreeFrame(
                sentence_id=sentence_id,
                text="",
                nodes=[],
                edges=[],
                root_id=-1,
                max_depth=0,
                depth_distribution=[1.0],
            )

        if not self.force_cleanroom and self._janome_tokenizer is not None:
            try:
                return self._parse_janome(clean_text, sentence_id)
            except Exception:
                pass

        # Fallback to cleanroom rule-based automaton (deterministic, external-dependency-free)
        return self._parse_cleanroom(clean_text, sentence_id)

    def parse_text(self, text: str) -> List[SyntaxTreeFrame]:
        """Splits multiline text into sentences and parses each into SyntaxTreeFrame."""
        clean = text.strip()
        if not clean:
            return []

        # Split sentences by Japanese sentence delimiters
        raw_sents = re.split(r'(?<=[。！？!?\n])', clean)
        sents = [s.strip() for s in raw_sents if s.strip()]
        if not sents:
            sents = [clean]

        return [self.parse_sentence(s, sentence_id=i) for i, s in enumerate(sents)]

    def _parse_janome(self, sentence: str, sentence_id: int) -> SyntaxTreeFrame:
        """Parses sentence into SyntaxTreeFrame using Janome morphological analyzer."""
        lex = self._get_lexicon()
        toks = list(self._janome_tokenizer.tokenize(sentence))
        if not toks:
            return self._parse_cleanroom(sentence, sentence_id)

        # Group morphemes into bunketsu chunks
        chunks: List[List[Any]] = []
        curr: List[Any] = []
        for tok in toks:
            pos = tok.part_of_speech.split(',')[0]
            detail = tok.part_of_speech.split(',')[1]
            if curr and pos in ('名詞', '動詞', '形容詞', '副詞', '接続詞', '連体詞') and detail not in ('非自立', '接尾'):
                chunks.append(curr)
                curr = [tok]
            else:
                curr.append(tok)
        if curr:
            chunks.append(curr)

        nodes: List[SyntaxNode] = []
        chunk_tails: List[str] = []
        for idx, chunk_toks in enumerate(chunks):
            surf = "".join(t.surface for t in chunk_toks)
            head_tok = chunk_toks[0]
            tail_toks = chunk_toks[1:]
            tail = "".join(t.surface for t in tail_toks)
            chunk_tails.append(tail)
            stem = head_tok.surface
            lemma = head_tok.base_form if head_tok.base_form != '*' else head_tok.surface

            is_mim = is_mimetic_word(stem) or is_mimetic_word(surf)
            pos = head_tok.part_of_speech.split(',')[0]
            pos_detail = chunk_toks[-1].part_of_speech.split(',')[1] if tail_toks else head_tok.part_of_speech.split(',')[1]

            is_neg = (
                any(neg in surf for neg in NEGATION_PATTERNS)
                or any(t.base_form in ("ない", "ぬ", "ず", "まい") or any(neg in t.surface for neg in NEGATION_PATTERNS) for t in chunk_toks)
            )
            is_spec = (
                any(spec in surf for spec in SPECULATIVE_PATTERNS)
                or any(t.base_form in ("らしい", "ようだ", "そうだ") or any(spec in t.surface for spec in SPECULATIVE_PATTERNS) for t in chunk_toks)
            )

            hth_pri = 0
            hth_sec = 0
            entry = lex.lookup(lemma)
            if entry is None and stem != lemma:
                entry = lex.lookup(stem)
            if entry is None:
                clean_s = re.sub(r'[。！？!?、，\s]+$', '', surf)
                entry = lex.lookup(clean_s)
            if entry is not None:
                hth_pri = entry.primary_physical_hth
                hth_sec = entry.secondary_abstract_hth
                lemma = entry.word
            elif is_mim:
                mim_res = synthesize_mimetic_8d(surf)
                nibs = [int(np.clip(v * 15.0, 0, 15)) for v in mim_res.sensory_vec]
                sens32 = pack_sensory_32bit(nibs)
                hth_pri = pack_poly_64bit(0x15555555, 0b11, sens32)
                hth_sec = pack_poly_64bit(0x0AAAAAAA, 0b11, sens32)

            node = SyntaxNode(
                node_id=idx,
                surface=surf,
                lemma=lemma,
                pos=pos,
                pos_detail=pos_detail,
                hth_primary=hth_pri,
                hth_secondary=hth_sec,
                is_negation_scope=is_neg,
                is_speculative_scope=is_spec,
            )
            nodes.append(node)

        return self._build_tree_frame(sentence_id, sentence, nodes, chunk_tails)

    def _parse_cleanroom(self, sentence: str, sentence_id: int) -> SyntaxTreeFrame:
        """Deterministic Bunketsu and Head-Final dependency tree builder."""
        lex = self._get_lexicon()

        # Step 1: Extract chunks
        chunks = self._segment_bunketsu(sentence)
        if not chunks:
            # Fallback single node
            chunks = [(sentence, sentence, "")]

        # Step 2: Build SyntaxNodes
        nodes: List[SyntaxNode] = []
        for idx, (surface, stem, tail) in enumerate(chunks):
            # Check mimetic
            is_mim = is_mimetic_word(stem) or is_mimetic_word(surface)

            # Determine coarse & fine POS
            pos, pos_detail = self._infer_pos(surface, stem, tail, is_mim)

            # Check negation / speculation
            alpha = evaluate_modality_alpha(tail if tail else surface)
            is_neg = (alpha < 0.0) or any(neg in surface for neg in NEGATION_PATTERNS)
            is_spec = (0.0 < alpha < 1.0) or any(spec in surface for spec in SPECULATIVE_PATTERNS)

            # HTH lookup with punctuation stripping
            hth_pri = 0
            hth_sec = 0
            clean_stem = re.sub(r'[。！？!?、，\s]+$', '', stem)
            clean_surf = re.sub(r'[。！？!?、，\s]+$', '', surface)
            entry = lex.lookup(clean_stem) if clean_stem else None
            if entry is None and clean_surf:
                entry = lex.lookup(clean_surf)

            canonical_lemma = entry.word if entry is not None else (clean_stem if clean_stem else clean_surf)
            if entry is not None:
                hth_pri = entry.primary_physical_hth
                hth_sec = entry.secondary_abstract_hth
            elif is_mim:
                # OPEN-03 sound symbolism synthesis for unknown mimetic
                mim_res = synthesize_mimetic_8d(clean_surf if clean_surf else surface)
                # Pack synthesized sensory nibbles into synthetic HTH
                nibs = [int(np.clip(v * 15.0, 0, 15)) for v in mim_res.sensory_vec]
                sens32 = pack_sensory_32bit(nibs)
                hth_pri = pack_poly_64bit(0x15555555, 0b11, sens32)
                hth_sec = pack_poly_64bit(0x0AAAAAAA, 0b11, sens32)

            node = SyntaxNode(
                node_id=idx,
                surface=surface,
                lemma=canonical_lemma if canonical_lemma else surface,
                pos=pos,
                pos_detail=pos_detail,
                hth_primary=hth_pri,
                hth_secondary=hth_sec,
                is_negation_scope=is_neg,
                is_speculative_scope=is_spec,
            )
            nodes.append(node)

        chunk_tails = [c[2] for c in chunks]
        return self._build_tree_frame(sentence_id, sentence, nodes, chunk_tails)

    def _build_tree_frame(
        self,
        sentence_id: int,
        sentence: str,
        nodes: List[SyntaxNode],
        chunk_tails: List[str],
    ) -> SyntaxTreeFrame:
        """Shared dependency builder, scope propagator, and radial entropy evaluator."""
        K = len(nodes)
        if K == 0:
            return SyntaxTreeFrame(sentence_id=sentence_id, text=sentence, root_id=-1)

        root_id = K - 1
        edges: List[SyntaxEdge] = []
        head_indices = [-1] * K

        for k in range(K - 1):
            src = nodes[k]
            tail_k = chunk_tails[k] if k < len(chunk_tails) else ""
            target_id = root_id
            relation = "mod"

            # Rule 1: Adnominal modification
            if tail_k == "の" or any(tail_k.endswith(x) for x in ("な", "た", "る", "い")):
                target_id = k + 1
                relation = "mod"
            # Rule 2: Adverbs & Mimetics
            elif src.pos == "副詞" or is_mimetic_word(src.surface) or any(tail_k.endswith(x) for x in ("と", "に", "く")):
                pred_idx = self._find_following_predicate(nodes, k + 1)
                target_id = pred_idx if pred_idx is not None else root_id
                relation = "mod"
            # Rule 3: Case particles
            elif any(p in tail_k for p in CASE_PARTICLES):
                relation = "subj" if "が" in tail_k else ("obj" if "を" in tail_k else "mod")
                pred_idx = self._find_following_predicate(nodes, k + 1, stop_at_comma=True)
                target_id = pred_idx if pred_idx is not None else root_id
            # Rule 4: Topic / Conjunction particles
            elif any(p in tail_k for p in TOPIC_PARTICLES) or any(c in tail_k for c in CONJUNCTION_MARKERS):
                relation = "subj" if any(p in tail_k for p in TOPIC_PARTICLES) else "conj"
                target_id = root_id
            else:
                target_id = root_id
                relation = "mod"

            head_indices[k] = target_id
            jump_d = jump_distance_64bit(src.hth_primary, nodes[target_id].hth_primary) if (src.hth_primary and nodes[target_id].hth_primary) else 0.0
            edges.append(SyntaxEdge(source_id=k, target_id=target_id, relation=relation, jump_distance=jump_d))

        # Propagate Negation & Speculation scopes along dependency paths (OPEN-02)
        for k in range(K):
            curr = k
            visited = {curr}
            neg_path = False
            spec_path = False
            while curr != -1 and curr < K:
                if nodes[curr].is_negation_scope:
                    neg_path = True
                if nodes[curr].is_speculative_scope:
                    spec_path = True
                if curr == root_id:
                    break
                next_head = head_indices[curr]
                if next_head in visited or next_head == -1:
                    break
                visited.add(next_head)
                curr = next_head
            if neg_path:
                nodes[k].is_negation_scope = True
            if spec_path:
                nodes[k].is_speculative_scope = True

        # Tree Depths & radial entropy
        depths = [0] * K
        for k in range(K - 1):
            d = 1
            curr = head_indices[k]
            visited = {k}
            while curr != -1 and curr != root_id and curr not in visited and d < K:
                visited.add(curr)
                curr = head_indices[curr]
                d += 1
            depths[k] = d
        depths[root_id] = 0

        max_depth = max(depths) if depths else 0
        depth_counts = [0] * (max_depth + 1)
        for d in depths:
            depth_counts[d] += 1
        depth_distribution = [c / max(K, 1) for c in depth_counts]
        radial_entropy = -sum(p * math.log(p + 1e-9) for p in depth_distribution if p > 0.0)

        return SyntaxTreeFrame(
            sentence_id=sentence_id,
            text=sentence,
            nodes=nodes,
            edges=edges,
            root_id=root_id,
            max_depth=max_depth,
            depth_distribution=depth_distribution,
            radial_entropy=radial_entropy,
        )

    def _segment_bunketsu(self, sentence: str) -> List[Tuple[str, str, str]]:
        """Segments sentence into (surface, stem, tail) tuples."""
        matches = list(BUNKETSU_CHUNK_RE.finditer(sentence))
        if not matches:
            return [(sentence, sentence, "")]

        chunks: List[Tuple[str, str, str]] = []
        last_end = 0

        for m in matches:
            span_start, span_end = m.span()
            if span_start > last_end:
                gap = sentence[last_end:span_start].strip()
                if gap:
                    chunks.append((gap, gap, ""))

            raw_chunk = m.group(0)
            if not raw_chunk:
                continue

            # Separate stem and tail
            stem, tail = self._split_stem_tail(raw_chunk)
            chunks.append((raw_chunk, stem, tail))
            last_end = span_end

        if last_end < len(sentence):
            rem = sentence[last_end:].strip()
            if rem:
                stem, tail = self._split_stem_tail(rem)
                chunks.append((rem, stem, tail))

        return chunks

    def _split_stem_tail(self, chunk: str) -> Tuple[str, str]:
        """Separates content stem from functional tail (particles/inflections)."""
        core = re.sub(r'[。！？!?、，\s]+$', '', chunk)
        tail_match = re.search(
            r'(ないで|なかった|なければ|ない|ぬ|ず|まい|だろう|でしょう|かもしれない|かのよう|みたい|らしい|べき|ため|ように|から|より|けれど|のに|ながら|たり|ても|では|には|とは|へは|まで|だけ|ほど|くらい|など|って|という|として|[がをにへでとはもやの]+|[ぁ-ん]{1,3})$',
            core,
        )
        if tail_match and tail_match.start() > 0:
            stem = core[:tail_match.start()]
            tail = chunk[len(stem):]
            return stem, tail
        return core, chunk[len(core):]

    def _infer_pos(self, surface: str, stem: str, tail: str, is_mim: bool) -> Tuple[str, str]:
        """Infers coarse and fine POS from morphology."""
        if is_mim:
            return "副詞", "オノマトペ"

        clean_surf = re.sub(r'[。！？!?、，\s]+$', '', surface)
        clean_tail = re.sub(r'[。！？!?、，\s]+$', '', tail)

        # Check auxiliary verbs (negation / speculation)
        if any(neg in clean_surf for neg in NEGATION_PATTERNS) or any(neg in clean_tail for neg in NEGATION_PATTERNS):
            return "助動詞", "否定"
        if any(spec in clean_surf for spec in SPECULATIVE_PATTERNS) or any(spec in clean_tail for spec in SPECULATIVE_PATTERNS):
            return "助動詞", "推量"

        if any(p in clean_tail for p in CASE_PARTICLES):
            return "名詞", "格助詞"
        if any(p in clean_tail for p in TOPIC_PARTICLES):
            return "名詞", "係助詞"
        if clean_tail == "の":
            return "名詞", "連体助詞"

        # Check adjectives
        if clean_surf.endswith("い") or clean_tail.endswith("い") or clean_surf.endswith("たい"):
            return "形容詞", "自立"

        # Check verbs
        if any(clean_tail.endswith(x) or clean_surf.endswith(x) for x in ("る", "た", "て", "ます", "う", "く", "す", "つ", "ぬ", "ふ", "む")):
            return "動詞", "自立"

        # Fallback noun
        return "名詞", "一般"

    def _find_following_predicate(
        self,
        nodes: List[SyntaxNode],
        start_idx: int,
        stop_at_comma: bool = False,
    ) -> Optional[int]:
        """Finds nearest following predicate (verb/adjective) node."""
        for idx in range(start_idx, len(nodes)):
            if stop_at_comma and "、" in nodes[idx].surface:
                if nodes[idx].pos in ("動詞", "形容詞"):
                    return idx
                break
            if nodes[idx].pos in ("動詞", "形容詞"):
                return idx
        return None
