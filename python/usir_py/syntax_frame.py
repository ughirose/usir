"""Hexagonal SyntaxTreeFrame Interface for USIR.

Pure data structures decoupling the USIR Core from external parsers,
tokenizers, and dependency analyzers.
Reference: 10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md Section 2.1
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SyntaxNode:
    """A single syntactic unit (bunketsu chunk or morpheme token)."""

    node_id: int                 # Token / bunketsu ID (0-indexed)
    surface: str                 # Surface string
    lemma: str                   # Base dictionary lemma for HTH lookup
    pos: str                     # Coarse POS ("名詞", "動詞", "形容詞", "副詞", "助詞", "助動詞", "記号")
    pos_detail: str              # Fine POS ("格助詞", "係助詞", "接続助詞", "終助詞", "非自立", etc.)
    hth_primary: int = 0         # High 64-bit uint64: Primary physical sense
    hth_secondary: int = 0       # Low 64-bit uint64: Secondary abstract/figurative sense
    is_negation_scope: bool = False     # In negation scope (ない, ぬ, ず, etc.)
    is_speculative_scope: bool = False  # In speculative/simulative scope (だろう, かのよう, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntaxNode:
        return cls(
            node_id=int(data["node_id"]),
            surface=str(data["surface"]),
            lemma=str(data.get("lemma", data["surface"])),
            pos=str(data.get("pos", "名詞")),
            pos_detail=str(data.get("pos_detail", "")),
            hth_primary=int(data.get("hth_primary", 0)),
            hth_secondary=int(data.get("hth_secondary", 0)),
            is_negation_scope=bool(data.get("is_negation_scope", False)),
            is_speculative_scope=bool(data.get("is_speculative_scope", False)),
        )


@dataclass
class SyntaxEdge:
    """Directed dependency edge from modifier/dependent to head/governor."""

    source_id: int               # Source dependent node_id
    target_id: int               # Target head/governor node_id
    relation: str                # Dependency relation ("subj", "obj", "mod", "conj", "terminal")
    jump_distance: float = 0.0   # Hyperbolic tree jump distance D_jump in [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntaxEdge:
        return cls(
            source_id=int(data["source_id"]),
            target_id=int(data["target_id"]),
            relation=str(data.get("relation", "mod")),
            jump_distance=float(data.get("jump_distance", 0.0)),
        )


@dataclass
class SyntaxTreeFrame:
    """Hexagonal sentence-level syntactic frame passed to USIR core."""

    sentence_id: int
    text: str
    nodes: List[SyntaxNode] = field(default_factory=list)
    edges: List[SyntaxEdge] = field(default_factory=list)
    root_id: int = -1             # Sentence-final root predicate node_id
    max_depth: int = 0           # Maximum tree depth from root
    depth_distribution: List[float] = field(default_factory=list) # Node ratio per depth layer p_d
    radial_entropy: float = 0.0  # Tree radial entropy H_radial(T_m) = -sum p_d ln(p_d + 1e-9)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "root_id": self.root_id,
            "max_depth": self.max_depth,
            "depth_distribution": list(self.depth_distribution),
            "radial_entropy": float(self.radial_entropy),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntaxTreeFrame:
        return cls(
            sentence_id=int(data.get("sentence_id", 0)),
            text=str(data.get("text", "")),
            nodes=[SyntaxNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[SyntaxEdge.from_dict(e) for e in data.get("edges", [])],
            root_id=int(data.get("root_id", -1)),
            max_depth=int(data.get("max_depth", 0)),
            depth_distribution=[float(x) for x in data.get("depth_distribution", [])],
            radial_entropy=float(data.get("radial_entropy", 0.0)),
        )
