/**
 * Hexagonal SyntaxTreeFrame Interface for USIR (TypeScript).
 * Pure data structures decoupling the USIR Core from external parsers,
 * tokenizers, and dependency analyzers.
 * Reference: 10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md Section 2.1
 */

export interface SyntaxNode {
  node_id: number;                // Token / bunketsu ID (0-indexed)
  surface: string;               // Surface string
  lemma: string;                 // Base dictionary lemma for HTH lookup
  pos: string;                   // Coarse POS ("名詞", "動詞", "形容詞", "副詞", "助詞", "助動詞", "記号")
  pos_detail: string;            // Fine POS ("格助詞", "係助詞", "接続助詞", "終助詞", "非自立", etc.)
  hth_primary: bigint;           // High 64-bit uint64: Primary physical sense
  hth_secondary: bigint;         // Low 64-bit uint64: Secondary abstract/figurative sense
  is_negation_scope: boolean;    // In negation scope (ない, ぬ, ず, etc.)
  is_speculative_scope: boolean; // In speculative/simulative scope (だろう, かのよう, etc.)
}

export interface SyntaxEdge {
  source_id: number;              // Source dependent node_id
  target_id: number;              // Target head/governor node_id
  relation: string;               // Dependency relation ("subj", "obj", "mod", "conj", "terminal")
  jump_distance: number;          // Hyperbolic tree jump distance D_jump in [0.0, 1.0]
}

export interface SyntaxTreeFrame {
  sentence_id: number;
  text: string;
  nodes: SyntaxNode[];
  edges: SyntaxEdge[];
  root_id: number;                // Sentence-final root predicate node_id
  max_depth: number;              // Maximum tree depth from root
  depth_distribution: number[];   // Node ratio per depth layer p_d
  radial_entropy: number;         // Tree radial entropy H_radial(T_m) = -sum p_d ln(p_d + 1e-9)
}

export function computeRadialEntropy(depthDist: number[]): number {
  let entropy = 0.0;
  for (const p of depthDist) {
    if (p > 0) {
      entropy -= p * Math.log(p + 1e-9);
    }
  }
  return entropy;
}

export function createSyntaxNode(partial: Partial<SyntaxNode> & { node_id: number; surface: string }): SyntaxNode {
  return {
    node_id: partial.node_id,
    surface: partial.surface,
    lemma: partial.lemma ?? partial.surface,
    pos: partial.pos ?? '名詞',
    pos_detail: partial.pos_detail ?? '',
    hth_primary: partial.hth_primary ?? 0n,
    hth_secondary: partial.hth_secondary ?? 0n,
    is_negation_scope: partial.is_negation_scope ?? false,
    is_speculative_scope: partial.is_speculative_scope ?? false,
  };
}

export function createSyntaxEdge(partial: Partial<SyntaxEdge> & { source_id: number; target_id: number }): SyntaxEdge {
  return {
    source_id: partial.source_id,
    target_id: partial.target_id,
    relation: partial.relation ?? 'mod',
    jump_distance: partial.jump_distance ?? 0.0,
  };
}

export function createSyntaxTreeFrame(partial: Partial<SyntaxTreeFrame> & { sentence_id: number; text: string }): SyntaxTreeFrame {
  const depthDist = partial.depth_distribution ?? [];
  const entropy = partial.radial_entropy ?? computeRadialEntropy(depthDist);
  return {
    sentence_id: partial.sentence_id,
    text: partial.text,
    nodes: partial.nodes ?? [],
    edges: partial.edges ?? [],
    root_id: partial.root_id ?? -1,
    max_depth: partial.max_depth ?? 0,
    depth_distribution: depthDist,
    radial_entropy: entropy,
  };
}
