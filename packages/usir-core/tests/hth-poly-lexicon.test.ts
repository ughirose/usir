import { describe, expect, it } from 'vitest';
import * as path from 'node:path';
import {
  HTHPolyLexicon,
  EXPECTED_BINARY_SIZE,
  TOTAL_WORDS,
  lcaDepth64bit,
  jumpDistance64bit,
  packPoly64bit,
  unpackPoly64bit,
  pack30bitOctree,
  unpack30bitOctree,
} from '../src/sensor/hthLexicon.js';
import {
  createSyntaxNode,
  createSyntaxEdge,
  createSyntaxTreeFrame,
} from '../src/schema/syntaxFrame.js';

describe('USIR Hexagonal Phase A: HTHPolyLexicon & SyntaxFrame (TS)', () => {
  const binPath = path.resolve(process.cwd(), 'python/usir_py/data/hth_poly_30k.bin');
  const metaPath = path.resolve(process.cwd(), 'python/usir_py/data/hth_poly_30k_meta.json');
  const lexicon = new HTHPolyLexicon(binPath, metaPath);

  it('verifies binary size and total words', () => {
    expect(lexicon.length).toBe(TOTAL_WORDS);
    expect(lexicon.words.length).toBe(TOTAL_WORDS);
  });

  it('performs O(1) index and word lookups', () => {
    const e0 = lexicon.lookupByIndex(0);
    expect(e0.index).toBe(0);
    expect(e0.primary_physical_hth).toBeDefined();
    expect(e0.primary_sensory_vector.length).toBe(8);

    const cold = lexicon.lookup('冷たい');
    expect(cold).not.toBeNull();
    expect(cold!.word).toBe('冷たい');
    expect(cold!.primary_sensory_vector.length).toBe(8);
  });

  it('resolves dynamic sense slots according to concreteness C(u) (OPEN-01)', () => {
    // High concreteness -> primary
    const resHigh = lexicon.resolveSense('冷たい', 0.85);
    expect(resHigh).not.toBeNull();
    expect(resHigh!.slot).toBe('primary_physical');

    // Low concreteness -> secondary
    const resLow = lexicon.resolveSense('冷たい', 0.2);
    expect(resLow).not.toBeNull();
    expect(resLow!.slot).toBe('secondary_abstract');

    // Low concreteness with damping
    const resDamped = lexicon.resolveSense('冷たい', 0.2, true);
    expect(resDamped).not.toBeNull();
    expect(resDamped!.sensoryVector[0]).toBeCloseTo(resLow!.sensoryVector[0]! * 0.2, 5);
  });

  it('computes 64-bit LCA depth and discrete hyperbolic jump distance', () => {
    expect(jumpDistance64bit(0x1234n, 0x1234n)).toBe(0.0);
    expect(lcaDepth64bit(0x1234n, 0x1234n)).toBe(10);

    const dSelf = lexicon.discreteJump('冷たい', '冷たい');
    expect(dSelf).toBe(0.0);

    const dRelated = lexicon.discreteJump('氷', '雪');
    expect(dRelated).not.toBeNull();
    expect(dRelated!).toBeLessThan(1.0);
  });

  it('creates pure SyntaxTreeFrame data structures', () => {
    const n0 = createSyntaxNode({
      node_id: 0,
      surface: '冷たい',
      lemma: '冷たい',
      pos: '形容詞',
      pos_detail: '自立',
      hth_primary: 100n,
      hth_secondary: 200n,
    });
    const n1 = createSyntaxNode({
      node_id: 1,
      surface: '氷',
      lemma: '氷',
      pos: '名詞',
      pos_detail: '一般',
      hth_primary: 300n,
      hth_secondary: 400n,
    });
    const edge = createSyntaxEdge({
      source_id: 0,
      target_id: 1,
      relation: 'mod',
      jump_distance: 0.2,
    });
    const frame = createSyntaxTreeFrame({
      sentence_id: 1,
      text: '冷たい氷',
      nodes: [n0, n1],
      edges: [edge],
      root_id: 1,
      max_depth: 1,
      depth_distribution: [0.5, 0.5],
    });

    expect(frame.sentence_id).toBe(1);
    expect(frame.nodes.length).toBe(2);
    expect(frame.edges.length).toBe(1);
    expect(frame.edges[0]!.jump_distance).toBe(0.2);
  });
});
