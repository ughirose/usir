/**
 * 128-bit Dual-Sense HTH (Hyperbolic-Thermodynamic Hash) Poly Lexicon (TypeScript).
 * O(1) direct binary reader for strictly 480,000 Bytes (30,000 words x 16 Bytes).
 * Isomorphic to Python `usir_py.hth_lexicon.HTHPolyLexicon`.
 *
 * Architecture Reference:
 * - 10_SPEC_PHASE_A_HEXAGONAL_AND_HTH_LEXICON.md Section 1.1 & Section 3.1
 * - usir_adversarial_stress_test_and_feasibility_audit.md Section 24.2
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

export const TOTAL_WORDS = 30_000;
export const BYTES_PER_ENTRY = 16;
export const EXPECTED_BINARY_SIZE = TOTAL_WORDS * BYTES_PER_ENTRY; // 480,000 Bytes
export const ENTRIES_PER_QUANTILE_BIN = 1_875;

// ==============================================================================
// Bit Manipulation and Tree Operations
// ==============================================================================

export function pack30bitOctree(levels10: number[]): number {
  let code = 0;
  for (let lvl = 0; lvl < Math.min(10, levels10.length); lvl++) {
    const digit = ((levels10[lvl] ?? 0) | 0) & 0x7;
    const shift = 3 * (9 - lvl);
    code |= digit << shift;
  }
  return (code & 0x3fffffff) >>> 0;
}

export function unpack30bitOctree(code: number): number[] {
  const c = (code & 0x3fffffff) >>> 0;
  const levels: number[] = new Array(10);
  for (let lvl = 0; lvl < 10; lvl++) {
    const shift = 3 * (9 - lvl);
    levels[lvl] = (c >>> shift) & 0x7;
  }
  return levels;
}

export function packSensory32bit(nibbles: number[]): number {
  let lower32 = 0;
  for (let d = 0; d < Math.min(8, nibbles.length); d++) {
    const val = ((nibbles[d] ?? 0) | 0) & 0xf;
    lower32 |= val << (4 * d);
  }
  return lower32 >>> 0;
}

export function unpackSensory32bit(sensory32bit: number): number[] {
  const lower32 = (sensory32bit & 0xffffffff) >>> 0;
  const nibbles: number[] = new Array(8);
  for (let d = 0; d < 8; d++) {
    nibbles[d] = (lower32 >>> (4 * d)) & 0xf;
  }
  return nibbles;
}

export function sensoryNibblesToFloats(nibbles: number[]): number[] {
  const floats: number[] = new Array(8);
  for (let d = 0; d < 8; d++) {
    floats[d] = (nibbles[d] ?? 0) / 15.0;
  }
  return floats;
}

export function packPoly64bit(octree30bit: number, tag2bit: number, sensory32bit: number): bigint {
  const oct = BigInt((octree30bit & 0x3fffffff) >>> 0) << 34n;
  const tag = BigInt((tag2bit & 0x3) >>> 0) << 32n;
  const sens = BigInt((sensory32bit & 0xffffffff) >>> 0);
  return (oct | tag | sens) & 0xffffffffffffffffn;
}

export function unpackPoly64bit(val: bigint): { octree30bit: number; tag2bit: number; sensory32bit: number } {
  const h = BigInt(val) & 0xffffffffffffffffn;
  const octree30bit = Number((h >> 34n) & 0x3fffffffn);
  const tag2bit = Number((h >> 32n) & 0x3n);
  const sensory32bit = Number(h & 0xffffffffn);
  return { octree30bit, tag2bit, sensory32bit };
}

export function lcaDepth64bit(h1: bigint, h2: bigint): number {
  const xorDiff = (BigInt(h1) ^ BigInt(h2)) & 0xffffffffffffffffn;
  if (xorDiff === 0n) {
    return 10;
  }
  const binStr = xorDiff.toString(2);
  const lzcnt64 = 64 - binStr.length;
  const matchingOctreeBits = Math.min(30, lzcnt64);
  return Math.floor(matchingOctreeBits / 3);
}

export function jumpDistance64bit(h1: bigint, h2: bigint): number {
  return 1.0 - lcaDepth64bit(h1, h2) / 10.0;
}

// ==============================================================================
// Data Structures
// ==============================================================================

export interface DualSenseEntry {
  word: string;
  index: number;
  primary_physical_hth: bigint;
  secondary_abstract_hth: bigint;
  primary_octree: number;
  primary_pos: number;
  primary_sensory_nibbles: number[];
  primary_sensory_vector: number[];
  secondary_octree: number;
  secondary_flag: number;
  secondary_sensory_nibbles: number[];
  secondary_sensory_vector: number[];
  primary_entry: bigint;
  secondary_entry: bigint;
}

export interface SenseResolutionResult {
  hth: bigint;
  sensoryVector: number[];
  slot: 'primary_physical' | 'secondary_abstract';
}

// ==============================================================================
// HTHPolyLexicon Class
// ==============================================================================

export class HTHPolyLexicon {
  private view: DataView;
  private buffer: ArrayBuffer;
  public words: string[] = [];
  public wordToIndex: Map<string, number> = new Map();
  public sha256 = '';
  public version = '1.0.0';

  constructor(
    source?: ArrayBuffer | Uint8Array | Buffer | string,
    metaSource?: { words?: string[]; sha256?: string; version?: string } | string
  ) {
    let uint8: Uint8Array;

    if (typeof source === 'string' || !source) {
      const resolvedPath = HTHPolyLexicon.findBinaryPath(source);
      if (!fs.existsSync(resolvedPath)) {
        throw new Error(`HTH poly binary file not found: ${resolvedPath}`);
      }
      const buf = fs.readFileSync(resolvedPath);
      uint8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } else if (source instanceof Uint8Array) {
      uint8 = source;
    } else if (source instanceof ArrayBuffer) {
      uint8 = new Uint8Array(source);
    } else {
      // Buffer / other TypedArray
      const b = source as Buffer;
      uint8 = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    }

    if (uint8.byteLength !== EXPECTED_BINARY_SIZE) {
      throw new Error(
        `Binary size mismatch: expected strictly ${EXPECTED_BINARY_SIZE} Bytes, got ${uint8.byteLength} Bytes`
      );
    }

    const copy = new ArrayBuffer(uint8.byteLength);
    new Uint8Array(copy).set(uint8);
    this.buffer = copy;
    this.view = new DataView(this.buffer);

    // Load companion metadata
    this.loadMeta(metaSource);
  }

  private static findBinaryPath(customPath?: string): string {
    if (customPath && fs.existsSync(customPath)) {
      return customPath;
    }
    const candidates = [
      customPath,
      path.resolve(process.cwd(), 'python/usir_py/data/hth_poly_30k.bin'),
      path.resolve(process.cwd(), '../python/usir_py/data/hth_poly_30k.bin'),
      path.resolve(process.cwd(), '../../python/usir_py/data/hth_poly_30k.bin'),
      path.resolve(__dirname, '../../../../python/usir_py/data/hth_poly_30k.bin'),
    ].filter((p): p is string => Boolean(p));

    for (const c of candidates) {
      if (fs.existsSync(c)) {
        return c;
      }
    }
    return candidates[0] ?? 'hth_poly_30k.bin';
  }

  private loadMeta(metaSource?: { words?: string[]; sha256?: string; version?: string } | string): void {
    if (typeof metaSource === 'object' && metaSource !== null) {
      this.words = metaSource.words ?? [];
      this.sha256 = metaSource.sha256 ?? '';
      this.version = metaSource.version ?? '1.0.0';
    } else {
      const metaPath =
        typeof metaSource === 'string' && fs.existsSync(metaSource)
          ? metaSource
          : HTHPolyLexicon.findMetaPath();
      if (metaPath && fs.existsSync(metaPath)) {
        try {
          const raw = fs.readFileSync(metaPath, 'utf-8');
          const data = JSON.parse(raw);
          this.words = data.words ?? [];
          this.sha256 = data.sha256 ?? '';
          this.version = data.version ?? '1.0.0';
        } catch {
          // Fallback if parsing fails
        }
      }
    }
    for (let i = 0; i < this.words.length; i++) {
      this.wordToIndex.set(this.words[i]!, i);
    }
  }

  private static findMetaPath(): string | null {
    const candidates = [
      path.resolve(process.cwd(), 'python/usir_py/data/hth_poly_30k_meta.json'),
      path.resolve(process.cwd(), '../python/usir_py/data/hth_poly_30k_meta.json'),
      path.resolve(process.cwd(), '../../python/usir_py/data/hth_poly_30k_meta.json'),
      path.resolve(__dirname, '../../../../python/usir_py/data/hth_poly_30k_meta.json'),
    ];
    for (const c of candidates) {
      if (fs.existsSync(c)) {
        return c;
      }
    }
    return null;
  }

  public get length(): number {
    return TOTAL_WORDS;
  }

  public has(word: string): boolean {
    return this.wordToIndex.has(word);
  }

  public getIndex(word: string): number | undefined {
    return this.wordToIndex.get(word);
  }

  public getRaw128bitByIndex(idx: number): [bigint, bigint] {
    if (idx < 0 || idx >= TOTAL_WORDS) {
      throw new RangeError(`Index ${idx} out of range [0, ${TOTAL_WORDS})`);
    }
    const offset = idx * BYTES_PER_ENTRY;
    const pri64 = this.view.getBigUint64(offset, true);
    const sec64 = this.view.getBigUint64(offset + 8, true);
    return [pri64, sec64];
  }

  public getRaw128bit(word: string): [bigint, bigint] | null {
    const idx = this.wordToIndex.get(word);
    if (idx === undefined) {
      return null;
    }
    return this.getRaw128bitByIndex(idx);
  }

  public lookupByIndex(idx: number): DualSenseEntry {
    const [pri64, sec64] = this.getRaw128bitByIndex(idx);
    const word = idx < this.words.length ? this.words[idx]! : `word_${idx}`;

    const { octree30bit: priOct, tag2bit: priPos, sensory32bit: priSens } = unpackPoly64bit(pri64);
    const priNibs = unpackSensory32bit(priSens);
    const priVec = sensoryNibblesToFloats(priNibs);

    const { octree30bit: secOct, tag2bit: secFlag, sensory32bit: secSens } = unpackPoly64bit(sec64);
    const secNibs = unpackSensory32bit(secSens);
    const secVec = sensoryNibblesToFloats(secNibs);

    return {
      word,
      index: idx,
      primary_physical_hth: pri64,
      secondary_abstract_hth: sec64,
      primary_octree: priOct,
      primary_pos: priPos,
      primary_sensory_nibbles: priNibs,
      primary_sensory_vector: priVec,
      secondary_octree: secOct,
      secondary_flag: secFlag,
      secondary_sensory_nibbles: secNibs,
      secondary_sensory_vector: secVec,
      primary_entry: pri64,
      secondary_entry: sec64,
    };
  }

  public lookup(word: string): DualSenseEntry | null {
    if (!word) {
      return null;
    }
    let idx = this.wordToIndex.get(word);
    if (idx === undefined) {
      idx = this.wordToIndex.get(word.trim());
    }
    if (idx === undefined) {
      return null;
    }
    return this.lookupByIndex(idx);
  }

  public resolveSense(
    word: string,
    contextConcreteness: number,
    applyDamping = false
  ): SenseResolutionResult | null {
    const entry = this.lookup(word);
    if (!entry) {
      return null;
    }

    const cU = Number.isNaN(contextConcreteness) || !Number.isFinite(contextConcreteness)
      ? 0.5
      : Math.min(1.0, Math.max(0.0, contextConcreteness));

    if (cU >= 0.45) {
      return {
        hth: entry.primary_physical_hth,
        sensoryVector: [...entry.primary_sensory_vector],
        slot: 'primary_physical',
      };
    } else if (cU < 0.3) {
      const vec = applyDamping
        ? entry.secondary_sensory_vector.map((v) => v * cU)
        : [...entry.secondary_sensory_vector];
      return {
        hth: entry.secondary_abstract_hth,
        sensoryVector: vec,
        slot: 'secondary_abstract',
      };
    } else {
      const alpha = (cU - 0.3) / 0.15;
      const secVec = applyDamping
        ? entry.secondary_sensory_vector.map((v) => v * cU)
        : entry.secondary_sensory_vector;
      const blendedVec: number[] = new Array(8);
      for (let i = 0; i < 8; i++) {
        blendedVec[i] = alpha * entry.primary_sensory_vector[i]! + (1.0 - alpha) * secVec[i]!;
      }
      const chosenHth = alpha >= 0.5 ? entry.primary_physical_hth : entry.secondary_abstract_hth;
      const slot = alpha >= 0.5 ? 'primary_physical' : 'secondary_abstract';
      return {
        hth: chosenHth,
        sensoryVector: blendedVec,
        slot,
      };
    }
  }

  public discreteJump(
    wordU: string,
    wordV: string,
    slot = 'primary',
    slotV?: string
  ): number | null {
    const entryU = this.lookup(wordU);
    const entryV = this.lookup(wordV);
    if (!entryU || !entryV) {
      return null;
    }

    const pickHth = (entry: DualSenseEntry, s: string): bigint => {
      const sClean = s.toLowerCase().trim();
      if (['primary', 'primary_physical', 'physical', 'pri', '1'].includes(sClean)) {
        return entry.primary_physical_hth;
      } else if (['secondary', 'secondary_abstract', 'abstract', 'sec', '2'].includes(sClean)) {
        return entry.secondary_abstract_hth;
      }
      throw new Error(`Unknown HTH sense slot '${s}'`);
    };

    const hu = pickHth(entryU, slot);
    const hv = pickHth(entryV, slotV ?? slot);
    return jumpDistance64bit(hu, hv);
  }
}
