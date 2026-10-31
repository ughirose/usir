/**
 * Universal Style Genome (USIR v2.0) Zero-Leakage 64-bit HTH Lexicon & Linguistic Frontend.
 * Module 1: Deterministic 256-node Laplacian diffusion, closed-functional-grammar lattice,
 * OPEN-01 (Polysemy), OPEN-02 (Dual Negation), OPEN-03 (Mimetic), OPEN-04 (Bakhtinian 2-Channel),
 * and M1 Music Functor.
 */

export interface HTHEntry {
  lemma: string;
  pos: number; // 0=Noun, 1=Verb, 2=Adjective, 3=Adverb/Mimetic
  octree30: number; // 30-bit Octree integer (10 levels x 3 bits)
  sensory8d: number[]; // 8D sensory/ontological vector in [0, 1]
  concreteness: number; // C(w) in [0, 1]
  clusterId: number; // 0..255
  mimeticImmunity?: boolean;
  isMimetic?: boolean;
}

export interface TokenAnalysis {
  surface: string;
  lemma: string;
  pos: number;
  entry: HTHEntry;
  effectiveSensory8d: number[];
  isNegated: boolean;
  modalityAlpha: number;
  polysemyState: 'literal' | 'poetic_metaphor' | 'dead_metaphor';
  isMimetic: boolean;
  mimeticImmunity: boolean;
}

export interface MusicNoteEvent {
  pitch: number; // MIDI note 0..127
  startTime: number; // seconds or beats
  duration: number; // seconds or beats
  velocity: number; // MIDI velocity 0..127
  channel?: number;
}

export function clz30(x: number): number {
  const masked = (x & 0x3fffffff) >>> 0;
  return masked === 0 ? 30 : Math.clz32(masked) - 2;
}

export function lcaDepth30Bit(hU: number, hV: number): number {
  return Math.floor(clz30((hU ^ hV) & 0x3fffffff) / 3);
}

export function discreteJumpDistance(hU: number, hV: number): number {
  return 1.0 - lcaDepth30Bit(hU, hV) / 10.0;
}

export function pack30bitOctree(levels10: number[]): number {
  let code = 0;
  for (let lvl = 0; lvl < Math.min(10, levels10.length); lvl++) {
    const digit = (levels10[lvl] ?? 0) & 0x7;
    const shift = 3 * (9 - lvl);
    code |= digit << shift;
  }
  return (code & 0x3fffffff) >>> 0;
}

export function packLower32Sensory(vec8d: number[] | Float64Array): number {
  let lower32 = 0;
  for (let d = 0; d < 8; d++) {
    const val = vec8d[d] ?? 0.0;
    const q = Math.min(15, Math.max(0, Math.round(val * 15.0)));
    lower32 |= (q & 0xf) << (4 * d);
  }
  return lower32 >>> 0;
}

export function unpackLower32Sensory(lower32: number): number[] {
  const vec: number[] = new Array(8);
  for (let d = 0; d < 8; d++) {
    const nibble = (lower32 >>> (4 * d)) & 0xf;
    vec[d] = nibble / 15.0;
  }
  return vec;
}

export function pack64bitHth(pos2bit: number, octree30bit: number, lower32Sensory: number): bigint {
  const upper32 = BigInt(((pos2bit & 0x3) << 30) | (octree30bit & 0x3fffffff));
  const lower = BigInt(lower32Sensory >>> 0);
  return ((upper32 & 0xffffffffn) << 32n) | (lower & 0xffffffffn);
}

export function unpack64bitHth(hth64: bigint): { pos: number; octree30: number; sensory8d: number[] } {
  const h = BigInt(hth64);
  const upper32 = Number((h >> 32n) & 0xffffffffn);
  const lower32 = Number(h & 0xffffffffn);
  const pos = (upper32 >>> 30) & 0x3;
  const octree30 = upper32 & 0x3fffffff;
  const sensory8d = unpackLower32Sensory(lower32);
  return { pos, octree30, sensory8d };
}

export interface DualSenseHthEntry {
  word: string;
  primary_physical_hth: bigint;
  secondary_abstract_hth: bigint;
  primaryPhysicalHth?: bigint;
  secondaryAbstractHth?: bigint;
  primary_entry: HTHEntry;
  secondary_entry: HTHEntry;
  primaryEntry?: HTHEntry;
  secondaryEntry?: HTHEntry;
}

export interface DualSenseResolutionResult {
  selectedSlot: 'primary_physical' | 'secondary_abstract';
  selected_slot: 'primary_physical' | 'secondary_abstract';
  selectedHth: bigint;
  selected_hth: bigint;
  full128bit: [bigint, bigint];
  full_128bit: [bigint, bigint];
  primary64bit: bigint;
  primary_64bit: bigint;
  secondary64bit: bigint;
  secondary_64bit: bigint;
  contextConcreteness: number;
  context_concreteness: number;
  effectiveSensory: number[];
  effective_sensory: number[];
  word: string;
  contextWord: string;
  context_word: string;
}

function _makeDualSense(
  word: string,
  pos2bit: number,
  pos: number,
  priOctree: number,
  priSensory: number[],
  priConcreteness: number,
  secOctree: number,
  secSensory: number[],
  secConcreteness: number
): DualSenseHthEntry {
  const priRaw = pack64bitHth(pos2bit, priOctree, packLower32Sensory(priSensory));
  const secRaw = pack64bitHth(pos2bit, secOctree, packLower32Sensory(secSensory));
  const priEntry: HTHEntry = {
    lemma: word,
    pos,
    octree30: priOctree & 0x3fffffff,
    sensory8d: [...priSensory],
    concreteness: priConcreteness,
    clusterId: 0,
    mimeticImmunity: false,
    isMimetic: false,
  };
  const secEntry: HTHEntry = {
    lemma: word,
    pos,
    octree30: secOctree & 0x3fffffff,
    sensory8d: [...secSensory],
    concreteness: secConcreteness,
    clusterId: 0,
    mimeticImmunity: false,
    isMimetic: false,
  };
  return {
    word,
    primary_physical_hth: priRaw,
    secondary_abstract_hth: secRaw,
    primaryPhysicalHth: priRaw,
    secondaryAbstractHth: secRaw,
    primary_entry: priEntry,
    secondary_entry: secEntry,
    primaryEntry: priEntry,
    secondaryEntry: secEntry,
  };
}

export const DUAL_SENSE_LEXICON: Record<string, DualSenseHthEntry> = {
  冷たい: _makeDualSense(
    '冷たい', 2, 2,
    0x20049249, [0.3, 0.1, 0.9, 0.7, 0.2, 0.4, 0.0, 0.2], 0.85,
    0x00049249, [0.1, 0.0, 0.1, 0.1, 0.0, 0.2, 0.8, 0.6], 0.22
  ),
  落とす: _makeDualSense(
    '落とす', 1, 1,
    0x1836db6d, [0.2, 0.4, 0.3, 0.2, 0.9, 0.5, 0.0, 0.1], 0.80,
    0x00092492, [0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.85, 0.4], 0.25
  ),
  走る: _makeDualSense(
    '走る', 1, 1,
    0x183b6db6, [0.3, 0.2, 0.4, 0.8, 0.95, 0.6, 0.0, 0.0], 0.88,
    0x000db6db, [0.1, 0.0, 0.0, 0.0, 0.3, 0.3, 0.85, 0.2], 0.24
  ),
  深い: _makeDualSense(
    '深い', 2, 2,
    0x2004b6db, [0.4, 0.1, 0.5, 0.3, 0.1, 0.9, 0.0, 0.4], 0.82,
    0x0006db6d, [0.0, 0.0, 0.0, 0.1, 0.0, 0.4, 0.9, 0.7], 0.20
  ),
  高い: _makeDualSense(
    '高い', 2, 2,
    0x08012492, [0.7, 0.1, 0.3, 0.2, 0.2, 0.95, 0.0, 0.1], 0.85,
    0x08016db6, [0.1, 0.0, 0.0, 0.0, 0.1, 0.3, 0.85, 0.2], 0.28
  ),
  重い: _makeDualSense(
    '重い', 2, 2,
    0x10024924, [0.2, 0.1, 0.8, 0.8, 0.7, 0.4, 0.0, 0.1], 0.86,
    0x00004924, [0.0, 0.0, 0.1, 0.3, 0.0, 0.2, 0.85, 0.5], 0.25
  ),
  光る: _makeDualSense(
    '光る', 1, 1,
    0x28052492, [0.98, 0.1, 0.2, 0.1, 0.3, 0.5, 0.0, 0.0], 0.90,
    0x0000b6db, [0.3, 0.0, 0.0, 0.0, 0.1, 0.2, 0.8, 0.3], 0.30
  ),
  闇: _makeDualSense(
    '闇', 0, 0,
    0x28056db6, [0.05, 0.1, 0.3, 0.4, 0.1, 0.7, 0.0, 0.85], 0.78,
    0x00001249, [0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.9, 0.6], 0.22
  ),
  切る: _makeDualSense(
    '切る', 1, 1,
    0x1002b6db, [0.4, 0.3, 0.85, 0.4, 0.9, 0.4, 0.0, 0.0], 0.89,
    0x00009249, [0.0, 0.1, 0.0, 0.0, 0.3, 0.2, 0.85, 0.4], 0.26
  ),
  立つ: _makeDualSense(
    '立つ', 1, 1,
    0x18032492, [0.3, 0.1, 0.5, 0.85, 0.6, 0.7, 0.0, 0.0], 0.84,
    0x00002492, [0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.95, 0.2], 0.18
  ),
  熱い: _makeDualSense(
    '熱い', 2, 2,
    0x20042492, [0.6, 0.1, 0.95, 0.8, 0.3, 0.3, 0.0, 0.1], 0.88,
    0x38072492, [0.2, 0.1, 0.2, 0.5, 0.3, 0.1, 0.4, 0.3], 0.35
  ),
  甘い: _makeDualSense(
    '甘い', 2, 2,
    0x30062492, [0.3, 0.0, 0.4, 0.95, 0.1, 0.2, 0.0, 0.1], 0.82,
    0x00006db6, [0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.85, 0.3], 0.21
  ),
  固い: _makeDualSense(
    '固い', 2, 2,
    0x10021249, [0.2, 0.1, 0.95, 0.5, 0.4, 0.5, 0.0, 0.0], 0.88,
    0x00001249, [0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.9, 0.2], 0.24
  ),
  開く: _makeDualSense(
    '開く', 1, 1,
    0x0801b6db, [0.5, 0.3, 0.6, 0.3, 0.7, 0.8, 0.0, 0.0], 0.85,
    0x0000b6db, [0.1, 0.1, 0.0, 0.0, 0.2, 0.3, 0.85, 0.2], 0.27
  ),
  閉じる: _makeDualSense(
    '閉じる', 1, 1,
    0x08016db6, [0.4, 0.3, 0.6, 0.3, 0.7, 0.8, 0.0, 0.2], 0.85,
    0x00006db6, [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.9, 0.4], 0.22
  ),
  浮かぶ: _makeDualSense(
    '浮かぶ', 1, 1,
    0x20046db6, [0.5, 0.1, 0.6, 0.3, 0.6, 0.7, 0.0, 0.3], 0.83,
    0x00004924, [0.2, 0.0, 0.0, 0.2, 0.1, 0.3, 0.75, 0.5], 0.29
  ),
  沈む: _makeDualSense(
    '沈む', 1, 1,
    0x20049249, [0.3, 0.1, 0.6, 0.3, 0.6, 0.7, 0.0, 0.6], 0.82,
    0x38076db6, [0.0, 0.0, 0.0, 0.3, 0.1, 0.2, 0.6, 0.75], 0.26
  ),
  揺れる: _makeDualSense(
    '揺れる', 1, 1,
    0x18034924, [0.4, 0.2, 0.6, 0.7, 0.85, 0.6, 0.0, 0.1], 0.85,
    0x000036db, [0.1, 0.0, 0.0, 0.1, 0.3, 0.2, 0.85, 0.3], 0.26
  ),
  響く: _makeDualSense(
    '響く', 1, 1,
    0x2805b6db, [0.2, 0.95, 0.4, 0.5, 0.5, 0.7, 0.0, 0.2], 0.86,
    0x0000b6db, [0.0, 0.2, 0.0, 0.1, 0.2, 0.3, 0.85, 0.4], 0.27
  ),
  掴む: _makeDualSense(
    '掴む', 1, 1,
    0x18039249, [0.3, 0.1, 0.95, 0.6, 0.9, 0.5, 0.0, 0.0], 0.89,
    0x00009249, [0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.9, 0.2], 0.23
  ),
};

export function getDualSenseEntry(word: string): DualSenseHthEntry | undefined {
  return DUAL_SENSE_LEXICON[word];
}

export function poincareHyperbolicDistance(u: HTHEntry, v: HTHEntry, tauTree: number = 0.0): number {
  const lca = lcaDepth30Bit(u.octree30, v.octree30);
  if (tauTree <= 1e-6) {
    return 1.0 - lca / 10.0;
  }
  let num = 0.0;
  let den = 0.0;
  const inv2Var = 1.0 / (2.0 * tauTree * tauTree);
  for (let k = 0; k <= 10; k++) {
    const diff = k - lca;
    const w = Math.exp(-diff * diff * inv2Var);
    num += (1.0 - k / 10.0) * w;
    den += w;
  }
  return den > 1e-12 ? num / den : 1.0 - lca / 10.0;
}

export interface LaplacianOntologyTable {
  diffused8d: Float64Array; // 256 * 8
  octantPaths: number[][]; // 256 * 3
}

let cachedOntology: LaplacianOntologyTable | null = null;

export function buildDiffusedClusterTable(alpha: number = 0.85): LaplacianOntologyTable {
  if (cachedOntology && alpha === 0.85) return cachedOntology;

  const octPrototypes = [
    [0.05, 0.05, 0.05, 0.05, 0.10, 0.45, 0.95, 0.20],
    [0.15, 0.05, 0.15, 0.05, 0.15, 0.85, 0.80, 0.25],
    [0.55, 0.20, 0.75, 0.20, 0.25, 0.75, 0.30, 0.20],
    [0.45, 0.45, 0.55, 0.65, 0.95, 0.50, 0.15, 0.10],
    [0.75, 0.55, 0.90, 0.45, 0.55, 0.60, 0.05, 0.45],
    [0.95, 0.90, 0.50, 0.40, 0.45, 0.40, 0.05, 0.35],
    [0.65, 0.40, 0.85, 0.95, 0.60, 0.35, 0.05, 0.20],
    [0.30, 0.25, 0.35, 0.25, 0.10, 0.55, 0.15, 0.95],
  ];

  const baseClusters: { path: [number, number, number]; seed: number[] }[] = [];
  for (let o1 = 0; o1 < 8; o1++) {
    for (let o2 = 0; o2 < 8; o2++) {
      const b1 = octPrototypes[o1]!;
      const b2 = octPrototypes[o2]!;
      const s = new Array<number>(8);
      for (let d = 0; d < 8; d++) s[d] = Math.min(1.0, Math.max(0.0, 0.72 * b1[d]! + 0.28 * b2[d]!));
      baseClusters.push({ path: [o1, o2, (o1 * 3 + o2 * 5) % 8], seed: s });
    }
  }

  const N = 256;
  const adj = new Float64Array(N * N);
  const octantPaths: number[][] = [];
  const seedValues = new Float64Array(N * 8);

  for (let i = 0; i < 64; i++) {
    const c = baseClusters[i]!;
    octantPaths.push(c.path);
    for (let d = 0; d < 8; d++) seedValues[i * 8 + d] = c.seed[d]!;
  }

  for (let i = 0; i < 64; i++) {
    const pi = baseClusters[i]!.path;
    for (let j = i + 1; j < 64; j++) {
      const pj = baseClusters[j]!.path;
      let w = 0.02;
      if (pi[0] === pj[0]) w = pi[1] === pj[1] ? 1.0 : 0.45;
      else if (pi[1] === pj[0] || pi[0] === pj[1]) w = 0.25;
      adj[i * N + j] = adj[j * N + i] = w;
    }
  }

  for (let u = 64; u < N; u++) {
    const pa = u % 64;
    const pb = (u * 7 + 13) % 64;
    const pc = (u * 19 + 29) % 64;
    adj[u * N + pa] = adj[pa * N + u] = 0.80;
    adj[u * N + pb] = adj[pb * N + u] = 0.50;
    adj[u * N + pc] = adj[pc * N + u] = 0.25;
    const prev = 64 + ((u - 64 - 1) % 192);
    adj[u * N + prev] = adj[prev * N + u] = 0.30;
    const pathA = baseClusters[pa]!.path;
    const pathB = baseClusters[pb]!.path;
    octantPaths.push([pathA[0], pathB[0], (u + pathA[2]) % 8]);
  }

  const degInvSqrt = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let sum = 0.0;
    for (let j = 0; j < N; j++) sum += adj[i * N + j]!;
    degInvSqrt[i] = sum > 1e-12 ? 1.0 / Math.sqrt(sum) : 0.0;
  }

  const sys = new Float64Array(N * N);
  const rhs = new Float64Array(N * 8);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const sn = degInvSqrt[i]! * adj[i * N + j]! * degInvSqrt[j]!;
      sys[i * N + j] = (i === j ? 1.0 : 0.0) - alpha * sn;
    }
    for (let d = 0; d < 8; d++) rhs[i * 8 + d] = (1.0 - alpha) * seedValues[i * 8 + d]!;
  }

  // Gaussian elimination with partial pivoting
  for (let i = 0; i < N; i++) {
    let maxR = i;
    let maxV = Math.abs(sys[i * N + i]!);
    for (let k = i + 1; k < N; k++) {
      const v = Math.abs(sys[k * N + i]!);
      if (v > maxV) { maxV = v; maxR = k; }
    }
    if (maxR !== i) {
      for (let j = i; j < N; j++) {
        const t = sys[i * N + j]!; sys[i * N + j] = sys[maxR * N + j]!; sys[maxR * N + j] = t;
      }
      for (let d = 0; d < 8; d++) {
        const t = rhs[i * 8 + d]!; rhs[i * 8 + d] = rhs[maxR * 8 + d]!; rhs[maxR * 8 + d] = t;
      }
    }
    const piv = sys[i * N + i]!;
    for (let k = i + 1; k < N; k++) {
      const f = sys[k * N + i]! / piv;
      for (let j = i + 1; j < N; j++) sys[k * N + j]! -= f * sys[i * N + j]!;
      for (let d = 0; d < 8; d++) rhs[k * 8 + d]! -= f * rhs[i * 8 + d]!;
    }
  }

  const solved = new Float64Array(N * 8);
  for (let i = N - 1; i >= 0; i--) {
    const piv = sys[i * N + i]!;
    for (let d = 0; d < 8; d++) {
      let s = rhs[i * 8 + d]!;
      for (let j = i + 1; j < N; j++) s -= sys[i * N + j]! * solved[j * 8 + d]!;
      solved[i * 8 + d] = s / piv;
    }
  }

  const colMin = new Float64Array(8).fill(Infinity);
  const colMax = new Float64Array(8).fill(-Infinity);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < 8; d++) {
      const v = solved[i * 8 + d]!;
      if (v < colMin[d]!) colMin[d] = v;
      if (v > colMax[d]!) colMax[d] = v;
    }
  }

  const diffused8d = new Float64Array(N * 8);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < 8; d++) {
      const denom = (colMax[d]! - colMin[d]!) > 1e-9 ? (colMax[d]! - colMin[d]!) : 1.0;
      const norm = (solved[i * 8 + d]! - colMin[d]!) / denom;
      const finalVal = i < 64 ? 0.65 * seedValues[i * 8 + d]! + 0.35 * norm : norm;
      diffused8d[i * 8 + d] = Math.min(1.0, Math.max(0.0, finalVal));
    }
  }

  cachedOntology = { diffused8d, octantPaths };
  return cachedOntology;
}

export function synthesizeMimeticHTH(surface: string): HTHEntry {
  const s = new Float64Array(8);
  const add = (idx: number, v: number): void => {
    s[idx] = (s[idx] ?? 0) + v;
  };
  const first = surface.charAt(0);
  // Voiced obstruents (濁音: g, z, d, b)
  if (/[がぎぐげござじずぜぞだぢづでどばびぶべぼガギグゲゴザジズゼゾダヂヅデドバビブベボ]/.test(first)) {
    add(4, 6); add(2, 5); add(6, 4);
  }
  // Voiceless fricatives/affricates (s, sh, h, f)
  if (/[さしすせそはひふへほちつサシスセソハヒフヘホチツ]/.test(first)) {
    add(1, 5); add(2, 4); add(5, 4);
  }
  // Bilabial plosive (半濁音: p)
  if (/[ぱぴぷぺぽパピプペポ]/.test(first)) {
    add(0, 5); add(4, 6);
  }
  // Geminate っ/ッ or ending with り
  if (/[っッ]/.test(surface) || surface.endsWith('り') || surface.endsWith('リ')) {
    add(4, 5);
  }
  // Moraic nasal ん/ン or prolonged ー
  if (/[んンー]/.test(surface)) {
    add(7, 6); add(1, 4);
  }
  // ABAB Reduplication
  if (surface.length >= 4 && surface.slice(0, Math.floor(surface.length / 2)) === surface.slice(Math.floor(surface.length / 2))) {
    add(4, 2);
  }

  const s8 = Array.from(s, (v) => Math.min(1.0, Math.max(0.0, v / 15.0)));
  const s0 = s[0] ?? 0;
  const s1 = s[1] ?? 0;
  const s4 = s[4] ?? 0;
  const s7 = s[7] ?? 0;
  const primOctant = s4 >= 6 ? 3 : s0 >= 5 ? 5 : s1 >= 5 ? 5 : s7 >= 6 ? 7 : 6;
  const octree30 = ((primOctant & 0x7) << 27) | 0x0124924;

  return {
    lemma: surface,
    pos: 3,
    octree30,
    sensory8d: s8,
    concreteness: 0.85,
    clusterId: primOctant * 8 + 3,
    mimeticImmunity: true,
    isMimetic: true,
  };
}

export class HTHLexicon {
  private readonly ontology: LaplacianOntologyTable;

  constructor() {
    this.ontology = buildDiffusedClusterTable(0.85);
  }

  public lookup(lemma: string, posHint?: number): HTHEntry {
    if (this.isMimeticForm(lemma)) {
      return synthesizeMimeticHTH(lemma);
    }
    const code = lemma.length > 0 ? (lemma.codePointAt(0) ?? 0x4e00) : 0x4e00;
    const oct1 = ((this.getPrimaryOctantFromCodepoint(code) % 8) + 8) % 8;
    const oct2 = (((Math.floor(code / 17) ^ (code & 0x7)) % 8) + 8) % 8;
    const clusterId = (((oct1 * 8 + oct2) % 256) + 256) % 256;

    const sensory: number[] = [];
    for (let d = 0; d < 8; d++) {
      sensory.push(this.ontology.diffused8d[clusterId * 8 + d]!);
    }

    const path = this.ontology.octantPaths[clusterId] ?? [oct1, oct2, 0];
    let octree30 = 0;
    for (let lvl = 0; lvl < 10; lvl++) {
      const digit = lvl < 3 ? (path[lvl] ?? 0) : (clusterId * 13 + lvl * 7 + (code & 0x7)) % 8;
      octree30 |= (digit & 0x7) << (3 * (9 - lvl));
    }
    octree30 = (octree30 & 0x3fffffff) >>> 0;

    const isPhysical = oct1 >= 2 && oct1 <= 6;
    let maxS = 0.0;
    for (let d = 0; d < 8; d++) if (sensory[d]! > maxS) maxS = sensory[d]!;
    const concreteness = isPhysical ? 0.90 : maxS * 0.5;

    return {
      lemma,
      pos: posHint ?? (lemma.endsWith('い') ? 2 : lemma.endsWith('る') || lemma.endsWith('す') ? 1 : 0),
      octree30,
      sensory8d: sensory,
      concreteness,
      clusterId,
      mimeticImmunity: false,
      isMimetic: false,
    };
  }

  private isMimeticForm(w: string): boolean {
    if (w.length < 2) return false;
    if (/[っッんンー]/.test(w) && (w.endsWith('と') || w.endsWith('り') || w.endsWith('っと'))) return true;
    if (w.length >= 4 && w.slice(0, Math.floor(w.length / 2)) === w.slice(Math.floor(w.length / 2))) return true;
    return false;
  }

  private getPrimaryOctantFromCodepoint(code: number): number {
    if (code >= 0x3000 && code <= 0x303f) return 1; // CJK symbols & punctuation (e.g. 々)
    if (code >= 0x3040 && code <= 0x309f) return 1; // Hiragana functional
    if (code >= 0x30a0 && code <= 0x30ff) return 5; // Katakana acoustic
    // Standard Unicode CJK Kangxi Radical Block Ranges:
    if (code >= 0x51ab && code <= 0x51dd) return 4; // Radical 15: 冫 (Ice / Cold / Freeze)
    if (code >= 0x6c34 && code <= 0x7060) return 4; // Radical 85: 水/氵 (Water / Fluid / Stream)
    if (code >= 0x706b && code <= 0x7220) return 4; // Radical 86: 火/灬 (Fire / Thermal / Burn)
    if (code >= 0x96e8 && code <= 0x9740) return 4; // Radical 173: 雨 (Rain / Meteorological / Sky)
    if (code >= 0x5c70 && code <= 0x5db0) return 4; // Radical 46: 山 (Mountain / Topography)
    if (code >= 0x77f3 && code <= 0x7920) return 2; // Radical 112: 石 (Stone / Rock / Mineral)
    if (code >= 0x6728 && code <= 0x6c00) return 6; // Radical 75: 木 (Flora / Organic)
    if (code >= 0x8278 && code <= 0x8660) return 6; // Radical 140: 艸/艹 & 142: 虫 (Plants / Insects)
    if (code >= 0x9b5a && code <= 0x9ce0) return 6; // Radical 195: 魚 & 196: 鳥 (Fauna)
    if (code >= 0x99ac && code <= 0x9a6b) return 6; // Radical 187: 馬 (Fauna / Equine)
    if (code >= 0x5fc3 && code <= 0x61f0) return 7; // Radical 61: 心/忄 (Emotion / Affective)
    if (code >= 0x624b && code <= 0x6520) return 3; // Radical 64: 手/扌 (Hand / Action / Kinetic)
    if (code >= 0x8d70 && code <= 0x8f50) return 3; // Radical 156/162: 走/辵/辶 (Motion / Walk / Run)
    if (code >= 0x65e5 && code <= 0x6700) return 5; // Radical 72: 日 (Sun / Light / Time)
    if (code >= 0x76ee && code <= 0x77f0) return 5; // Radical 109: 目 (Eye / Vision / Chromatic)
    if (code >= 0x898b && code <= 0x89e0) return 5; // Radical 147: 見 (See / Gaze)
    if (code >= 0x91d1 && code <= 0x93f0) return 2; // Radical 167: 金 (Metal / Solid)
    if (code >= 0x9580 && code <= 0x95f0) return 1; // Radical 169: 門 (Architecture / Boundary)
    if (code >= 0x8a00 && code <= 0x8c30) return 0; // Radical 149: 言 (Language / Formal logic)
    const base = code >= 0x4e00 ? Math.floor((code - 0x4e00) / 2500) : Math.floor(code / 1000);
    return ((base % 8) + 8) % 8;
  }

  public applyOpen01PolysemyDamping(u: HTHEntry, v: HTHEntry): {
    effectiveSensory: number[];
    state: 'literal' | 'poetic_metaphor' | 'dead_metaphor';
  } {
    let dot = 0.0;
    let normU = 0.0;
    let normV = 0.0;
    for (let d = 0; d < 8; d++) {
      dot += u.sensory8d[d]! * v.sensory8d[d]!;
      normU += u.sensory8d[d]! * u.sensory8d[d]!;
      normV += v.sensory8d[d]! * v.sensory8d[d]!;
    }
    const denom = Math.sqrt(normU * normV);
    const gBridge = denom > 1e-9 ? dot / denom : 0.0;

    if (u.concreteness >= 0.5 && v.concreteness >= 0.5) {
      return { effectiveSensory: [...v.sensory8d], state: 'literal' };
    }
    if (u.concreteness < 0.5 && gBridge >= 0.35) {
      return { effectiveSensory: [...v.sensory8d], state: 'poetic_metaphor' };
    }
    if (u.concreteness < 0.3 && gBridge < 0.35) {
      const damped = v.sensory8d.map((val) => val * u.concreteness);
      return { effectiveSensory: damped, state: 'dead_metaphor' };
    }
    const scale = Math.max(u.concreteness, gBridge);
    return { effectiveSensory: v.sensory8d.map((x) => x * scale), state: 'dead_metaphor' };
  }

  public segmentText(text: string): TokenAnalysis[] {
    const rawTokens: { surface: string; isPunct: boolean }[] = [];
    let cur = '';
    let curType = -1; // 0=punct, 1=kanji, 2=hiragana, 3=katakana, 4=other

    const getCharType = (c: string): number => {
      if (/[、。！？!?「」『』（）…―──〜~\s]/.test(c)) return 0;
      const cp = c.codePointAt(0) ?? 0;
      if (cp >= 0x4e00 && cp <= 0x9fff) return 1;
      if (cp >= 0x3040 && cp <= 0x309f) return 2;
      if (cp >= 0x30a0 && cp <= 0x30ff) return 3;
      return 4;
    };

    for (const ch of text) {
      const ct = getCharType(ch);
      if (ct === 0) {
        if (cur.length > 0) { rawTokens.push({ surface: cur, isPunct: false }); cur = ''; }
        rawTokens.push({ surface: ch, isPunct: true });
        curType = -1;
      } else {
        if (curType !== -1 && curType !== ct && (curType === 1 && ct !== 2)) {
          rawTokens.push({ surface: cur, isPunct: false });
          cur = ch;
        } else {
          cur += ch;
        }
        curType = ct;
      }
    }
    if (cur.length > 0) rawTokens.push({ surface: cur, isPunct: false });

    const tokens: TokenAnalysis[] = [];
    const negPattern = /(ない|ぬ|ず|なし|なかった|なく)$/;
    const specPattern = /(だろう|らしい|ようだ|そうだ|まい)$/;

    for (let i = 0; i < rawTokens.length; i++) {
      const item = rawTokens[i]!;
      if (item.isPunct) continue;
      const s = item.surface;
      const isNeg = negPattern.test(s) || /^[不非無未]/.test(s);
      const isSpec = specPattern.test(s);
      const alpha = isNeg ? -0.5 : isSpec ? 0.6 : 1.0;
      const entry = this.lookup(s);

      tokens.push({
        surface: s,
        lemma: s,
        pos: entry.pos,
        entry,
        effectiveSensory8d: [...entry.sensory8d],
        isNegated: isNeg,
        modalityAlpha: alpha,
        polysemyState: 'literal',
        isMimetic: entry.isMimetic ?? false,
        mimeticImmunity: entry.mimeticImmunity ?? false,
      });
    }

    // Apply OPEN-01 dependency pairs (Noun -> Predicate)
    for (let i = 0; i < tokens.length - 1; i++) {
      const u = tokens[i]!;
      const v = tokens[i + 1]!;
      if (u.pos === 0 && (v.pos === 1 || v.pos === 2)) {
        const { effectiveSensory, state } = this.applyOpen01PolysemyDamping(u.entry, v.entry);
        v.effectiveSensory8d = effectiveSensory;
        v.polysemyState = state;
      }
    }

    // Apply OPEN-02 Dual Negation
    for (const t of tokens) {
      if (t.isNegated) {
        t.effectiveSensory8d = t.effectiveSensory8d.map((val) => val * Math.max(0, t.modalityAlpha));
      }
    }

    return tokens;
  }

  public separateNarrativeAndDialogue(text: string): {
    narrativeText: string;
    dialogueText: string;
    dialogueRatio: number;
    alternationCount: number;
  } {
    let narrative = '';
    let dialogue = '';
    let inQuote = false;
    let alternations = 0;

    for (let i = 0; i < text.length; i++) {
      const ch = text.charAt(i);
      if (ch === '「' || ch === '『') {
        inQuote = true;
        alternations++;
      } else if (ch === '」' || ch === '』') {
        inQuote = false;
        alternations++;
      } else {
        if (inQuote) dialogue += ch;
        else narrative += ch;
      }
    }

    const totalLen = Math.max(1, narrative.length + dialogue.length);
    return {
      narrativeText: narrative,
      dialogueText: dialogue,
      dialogueRatio: dialogue.length / totalLen,
      alternationCount: alternations,
    };
  }

  public getConcreteness(word: string): number {
    const entry = this.lookup(word);
    return entry.concreteness;
  }

  public resolveDualSenseHth(word: string, contextWord: string): DualSenseResolutionResult {
    return resolveDualSenseHth(word, contextWord, this);
  }
}

export function resolveDualSenseHth(
  word: string,
  contextWord: string,
  lexicon?: HTHLexicon
): DualSenseResolutionResult {
  const lex = lexicon ?? new HTHLexicon();
  const cU = lex.getConcreteness(contextWord);
  const entry = DUAL_SENSE_LEXICON[word];

  let selectedSlot: 'primary_physical' | 'secondary_abstract';
  let selectedHth: bigint;
  let pri64: bigint;
  let sec64: bigint;
  let effectiveSensory: number[];

  if (entry) {
    if (cU >= 0.52) {
      selectedSlot = 'primary_physical';
      selectedHth = entry.primary_physical_hth;
      effectiveSensory = [...entry.primary_entry.sensory8d];
    } else {
      selectedSlot = 'secondary_abstract';
      selectedHth = entry.secondary_abstract_hth;
      effectiveSensory = [...entry.secondary_entry.sensory8d];
    }
    pri64 = entry.primary_physical_hth;
    sec64 = entry.secondary_abstract_hth;
  } else {
    const tok = lex.lookup(word);
    const packed = pack64bitHth(tok.pos, tok.octree30, packLower32Sensory(tok.sensory8d));
    selectedSlot = cU >= 0.52 ? 'primary_physical' : 'secondary_abstract';
    selectedHth = packed;
    pri64 = packed;
    sec64 = packed;
    effectiveSensory = [...tok.sensory8d];
  }

  return {
    selectedSlot,
    selected_slot: selectedSlot,
    selectedHth,
    selected_hth: selectedHth,
    full128bit: [pri64, sec64],
    full_128bit: [pri64, sec64],
    primary64bit: pri64,
    primary_64bit: pri64,
    secondary64bit: sec64,
    secondary_64bit: sec64,
    contextConcreteness: cU,
    context_concreteness: cU,
    effectiveSensory,
    effective_sensory: effectiveSensory,
    word,
    contextWord,
    context_word: contextWord,
  };
}

export function extractMusicFunctorSignals(notes: MusicNoteEvent[]): number[][] {
  if (notes.length === 0) return [[0, 0, 0, 0]];
  const sorted = [...notes].sort((a, b) => a.startTime - b.startTime);
  const minTime = sorted[0]!.startTime;
  let maxTime = minTime;
  for (const n of sorted) {
    const end = n.startTime + n.duration;
    if (end > maxTime) maxTime = end;
  }
  const totalDuration = Math.max(0.1, maxTime - minTime);
  const M = Math.max(8, Math.min(64, Math.round(totalDuration * 2.0)));
  const dt = totalDuration / M;

  const result: number[][] = [];
  for (let m = 0; m < M; m++) {
    const tStart = minTime + m * dt;
    const tEnd = tStart + dt;
    const active = sorted.filter((n) => n.startTime < tEnd && n.startTime + n.duration > tStart);

    if (active.length === 0) {
      result.push([0.0, 0.0, 1.0, 0.0]); // Pure void/rest window
      continue;
    }

    let pitchSum = 0.0;
    let velSum = 0.0;
    for (const n of active) {
      pitchSum += n.pitch;
      velSum += n.velocity;
    }
    const meanPitch = pitchSum / active.length;
    const meanVel = velSum / active.length;

    let varPitch = 0.0;
    for (const n of active) varPitch += (n.pitch - meanPitch) * (n.pitch - meanPitch);
    const stdPitch = Math.sqrt(varPitch / active.length);

    // s1: Flux / Information density (pitch variance & dynamics)
    const s1 = Math.min(1.0, Math.max(0.0, (meanPitch - 48.0) / 48.0 * 0.5 + stdPitch / 24.0 * 0.3 + meanVel / 127.0 * 0.2));

    // s2: Lattice / Rhythm regularity (IOI autocorrelation / note density)
    const density = Math.min(1.0, active.length / 6.0);
    const s2 = density;

    // s3: Void / Harmonic Rest (unoccupied time in window)
    let occupied = 0.0;
    for (const n of active) occupied += Math.min(tEnd, n.startTime + n.duration) - Math.max(tStart, n.startTime);
    const voidRatio = Math.min(1.0, Math.max(0.0, 1.0 - occupied / dt));
    const s3 = voidRatio;

    // s4: Shock / Interval jumps
    let maxJump = 0.0;
    for (let i = 0; i < active.length - 1; i++) {
      const j = Math.abs(active[i + 1]!.pitch - active[i]!.pitch);
      if (j > maxJump) maxJump = j;
    }
    const s4 = Math.min(1.0, maxJump / 12.0);

    result.push([s1, s2, s3, s4]);
  }

  return result;
}
