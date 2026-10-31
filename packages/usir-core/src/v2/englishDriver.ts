/**
 * USIR v3.0 Cross-Linguistic Isomorphic Driver (Gap 7: usir-driver-en).
 *
 * Implements 1-to-1 structural isomorphism between Japanese and English:
 * 1. Kango (漢語) <-> Latinate / Greek abstract words & formal/legal markers
 * 2. Wago (和語) <-> Anglo-Saxon concrete monosyllabic/disyllabic stems
 * 3. Mora Cadence <-> English Syllable Count & Stress Rhythm
 * 4. OPEN-02 & OPEN-05 Isomorphism (Apophatic Negation & Paratactic Subtraction)
 * 5. English AI Slop / Dead Metaphor Detection
 * 6. 24D Symplectic Extraction & 5-Phase Landau Classification
 */

import { extract24dV2 } from './waveletHermiteEncoder.js';

// ============================================================================
// Lexical & Morphological Isomorphism Tables
// ============================================================================

export const LATINATE_SUFFIXES: readonly string[] = [
  'tion', 'sion', 'ity', 'ment', 'ence', 'ance', 'ism', 'ology',
  'ature', 'tude', 'ous', 'ive', 'al', 'ical', 'ary', 'ate',
  'ian', 'ible', 'able', 'ise', 'ize', 'ant', 'ent', 'ic',
];

export const LEGAL_MATH_MARKERS: ReadonlySet<string> = new Set([
  'shall', 'pursuant', 'thereof', 'herein', 'hereto', 'notwithstanding',
  'whereas', 'hereunder', 'foregoing', 'inasmuch', 'iff', 'compact',
  'homomorphism', 'subspace', 'isomorphism', 'topology', 'diffeomorphism',
  'manifold', 'functor', 'symplectic', 'lemma', 'theorem', 'corollary',
  'definition', 'proposition', 'axiom', 'provided', 'jurisdiction',
  'statute', 'obligation', 'indemnity', 'representation', 'warranty',
  'hereby', 'therein', 'codomain', 'homeomorphic', 'continuous', 'bounded',
  'homeomorphism', 'affine', 'bijection', 'submanifold', 'vector',
]);

export const SAXON_CONCRETE_STEMS: ReadonlySet<string> = new Set([
  'stone', 'rain', 'wind', 'cold', 'dark', 'water', 'leaf', 'bone',
  'shadow', 'light', 'mud', 'wood', 'hill', 'river', 'sky', 'dust',
  'grass', 'snow', 'fire', 'blood', 'iron', 'glass', 'bell', 'fog',
  'silence', 'breath', 'tree', 'night', 'sea', 'earth', 'sun', 'moon',
  'hand', 'eye', 'foot', 'walk', 'fall', 'run', 'stand', 'lay',
  'sit', 'hear', 'see', 'door', 'road', 'wall', 'bread', 'fish',
  'boat', 'flesh', 'salt', 'ash', 'bark', 'root', 'bird', 'frost',
  'ice', 'path', 'smoke', 'trout', 'lake', 'pool', 'stream', 'deep',
  'black', 'white', 'green', 'red', 'grey', 'gray', 'hot', 'still',
]);

export const APOPHATIC_NEGATIONS: ReadonlySet<string> = new Set([
  'not', 'never', 'neither', 'nor', 'nothing', 'nowhere', 'without',
  'none', 'no', 'nobody', 'scarcely', 'barely', 'void', 'empty',
  'hollow', 'unseen', 'silent', 'silence',
]);

export const AI_SLOP_MARKERS: readonly string[] = [
  'delve', 'tapestry', 'testament', 'underscores', 'multifaceted',
  'pivotal', 'crucial', 'moreover', 'furthermore', 'it is important to note',
  'in conclusion', 'comprehensive framework', 'seamless integration', 'foster',
  'navigate the complexities', 'beacon', 'vital', 'harness', 'embark',
  'paramount', 'interplay', 'vibrant', 'rich tapestry', 'testament to',
  'delve into', 'poised to', 'elevate', 'ever-evolving', "in today's world",
  'dynamic landscape', 'shed light', 'in summary',
];

export const SHOCK_WORDS: ReadonlySet<string> = new Set([
  'blood', 'shriek', 'darkness', 'abyss', 'terror', 'crash', 'sudden',
  'suddenly', 'fury', 'death', 'madness', 'horror', 'scream', 'rupture',
  'burst', 'wound', 'shatter', 'agony', 'shudder', 'climax', 'struck',
  'blast', 'frightful', 'trembling', 'yawning', 'violent', 'panic',
  'shattered', 'shrieked', 'crashed', 'screamed', 'ripped',
]);

export const ADVERB_ADJECTIVE_SUFFIXES: readonly string[] = [
  'ly', 'ful', 'less', 'able', 'ible', 'ous', 'ious', 'ive',
  'ish', 'ic', 'al', 'est', 'er',
];

// ============================================================================
// Syllable Counter & Clause Segmentation
// ============================================================================

export function countSyllables(word: string): number {
  let w = word.toLowerCase().replace(/[^a-z]/g, '');
  if (!w) return 0;
  if (w.length <= 3) return 1;

  // Separate common vowel hiatuses that form 2 syllables
  w = w.replace(/([iu])([aeo])/g, '$1 $2');
  w = w.replace(/([eo])([iu])/g, '$1 $2');

  const vowelRuns = w.match(/[aeiouy]+/g);
  let count = vowelRuns ? vowelRuns.length : 1;

  if (w.endsWith('e') && !w.endsWith('le') && !w.endsWith('ee')) {
    count--;
  }

  if (w.endsWith('ed') && !w.endsWith('ded') && !w.endsWith('ted')) {
    if (count > 1) count--;
  }

  if (w.endsWith('es') && !(w.endsWith('ses') || w.endsWith('zes') || w.endsWith('ches') || w.endsWith('shes'))) {
    if (count > 1) count--;
  }

  if (w.endsWith('ism')) {
    count++;
  }

  return Math.max(count, 1);
}

export function splitEnglishClauses(text: string): [string, string][] {
  const tokens = text.split(/([.\n?!;:—–,]+)/);
  const clauses: [string, string][] = [];

  let i = 0;
  while (i < tokens.length) {
    const c = tokens[i]?.trim() ?? '';
    const delim = tokens[i + 1]?.trim() ?? '';
    if ((c.match(/[a-zA-Z]+/g) ?? []).length >= 1) {
      clauses.push([c, delim]);
    }
    i += 2;
  }

  if (clauses.length === 0) {
    return [[text.trim() || 'silence', '.']];
  }
  return clauses;
}

// ============================================================================
// 4D Base Signal Extraction (Flux, Lattice, Void, Shock)
// ============================================================================

export function extractClauseSignalsEnglish(
  clause: string,
  delim: string = '',
  _prevClause?: string
): number[] {
  const words = (clause.toLowerCase().match(/\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b/g) ?? []);
  const nTok = words.length;
  if (nTok === 0) {
    return [0.1, 0.5, 0.5, 0.0];
  }

  // 1. Channel 1: Flux (Dynamic range, radial entropy, bigram surprise)
  const lengths = words.map((w) => w.length);
  const totChars = lengths.reduce((acc, l) => acc + l, 0);
  let hRadial = 0.20;
  if (totChars > 0 && nTok > 1) {
    const pLen = lengths.map((l) => l / totChars);
    const sumEnt = pLen.reduce((acc, p) => acc + (p > 0 ? p * Math.log2(p + 1e-12) : 0), 0);
    hRadial = -sumEnt / Math.log2(Math.max(nTok, 2));
  }

  const cleanChars = words.join('');
  let surprise = 0.15;
  if (cleanChars.length >= 3) {
    const bigrams: string[] = [];
    for (let i = 0; i < cleanChars.length - 1; i++) {
      bigrams.push(cleanChars.substring(i, i + 2));
    }
    const counts = new Map<string, number>();
    for (const bg of bigrams) {
      counts.set(bg, (counts.get(bg) ?? 0) + 1);
    }
    const nBg = bigrams.length;
    let bgEntropy = 0;
    for (const c of counts.values()) {
      const p = c / nBg;
      bgEntropy -= p * Math.log2(p);
    }
    surprise = bgEntropy / Math.log2(Math.max(nBg, 2));
  }

  const syllables = words.map(countSyllables);
  const totSyl = syllables.reduce((a, b) => a + b, 0);
  const s1 = Math.min(1.0, Math.max(0.0, 0.35 * hRadial + 0.35 * surprise + 0.30 * Math.min(1.0, totSyl / 12.0)));

  // 2. Channel 2: Lattice (Latinate/Greek abstract words + Legal/Math markers + Cadence)
  const latinateCount = words.filter(
    (w) => LATINATE_SUFFIXES.some((sfx) => w.endsWith(sfx)) || LEGAL_MATH_MARKERS.has(w)
  ).length;
  const latinateRatio = latinateCount / nTok;

  const legalCount = words.filter((w) => LEGAL_MATH_MARKERS.has(w)).length;
  const legalRatio = legalCount / nTok;

  const cadenceDist = Math.min(...[4, 8, 10, 12, 16].map((c) => Math.abs(totSyl - c)));
  const cadenceReg = Math.max(0.0, 1.0 - cadenceDist / 8.0);

  let s2: number;
  if (legalCount >= 1 || latinateRatio > 0.25) {
    s2 = Math.min(1.0, Math.max(0.0, 0.40 * latinateRatio + 0.45 * legalRatio + 0.15 * cadenceReg + 0.30));
  } else {
    s2 = Math.min(1.0, Math.max(0.0, 0.30 * latinateRatio + 0.50 * cadenceReg + 0.20 * (1.0 - surprise)));
  }

  // 3. Channel 3: Void [間] (Saxon concreteness * subtraction rate * apophatic negation)
  const saxonCount = words.filter((w) => SAXON_CONCRETE_STEMS.has(w)).length;
  const modCount = words.filter(
    (w) => ADVERB_ADJECTIVE_SUFFIXES.some((sfx) => w.endsWith(sfx)) && !SAXON_CONCRETE_STEMS.has(w)
  ).length;
  const subtractionRate = Math.exp(-0.50 * modCount);

  const hasApophatic = words.some((w) => APOPHATIC_NEGATIONS.has(w));
  const apophaticFactor = hasApophatic ? 1.35 : 1.0;

  const clauseLower = clause.toLowerCase();
  const slopHits = AI_SLOP_MARKERS.filter((slop) => clauseLower.includes(slop)).length;
  const slopPenalty = Math.max(0.0, 1.0 - 0.70 * slopHits);
  const latinPenalty = Math.max(0.1, 1.0 - 0.60 * latinateRatio);

  let baseVoid: number;
  if (saxonCount >= 1 || hasApophatic) {
    baseVoid = 0.45 + 0.35 * Math.min(saxonCount / 2.0, 1.0);
  } else {
    baseVoid = 0.15;
  }

  const s3 = Math.min(
    1.0,
    Math.max(
      0.0,
      baseVoid * subtractionRate * apophaticFactor * slopPenalty * latinPenalty
    )
  );

  // 4. Channel 4: Shock (Visceral rupture + Sudden caesura + Syllable leap)
  const shockCount = words.filter((w) => SHOCK_WORDS.has(w)).length;
  const shockDensity = Math.min(1.0, shockCount * 0.45);

  const puncShock = delim.includes('!') || delim.includes('—') || delim.includes('--') ? 0.35 : 0.0;
  const hasAdversative = ['but', 'yet', 'suddenly', 'however', 'though'].includes(words[0] ?? '');
  const advBoost = hasAdversative ? 0.20 : 0.0;

  let s4: number;
  if (shockCount >= 1 && puncShock > 0) {
    s4 = Math.min(1.0, Math.max(0.0, 0.50 * shockDensity + 0.30 * puncShock + 0.35));
  } else if (shockCount >= 1) {
    s4 = Math.min(1.0, Math.max(0.0, 0.60 * shockDensity + 0.20 * puncShock + 0.20 * advBoost));
  } else {
    s4 = Math.min(0.25, Math.max(0.0, 0.40 * puncShock + 0.30 * advBoost));
  }

  if (slopHits >= 1 || legalCount >= 1) {
    s4 = Math.min(s4 * 0.20, 0.06);
  }

  return [s1, s2, s3, s4];
}

export function extractEnglishSignals(text: string): number[][] {
  const clauses = splitEnglishClauses(text);
  const signals: number[][] = [];
  for (let i = 0; i < clauses.length; i++) {
    const [c, d] = clauses[i]!;
    const prevC = i > 0 ? clauses[i - 1]?.[0] : undefined;
    signals.push(extractClauseSignalsEnglish(c, d, prevC));
  }
  return signals;
}

// ============================================================================
// Symplectic 24D Extraction
// ============================================================================

export function extractUsir24dEnglish(
  text: string,
  useMca: boolean = false
): Float64Array {
  const signals = extractEnglishSignals(text);
  return extract24dV2(signals, useMca);
}

// ============================================================================
// English Landau 5-Phase Classification
// ============================================================================

export function classifyEnglishPhase(text: string): string {
  const textLower = text.toLowerCase();
  const words = textLower.match(/\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b/g) ?? [];
  const nTok = words.length;
  if (nTok === 0) return 'III_Void';

  const clauses = splitEnglishClauses(text);
  const nClauses = Math.max(clauses.length, 1);

  const z = extractUsir24dEnglish(text, false);
  const q = Array.from(z.slice(0, 12));

  // 1. AI Slop Detection (V_Deadzone Gate)
  const slopCount = AI_SLOP_MARKERS.filter((marker) => textLower.includes(marker)).length;
  if (slopCount >= 2) {
    return 'V_Deadzone';
  }

  // 2. Legal / Mathematical Axiom Detection (I_Crystalline Gate)
  const legalCount = words.filter((w) => LEGAL_MATH_MARKERS.has(w)).length;
  const latinateCount = words.filter((w) => LATINATE_SUFFIXES.some((sfx) => w.endsWith(sfx))).length;
  const latinateRatio = latinateCount / nTok;

  if (legalCount >= 2 || (latinateRatio >= 0.20 && legalCount >= 1)) {
    if (q[3]! < 0.25 && q[11]! < 0.20) {
      return 'I_Crystalline';
    }
  }

  // 3. Purple Prose & Excessive Ornamentation (IV_Turbulent Gate)
  const modCount = words.filter(
    (w) => ADVERB_ADJECTIVE_SUFFIXES.some((sfx) => w.endsWith(sfx)) && !SAXON_CONCRETE_STEMS.has(w)
  ).length;
  const modDensity = modCount / nClauses;
  if (modDensity >= 1.60 && q[10]! < 0.25) {
    return 'IV_Turbulent';
  }

  // 4. Masterpiece Climax Prose (II_Critical Gate)
  const shockCount = words.filter((w) => SHOCK_WORDS.has(w)).length;
  const hasPuncShock = clauses.some(([_, d]) => d.includes('!') || d.includes('—'));
  if ((shockCount >= 1 || hasPuncShock) && q[11]! >= 0.20) {
    return 'II_Critical';
  }

  // 5. Minimalist Subtraction & Apophatic Stillness (III_Void Gate)
  const saxonCount = words.filter((w) => SAXON_CONCRETE_STEMS.has(w)).length;
  if (q[10]! >= 0.40 && modDensity <= 1.25 && saxonCount >= 2) {
    return 'III_Void';
  }

  return 'V_Deadzone';
}
