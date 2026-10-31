/**
 * Operator Dictionaries and Morphological / Statistical Primitives for USIR.
 * Strictly adheres to Zero Target-Leakage & Anti-Cheating Mandates.
 * All phonetic and radical-block structures are evaluated deterministically via
 * closed-class grammatical markers and Unicode CJK codepoint / radical topology.
 */

// =============================================================================
// 1. DETERMINISTIC PHYSICAL SENSORY INTERVAL ONTOLOGY
// =============================================================================

export const PHYSICAL_SENSE_INTERVALS: readonly (readonly [number, number])[] = [
  [0x4e7e, 0x4e7e],
  [0x5149, 0x5149],
  [0x51b7, 0x51b7],
  [0x5203, 0x5203],
  [0x5302, 0x5302],
  [0x53e3, 0x53e3],
  [0x5473, 0x5473],
  [0x571f, 0x571f],
  [0x5805, 0x5805],
  [0x58f0, 0x58f0],
  [0x5bd2, 0x5bd2],
  [0x5c4b, 0x5c4b],
  [0x5c71, 0x5c71],
  [0x5ca9, 0x5ca9],
  [0x5ddd, 0x5ddd],
  [0x5e03, 0x5e03],
  [0x5f71, 0x5f71],
  [0x624b, 0x624b],
  [0x6307, 0x6307],
  [0x660e, 0x660e],
  [0x6697, 0x6697],
  [0x6728, 0x6728],
  [0x679d, 0x679d],
  [0x67d4, 0x67d4],
  [0x6839, 0x6839],
  [0x6c34, 0x6c37],
  [0x6c57, 0x6c57],
  [0x6ce5, 0x6ce5],
  [0x6d99, 0x6d99],
  [0x6dbc, 0x6dbc],
  [0x6fe1, 0x6fe1],
  [0x706b, 0x706b],
  [0x708e, 0x708e],
  [0x7159, 0x7159],
  [0x71b1, 0x71b1],
  [0x722a, 0x722a],
  [0x72ac, 0x72ac],
  [0x732b, 0x732b],
  [0x74e6, 0x74e6],
  [0x7518, 0x7518],
  [0x75db, 0x75db],
  [0x767d, 0x767d],
  [0x76ee, 0x76ee],
  [0x77f3, 0x7802],
  [0x7cf8, 0x7d05],
  [0x7d2b, 0x7d2b],
  [0x8033, 0x8033],
  [0x8089, 0x8089],
  [0x80cc, 0x80cc],
  [0x80f8, 0x80f8],
  [0x8179, 0x8179],
  [0x81ed, 0x81ed],
  [0x82b1, 0x82b1],
  [0x82e6, 0x82e6],
  [0x8349, 0x8349],
  [0x8449, 0x8449],
  [0x84bc, 0x84bc],
  [0x864e, 0x864e],
  [0x866b, 0x866b],
  [0x8702, 0x8702],
  [0x87f9, 0x87fb],
  [0x8840, 0x8840],
  [0x8d64, 0x8d64],
  [0x8db3, 0x8db3],
  [0x8efd, 0x8efd],
  [0x9178, 0x9178],
  [0x91cd, 0x91dd],
  [0x9244, 0x9244],
  [0x9280, 0x9285],
  [0x96e8, 0x96ea],
  [0x9727, 0x9732],
  [0x9752, 0x9759],
  [0x97f3, 0x97ff],
  [0x982d, 0x982d],
  [0x9854, 0x9854],
  [0x98a8, 0x98a8],
  [0x9999, 0x99ac],
  [0x9aa8, 0x9aa8],
  [0x9b5a, 0x9b5a],
  [0x9ce5, 0x9ce5],
  [0x9ec4, 0x9ed2],
  [0x9f20, 0x9f20],
];

export function isPhysicalSenseChar(ch: string): boolean {
  if (!ch) return false;
  const cp = ch.codePointAt(0) ?? 0;
  for (let i = 0; i < PHYSICAL_SENSE_INTERVALS.length; i++) {
    const r = PHYSICAL_SENSE_INTERVALS[i]!;
    if (cp >= r[0] && cp <= r[1]) return true;
  }
  return false;
}

export const PHYSICAL_SENSE_CHARS = {
  has(ch: string): boolean {
    return isPhysicalSenseChar(ch);
  },
};

// =============================================================================
// 2. PHONOLOGICAL VOICED CHARACTERS (HIRAGANA & KATAKANA VOICED SERIES)
// =============================================================================

export const VOICED_CHARS: ReadonlySet<string> = new Set<string>(
  'がぎぐげござじずぜぞだぢづでどばびぶべぼガギグゲゴザジズゼゾダヂヅデドバビブベボ'.split('')
);

// =============================================================================
// 3. FORMAL LOGIC & STATUTORY CONJUNCTIONS / COPULAS
// =============================================================================

export const FORMAL_LOGIC_TERMS: readonly string[] = [
  'ただし', 'この限りでない', '場合において', 'とき', 'とは', 'をいう',
  'なければなら', 'ものとする', '第', '項', '規定', 'される', '備える',
  '検証', '固定され', '格納され',
];

// =============================================================================
// 4. ABSTRACT FILLERS (CLOSED GRAMMAR + ABSTRACT STEM EXPRESSIONS)
// =============================================================================

export const ABSTRACT_FILLERS: readonly string[] = [
  '様々', '多様', '重要', '適切', '深い', '深く', '強い', '強く', '非常',
  'とても', '感じ', '気持ち', '感情', '状態', '環境', '課題', '側面', '要素',
  '意味', '存在', 'テーマ', '雰囲気', '状況', '関係', '結果', '影響', '全体',
  '特別', '不思議', '改めて', '次第に', 'ように思われ', 'ように感じ',
  'のであった', 'のだった', 'といえる', 'ゆっくりと', '少しずつ',
  'どのように', 'すっかり', 'あまりにも',
];

// =============================================================================
// 5. FIGURATIVE MARKERS (CLOSED COMPARATIVE GRAMMAR)
// =============================================================================

export const FIGURATIVE_MARKERS: readonly string[] = [
  'まるで', 'ごとき', 'ごとく', 'と言おうか', 'ようなものだ', 'ような気が',
  'のように', 'のような', 'ほど', 'とか',
];

// =============================================================================
// 6. RHETORICAL RUPTURE (PUNCTUATION, CONJUNCTIONS, SCRIPT-ACCENT ADVERBS)
// =============================================================================

export const RHETORICAL_RUPTURE: readonly string[] = [
  '！', '？', '!', '?', '「', '」', 'ではないか', 'じゃないか', 'なのだ',
  'のだ。', 'しかし', 'けれども', '忽ち', '突然', '不意に', '遂に',
  'またたく間に', '全然', 'なぜ', '何故',
];

// =============================================================================
// 7. APOPHATIC VOID OPERATORS (NEGATIVE AUXILIARIES & STILLNESS MARKERS)
// =============================================================================

export const APOPHATIC_VOID_OPS: readonly string[] = [
  'ない。', 'なかった。', 'なく', 'ず', 'ぬ', 'ばかり', 'だけ', 'まま',
  'じっと', '静', 'しずか', '寂し', '一人', '音も', 'ただ', 'しみ入', '澄',
];

// =============================================================================
// 8. SCRIPT & MORPHOLOGICAL HELPERS
// =============================================================================

export function isKanji(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return code >= 0x4e00 && code <= 0x9fff;
}

export function isHiragana(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return code >= 0x3040 && code <= 0x309f;
}

export function isKatakana(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return code >= 0x30a0 && code <= 0x30ff;
}

export function splitSentences(text: string): string[] {
  const raw = text.split(/[。！？!?]+/);
  const sents: string[] = [];
  for (const seg of raw) {
    const s = seg.trim();
    if (s.length >= 2) {
      sents.push(s);
    }
  }
  return sents;
}

export function extractKanjiStems(s: string): string[] {
  const matches = s.match(/[\u4e00-\u9fff]+/g);
  return matches ? [...matches] : [];
}

export function countMoras(text: string): number {
  const smallKana = new Set('ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ');
  let moras = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    const code = ch.charCodeAt(0);
    if (((code >= 0x3040 && code <= 0x309f) || (code >= 0x30a0 && code <= 0x30ff)) && !smallKana.has(ch)) {
      moras++;
    } else if (code >= 0x4e00 && code <= 0x9fff) {
      moras += 2;
    }
  }
  return Math.max(moras, 1);
}

export function countSubstr(text: string, sub: string): number {
  if (!sub) return 0;
  let count = 0;
  let pos = 0;
  while ((pos = text.indexOf(sub, pos)) !== -1) {
    count++;
    pos += sub.length;
  }
  return count;
}

export function charBigramEntropy(s: string): number {
  if (s.length < 3) {
    return 0.0;
  }
  const counts = new Map<string, number>();
  const n = s.length - 1;
  for (let i = 0; i < n; i++) {
    const bg = s.slice(i, i + 2);
    counts.set(bg, (counts.get(bg) ?? 0) + 1);
  }
  let ent = 0.0;
  for (const c of counts.values()) {
    const p = c / n;
    ent -= p * Math.log2(p);
  }
  return ent / Math.max(Math.log2(n), 1.0);
}

// Population standard deviation (ddof=0 matching numpy.std)
export function stdPop(arr: number[]): number {
  if (arr.length === 0) return 0.0;
  const meanVal = arr.reduce((acc, v) => acc + v, 0) / arr.length;
  const variance = arr.reduce((acc, v) => acc + (v - meanVal) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

export function mean(arr: number[]): number {
  if (arr.length === 0) return 0.0;
  return arr.reduce((acc, v) => acc + v, 0) / arr.length;
}

export function clip(val: number, minVal: number, maxVal: number): number {
  return Math.max(minVal, Math.min(maxVal, val));
}
