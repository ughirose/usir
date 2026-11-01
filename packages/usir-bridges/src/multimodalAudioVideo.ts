/**
 * @nishiki/usir-bridges: Multimodal Audio & Voronoi Genesis Synthesis
 * Workstream 2: Voronoi Genesis Generator, Polyphonic Web Audio Synthesizer & SMF MIDI Binary Encoder.
 */

import { evaluateStyleMolecules, type StyleMoleculesResult } from '@nishiki/usir-core';

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface DominantElement {
  id: string;      // e.g. 'Sg-03'
  name: string;    // e.g. 'Pr'
  code: string;    // e.g. 'Sg-03 Pr'
  labelJa: string; // e.g. '孔 (Porosity)'
  labelEn: string; // e.g. 'Porosity'
  value: number;
}

export interface HamiltonianDirective {
  targetCvL: number;
  targetVoidPauseRatio: number;
  syntaxHopDistance: number;
  terminalAnchor: string;
  instructionsJa: string;
  instructionsEn: string;
}

export interface VoronoiGenesisArtifact {
  styleId: string;
  styleNameJa: string;
  styleNameEn: string;
  dominantElements: DominantElement[];
  moleculeProfile: {
    yugen: number;
    wabi: number;
    qiyun: number;
    iki: number;
    dominantMolecule: string;
  };
  generatedProseJa: string;
  generatedProseEn: string;
  hamiltonianDirective: HamiltonianDirective;
}

export type UsirAcousticRegime =
  | 'V_Deadzone'
  | 'III_Void'
  | 'II_Critical'
  | 'I_Crystalline'
  | 'Genesis';

export interface UsirMusicNote {
  pitch: number;        // MIDI pitch 0-127 (e.g. 60 = C4)
  startTime: number;    // start time in seconds
  duration: number;     // duration in seconds
  velocity: number;     // MIDI velocity 1-127
  channel?: number;     // MIDI channel 0-15
}

export interface UsirMusicTrack {
  name: string;
  program: number;      // General MIDI patch (0 = Acoustic Grand, 48 = Strings, 80 = Square Lead)
  channel: number;
  instrumentType: 'sine' | 'triangle' | 'square' | 'sawtooth' | 'organ' | 'piano' | 'pad';
  notes: UsirMusicNote[];
}

export interface UsirMusicStats {
  restRatio: number;
  velocityStd: number;
  meanVelocity: number;
  ioiVariance: number;
  maxIntervalLeap: number;
  notesCount: number;
}

export interface UsirMusicComposition {
  title: string;
  mode: UsirAcousticRegime;
  bpm: number;
  timeSignature: [number, number];
  scaleName: string;
  tracks: UsirMusicTrack[];
  reverbWet: number;
  restRatio: number;
  rubatoDepth: number;
  solitonAccentGain: number;
  harmonicDissonance: number;
  durationSec: number;
  stats: UsirMusicStats;
}

// ============================================================================
// Stylistic Element Metadata
// ============================================================================

export const SG_ELEMENTS_META: readonly {
  id: string;
  name: string;
  code: string;
  labelJa: string;
  labelEn: string;
  jaEpithet: string;
  enEpithet: string;
}[] = [
  { id: 'Sg-01', name: 'Gr', code: 'Sg-01 Gr', labelJa: '密 (Density/Grain)', labelEn: 'Density/Grain', jaEpithet: '高密度流体', enEpithet: 'Dense Grain Fluid' },
  { id: 'Sg-02', name: 'Cr', code: 'Sg-02 Cr', labelJa: '晶 (Crystal/Curvature)', labelEn: 'Crystal/Curvature', jaEpithet: '幾何結晶格子', enEpithet: 'Crystalline Lattice' },
  { id: 'Sg-03', name: 'Pr', code: 'Sg-03 Pr', labelJa: '孔 (Porosity/Simplex)', labelEn: 'Porosity/Simplex', jaEpithet: '多孔性空隙', enEpithet: 'Porous Void' },
  { id: 'Sg-04', name: 'Jt', code: 'Sg-04 Jt', labelJa: '揺 (Jitter/Fluctuation)', labelEn: 'Jitter/Fluctuation', jaEpithet: '量子微小揺らぎ', enEpithet: 'Quantum Fluctuation' },
  { id: 'Sg-05', name: 'Cp', code: 'Sg-05 Cp', labelJa: '層 (Coupling/Complex)', labelEn: 'Coupling/Complex', jaEpithet: '多重複合層', enEpithet: 'Multi-Layer Coupling' },
  { id: 'Sg-06', name: 'Sy', code: 'Sg-06 Sy', labelJa: '対 (Symmetry/Lattice)', labelEn: 'Symmetry/Lattice', jaEpithet: '鏡像対称律', enEpithet: 'Mirror Symmetry' },
  { id: 'Sg-07', name: 'Cs', code: 'Sg-07 Cs', labelJa: '切 (Cut/Contrast)', labelEn: 'Cut/Contrast', jaEpithet: '極限明暗切断', enEpithet: 'Stark Contrast Cut' },
  { id: 'Sg-08', name: 'Lp', code: 'Sg-08 Lp', labelJa: '跳 (Leap/Soliton)', labelEn: 'Leap/Soliton', jaEpithet: '双曲ソリトン', enEpithet: 'Hyperbolic Soliton' },
  { id: 'Sg-09', name: 'Dy', code: 'Sg-09 Dy', labelJa: '幅 (Dynamic Range)', labelEn: 'Dynamic Range', jaEpithet: '広域振幅ダイナミクス', enEpithet: 'Wide-Range Amplitude' },
  { id: 'Sg-10', name: 'At', code: 'Sg-10 At', labelJa: '帰 (Attenuation/Anchor)', labelEn: 'Attenuation/Anchor', jaEpithet: '求心的回帰', enEpithet: 'Centripetal Attenuation' },
  { id: 'Sg-11', name: 'Ma', code: 'Sg-11 Ma', labelJa: '間 (Void/Silence)', labelEn: 'Void/Silence', jaEpithet: '真空共鳴余白', enEpithet: 'Vacuum Resonance' },
  { id: 'Sg-12', name: 'Ru', code: 'Sg-12 Ru', labelJa: '破 (Rupture/Catharsis)', labelEn: 'Rupture/Catharsis', jaEpithet: '激震カタルシス', enEpithet: 'Catastrophic Rupture' },
];

// ============================================================================
// Molecule Metadata & Naming Dictionaries
// ============================================================================

const MOLECULE_NAMES: Record<
  string,
  { jaPrefix: string; enPrefix: string; jaEnSub: string }
> = {
  Mol_Yugen: {
    jaPrefix: '零度超流動',
    enPrefix: 'Zero-Degree Superfluid',
    jaEnSub: 'Cold Hyperbolic Zen',
  },
  Mol_Wabi: {
    jaPrefix: '枯淡侘寂',
    enPrefix: 'Wabi Rusticity',
    jaEnSub: 'Subtle Asymmetry',
  },
  Mol_QiYun: {
    jaPrefix: '気韻生動',
    enPrefix: 'Vital Circulation',
    jaEnSub: 'Vital Dynamic Flow',
  },
  Mol_Iki: {
    jaPrefix: '粋巧切断',
    enPrefix: 'Chic Severance',
    jaEnSub: 'Lucid Sharp Edge',
  },
};

/**
 * Analyzes a 24D symplectic state vector z24 = (q_12, p_12) and discovers
 * an uncharted literary/poetic style artifact complete with dynamic naming,
 * cleanroom poetic prose in Japanese and English, and concrete Hamiltonian directives.
 */
export function generateVoronoiGenesisArtifact(
  z24: number[] | Float64Array,
  _language: 'ja' | 'en' = 'ja',
  seedIndex?: number
): VoronoiGenesisArtifact {
  const z = Array.from(z24);
  const q = new Array<number>(12);
  const p = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    q[i] = Math.max(0.0, Math.min(1.0, z[i] ?? 0.0));
    p[i] = z.length >= 24 ? (z[12 + i] ?? 0.0) : 0.0;
  }

  // 1. Evaluate Aesthetic Molecules
  const molRes: StyleMoleculesResult = evaluateStyleMolecules(z.length === 24 ? z : q);

  // 2. Identify Dominant Elements
  // We rank elements by active weight: q_i + 0.35 * |p_i|
  const ranked = SG_ELEMENTS_META.map((meta, i) => ({
    ...meta,
    value: q[i]!,
    effectiveWeight: q[i]! + 0.35 * Math.abs(p[i]!),
  })).sort((a, b) => b.effectiveWeight - a.effectiveWeight);

  const topElements: DominantElement[] = ranked.slice(0, 3).map((item) => ({
    id: item.id,
    name: item.name,
    code: item.code,
    labelJa: item.labelJa,
    labelEn: item.labelEn,
    value: Number(item.value.toFixed(4)),
  }));

  // 3. Dynamic Style Naming
  const idx = seedIndex ?? 1;
  const idxStr = String(idx).padStart(2, '0');
  const styleId = `GENESIS-${idxStr}`;

  const dominantMolKey = molRes.dominantMolecule || 'Mol_Yugen';
  const molMeta = MOLECULE_NAMES[dominantMolKey] ?? MOLECULE_NAMES.Mol_Yugen!;
  const topElemMeta = ranked[0]!;

  const styleNameJa = `【未踏様式 #${idxStr}: ${molMeta.jaPrefix}×${topElemMeta.jaEpithet}（${molMeta.jaEnSub}）】`;
  const styleNameEn = `[Uncharted Style #${idxStr}: ${molMeta.enPrefix} × ${topElemMeta.enEpithet}]`;

  // 4. Derive Hamiltonian Writing Directives
  // Target CV_L (Sentence length variance) driven by Sg-09 Dy and momentum p[8]
  const targetCvL = Number(
    Math.min(1.20, Math.max(0.25, 0.35 + 0.70 * q[8]! + 0.20 * Math.abs(p[8]!))).toFixed(2)
  );

  // Target Void Pause Ratio driven by Sg-03 Pr (0.2), Sg-07 Cs (0.3), Sg-11 Ma (0.5)
  const targetVoidPauseRatio = Number(
    Math.min(0.85, Math.max(0.05, 0.20 * q[2]! + 0.30 * q[6]! + 0.50 * q[10]!)).toFixed(2)
  );

  // Syntax Hop Distance driven by Sg-08 Lp (Leap) and Sg-12 Ru (Rupture)
  const syntaxHopDistance = Number(
    Math.min(6.0, Math.max(1.0, 1.0 + 3.2 * q[7]! + 2.0 * q[11]! + 0.5 * Math.abs(p[7]!))).toFixed(2)
  );

  let terminalAnchor = '余白への漸近的開放 (Asymptotic Open Ellipsis)';
  if (q[9]! > 0.55 && q[1]! > 0.4) {
    terminalAnchor = '体言止め・幾何学的名詞終止 (Geometric Noun Anchor)';
  } else if (q[11]! > 0.6) {
    terminalAnchor = 'カタルシス断絶・破調結び (Cathartic Abrupt Rupture)';
  } else if (q[0]! > 0.6) {
    terminalAnchor = '高密度複合述定・重層終止 (Dense Compound Copula)';
  }

  const instructionsJa = [
    `1. 文長変動係数 (CV_L) 目標値: ${targetCvL} (超短文と複文のダイナミックレンジ比)`,
    `2. 間・休止率 (Void Pause Ratio) 目標値: ${(targetVoidPauseRatio * 100).toFixed(0)}% (読点・行間・沈黙余白の面積確保)`,
    `3. 構文跳躍距離 (Hop Distance) 目標値: ${syntaxHopDistance} (意味論的ソリトン転換とメタファー飛翔)`,
    `4. 終止結び様式: ${terminalAnchor}`,
    `5. 優勢美学分子: ${dominantMolKey} (${(molRes[dominantMolKey.replace('Mol_', '').toLowerCase() as 'yugen' | 'wabi' | 'qiyun' | 'iki'] * 100).toFixed(1)}%)`,
  ].join('\n');

  const instructionsEn = [
    `1. Sentence Length CV_L Target: ${targetCvL} (staccato vs compound sentence ratio)`,
    `2. Void Pause Ratio Target: ${(targetVoidPauseRatio * 100).toFixed(0)}% (inter-clause spacing and silent whitespace)`,
    `3. Syntax Hop Distance: ${syntaxHopDistance} (semantic leap and metaphoric soliton amplitude)`,
    `4. Terminal Anchor: ${terminalAnchor}`,
    `5. Dominant Aesthetic Molecule: ${dominantMolKey}`,
  ].join('\n');

  const hamiltonianDirective: HamiltonianDirective = {
    targetCvL,
    targetVoidPauseRatio,
    syntaxHopDistance,
    terminalAnchor,
    instructionsJa,
    instructionsEn,
  };

  // 5. Generate Cleanroom Evocative Literary Prose
  const { jaProse, enProse } = composeGenesisProse(
    q,
    p,
    topElements,
    dominantMolKey,
    targetVoidPauseRatio,
    targetCvL,
    syntaxHopDistance
  );

  return {
    styleId,
    styleNameJa,
    styleNameEn,
    dominantElements: topElements,
    moleculeProfile: {
      yugen: Number(molRes.yugen.toFixed(4)),
      wabi: Number(molRes.wabi.toFixed(4)),
      qiyun: Number(molRes.qiyun.toFixed(4)),
      iki: Number(molRes.iki.toFixed(4)),
      dominantMolecule: dominantMolKey,
    },
    generatedProseJa: jaProse,
    generatedProseEn: enProse,
    hamiltonianDirective,
  };
}

/**
 * Cleanroom multi-sentence literary/poetic prose generator.
 * Directly reflects the void pause ratio, sentence length variance,
 * soliton leap amplitude, and dominant aesthetic molecule.
 */
function composeGenesisProse(
  q: number[],
  _p: number[],
  topElements: DominantElement[],
  dominantMol: string,
  voidRatio: number,
  cvL: number,
  hopDist: number
): { jaProse: string; enProse: string } {
  const topId = topElements[0]?.id ?? 'Sg-08';
  const highVoid = voidRatio >= 0.45;
  const highCv = cvL >= 0.70;
  const highLeap = hopDist >= 3.5;

  let jaS1 = '';
  let jaS2 = '';
  let jaS3 = '';
  let enS1 = '';
  let enS2 = '';
  let enS3 = '';

  // Sentence 1: Incipient Theme & Spatial Anchor
  if (dominantMol === 'Mol_Yugen') {
    if (highVoid) {
      jaS1 = '星の冷えた灰が、無音の水平線へ沈む。';
      enS1 = 'The chilled ash of distant stars settles upon a soundless horizon.';
    } else {
      jaS1 = '蒼穹の底に沈む群青の気流が、幾重もの薄氷を抱き留めて震えている。';
      enS1 = 'An ultramarine current sunk at the base of the azure sky trembles, cradling layers of gossamer ice.';
    }
  } else if (dominantMol === 'Mol_Wabi') {
    if (highVoid) {
      jaS1 = '風が枯れ草の葉先をかすめ、微かな渇いた音を砂に落とした。';
      enS1 = 'Wind grazes the tips of withered reeds, dropping a faint, arid resonance onto the sand.';
    } else {
      jaS1 = '雨滲む無釉の小壺の肌を撫でれば、不整な土の息づかいが手のひらに伝わる。';
      enS1 = 'Fingers brushing the unglazed stoneware jug discern the irregular, earthy breath of clay.';
    }
  } else if (dominantMol === 'Mol_QiYun') {
    jaS1 = '凍土を割り開く水脈の奔流が、天空の風を巻き込んで旋回する。';
    enS1 = 'A torrent of subglacial springs breaches the frozen soil, whirling aloft to engulf the celestial winds.';
  } else {
    // Mol_Iki
    jaS1 = '夕立上がりの紺絣に差す一筋の残光が、濡れた石畳を斜めに切断した。';
    enS1 = 'A shaft of afterglow piercing indigo kasuri after the sudden shower cuts diagonally across the wet stone path.';
  }

  // Sentence 2: Dynamic Modulation, Texture & Soliton Leap
  if (highCv) {
    // Staccato abrupt clause inserted before deep undulating clause
    if (topId === 'Sg-08' || highLeap) {
      jaS2 = '突風。――次の刹那、視界の縁で青白い双曲線の光条が跳躍し、千年の沈黙を抱えた岩壁を一挙に宇宙の涯へと引きちぎる。';
      enS2 = 'A gust. ――In the next instant, a pale hyperbolic arc of light leaps at the periphery of sight, wrenching the millennial silence of the rock wall toward the outer rim of the cosmos.';
    } else if (topId === 'Sg-12') {
      jaS2 = '亀裂。研ぎ澄まされた刃先の如き閃光が走り、張り詰めていたすべての均衡が灼熱の飛沫となって散った。';
      enS2 = 'A fracture. A flash sharp as a honed edge strikes, and all strained equilibrium scatters into searing droplets.';
    } else {
      jaS2 = '静止。整然たる幾何学の結晶格子が、光の屈折角を寸分の狂いもなく整列させていく。';
      enS2 = 'Stasis. A disciplined crystalline lattice aligns each refraction angle with unyielding precision.';
    }
  } else {
    // Measured, undulating syntactic flow
    if (highVoid) {
      jaS2 = 'どこまでも広がる白銀の静寂のなか、遠い鐘の余韻だけが、言葉の輪郭を削ぎ落としながら透き通っていく。';
      enS2 = 'Across the boundless expanse of silver silence, only the lingering toll of a distant bell dissolves the boundaries of words into transparency.';
    } else {
      jaS2 = '幾重にも折り重なる言葉の階層が、かすかな熱量を放ちながら、新たな生命の流路を織り成していく。';
      enS2 = 'Layer upon layer of semantic folds radiate subtle warmth, weaving novel conduits for vital momentum.';
    }
  }

  // Sentence 3: Terminal Cadence & Anchor
  if (q[9]! > 0.55 && q[1]! > 0.4) {
    jaS3 = '帰還の軌跡を描き、すべては冷厳たる白磁の幾何学。';
    enS3 = 'Tracing an arc of centripetal return, all resolves into the austere geometry of white porcelain.';
  } else if (highVoid) {
    jaS3 = '消え去る息の痕跡……ただ澄明なる真空の深淵が、その奥に息づいている。';
    enS3 = 'A vanishing breath…… only the clear vacuum of the abyss breathes in the quiet depths.';
  } else if (q[11]! > 0.55) {
    jaS3 = 'そして、沈黙の閾値は完全に突破された。';
    enS3 = 'And with that, the threshold of silence was irrecoverably breached.';
  } else {
    jaS3 = 'やがて微光が満ち、静かな調和の円環が閉じられていく。';
    enS3 = 'At last, a quiet radiance swells, gently closing the circle of harmonic stillness.';
  }

  const jaProse = `${jaS1} ${jaS2} ${jaS3}`;
  const enProse = `${enS1} ${enS2} ${enS3}`;

  return { jaProse, enProse };
}

// ============================================================================
// Polyphonic Web Audio Synthesizer Descriptor
// ============================================================================

/**
 * Computes statistical and information-theoretic descriptors of a composition.
 */
export function computeMusicStats(
  melodyNotes: UsirMusicNote[],
  totalDuration: number
): UsirMusicStats {
  if (melodyNotes.length === 0) {
    return {
      restRatio: 1.0,
      velocityStd: 0.0,
      meanVelocity: 0.0,
      ioiVariance: 0.0,
      maxIntervalLeap: 0,
      notesCount: 0,
    };
  }

  // 1. Velocity statistics
  let sumVel = 0;
  for (const n of melodyNotes) sumVel += n.velocity;
  const meanVelocity = sumVel / melodyNotes.length;

  let sumSqDiff = 0;
  for (const n of melodyNotes) {
    sumSqDiff += (n.velocity - meanVelocity) ** 2;
  }
  const velocityStd = Number(Math.sqrt(sumSqDiff / melodyNotes.length).toFixed(4));

  // 2. Inter-Onset-Interval (IOI) variance
  const iois: number[] = [];
  for (let i = 0; i < melodyNotes.length - 1; i++) {
    iois.push(melodyNotes[i + 1]!.startTime - melodyNotes[i]!.startTime);
  }
  let ioiVariance = 0;
  if (iois.length > 1) {
    const meanIoi = iois.reduce((a, b) => a + b, 0) / iois.length;
    const sqDiffIoi = iois.reduce((acc, v) => acc + (v - meanIoi) ** 2, 0);
    ioiVariance = Number((sqDiffIoi / iois.length).toFixed(6));
  }

  // 3. Max Interval Leap between successive melody notes
  let maxIntervalLeap = 0;
  for (let i = 0; i < melodyNotes.length - 1; i++) {
    const leap = Math.abs(melodyNotes[i + 1]!.pitch - melodyNotes[i]!.pitch);
    if (leap > maxIntervalLeap) maxIntervalLeap = leap;
  }

  // 4. Exact Rest Ratio (silence fraction in [0, 1])
  const intervals = melodyNotes.map((n) => ({
    start: n.startTime,
    end: n.startTime + n.duration,
  }));
  intervals.sort((a, b) => a.start - b.start);

  let activeSoundTime = 0;
  let currentStart = intervals[0]!.start;
  let currentEnd = intervals[0]!.end;

  for (let i = 1; i < intervals.length; i++) {
    const next = intervals[i]!;
    if (next.start <= currentEnd + 1e-5) {
      currentEnd = Math.max(currentEnd, next.end);
    } else {
      activeSoundTime += currentEnd - currentStart;
      currentStart = next.start;
      currentEnd = next.end;
    }
  }
  activeSoundTime += currentEnd - currentStart;

  const totalEffectiveTime = Math.max(
    totalDuration,
    intervals[intervals.length - 1]!.end
  );
  const restRatio = Number(
    Math.max(0.0, Math.min(1.0, 1.0 - activeSoundTime / totalEffectiveTime)).toFixed(4)
  );

  return {
    restRatio,
    velocityStd,
    meanVelocity: Number(meanVelocity.toFixed(2)),
    ioiVariance,
    maxIntervalLeap,
    notesCount: melodyNotes.length,
  };
}

/**
 * Maps any 24D symplectic state vector into a polyphonic musical composition specification
 * realizing one of the three required acoustic regimes:
 * - V_Deadzone: Flat velocity 64, restRatio == 0, ioiVariance == 0, mechanical arpeggio
 * - III_Void: Miyako-bushi/In-sen pentatonic, restRatio >= 0.45, sine/triangle timbre, deep reverb
 * - II_Critical / Genesis: 1/f rubato (depth > 0.25), wide dramatic interval leaps >= 8 st, sharp accents
 */
export function synthesizeMusicFromUsir24d(
  z24: number[] | Float64Array,
  mode?: UsirAcousticRegime,
  durationSec: number = 16.0
): UsirMusicComposition {
  const z = Array.from(z24);
  const q = new Array<number>(12);
  const p = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    q[i] = Math.max(0.0, Math.min(1.0, z[i] ?? 0.0));
    p[i] = z.length >= 24 ? (z[12 + i] ?? 0.0) : 0.0;
  }

  // Infer regime if not explicitly specified
  let effectiveMode: UsirAcousticRegime = mode ?? 'II_Critical';
  if (!mode) {
    const voidScore = 0.20 * q[2]! + 0.30 * q[6]! + 0.50 * q[10]!;
    const isFlat = q.every((v) => Math.abs(v - q[0]!) < 0.05 && v < 0.25);
    if (voidScore >= 0.50) {
      effectiveMode = 'III_Void';
    } else if (isFlat) {
      effectiveMode = 'V_Deadzone';
    } else if (q[7]! > 0.50 || q[11]! > 0.50 || q[8]! > 0.50) {
      effectiveMode = 'Genesis';
    } else if (q[1]! > 0.50 && q[5]! > 0.50) {
      effectiveMode = 'I_Crystalline';
    } else {
      effectiveMode = 'II_Critical';
    }
  }

  if (effectiveMode === 'V_Deadzone') {
    return generateDeadzoneComposition(durationSec);
  } else if (effectiveMode === 'III_Void') {
    return generateVoidComposition(q, p, durationSec);
  } else if (effectiveMode === 'I_Crystalline') {
    return generateCrystallineComposition(q, p, durationSec);
  } else {
    // II_Critical or Genesis
    return generateCriticalComposition(q, p, effectiveMode, durationSec);
  }
}

// ============================================================================
// Regime Generators
// ============================================================================

function generateDeadzoneComposition(durationSec: number): UsirMusicComposition {
  const bpm = 120;
  const eighth = 60 / (bpm * 2); // 0.25s
  const numNotes = Math.floor(durationSec / eighth);

  // Arpeggio repeating [60, 64, 67, 64] -> small leaps <= 4 semitones
  const arpeggio = [60, 64, 67, 64];
  const melodyNotes: UsirMusicNote[] = [];

  for (let i = 0; i < numNotes; i++) {
    melodyNotes.push({
      pitch: arpeggio[i % arpeggio.length]!,
      startTime: i * eighth,
      duration: eighth, // exact fit, zero rest
      velocity: 64,     // strictly flat 64
      channel: 0,
    });
  }

  // Chords Track: Constant C-major triad (48, 52, 55) at vel 64
  const chordNotes: UsirMusicNote[] = [];
  const barSec = 60 / bpm * 4; // 2.0s
  const numBars = Math.floor(durationSec / barSec);
  for (let b = 0; b < numBars; b++) {
    const t = b * barSec;
    for (const p of [48, 52, 55]) {
      chordNotes.push({
        pitch: p,
        startTime: t,
        duration: barSec,
        velocity: 64,
        channel: 1,
      });
    }
  }

  // Bass Track: Continuous C2 (36) notes
  const bassNotes: UsirMusicNote[] = [];
  const quarter = 60 / bpm; // 0.5s
  const numQuarters = Math.floor(durationSec / quarter);
  for (let qIdx = 0; qIdx < numQuarters; qIdx++) {
    bassNotes.push({
      pitch: 36,
      startTime: qIdx * quarter,
      duration: quarter,
      velocity: 64,
      channel: 2,
    });
  }

  const tracks: UsirMusicTrack[] = [
    {
      name: 'Mechanical Lead',
      program: 80, // Lead 1 (square)
      channel: 0,
      instrumentType: 'square',
      notes: melodyNotes,
    },
    {
      name: 'Deadzone Pad',
      program: 48, // String Ensemble 1
      channel: 1,
      instrumentType: 'sawtooth',
      notes: chordNotes,
    },
    {
      name: 'Deadzone Bass',
      program: 38, // Synth Bass 1
      channel: 2,
      instrumentType: 'square',
      notes: bassNotes,
    },
  ];

  const stats = computeMusicStats(melodyNotes, durationSec);

  return {
    title: 'Acoustic Regime V: AI Deadzone (Mechanical Monotony)',
    mode: 'V_Deadzone',
    bpm,
    timeSignature: [4, 4],
    scaleName: 'C-Major Arpeggio (Mechanical)',
    tracks,
    reverbWet: 0.05,
    restRatio: 0.0,
    rubatoDepth: 0.0,
    solitonAccentGain: 0.0,
    harmonicDissonance: 0.0,
    durationSec,
    stats,
  };
}

function generateVoidComposition(
  _q: number[],
  _p: number[],
  durationSec: number
): UsirMusicComposition {
  const bpm = 58; // Meditative tempo in [54, 68]
  // Miyako-bushi / In-sen scale: C, Db, F, G, Bb, C5, Db5, F5
  const miyakoBushiScale = [60, 61, 65, 67, 70, 72, 73, 77];

  const melodyNotes: UsirMusicNote[] = [];
  const stepTime = 2.2; // Generous temporal spacing
  const noteDuration = 0.9; // Sounding duration -> rest ratio ~ 0.59 >= 0.45
  const numSteps = Math.floor(durationSec / stepTime);

  // Gentle velocity shading around 46 +/- 8
  const baseVelocities = [42, 48, 38, 52, 45, 54, 40, 50];
  const melodicPattern = [0, 2, 1, 3, 4, 3, 2, 5];

  for (let i = 0; i < numSteps; i++) {
    const scaleIdx = melodicPattern[i % melodicPattern.length]!;
    const pitch = miyakoBushiScale[scaleIdx]!;
    const velocity = baseVelocities[i % baseVelocities.length]!;

    melodyNotes.push({
      pitch,
      startTime: Number((i * stepTime + 0.15 * Math.sin(i * 1.2)).toFixed(3)),
      duration: noteDuration,
      velocity,
      channel: 0,
    });
  }

  // Pad Track: Ethereal open fifths (C3, G3) with long decays
  const chordNotes: UsirMusicNote[] = [];
  const padStep = 5.0;
  const numPads = Math.floor(durationSec / padStep);
  for (let b = 0; b < numPads; b++) {
    const t = b * padStep;
    chordNotes.push(
      { pitch: 48, startTime: t, duration: 3.2, velocity: 36, channel: 1 },
      { pitch: 55, startTime: t, duration: 3.2, velocity: 34, channel: 1 }
    );
  }

  // Bass Track: Deep tonic drone (C2 = 36)
  const bassNotes: UsirMusicNote[] = [];
  const bassStep = 8.0;
  const numBasses = Math.floor(durationSec / bassStep);
  for (let b = 0; b < numBasses; b++) {
    bassNotes.push({
      pitch: 36,
      startTime: b * bassStep,
      duration: 5.5,
      velocity: 42,
      channel: 2,
    });
  }

  const tracks: UsirMusicTrack[] = [
    {
      name: 'Superfluid Void Lead',
      program: 73, // Flute / Shakuhachi
      channel: 0,
      instrumentType: 'sine',
      notes: melodyNotes,
    },
    {
      name: 'Ambient Resonance Pad',
      program: 89, // Pad 2 (warm)
      channel: 1,
      instrumentType: 'triangle',
      notes: chordNotes,
    },
    {
      name: 'Tonic Pedal Drone',
      program: 42, // Cello
      channel: 2,
      instrumentType: 'sine',
      notes: bassNotes,
    },
  ];

  const stats = computeMusicStats(melodyNotes, durationSec);

  return {
    title: 'Acoustic Regime III: Superfluid Void (Miyako-bushi & Deep Reverb)',
    mode: 'III_Void',
    bpm,
    timeSignature: [4, 4],
    scaleName: 'Miyako-bushi / In-sen Pentatonic',
    tracks,
    reverbWet: 0.70,
    restRatio: stats.restRatio,
    rubatoDepth: 0.18,
    solitonAccentGain: 0.12,
    harmonicDissonance: 0.08,
    durationSec,
    stats,
  };
}

function generateCriticalComposition(
  q: number[],
  p: number[],
  mode: 'II_Critical' | 'Genesis',
  durationSec: number
): UsirMusicComposition {
  const bpm = 96;
  const rubatoDepth = Math.max(0.28, 0.25 + 0.30 * q[8]! + 0.20 * Math.abs(p[8]!));

  // Synesthetic Lydian / Chromatic scale with wide leaps
  // Explicit leap patterns: e.g. 60 -> 72 (12 st leap), 62 -> 74 (12 st leap), 74 -> 61 (13 st leap)
  const melodyPitches = [60, 72, 65, 67, 74, 62, 74, 61, 73, 67, 70, 75];
  // Sharp dynamic accents driven by Sg-12 Ru: contrast between quiet 42 and accented 112
  const velocities = [45, 114, 52, 60, 118, 48, 110, 42, 115, 58, 64, 112];
  const durations = [0.35, 0.70, 0.40, 0.30, 0.85, 0.35, 0.65, 0.30, 0.80, 0.45, 0.40, 0.75];

  const melodyNotes: UsirMusicNote[] = [];
  let currentTime = 0.1;
  let idx = 0;

  while (currentTime + 0.5 < durationSec) {
    const pitch = melodyPitches[idx % melodyPitches.length]!;
    const velocity = velocities[idx % velocities.length]!;
    const dur = durations[idx % durations.length]!;

    // 1/f Rubato micro-timing jitter
    const microJitter = (Math.sin(idx * 1.618) + Math.cos(idx * 0.77)) * rubatoDepth * 0.08;
    const actualStart = Math.max(0.0, currentTime + microJitter);

    melodyNotes.push({
      pitch,
      startTime: Number(actualStart.toFixed(3)),
      duration: Number(dur.toFixed(3)),
      velocity,
      channel: 0,
    });

    // Advance time with varying IOI
    const baseIoi = 0.55 + 0.20 * Math.sin(idx * 2.1);
    currentTime += baseIoi + microJitter * 0.5;
    idx++;
  }

  // Chords: Rich jazz/impressionist voicings (Cmaj9, F#m7b5, Abmaj7#11)
  const chordVoicings = [
    [48, 55, 59, 62, 64], // Cmaj9
    [42, 54, 57, 60, 64], // F#m7b5
    [44, 51, 55, 58, 62], // Abmaj7#11
    [45, 52, 55, 59, 62], // Am9
  ];
  const chordNotes: UsirMusicNote[] = [];
  const barSec = 60 / bpm * 4; // 2.5s
  const numBars = Math.floor(durationSec / barSec);

  for (let b = 0; b < numBars; b++) {
    const voicing = chordVoicings[b % chordVoicings.length]!;
    const t = b * barSec;
    for (const p of voicing) {
      chordNotes.push({
        pitch: p,
        startTime: Number(t.toFixed(3)),
        duration: Number((barSec * 0.9).toFixed(3)),
        velocity: 58,
        channel: 1,
      });
    }
  }

  // Bass Track: Syncopated fundamental notes
  const bassNotes: UsirMusicNote[] = [];
  const bassRoots = [36, 42, 44, 45]; // C, F#, Ab, A
  for (let b = 0; b < numBars; b++) {
    const root = bassRoots[b % bassRoots.length]!;
    const t = b * barSec;
    bassNotes.push(
      { pitch: root, startTime: t, duration: 0.6, velocity: 85, channel: 2 },
      { pitch: root, startTime: t + 1.25, duration: 0.5, velocity: 70, channel: 2 }
    );
  }

  const tracks: UsirMusicTrack[] = [
    {
      name: mode === 'Genesis' ? 'Genesis Soliton Lead' : 'Critical Dynamics Lead',
      program: 19, // Church Organ / Synth
      channel: 0,
      instrumentType: 'sawtooth',
      notes: melodyNotes,
    },
    {
      name: 'Harmonic Rich Pad',
      program: 50, // Synth Strings 1
      channel: 1,
      instrumentType: 'pad',
      notes: chordNotes,
    },
    {
      name: 'Syncopated Dynamic Bass',
      program: 35, // Fretless Bass
      channel: 2,
      instrumentType: 'square',
      notes: bassNotes,
    },
  ];

  const stats = computeMusicStats(melodyNotes, durationSec);

  return {
    title: mode === 'Genesis'
      ? 'Acoustic Regime Genesis: Unexplored Hyperbolic Soliton'
      : 'Acoustic Regime II: Self-Organized Critical (1/f Rubato & Wide Leaps)',
    mode,
    bpm,
    timeSignature: [4, 4],
    scaleName: 'Hamiltonian Synesthetic Lydian / Chromatic',
    tracks,
    reverbWet: 0.35,
    restRatio: stats.restRatio,
    rubatoDepth,
    solitonAccentGain: 1.25,
    harmonicDissonance: 0.45,
    durationSec,
    stats,
  };
}

function generateCrystallineComposition(
  _q: number[],
  _p: number[],
  durationSec: number
): UsirMusicComposition {
  const bpm = 108;
  const sixteenth = 60 / (bpm * 4); // 0.1388s
  const melodyNotes: UsirMusicNote[] = [];
  // Crystalline Dorian sequence with balanced symmetry
  const dorianPitches = [62, 65, 67, 69, 72, 69, 67, 65];
  const totalNotes = Math.floor(durationSec / sixteenth);

  for (let i = 0; i < totalNotes; i++) {
    melodyNotes.push({
      pitch: dorianPitches[i % dorianPitches.length]!,
      startTime: Number((i * sixteenth).toFixed(3)),
      duration: Number((sixteenth * 0.9).toFixed(3)),
      velocity: (i % 4 === 0) ? 75 : 58,
      channel: 0,
    });
  }

  const tracks: UsirMusicTrack[] = [
    {
      name: 'Crystalline Arpeggiator',
      program: 11, // Vibraphone / Glockenspiel
      channel: 0,
      instrumentType: 'triangle',
      notes: melodyNotes,
    },
  ];

  const stats = computeMusicStats(melodyNotes, durationSec);

  return {
    title: 'Acoustic Regime I: Crystalline Lattice (Symmetric Dorian)',
    mode: 'I_Crystalline',
    bpm,
    timeSignature: [4, 4],
    scaleName: 'D-Dorian Symmetrical Crystal',
    tracks,
    reverbWet: 0.25,
    restRatio: stats.restRatio,
    rubatoDepth: 0.05,
    solitonAccentGain: 0.40,
    harmonicDissonance: 0.15,
    durationSec,
    stats,
  };
}

// ============================================================================
// Standard MIDI File (SMF Format 1) Binary Encoder
// ============================================================================

/**
 * Encodes an integer into MIDI Variable-Length Quantity (VLQ) bytes.
 */
export function encodeVlq(value: number): number[] {
  let v = Math.max(0, Math.floor(value));
  const buffer: number[] = [v & 0x7f];
  v = v >> 7;
  while (v > 0) {
    buffer.unshift((v & 0x7f) | 0x80);
    v = v >> 7;
  }
  return buffer;
}

/**
 * Encodes a polyphonic UsirMusicComposition into a valid Standard MIDI File (SMF Format 1) binary.
 */
export function encodeCompositionToMidiBytes(comp: UsirMusicComposition): Uint8Array {
  const division = 480; // 480 ticks per quarter note
  const ticksPerSecond = 8 * comp.bpm; // (480 / 60) * bpm

  // Helper to convert string to byte array
  const strToBytes = (s: string): number[] => {
    const bytes: number[] = [];
    for (let i = 0; i < s.length; i++) {
      bytes.push(s.charCodeAt(i) & 0x7f);
    }
    return bytes;
  };

  // 1. Build Conductor Track (Tempo & Time Signature)
  const conductorEvents: number[] = [];

  // Track Name
  const seqNameBytes = strToBytes(comp.title || 'USIR Master');
  conductorEvents.push(0x00, 0xff, 0x03, ...encodeVlq(seqNameBytes.length), ...seqNameBytes);

  // Time Signature: nn=4, dd=2 (2^2=4), cc=24, bb=8
  conductorEvents.push(0x00, 0xff, 0x58, 0x04, comp.timeSignature[0] ?? 4, 2, 24, 8);

  // Set Tempo: microseconds per quarter note = round(60,000,000 / bpm)
  const usPerQuarter = Math.round(60_000_000 / comp.bpm);
  conductorEvents.push(
    0x00,
    0xff,
    0x51,
    0x03,
    (usPerQuarter >> 16) & 0xff,
    (usPerQuarter >> 8) & 0xff,
    usPerQuarter & 0xff
  );

  // End of Track
  conductorEvents.push(0x00, 0xff, 0x2f, 0x00);

  // 2. Build Instrument Tracks
  const trackChunks: number[][] = [];

  // Add conductor track chunk
  const conductorLen = conductorEvents.length;
  trackChunks.push([
    0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
    (conductorLen >> 24) & 0xff,
    (conductorLen >> 16) & 0xff,
    (conductorLen >> 8) & 0xff,
    conductorLen & 0xff,
    ...conductorEvents,
  ]);

  for (const track of comp.tracks) {
    const trackEvents: number[] = [];

    // Track Name
    const trackNameBytes = strToBytes(track.name || 'Track');
    trackEvents.push(0x00, 0xff, 0x03, ...encodeVlq(trackNameBytes.length), ...trackNameBytes);

    // Program Change: at delta 0
    const ch = (track.channel ?? 0) & 0x0f;
    trackEvents.push(0x00, 0xc0 | ch, (track.program ?? 0) & 0x7f);

    // Convert notes into NoteOn and NoteOff events
    interface MidiEvent {
      tick: number;
      type: 'off' | 'on';
      pitch: number;
      velocity: number;
    }
    const events: MidiEvent[] = [];

    for (const note of track.notes) {
      const startTick = Math.max(0, Math.round(note.startTime * ticksPerSecond));
      const durTicks = Math.max(1, Math.round(note.duration * ticksPerSecond));
      const endTick = startTick + durTicks;

      events.push({
        tick: startTick,
        type: 'on',
        pitch: Math.max(0, Math.min(127, note.pitch)),
        velocity: Math.max(1, Math.min(127, note.velocity)),
      });
      events.push({
        tick: endTick,
        type: 'off',
        pitch: Math.max(0, Math.min(127, note.pitch)),
        velocity: 0,
      });
    }

    // Sort events chronologically. If tick is equal, 'off' comes before 'on'
    events.sort((a, b) => {
      if (a.tick !== b.tick) return a.tick - b.tick;
      if (a.type !== b.type) return a.type === 'off' ? -1 : 1;
      return a.pitch - b.pitch;
    });

    let lastTick = 0;
    for (const ev of events) {
      const delta = ev.tick - lastTick;
      const deltaBytes = encodeVlq(delta);
      trackEvents.push(...deltaBytes);

      if (ev.type === 'on') {
        trackEvents.push(0x90 | ch, ev.pitch, ev.velocity);
      } else {
        trackEvents.push(0x80 | ch, ev.pitch, 0x00);
      }
      lastTick = ev.tick;
    }

    // End of Track
    trackEvents.push(0x00, 0xff, 0x2f, 0x00);

    const tLen = trackEvents.length;
    trackChunks.push([
      0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
      (tLen >> 24) & 0xff,
      (tLen >> 16) & 0xff,
      (tLen >> 8) & 0xff,
      tLen & 0xff,
      ...trackEvents,
    ]);
  }

  // 3. Assemble Full SMF File (MThd + all MTrk chunks)
  const totalTracks = trackChunks.length;
  const headerChunk = [
    0x4d, 0x54, 0x68, 0x64, // 'MThd'
    0x00, 0x00, 0x00, 0x06, // length 6
    0x00, 0x01,             // format 1 (multitrack)
    (totalTracks >> 8) & 0xff,
    totalTracks & 0xff,     // track count
    (division >> 8) & 0xff,
    division & 0xff,        // division (480 ticks/quarter)
  ];

  let totalByteCount = headerChunk.length;
  for (const chunk of trackChunks) {
    totalByteCount += chunk.length;
  }

  const out = new Uint8Array(totalByteCount);
  out.set(headerChunk, 0);
  let offset = headerChunk.length;
  for (const chunk of trackChunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }

  return out;
}

/**
 * Creates a Data URI ('data:audio/midi;base64,...') from a UsirMusicComposition
 * for immediate one-click downloading or embedding in HTML.
 */
export function createMidiDataUri(comp: UsirMusicComposition): string {
  const bytes = encodeCompositionToMidiBytes(comp);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  let base64 = '';
  if (typeof btoa === 'function') {
    base64 = btoa(binary);
  } else if (typeof Buffer !== 'undefined') {
    base64 = Buffer.from(bytes).toString('base64');
  } else {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    for (let i = 0; i < bytes.length; i += 3) {
      const b0 = bytes[i]!;
      const b1 = i + 1 < bytes.length ? bytes[i + 1]! : 0;
      const b2 = i + 2 < bytes.length ? bytes[i + 2]! : 0;
      base64 += chars[b0 >> 2]!;
      base64 += chars[((b0 & 3) << 4) | (b1 >> 4)]!;
      base64 += i + 1 < bytes.length ? chars[((b1 & 15) << 2) | (b2 >> 6)]! : '=';
      base64 += i + 2 < bytes.length ? chars[b2 & 63]! : '=';
    }
  }
  return `data:audio/midi;base64,${base64}`;
}

// ============================================================================
// Web Audio API Polyphonic Synthesizer Engine
// ============================================================================

/**
 * Real-time Web Audio polyphonic synthesizer player.
 * Schedules melody, chord pads, basslines with ADSR envelope, biquad filter,
 * and algorithmic synthetic impulse-response convolution reverb.
 */
export function playCompositionOnWebAudio(
  ctx: any,
  comp: UsirMusicComposition
): { stop: () => void } {
  const activeNodes: any[] = [];
  const baseTime = ctx.currentTime ?? 0.0;

  try {
    // 1. Master Output Gain
    const masterGain = ctx.createGain();
    masterGain.gain.setValueAtTime(0.85, baseTime);
    masterGain.connect(ctx.destination);
    activeNodes.push(masterGain);

    // 2. Algorithmic Convolution Reverb (Synthetic Impulse Response)
    let wetGain: any = null;
    let dryGain: any = null;
    let convolver: any = null;

    if (typeof ctx.createBuffer === 'function' && typeof ctx.createConvolver === 'function') {
      const sampleRate = ctx.sampleRate || 44100;
      const decayTime = 2.4;
      const length = Math.floor(sampleRate * decayTime);
      const impulseBuffer = ctx.createBuffer(2, length, sampleRate);

      for (let ch = 0; ch < 2; ch++) {
        const channelData = impulseBuffer.getChannelData(ch);
        for (let i = 0; i < length; i++) {
          const t = i / sampleRate;
          channelData[i] = (Math.random() * 2 - 1) * Math.exp(-t * (ch === 0 ? 3.0 : 3.4));
        }
      }

      convolver = ctx.createConvolver();
      convolver.buffer = impulseBuffer;

      wetGain = ctx.createGain();
      wetGain.gain.setValueAtTime(comp.reverbWet, baseTime);

      dryGain = ctx.createGain();
      dryGain.gain.setValueAtTime(Math.max(0.1, 1.0 - comp.reverbWet * 0.4), baseTime);

      convolver.connect(wetGain);
      wetGain.connect(masterGain);
      dryGain.connect(masterGain);

      activeNodes.push(convolver, wetGain, dryGain);
    }

    // 3. Instrument Tracks & Polyphonic Voice Scheduling
    for (const track of comp.tracks) {
      const trackFilter = ctx.createBiquadFilter();
      trackFilter.type = 'lowpass';
      trackFilter.frequency.setValueAtTime(
        track.instrumentType === 'sine' ? 2400 : 4200,
        baseTime
      );

      if (dryGain && convolver) {
        trackFilter.connect(dryGain);
        trackFilter.connect(convolver);
      } else {
        trackFilter.connect(masterGain);
      }
      activeNodes.push(trackFilter);

      // Map instrument type to native oscillator waveform
      let oscType: 'sine' | 'triangle' | 'square' | 'sawtooth' = 'sine';
      if (track.instrumentType === 'square') oscType = 'square';
      else if (track.instrumentType === 'sawtooth' || track.instrumentType === 'organ') oscType = 'sawtooth';
      else if (track.instrumentType === 'triangle' || track.instrumentType === 'pad' || track.instrumentType === 'piano') oscType = 'triangle';

      for (const note of track.notes) {
        const noteStart = baseTime + note.startTime;
        const noteEnd = noteStart + Math.max(0.05, note.duration);
        const freq = 440 * Math.pow(2, (note.pitch - 69) / 12);
        const peakGain = (note.velocity / 127) * 0.22;

        const osc = ctx.createOscillator();
        const noteGain = ctx.createGain();

        osc.type = oscType;
        osc.frequency.setValueAtTime(freq, noteStart);

        // ADSR Envelope
        const attack = 0.02;
        const decay = 0.06;
        const sustain = peakGain * 0.75;
        const release = 0.08;

        noteGain.gain.setValueAtTime(0.0001, noteStart);
        if (typeof noteGain.gain.linearRampToValueAtTime === 'function') {
          noteGain.gain.linearRampToValueAtTime(peakGain, noteStart + attack);
          noteGain.gain.linearRampToValueAtTime(sustain, noteStart + attack + decay);
          noteGain.gain.setValueAtTime(sustain, noteEnd);
          noteGain.gain.linearRampToValueAtTime(0.0001, noteEnd + release);
        } else {
          noteGain.gain.setValueAtTime(peakGain, noteStart);
          noteGain.gain.setValueAtTime(0.0001, noteEnd + release);
        }

        osc.connect(noteGain);
        noteGain.connect(trackFilter);

        if (typeof osc.start === 'function') {
          osc.start(noteStart);
          osc.stop(noteEnd + release + 0.05);
        }

        activeNodes.push(osc, noteGain);
      }
    }
  } catch (_e) {
    // Non-blocking fail-safe in headless/virtual environments
  }

  return {
    stop: () => {
      for (const node of activeNodes) {
        try {
          if (typeof node.stop === 'function') node.stop();
          if (typeof node.disconnect === 'function') node.disconnect();
        } catch (_err) {
          // Ignore clean-up errors
        }
      }
    },
  };
}

/**
 * Returns a standalone, zero-dependency browser-compatible JavaScript source code
 * that implements the Web Audio polyphonic synthesizer and scheduler.
 */
export function createWebAudioSynthSourceCode(): string {
  return `
// USIR Web Audio Polyphonic Synthesizer Engine
export function playUsirComposition(ctx, comp) {
  const baseTime = ctx.currentTime || 0;
  const master = ctx.createGain();
  master.gain.setValueAtTime(0.85, baseTime);
  master.connect(ctx.destination);

  let wet = ctx.createGain();
  wet.gain.setValueAtTime(comp.reverbWet || 0.3, baseTime);
  let dry = ctx.createGain();
  dry.gain.setValueAtTime(1.0 - (comp.reverbWet || 0.3) * 0.4, baseTime);

  const sampleRate = ctx.sampleRate || 44100;
  const length = Math.floor(sampleRate * 2.2);
  const impulse = ctx.createBuffer(2, length, sampleRate);
  for (let c = 0; c < 2; c++) {
    const d = impulse.getChannelData(c);
    for (let i = 0; i < length; i++) {
      d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (sampleRate * 0.7));
    }
  }
  const conv = ctx.createConvolver();
  conv.buffer = impulse;
  conv.connect(wet);
  wet.connect(master);
  dry.connect(master);

  const activeOscs = [];
  for (const track of comp.tracks) {
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(track.instrumentType === 'sine' ? 2400 : 4200, baseTime);
    filter.connect(dry);
    filter.connect(conv);

    const typeMap = { square: 'square', sawtooth: 'sawtooth', organ: 'sawtooth', triangle: 'triangle', pad: 'triangle', piano: 'triangle', sine: 'sine' };
    const oscType = typeMap[track.instrumentType] || 'sine';

    for (const note of track.notes) {
      const start = baseTime + note.startTime;
      const end = start + Math.max(0.05, note.duration);
      const freq = 440 * Math.pow(2, (note.pitch - 69) / 12);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = oscType;
      osc.frequency.setValueAtTime(freq, start);

      const peak = (note.velocity / 127) * 0.22;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.linearRampToValueAtTime(peak, start + 0.02);
      gain.gain.linearRampToValueAtTime(peak * 0.7, start + 0.08);
      gain.gain.setValueAtTime(peak * 0.7, end);
      gain.gain.linearRampToValueAtTime(0.0001, end + 0.08);

      osc.connect(gain);
      gain.connect(filter);
      osc.start(start);
      osc.stop(end + 0.1);
      activeOscs.push(osc);
    }
  }

  return {
    stop: () => activeOscs.forEach(o => { try { o.stop(); } catch(e){} })
  };
}
`.trim();
}
