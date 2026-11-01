/**
 * @nishiki/usir-bridges: 60fps Cinematic Visual Simulator & Multimodal Storyboard Timeline
 * Workstream 3 Implementation (USIR v3.0 Stage 4 / TASK-362).
 * 
 * Provides:
 * 1. 60fps Real-Time Cinematic Visual Simulator State (`computeCinematicShaderState`)
 *    Maps 24D Symplectic State z = (q_12, p_12) into HTML5 Canvas shader parameters:
 *    - Flux (Sg-01, Sg-05, Sg-09) -> particleCount, textureGrainDensity, bokehLuminance
 *    - Lattice (Sg-02, Sg-06, Sg-10) -> gridSymmetryAlpha, cameraStability, horizonLock
 *    - Void (Sg-03, Sg-07, Sg-11) -> negativeSpaceRatio, mistOpacity, fadeDecayRate
 *    - Shock (Sg-04, Sg-08, Sg-12) & Momentum p -> cameraZoomAccel, panVelocity, chiaroscuroRimLight, cutImpulseProbability
 * 2. Paragraph-Synchronized Multimodal Storyboard Timeline Generator (`generateMultimodalStoryboard`)
 *    Synchronized 4-Column Table:
 *    ① Text Paragraph (序 Mode 0: Setup -> 破 Mode 1: Soliton -> 間 Mode 2: Void)
 *    ② Music Cue (BPM, tonality, rest ratio %, preview composition with playable notes)
 *    ③ Image Generation AI Prompt (Flux.1 / Midjourney v6 / ImageFX ready)
 *    ④ Video Generation AI Camera & Cut Script (Sora / Runway Gen-3 / Google Veo ready)
 * 3. 60fps HTML5 Canvas Cinematic Simulator Rendering Function (`renderCinematicFrame`)
 */

import {
  extractUsir24d,
  transpileToMusic,
  transpileToVisual,
  type MusicTranspiledParams,
  type DiffusionTranspiledParams,
  OBSIDIAN_VOID_MU,
  AMBER_CRITICAL_MU,
  PORCELAIN_CRYSTAL_MU,
  LIGHTNING_RUPTURE_MU,
  DEADZONE_MU,
} from '@nishiki/usir-core';

export {
  OBSIDIAN_VOID_MU,
  AMBER_CRITICAL_MU,
  PORCELAIN_CRYSTAL_MU,
  LIGHTNING_RUPTURE_MU,
  DEADZONE_MU,
};

// ============================================================================
// 1. Interfaces: 60fps Shader State & Canvas Simulator
// ============================================================================

export interface ColorPalette {
  primary: string;
  secondary: string;
  rim: string;
  background: string;
}

export interface CinematicShaderState {
  // Flux (密度: Sg-01, Sg-05, Sg-09)
  particleCount: number; // 12 -- 240 particles
  textureGrainDensity: number; // [0, 1]
  bokehLuminance: number; // [0, 1]

  // Lattice (秩序: Sg-02, Sg-06, Sg-10)
  gridSymmetryAlpha: number; // [0, 1] rule-of-thirds / golden ratio lattice overlay
  cameraStability: number; // [0, 1] tripod lock vs handheld drift
  horizonLock: boolean; // true if cameraStability >= 0.55

  // Void (間: Sg-03, Sg-07, Sg-11)
  negativeSpaceRatio: number; // [0.15, 0.88] letterbox/vignette black & negative space area
  mistOpacity: number; // [0, 0.85] silent atmospheric fog
  fadeDecayRate: number; // [0.01, 0.25] slow lingering afterimage decay rate per frame

  // Shock (破調: Sg-04, Sg-08, Sg-12) & Momentum p
  cameraZoomAccel: number; // rapid dolly/zoom acceleration
  panVelocity: [number, number]; // [vx, vy]
  chiaroscuroRimLight: number; // [0, 1] lightning-like rim light contrast ratio
  cutImpulseProbability: number; // hard-cut flash trigger when momentum norm ||p|| spikes

  // Lens & Color Palette
  colorPalette: ColorPalette;
  lensFocalLengthMm: number; // 24 | 35 | 50 | 85 | 135
  lensFocalLength: string; // e.g. "85mm f/1.4 anamorphic"
  dominantMode: string; // 'void' | 'critical' | 'crystalline' | 'turbulent' | 'deadzone' | 'genesis'
}

export interface CanvasRenderingContext2DLike {
  fillStyle: string | any;
  strokeStyle: string | any;
  lineWidth: number;
  globalAlpha: number;
  fillRect(x: number, y: number, w: number, h: number): void;
  strokeRect(x: number, y: number, w: number, h: number): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  stroke(): void;
  fill(): void;
  arc(x: number, y: number, radius: number, startAngle: number, endAngle: number): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  save(): void;
  restore(): void;
  createLinearGradient?(x0: number, y0: number, x1: number, y1: number): any;
  createRadialGradient?(x0: number, y0: number, r0: number, x1: number, y1: number, r1: number): any;
}

// ============================================================================
// 2. Interfaces: Multimodal Storyboard Timeline (4 Columns)
// ============================================================================

export interface MusicalNoteEvent {
  pitch: string; // e.g. "D3", "A3", "F4"
  freqHz: number; // e.g. 146.83, 220.00
  timeSec: number; // Start offset within section
  durationSec: number; // Note duration
  velocity: number; // 0.0 - 1.0 (accent gain)
}

export interface MusicPreviewComposition {
  notes: MusicalNoteEvent[];
  synthPreset: string; // e.g. "Ambient Sumi-e Bowed Pad & Shakuhachi"
  audioClipDescription: string;
  scaleName: string;
}

export interface StoryboardMusicCue {
  bpm: number; // e.g. 62, 108, 54
  tonality: string; // e.g. "D Miyako-bushi Pentatonic", "C# Harmonic Minor + Quartal Voicing"
  restRatioPercent: number; // e.g. 52 for 52%
  instrumentLead: string; // e.g. "Shakuhachi & Celesta", "Distorted Cello Soliton"
  previewComposition: MusicPreviewComposition;
}

export interface StoryboardImagePrompt {
  subjectScene: string;
  lensFocalLength: string; // e.g. "85mm f/1.4 anamorphic"
  lightingRatio: string; // e.g. "Chiaroscuro 8:1 rim lighting"
  negativeSpaceComposition: string; // e.g. "72% negative space composition, extreme minimalist framing"
  promptString: string; // Ready-to-copy complete prompt for Flux.1 / Midjourney v6 / ImageFX
  negativePrompt: string; // Ready-to-copy negative prompt
  aspectRatio: string; // e.g. "21:9"
  recommendedModel: string; // e.g. "Flux.1 Dev / Midjourney v6 / ImageFX"
}

export interface StoryboardVideoScript {
  durationSec: number; // e.g. 4.5, 2.2, 6.0
  cameraVector: string; // e.g. "Slow dolly-in (vz=+0.15) + locked horizon"
  speedCurve: string; // e.g. "cubic-bezier(0.16, 1, 0.3, 1) [Soliton Impulse]"
  fullVeoSoraPrompt: string; // Ready-to-copy prompt for Sora / Runway Gen-3 / Google Veo
  cutTransition: string; // e.g. "L-cut Dissolve through negative mist"
}

export interface StoryboardTextColumn {
  sectionLabel: string; // e.g. "Scene 01 [序 / Mode 0: 導入・秩序]"
  beforeParagraph: string;
  afterParagraph: string;
  z24: number[];
  deltaNorm: number; // Euclidean distance ||z_after - z_before||
}

export interface StoryboardRow {
  sceneIndex: number;
  textColumn: StoryboardTextColumn;
  musicCue: StoryboardMusicCue;
  imagePrompt: StoryboardImagePrompt;
  videoScript: StoryboardVideoScript;
  shaderState: CinematicShaderState;
}

export interface MultimodalStoryboard {
  rows: StoryboardRow[];
  targetMode: string;
  totalDurationSec: number;
  summary: {
    sceneCount: number;
    averageBpm: number;
    dominantTonality: string;
    overallStyleMode: string;
    negativeSpaceAveragePercent: number;
  };
}

// ============================================================================
// 3. Palettes & Mode Catalogs
// ============================================================================

export const CANONICAL_CINEMATIC_PALETTES: Record<string, ColorPalette> = {
  void: {
    background: '#07080c',
    primary: '#475569', // slate sumi-e wash
    secondary: '#1e293b', // deep ink shadow
    rim: '#94a3b8', // soft moonlit mist silver
  },
  critical: {
    background: '#120705',
    primary: '#ea580c', // ember crimson
    secondary: '#451a03', // dark ember core
    rim: '#fdba74', // intense white-amber rim light
  },
  crystalline: {
    background: '#070d18',
    primary: '#2563eb', // sapphire structure
    secondary: '#172554', // deep geometric navy
    rim: '#93c5fd', // crystalline frost specular
  },
  turbulent: {
    background: '#16040a',
    primary: '#e11d48', // violent magenta
    secondary: '#4c0519', // blood shadow
    rim: '#f43f5e', // high-voltage neon shock
  },
  deadzone: {
    background: '#18181b',
    primary: '#71717a', // flat uninflected gray
    secondary: '#3f3f46', // dull zinc shadow
    rim: '#a1a1aa', // diffuse flat light
  },
  genesis: {
    background: '#060614',
    primary: '#7c3aed', // transcendent void purple
    secondary: '#2e1065', // abyssal violet
    rim: '#38bdf8', // celestial aurora cyan
  },
};

function clamp(val: number, minVal: number, maxVal: number): number {
  return Math.min(maxVal, Math.max(minVal, val));
}

export function detectDominantMode(z24: number[] | Float64Array): string {
  const z = Array.from(z24);
  const q = z.slice(0, 12);
  const p = z.slice(12, 24);

  const qL = (q[1]! + q[5]! + q[9]!) / 3.0;
  const qV = (q[2]! + q[6]! + q[10]!) / 3.0;
  const qS = (q[3]! + q[7]! + q[11]!) / 3.0;
  const pNorm = Math.sqrt(p.reduce((acc, v) => acc + v * v, 0) / 12.0);

  // Deadzone detection: momentum near zero and flat medium distribution
  if (pNorm < 0.08 && qS < 0.25 && qV < 0.35 && qL < 0.65) {
    return 'deadzone';
  }
  // Void detection: Void group dominates
  if (qV > qL && qV > qS && qV >= 0.42) {
    return 'void';
  }
  // Shock / Turbulent detection
  if (qS > 0.60 || pNorm > 0.65) {
    return 'turbulent';
  }
  if (qS > 0.40 || pNorm > 0.35) {
    return 'critical';
  }
  // Lattice / Crystalline detection
  if (qL > 0.50 && qL >= qV && qL >= qS) {
    return 'crystalline';
  }
  // Default to critical or void based on higher component
  return qV > qS ? 'void' : 'critical';
}

// ============================================================================
// 4. Musical Synthesizer & Motif Generator
// ============================================================================

interface ScaleDefinition {
  name: string;
  pitches: string[];
  freqs: number[];
  synthPreset: string;
  leadInstrument: string;
}

const SCALE_PRESETS: Record<string, ScaleDefinition> = {
  void: {
    name: 'D Miyako-bushi Pentatonic (陰旋法)',
    pitches: ['D3', 'Eb3', 'G3', 'A3', 'Bb3', 'D4', 'Eb4'],
    freqs: [146.83, 155.56, 196.00, 220.00, 233.08, 293.66, 311.13],
    synthPreset: 'Ambient Sumi-e Bowed Pad & Shakuhachi Breath',
    leadInstrument: 'Shakuhachi & Celesta (Whispering Breath)',
  },
  critical: {
    name: 'C# Harmonic Minor + Quartal Voicing',
    pitches: ['C#3', 'E3', 'F#3', 'G#3', 'A3', 'C4', 'C#4'],
    freqs: [138.59, 164.81, 185.00, 207.65, 220.00, 261.63, 277.18],
    synthPreset: 'Cinematic Aggressive Soliton Sub & Bowed Metal',
    leadInstrument: 'Distorted Cello Soliton & Taiko Impulse',
  },
  crystalline: {
    name: 'F Lydian Suspended (Crystalline Symmetry)',
    pitches: ['F3', 'G3', 'A3', 'B3', 'C4', 'D4', 'E4'],
    freqs: [174.61, 196.00, 220.00, 246.94, 261.63, 293.66, 329.63],
    synthPreset: 'Glass Marimba & Crystalline Sieve Strings',
    leadInstrument: 'String Quartet & Crystal Marimba',
  },
  turbulent: {
    name: 'B Phrygian Dominant (Rupture Flash)',
    pitches: ['B2', 'C3', 'D#3', 'E3', 'F#3', 'G3', 'A3'],
    freqs: [123.47, 130.81, 155.56, 164.81, 185.00, 196.00, 220.00],
    synthPreset: 'Industrial Distortion Wave & Sub-bass Impact',
    leadInstrument: 'Electric Cello Overdrive & Rupture Percussion',
  },
  deadzone: {
    name: 'C Major Uninflected Equal-Tempered (Deadzone Static)',
    pitches: ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3'],
    freqs: [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94],
    synthPreset: 'General MIDI Flat Electric Piano (Rigid Grid)',
    leadInstrument: 'Monotone GM Synth Piano (Quantized)',
  },
  genesis: {
    name: 'G# Acoustic Scale / Lydian Dominant (Voronoi Horizon)',
    pitches: ['G#3', 'A#3', 'C4', 'D4', 'D#4', 'F4', 'F#4'],
    freqs: [207.65, 233.08, 261.63, 293.66, 311.13, 349.23, 369.99],
    synthPreset: 'Microtonal Glass Harmonica & Transcendent Shimmer Pad',
    leadInstrument: 'Microtonal Glass Harmonica & Shimmer Pad',
  },
};

export function extractParagraphKeywords(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) {
    return 'cinematic atmospheric void, quiet emptiness';
  }

  // Extract explicit sensory imagery words in Japanese or English
  const jpKeywords: string[] = [];
  const jpMatches = trimmed.match(/[\u4e00-\u9faf]{2,4}|[\u30a0-\u30ff]{2,6}/g);
  if (jpMatches && jpMatches.length > 0) {
    for (const m of jpMatches) {
      if (
        !jpKeywords.includes(m) &&
        !['こと', 'もの', 'よう', 'ため', 'これ', 'それ', 'その', 'ない', 'ある', 'いる'].includes(m)
      ) {
        jpKeywords.push(m);
        if (jpKeywords.length >= 4) break;
      }
    }
  }

  const enKeywords: string[] = [];
  const enMatches = trimmed.match(/[a-zA-Z]{4,15}/g);
  if (enMatches && enMatches.length > 0) {
    for (const w of enMatches) {
      const lower = w.toLowerCase();
      if (
        !enKeywords.includes(lower) &&
        !['this', 'that', 'with', 'from', 'have', 'were', 'been', 'which', 'their', 'there'].includes(lower)
      ) {
        enKeywords.push(lower);
        if (enKeywords.length >= 4) break;
      }
    }
  }

  const found = [...jpKeywords, ...enKeywords];
  if (found.length > 0) {
    return found.join(', ');
  }

  // Fallback to first clean clause or sentence
  const firstSentence = trimmed.split(/[。！？.!?\n]/)[0]?.trim();
  if (firstSentence && firstSentence.length > 0) {
    return firstSentence.slice(0, 32);
  }
  return 'minimalist cinematic composition, high textural contrast';
}

export function synthesizeScaleMotif(
  mode: string,
  bpm: number,
  _sceneIndex: number,
  restRatio: number
): MusicalNoteEvent[] {
  const scale = SCALE_PRESETS[mode] || SCALE_PRESETS['void']!;
  const beatSec = 60.0 / bpm;
  const notes: MusicalNoteEvent[] = [];

  if (mode === 'void') {
    // Sparse, lingering apophatic notes with wide rests
    const indices = [0, 2, 4, 1];
    let currentTime = 0.5 * beatSec;
    for (let i = 0; i < indices.length; i++) {
      const pitchIdx = indices[i]! % scale.pitches.length;
      const duration = beatSec * (1.2 + 0.6 * (1.0 - restRatio));
      notes.push({
        pitch: scale.pitches[pitchIdx]!,
        freqHz: scale.freqs[pitchIdx]!,
        timeSec: Number(currentTime.toFixed(3)),
        durationSec: Number(duration.toFixed(3)),
        velocity: Number((0.55 + 0.15 * Math.sin(i * 1.5)).toFixed(2)),
      });
      currentTime += duration + beatSec * (1.5 + 3.0 * restRatio);
    }
  } else if (mode === 'critical' || mode === 'turbulent') {
    // Dense, syncopated soliton pulses with aggressive dynamic accents
    const indices = [0, 3, 2, 5, 4, 6, 0];
    let currentTime = 0.0;
    for (let i = 0; i < indices.length; i++) {
      const pitchIdx = indices[i]! % scale.pitches.length;
      const duration = beatSec * (0.35 + 0.15 * (i % 2));
      const velocity = i === 3 || i === 6 ? 0.95 : 0.72 + 0.1 * (i % 3);
      notes.push({
        pitch: scale.pitches[pitchIdx]!,
        freqHz: scale.freqs[pitchIdx]!,
        timeSec: Number(currentTime.toFixed(3)),
        durationSec: Number(duration.toFixed(3)),
        velocity: Number(velocity.toFixed(2)),
      });
      currentTime += duration + beatSec * (0.15 + 0.35 * restRatio);
    }
  } else if (mode === 'crystalline') {
    // Balanced geometric lattice motif (equal spacing, rule-of-thirds symmetry)
    const indices = [0, 2, 4, 6, 4, 2];
    let currentTime = 0.25 * beatSec;
    for (let i = 0; i < indices.length; i++) {
      const pitchIdx = indices[i]! % scale.pitches.length;
      const duration = beatSec * 0.75;
      notes.push({
        pitch: scale.pitches[pitchIdx]!,
        freqHz: scale.freqs[pitchIdx]!,
        timeSec: Number(currentTime.toFixed(3)),
        durationSec: Number(duration.toFixed(3)),
        velocity: 0.70,
      });
      currentTime += duration + beatSec * 0.5;
    }
  } else if (mode === 'deadzone') {
    // Flat, uninflected mechanical beats on strict quarter notes
    const indices = [0, 1, 2, 0];
    let currentTime = 0.0;
    for (let i = 0; i < indices.length; i++) {
      const pitchIdx = indices[i]! % scale.pitches.length;
      notes.push({
        pitch: scale.pitches[pitchIdx]!,
        freqHz: scale.freqs[pitchIdx]!,
        timeSec: Number(currentTime.toFixed(3)),
        durationSec: Number((beatSec * 0.8).toFixed(3)),
        velocity: 0.65,
      });
      currentTime += beatSec;
    }
  } else {
    // Genesis / unexplored void
    const indices = [0, 3, 4, 1, 6];
    let currentTime = 0.4 * beatSec;
    for (let i = 0; i < indices.length; i++) {
      const pitchIdx = indices[i]! % scale.pitches.length;
      const duration = beatSec * 1.0;
      notes.push({
        pitch: scale.pitches[pitchIdx]!,
        freqHz: scale.freqs[pitchIdx]!,
        timeSec: Number(currentTime.toFixed(3)),
        durationSec: Number(duration.toFixed(3)),
        velocity: Number((0.60 + 0.25 * (i / 5)).toFixed(2)),
      });
      currentTime += duration + beatSec * 0.8;
    }
  }

  return notes;
}

// ============================================================================
// 5. 60fps Cinematic Shader State Engine
// ============================================================================

export function computeCinematicShaderState(
  z24: number[] | Float64Array,
  modeOverride?: string
): CinematicShaderState {
  const z = Array.from(z24);
  while (z.length < 24) {
    z.push(0.0);
  }

  const q = z.slice(0, 12);
  const p = z.slice(12, 24);

  // Group Aggregations
  // Flux: Sg-01 (micro), Sg-05 (meso), Sg-09 (macro)
  const qFlux = (q[0]! + q[4]! + q[8]!) / 3.0;
  const pFlux = (p[0]! + p[4]! + p[8]!) / 3.0;

  // Lattice: Sg-02 (micro), Sg-06 (meso), Sg-10 (macro)
  const qLattice = (q[1]! + q[5]! + q[9]!) / 3.0;
  const pLattice = (p[1]! + p[5]! + p[9]!) / 3.0;

  // Void: Sg-03 (micro), Sg-07 (meso), Sg-11 (macro)
  const qVoid = (q[2]! + q[6]! + q[10]!) / 3.0;
  const pVoid = (p[2]! + p[6]! + p[10]!) / 3.0;

  // Shock: Sg-04 (micro), Sg-08 (meso), Sg-12 (macro)
  const qShock = (q[3]! + q[7]! + q[11]!) / 3.0;
  const pShock = (p[3]! + p[7]! + p[11]!) / 3.0;

  // Momentum Root Mean Square Norm
  const pNorm = Math.sqrt(p.reduce((acc, v) => acc + v * v, 0) / 12.0);

  // Dominant Mode
  const detectedMode = detectDominantMode(z);
  const mode = (modeOverride || detectedMode).toLowerCase();

  // 1. Flux -> particleCount, textureGrainDensity, bokehLuminance
  const particleCount = Math.round(
    clamp(12 + 228 * (0.65 * qFlux + 0.35 * Math.tanh(pFlux * 2.0 + 0.2)), 12, 240)
  );
  const textureGrainDensity = clamp(
    0.05 + 0.60 * qFlux + 0.30 * qShock + 0.15 * Math.tanh(pShock),
    0.0,
    1.0
  );
  const bokehLuminance = clamp(
    0.15 + 0.55 * q[4]! + 0.25 * q[8]! + 0.20 * Math.tanh(pFlux),
    0.0,
    1.0
  );

  // 2. Lattice -> gridSymmetryAlpha, cameraStability, horizonLock
  const gridSymmetryAlpha = clamp(
    0.85 * qLattice - 0.40 * qShock + 0.15,
    0.0,
    1.0
  );
  const cameraStability = clamp(
    0.90 * qLattice - 0.65 * qShock - 0.35 * Math.tanh(pNorm * 2.0) + 0.10 * Math.tanh(pLattice) + 0.15,
    0.0,
    1.0
  );
  const horizonLock = cameraStability >= 0.55;

  // 3. Void -> negativeSpaceRatio, mistOpacity, fadeDecayRate
  // negativeSpaceRatio in [0.15, 0.88]
  const modeVoidBoost = mode === 'void' ? 0.18 : 0.0;
  const rawNegativeSpace =
    0.15 + 0.73 * (0.25 * q[2]! + 0.35 * q[6]! + 0.40 * q[10]!) + 0.05 * Math.tanh(pVoid) + modeVoidBoost;
  const negativeSpaceRatio = clamp(rawNegativeSpace, 0.15, 0.88);

  // mistOpacity in [0, 0.85]
  const rawMist =
    0.85 * (0.25 * q[2]! + 0.35 * q[6]! + 0.40 * q[10]!) * (1.0 - 0.45 * qShock) +
    (mode === 'void' ? 0.15 : 0.0);
  const mistOpacity = clamp(rawMist, 0.0, 0.85);

  // fadeDecayRate in [0.01, 0.25] (lingering afterimage: lower rate = longer linger)
  const fadeDecayRate = clamp(
    0.22 - 0.19 * qVoid + 0.08 * qShock - (mode === 'void' ? 0.04 : 0.0),
    0.01,
    0.25
  );

  // 4. Shock & Momentum p -> cameraZoomAccel, panVelocity, chiaroscuroRimLight, cutImpulseProbability
  const cameraZoomAccel = clamp(
    0.05 + 2.4 * qShock + 1.8 * Math.tanh(pNorm * 2.0),
    0.0,
    4.0
  );

  const vx = clamp((p[3]! - p[7]!) * 1.8 + (qShock - qLattice) * 0.4, -3.0, 3.0);
  const vy = clamp((p[11]! - p[8]!) * 1.5, -2.5, 2.5);
  const panVelocity: [number, number] = [Number(vx.toFixed(3)), Number(vy.toFixed(3))];

  const chiaroscuroRimLight = clamp(
    0.10 + 0.75 * qShock + 0.25 * Math.tanh(pNorm * 1.5),
    0.0,
    1.0
  );

  // Hard cut impulse is triggered by Shock momentum conjugate components (p3, p7, p11)
  const pShockNorm = Math.sqrt(((p[3]! * p[3]!) + (p[7]! * p[7]!) + (p[11]! * p[11]!)) / 3.0);
  const cutImpulseProbability = clamp(
    Math.tanh(Math.max(0, pShockNorm - 0.25) * 2.5) * 0.65 + 0.35 * qShock,
    0.0,
    1.0
  );

  // 5. Lens Focal Length
  let lensFocalLengthMm = 50;
  let lensFocalLength = '50mm f/1.2 standard cinema';

  if (mode === 'deadzone') {
    lensFocalLengthMm = 50;
    lensFocalLength = '50mm kit lens flat neutral';
  } else if (mode === 'void' || (qVoid > 0.48 && qVoid >= qShock && qVoid >= qLattice)) {
    if (qVoid > 0.65) {
      lensFocalLengthMm = 135;
      lensFocalLength = '135mm f/2.0 telephoto void compression';
    } else {
      lensFocalLengthMm = 85;
      lensFocalLength = '85mm f/1.4 intimate telephoto anamorphic';
    }
  } else if (mode === 'critical' || mode === 'turbulent' || qShock > 0.45 || pNorm > 0.45) {
    lensFocalLengthMm = 24;
    lensFocalLength = '24mm f/2.8 dynamic ultra-wide forced perspective';
  } else if (mode === 'crystalline' || qLattice > 0.55) {
    lensFocalLengthMm = 35;
    lensFocalLength = '35mm f/1.8 narrative geometric prime';
  } else if (mode === 'genesis') {
    lensFocalLengthMm = 85;
    lensFocalLength = '85mm f/1.4 anamorphic spherical tilt';
  }

  // 6. Color Palette
  const colorPalette = CANONICAL_CINEMATIC_PALETTES[mode] || CANONICAL_CINEMATIC_PALETTES['void']!;

  return {
    particleCount,
    textureGrainDensity: Number(textureGrainDensity.toFixed(3)),
    bokehLuminance: Number(bokehLuminance.toFixed(3)),
    gridSymmetryAlpha: Number(gridSymmetryAlpha.toFixed(3)),
    cameraStability: Number(cameraStability.toFixed(3)),
    horizonLock,
    negativeSpaceRatio: Number(negativeSpaceRatio.toFixed(3)),
    mistOpacity: Number(mistOpacity.toFixed(3)),
    fadeDecayRate: Number(fadeDecayRate.toFixed(4)),
    cameraZoomAccel: Number(cameraZoomAccel.toFixed(3)),
    panVelocity,
    chiaroscuroRimLight: Number(chiaroscuroRimLight.toFixed(3)),
    cutImpulseProbability: Number(cutImpulseProbability.toFixed(3)),
    colorPalette,
    lensFocalLengthMm,
    lensFocalLength,
    dominantMode: mode,
  };
}

// ============================================================================
// 6. Paragraph-Synchronized Multimodal Storyboard Timeline Generator
// ============================================================================

function splitTextIntoParagraphs(text: string): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [''];

  // 1. Try double newline split
  let parts = trimmed.split(/\r?\n\s*\r?\n/).map((s) => s.trim()).filter((s) => s.length > 0);
  if (parts.length >= 3) return parts;

  // 2. Try single newline split
  parts = trimmed.split(/\r?\n/).map((s) => s.trim()).filter((s) => s.length > 0);
  if (parts.length >= 3) return parts;

  // 3. Try sentence split (Japanese or English)
  const sentences = trimmed
    .split(/(?<=[。！？.!?])\s*/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  if (sentences.length >= 3) {
    const chunkSize = Math.ceil(sentences.length / 3);
    return [
      sentences.slice(0, chunkSize).join(' '),
      sentences.slice(chunkSize, chunkSize * 2).join(' '),
      sentences.slice(chunkSize * 2).join(' '),
    ].filter((s) => s.length > 0);
  }

  // 4. Fallback to length chunking
  const len = trimmed.length;
  const c1 = Math.floor(len / 3);
  const c2 = Math.floor((len * 2) / 3);
  return [trimmed.slice(0, c1), trimmed.slice(c1, c2), trimmed.slice(c2)].filter((s) => s.length > 0);
}

export function splitIntoScenes(
  beforeText: string,
  afterText: string
): { before: string[]; after: string[] } {
  let beforeParas = splitTextIntoParagraphs(beforeText);
  let afterParas = splitTextIntoParagraphs(afterText);

  const targetCount = Math.max(beforeParas.length, afterParas.length, 3);

  while (beforeParas.length < targetCount) {
    const last = beforeParas[beforeParas.length - 1] || '';
    beforeParas.push(last);
  }
  while (afterParas.length < targetCount) {
    const last = afterParas[afterParas.length - 1] || '';
    afterParas.push(last);
  }

  return { before: beforeParas, after: afterParas };
}

export function generateMultimodalStoryboard(
  beforeText: string,
  afterText: string,
  targetMode?: string
): MultimodalStoryboard {
  const { before, after } = splitIntoScenes(beforeText, afterText);
  const N = before.length;

  const rows: StoryboardRow[] = [];
  let totalDurationSec = 0;
  let bpmSum = 0;
  let restRatioSum = 0;

  for (let k = 0; k < N; k++) {
    const bPara = before[k]!;
    const aPara = after[k]!;

    // 1. Scene Label following Jo-Ha-Kyu / Ma
    let sectionLabel = '';
    let sceneProgressionMode = 'crystalline';

    if (k === 0) {
      sectionLabel = `Scene 01 [序 / Mode 0: 導入・秩序]`;
      sceneProgressionMode = targetMode ? targetMode : 'crystalline';
    } else if (k === N - 1) {
      sectionLabel = `Scene ${String(k + 1).padStart(2, '0')} [間 / Mode 2: 静寂・余韻]`;
      sceneProgressionMode = targetMode ? targetMode : 'void';
    } else {
      const subIdx = k === 1 && N === 3 ? '' : `-${k}`;
      sectionLabel = `Scene ${String(k + 1).padStart(2, '0')} [破${subIdx} / Mode 1: 臨界・破調]`;
      sceneProgressionMode = targetMode ? targetMode : 'critical';
    }

    // 2. 24D Extraction
    const resAfter = extractUsir24d(aPara);
    const resBefore = extractUsir24d(bPara);
    const z24 = resAfter.z_24d;

    // Delta norm between before and after
    let deltaNorm = 0.0;
    for (let d = 0; d < 24; d++) {
      const diff = (resAfter.z_24d[d] || 0) - (resBefore.z_24d[d] || 0);
      deltaNorm += diff * diff;
    }
    deltaNorm = Math.sqrt(deltaNorm);

    // Compute Shader State
    const shader = computeCinematicShaderState(z24, sceneProgressionMode);

    // Transpile to Music & Diffusion
    const musicParams = transpileToMusic(z24) as MusicTranspiledParams;
    const diffParams = transpileToVisual(z24) as DiffusionTranspiledParams;
    const aspectRatio = diffParams.composition_symmetry > 0.65 ? '16:9' : '21:9';

    // 3. Music Cue Calculation
    let bpm = 72;
    if (sceneProgressionMode === 'void') {
      bpm = Math.round(52 + 16 * (1.0 - musicParams.rest_ratio)); // 52 - 68 BPM
    } else if (sceneProgressionMode === 'critical' || sceneProgressionMode === 'turbulent') {
      bpm = Math.round(102 + 32 * musicParams.bpm_fluctuation); // 102 - 134 BPM
    } else if (sceneProgressionMode === 'crystalline') {
      bpm = Math.round(76 + 18 * musicParams.bpm_fluctuation); // 76 - 94 BPM
    } else if (sceneProgressionMode === 'deadzone') {
      bpm = 80;
    } else {
      bpm = Math.round(64 + 20 * musicParams.bpm_fluctuation);
    }

    const restRatioPercent = Math.round(clamp(musicParams.rest_ratio * 100, 10, 85));
    const scale = SCALE_PRESETS[shader.dominantMode] || SCALE_PRESETS['void']!;

    const notes = synthesizeScaleMotif(
      shader.dominantMode,
      bpm,
      k,
      musicParams.rest_ratio
    );

    const musicCue: StoryboardMusicCue = {
      bpm,
      tonality: scale.name,
      restRatioPercent,
      instrumentLead: scale.leadInstrument,
      previewComposition: {
        notes,
        synthPreset: scale.synthPreset,
        audioClipDescription: `Synthesized ${scale.name} at ${bpm} BPM with ${restRatioPercent}% rest ratio`,
        scaleName: scale.name,
      },
    };

    // 4. Image Generation AI Prompt (Flux.1 / Midjourney v6 / ImageFX)
    const subjectKeywords = extractParagraphKeywords(aPara);

    let lightingDesc = '';
    let negativeSpaceDesc = '';

    if (shader.dominantMode === 'void') {
      lightingDesc = 'Diffused overcast sumi-e mist, low-key shadow gradient, delicate silver rim illumination (ratio 2:1)';
      negativeSpaceDesc = `${Math.round(shader.negativeSpaceRatio * 100)}% negative space composition, extreme minimalist framing, apophatic breathing room`;
    } else if (shader.dominantMode === 'critical' || shader.dominantMode === 'turbulent') {
      lightingDesc = `Chiaroscuro 8:1 high-contrast rim lighting, piercing directional edge light (${Math.round(shader.chiaroscuroRimLight * 100)}% rim intensity), deep obsidian shadows`;
      negativeSpaceDesc = `${Math.round(shader.negativeSpaceRatio * 100)}% dynamic diagonal tension, rule of thirds rupture, asymmetric kinetic framing`;
    } else if (shader.dominantMode === 'crystalline') {
      lightingDesc = 'Pristine architectural daylight, balanced 3:1 fill, geometric specular highlights';
      negativeSpaceDesc = `${Math.round(shader.negativeSpaceRatio * 100)}% golden-ratio lattice balance, centered architectural symmetry`;
    } else if (shader.dominantMode === 'deadzone') {
      lightingDesc = 'Flat overhead fluorescent lighting, zero rim contrast, 1:1 washed out diffuse fill';
      negativeSpaceDesc = '30% crowded cluttered frame, accidental center-heavy clutter';
    } else {
      lightingDesc = 'Bioluminescent aurora edge lighting, iridescent rim illumination, deep cosmic ambient glow';
      negativeSpaceDesc = `${Math.round(shader.negativeSpaceRatio * 100)}% expansive cosmic void, celestial negative framing`;
    }

    const promptString = [
      `cinematic film still, ${subjectKeywords}`,
      `shot on ARRI Alexa 65`,
      shader.lensFocalLength,
      lightingDesc,
      negativeSpaceDesc,
      `photorealistic 8k UHD, masterwork --ar 21:9 --style raw`,
    ].join(', ');

    const negativePrompt =
      'blurry, flat lighting, oversaturated, amateur, text, watermark, bad anatomy, artificial plastic sheen, AI deadzone, repetitive clutter';

    const imagePrompt: StoryboardImagePrompt = {
      subjectScene: subjectKeywords,
      lensFocalLength: shader.lensFocalLength,
      lightingRatio: lightingDesc,
      negativeSpaceComposition: negativeSpaceDesc,
      promptString,
      negativePrompt,
      aspectRatio,
      recommendedModel: 'Flux.1 Dev / Midjourney v6 / ImageFX',
    };

    // 5. Video Generation AI Camera & Cut Script (Sora / Runway Gen-3 / Google Veo)
    let cutDurationSec = 4.5;
    let cameraVector = '';
    let speedCurve = '';
    let cutTransition = '';

    if (shader.dominantMode === 'void') {
      cutDurationSec = Number((5.2 + 2.5 * shader.negativeSpaceRatio).toFixed(1));
      cameraVector = 'Static locked-off shot + slow rack focus to negative space (vz=+0.02) + locked horizon';
      speedCurve = 'ease-out-sine [Void Lingering Decay]';
      cutTransition = 'L-cut dissolve through lingering negative mist';
    } else if (shader.dominantMode === 'critical' || shader.dominantMode === 'turbulent') {
      cutDurationSec = Number((1.8 + 1.2 * (1.0 - shader.chiaroscuroRimLight)).toFixed(1));
      cameraVector = `Handheld whip-pan (vx=${shader.panVelocity[0]}) + rapid push-in (vz=+1.4) + Dutch tilt 8deg`;
      speedCurve = 'cubic-bezier(0.16, 1, 0.3, 1) [Soliton Impulse]';
      cutTransition = 'Hard jump-cut on soliton acoustic impulse';
    } else if (shader.dominantMode === 'crystalline') {
      cutDurationSec = Number((4.0 + 1.0 * shader.gridSymmetryAlpha).toFixed(1));
      cameraVector = 'Slow smooth dolly-in (vz=+0.15) + locked horizon (tilt 0deg)';
      speedCurve = 'linear [Crystalline Lock]';
      cutTransition = 'Match cut along golden-ratio lattice lines';
    } else if (shader.dominantMode === 'deadzone') {
      cutDurationSec = 4.0;
      cameraVector = 'Motionless static camera, zero dolly, neutral eye level';
      speedCurve = 'linear [Flat Deadzone]';
      cutTransition = 'Abrupt cut';
    } else {
      cutDurationSec = 5.0;
      cameraVector = 'Orbital spiral crane shot (angVel=0.12rad/s) + expanding focal horizon';
      speedCurve = 'cubic-bezier(0.25, 0.1, 0.25, 1) [Harmonic Wave]';
      cutTransition = 'Prismatic cross-fade';
    }

    const fullVeoSoraPrompt = [
      `Cinematic cut (${cutDurationSec}s duration).`,
      `Camera Direction: ${cameraVector}, timing curve: ${speedCurve}.`,
      `Visual Scene: ${subjectKeywords}.`,
      `Lighting: ${lightingDesc}.`,
      `Lens: ${shader.lensFocalLength}.`,
      `Transition: ${cutTransition}.`,
      `Hyper-realistic 60fps cinematic motion blur, 4K master grade.`,
    ].join(' ');

    const videoScript: StoryboardVideoScript = {
      durationSec: cutDurationSec,
      cameraVector,
      speedCurve,
      fullVeoSoraPrompt,
      cutTransition,
    };

    const textColumn: StoryboardTextColumn = {
      sectionLabel,
      beforeParagraph: bPara,
      afterParagraph: aPara,
      z24,
      deltaNorm: Number(deltaNorm.toFixed(4)),
    };

    rows.push({
      sceneIndex: k + 1,
      textColumn,
      musicCue,
      imagePrompt,
      videoScript,
      shaderState: shader,
    });

    totalDurationSec += cutDurationSec;
    bpmSum += bpm;
    restRatioSum += restRatioPercent;
  }

  const primaryMode = targetMode || (rows[0] ? rows[0].shaderState.dominantMode : 'void');

  return {
    rows,
    targetMode: primaryMode,
    totalDurationSec: Number(totalDurationSec.toFixed(1)),
    summary: {
      sceneCount: N,
      averageBpm: Math.round(bpmSum / N),
      dominantTonality: rows[0]?.musicCue.tonality || '',
      overallStyleMode: primaryMode,
      negativeSpaceAveragePercent: Math.round(restRatioSum / N),
    },
  };
}

// ============================================================================
// 7. 60fps HTML5 Canvas Cinematic Visual Simulator Renderer
// ============================================================================

export function renderCinematicFrame(
  ctx: CanvasRenderingContext2DLike,
  state: CinematicShaderState,
  frameIndex: number,
  width: number,
  height: number
): void {
  const t = frameIndex / 60.0; // Elapsed time in seconds

  // 1. Clear & Background Fill
  ctx.save();
  ctx.fillStyle = state.colorPalette.background;
  ctx.fillRect(0, 0, width, height);

  // 2. Cinematic Letterbox & Negative Space Margins
  // negativeSpaceRatio determines the proportion of darkened cinematic void
  const letterboxHeight = Math.floor(height * (state.negativeSpaceRatio * 0.22));
  if (letterboxHeight > 0) {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, letterboxHeight);
    ctx.fillRect(0, height - letterboxHeight, width, letterboxHeight);
  }

  // 3. Atmospheric Fog / Mist Layers
  if (state.mistOpacity > 0.05) {
    const mistBreath = 0.85 + 0.15 * Math.sin(t * 0.8);
    ctx.fillStyle = state.colorPalette.secondary;
    ctx.globalAlpha = clamp(state.mistOpacity * mistBreath * 0.45, 0.0, 1.0);
    ctx.fillRect(0, letterboxHeight, width, height - 2 * letterboxHeight);
  }

  // 4. Rule-of-Thirds & Golden-Ratio Lattice Overlay
  if (state.gridSymmetryAlpha > 0.05) {
    ctx.strokeStyle = state.colorPalette.primary;
    ctx.globalAlpha = clamp(state.gridSymmetryAlpha * 0.35, 0.0, 1.0);
    ctx.lineWidth = 1;

    // Rule of thirds lines
    const xThird1 = width / 3.0;
    const xThird2 = (width * 2.0) / 3.0;
    const yThird1 = height / 3.0;
    const yThird2 = (height * 2.0) / 3.0;

    ctx.beginPath();
    ctx.moveTo(xThird1, letterboxHeight);
    ctx.lineTo(xThird1, height - letterboxHeight);
    ctx.moveTo(xThird2, letterboxHeight);
    ctx.lineTo(xThird2, height - letterboxHeight);
    ctx.moveTo(0, yThird1);
    ctx.lineTo(width, yThird1);
    ctx.moveTo(0, yThird2);
    ctx.lineTo(width, yThird2);
    ctx.stroke();
  }

  // 5. Bokeh & Kinetic Particle Simulation
  const count = state.particleCount;
  if (count > 0) {
    const vx = state.panVelocity[0] * 12.0;
    const vy = state.panVelocity[1] * 8.0;

    for (let i = 0; i < count; i++) {
      // Deterministic pseudo-random seed per particle
      const seedX = Math.sin(i * 12.9898 + 43.12) * 43758.5453;
      const seedY = Math.cos(i * 78.233 + 12.54) * 43758.5453;
      const baseX = Math.abs(seedX - Math.floor(seedX)) * width;
      const baseY = Math.abs(seedY - Math.floor(seedY)) * height;

      // Particle motion
      const px = (baseX + t * vx + Math.sin(t + i) * 6.0) % width;
      const py = (baseY + t * vy + Math.cos(t * 0.7 + i) * 4.0) % height;

      const normY = py < letterboxHeight || py > height - letterboxHeight ? 0 : 1;
      if (normY === 0) continue;

      const radius = 1.0 + 4.0 * state.bokehLuminance * (0.8 + 0.2 * Math.sin(t * 2.0 + i));
      ctx.fillStyle = state.colorPalette.rim;
      ctx.globalAlpha = clamp(0.2 + 0.6 * state.bokehLuminance, 0.0, 1.0);

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 6. Chiaroscuro Rim Light (Vignette Gradient)
  if (state.chiaroscuroRimLight > 0.1) {
    ctx.strokeStyle = state.colorPalette.rim;
    ctx.lineWidth = Math.max(2, Math.floor(state.chiaroscuroRimLight * 6));
    ctx.globalAlpha = clamp(state.chiaroscuroRimLight * 0.65, 0.0, 1.0);
    ctx.strokeRect(0, letterboxHeight, width, height - 2 * letterboxHeight);
  }

  // 7. Hard-Cut Soliton Impulse Flash
  if (state.cutImpulseProbability > 0.85 && frameIndex % 60 === 0) {
    ctx.fillStyle = '#ffffff';
    ctx.globalAlpha = 0.45;
    ctx.fillRect(0, 0, width, height);
  }

  ctx.restore();
}
