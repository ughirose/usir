/**
 * USIR v3.0 Cross-Modal Functors & Synesthetic Transpiler (Gap 6).
 *
 * Implements:
 * 1. UI/DOM Layout Tree Functor F_ui (extractUiDom24d)
 * 2. Visual Saccade Trajectory Functor F_visual (extractVisualSaccade24d)
 * 3. Synesthetic Transpiler (transpileCrossModal)
 */

import { extract24dV2 } from './waveletHermiteEncoder.js';

// ============================================================================
// Functor Component Descriptors
// ============================================================================

export interface UiDomNode {
  depth?: number;             // Nesting depth in DOM hierarchy (>= 1.0)
  whitespace_ratio?: number;  // Fraction of element bounding box as whitespace/padding in [0, 1]
  whitespaceRatio?: number;   // camelCase alias
  font_scale_ratio?: number;  // Relative font scale ratio to base font (e.g. 1.0, 1.5, 2.5)
  fontScaleRatio?: number;    // camelCase alias
  card_uniformity?: number;   // Consistency of border radius / background / elevation in [0, 1]
  cardUniformity?: number;    // camelCase alias
  contrast_jump?: number;     // Delta luminance / color contrast relative to parent/neighbor in [0, 1]
  contrastJump?: number;      // camelCase alias
  padding_rhythm?: number;    // Adherence to spacing scale (4/8/16/24/32px) in [0, 1]
  paddingRhythm?: number;     // camelCase alias
}

export interface VisualSaccadePatch {
  luminance_contrast?: number;    // Michelson or Weber contrast in [0, 1]
  luminanceContrast?: number;     // camelCase alias
  edge_density?: number;          // Sobel/Canny edge density in [0, 1]
  edgeDensity?: number;           // camelCase alias
  negative_space_ratio?: number;  // Fraction of uniform / untextured background in [0, 1]
  negativeSpaceRatio?: number;    // camelCase alias
  chromatic_jump?: number;        // Color shift / delta-E jump in [0, 1]
  chromaticJump?: number;         // camelCase alias
  duration_ms?: number;           // Fixation duration in ms
  durationMs?: number;            // camelCase alias
  x?: number;                     // Spatial coordinate x in [0, 1]
  y?: number;                     // Spatial coordinate y in [0, 1]
}

// ============================================================================
// Transpiled Generative Parameter Interfaces
// ============================================================================

export interface MusicTranspiledParams {
  bpm_fluctuation: number;
  harmonic_dissonance: number;
  rest_ratio: number;
  rubato_depth: number;
  soliton_accent_gain: number;
  arch_return_strength: number;

  // camelCase aliases
  bpmFluctuation?: number;
  harmonicDissonance?: number;
  restRatio?: number;
  rubatoDepth?: number;
  solitonAccentGain?: number;
  archReturnStrength?: number;
}

export interface DiffusionTranspiledParams {
  cfg_scale: number;
  high_freq_noise_gain: number;
  negative_space_ratio: number;
  color_contrast_temperature: number;
  composition_symmetry: number;
  edge_sharpness: number;

  // camelCase aliases
  cfgScale?: number;
  highFreqNoiseGain?: number;
  negativeSpaceRatio?: number;
  colorContrastTemperature?: number;
  compositionSymmetry?: number;
  edgeSharpness?: number;
}

export interface UiCssTranspiledParams {
  whitespace_multiplier: number;
  contrast_ratio: number;
  border_radius_px: number;
  typography_scale_ratio: number;
  grid_irregularity: number;
  transition_duration_ms: number;

  // camelCase aliases
  whitespaceMultiplier?: number;
  contrastRatio?: number;
  borderRadiusPx?: number;
  typographyScaleRatio?: number;
  gridIrregularity?: number;
  transitionDurationMs?: number;
}

export type TranspiledParams =
  | MusicTranspiledParams
  | DiffusionTranspiledParams
  | UiCssTranspiledParams;

// ============================================================================
// Functor F_ui: UI/DOM Layout Tree Functor
// ============================================================================

function parseHtmlToNodes(htmlStr: string): UiDomNode[] {
  const tagRegex = /<([a-zA-Z0-9]+)([^>]*)>(.*?)(?=<\s*\/?[a-zA-Z0-9]+|$)/gis;
  const matches = [...htmlStr.matchAll(tagRegex)];

  if (matches.length === 0) {
    return [{
      depth: 1.0,
      whitespace_ratio: 0.3,
      font_scale_ratio: 1.0,
      card_uniformity: 0.6,
      contrast_jump: 0.1,
      padding_rhythm: 0.6,
    }];
  }

  const nodes: UiDomNode[] = [];
  let currentDepth = 1;

  for (const m of matches) {
    const tagName = m[1]?.toLowerCase() ?? '';
    const attrs = m[2]?.toLowerCase() ?? '';
    const content = m[3] ?? '';

    if (['html', 'body', 'head', 'meta', 'link', 'script', 'style'].includes(tagName)) {
      continue;
    }

    if (['div', 'section', 'article', 'main', 'aside', 'nav', 'ul', 'ol'].includes(tagName)) {
      currentDepth = Math.min(currentDepth + 1, 8);
    }

    let fontScale = 1.0;
    if (tagName === 'h1') fontScale = 2.5;
    else if (tagName === 'h2') fontScale = 2.0;
    else if (tagName === 'h3') fontScale = 1.5;
    else if (['h4', 'h5', 'h6'].includes(tagName)) fontScale = 1.25;
    else if (['small', 'sub', 'sup'].includes(tagName)) fontScale = 0.8;

    let whitespaceRatio = 0.2;
    if (/p-|m-|gap-|py-|px-|padding|margin/.test(attrs)) whitespaceRatio += 0.3;
    if (/empty/.test(attrs) || !content.trim()) whitespaceRatio += 0.4;
    whitespaceRatio = Math.min(1.0, whitespaceRatio);

    let cardUniformity = 0.5;
    if (/card|rounded|shadow|border|grid|flex/.test(attrs)) cardUniformity = 0.85;

    let contrastJump = 0.1;
    if (/bg-primary|bg-black|bg-accent|text-white|btn|danger|warning/.test(attrs)) {
      contrastJump = 0.75;
    } else if (['button', 'a', 'mark'].includes(tagName)) {
      contrastJump = 0.60;
    }

    let paddingRhythm = 0.6;
    if (/gap-4|p-4|m-4|p-2|p-8/.test(attrs)) paddingRhythm = 0.90;

    nodes.push({
      depth: currentDepth,
      whitespace_ratio: whitespaceRatio,
      font_scale_ratio: fontScale,
      card_uniformity: cardUniformity,
      contrast_jump: contrastJump,
      padding_rhythm: paddingRhythm,
    });
  }

  if (nodes.length === 0) {
    nodes.push({
      depth: 1.0,
      whitespace_ratio: 0.2,
      font_scale_ratio: 1.0,
      card_uniformity: 0.5,
      contrast_jump: 0.1,
      padding_rhythm: 0.5,
    });
  }

  return nodes;
}

export function extractUiDomSignals(
  nodesOrHtml: string | (UiDomNode | Record<string, unknown>)[]
): number[][] {
  const nodes: UiDomNode[] =
    typeof nodesOrHtml === 'string'
      ? parseHtmlToNodes(nodesOrHtml)
      : nodesOrHtml.map((item) => ({
          depth: Number(item.depth ?? 1.0),
          whitespace_ratio: Number(item.whitespace_ratio ?? item.whitespaceRatio ?? 0.2),
          font_scale_ratio: Number(item.font_scale_ratio ?? item.fontScaleRatio ?? 1.0),
          card_uniformity: Number(item.card_uniformity ?? item.cardUniformity ?? 0.5),
          contrast_jump: Number(item.contrast_jump ?? item.contrastJump ?? 0.1),
          padding_rhythm: Number(item.padding_rhythm ?? item.paddingRhythm ?? 0.5),
        }));

  const m = nodes.length;
  if (m === 0) {
    return new Array(16).fill(0).map(() => [0, 0, 0, 0]);
  }

  const signals: number[][] = [];
  for (let i = 0; i < m; i++) {
    const n = nodes[i]!;
    const prevFont = i > 0 ? (nodes[i - 1]?.font_scale_ratio ?? n.font_scale_ratio ?? 1.0) : (n.font_scale_ratio ?? 1.0);
    const fontDelta = Math.abs((n.font_scale_ratio ?? 1.0) - prevFont);

    // 1. Flux
    const s1 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.40 * Math.min((n.font_scale_ratio ?? 1.0) / 3.0, 1.0) +
          0.35 * Math.min((n.depth ?? 1.0) / 8.0, 1.0) +
          0.25 * Math.min(fontDelta, 1.0)
      )
    );

    // 2. Lattice
    const s2 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.50 * Math.min(Math.max(n.card_uniformity ?? 0.5, 0.0), 1.0) +
          0.50 * Math.min(Math.max(n.padding_rhythm ?? 0.5, 0.0), 1.0)
      )
    );

    // 3. Void [間]
    const depthPenalty = Math.max(0.0, 1.0 - (n.depth ?? 1.0) / 10.0);
    const s3 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.75 * Math.min(Math.max(n.whitespace_ratio ?? 0.2, 0.0), 1.0) +
          0.25 * depthPenalty
      )
    );

    // 4. Shock
    const cJump = Math.min(Math.max(n.contrast_jump ?? 0.1, 0.0), 1.0);
    const cardUnif = Math.min(Math.max(n.card_uniformity ?? 0.5, 0.0), 1.0);
    const s4 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.70 * cJump + 0.30 * (cJump > 0.4 ? 1.0 - cardUnif : 0.0)
      )
    );

    signals.push([s1, s2, s3, s4]);
  }

  return signals;
}

export function extractUiDom24d(
  nodesOrHtml: string | (UiDomNode | Record<string, unknown>)[],
  useMca: boolean = true
): Float64Array {
  const signals = extractUiDomSignals(nodesOrHtml);
  return extract24dV2(signals, useMca);
}

// ============================================================================
// Functor F_visual: Visual Saccade Trajectory Functor
// ============================================================================

export function extractVisualSaccadeSignals(
  patches: (VisualSaccadePatch | Record<string, unknown>)[]
): number[][] {
  const cleanPatches = patches.map((p) => ({
    luminance_contrast: Number(p.luminance_contrast ?? p.luminanceContrast ?? 0.2),
    edge_density: Number(p.edge_density ?? p.edgeDensity ?? 0.3),
    negative_space_ratio: Number(p.negative_space_ratio ?? p.negativeSpaceRatio ?? 0.4),
    chromatic_jump: Number(p.chromatic_jump ?? p.chromaticJump ?? 0.1),
    duration_ms: Number(p.duration_ms ?? p.durationMs ?? 200.0),
    x: Number(p.x ?? 0.5),
    y: Number(p.y ?? 0.5),
  }));

  const m = cleanPatches.length;
  if (m === 0) {
    return new Array(16).fill(0).map(() => [0, 0, 0, 0]);
  }

  const signals: number[][] = [];
  for (let i = 0; i < m; i++) {
    const p = cleanPatches[i]!;
    let saccadeDist = 0.1;
    if (i > 0) {
      const prev = cleanPatches[i - 1]!;
      const dx = p.x - prev.x;
      const dy = p.y - prev.y;
      saccadeDist = Math.sqrt(dx * dx + dy * dy);
    }

    // 1. Flux
    const s1 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.45 * Math.min(Math.max(p.edge_density, 0.0), 1.0) +
          0.30 * Math.min(saccadeDist * 1.5, 1.0) +
          0.25 * Math.min(Math.max(p.chromatic_jump, 0.0), 1.0)
      )
    );

    // 2. Lattice
    const s2 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.55 * (1.0 - Math.abs(p.edge_density - 0.5) * 1.8) +
          0.45 * (1.0 - Math.min(Math.max(p.chromatic_jump, 0.0), 1.0))
      )
    );

    // 3. Void [間]
    const s3 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.80 * Math.min(Math.max(p.negative_space_ratio, 0.0), 1.0) +
          0.20 * (1.0 - Math.min(Math.max(p.edge_density, 0.0), 1.0))
      )
    );

    // 4. Shock
    const s4 = Math.min(
      1.0,
      Math.max(
        0.0,
        0.60 * Math.min(Math.max(p.luminance_contrast, 0.0), 1.0) +
          0.40 * Math.min(Math.max(p.chromatic_jump, 0.0), 1.0)
      )
    );

    signals.push([s1, s2, s3, s4]);
  }

  return signals;
}

export function extractVisualSaccade24d(
  patches: (VisualSaccadePatch | Record<string, unknown>)[],
  useMca: boolean = true
): Float64Array {
  const signals = extractVisualSaccadeSignals(patches);
  return extract24dV2(signals, useMca);
}

// ============================================================================
// Synesthetic Transpiler
// ============================================================================

export function transpileCrossModal(
  z24d: Float64Array | number[],
  targetModality: 'music' | 'diffusion' | 'ui_css' | string
): TranspiledParams {
  const z = Array.from(z24d);
  if (z.length < 24) {
    throw new ValueError(`State vector must have 24 dimensions, got ${z.length}`);
  }

  const q = z.slice(0, 12);
  const p = z.slice(12, 24);

  const mod = targetModality.trim().toLowerCase();

  const clip = (val: number, minVal: number, maxVal: number) =>
    Math.min(maxVal, Math.max(minVal, val));

  if (mod === 'music') {
    const bpm_fluctuation = clip(
      0.05 + 0.35 * (0.4 * q[0]! + 0.3 * q[4]! + 0.3 * q[8]!) + 0.15 * Math.tanh(p[0]! + p[4]!),
      0.0,
      1.0
    );
    const harmonic_dissonance = clip(
      0.50 * (q[3]! + q[7]! + q[11]!) / 3.0 - 0.30 * (q[1]! + q[5]! + q[9]!) / 3.0 + 0.25,
      0.0,
      1.0
    );
    const rest_ratio = clip(
      0.20 * q[2]! + 0.30 * q[6]! + 0.50 * q[10]!,
      0.0,
      1.0
    );
    const rubato_depth = clip(
      0.10 + 0.50 * q[8]! + 0.30 * (1.0 - q[1]!) * Math.tanh(p[8]!),
      0.0,
      1.0
    );
    const soliton_accent_gain = clip(
      0.70 * q[11]! + 0.30 * q[7]! + 0.20 * Math.tanh(p[11]!),
      0.0,
      2.0
    );
    const arch_return_strength = clip(
      0.80 * q[9]! + 0.20 * q[5]!,
      0.0,
      1.0
    );

    return {
      bpm_fluctuation,
      harmonic_dissonance,
      rest_ratio,
      rubato_depth,
      soliton_accent_gain,
      arch_return_strength,
      // camelCase aliases
      bpmFluctuation: bpm_fluctuation,
      harmonicDissonance: harmonic_dissonance,
      restRatio: rest_ratio,
      rubatoDepth: rubato_depth,
      solitonAccentGain: soliton_accent_gain,
      archReturnStrength: arch_return_strength,
    };
  } else if (mod === 'diffusion') {
    const cfg_scale = clip(
      4.0 + 8.0 * (0.4 * q[1]! + 0.4 * q[5]! + 0.2 * q[9]!) - 3.0 * q[10]!,
      1.5,
      14.0
    );
    const high_freq_noise_gain = clip(
      0.10 + 0.65 * q[11]! + 0.25 * q[3]! + 0.20 * Math.tanh(p[3]! + p[7]!),
      0.0,
      1.5
    );
    const negative_space_ratio = clip(
      0.15 + 0.25 * q[2]! + 0.25 * q[6]! + 0.35 * q[10]!,
      0.0,
      1.0
    );
    const color_contrast_temperature = clip(
      0.20 + 0.40 * q[3]! + 0.40 * q[7]! + 0.20 * q[11]!,
      0.0,
      1.0
    );
    const composition_symmetry = clip(
      0.60 * q[9]! + 0.40 * q[1]! - 0.30 * q[3]! - 0.30 * q[11]! + 0.20,
      0.0,
      1.0
    );
    const edge_sharpness = clip(
      0.50 * q[1]! + 0.30 * q[5]! + 0.40 * q[3]! - 0.30 * q[10]!,
      0.0,
      1.0
    );

    return {
      cfg_scale,
      high_freq_noise_gain,
      negative_space_ratio,
      color_contrast_temperature,
      composition_symmetry,
      edge_sharpness,
      // camelCase aliases
      cfgScale: cfg_scale,
      highFreqNoiseGain: high_freq_noise_gain,
      negativeSpaceRatio: negative_space_ratio,
      colorContrastTemperature: color_contrast_temperature,
      compositionSymmetry: composition_symmetry,
      edgeSharpness: edge_sharpness,
    };
  } else if (mod === 'ui_css' || mod === 'uicss' || mod === 'ui') {
    const whitespace_multiplier = clip(
      0.80 + 2.50 * (0.2 * q[2]! + 0.3 * q[6]! + 0.5 * q[10]!),
      0.5,
      4.0
    );
    const contrast_ratio = clip(
      3.0 + 12.0 * (0.5 * q[1]! + 0.3 * q[3]! + 0.2 * q[11]!),
      3.0,
      18.0
    );
    const border_radius_px = clip(
      16.0 * (0.4 * q[2]! + 0.4 * q[10]! + 0.2 * q[9]!) - 10.0 * (q[1]! + q[11]!),
      0.0,
      32.0
    );
    const typography_scale_ratio = clip(
      1.10 + 0.45 * (0.4 * q[0]! + 0.3 * q[8]! + 0.3 * q[11]!),
      1.05,
      1.60
    );
    const grid_irregularity = clip(
      0.45 * q[3]! + 0.30 * q[7]! + 0.40 * q[11]! - 0.35 * q[1]! + 0.10,
      0.0,
      1.0
    );
    const transition_duration_ms = clip(
      150.0 + 500.0 * q[10]! - 250.0 * q[11]!,
      50.0,
      800.0
    );

    return {
      whitespace_multiplier,
      contrast_ratio,
      border_radius_px,
      typography_scale_ratio,
      grid_irregularity,
      transition_duration_ms,
      // camelCase aliases
      whitespaceMultiplier: whitespace_multiplier,
      contrastRatio: contrast_ratio,
      borderRadiusPx: border_radius_px,
      typographyScaleRatio: typography_scale_ratio,
      gridIrregularity: grid_irregularity,
      transitionDurationMs: transition_duration_ms,
    };
  } else {
    throw new Error(`Unknown target modality: ${targetModality}. Must be 'music', 'diffusion', or 'ui_css'.`);
  }
}

class ValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValueError';
  }
}

export const transpileToMusic = (z: Float64Array | number[]): MusicTranspiledParams =>
  transpileCrossModal(z, 'music') as MusicTranspiledParams;

export const transpileToVisual = (z: Float64Array | number[]): DiffusionTranspiledParams =>
  transpileCrossModal(z, 'diffusion') as DiffusionTranspiledParams;

export const transpileToCssVariables = (z: Float64Array | number[]): UiCssTranspiledParams =>
  transpileCrossModal(z, 'ui_css') as UiCssTranspiledParams;

export const StyleToMusicFunctor = { transpile: transpileToMusic };
export const StyleToVisualFunctor = { transpile: transpileToVisual };
export const StyleToHapticFunctor = { transpile: transpileToCssVariables };

