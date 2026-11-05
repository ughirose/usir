/**
 * @nishiki/usir-bridges: Universal Cross-Modal Functors F_visual & F_music
 * Phase E Implementation (WS-E1).
 *
 * Implements bidirectional transpilations between 24D USIR Symplectic Manifold
 * and visual/musical generative spaces:
 * 1. F_visual:
 *    - Forward: 2D image features (4-direction Gabor frequencies, color entropy,
 *      chiaroscuro gradient, spatial quadtree depth, saliency scanpath) -> 24D .usir
 *    - Inverse (F_visual^{-1}): 24D .usir -> Diffusion generative parameters
 *      (CFG scale, high-frequency noise gain, composition margin ratio).
 * 2. F_music:
 *    - Forward: Note/chord sequences (Circle of Fifths torus distance, dissonance,
 *      syncopation, rest ratio) -> 24D .usir
 *    - Inverse (F_music^{-1}): 24D .usir -> WebAudio / MIDI sequence controls
 *      (BPM, harmonic tension, rest interval, reverb decay).
 */

// ============================================================================
// 1. Visual Modality Interfaces & Functors (F_visual)
// ============================================================================

export interface SaliencyFixation {
  x: number;          // Normalized coordinate x in [0.0, 1.0]
  y: number;          // Normalized coordinate y in [0.0, 1.0]
  intensity: number;  // Saliency peak intensity in [0.0, 1.0]
  durationMs?: number;
}

export interface VisualFeatures2D {
  gaborFrequencies: [number, number, number, number]; // [0 deg, 45 deg, 90 deg, 135 deg] energy
  colorEntropy: number;           // Color space entropy in [0.0, 1.0]
  chiaroscuroGradient: number;    // Local luminance contrast gradient in [0.0, 1.0]
  quadtreeDepthEntropy: number;   // Spatial quadtree depth distribution entropy in [0.0, 1.0]
  saliencyScanpath: SaliencyFixation[];
  compositionMarginRatio?: number; // Background / margin whitespace in [0.0, 1.0]
  /** Ordered local blocks / fixation patches. With >= 2, all 12 momentum coordinates are measured. */
  patchFeatures?: VisualFeatures2D[];
}

export interface ImageData2D {
  width: number;
  height: number;
  // Grayscale or RGB pixel buffer. If Uint8Array of length width*height, treated as grayscale.
  // If length width*height*3 or *4, treated as RGB/RGBA.
  data: number[] | Uint8Array | Float32Array;
}

export interface VisualGenerativeParams {
  cfgScale: number;                  // Classifier-Free Guidance scale in [1.5, 14.0]
  highFreqNoiseGain: number;         // High frequency noise / grain gain in [0.0, 2.0]
  compositionMarginRatio: number;    // Composition margin / negative space in [0.0, 1.0]
  colorContrastTemperature: number;  // Warm/cool chromatic contrast in [0.0, 1.0]
  edgeSharpness: number;             // Sharpness / high-pass filter weight in [0.0, 1.0]
  compositionSymmetry: number;       // Structural bilateral / quad symmetry in [0.0, 1.0]
}

/**
 * Extracts 2D visual features from a raw pixel buffer using Gabor filters,
 * color entropy, chiaroscuro contrast gradient, and spatial quadtree decomposition.
 */
function extractVisualSummary(image: ImageData2D): VisualFeatures2D {
  const { width, height, data } = image;
  const numPixels = width * height;
  if (numPixels === 0) {
    return {
      gaborFrequencies: [0.25, 0.25, 0.25, 0.25],
      colorEntropy: 0.5,
      chiaroscuroGradient: 0.3,
      quadtreeDepthEntropy: 0.5,
      saliencyScanpath: [{ x: 0.5, y: 0.5, intensity: 0.5 }],
      compositionMarginRatio: 0.3,
    };
  }

  const channels = data.length >= numPixels * 3 ? (data.length >= numPixels * 4 ? 4 : 3) : 1;
  const gray = new Float32Array(numPixels);

  if (channels === 1) {
    for (let i = 0; i < numPixels; i++) {
      gray[i] = (data[i] as number) / 255.0;
    }
  } else {
    for (let i = 0; i < numPixels; i++) {
      const offset = i * channels;
      const r = (data[offset] as number) / 255.0;
      const g = (data[offset + 1] as number) / 255.0;
      const b = (data[offset + 2] as number) / 255.0;
      gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }

  // 1. Color / Luminance Histogram Entropy
  const numBins = 16;
  const hist = new Float32Array(numBins);
  for (let i = 0; i < numPixels; i++) {
    const bin = Math.min(numBins - 1, Math.floor(gray[i]! * numBins));
    hist[bin] = (hist[bin] || 0) + 1.0;
  }
  let colorEntropy = 0.0;
  for (let b = 0; b < numBins; b++) {
    const p = hist[b]! / numPixels;
    if (p > 1e-7) {
      colorEntropy -= p * Math.log2(p);
    }
  }
  // Normalize by log2(16) = 4
  colorEntropy = Math.min(1.0, colorEntropy / 4.0);

  // 2. Chiaroscuro Gradient (RMS Sobel luminance contrast)
  let gradSum = 0.0;
  let gradCount = 0;
  const gradMap = new Float32Array(numPixels);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const gx =
        -gray[(y - 1) * width + (x - 1)]! +
        gray[(y - 1) * width + (x + 1)]! -
        2.0 * gray[y * width + (x - 1)]! +
        2.0 * gray[y * width + (x + 1)]! -
        gray[(y + 1) * width + (x - 1)]! +
        gray[(y + 1) * width + (x + 1)]!;
      const gy =
        -gray[(y - 1) * width + (x - 1)]! -
        2.0 * gray[(y - 1) * width + x]! -
        gray[(y - 1) * width + (x + 1)]! +
        gray[(y + 1) * width + (x - 1)]! +
        2.0 * gray[(y + 1) * width + x]! +
        gray[(y + 1) * width + (x + 1)]!;
      const mag = Math.sqrt(gx * gx + gy * gy) / 4.0;
      gradMap[idx] = mag;
      gradSum += mag;
      gradCount++;
    }
  }
  const chiaroscuroGradient = gradCount > 0 ? Math.min(1.0, (gradSum / gradCount) * 2.5) : 0.2;

  // 3. 4-Direction Gabor Spatial Frequency Approximation
  let e0 = 0.0;   // Horizontal (0 deg)
  let e45 = 0.0;  // Diagonal 45 deg
  let e90 = 0.0;  // Vertical (90 deg)
  let e135 = 0.0; // Diagonal 135 deg

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const c = gray[y * width + x]!;
      const hDiff = Math.abs(c - gray[y * width + (x + 1)]!);
      const vDiff = Math.abs(c - gray[(y + 1) * width + x]!);
      const d1Diff = Math.abs(c - gray[(y + 1) * width + (x + 1)]!);
      const d2Diff = Math.abs(c - gray[(y + 1) * width + (x - 1)]!);

      e0 += hDiff;
      e90 += vDiff;
      e45 += d1Diff;
      e135 += d2Diff;
    }
  }
  const rawTotalGabor = e0 + e45 + e90 + e135;
  const gaborFrequencies: [number, number, number, number] =
    rawTotalGabor < 1e-6
      ? [0.25, 0.25, 0.25, 0.25]
      : [
          e0 / rawTotalGabor,
          e45 / rawTotalGabor,
          e90 / rawTotalGabor,
          e135 / rawTotalGabor,
        ];

  // 4. Spatial Quadtree Decomposition & Depth Entropy
  const maxDepth = 4;
  const depthCounts = new Array<number>(maxDepth + 1).fill(0);

  function splitQuadtree(x0: number, y0: number, w: number, h: number, depth: number): void {
    if (depth >= maxDepth || w <= 4 || h <= 4) {
      depthCounts[depth] = (depthCounts[depth] || 0) + 1;
      return;
    }
    // Calculate variance of region
    let sum = 0.0;
    let sumSq = 0.0;
    let count = 0;
    for (let y = y0; y < y0 + h; y++) {
      for (let x = x0; x < x0 + w; x++) {
        const val = gray[y * width + x]!;
        sum += val;
        sumSq += val * val;
        count++;
      }
    }
    const variance = count > 0 ? sumSq / count - (sum / count) ** 2 : 0;
    if (variance < 0.005) {
      depthCounts[depth] = (depthCounts[depth] || 0) + 1;
    } else {
      const hw = Math.floor(w / 2);
      const hh = Math.floor(h / 2);
      splitQuadtree(x0, y0, hw, hh, depth + 1);
      splitQuadtree(x0 + hw, y0, w - hw, hh, depth + 1);
      splitQuadtree(x0, y0 + hh, hw, h - hh, depth + 1);
      splitQuadtree(x0 + hw, y0 + hh, w - hw, h - hh, depth + 1);
    }
  }

  splitQuadtree(0, 0, width, height, 0);

  const totalLeaves = depthCounts.reduce((a, b) => a + b, 0);
  let quadtreeDepthEntropy = 0.0;
  for (let d = 0; d <= maxDepth; d++) {
    const p = depthCounts[d]! / (totalLeaves || 1);
    if (p > 1e-7) {
      quadtreeDepthEntropy -= p * Math.log2(p);
    }
  }
  quadtreeDepthEntropy = Math.min(1.0, quadtreeDepthEntropy / Math.log2(maxDepth + 1));

  // 5. Saliency Scanpath (Local extrema of gradient magnitude)
  const saliencyScanpath: SaliencyFixation[] = [];
  const step = Math.max(4, Math.floor(Math.min(width, height) / 8));
  for (let y = step; y < height - step; y += step) {
    for (let x = step; x < width - step; x += step) {
      const val = gradMap[y * width + x]!;
      if (val > chiaroscuroGradient * 0.8) {
        saliencyScanpath.push({
          x: x / width,
          y: y / height,
          intensity: Math.min(1.0, val * 2.0),
        });
      }
    }
  }
  if (saliencyScanpath.length === 0) {
    saliencyScanpath.push({ x: 0.5, y: 0.5, intensity: 0.5 });
  }

  const compositionMarginRatio = Math.max(0.0, Math.min(1.0, 1.0 - chiaroscuroGradient));

  return {
    gaborFrequencies,
    colorEntropy,
    chiaroscuroGradient,
    quadtreeDepthEntropy,
    saliencyScanpath,
    compositionMarginRatio,
  };
}

/**
 * Extracts 2D visual features for the whole image and, for images large enough, for a g x g grid of
 * local blocks in row-major order (patchFeatures). The block series gives the window-to-window
 * variation from which all 12 momentum coordinates are measured.
 */
export function extractVisualFeaturesFromImage(image: ImageData2D): VisualFeatures2D {
  const summary = extractVisualSummary(image);
  const { width, height, data } = image;
  const g = Math.min(3, Math.floor(Math.min(width, height) / 16));
  if (g < 2) return summary;

  const numPixels = width * height;
  const channels = data.length >= numPixels * 3 ? (data.length >= numPixels * 4 ? 4 : 3) : 1;
  const patches: VisualFeatures2D[] = [];
  for (let by = 0; by < g; by++) {
    for (let bx = 0; bx < g; bx++) {
      const x0 = Math.floor((bx * width) / g);
      const x1 = Math.floor(((bx + 1) * width) / g);
      const y0 = Math.floor((by * height) / g);
      const y1 = Math.floor(((by + 1) * height) / g);
      const w = x1 - x0;
      const h = y1 - y0;
      const sub = new Array<number>(w * h * channels);
      let o = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          for (let c = 0; c < channels; c++) {
            sub[o++] = data[(y * width + x) * channels + c] as number;
          }
        }
      }
      patches.push(extractVisualSummary({ width: w, height: h, data: sub }));
    }
  }
  return { ...summary, patchFeatures: patches };
}

const clamp01 = (x: number): number => Math.max(0.0, Math.min(1.0, x));

/**
 * p_i = sqrt(1/(M-1) sum_m (q_{m+1,i} - q_{m,i})^2): window-to-window RMS velocity of each element.
 * Needs M >= 2 windows; returns zeros otherwise. Non-negative and not a copy of static q.
 */
export function momentumFromWindowSeries(qSeries: number[][]): number[] {
  const p = new Array<number>(12).fill(0);
  if (qSeries.length < 2) return p;
  for (let i = 0; i < 12; i++) {
    let sum = 0.0;
    for (let m = 1; m < qSeries.length; m++) {
      const d = qSeries[m]![i]! - qSeries[m - 1]![i]!;
      sum += d * d;
    }
    p[i] = Math.sqrt(sum / (qSeries.length - 1));
  }
  return p;
}

/**
 * Equilibrium Momentum Reconstruction (summary-only path). Mirrors python
 * `momentum_from_summary` in nishiki_unified_field.py. g = Lambda (q - mu) with a fixed 4-group
 * coupling matrix (unit diagonal, 0.3 within group, 0.1 across groups; eigenvalues 0.7/1.3/2.5,
 * positive definite); p_i = SCALE * sqrt(g_i^2 + FLOOR^2) > 0. Legacy proxy dims (non-zero) combine
 * as hypot(proxy_i, p_i).
 */
const EQ_GROUPS: number[][] = [[0, 3, 7], [1, 4, 8], [2, 5, 9], [6, 10, 11]];
const EQ_W_WITHIN = 0.3;
const EQ_W_CROSS = 0.1;
const EQ_MU = 0.5;
const EQ_FLOOR = 0.02;
const EQ_SCALE = 2.2;

export function momentumFromSummary(q: number[], legacyProxy?: number[]): number[] {
  const groupOf = new Array<number>(12).fill(0);
  EQ_GROUPS.forEach((g, gi) => g.forEach((i) => { groupOf[i] = gi; }));
  const p = new Array<number>(12).fill(0);
  for (let i = 0; i < 12; i++) {
    let g = 0.0;
    for (let j = 0; j < 12; j++) {
      const lam = i === j ? 1.0 : groupOf[i] === groupOf[j] ? EQ_W_WITHIN : EQ_W_CROSS;
      g += lam * (q[j]! - EQ_MU);
    }
    p[i] = EQ_SCALE * Math.sqrt(g * g + EQ_FLOOR * EQ_FLOOR);
    const proxy = legacyProxy?.[i] ?? 0.0;
    if (proxy !== 0.0) p[i] = Math.hypot(proxy, p[i]!);
  }
  return p;
}

interface VisualQTerms {
  q: number[];
  saccadeMean: number;
  gaborVar: number;
  chiaro: number;
  marginRatio: number;
}

function visualQAndTerms(features: VisualFeatures2D): VisualQTerms {
  const gabor = features.gaborFrequencies;
  const colorEnt = clamp01(features.colorEntropy);
  const chiaro = clamp01(features.chiaroscuroGradient);
  const quadEnt = clamp01(features.quadtreeDepthEntropy);
  const scanpath = features.saliencyScanpath;

  let saccadeMean = 0.2;
  if (scanpath && scanpath.length >= 2) {
    let sumDist = 0.0;
    for (let i = 1; i < scanpath.length; i++) {
      const dx = scanpath[i]!.x - scanpath[i - 1]!.x;
      const dy = scanpath[i]!.y - scanpath[i - 1]!.y;
      sumDist += Math.sqrt(dx * dx + dy * dy);
    }
    saccadeMean = sumDist / (scanpath.length - 1);
  }

  const gaborMean = (gabor[0] + gabor[1] + gabor[2] + gabor[3]) / 4.0;
  const gaborVar =
    ((gabor[0] - gaborMean) ** 2 +
      (gabor[1] - gaborMean) ** 2 +
      (gabor[2] - gaborMean) ** 2 +
      (gabor[3] - gaborMean) ** 2) /
    4.0;

  const marginRatio =
    features.compositionMarginRatio !== undefined ? clamp01(features.compositionMarginRatio) : 1.0 - chiaro;

  const q = new Array<number>(12).fill(0);
  // Group I: Flux
  q[0] = clamp01(0.5 * saccadeMean + 0.5 * gaborVar * 4.0);
  q[3] = clamp01(0.4 * gabor[0] + 0.6 * saccadeMean);
  q[7] = clamp01(0.6 * gaborVar + 0.4 * colorEnt);
  // Group II: Lattice
  q[1] = clamp01(quadEnt);
  q[4] = clamp01(0.5 * (1.0 - gaborVar * 2.0) + 0.5 * quadEnt);
  q[8] = clamp01(0.7 * (1.0 - colorEnt) + 0.3 * quadEnt);
  // Group III: Void
  q[2] = clamp01(marginRatio);
  q[5] = clamp01(0.6 * marginRatio + 0.4 * (1.0 - quadEnt));
  q[9] = clamp01(0.8 * marginRatio + 0.2 * (1.0 - chiaro));
  // Group IV: Shock
  q[6] = clamp01(chiaro);
  q[10] = clamp01(0.7 * chiaro + 0.3 * colorEnt);
  q[11] = clamp01(0.8 * chiaro + 0.2 * gaborVar * 5.0);

  return { q, saccadeMean, gaborVar, chiaro, marginRatio };
}

/**
 * Forward Visual Functor F_visual: Maps 2D visual features to 24D USIR symplectic state z = (q, p).
 * With >= 2 patchFeatures all 12 momentum coordinates are the RMS patch-to-patch change of the
 * per-patch elements. Without patches p is the Equilibrium Momentum Reconstruction
 * (momentumFromSummary; all 12 > 0, legacy proxy dims p[0,2,6,11] combined).
 */
export function visualFunctorForward(features: VisualFeatures2D): number[] {
  const t = visualQAndTerms(features);
  let p: number[];
  const patches = features.patchFeatures ?? [];
  if (patches.length >= 2) {
    p = momentumFromWindowSeries(patches.map((pf) => visualQAndTerms(pf).q));
  } else {
    p = new Array<number>(12).fill(0);
    p[0] = t.saccadeMean * 1.5;
    p[2] = t.marginRatio * 0.8;
    p[6] = t.chiaro * 1.8;
    p[11] = t.chiaro * 2.0 + t.gaborVar;
    p = momentumFromSummary(t.q, p);
  }
  return [...t.q, ...p];
}

/**
 * Inverse Visual Functor F_visual^{-1}: Maps 24D USIR symplectic state to Diffusion parameters.
 */
export function visualFunctorInverse(z24d: number[] | Float32Array | Float64Array): VisualGenerativeParams {
  if (z24d.length < 24) {
    throw new Error(`Expected 24-dimensional USIR state vector, got ${z24d.length}`);
  }
  const z = Array.from(z24d);
  const q = z.slice(0, 12);
  const p = z.slice(12, 24);

  // CFG Scale: Driven by Lattice (Sg-02, Sg-05, Sg-09) and restrained by Void (Sg-10)
  const cfgScale = Math.max(
    1.5,
    Math.min(
      14.0,
      3.5 + 8.5 * (0.4 * (q[1] || 0) + 0.4 * (q[4] || 0) + 0.2 * (q[8] || 0)) - 2.5 * (q[9] || 0)
    )
  );

  // High Frequency Noise Gain: Driven by Shock (Sg-12, Sg-07) and momentum
  const highFreqNoiseGain = Math.max(
    0.0,
    Math.min(
      2.0,
      0.1 + 0.7 * (q[11] || 0) + 0.3 * (q[6] || 0) + 0.15 * Math.tanh((p[6] || 0) + (p[11] || 0))
    )
  );

  // Composition Margin Ratio: Driven by Void (Sg-03, Sg-06, Sg-10)
  const compositionMarginRatio = Math.max(
    0.0,
    Math.min(
      1.0,
      0.1 + 0.3 * (q[2] || 0) + 0.3 * (q[5] || 0) + 0.4 * (q[9] || 0)
    )
  );

  // Chromatic contrast temperature
  const colorContrastTemperature = Math.max(
    0.0,
    Math.min(1.0, 0.2 + 0.5 * (q[10] || 0) + 0.3 * (q[7] || 0))
  );

  // Edge sharpness
  const edgeSharpness = Math.max(
    0.0,
    Math.min(1.0, 0.4 * (q[1] || 0) + 0.4 * (q[6] || 0) + 0.2 * (1.0 - (q[2] || 0)))
  );

  // Composition symmetry
  const compositionSymmetry = Math.max(
    0.0,
    Math.min(
      1.0,
      0.6 * (q[8] || 0) + 0.4 * (q[1] || 0) - 0.3 * (q[6] || 0) - 0.3 * (q[11] || 0) + 0.2
    )
  );

  return {
    cfgScale,
    highFreqNoiseGain,
    compositionMarginRatio,
    colorContrastTemperature,
    edgeSharpness,
    compositionSymmetry,
  };
}


// ============================================================================
// 2. Music Modality Interfaces & Functors (F_music)
// ============================================================================

export interface MusicNoteEvent {
  pitch: number;        // MIDI pitch 0-127 (e.g. 60 = C4)
  duration: number;     // Duration in seconds
  velocity: number;     // MIDI velocity 1-127
  startTime: number;    // Absolute start time in seconds
}

export interface MusicFeatures {
  torusDistance: number;   // Circle of Fifths torus distance in [0.0, 1.0]
  dissonance: number;      // Harmonic sensory dissonance in [0.0, 1.0]
  syncopation: number;     // Metric syncopation degree in [0.0, 1.0]
  restRatio: number;       // Silence / pause ratio in [0.0, 1.0]
  bpm?: number;            // Tempo in BPM (default 100)
  /** Ordered per-bar (time-window) features. With >= 2, all 12 momentum coordinates are measured. */
  barFeatures?: MusicFeatures[];
}

export interface WebAudioNoteDirective {
  note: number;
  time: number;
  duration: number;
  gain: number;
}

export interface MusicGenerativeParams {
  bpm: number;                         // Tempo in BPM in [50, 190]
  harmonicTension: number;             // Harmonic tension / dissonance in [0.0, 1.0]
  restInterval: number;                // Pause spacing / silence in [0.0, 1.0]
  reverbDecay: number;                 // Reverb decay tail in seconds [0.4, 6.0]
  rubatoDepth: number;                 // Rubato tempo fluctuation in [0.0, 1.0]
  solitonAccentGain: number;           // Soliton accent dynamic surge in [0.0, 2.0]
  webAudioSequence: WebAudioNoteDirective[];
}

/**
 * Calculates Circle of Fifths Torus distance between pitch classes c1, c2 in {0..11}.
 * Distance = min(|7(c1 - c2) mod 12|, 12 - |7(c1 - c2) mod 12|) / 6.0 in [0, 1].
 */
export function circleOfFifthsDistance(c1: number, c2: number): number {
  const step = Math.abs((7 * (c1 - c2)) % 12);
  const circ = Math.min(step, 12 - step);
  return circ / 6.0;
}

/**
 * Plomp-Levelt interval dissonance approximation between two semitone pitches.
 */
export function intervalDissonance(p1: number, p2: number): number {
  const interval = Math.abs(p1 - p2) % 12;
  // Dissonance curve: 1 (minor 2nd) and 6 (tritone) and 11 (major 7th) highest
  switch (interval) {
    case 0:  return 0.0; // Unison
    case 1:  return 1.0; // Minor 2nd (harsh)
    case 2:  return 0.4; // Major 2nd
    case 3:  return 0.2; // Minor 3rd
    case 4:  return 0.1; // Major 3rd
    case 5:  return 0.15;// Perfect 4th
    case 6:  return 0.85;// Tritone
    case 7:  return 0.05;// Perfect 5th
    case 8:  return 0.25;// Minor 6th
    case 9:  return 0.2; // Major 6th
    case 10: return 0.5; // Minor 7th
    case 11: return 0.95;// Major 7th
    default: return 0.3;
  }
}

/**
 * Extracts musical features from a note sequence.
 */
function extractMusicSummary(notes: MusicNoteEvent[]): MusicFeatures {
  if (notes.length === 0) {
    return {
      torusDistance: 0.2,
      dissonance: 0.1,
      syncopation: 0.2,
      restRatio: 0.5,
      bpm: 100,
    };
  }

  // Sort notes by startTime
  const sorted = [...notes].sort((a, b) => a.startTime - b.startTime);
  let totalDuration = 0.0;
  for (const n of sorted) {
    const endT = n.startTime + n.duration;
    if (endT > totalDuration) {
      totalDuration = endT;
    }
  }

  // 1. Circle of Fifths Torus Distance
  let torusDistSum = 0.0;
  let torusCount = 0;
  for (let i = 1; i < sorted.length; i++) {
    const pc1 = sorted[i - 1]!.pitch % 12;
    const pc2 = sorted[i]!.pitch % 12;
    torusDistSum += circleOfFifthsDistance(pc1, pc2);
    torusCount++;
  }
  const torusDistance = torusCount > 0 ? torusDistSum / torusCount : 0.2;

  // 2. Dissonance
  let dissSum = 0.0;
  let dissCount = 0;
  for (let i = 1; i < sorted.length; i++) {
    dissSum += intervalDissonance(sorted[i - 1]!.pitch, sorted[i]!.pitch);
    dissCount++;
  }
  const dissonance = dissCount > 0 ? dissSum / dissCount : 0.1;

  // 3. Rest Ratio
  let activeDuration = 0.0;
  for (const n of sorted) {
    activeDuration += n.duration;
  }
  const restRatio = totalDuration > 0 ? Math.max(0.0, Math.min(1.0, 1.0 - activeDuration / totalDuration)) : 0.2;

  // 4. Syncopation (Inter-Onset-Interval variation and off-beat energy)
  let ioiVar = 0.0;
  if (sorted.length > 2) {
    const iois: number[] = [];
    for (let i = 1; i < sorted.length; i++) {
      iois.push(sorted[i]!.startTime - sorted[i - 1]!.startTime);
    }
    const meanIoi = iois.reduce((a, b) => a + b, 0) / iois.length;
    const varIoi = iois.reduce((a, b) => a + (b - meanIoi) ** 2, 0) / iois.length;
    ioiVar = Math.min(1.0, Math.sqrt(varIoi) / (meanIoi || 1.0));
  }
  const syncopation = Math.min(1.0, 0.5 * ioiVar + 0.5 * torusDistance);

  // Estimate BPM from median positive IOI
  let bpm = 100.0;
  if (sorted.length > 1) {
    const positiveIois: number[] = [];
    for (let i = 1; i < sorted.length; i++) {
      const ioi = sorted[i]!.startTime - sorted[i - 1]!.startTime;
      if (ioi > 0.05) {
        positiveIois.push(ioi);
      }
    }
    if (positiveIois.length > 0) {
      positiveIois.sort((a, b) => a - b);
      const medianIoi = positiveIois[Math.floor(positiveIois.length / 2)]!;
      let estBpm = 60.0 / medianIoi;
      while (estBpm < 50.0 && estBpm > 0) estBpm *= 2.0;
      while (estBpm > 190.0) estBpm /= 2.0;
      bpm = Math.max(50.0, Math.min(190.0, Math.round(estBpm)));
    }
  }

  return {
    torusDistance,
    dissonance,
    syncopation,
    restRatio,
    bpm,
  };
}

interface MusicQTerms {
  q: number[];
  normBpm: number;
  rest: number;
  diss: number;
  torusDist: number;
}

function musicQAndTerms(features: MusicFeatures): MusicQTerms {
  const torusDist = clamp01(features.torusDistance);
  const diss = clamp01(features.dissonance);
  const synco = clamp01(features.syncopation);
  const rest = clamp01(features.restRatio);
  const bpm = features.bpm ?? 100.0;
  const normBpm = clamp01((bpm - 60.0) / 120.0);

  const q = new Array<number>(12).fill(0);
  q[0] = clamp01(0.6 * normBpm + 0.4 * synco);
  q[3] = clamp01(0.5 * synco + 0.5 * torusDist);
  q[7] = clamp01(0.7 * synco + 0.3 * normBpm);
  const consonance = 1.0 - diss;
  q[1] = clamp01(consonance);
  q[4] = clamp01(0.6 * consonance + 0.4 * (1.0 - torusDist));
  q[8] = clamp01(0.8 * consonance + 0.2 * (1.0 - synco));
  q[2] = clamp01(rest);
  q[5] = clamp01(0.7 * rest + 0.3 * (1.0 - normBpm));
  q[9] = clamp01(0.9 * rest + 0.1 * (1.0 - diss));
  q[6] = clamp01(diss);
  q[10] = clamp01(0.6 * diss + 0.4 * torusDist);
  q[11] = clamp01(0.7 * diss + 0.3 * synco);
  return { q, normBpm, rest, diss, torusDist };
}

/**
 * Extracts whole-piece music features and per-bar features (4 beats at the estimated tempo).
 * Bars containing no note onset are kept as silent windows so Void changes enter the momentum.
 */
export function extractMusicFeaturesFromNotes(notes: MusicNoteEvent[]): MusicFeatures {
  const summary = extractMusicSummary(notes);
  if (notes.length === 0) return summary;
  const barSec = (4 * 60.0) / (summary.bpm ?? 100.0);
  let endT = 0.0;
  for (const n of notes) endT = Math.max(endT, n.startTime + n.duration);
  const t0 = notes.reduce((m, n) => Math.min(m, n.startTime), Infinity);
  const nBars = Math.ceil((endT - t0) / barSec);
  if (nBars < 2) return summary;

  const bars: MusicFeatures[] = [];
  for (let b = 0; b < nBars; b++) {
    const lo = t0 + b * barSec;
    const hi = lo + barSec;
    const inBar = notes
      .filter((n) => n.startTime >= lo && n.startTime < hi)
      .map((n) => ({ ...n, startTime: n.startTime - lo }));
    if (inBar.length === 0) {
      bars.push({ torusDistance: summary.torusDistance, dissonance: 0.0, syncopation: 0.0, restRatio: 1.0, bpm: summary.bpm });
    } else {
      const f = extractMusicSummary(inBar);
      // Rest ratio is measured against the fixed bar length, not the span of the notes present.
      const active = inBar.reduce((a, n) => a + Math.min(n.duration, barSec), 0.0);
      bars.push({ ...f, restRatio: clamp01(1.0 - active / barSec), bpm: summary.bpm });
    }
  }
  return { ...summary, barFeatures: bars };
}

/**
 * Forward Music Functor F_music: Maps musical features to 24D USIR symplectic state z = (q, p).
 * With >= 2 barFeatures all 12 momentum coordinates are the RMS bar-to-bar change of the per-bar
 * elements; otherwise p is the Equilibrium Momentum Reconstruction (momentumFromSummary).
 */
export function musicFunctorForward(features: MusicFeatures): number[] {
  const t = musicQAndTerms(features);
  let p: number[];
  const bars = features.barFeatures ?? [];
  if (bars.length >= 2) {
    p = momentumFromWindowSeries(bars.map((bf) => musicQAndTerms(bf).q));
  } else {
    p = new Array<number>(12).fill(0);
    p[0] = t.normBpm * 1.5;
    p[2] = t.rest * 0.9;
    p[6] = t.diss * 1.6;
    p[11] = t.diss * 1.8 + t.torusDist;
    p = momentumFromSummary(t.q, p);
  }
  return [...t.q, ...p];
}

const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
// Low-discrepancy (golden-ratio family) sequences: deterministic and identical to the Python port.
const PHI = 0.6180339887498949;
const PLASTIC = 0.7548776662466927;
const PLASTIC2 = 0.5698402909980532;
const SILVER = 0.3819660112501051;
const frac = (x: number): number => x - Math.floor(x);

/**
 * 24D-driven note sequence on an eighth-note grid. Flux sets onset density, Void removes slots
 * (real rests), Lattice damps leaps (voice-leading), Shock drives leaps / circle-of-fifths tonic
 * motion / tritone resolution, syncopation displaces onsets by a quarter beat.
 */
export function synthesizeMusicSequence(
  q: number[],
  p: number[],
  bpm: number,
  harmonicTension: number,
  restInterval: number,
  solitonAccentGain: number,
  nSlots = 16
): WebAudioNoteDirective[] {
  const beatSec = 60.0 / bpm;
  const slotSec = beatSec / 2.0;
  const flux = 0.5 * q[0]! + 0.3 * q[3]! + 0.2 * q[7]!;
  const lattice = (q[1]! + q[4]! + q[8]!) / 3.0;
  const density = Math.max(0.1, Math.min(1.0, 0.35 + 0.6 * flux - 0.7 * restInterval));
  const leapProb = Math.max(
    0.0,
    Math.min(0.95, 0.1 + 0.8 * harmonicTension * (1.0 - 0.5 * lattice) + 0.3 * Math.tanh(p[11]!))
  );
  const leapSize = 2 + Math.floor(harmonicTension * 4.0);
  const fifthsSteps = Math.floor(q[10]! * 3.0);
  const syncoProb = Math.max(0.0, Math.min(0.9, 0.8 * q[7]! + 0.2 * Math.tanh(p[7]!)));
  const gainBase = Math.max(0.1, Math.min(1.0, 0.4 + 0.3 * solitonAccentGain));

  let onsets: number[] = [];
  for (let sl = 0; sl < nSlots; sl++) {
    if (frac(sl * PHI + 0.5) < density) onsets.push(sl);
  }
  if (onsets.length === 0) onsets = [0];

  const seq: WebAudioNoteDirective[] = [];
  let degree = 0;
  for (let k = 0; k < onsets.length; k++) {
    const slot = onsets[k]!;
    const phase = Math.floor(k / 4);
    let tonic = 60 + ((7 * fifthsSteps * phase) % 12);
    if (tonic > 66) tonic -= 12;
    const direction = frac(k * PLASTIC2 + 0.1) < 0.5 ? 1 : -1;
    if (frac(k * PLASTIC + 0.2) < leapProb) {
      degree += direction * leapSize;
    } else {
      degree += direction;
    }
    degree = Math.max(-7, Math.min(14, degree));
    const octave = Math.floor(degree / 7);
    const idx = degree - octave * 7;
    let pitch = tonic + 12 * octave + MAJOR_SCALE[idx]!;
    if (harmonicTension > 0.6 && k % 4 === 2) {
      pitch = tonic + 6;
    } else if (harmonicTension > 0.6 && k % 4 === 3) {
      pitch = tonic + 7;
    }
    pitch = Math.max(36, Math.min(96, pitch));

    const onsetOffset = frac(k * SILVER + 0.3) < syncoProb ? beatSec / 4.0 : 0.0;
    const duration = slotSec * (1.0 - restInterval * 0.5);
    const gain = Math.max(0.1, Math.min(1.0, gainBase + (k % 4 === 0 ? 0.2 * solitonAccentGain : 0.0)));
    seq.push({ note: pitch, time: slot * slotSec + onsetOffset, duration, gain });
  }
  return seq;
}

/**
 * Inverse Music Functor F_music^{-1}: Maps 24D state to WebAudio / MIDI sequence controls.
 */
export function musicFunctorInverse(z24d: number[] | Float32Array | Float64Array): MusicGenerativeParams {
  if (z24d.length < 24) {
    throw new Error(`Expected 24-dimensional USIR state vector, got ${z24d.length}`);
  }
  const z = Array.from(z24d);
  const q = z.slice(0, 12);
  const p = z.slice(12, 24);

  // BPM: Driven by Flux (q0, q3, q7) and momentum p0
  const bpm = Math.max(
    50.0,
    Math.min(
      190.0,
      60.0 +
        100.0 * (0.5 * (q[0] || 0) + 0.3 * (q[3] || 0) + 0.2 * (q[7] || 0)) +
        20.0 * Math.tanh(p[0] || 0)
    )
  );

  // Harmonic Tension: Driven by Shock (q6, q10, q11) and dissonance
  const harmonicTension = Math.max(
    0.0,
    Math.min(
      1.0,
      0.4 * (q[6] || 0) + 0.3 * (q[10] || 0) + 0.3 * (q[11] || 0) - 0.2 * (q[1] || 0)
    )
  );

  // Rest Interval: Driven by Void (q2, q5, q9)
  const restInterval = Math.max(
    0.0,
    Math.min(
      1.0,
      0.3 * (q[2] || 0) + 0.3 * (q[5] || 0) + 0.4 * (q[9] || 0)
    )
  );

  // Reverb Decay (seconds): Driven by Void depth (obsidian stillness)
  const reverbDecay = Math.max(
    0.4,
    Math.min(6.0, 0.5 + 4.5 * (q[9] || 0) + 1.5 * (q[2] || 0))
  );

  const rubatoDepth = Math.max(
    0.0,
    Math.min(1.0, 0.1 + 0.5 * (q[7] || 0) + 0.2 * Math.tanh(p[7] || 0))
  );

  const solitonAccentGain = Math.max(
    0.0,
    Math.min(
      2.0,
      0.6 * (q[11] || 0) + 0.4 * (q[6] || 0) + 0.3 * Math.tanh(p[11] || 0)
    )
  );

  const webAudioSequence = synthesizeMusicSequence(
    q.map((v) => v || 0),
    p.map((v) => v || 0),
    bpm,
    harmonicTension,
    restInterval,
    solitonAccentGain
  );

  return {
    bpm,
    harmonicTension,
    restInterval,
    reverbDecay,
    rubatoDepth,
    solitonAccentGain,
    webAudioSequence,
  };
}
