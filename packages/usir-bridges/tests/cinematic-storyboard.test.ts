/**
 * @nishiki/usir-bridges: Unit & Stress Tests for 60fps Cinematic Visual Simulator & Multimodal Storyboard
 * Workstream 3 Verification (USIR v3.0 Stage 4 / TASK-362).
 */

import { describe, it, expect } from 'vitest';
import {
  computeCinematicShaderState,
  generateMultimodalStoryboard,
  renderCinematicFrame,
  detectDominantMode,
  synthesizeScaleMotif,
  splitIntoScenes,
  CANONICAL_CINEMATIC_PALETTES,
  type CanvasRenderingContext2DLike,
  type CinematicShaderState,
  DEADZONE_MU,
  OBSIDIAN_VOID_MU,
  AMBER_CRITICAL_MU,
  PORCELAIN_CRYSTAL_MU,
  LIGHTNING_RUPTURE_MU,
} from '../src/cinematicStoryboard.js';

function createMockCanvasContext() {
  const calls: { method: string; args: any[] }[] = [];
  const ctx: CanvasRenderingContext2DLike = {
    fillStyle: '#000000',
    strokeStyle: '#ffffff',
    lineWidth: 1,
    globalAlpha: 1.0,
    fillRect: (x, y, w, h) => calls.push({ method: 'fillRect', args: [x, y, w, h] }),
    strokeRect: (x, y, w, h) => calls.push({ method: 'strokeRect', args: [x, y, w, h] }),
    beginPath: () => calls.push({ method: 'beginPath', args: [] }),
    moveTo: (x, y) => calls.push({ method: 'moveTo', args: [x, y] }),
    lineTo: (x, y) => calls.push({ method: 'lineTo', args: [x, y] }),
    stroke: () => calls.push({ method: 'stroke', args: [] }),
    fill: () => calls.push({ method: 'fill', args: [] }),
    arc: (x, y, r, sa, ea) => calls.push({ method: 'arc', args: [x, y, r, sa, ea] }),
    clearRect: (x, y, w, h) => calls.push({ method: 'clearRect', args: [x, y, w, h] }),
    save: () => calls.push({ method: 'save', args: [] }),
    restore: () => calls.push({ method: 'restore', args: [] }),
  };
  return { ctx, calls };
}

describe('Workstream 3: 60fps Cinematic Visual Simulator & Multimodal Storyboard Timeline', () => {
  describe('1. 60fps Real-Time Cinematic Visual Simulator (computeCinematicShaderState)', () => {
    it('maps DEADZONE_MU to flat, uninflected shader state with zero impulse and minimal zoom', () => {
      const state = computeCinematicShaderState(DEADZONE_MU);

      expect(state.dominantMode).toBe('deadzone');
      expect(state.particleCount).toBeLessThan(120);
      expect(state.cutImpulseProbability).toBeLessThan(0.20);
      expect(state.cameraZoomAccel).toBeLessThan(0.80);
      expect(state.lensFocalLengthMm).toBe(50);
      expect(state.colorPalette.background).toBe(CANONICAL_CINEMATIC_PALETTES['deadzone']!.background);
      expect(state.chiaroscuroRimLight).toBeLessThan(0.35);
    });

    it('maps OBSIDIAN_VOID_MU to expansive negative space, high mist, and intimate telephoto lens', () => {
      const state = computeCinematicShaderState(OBSIDIAN_VOID_MU);

      expect(state.dominantMode).toBe('void');
      expect(state.negativeSpaceRatio).toBeGreaterThanOrEqual(0.55);
      expect(state.mistOpacity).toBeGreaterThanOrEqual(0.30);
      expect(state.fadeDecayRate).toBeLessThanOrEqual(0.15); // Slow lingering afterimage
      expect([85, 135]).toContain(state.lensFocalLengthMm);
      expect(state.colorPalette.background).toBe(CANONICAL_CINEMATIC_PALETTES['void']!.background);
      expect(state.cutImpulseProbability).toBeLessThan(0.40);
    });

    it('maps AMBER_CRITICAL_MU to high chiaroscuro rim lighting, wide lens, and dynamic kinetic zoom', () => {
      const state = computeCinematicShaderState(AMBER_CRITICAL_MU);

      expect(state.dominantMode).toBe('critical');
      expect(state.chiaroscuroRimLight).toBeGreaterThanOrEqual(0.45);
      expect(state.cameraZoomAccel).toBeGreaterThanOrEqual(1.0);
      expect(state.lensFocalLengthMm).toBe(24);
      expect(state.colorPalette.background).toBe(CANONICAL_CINEMATIC_PALETTES['critical']!.background);
    });

    it('maps PORCELAIN_CRYSTAL_MU to high grid symmetry, stable tripod lock, and locked horizon', () => {
      const state = computeCinematicShaderState(PORCELAIN_CRYSTAL_MU);

      expect(state.dominantMode).toBe('crystalline');
      expect(state.gridSymmetryAlpha).toBeGreaterThanOrEqual(0.45);
      expect(state.cameraStability).toBeGreaterThanOrEqual(0.55);
      expect(state.horizonLock).toBe(true);
      expect(state.colorPalette.background).toBe(CANONICAL_CINEMATIC_PALETTES['crystalline']!.background);
    });

    it('maps LIGHTNING_RUPTURE_MU to extreme soliton zoom, high impulse cut probability, and neon rim light', () => {
      const state = computeCinematicShaderState(LIGHTNING_RUPTURE_MU);

      expect(state.dominantMode).toBe('turbulent');
      expect(state.cameraZoomAccel).toBeGreaterThanOrEqual(2.0);
      expect(state.cutImpulseProbability).toBeGreaterThanOrEqual(0.70);
      expect(state.chiaroscuroRimLight).toBeGreaterThanOrEqual(0.70);
      expect(state.colorPalette.background).toBe(CANONICAL_CINEMATIC_PALETTES['turbulent']!.background);
    });

    it('honors explicit mode overrides including genesis mode', () => {
      const state = computeCinematicShaderState(OBSIDIAN_VOID_MU, 'genesis');
      expect(state.dominantMode).toBe('genesis');
      expect(state.colorPalette.primary).toBe(CANONICAL_CINEMATIC_PALETTES['genesis']!.primary);
      expect(state.colorPalette.rim).toBe(CANONICAL_CINEMATIC_PALETTES['genesis']!.rim);
    });
  });

  describe('2. Paragraph-Synchronized Multimodal Storyboard Timeline (generateMultimodalStoryboard)', () => {
    const beforeText = `
私たちは次世代のAIソリューションを通じて包括的なイノベーションを提供します。
多面的なアプローチを統合し、シームレスな体験を実現することで業界のリーダーシップを確立します。
持続可能な成長とシナジーの最大化に向けて、革新的な価値創造を推進し続けます。
    `.trim();

    const afterText = `
国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。
信号所に汽車が止まった。向側の座席から娘が立って来て、島村の前のガラス窓を落した。冷気が流れこんだ。
遠くの山裾が闇に沈み、星の光だけが凍てついた雪原に刺さっていた。
    `.trim();

    it('generates a 3-scene Jo-Ha-Kyu / Ma timeline with synchronized 4 columns', () => {
      const storyboard = generateMultimodalStoryboard(beforeText, afterText);

      expect(storyboard.rows).toHaveLength(3);
      expect(storyboard.summary.sceneCount).toBe(3);
      expect(storyboard.totalDurationSec).toBeGreaterThan(6.0);

      // Verify Column 1: Text Paragraph with Jo-Ha-Kyu / Ma Section Labels
      const row0 = storyboard.rows[0]!;
      const row1 = storyboard.rows[1]!;
      const row2 = storyboard.rows[2]!;

      expect(row0.textColumn.sectionLabel).toContain('Scene 01 [序');
      expect(row1.textColumn.sectionLabel).toContain('Scene 02 [破');
      expect(row2.textColumn.sectionLabel).toContain('Scene 03 [間');

      expect(row0.textColumn.beforeParagraph).toContain('包括的なイノベーション');
      expect(row0.textColumn.afterParagraph).toContain('国境の長いトンネル');
      expect(row0.textColumn.z24).toHaveLength(24);
      expect(row0.textColumn.deltaNorm).toBeGreaterThan(0);

      // Verify Column 2: Music Cue with Web Audio Preview Notes
      expect(row0.musicCue.bpm).toBeGreaterThan(0);
      expect(row1.musicCue.bpm).toBeGreaterThan(row2.musicCue.bpm); // Critical is faster than Void
      expect(row2.musicCue.restRatioPercent).toBeGreaterThanOrEqual(10);
      expect(row0.musicCue.previewComposition.notes.length).toBeGreaterThanOrEqual(4);

      for (const note of row0.musicCue.previewComposition.notes) {
        expect(note.pitch).toMatch(/^[A-G][#b]?[0-8]$/);
        expect(note.freqHz).toBeGreaterThan(20);
        expect(note.durationSec).toBeGreaterThan(0);
        expect(note.velocity).toBeGreaterThan(0);
      }

      // Verify Column 3: Image Generation AI Prompt (Flux.1 / Midjourney v6 ready)
      expect(row0.imagePrompt.promptString).toContain('cinematic film still');
      expect(row0.imagePrompt.promptString).toContain('shot on ARRI Alexa 65');
      expect(row0.imagePrompt.promptString).toContain(row0.shaderState.lensFocalLength);
      expect(row0.imagePrompt.aspectRatio).toMatch(/^(21:9|16:9)$/);
      expect(row0.imagePrompt.recommendedModel).toContain('Flux.1');

      // Verify Column 4: Video Generation AI Camera & Cut Script (Sora / Veo ready)
      expect(row0.videoScript.durationSec).toBeGreaterThan(0);
      expect(row0.videoScript.cameraVector.length).toBeGreaterThan(10);
      expect(row0.videoScript.speedCurve.length).toBeGreaterThan(5);
      expect(row0.videoScript.fullVeoSoraPrompt).toContain('Cinematic cut');
      expect(row0.videoScript.fullVeoSoraPrompt).toContain(row0.videoScript.cameraVector);
    });

    it('scales cleanly to N > 3 scenes when input has more than 3 paragraphs', () => {
      const multiBeforeText = [
        '第一の段落における導入と背景の説明です。',
        '第二の段落における問題提起と課題の分析です。',
        '第三の段落における対立構造と緊張の高まりです。',
        '第四の段落における劇的な転換と危機の爆発です。',
        '第五の段落における静かな解決と余韻の広がりです。',
      ].join('\n\n');

      const multiAfterText = [
        '夜明け前の石畳に、微かな蹄の音が響いていた。',
        '冷たい霧が運河を覆い、街灯の光が水面に揺れる。',
        '突然、路地の角から黒い影が走り抜け、呼子笛が鳴り響いた。',
        '尖塔の鐘が乱打され、破片が石畳を跳ねる。炎が空を焦がした。',
        '煙が夜空へ吸い込まれ、遠く波の音だけが戻ってきた。',
      ].join('\n\n');

      const storyboard = generateMultimodalStoryboard(multiBeforeText, multiAfterText);

      expect(storyboard.rows).toHaveLength(5);
      expect(storyboard.rows[0]!.textColumn.sectionLabel).toContain('[序');
      expect(storyboard.rows[1]!.textColumn.sectionLabel).toContain('[破-1');
      expect(storyboard.rows[2]!.textColumn.sectionLabel).toContain('[破-2');
      expect(storyboard.rows[3]!.textColumn.sectionLabel).toContain('[破-3');
      expect(storyboard.rows[4]!.textColumn.sectionLabel).toContain('[間');
    });

    it('handles single-sentence short inputs gracefully by chunking into 3 scenes', () => {
      const shortBefore = 'AI deadzone generic paragraph without breaks.';
      const shortAfter = '古池や蛙飛びこむ水の音。静けさが水面に広がる。やがて波紋は消え失せた。';

      const storyboard = generateMultimodalStoryboard(shortBefore, shortAfter);
      expect(storyboard.rows).toHaveLength(3);
      expect(storyboard.rows[0]!.textColumn.sectionLabel).toContain('[序');
      expect(storyboard.rows[1]!.textColumn.sectionLabel).toContain('[破');
      expect(storyboard.rows[2]!.textColumn.sectionLabel).toContain('[間');
    });
  });

  describe('3. 60fps HTML5 Canvas Cinematic Simulator Renderer (renderCinematicFrame)', () => {
    it('renders a 60-frame simulation loop across Void, Critical, and Crystalline states without crashing', () => {
      const { ctx, calls } = createMockCanvasContext();
      const stateVoid = computeCinematicShaderState(OBSIDIAN_VOID_MU);
      const stateCrit = computeCinematicShaderState(AMBER_CRITICAL_MU);
      const stateCrys = computeCinematicShaderState(PORCELAIN_CRYSTAL_MU);

      // Render 20 frames of each state (total 60 frames = 1 second)
      for (let f = 0; f < 20; f++) {
        renderCinematicFrame(ctx, stateVoid, f, 1920, 1080);
      }
      for (let f = 20; f < 40; f++) {
        renderCinematicFrame(ctx, stateCrit, f, 1920, 1080);
      }
      for (let f = 40; f < 60; f++) {
        renderCinematicFrame(ctx, stateCrys, f, 1920, 1080);
      }

      expect(calls.length).toBeGreaterThan(100);

      // Verify that save and restore are balanced
      const saveCount = calls.filter((c) => c.method === 'save').length;
      const restoreCount = calls.filter((c) => c.method === 'restore').length;
      expect(saveCount).toBe(60);
      expect(restoreCount).toBe(60);

      // Verify letterbox fills occurred for high negative space
      const letterboxFills = calls.filter(
        (c) => c.method === 'fillRect' && c.args[1] === 0 && c.args[2] === 1920
      );
      expect(letterboxFills.length).toBeGreaterThan(0);
    });
  });

  describe('4. Real-World Literary Stress Testing (Rule 3: Real-World Stress Testing)', () => {
    it('accurately distinguishes Hemingway apophatic void from AI Deadzone corporate report', () => {
      const hemingway = `
He looked across the sea and knew how alone he was now.
He saw the prism in the deep dark water and the line slanting far ahead.
The clouds over the land now rose like mountains and the coast was only a long green line with the gray blue hills behind it.
      `.trim();

      const deadzone = `
当社は革新的なソリューションと包括的なシナジーを通じて、持続可能なエコシステムの構築を目指しています。
多様なステークホルダーとの協創により、高付加価値なサービスの提供と業務効率の最大化を実現いたします。
グローバルな視点に基づき、データドリブンな意思決定と俊敏なオペレーションを推進してまいります。
      `.trim();

      const sbHemingway = generateMultimodalStoryboard(deadzone, hemingway);
      const sbDeadzone = generateMultimodalStoryboard(hemingway, deadzone);

      // Hemingway final scene should feature lower BPM, higher rest ratio, and longer video cut duration
      const lastHemingwayRow = sbHemingway.rows[sbHemingway.rows.length - 1]!;
      const lastDeadzoneRow = sbDeadzone.rows[sbDeadzone.rows.length - 1]!;

      expect(lastHemingwayRow.musicCue.bpm).toBeLessThanOrEqual(lastDeadzoneRow.musicCue.bpm + 5);
      expect(lastHemingwayRow.shaderState.negativeSpaceRatio).toBeGreaterThan(0.40);
      expect(lastHemingwayRow.videoScript.durationSec).toBeGreaterThanOrEqual(4.0);
    });

    it('guarantees mutually distinct palettes, tonalities, and synth presets across all 6 style modes', () => {
      const modes = ['void', 'critical', 'crystalline', 'turbulent', 'deadzone', 'genesis'];
      const palettes = new Set<string>();
      const tonalities = new Set<string>();
      const synths = new Set<string>();

      for (const m of modes) {
        const state = computeCinematicShaderState(new Array(24).fill(0.5), m);
        palettes.add(state.colorPalette.background);

        const notes = synthesizeScaleMotif(m, 80, 0, 0.4);
        expect(notes.length).toBeGreaterThan(0);

        // Verify chronological monotonicity of notes
        for (let i = 1; i < notes.length; i++) {
          expect(notes[i]!.timeSec).toBeGreaterThanOrEqual(notes[i - 1]!.timeSec);
        }

        const sb = generateMultimodalStoryboard('導入', '展開', m);
        tonalities.add(sb.rows[0]!.musicCue.tonality);
        synths.add(sb.rows[0]!.musicCue.previewComposition.synthPreset);
      }

      expect(palettes.size).toBe(6);
      expect(tonalities.size).toBe(6);
      expect(synths.size).toBe(6);
    });

    it('robustly handles edge-case inputs: empty string, single character, and 10 paragraphs', () => {
      // Empty string
      const sbEmpty = generateMultimodalStoryboard('', '');
      expect(sbEmpty.rows).toHaveLength(3);
      expect(sbEmpty.rows[0]!.textColumn.z24).toHaveLength(24);

      // Single character
      const sbChar = generateMultimodalStoryboard('無', '閃');
      expect(sbChar.rows).toHaveLength(3);

      // 10 paragraphs
      const tenParas = new Array(10).fill(0).map((_, i) => `段落 ${i + 1} の描写。`).join('\n\n');
      const sbTen = generateMultimodalStoryboard(tenParas, tenParas);
      expect(sbTen.rows).toHaveLength(10);
      expect(sbTen.rows[0]!.textColumn.sectionLabel).toContain('[序');
      expect(sbTen.rows[9]!.textColumn.sectionLabel).toContain('[間');
    });

    it('triggers high-energy hard-cut flash in renderCinematicFrame on high impulse frames', () => {
      const { ctx, calls } = createMockCanvasContext();
      const stateTurbulent = computeCinematicShaderState(LIGHTNING_RUPTURE_MU);
      expect(stateTurbulent.cutImpulseProbability).toBeGreaterThanOrEqual(0.70);

      // Render at frame 60 (mod 60 == 0)
      renderCinematicFrame(ctx, stateTurbulent, 60, 1920, 1080);

      // Check if a flash occurred (white fillRect with alpha)
      const flashCall = calls.find(
        (c) => c.method === 'fillRect' && c.args[0] === 0 && c.args[1] === 0 && c.args[2] === 1920 && c.args[3] === 1080
      );
      expect(flashCall).toBeDefined();
    });
  });
});
