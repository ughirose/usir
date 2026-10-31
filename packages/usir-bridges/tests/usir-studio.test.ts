import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vm from 'node:vm';

describe('Stage 4 & v3.0: USIR Studio Interactive Web GUI, Chemistry & Ecosystem Functors', () => {
  const htmlPath = path.resolve(__dirname, '../public/usir-studio.html');
  const html = fs.readFileSync(htmlPath, 'utf-8');

  it('enforces Docked Atelier UI Constitution (Rule 5) and renders all 3 panes, presets, SVGs, and 12-element spectrum', () => {
    expect(html).toContain('id="usir-docked-drawer"');
    expect(html).toContain('data-placement="right-detail-drawer"');
    expect(html).toContain('data-allow-popup-modal="false"');
    expect(html).toContain('width: 25%');

    // 5 Japanese + 1 English Presets
    for (const presetId of ['VOID-03', 'CRIT-05', 'CRYS-01', 'DEAD-01', 'HYBRID-5P', 'EN-HEMINGWAY-VOID']) {
      expect(html).toContain(`data-preset-id="${presetId}"`);
    }

    expect(html).toContain('id="usir-phase-map"');
    expect(html).toContain('id="usir-hsmm-oscilloscope"');
    expect(html).toContain('id="usir-sinkhole-alerts"');

    // Spectrum Bars
    for (let i = 1; i <= 12; i++) {
      const code = `Sg-${String(i).padStart(2, '0')}`;
      expect(html).toContain(`data-element="${code}"`);
    }

    // Sliders
    for (const sliderId of ['slider-flux', 'slider-lattice', 'slider-void', 'slider-shock']) {
      expect(html).toContain(`id="${sliderId}"`);
    }

    // v3.0 Panels & Markers
    expect(html).toContain('id="usir-lang-badge"');
    expect(html).toContain('id="usir-chemistry-panel"');
    expect(html).toContain('id="mol-yugen"');
    expect(html).toContain('id="mol-wabi"');
    expect(html).toContain('id="mol-qiyun"');
    expect(html).toContain('id="mol-iki"');
    expect(html).toContain('id="btn-usir-genesis"');
    expect(html).toContain('id="usir-genesis-panel"');
    expect(html).toContain('id="genesis-star-marker"');
    expect(html).toContain('id="usir-genesis-directive"');
    expect(html).toContain('id="usir-cross-modal-panel"');
    expect(html).toContain('id="cm-music-summary"');
    expect(html).toContain('id="cm-visual-summary"');
    expect(html).toContain('id="cm-live-ui-card"');
  });

  function setupStudioVM() {
    const scriptMatches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
    expect(scriptMatches.length).toBeGreaterThanOrEqual(2);

    const elements = new Map<string, any>();
    const context: Record<string, unknown> = {
      TextEncoder,
      TextDecoder,
      Uint8Array,
      Float32Array,
      Float64Array,
      DataView,
      ArrayBuffer,
      Math,
      JSON,
      Map,
      Set,
      console,
      document: {
        getElementById: (id: string) => {
          if (!elements.has(id)) {
            const styles: Record<string, string> = {};
            const attrs: Record<string, string> = {};
            elements.set(id, {
              textContent: '',
              innerHTML: '',
              value: '',
              points: '',
              style: {
                ...styles,
                cssText: '',
                setProperty: (k: string, v: string) => { styles[k] = v; },
                getPropertyValue: (k: string) => styles[k] || '',
              },
              setAttribute: (k: string, v: string) => { attrs[k] = v; },
              getAttribute: (k: string) => attrs[k] || '',
            });
          }
          return elements.get(id)!;
        },
      },
    };
    context.window = context;
    context.globalThis = context;
    vm.createContext(context);

    for (const match of scriptMatches) {
      vm.runInContext(match[1]!, context);
    }

    return {
      studio: context.USIRStudio as any,
      elements,
    };
  }

  it('executes embedded USIRStudio engine across all 5 presets and pinpoints Deadzone Sinkholes', () => {
    const { studio, elements } = setupStudioVM();
    expect(studio).toBeDefined();

    // 1. VOID-03 -> III_Void & MicroGate PASS
    const void03 = studio.loadPreset('VOID-03');
    expect(void03.classification.phase).toBe('III_Void');
    expect(void03.microGate.passed).toBe(true);
    expect(elements.get('usir-phase-badge')?.textContent).toBe('III_Void');

    // 2. CRIT-05 -> II_Critical & MicroGate PASS
    const crit05 = studio.loadPreset('CRIT-05');
    expect(crit05.classification.phase).toBe('II_Critical');
    expect(crit05.microGate.passed).toBe(true);

    // 3. CRYS-01 -> I_Crystalline
    const crys01 = studio.loadPreset('CRYS-01');
    expect(crys01.classification.phase).toBe('I_Crystalline');

    // 4. DEAD-01 -> V_Deadzone & MicroGate BLOCK
    const dead01 = studio.loadPreset('DEAD-01');
    expect(dead01.classification.phase).toBe('V_Deadzone');
    expect(dead01.microGate.passed).toBe(false);

    // 5. HYBRID-5P -> Pinpoints Paragraph Index 2 as Deadzone Sinkhole
    const hybrid = studio.loadPreset('HYBRID-5P');
    expect(hybrid.sinkholes.hasSinkholes).toBe(true);
    expect(hybrid.sinkholes.sinkholes.some((s: any) => s.paragraphIndex === 2)).toBe(true);
    expect(elements.get('usir-sinkhole-alerts')?.innerHTML).toContain('data-paragraph-index="2"');

    // 6. Equalizer Slider Modulation
    const beforeVoidQ = hybrid.modulatedQ[2]!;
    const modulated = studio.setGains({ alpha_void: 1.6, alpha_shock: 0.4 });
    expect(modulated.modulatedQ[2]!).toBeGreaterThanOrEqual(beforeVoidQ);
    expect(modulated.directives.compiledPrompt).toContain('USIR Phase Transition Steering Directive');
  });

  it('computes and updates the 4 aesthetic molecules (yugen, wabi, qiyun, iki) across presets', () => {
    const { studio, elements } = setupStudioVM();

    // 1. VOID-03: Soseki Kusamakura has prominent Yugen and Wabi
    const void03 = studio.loadPreset('VOID-03');
    expect(void03.molecules).toBeDefined();
    expect(void03.molecules.yugen).toBeGreaterThan(0.60);
    expect(void03.molecules.wabi).toBeGreaterThan(0.45);
    expect(elements.get('mol-yugen')?.textContent).toContain('幽玄 (Yugen)');
    expect(elements.get('mol-wabi')?.textContent).toContain('侘び (Wabi)');

    // 2. CRIT-05: Kajii Motojiro has active momentum circulation and QiYun / Iki
    const crit05 = studio.loadPreset('CRIT-05');
    expect(crit05.molecules).toBeDefined();
    expect(crit05.molecules.qiyun).toBeGreaterThan(0.35);
    expect(crit05.molecules.iki).toBeGreaterThan(0.35);
    expect(elements.get('mol-qiyun')?.textContent).toContain('気韻生動 (QiYun)');
    expect(elements.get('mol-iki')?.textContent).toContain('粋 (Iki)');

    // Coupling energy
    expect(void03.molecules.synestheticCouplingEnergy).toBeGreaterThan(0.5);
    expect(crit05.molecules.synestheticCouplingEnergy).toBeGreaterThan(0.5);
  });

  it('triggers usir-genesis to discover uncharted style coordinate within chi2_{24, 0.95} ellipsoid and plots star marker', () => {
    const { studio, elements } = setupStudioVM();

    expect(typeof studio.triggerGenesis).toBe('function');
    const genesis = studio.triggerGenesis();

    // 1. Rigorous statistical ellipsoid constraint
    expect(genesis).toBeDefined();
    expect(genesis.insideTrustRegion).toBe(true);
    expect(genesis.mahalanobisD2).toBeLessThanOrEqual(36.415028 + 1e-4);
    expect(genesis.minRiemannianDistance).toBeGreaterThan(1.0);
    expect(genesis.zGenesis.length).toBe(24);

    // 2. DOM updates
    const directiveText = elements.get('usir-genesis-directive')?.textContent || '';
    expect(directiveText).toContain('[USIR-GENESIS SYNTHESIZED HAMILTONIAN DIRECTIVE]');
    expect(directiveText).toContain('Uncharted Voronoi Coordinate');
    expect(directiveText).toContain('Molecular Recipe:');
    expect(directiveText).toContain('Hamiltonian Tuning:');

    // 3. Gold star marker coordinates plotted
    const starEl = elements.get('genesis-star-marker');
    expect(starEl).toBeDefined();
    const pointsStr = starEl.points || starEl.getAttribute('points') || '';
    expect(pointsStr.split(' ').length).toBe(10);
  });

  it('transpiles 24D symplectic state to Music, Diffusion visual, and live UI CSS variables', () => {
    const { studio, elements } = setupStudioVM();

    const void03 = studio.loadPreset('VOID-03');
    expect(void03.crossModal).toBeDefined();
    const { music, visual, uiCss } = void03.crossModal;

    // Void maps to high rest_ratio, negative_space_ratio, whitespace_multiplier
    expect(music.rest_ratio).toBeGreaterThan(0.35);
    expect(visual.negative_space_ratio).toBeGreaterThan(0.35);
    expect(uiCss.whitespace_multiplier).toBeGreaterThan(1.0);

    // Summary text rendered
    expect(elements.get('cm-music-summary')?.textContent).toContain('Rest/Ma:');
    expect(elements.get('cm-visual-summary')?.textContent).toContain('Negative Space:');

    // CSS variables applied to live UI card
    const card = elements.get('cm-live-ui-card');
    expect(card).toBeDefined();
    expect(card.innerHTML).toContain('Live Synesthetic UI Preview Card');
    const cssText = card.style.cssText || card.getAttribute('style') || '';
    expect(cssText).toContain('--usir-padding:');
    expect(cssText).toContain('--usir-radius:');
    expect(cssText).toContain('--usir-contrast:');

    // Critical preset has higher articulation & CFG
    const crit05 = studio.loadPreset('CRIT-05');
    expect(crit05.crossModal.music.soliton_accent_gain).toBeGreaterThan(0.3);
    expect(crit05.crossModal.visual.cfg_scale).toBeGreaterThan(4.0);

    // Direct helper method
    const liveMusic = studio.transpileCurrentState('music');
    expect(liveMusic.bpm_fluctuation).toBeGreaterThanOrEqual(0.0);
    const liveVisual = studio.transpileCurrentState('diffusion');
    expect(liveVisual.cfg_scale).toBeGreaterThanOrEqual(1.5);
    const liveUi = studio.transpileCurrentState('ui_css');
    expect(liveUi.contrast_ratio).toBeGreaterThanOrEqual(3.0);
  });

  it('automatically detects English input and seamlessly switches to English isomorphic driver', () => {
    const { studio, elements } = setupStudioVM();

    // 1. English Hemingway Preset
    const hemingway = studio.loadPreset('EN-HEMINGWAY-VOID');
    expect(hemingway.language).toBe('en');
    expect(hemingway.classification.phase).toBe('III_Void');
    expect(elements.get('usir-lang-badge')?.textContent).toBe('EN (English Isomorphic)');
    expect(hemingway.molecules.wabi).toBeGreaterThan(0.10);
    expect(hemingway.crossModal.music.rest_ratio).toBeGreaterThan(0.20);

    // 2. Switch back to Japanese
    const void03 = studio.loadPreset('VOID-03');
    expect(void03.language).toBe('ja');
    expect(elements.get('usir-lang-badge')?.textContent).toBe('JA (Japanese)');

    // 3. Arbitrary English text auto-detection in analyzeText
    const customEn = studio.analyzeText(
      'The river was green. Cold water ran over the white stones. Nick looked down through the dark pool and saw the trout. He did not move.'
    );
    expect(customEn.language).toBe('en');
    expect(customEn.classification.phase).toBe('III_Void');
    expect(elements.get('usir-lang-badge')?.textContent).toBe('EN (English Isomorphic)');

    // 4. Arbitrary Japanese text auto-detection in analyzeText
    const customJa = studio.analyzeText('山路を登りながら、こう考えた。智に働けば角が立つ。情に棹させば流される。');
    expect(customJa.language).toBe('ja');
    expect(elements.get('usir-lang-badge')?.textContent).toBe('JA (Japanese)');
  });
});
