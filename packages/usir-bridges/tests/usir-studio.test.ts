import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vm from 'node:vm';

describe('Stage 4: USIR Studio Interactive Web GUI & Embedded Symplectic Engine', () => {
  const htmlPath = path.resolve(__dirname, '../public/usir-studio.html');
  const html = fs.readFileSync(htmlPath, 'utf-8');

  it('enforces Docked Atelier UI Constitution (Rule 5) and renders all 3 panes, 5 presets, SVGs, and 12-element spectrum', () => {
    expect(html).toContain('id="usir-docked-drawer"');
    expect(html).toContain('data-placement="right-detail-drawer"');
    expect(html).toContain('data-allow-popup-modal="false"');
    expect(html).toContain('width: 25%');

    for (const presetId of ['VOID-03', 'CRIT-05', 'CRYS-01', 'DEAD-01', 'HYBRID-5P']) {
      expect(html).toContain(`data-preset-id="${presetId}"`);
    }

    expect(html).toContain('id="usir-phase-map"');
    expect(html).toContain('id="usir-hsmm-oscilloscope"');
    expect(html).toContain('id="usir-sinkhole-alerts"');

    for (let i = 1; i <= 12; i++) {
      const code = `Sg-${String(i).padStart(2, '0')}`;
      expect(html).toContain(`data-element="${code}"`);
    }

    for (const sliderId of ['slider-flux', 'slider-lattice', 'slider-void', 'slider-shock']) {
      expect(html).toContain(`id="${sliderId}"`);
    }
  });

  it('executes embedded USIRStudio engine across all 5 presets and pinpoints Deadzone Sinkholes', () => {
    const scriptMatches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
    expect(scriptMatches.length).toBeGreaterThanOrEqual(2);

    const elements = new Map<string, { textContent: string; innerHTML: string; value: string }>();
    const context: Record<string, unknown> = {
      TextEncoder,
      TextDecoder,
      Uint8Array,
      Float32Array,
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
            elements.set(id, { textContent: '', innerHTML: '', value: '' });
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

    const studio = context.USIRStudio as {
      loadPreset: (id: string) => {
        classification: { phase: string };
        microGate: { passed: boolean };
        sinkholes: { hasSinkholes: boolean; sinkholes: Array<{ paragraphIndex: number }> };
        modulatedQ: number[];
        directives: { compiledPrompt: string };
      };
      setGains: (g: Record<string, number>) => {
        modulatedQ: number[];
        directives: { compiledPrompt: string };
      };
    };

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
    expect(hybrid.sinkholes.sinkholes.some((s) => s.paragraphIndex === 2)).toBe(true);
    expect(elements.get('usir-sinkhole-alerts')?.innerHTML).toContain('data-paragraph-index="2"');

    // 6. Equalizer Slider Modulation
    const beforeVoidQ = hybrid.modulatedQ[2]!;
    const modulated = studio.setGains({ alpha_void: 1.6, alpha_shock: 0.4 });
    expect(modulated.modulatedQ[2]!).toBeGreaterThanOrEqual(beforeVoidQ);
    expect(modulated.directives.compiledPrompt).toContain('USIR Phase Transition Steering Directive');
  });
});
