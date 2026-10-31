import * as fs from 'node:fs';
import * as path from 'node:path';
import { buildSync } from '../../../node_modules/.pnpm/esbuild@0.21.5/node_modules/esbuild/lib/main.js';

const repoRoot = 'C:/dev/usir';
const fixturePath = path.join(repoRoot, 'packages/usir-core/tests/fixtures/python_reference_40.json');
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
const getText = (id) => fixture.find((d) => d.id === id).text;

const presets = {
  'VOID-03': { id: 'VOID-03', title: '夏目漱石：草枕 [VOID-03]', expectedPhase: 'III_Void', text: getText('VOID-03') },
  'CRIT-05': { id: 'CRIT-05', title: '梶井基次郎：桜の樹の下には [CRIT-05]', expectedPhase: 'II_Critical', text: getText('CRIT-05') },
  'CRYS-01': { id: 'CRYS-01', title: '民法条文 [CRYS-01]', expectedPhase: 'I_Crystalline', text: getText('CRYS-01') },
  'DEAD-01': { id: 'DEAD-01', title: 'AI模倣スロップ [DEAD-01]', expectedPhase: 'V_Deadzone', text: getText('DEAD-01') },
  'HYBRID-5P': {
    id: 'HYBRID-5P',
    title: '5段落ハイブリッド長文 [中だるみ混入デモ]',
    expectedPhase: 'III_Void',
    text: [getText('VOID-01'), getText('CRIT-01'), getText('DEAD-01'), getText('VOID-02'), getText('CRIT-03')].join('\n\n'),
  },
};

const bundleResult = buildSync({
  entryPoints: [path.join(repoRoot, 'packages/usir-core/dist/index.js')],
  bundle: true,
  format: 'iife',
  globalName: 'USIRCore',
  platform: 'browser',
  write: false,
});
const coreBundleJs = bundleResult.outputFiles[0].text;

const studioAppJs = `
(function(root) {
  const core = root.USIRCore;
  const PRESETS = ${JSON.stringify(presets)};
  const ELEMENT_META = [
    { idx: 0, code: 'Sg-01 Gr', group: 'Flux' }, { idx: 4, code: 'Sg-05 Cp', group: 'Flux' }, { idx: 8, code: 'Sg-09 Dy', group: 'Flux' },
    { idx: 1, code: 'Sg-02 Cr', group: 'Lattice' }, { idx: 5, code: 'Sg-06 Sy', group: 'Lattice' }, { idx: 9, code: 'Sg-10 At', group: 'Lattice' },
    { idx: 2, code: 'Sg-03 Pr', group: 'Void' }, { idx: 6, code: 'Sg-07 Cs', group: 'Void' }, { idx: 10, code: 'Sg-11 Ma', group: 'Void' },
    { idx: 3, code: 'Sg-04 Jt', group: 'Shock' }, { idx: 7, code: 'Sg-08 Lp', group: 'Shock' }, { idx: 11, code: 'Sg-12 Ru', group: 'Shock' }
  ];

  let state = {
    presetId: 'VOID-03',
    text: PRESETS['VOID-03'].text,
    targetPhase: 'III_Void',
    gains: { alpha_flux: 1.0, alpha_lattice: 1.0, alpha_void: 1.0, alpha_shock: 1.0 },
    lastAnalysis: null
  };

  function toHex(u8) {
    return Array.from(u8.slice(0, 32)).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function analyzeText(text, targetPhase = state.targetPhase, gains = state.gains) {
    const metrics = core.extractUsir24d(text);
    const classification = core.classifyPhase(metrics.z_24d);
    const microGate = core.runMicroConstraintGate(text, { targetPhase, registerMode: 'classical_literary' });
    const trajectoryWave = core.extractTrajectoryWave(text, { windowSize: 3, stride: 1 });
    const hsmm = core.decodeHsmmModes(trajectoryWave);
    const sinkholes = core.detectDeadzoneSinkholes(text);
    const directives = core.generateRewriteDirectives(text, targetPhase, metrics);
    const modulatedQ = core.applyEqualizer(metrics.q, gains);
    const tierS = core.encodeTierS(metrics);
    const rewrittenPreview = core.steerPhaseRewrite(text, targetPhase);
    const rewrittenMetrics = core.extractUsir24d(rewrittenPreview);
    const result = {
      text, targetPhase, gains, metrics, classification, microGate,
      trajectoryWave, hsmm, sinkholes, directives, modulatedQ,
      tierSHex: toHex(tierS), rewrittenPreview, rewrittenMetrics
    };
    state.text = text;
    state.targetPhase = targetPhase;
    state.gains = { ...gains };
    state.lastAnalysis = result;
    renderDOM(result);
    return result;
  }

  function renderDOM(res) {
    if (typeof document === 'undefined' || !document.getElementById) return;
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setTxt('usir-phase-badge', res.classification.phase);
    setTxt('usir-cv-badge', 'CV_L: ' + res.metrics.len_cv.toFixed(3));
    setTxt('usir-cadence-badge', 'Alive: ' + res.metrics.cadence_alive.toFixed(2));
    setTxt('usir-micro-gate-badge', res.microGate.passed ? 'MICRO GATE: PASS (0 violations)' : 'MICRO GATE: BLOCK (' + res.microGate.violations.length + ')');
    setTxt('usir-compiled-prompt', res.directives.compiledPrompt);
    setTxt('usir-tiers-hex', res.tierSHex);
    const shEl = document.getElementById('usir-sinkhole-alerts');
    if (shEl) {
      shEl.innerHTML = res.sinkholes.hasSinkholes
        ? res.sinkholes.sinkholes.map(s => '<div class="sinkhole-alert" data-paragraph-index="' + s.paragraphIndex + '">⚠️ Deadzone Sinkhole at Paragraph #' + (s.paragraphIndex + 1) + ' (Index ' + s.paragraphIndex + ') | CV_L=' + s.localCvL.toFixed(3) + '</div>').join('')
        : '<div class="sinkhole-ok">No Deadzone Sinkholes Detected</div>';
    }
  }

  function loadPreset(presetId) {
    const p = PRESETS[presetId];
    if (!p) throw new Error('Unknown preset: ' + presetId);
    state.presetId = presetId;
    if (typeof document !== 'undefined' && document.getElementById) {
      const ed = document.getElementById('usir-editor');
      if (ed) ed.value = p.text;
    }
    const target = p.expectedPhase === 'V_Deadzone' ? 'III_Void' : p.expectedPhase;
    return analyzeText(p.text, target, state.gains);
  }

  function setGains(newGains) {
    state.gains = { ...state.gains, ...newGains };
    return analyzeText(state.text, state.targetPhase, state.gains);
  }

  function setTargetPhase(targetPhase) {
    state.targetPhase = targetPhase;
    return analyzeText(state.text, targetPhase, state.gains);
  }

  root.USIRStudio = {
    PRESETS, ELEMENT_META, analyzeText, loadPreset, setGains, setTargetPhase,
    getState: () => state,
    runMicroGate: (t, phase) => core.runMicroConstraintGate(t, { targetPhase: phase || state.targetPhase })
  };
})(typeof window !== 'undefined' ? window : globalThis);
`;

const html = `<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <title>USIR Studio v1.0 — 5-Phase Symplectic Oscilloscope & Equalizer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0b0f17; color: #e2e8f0; font-family: system-ui, sans-serif; display: flex; height: 100vh; overflow: hidden; }
    #left-pane { width: 30%; padding: 16px; border-right: 1px solid #1e293b; display: flex; flex-direction: column; gap: 10px; }
    #center-pane { width: 45%; padding: 16px; border-right: 1px solid #1e293b; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
    #usir-docked-drawer { width: 25%; padding: 16px; background: #0f172a; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
    textarea { width: 100%; flex: 1; background: #111827; color: #f8fafc; border: 1px solid #334155; padding: 10px; border-radius: 6px; }
    .preset-bar { display: flex; flex-wrap: wrap; gap: 6px; }
    button { background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 6px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
    .badge { padding: 4px 8px; border-radius: 4px; background: #1e293b; font-size: 12px; font-family: monospace; }
    .sinkhole-alert { background: #450a0a; color: #fca5a5; padding: 8px; border: 1px solid #ef4444; border-radius: 4px; font-size: 12px; }
    pre { background: #090d16; padding: 8px; border-radius: 4px; font-size: 11px; white-space: pre-wrap; max-height: 160px; overflow-y: auto; }
  </style>
</head>
<body>
  <section id="left-pane">
    <h2>USIR Studio v1.0</h2>
    <div class="preset-bar">
      <button data-preset-id="VOID-03" onclick="USIRStudio.loadPreset('VOID-03')">夏目漱石：草枕 [VOID-03]</button>
      <button data-preset-id="CRIT-05" onclick="USIRStudio.loadPreset('CRIT-05')">梶井基次郎：桜の樹の下には [CRIT-05]</button>
      <button data-preset-id="CRYS-01" onclick="USIRStudio.loadPreset('CRYS-01')">民法条文 [CRYS-01]</button>
      <button data-preset-id="DEAD-01" onclick="USIRStudio.loadPreset('DEAD-01')">AI模倣スロップ [DEAD-01]</button>
      <button data-preset-id="HYBRID-5P" onclick="USIRStudio.loadPreset('HYBRID-5P')">5段落ハイブリッド長文 [中だるみ混入デモ]</button>
    </div>
    <div><span id="usir-cv-badge" class="badge">CV_L: --</span> <span id="usir-cadence-badge" class="badge">Alive: --</span></div>
    <textarea id="usir-editor" oninput="USIRStudio.analyzeText(this.value)"></textarea>
  </section>
  <section id="center-pane">
    <h3>5-Phase Thermodynamic State Diagram (κ_kin vs φ_void)</h3>
    <svg id="usir-phase-map" viewBox="0 0 400 220" width="100%" height="220">
      <rect width="400" height="220" fill="#090d16" stroke="#334155" />
      <circle id="particle-porcelain" cx="80" cy="160" r="5" fill="#e2e8f0" />
      <circle id="particle-amber" cx="260" cy="90" r="5" fill="#f59e0b" />
      <circle id="particle-obsidian" cx="120" cy="40" r="5" fill="#38bdf8" />
      <circle id="particle-lightning" cx="340" cy="140" r="5" fill="#a855f7" />
      <circle id="usir-live-point" cx="200" cy="110" r="7" fill="#10b981" />
    </svg>
    <h3>3-Mode HSMM Trajectory Oscilloscope (序:Mode0 / 破:Mode1 / 間:Mode2)</h3>
    <svg id="usir-hsmm-oscilloscope" viewBox="0 0 400 140" width="100%" height="140">
      <rect width="400" height="140" fill="#090d16" stroke="#334155" />
    </svg>
    <div id="usir-sinkhole-alerts"></div>
  </section>
  <aside id="usir-docked-drawer" style="width: 25%;" data-placement="right-detail-drawer" data-allow-popup-modal="false" data-auto-interrupt-typing="false">
    <h3>Docked Atelier Equalizer</h3>
    <div id="usir-phase-badge" class="badge">--</div>
    <div id="usir-micro-gate-badge" class="badge">--</div>
    <div id="usir-spectrum-bars">
      <div data-element="Sg-01">Sg-01 Gr</div><div data-element="Sg-02">Sg-02 Cr</div><div data-element="Sg-03">Sg-03 Pr</div><div data-element="Sg-04">Sg-04 Jt</div>
      <div data-element="Sg-05">Sg-05 Cp</div><div data-element="Sg-06">Sg-06 Sy</div><div data-element="Sg-07">Sg-07 Cs</div><div data-element="Sg-08">Sg-08 Lp</div>
      <div data-element="Sg-09">Sg-09 Dy</div><div data-element="Sg-10">Sg-10 At</div><div data-element="Sg-11">Sg-11 Ma</div><div data-element="Sg-12">Sg-12 Ru</div>
    </div>
    <div>
      <label>Flux <input id="slider-flux" type="range" min="0" max="2" step="0.05" value="1" /></label>
      <label>Lattice <input id="slider-lattice" type="range" min="0" max="2" step="0.05" value="1" /></label>
      <label>Void <input id="slider-void" type="range" min="0" max="2" step="0.05" value="1" /></label>
      <label>Shock <input id="slider-shock" type="range" min="0" max="2" step="0.05" value="1" /></label>
      <select id="select-target-phase"><option value="III_Void">III_Void</option><option value="II_Critical">II_Critical</option><option value="I_Crystalline">I_Crystalline</option></select>
    </div>
    <button id="btn-copy-prompt">Copy Hamiltonian Prompt</button>
    <button id="btn-preview-transition">Preview Closed-Loop Phase Transition</button>
    <pre id="usir-compiled-prompt"></pre>
    <pre id="usir-tiers-hex"></pre>
  </aside>
  <script>${coreBundleJs}</script>
  <script>${studioAppJs}</script>
</body>
</html>`;

const outDir = path.join(repoRoot, 'packages/usir-bridges/public');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'usir-studio.html'), html, 'utf-8');
console.log('Generated usir-studio.html successfully (' + html.length + ' bytes).');
