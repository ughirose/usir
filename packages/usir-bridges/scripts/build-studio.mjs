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
  'EN-HEMINGWAY-VOID': {
    id: 'EN-HEMINGWAY-VOID',
    title: 'Ernest Hemingway: Big Two-Hearted River [EN-HEMINGWAY-VOID]',
    expectedPhase: 'III_Void',
    text: 'The river was green. Cold water ran over the white stones. Nick looked down through the dark pool and saw the trout. He did not move. The sun was hot on his back. There was no sound in the trees. The wind blew from the hills, cold and clean.',
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
    lastAnalysis: null,
    genesis: null
  };

  function toHex(u8) {
    return Array.from(u8.slice(0, 32)).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function analyzeText(text, targetPhase = state.targetPhase, gains = state.gains) {
    // 1. Language Auto-Detection
    const asciiCount = (text.match(/[a-zA-Z]/g) || []).length;
    const nonSpaceCount = text.replace(/\\s+/g, '').length;
    const isEnglish = nonSpaceCount > 0 && (asciiCount / nonSpaceCount > 0.50);
    const language = isEnglish ? 'en' : 'ja';

    let metrics, classification, microGate, directives, modulatedQ, tierS, rewrittenPreview, rewrittenMetrics;

    if (isEnglish) {
      const zEnFloat = core.extractUsir24dEnglish(text);
      const zEn = Array.from(zEnFloat);
      const enPhase = core.classifyEnglishPhase(text);
      const baseMetrics = core.extractUsir24d(text);
      metrics = {
        ...baseMetrics,
        z_24d: zEn,
        q: zEn.slice(0, 12),
        p: zEn.slice(12, 24),
      };
      classification = { phase: enPhase, distance: 0.0 };
      microGate = {
        passed: enPhase !== 'V_Deadzone',
        violations: enPhase === 'V_Deadzone' ? ['English AI Slop detected'] : [],
      };
      directives = core.generateRewriteDirectives(text, targetPhase, metrics);
      modulatedQ = core.applyEqualizer(metrics.q, gains);
      tierS = core.encodeTierS(metrics);
      rewrittenPreview = text;
      rewrittenMetrics = metrics;
    } else {
      metrics = core.extractUsir24d(text);
      classification = core.classifyPhase(metrics.z_24d);
      microGate = core.runMicroConstraintGate(text, { targetPhase, registerMode: 'classical_literary' });
      directives = core.generateRewriteDirectives(text, targetPhase, metrics);
      modulatedQ = core.applyEqualizer(metrics.q, gains);
      tierS = core.encodeTierS(metrics);
      rewrittenPreview = core.steerPhaseRewrite(text, targetPhase);
      rewrittenMetrics = core.extractUsir24d(rewrittenPreview);
    }

    const trajectoryWave = core.extractTrajectoryWave(text, { windowSize: 3, stride: 1 });
    const hsmm = core.decodeHsmmModes(trajectoryWave);
    const sinkholes = core.detectDeadzoneSinkholes(text);

    // 2. Style Chemistry (4 Major Aesthetic Molecules)
    const molecules = core.computeStyleMolecules(metrics.q, metrics.p);

    // 3. Cross-Modal Synesthetic Transpiler
    const music = core.transpileToMusic(metrics.z_24d);
    const visual = core.transpileToVisual(metrics.z_24d);
    const uiCss = core.transpileToCssVariables(metrics.z_24d);
    const crossModal = { music, visual, uiCss };

    const result = {
      text, targetPhase, gains, language, metrics, classification, microGate,
      trajectoryWave, hsmm, sinkholes, directives, modulatedQ,
      tierSHex: toHex(tierS), rewrittenPreview, rewrittenMetrics,
      molecules, crossModal
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

    // Badges
    setTxt('usir-phase-badge', res.classification.phase);
    setTxt('usir-cv-badge', 'CV_L: ' + res.metrics.len_cv.toFixed(3));
    setTxt('usir-cadence-badge', 'Alive: ' + res.metrics.cadence_alive.toFixed(2));
    setTxt('usir-micro-gate-badge', res.microGate.passed ? 'MICRO GATE: PASS (0 violations)' : 'MICRO GATE: BLOCK (' + res.microGate.violations.length + ')');
    setTxt('usir-compiled-prompt', res.directives.compiledPrompt);
    setTxt('usir-tiers-hex', res.tierSHex);

    // Language Badge
    setTxt('usir-lang-badge', res.language === 'en' ? 'EN (English Isomorphic)' : 'JA (Japanese)');

    // Deadzone Sinkholes
    const shEl = document.getElementById('usir-sinkhole-alerts');
    if (shEl) {
      shEl.innerHTML = res.sinkholes.hasSinkholes
        ? res.sinkholes.sinkholes.map(s => '<div class="sinkhole-alert" data-paragraph-index="' + s.paragraphIndex + '">⚠️ Deadzone Sinkhole at Paragraph #' + (s.paragraphIndex + 1) + ' (Index ' + s.paragraphIndex + ') | CV_L=' + s.localCvL.toFixed(3) + '</div>').join('')
        : '<div class="sinkhole-ok">No Deadzone Sinkholes Detected</div>';
    }

    // Style Chemistry (4 Aesthetic Molecules)
    const mol = res.molecules;
    if (mol) {
      const updateMol = (id, label, val) => {
        const el = document.getElementById(id);
        if (el) {
          el.textContent = label + ': ' + val.toFixed(3);
          el.innerHTML = '<div style="display: flex; justify-content: space-between;"><span>' + label + '</span><span>' + val.toFixed(3) + '</span></div><div style="background: #1e293b; height: 4px; border-radius: 2px; margin-top: 2px;"><div style="background: #38bdf8; width: ' + (val * 100).toFixed(1) + '%; height: 100%; border-radius: 2px;"></div></div>';
        }
      };
      updateMol('mol-yugen', '幽玄 (Yugen)', mol.Mol_Yugen);
      updateMol('mol-wabi', '侘び (Wabi)', mol.Mol_Wabi);
      updateMol('mol-qiyun', '気韻生動 (QiYun)', mol.Mol_QiYun);
      updateMol('mol-iki', '粋 (Iki)', mol.Mol_Iki);
    }

    // Cross-Modal Synesthetic Transpiler Preview
    const cm = res.crossModal;
    if (cm) {
      const m = cm.music;
      setTxt('cm-music-summary', 'BPM Fluctuation: ' + (m.bpm_fluctuation * 100).toFixed(1) + '% | Dissonance: ' + (m.harmonic_dissonance * 100).toFixed(1) + '% | Rest/Ma: ' + (m.rest_ratio * 100).toFixed(1) + '% | Articulation: ' + m.soliton_accent_gain.toFixed(2));

      const v = cm.visual;
      setTxt('cm-visual-summary', 'CFG Scale: ' + v.cfg_scale.toFixed(1) + ' | High-Freq Detail: ' + v.high_freq_noise_gain.toFixed(2) + ' | Negative Space: ' + (v.negative_space_ratio * 100).toFixed(1) + '% | Color Temp: ' + (v.color_contrast_temperature * 100).toFixed(1) + '%');

      const u = cm.uiCss;
      const card = document.getElementById('cm-live-ui-card');
      if (card) {
        const padPx = Math.round(8 * u.whitespace_multiplier);
        const radPx = Math.round(u.border_radius_px);
        const contrast = u.contrast_ratio.toFixed(1);
        const weight = Math.round(400 * u.typography_scale_ratio);
        const durMs = Math.round(u.transition_duration_ms);

        const styleStr = '--usir-padding: ' + padPx + 'px; --usir-radius: ' + radPx + 'px; --usir-contrast: ' + contrast + '; --usir-font-weight: ' + weight + '; --usir-duration: ' + durMs + 'ms; padding: var(--usir-padding); border-radius: var(--usir-radius); font-weight: var(--usir-font-weight); transition: all var(--usir-duration) ease;';

        if (card.setAttribute) card.setAttribute('style', styleStr);
        card.style = card.style || {};
        card.style.cssText = styleStr;
        if (typeof card.style.setProperty === 'function') {
          card.style.setProperty('--usir-padding', padPx + 'px');
          card.style.setProperty('--usir-radius', radPx + 'px');
          card.style.setProperty('--usir-contrast', contrast);
          card.style.setProperty('--usir-font-weight', String(weight));
          card.style.setProperty('--usir-duration', durMs + 'ms');
        }

        card.innerHTML = '<div class="card-inner"><strong>Live Synesthetic UI Preview Card</strong><p style="margin-top: 4px; font-size: 11px;">Padding: ' + padPx + 'px | Radius: ' + radPx + 'px | Contrast: ' + contrast + ' | Weight: ' + weight + '</p></div>';
      }
    }
  }

  function renderGenesisDOM(res) {
    if (typeof document === 'undefined' || !document.getElementById) return;
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    const q = res.zGenesis.slice(0, 12);
    const p = res.zGenesis.slice(12, 24);
    const pKin = Math.sqrt(p.reduce((s, v) => s + v * v, 0) / 12);
    const qVoid = (q[2] + q[6] + q[10]) / 3;
    const cx = Math.max(30, Math.min(370, 50 + pKin * 300));
    const cy = Math.max(30, Math.min(190, 190 - qVoid * 150));

    const starEl = document.getElementById('genesis-star-marker');
    if (starEl) {
      let pts = [];
      for (let i = 0; i < 10; i++) {
        const angle = (i * Math.PI) / 5 - Math.PI / 2;
        const r = i % 2 === 0 ? 10 : 5;
        pts.push((cx + r * Math.cos(angle)).toFixed(1) + ',' + (cy + r * Math.sin(angle)).toFixed(1));
      }
      const ptsStr = pts.join(' ');
      if (starEl.setAttribute) {
        starEl.setAttribute('points', ptsStr);
        starEl.setAttribute('style', 'display: inline;');
      } else {
        starEl.points = ptsStr;
        starEl.style = starEl.style || {};
        starEl.style.display = 'inline';
      }
    }

    const directive = '[USIR-GENESIS SYNTHESIZED HAMILTONIAN DIRECTIVE]\\n' +
      'Uncharted Voronoi Coordinate z* ∈ R^24 (d_M^2 = ' + res.mahalanobisD2.toFixed(3) + ' <= 36.415, min_k d_G = ' + res.minRiemannianDistance.toFixed(3) + ')\\n' +
      'Dominant Aesthetic Molecule: ' + res.molecules.dominantMolecule + ' (Coupling Energy: ' + res.molecules.synestheticCouplingEnergy.toFixed(3) + ')\\n' +
      'Molecular Recipe:\\n' +
      '  • 幽玄 (Yugen): ' + (res.molecules.Mol_Yugen * 100).toFixed(1) + '%\\n' +
      '  • 侘び (Wabi):  ' + (res.molecules.Mol_Wabi * 100).toFixed(1) + '%\\n' +
      '  • 気韻生動 (QiYun): ' + (res.molecules.Mol_QiYun * 100).toFixed(1) + '%\\n' +
      '  • 粋 (Iki):   ' + (res.molecules.Mol_Iki * 100).toFixed(1) + '%\\n' +
      'Hamiltonian Tuning:\\n' +
      '  α_flux=' + (1.0 + 0.4 * (q[0] - 0.5)).toFixed(2) + ', α_lattice=' + (1.0 + 0.4 * (q[1] - 0.5)).toFixed(2) + ', α_void=' + (1.0 + 0.4 * (q[2] - 0.5)).toFixed(2) + ', α_shock=' + (1.0 + 0.4 * (q[3] - 0.5)).toFixed(2) + '\\n' +
      'Genesis Phase Prompt Injection: Steer generator towards unexplored Riemannian style void without collapsing into Deadzone.';

    setTxt('usir-genesis-directive', directive);
  }

  function triggerGenesis() {
    const particles = Object.values(core.CANONICAL_PARTICLES);
    const genesisRes = core.discoverGenesisStyle(particles, undefined, { nRestarts: 4, nSteps: 25 });
    state.genesis = genesisRes;
    renderGenesisDOM(genesisRes);
    return genesisRes;
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

  function transpileCurrentState(targetModality) {
    if (!state.lastAnalysis) return null;
    return core.transpileCrossModal(state.lastAnalysis.metrics.z_24d, targetModality);
  }

  root.USIRStudio = {
    PRESETS, ELEMENT_META, analyzeText, loadPreset, setGains, setTargetPhase,
    triggerGenesis, transpileCurrentState,
    getState: () => state,
    runMicroGate: (t, phase) => core.runMicroConstraintGate(t, { targetPhase: phase || state.targetPhase })
  };
})(typeof window !== 'undefined' ? window : globalThis);
`;

const html = `<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <title>USIR Studio v3.0 — 5-Phase Symplectic Oscilloscope, Chemistry & Synesthetic Transpiler</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0b0f17; color: #e2e8f0; font-family: system-ui, sans-serif; display: flex; height: 100vh; overflow: hidden; }
    #left-pane { width: 30%; padding: 16px; border-right: 1px solid #1e293b; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
    #center-pane { width: 45%; padding: 16px; border-right: 1px solid #1e293b; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
    #usir-docked-drawer { width: 25%; padding: 16px; background: #0f172a; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
    textarea { width: 100%; min-height: 140px; flex: 1; background: #111827; color: #f8fafc; border: 1px solid #334155; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 12px; }
    .preset-bar { display: flex; flex-wrap: wrap; gap: 6px; }
    button { background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 6px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
    button:hover { background: #334155; }
    .badge { padding: 4px 8px; border-radius: 4px; background: #1e293b; font-size: 12px; font-family: monospace; display: inline-block; }
    .sinkhole-alert { background: #450a0a; color: #fca5a5; padding: 8px; border: 1px solid #ef4444; border-radius: 4px; font-size: 12px; }
    .sinkhole-ok { background: #064e3b; color: #6ee7b7; padding: 8px; border: 1px solid #10b981; border-radius: 4px; font-size: 12px; }
    pre { background: #090d16; padding: 8px; border-radius: 4px; font-size: 11px; white-space: pre-wrap; max-height: 160px; overflow-y: auto; border: 1px solid #1e293b; font-family: monospace; }
    .molecule-meter { background: #111827; border: 1px solid #1e293b; border-radius: 4px; padding: 6px; font-size: 11px; margin-top: 4px; }
    #usir-chemistry-panel, #usir-genesis-panel, #usir-cross-modal-panel { background: #0d1424; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; }
    #cm-live-ui-card { background: #1e293b; border: 1px solid #475569; padding: 12px; border-radius: 8px; font-size: 12px; }
  </style>
</head>
<body>
  <section id="left-pane">
    <h2>USIR Studio v3.0</h2>
    <div class="preset-bar">
      <button data-preset-id="VOID-03" onclick="USIRStudio.loadPreset('VOID-03')">夏目漱石：草枕 [VOID-03]</button>
      <button data-preset-id="CRIT-05" onclick="USIRStudio.loadPreset('CRIT-05')">梶井基次郎：桜の樹の下には [CRIT-05]</button>
      <button data-preset-id="CRYS-01" onclick="USIRStudio.loadPreset('CRYS-01')">民法条文 [CRYS-01]</button>
      <button data-preset-id="DEAD-01" onclick="USIRStudio.loadPreset('DEAD-01')">AI模倣スロップ [DEAD-01]</button>
      <button data-preset-id="HYBRID-5P" onclick="USIRStudio.loadPreset('HYBRID-5P')">5段落ハイブリッド長文 [中だるみ混入デモ]</button>
      <button data-preset-id="EN-HEMINGWAY-VOID" onclick="USIRStudio.loadPreset('EN-HEMINGWAY-VOID')">Hemingway: Big Two-Hearted River [EN-HEMINGWAY-VOID]</button>
    </div>
    <div>
      <span id="usir-lang-badge" class="badge">JA (Japanese)</span>
      <span id="usir-cv-badge" class="badge">CV_L: --</span>
      <span id="usir-cadence-badge" class="badge">Alive: --</span>
    </div>
    <textarea id="usir-editor" oninput="USIRStudio.analyzeText(this.value)"></textarea>
    <div id="usir-chemistry-panel">
      <h4 style="margin-bottom: 6px; font-size: 12px;">Style Chemistry (4 Major Aesthetic Molecules)</h4>
      <div id="mol-yugen" class="molecule-meter">幽玄 (Yugen): --</div>
      <div id="mol-wabi" class="molecule-meter">侘び (Wabi): --</div>
      <div id="mol-qiyun" class="molecule-meter">気韻生動 (QiYun): --</div>
      <div id="mol-iki" class="molecule-meter">粋 (Iki): --</div>
    </div>
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
      <polygon id="genesis-star-marker" points="200,100 203,107 210,107 205,111 207,118 200,114 193,118 195,111 190,107 197,107" fill="#fbbf24" stroke="#f59e0b" stroke-width="1.5" style="display: none;" />
    </svg>
    <div id="usir-genesis-panel">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <h4 style="margin: 0; font-size: 12px;">Uncharted Style Discovery (usir-genesis)</h4>
        <button id="btn-usir-genesis" onclick="USIRStudio.triggerGenesis()">✨ Discover Uncharted Style (usir-genesis)</button>
      </div>
      <pre id="usir-genesis-directive">[Click 'Discover Uncharted Style' to compute Voronoi void within χ²(24, 0.95)]</pre>
    </div>
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
    <div id="usir-cross-modal-panel">
      <h4 style="margin-bottom: 6px; font-size: 12px;">Cross-Modal Synesthetic Transpiler</h4>
      <div style="font-size: 11px; margin-bottom: 4px;">🎵 Music Generative Parameters:</div>
      <div id="cm-music-summary" class="badge">BPM: -- | Dissonance: -- | Rest: --</div>
      <div style="font-size: 11px; margin-top: 8px; margin-bottom: 4px;">🎨 Visual Diffusion Parameters:</div>
      <div id="cm-visual-summary" class="badge">CFG: -- | Detail: -- | Space: --</div>
      <div style="font-size: 11px; margin-top: 8px; margin-bottom: 4px;">🖥️ Live UI CSS Morphing Card:</div>
      <div id="cm-live-ui-card">
        Live Synesthetic UI Preview Card
      </div>
    </div>
  </aside>
  <script>${coreBundleJs}</script>
  <script>${studioAppJs}</script>
</body>
</html>`;

const outDir = path.join(repoRoot, 'packages/usir-bridges/public');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'usir-studio.html'), html, 'utf-8');
console.log('Generated usir-studio.html successfully (' + html.length + ' bytes).');
