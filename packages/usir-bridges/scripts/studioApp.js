(function(root) {
  const core = root.USIRCore;
  const bridges = root.USIRBridges;
  const PRESETS = root.__USIR_PRESETS__ || {};

  const ELEMENT_META = [
    { idx: 0, code: 'Sg-01 Gr', group: 'Flux' }, { idx: 4, code: 'Sg-05 Cp', group: 'Flux' }, { idx: 8, code: 'Sg-09 Dy', group: 'Flux' },
    { idx: 1, code: 'Sg-02 Cr', group: 'Lattice' }, { idx: 5, code: 'Sg-06 Sy', group: 'Lattice' }, { idx: 9, code: 'Sg-10 At', group: 'Lattice' },
    { idx: 2, code: 'Sg-03 Pr', group: 'Void' }, { idx: 6, code: 'Sg-07 Cs', group: 'Void' }, { idx: 10, code: 'Sg-11 Ma', group: 'Void' },
    { idx: 3, code: 'Sg-04 Jt', group: 'Shock' }, { idx: 7, code: 'Sg-08 Lp', group: 'Shock' }, { idx: 11, code: 'Sg-12 Ru', group: 'Shock' }
  ];

  let state = {
    presetId: 'VOID-03',
    useCaseId: 'case_1_literary',
    mode: 'instant',
    targetPhase: 'III_Void',
    text: (PRESETS['VOID-03'] && PRESETS['VOID-03'].text) || '',
    beforeText: '',
    afterText: '',
    coreFacts: [],
    gains: { alpha_flux: 1.0, alpha_lattice: 1.0, alpha_void: 1.0, alpha_shock: 1.0 },
    lastAnalysis: null,
    redlineResult: null,
    genesis: null,
    genesisArtifact: null,
    storyboard: null,
    activeAudioPlayer: null,
    activeAudioComposition: null,
    audioContext: null
  };

  function toHex(u8) {
    return Array.from(u8.slice(0, 32)).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function analyzeText(text, targetPhase = state.targetPhase, gains = state.gains) {
    if (!core) return null;
    const asciiCount = (text.match(/[a-zA-Z]/g) || []).length;
    const nonSpaceCount = text.replace(/\s+/g, '').length;
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
      directives = core.generateRewriteDirectives(text, targetPhase === 'Genesis' ? 'III_Void' : targetPhase, metrics);
      modulatedQ = core.applyEqualizer(metrics.q, gains);
      tierS = core.encodeTierS(metrics);
      rewrittenPreview = text;
      rewrittenMetrics = metrics;
    } else {
      metrics = core.extractUsir24d(text);
      classification = core.classifyPhase(metrics.z_24d);
      microGate = core.runMicroConstraintGate(text, { targetPhase: targetPhase === 'Genesis' ? 'III_Void' : targetPhase, registerMode: 'classical_literary' });
      directives = core.generateRewriteDirectives(text, targetPhase === 'Genesis' ? 'III_Void' : targetPhase, metrics);
      modulatedQ = core.applyEqualizer(metrics.q, gains);
      tierS = core.encodeTierS(metrics);
      rewrittenPreview = core.steerPhaseRewrite(text, targetPhase === 'Genesis' ? 'III_Void' : targetPhase);
      rewrittenMetrics = core.extractUsir24d(rewrittenPreview);
    }

    const trajectoryWave = core.extractTrajectoryWave(text, { windowSize: 3, stride: 1 });
    const hsmm = core.decodeHsmmModes(trajectoryWave);
    const sinkholes = core.detectDeadzoneSinkholes(text);
    const molecules = core.computeStyleMolecules(metrics.q, metrics.p);

    const music = core.transpileToMusic(metrics.z_24d);
    const visual = core.transpileToVisual(metrics.z_24d);
    const uiCss = core.transpileToCssVariables(metrics.z_24d);
    const crossModal = { music, visual, uiCss };

    let redlineResult = state.redlineResult;
    if (state.beforeText && state.beforeText !== text && !redlineResult) {
      redlineResult = core.computeRedlineDiffAndRoi(state.beforeText, text, targetPhase, state.coreFacts);
      state.redlineResult = redlineResult;
    }

    const result = {
      text, targetPhase, gains, language, metrics, classification, microGate,
      trajectoryWave, hsmm, sinkholes, directives, modulatedQ,
      tierSHex: toHex(tierS), rewrittenPreview, rewrittenMetrics,
      molecules, crossModal, redlineResult
    };

    state.text = text;
    state.targetPhase = targetPhase;
    state.gains = { ...gains };
    state.lastAnalysis = result;

    renderDOM(result);
    if (bridges && state.beforeText) {
      renderStoryboard(state.beforeText, text, targetPhase);
    }
    renderCinemaPreview(metrics.z_24d);
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
    setTxt('usir-lang-badge', res.language === 'en' ? 'EN (English Isomorphic)' : 'JA (Japanese)');

    const shEl = document.getElementById('usir-sinkhole-alerts');
    if (shEl) {
      shEl.innerHTML = res.sinkholes.hasSinkholes
        ? res.sinkholes.sinkholes.map(s => '<div class="sinkhole-alert" data-paragraph-index="' + s.paragraphIndex + '">⚠️ Deadzone Sinkhole at Paragraph #' + (s.paragraphIndex + 1) + ' (Index ' + s.paragraphIndex + ') | CV_L=' + s.localCvL.toFixed(3) + '</div>').join('')
        : '<div class="sinkhole-ok">No Deadzone Sinkholes Detected</div>';
    }

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

    if (res.redlineResult) {
      renderRedlineAndRoiDOM(res.redlineResult);
    }
  }

  function renderRedlineAndRoiDOM(diff) {
    if (typeof document === 'undefined' || !document.getElementById || !diff) return;
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    const setHtml = (id, v) => { const el = document.getElementById(id); if (el) el.innerHTML = v; };

    setHtml('usir-redline-html', diff.redlineHtml || '');

    if (diff.roiMetrics) {
      const roi = diff.roiMetrics;
      setTxt('roi-dropoff-risk', roi.estimatedDropoffRiskBefore.toFixed(1) + '% → ' + roi.estimatedDropoffRiskAfter.toFixed(1) + '%');
      setTxt('roi-slop-reduction', (roi.slopReductionRatio * 100).toFixed(1) + '% 削減');
      setTxt('roi-clarity-gain', roi.clarityAndImpactGain.toFixed(2) + 'x 向上');
      setTxt('roi-fact-retention', (diff.factRetentionRate * 100).toFixed(0) + '% 保持 (' + diff.retainedFacts.length + ' 確定事実)');
    }

    if (diff.rhythmBarcode) {
      const rb = diff.rhythmBarcode;
      const renderBars = (containerId, bars, lengths, color) => {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = bars.map((b, i) =>
          '<div class="rhythm-bar" style="height: ' + Math.max(12, Math.round(b * 100)) + '%; background: ' + color + ';" title="Len ' + lengths[i] + '"></div>'
        ).join('');
      };
      renderBars('barcode-before', rb.beforeBars, rb.beforeLengths, '#64748b');
      renderBars('barcode-after', rb.afterBars, rb.afterLengths, '#10b981');
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
    const cy = Math.max(30, Math.min(170, 170 - qVoid * 130));

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

    if (res.artifact) {
      setTxt('genesis-style-name', res.artifact.styleNameJa);
      setTxt('genesis-prose', res.artifact.generatedProseJa);
    }

    const directive = '[USIR-GENESIS SYNTHESIZED HAMILTONIAN DIRECTIVE]\n' +
      'Uncharted Voronoi Coordinate z* ∈ R^24 (d_M^2 = ' + res.mahalanobisD2.toFixed(3) + ' <= 36.415, min_k d_G = ' + res.minRiemannianDistance.toFixed(3) + ')\n' +
      'Dominant Aesthetic Molecule: ' + res.molecules.dominantMolecule + ' (Coupling Energy: ' + res.molecules.synestheticCouplingEnergy.toFixed(3) + ')\n' +
      'Molecular Recipe:\n' +
      '  • 幽玄 (Yugen): ' + (res.molecules.Mol_Yugen * 100).toFixed(1) + '%\n' +
      '  • 侘び (Wabi):  ' + (res.molecules.Mol_Wabi * 100).toFixed(1) + '%\n' +
      '  • 気韻生動 (QiYun): ' + (res.molecules.Mol_QiYun * 100).toFixed(1) + '%\n' +
      '  • 粋 (Iki):   ' + (res.molecules.Mol_Iki * 100).toFixed(1) + '%\n' +
      'Hamiltonian Tuning:\n' +
      '  α_flux=' + (1.0 + 0.4 * (q[0] - 0.5)).toFixed(2) + ', α_lattice=' + (1.0 + 0.4 * (q[1] - 0.5)).toFixed(2) + ', α_void=' + (1.0 + 0.4 * (q[2] - 0.5)).toFixed(2) + ', α_shock=' + (1.0 + 0.4 * (q[3] - 0.5)).toFixed(2) + '\n' +
      'Genesis Phase Prompt Injection: Steer generator towards unexplored Riemannian style void without collapsing into Deadzone.';

    setTxt('usir-genesis-directive', directive);
  }

  function renderStoryboard(beforeText, afterText, mode) {
    if (!bridges || !bridges.generateMultimodalStoryboard) return;
    const sb = bridges.generateMultimodalStoryboard(beforeText, afterText, mode);
    state.storyboard = sb;

    const container = document.getElementById('multimodal-storyboard-table');
    if (!container) return;

    let html = '<table class="storyboard-table"><thead><tr>' +
      '<th>Scene / Text</th>' +
      '<th>Music Cue</th>' +
      '<th>Image Prompt (Flux/Midjourney)</th>' +
      '<th>Video Script (Sora/Veo)</th>' +
      '</tr></thead><tbody>';

    for (const row of sb.rows) {
      html += '<tr>' +
        '<td><strong>' + row.textColumn.sectionLabel + '</strong><br/><span style="color:#94a3b8; font-size:9px;">Before: ' + row.textColumn.beforeParagraph.slice(0, 48) + '...</span><br/><span style="color:#6ee7b7; font-size:9px;">After: ' + row.textColumn.afterParagraph.slice(0, 48) + '...</span></td>' +
        '<td>BPM: ' + row.musicCue.bpm + '<br/>Tonality: ' + row.musicCue.tonality + '<br/>Rest: ' + row.musicCue.restRatioPercent + '%<br/><button class="btn-play-cue" onclick="USIRStudio.playAudio(\'' + row.shaderState.dominantMode + '\')">▶ 試聴</button></td>' +
        '<td><span style="color:#fbbf24; font-size:9px;">' + row.imagePrompt.subjectScene + '</span><br/>' + row.imagePrompt.lensFocalLength + '<br/><span style="color:#94a3b8; font-size:9px;">Space: ' + row.imagePrompt.negativeSpaceComposition + '</span></td>' +
        '<td>Duration: ' + row.videoScript.durationSec + 's<br/>Camera: ' + row.videoScript.cameraVector + '<br/>Curve: ' + row.videoScript.speedCurve + '</td>' +
        '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function renderCinemaPreview(z24) {
    if (!bridges || !bridges.computeCinematicShaderState || !bridges.renderCinematicFrame) return;
    const canvas = document.getElementById('cinematic-shader-canvas');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const shader = bridges.computeCinematicShaderState(z24, state.targetPhase === 'Genesis' ? 'genesis' : undefined);
    bridges.renderCinematicFrame(ctx, shader, 0, canvas.width || 400, canvas.height || 150);
  }

  function triggerGenesis(seedIndex = 1) {
    const particles = Object.values(core.CANONICAL_PARTICLES);
    const genesisRes = core.discoverGenesisStyle(particles, undefined, { nRestarts: 4, nSteps: 25 });
    let artifact = null;
    if (bridges && bridges.generateVoronoiGenesisArtifact) {
      artifact = bridges.generateVoronoiGenesisArtifact(genesisRes.zGenesis, 'ja', seedIndex);
    }
    const fullRes = { ...genesisRes, artifact };
    state.genesis = genesisRes;
    state.genesisArtifact = artifact;
    state.targetPhase = 'Genesis';
    renderGenesisDOM(fullRes);
    if (artifact) {
      state.text = artifact.generatedProseJa;
      if (typeof document !== 'undefined' && document.getElementById) {
        const ed = document.getElementById('usir-editor');
        if (ed) ed.value = state.text;
      }
      analyzeText(state.text, 'Genesis');
    }
    return fullRes;
  }

  function loadPreset(presetId) {
    const p = PRESETS[presetId];
    if (!p) throw new Error('Unknown preset: ' + presetId);
    state.presetId = presetId;
    state.beforeText = '';
    state.afterText = '';
    if (typeof document !== 'undefined' && document.getElementById) {
      const ed = document.getElementById('usir-editor');
      if (ed) ed.value = p.text;
    }
    const target = p.expectedPhase === 'V_Deadzone' ? 'III_Void' : p.expectedPhase;
    return analyzeText(p.text, target, state.gains);
  }

  function loadUseCase(useCaseId, targetMode) {
    if (!core || !core.PRESALES_SHOWCASE_CASES) return null;
    const uc = core.PRESALES_SHOWCASE_CASES[useCaseId];
    if (!uc) return null;
    state.useCaseId = useCaseId;
    state.beforeText = uc.beforeText;
    state.coreFacts = uc.coreFacts || [];

    const mode = targetMode || state.targetPhase || 'III_Void';
    const variant = uc.variants[mode] || Object.values(uc.variants)[0];
    state.afterText = variant.afterText;
    state.targetPhase = variant.mode;
    state.text = variant.afterText;
    state.redlineResult = variant.analysis;

    if (typeof document !== 'undefined' && document.getElementById) {
      const ed = document.getElementById('usir-editor');
      if (ed) ed.value = variant.afterText;
      const sel = document.getElementById('select-target-phase');
      if (sel) sel.value = variant.mode;
      const b1 = document.getElementById('btn-case-1');
      const b2 = document.getElementById('btn-case-2');
      const b3 = document.getElementById('btn-case-3');
      if (b1) b1.className = useCaseId === 'case_1_literary' ? 'active' : '';
      if (b2) b2.className = useCaseId === 'case_2_essay' ? 'active' : '';
      if (b3) b3.className = useCaseId === 'case_3_b2b_proposal' ? 'active' : '';
    }

    renderRedlineAndRoiDOM(variant.analysis);
    const analysisRes = analyzeText(variant.afterText, variant.mode);
    return { useCase: uc, variant, redlineResult: variant.analysis, analysis: analysisRes };
  }

  async function runIterativeRewrite(customOptions) {
    if (!bridges || !bridges.runLiveOrOfflineIterativeRewrite) return null;
    let endpoint = 'http://localhost:11434/api/generate';
    let model = 'llama3';
    if (typeof document !== 'undefined' && document.getElementById) {
      const epEl = document.getElementById('input-ai-endpoint');
      if (epEl && epEl.value) endpoint = epEl.value;
      const mEl = document.getElementById('input-ai-model');
      if (mEl && mEl.value) model = mEl.value;
    }

    const opts = {
      text: state.beforeText || state.text,
      targetMode: state.targetPhase,
      isLive: state.mode === 'live',
      endpoint,
      model,
      useCaseId: state.useCaseId,
      ...customOptions
    };

    const res = await bridges.runLiveOrOfflineIterativeRewrite(opts);
    state.afterText = res.finalText;
    state.text = res.finalText;
    state.redlineResult = res.redlineResult;

    if (typeof document !== 'undefined' && document.getElementById) {
      const ed = document.getElementById('usir-editor');
      if (ed) ed.value = res.finalText;
    }

    renderRedlineAndRoiDOM(res.redlineResult);
    analyzeText(res.finalText, state.targetPhase);
    return res;
  }

  function playAudio(which) {
    if (!bridges || !bridges.synthesizeMusicFromUsir24d) return null;
    let comp;
    if (which === 'before') {
      comp = bridges.synthesizeMusicFromUsir24d(core.DEADZONE_MU || new Float64Array(24), 'V_Deadzone', 16);
    } else {
      const z = state.lastAnalysis?.metrics.z_24d || core.OBSIDIAN_VOID_MU || new Float64Array(24);
      const m = state.targetPhase === 'Genesis' ? 'Genesis' : state.targetPhase;
      comp = bridges.synthesizeMusicFromUsir24d(z, m, 16);
    }
    state.activeAudioComposition = comp;

    if (typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext)) {
      if (!state.audioContext) {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (state.activeAudioPlayer) state.activeAudioPlayer.stop();
      state.activeAudioPlayer = bridges.playCompositionOnWebAudio(state.audioContext, comp);
    }

    const lbl = document.getElementById('audio-status-label');
    if (lbl) lbl.textContent = 'Playing: ' + comp.title.slice(0, 24) + '...';
    return comp;
  }

  function stopAudio() {
    if (state.activeAudioPlayer) {
      state.activeAudioPlayer.stop();
      state.activeAudioPlayer = null;
    }
    const lbl = document.getElementById('audio-status-label');
    if (lbl) lbl.textContent = 'Audio: Stopped';
  }

  function downloadMidi(customComp) {
    if (!bridges || !bridges.createMidiDataUri) return null;
    const comp = customComp || state.activeAudioComposition || bridges.synthesizeMusicFromUsir24d(core.OBSIDIAN_VOID_MU, 'III_Void', 16);
    const dataUri = bridges.createMidiDataUri(comp);
    if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
      const a = document.createElement('a');
      a.href = dataUri;
      a.download = (comp.title.replace(/[^a-zA-Z0-9_-]/g, '_') || 'usir-music') + '.mid';
      if (typeof a.click === 'function') a.click();
    }
    return dataUri;
  }

  function setMode(newMode) {
    state.mode = newMode;
    const bInst = document.getElementById('btn-mode-instant');
    const bLive = document.getElementById('btn-mode-live');
    if (bInst) bInst.className = newMode === 'instant' ? 'active' : '';
    if (bLive) bLive.className = newMode === 'live' ? 'active' : '';
  }

  function setGains(newGains) {
    state.gains = { ...state.gains, ...newGains };
    return analyzeText(state.text, state.targetPhase, state.gains);
  }

  function setTargetPhase(targetPhase) {
    state.targetPhase = targetPhase;
    if (state.useCaseId) {
      return loadUseCase(state.useCaseId, targetPhase);
    }
    return analyzeText(state.text, targetPhase, state.gains);
  }

  function transpileCurrentState(targetModality) {
    if (!state.lastAnalysis) return null;
    return core.transpileCrossModal(state.lastAnalysis.metrics.z_24d, targetModality);
  }

  root.USIRStudio = {
    PRESETS, ELEMENT_META, analyzeText, loadPreset, setGains, setTargetPhase,
    loadUseCase, runIterativeRewrite, triggerGenesis, playAudio, stopAudio, downloadMidi,
    renderCinemaPreview, renderStoryboard, setMode, transpileCurrentState,
    getState: () => state,
    runMicroGate: (t, phase) => core.runMicroConstraintGate(t, { targetPhase: phase || state.targetPhase })
  };
})(typeof window !== 'undefined' ? window : globalThis);
