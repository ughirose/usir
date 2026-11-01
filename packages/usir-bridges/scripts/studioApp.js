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

    let geodesic = null;
    let bridge = null;
    if (core.computeSemanticGeodesicCurvature) {
      geodesic = core.computeSemanticGeodesicCurvature(text);
    }
    if (core.pinBoundaryAnchors && core.solveBidirectionalCollisionBridge) {
      const domain = state.useCaseId === 'case_2_essay' ? 'article' : (state.useCaseId === 'case_3_b2b_proposal' ? 'cot_reasoning' : 'novel');
      const anchors = core.pinBoundaryAnchors(state.presetId || 'USIR Theme', domain);
      bridge = core.solveBidirectionalCollisionBridge(anchors, { k_candidates: 3 });
    }

    const result = {
      text, targetPhase, gains, language, metrics, classification, microGate,
      trajectoryWave, hsmm, sinkholes, directives, modulatedQ,
      tierSHex: toHex(tierS), rewrittenPreview, rewrittenMetrics,
      molecules, crossModal, redlineResult, geodesic, bridge
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

    if (res.geodesic) {
      renderGeodesicDOM(res.geodesic);
    }
    if (res.bridge) {
      renderBridgeDOM(res.bridge);
    }
  }

  function renderGeodesicDOM(geo) {
    if (typeof document === 'undefined' || !document.getElementById || !geo) return;
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    setTxt('geodesic-mean-curvature', geo.mean_curvature.toFixed(3));
    setTxt('geodesic-brownian-ratio', geo.brownian_ratio.toFixed(2));
    const warnEl = document.getElementById('geodesic-catalog-warning');
    if (warnEl) {
      if (geo.is_brownian_catalog) {
        warnEl.textContent = '⚠️ 小道具乱射・カタログ化検知 (Brownian Jump)';
        warnEl.style.color = '#ef4444';
      } else {
        warnEl.textContent = '正常 (Cohesive Geodesic Arc)';
        warnEl.style.color = '#10b981';
      }
    }
  }

  function renderBridgeDOM(bridge) {
    if (typeof document === 'undefined' || !document.getElementById || !bridge) return;
    const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    setTxt('bridge-status-badge', 'Bridge: 衝突溶接完了 (M=' + bridge.m_bridge.toFixed(2) + ')');
    setTxt('bridge-fwd-anchor', 'Anchor x_0: ' + bridge.anchors.x_0.slice(0, 32) + '...');
    setTxt('bridge-bwd-anchor', 'Anchor x_T: ' + bridge.anchors.x_T.slice(0, 32) + '...');
    setTxt('bridge-action-metric', 'Action Integral S_collision: ' + bridge.min_action.toFixed(3) + ' | Coupling M_bridge: ' + bridge.m_bridge.toFixed(3) + ' | Pair: (' + bridge.optimal_pair[0] + ', ' + bridge.optimal_pair[1] + ')');
    setTxt('bridge-welded-preview', 'Welded: ' + bridge.welded_text.slice(0, 140) + '...');

    renderBridgeCanvas(bridge);
  }

  function renderBridgeCanvas(bridge) {
    if (typeof document === 'undefined' || !document.getElementById) return;
    const canvas = document.getElementById('bridge-wave-canvas');
    if (!canvas || typeof canvas.getContext !== 'function') return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width || 400;
    const h = canvas.height || 70;
    ctx.clearRect(0, 0, w, h);

    // Left Forward Wave (Blue)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(10, h / 2);
    for (let x = 10; x <= w / 2; x += 10) {
      const y = h / 2 + Math.sin((x / 20) + (bridge.optimal_pair ? bridge.optimal_pair[0] : 0)) * 18;
      ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Right Backward Wave (Purple)
    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w - 10, h / 2);
    for (let x = w - 10; x >= w / 2; x -= 10) {
      const y = h / 2 + Math.sin(((w - x) / 20) + (bridge.optimal_pair ? bridge.optimal_pair[1] : 0)) * 18;
      ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Center Collision Point (Gold spark)
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  const CLASSIC_V3_SAMPLES = {
    'Sample-01': {
      title: '雨夜の湯治場と老陶工',
      v3Text: '障子を細く開けると、冷えた杉の匂いが雨とともに流れ込んできた。軒先から落ちる水滴が、土間の踏み石に一つ、また一つと鈍い音を立てて砕ける。谷底の湯治場は暮れ果てて、人の声はもう途絶えていた。渡り廊下の突き当たり、裸電球の傘の下に、削り屑の積もった轆轤が据えてある。濡れた粘土を指先で受ける老人は、こちらへ眼もくれず、ただ掌の重みを測るように背を丸めていた。水を含んだ灰色の土が、指の隙間から細くせり上がる。窯の焚口から漏れる微かな熱気だけが、乾いた板壁の節穴を赤く染めていた。言葉はなかった。ただ、轆轤の軸が軋むかすかな擦過音と、山を滑り降りてくる夜雨の気配ばかりが、薄暗い作業場を満たしていた。',
      domain: 'novel',
    },
    'Sample-02': {
      title: '未明の時計工房と解体',
      v3Text: '午前三時。路地の外灯が消え、硝子窓に青白い霜が降りた。解体台の白木には、研ぎ澄まされた鋼のピンセットと、油壺が三つ。銀の裏蓋を刃先でこじ開けた瞬間、乾いた金属の匂いが鼻孔を突いた。露わになった脱進機は、生き物の心臓のように細かく痙攣している。針が止まった。ルビーの軸受けに噛み込んだ微細な真鍮粉を、極細の豚毛で払う。息を止める。ピンセットの先端が微かに滑れば、渦巻く極細のヒゲゼンマイは一瞬で弾け、二度と戻らぬ銀の屑と化すのだ。カチリ、と小さな歯が噛み合った。再び規則的な鼓動が冷たい指先に伝わったとき、東の空がわずかに白み、凍りついた瓦屋根が青く光った。',
      domain: 'novel',
    },
    'Sample-03': {
      title: '深夜の埠頭と夜の積荷',
      v3Text: '潮風が塩の粉を吹きつけ、鉄扉の錆が赤い鱗となって剥がれ落ちた。第三突堤。吊り上げられた木箱の下で、濡れた麻縄が軋む。合図はない。男は湿った煙草の吸殻を靴底で踏み消し、革手袋の留め金を締めた。クレーンの太いワイヤーが唸りを上げ、油の浮いた黒い水面に長い飛沫を散らす。荷台の幌が跳ね上げられ、番号のない角材の束が無言で滑り込んだ。ディーゼル機関の低い振動が、足元のコンクリートを震わせる。時計を見る。三分の遅れ。闇の奥でヘッドライトが二度瞬き、黒い大型車は泥跳ねを上げて埠頭を滑り去った。あとに残ったのは、重油の焦げた臭いと、岸壁を叩く冷たい波頭の音だけだった。',
      domain: 'novel',
    },
  };

  function compareV3VsV4(sampleKey = 'Sample-01') {
    const sample = CLASSIC_V3_SAMPLES[sampleKey] || CLASSIC_V3_SAMPLES['Sample-01'];
    if (!core) return null;

    const geoV3 = core.computeSemanticGeodesicCurvature(sample.v3Text);
    const anchors = core.pinBoundaryAnchors(sample.title, sample.domain);
    const bridge = core.solveBidirectionalCollisionBridge(anchors, { k_candidates: 4 });
    const geoV4 = core.computeSemanticGeodesicCurvature(bridge.welded_text);

    renderGeodesicDOM(geoV4);
    renderBridgeDOM(bridge);

    const compEl = document.getElementById('v3-v4-comparison-view');
    if (compEl) {
      compEl.innerHTML =
        '<table style="width: 100%; border-collapse: collapse; font-size: 10px;">' +
        '<thead><tr style="border-bottom: 1px solid #334155;"><th style="text-align: left; padding: 3px;">世代</th><th style="padding: 3px;">平均曲率 κ</th><th style="padding: 3px;">ブラウン比 η</th><th style="padding: 3px;">判定</th></tr></thead>' +
        '<tbody>' +
        '<tr style="color: #f87171;"><td style="padding: 3px;">v3.0 片方向 (旧)</td><td style="text-align: center;">' + geoV3.mean_curvature.toFixed(3) + '</td><td style="text-align: center;">' + geoV3.brownian_ratio.toFixed(2) + '</td><td>⚠️ 小道具カタログ化</td></tr>' +
        '<tr style="color: #34d399; font-weight: bold;"><td style="padding: 3px;">v4.0 双方向 (新)</td><td style="text-align: center;">' + geoV4.mean_curvature.toFixed(3) + '</td><td style="text-align: center;">' + geoV4.brownian_ratio.toFixed(2) + '</td><td>⚡ 中央衝突溶接</td></tr>' +
        '</tbody></table>';
    }

    return { sample, geoV3, geoV4, bridge };
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
    planBidirectionalBridge: (theme, domain) => {
      const a = core.pinBoundaryAnchors(theme, domain || 'novel');
      const b = core.solveBidirectionalCollisionBridge(a, { k_candidates: 4 });
      renderBridgeDOM(b);
      return b;
    },
    inspectGeodesicCurvature: (text) => {
      const g = core.computeSemanticGeodesicCurvature(text);
      renderGeodesicDOM(g);
      return g;
    },
    compareV3VsV4,
    getState: () => state,
    runMicroGate: (t, phase) => core.runMicroConstraintGate(t, { targetPhase: phase || state.targetPhase })
  };
})(typeof window !== 'undefined' ? window : globalThis);
