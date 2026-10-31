import { describe, expect, it } from 'vitest';
import {
  createPlotailorDrawerState,
  PlotailorEqualizerController,
  assertRule5Compliance,
  WorldCraftHarmonicBridge,
  NanoStyleRingBuffer,
  NANO_FRAME_BYTE_LENGTH,
  NANO_RING_MAGIC,
} from '../src/index.js';
import type { UsirState } from '@nishiki/usir-core';

describe('@nishiki/usir-bridges Comprehensive Test Suite (TASK-362 & Ecosystem)', () => {
  // Sample literary texts
  const SOUSEKI_TEXT =
    '吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。';

  const AI_DEADZONE_TEXT =
    '人工知能の発展に伴い、多様なシステムにおいて効率的な処理が実現されている状況であると言える。自然言語処理技術の向上により、様々な分野において高度な自動化が進められている傾向が見受けられる。今後の展望としても、継続的な最適化が行われることが期待されると考えられるのである。このような動向を踏まえ、適切な対応策を講じることが重要であると思われる。';

  // ===========================================================================
  // SECTION 1: PLOTAILOR DOCKED ATELIER EQUALIZER (TASK-362)
  // ===========================================================================
  describe('PlotailorEqualizer (TASK-362)', () => {
    it('strictly enforces Rule 5 (Docked Atelier UI Constitution)', () => {
      // Default state complies with Rule 5
      const state = createPlotailorDrawerState(SOUSEKI_TEXT);
      expect(state.constitution.placement).toBe('right-detail-drawer');
      expect(state.constitution.widthRatio).toBe(0.25);
      expect(state.constitution.allowPopupModal).toBe(false);
      expect(state.constitution.autoInterruptTyping).toBe(false);

      // Bottom dock is also permitted
      const bottomState = createPlotailorDrawerState(SOUSEKI_TEXT, {
        placement: 'bottom-dock',
      });
      expect(bottomState.constitution.placement).toBe('bottom-dock');

      // Attempting to violate Rule 5 MUST throw
      expect(() => {
        assertRule5Compliance({ allowPopupModal: true });
      }).toThrow(/VIOLATION OF DOCKED ATELIER UI CONSTITUTION/);

      expect(() => {
        assertRule5Compliance({ autoInterruptTyping: true });
      }).toThrow(/VIOLATION OF DOCKED ATELIER UI CONSTITUTION/);

      expect(() => {
        assertRule5Compliance({ placement: 'modal' as any });
      }).toThrow(/VIOLATION OF DOCKED ATELIER UI CONSTITUTION/);

      expect(() => {
        new PlotailorEqualizerController(SOUSEKI_TEXT, { allowPopupModal: true });
      }).toThrow();
    });

    it('computes 5-phase indicator and flags AI deadzone with warning badge', () => {
      // Healthy text
      const controllerHealthy = new PlotailorEqualizerController(SOUSEKI_TEXT);
      const healthyState = controllerHealthy.getState();
      expect(healthyState.phaseIndicator.phase).not.toBe('V_Deadzone');
      expect(healthyState.phaseIndicator.deadzoneWarningBadge).toBeNull();
      expect(healthyState.phaseIndicator.isDeadzone).toBe(false);

      // Deadzone AI slop text
      const controllerDeadzone = new PlotailorEqualizerController(AI_DEADZONE_TEXT);
      const deadState = controllerDeadzone.getState();
      expect(deadState.phaseIndicator.isDeadzone).toBe(true);
      expect(deadState.phaseIndicator.deadzoneWarningBadge).not.toBeNull();
      expect(deadState.phaseIndicator.deadzoneWarningBadge?.active).toBe(true);
      expect(deadState.phaseIndicator.deadzoneWarningBadge?.color).toBe('#f59e0b');
    });

    it('organizes 12 spectrum bars into 4 orthogonal groups across 3 periods', () => {
      const controller = new PlotailorEqualizerController(SOUSEKI_TEXT);
      const { spectrumBars } = controller.getState();

      expect(spectrumBars.all12.length).toBe(12);
      expect(spectrumBars.flux.length).toBe(3);
      expect(spectrumBars.lattice.length).toBe(3);
      expect(spectrumBars.void.length).toBe(3);
      expect(spectrumBars.shock.length).toBe(3);

      // Check element IDs
      expect(spectrumBars.flux.map((i) => i.id)).toEqual(['Sg-01', 'Sg-05', 'Sg-09']);
      expect(spectrumBars.lattice.map((i) => i.id)).toEqual(['Sg-02', 'Sg-06', 'Sg-10']);
      expect(spectrumBars.void.map((i) => i.id)).toEqual(['Sg-03', 'Sg-07', 'Sg-11']);
      expect(spectrumBars.shock.map((i) => i.id)).toEqual(['Sg-04', 'Sg-08', 'Sg-12']);

      // Check values and momenta
      for (const bar of spectrumBars.all12) {
        expect(bar.value).toBeGreaterThanOrEqual(0.0);
        expect(bar.value).toBeLessThanOrEqual(1.0);
        expect(bar.momentum).toBeGreaterThanOrEqual(0.0);
      }
    });

    it('modulates state via 4-group gain sliders within [0.0, 2.0]', () => {
      const controller = new PlotailorEqualizerController(SOUSEKI_TEXT);
      const initialQ = controller.getState().extraction.q;

      // Boost void slider to 1.8 and shock to 1.5
      controller.setSliders({ void: 1.8, shock: 1.5 });
      let state = controller.getState();
      expect(state.sliders.void).toBe(1.8);
      expect(state.sliders.shock).toBe(1.5);
      expect(state.sliders.flux).toBe(1.0);

      // Modulated Void coordinates (q[2], q[6], q[10]) should scale up
      expect(state.modulatedQ[2]).toBeGreaterThanOrEqual(initialQ[2]!);
      expect(state.modulatedQ[6]).toBeGreaterThanOrEqual(initialQ[6]!);

      // Extreme values are clipped to [0.0, 2.0]
      controller.setSliders({ flux: 3.5, lattice: -1.0 });
      state = controller.getState();
      expect(state.sliders.flux).toBe(2.0);
      expect(state.sliders.lattice).toBe(0.0);

      // Reset sliders
      controller.resetSliders();
      expect(controller.getState().sliders).toEqual({
        flux: 1.0,
        lattice: 1.0,
        void: 1.0,
        shock: 1.0,
      });
    });

    it('generates one-click phase transition rewrite preview and applies it', () => {
      const controller = new PlotailorEqualizerController(AI_DEADZONE_TEXT);
      const preview = controller.generateRewriteProposal('III_Void');

      expect(preview.originalText).toBe(AI_DEADZONE_TEXT);
      expect(preview.targetPhase).toBe('III_Void');
      expect(preview.directives.mandatoryOperators.length).toBeGreaterThan(0);
      expect(preview.rewrittenText).not.toBe(AI_DEADZONE_TEXT);

      // Apply rewrite
      const appliedText = controller.applyRewrite();
      expect(appliedText).toBe(preview.rewrittenText);
      expect(controller.getState().text).toBe(preview.rewrittenText);
    });
  });

  // ===========================================================================
  // SECTION 2: WORLDCRAFT HARMONIC TENSION-RELEASE ADAPTER
  // ===========================================================================
  describe('WorldCraftHarmonicBridge', () => {
    const bridge = new WorldCraftHarmonicBridge();

    it('maps tension curves to PoE blended canonical style particles', () => {
      // Jo: low tension
      const joParticle = bridge.mapTensionToParticle(0.2, 'jo');
      expect(joParticle.mu_mix.length).toBe(24);
      expect(joParticle.precision_mix.length).toBe(24);
      expect(joParticle.sourceParticles).toContain('PORCELAIN_CRYSTAL');

      // Ha: medium tension
      const haParticle = bridge.mapTensionToParticle(0.65, 'ha');
      expect(haParticle.sourceParticles).toContain('AMBER_CRITICAL');

      // Kyu: high tension
      const kyuParticle = bridge.mapTensionToParticle(0.9, 'kyu');
      expect(kyuParticle.sourceParticles).toContain('LIGHTNING_RUPTURE');
    });

    it('evaluates tension-release trajectory consistency across multi-scene sequence', () => {
      const healthySequence = [
        {
          id: 'scene-01',
          text: '静かな朝だった。庭に白い霜が降りていた。誰もいない縁側で、湯呑みの湯気だけがゆっくりと立っていた。ただそれだけの時間があった。',
          stage: 'jo' as const,
        },
        {
          id: 'scene-02',
          text: 'だが、裏山が崩れた。突如として大地が激しい地鳴りを立てて震動し、黒い土煙が天を衝く。息を呑む暇すらなかった。瓦礫が一斉に崩落してきた。',
          stage: 'ha' as const,
        },
        {
          id: 'scene-03',
          text: '飛び退く！紙一重で梁をかわし、叫び声をあげながら広場へ疾走した。崩壊の轟音が背後を覆い尽くした。',
          stage: 'kyu' as const,
        },
      ];

      const evalResult = bridge.evaluateSequence(healthySequence);
      expect(evalResult.trajectory.length).toBe(3);
      expect(evalResult.dynamicRange).toBeGreaterThanOrEqual(0.15);
      expect(evalResult.deadzoneDetected).toBe(false);
      expect(evalResult.isConsistent).toBe(true);

      // Verify that Jo tension < Kyu tension
      const joTension = evalResult.trajectory[0]!.empiricalTension;
      const kyuTension = evalResult.trajectory[2]!.empiricalTension;
      expect(joTension).toBeLessThan(kyuTension);
    });

    it('flags AI deadzone slump in multi-scene sequence', () => {
      const slumpSequence = [
        {
          id: 'scene-01',
          text: SOUSEKI_TEXT,
          stage: 'jo' as const,
        },
        {
          id: 'scene-02',
          text: AI_DEADZONE_TEXT,
          stage: 'ha' as const,
        },
      ];

      const evalResult = bridge.evaluateSequence(slumpSequence);
      expect(evalResult.deadzoneDetected).toBe(true);
      expect(evalResult.isConsistent).toBe(false);
      expect(evalResult.recommendations.some((r) => r.includes('AI Deadzone Slump'))).toBe(true);
    });
  });

  // ===========================================================================
  // SECTION 3: NARRATIVE-NANO SPSC RING BUFFER BRIDGE
  // ===========================================================================
  describe('NanoStyleRingBuffer', () => {
    it('initializes with correct header layout and frame size', () => {
      const ring = new NanoStyleRingBuffer(16);
      expect(ring.capacity).toBe(16);
      expect(ring.isEmpty()).toBe(true);
      expect(ring.isFull()).toBe(false);
      expect(ring.availableRead()).toBe(0);
      expect(ring.availableWrite()).toBe(16);

      const raw = ring.getRawBuffer();
      const u32 = new Uint32Array(raw, 0, 4);
      expect(u32[0]).toBe(NANO_RING_MAGIC);
      expect(u32[1]).toBe(16);
      expect(u32[2]).toBe(NANO_FRAME_BYTE_LENGTH);
    });

    it('writes and reads 128B frames with zero allocation support', () => {
      const ring = new NanoStyleRingBuffer(8);
      const frame1 = new Uint8Array(128);
      frame1[0] = 0xaa;
      frame1[127] = 0x55;

      const frame2 = new Uint8Array(128);
      frame2[0] = 0xbb;
      frame2[127] = 0x66;

      expect(ring.writeFrame(frame1)).toBe(true);
      expect(ring.writeFrame(frame2)).toBe(true);
      expect(ring.availableRead()).toBe(2);
      expect(ring.availableWrite()).toBe(6);

      // Reusable scratch buffer for zero allocation reading
      const outScratch = new Uint8Array(128);
      const read1 = ring.readFrame(outScratch);
      expect(read1).toBe(outScratch);
      expect(read1![0]).toBe(0xaa);
      expect(read1![127]).toBe(0x55);

      const read2 = ring.readFrame();
      expect(read2).not.toBeNull();
      expect(read2![0]).toBe(0xbb);
      expect(read2![127]).toBe(0x66);

      expect(ring.isEmpty()).toBe(true);
      expect(ring.readFrame()).toBeNull();
    });

    it('performs round-trip streaming of Tier S binary structures', () => {
      const ring = new NanoStyleRingBuffer(4);

      const originalState: UsirState = {
        q: [0.6, 0.7, 0.4, 0.2, 0.5, 0.8, 0.3, 0.1, 0.7, 0.9, 0.2, 0.3],
        p: [0.3, 0.2, 0.5, 0.4, 0.6, 0.1, 0.3, 0.7, 0.4, 0.5, 0.2, 0.6],
        order_params: {
          psi_order: 0.45,
          phi_void: 0.52,
          kappa_kin: 0.38,
        },
        len_cv: 0.48,
      };

      expect(ring.writeTierS(originalState)).toBe(true);
      expect(ring.availableRead()).toBe(1);

      const decoded = ring.readTierS();
      expect(decoded).not.toBeNull();
      expect(decoded!.tier).toBe(1); // Tier S = 1
      expect(decoded!.len_cv).toBeCloseTo(0.48, 4);
      expect(decoded!.order_params.psi_order).toBeCloseTo(0.45, 4);
      expect(decoded!.order_params.phi_void).toBeCloseTo(0.52, 4);
      expect(decoded!.order_params.kappa_kin).toBeCloseTo(0.38, 4);

      for (let i = 0; i < 12; i++) {
        expect(decoded!.q[i]).toBeCloseTo(originalState.q[i]!, 4);
        expect(decoded!.p[i]).toBeCloseTo(originalState.p[i]!, 4);
      }
    });

    it('handles queue full and wraparound correctly', () => {
      const ring = new NanoStyleRingBuffer(3);
      const frame = new Uint8Array(128);

      expect(ring.writeFrame(frame)).toBe(true);
      expect(ring.writeFrame(frame)).toBe(true);
      expect(ring.writeFrame(frame)).toBe(true);
      expect(ring.isFull()).toBe(true);
      expect(ring.writeFrame(frame)).toBe(false); // Refused when full

      // Read one, then write one (wraparound)
      expect(ring.readFrame()).not.toBeNull();
      expect(ring.isFull()).toBe(false);
      expect(ring.writeFrame(frame)).toBe(true);

      ring.clear();
      expect(ring.isEmpty()).toBe(true);
      expect(ring.availableRead()).toBe(0);
    });
  });
});
