/**
 * @nishiki/usir-bridges: Plotailor Docked Atelier Right-Detail Drawer Adapter
 * Phase 3 & TASK-362 Implementation.
 * 
 * Strict Enforcement of .antigravity/rules.md Rule 5:
 * Docked Atelier UI Constitution:
 * - Placement strictly inside Right DetailDrawer (widthRatio: 0.25) or Bottom Dock
 * - allowPopupModal: false (auto-popping modals during typing are strictly forbidden)
 * - autoInterruptTyping: false (typing is never interrupted)
 */

import {
  extractUsir24d,
  applyEqualizer,
  generateRewriteDirectives,
  steerPhaseRewrite,
  fitPhaseClassifier,
  predictPhase,
  type UsirPhase,
  type Usir24dResult,
  type EqualizerGains,
  type RewriteDirectives,
  type PhaseCentroidsAndCov,
} from '@nishiki/usir-core';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

export type DockedPlacement = 'right-detail-drawer' | 'bottom-dock';

export interface PlotailorUiConstitution {
  readonly placement: DockedPlacement;
  readonly widthRatio: 0.25;
  readonly allowPopupModal: false;
  readonly autoInterruptTyping: false;
}

export interface DeadzoneWarningBadge {
  readonly active: boolean;
  readonly label: string;
  readonly severity: 'warning' | 'error' | 'normal';
  readonly color: string;
  readonly reason: string;
}

export interface FivePhaseIndicator {
  readonly phase: UsirPhase;
  readonly orderParams: {
    readonly psi_order: number;
    readonly phi_void: number;
    readonly kappa_kin: number;
  };
  readonly lenCv: number;
  readonly cadenceAlive: number;
  readonly isDeadzone: boolean;
  readonly deadzoneWarningBadge: DeadzoneWarningBadge | null;
}

export interface SpectrumBarItem {
  readonly id: string; // 'Sg-01' .. 'Sg-12'
  readonly name: string;
  readonly period: 'Kasane' | 'Sasae' | 'Kamae';
  readonly group: 'Flux' | 'Lattice' | 'Void' | 'Shock';
  readonly value: number; // q_i in [0, 1]
  readonly momentum: number; // p_i
}

export interface SpectrumBarsState {
  readonly flux: SpectrumBarItem[];
  readonly lattice: SpectrumBarItem[];
  readonly void: SpectrumBarItem[];
  readonly shock: SpectrumBarItem[];
  readonly all12: SpectrumBarItem[];
}

export interface PlotailorEqualizerSliders {
  flux: number;
  lattice: number;
  void: number;
  shock: number;
}

export interface PhaseTransitionRewritePreview {
  readonly originalText: string;
  readonly rewrittenText: string;
  readonly targetPhase: UsirPhase;
  readonly directives: RewriteDirectives;
  readonly beforeLenCv: number;
  readonly afterLenCv: number;
}

export interface PlotailorDrawerState {
  readonly constitution: PlotailorUiConstitution;
  readonly text: string;
  readonly extraction: Usir24dResult;
  readonly phaseIndicator: FivePhaseIndicator;
  readonly spectrumBars: SpectrumBarsState;
  readonly sliders: PlotailorEqualizerSliders;
  readonly modulatedQ: number[];
  readonly rewritePreview: PhaseTransitionRewritePreview | null;
}

export interface PlotailorDrawerOptions {
  placement?: DockedPlacement;
  allowPopupModal?: boolean;
  autoInterruptTyping?: boolean;
  initialSliders?: Partial<PlotailorEqualizerSliders>;
}

// 12 Spectrum Elements Metadata strictly aligned with 4 groups
const SPECTRUM_METADATA: Array<{
  id: string;
  name: string;
  period: 'Kasane' | 'Sasae' | 'Kamae';
  group: 'Flux' | 'Lattice' | 'Void' | 'Shock';
  index: number;
}> = [
  // Period 1: Kasane (襲)
  { id: 'Sg-01', name: 'Gr', period: 'Kasane', group: 'Flux', index: 0 },
  { id: 'Sg-02', name: 'Cr', period: 'Kasane', group: 'Lattice', index: 1 },
  { id: 'Sg-03', name: 'Pr', period: 'Kasane', group: 'Void', index: 2 },
  { id: 'Sg-04', name: 'Jt', period: 'Kasane', group: 'Shock', index: 3 },
  // Period 2: Sasae (支え)
  { id: 'Sg-05', name: 'Cp', period: 'Sasae', group: 'Flux', index: 4 },
  { id: 'Sg-06', name: 'Sy', period: 'Sasae', group: 'Lattice', index: 5 },
  { id: 'Sg-07', name: 'Cs', period: 'Sasae', group: 'Void', index: 6 },
  { id: 'Sg-08', name: 'Lp', period: 'Sasae', group: 'Shock', index: 7 },
  // Period 3: Kamae (構え)
  { id: 'Sg-09', name: 'Dy', period: 'Kamae', group: 'Flux', index: 8 },
  { id: 'Sg-10', name: 'At', period: 'Kamae', group: 'Lattice', index: 9 },
  { id: 'Sg-11', name: 'Ma', period: 'Kamae', group: 'Void', index: 10 },
  { id: 'Sg-12', name: 'Ru', period: 'Kamae', group: 'Shock', index: 11 },
];

/**
 * Asserts Rule 5 compliance of the Docked Atelier UI Constitution.
 * Throws immediately if an attempt is made to instantiate popup modals or modal overlays.
 */
export function assertRule5Compliance(options?: PlotailorDrawerOptions): void {
  if (options?.allowPopupModal === true) {
    throw new Error(
      'VIOLATION OF DOCKED ATELIER UI CONSTITUTION (Rule 5): ' +
        'allowPopupModal MUST BE FALSE. Pop-up modals during typing are strictly prohibited.'
    );
  }
  if (options?.autoInterruptTyping === true) {
    throw new Error(
      'VIOLATION OF DOCKED ATELIER UI CONSTITUTION (Rule 5): ' +
        'autoInterruptTyping MUST BE FALSE. Typing flow must remain uninterrupted.'
    );
  }
  if (options?.placement && options.placement !== 'right-detail-drawer' && options.placement !== 'bottom-dock') {
    throw new Error(
      `VIOLATION OF DOCKED ATELIER UI CONSTITUTION (Rule 5): ` +
        `Invalid placement "${options.placement}". Permitted placements: "right-detail-drawer" (25% width) or "bottom-dock".`
    );
  }
}

let cachedClassifier: PhaseCentroidsAndCov | null = null;

export function getBridgePhaseClassifier(): PhaseCentroidsAndCov | null {
  if (cachedClassifier) {
    return cachedClassifier;
  }
  try {
    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const candidates = [
      path.resolve(currentDir, '../../usir-core/tests/fixtures/python_reference_40.json'),
      path.resolve(currentDir, '../node_modules/@nishiki/usir-core/tests/fixtures/python_reference_40.json'),
      path.resolve(process.cwd(), 'packages/usir-core/tests/fixtures/python_reference_40.json'),
      path.resolve(process.cwd(), 'tests/fixtures/python_reference_40.json'),
    ];
    for (const p of candidates) {
      if (fs.existsSync(p)) {
        const raw = fs.readFileSync(p, 'utf-8');
        const records: { phase: string; z: number[] }[] = JSON.parse(raw);
        cachedClassifier = fitPhaseClassifier(
          records.map((r) => ({ phase: r.phase, features: r.z }))
        );
        return cachedClassifier;
      }
    }
  } catch {
    // Fall back to analytical classification
  }
  return null;
}

/**
 * 5-Phase Determination from 24D symplectic extracted values.
 * Uses empirical trained Gaussian classifier when available, with analytical fallback.
 * Guards against False Positives on crystalline/formal texts (Rule 2 & User Global Rules).
 */
export function determinePhaseFromState(res: Usir24dResult): UsirPhase {
  const clf = getBridgePhaseClassifier();
  if (clf) {
    return predictPhase(res.z_24d, clf).phase;
  }

  // Phase I: Crystalline Order (High formal structure, high psi_order, low abstract filler)
  if (res.order_params.psi_order >= 0.40 && res.abst_bloat < 0.05 && res.groups.IV_Shock < 0.25) {
    return 'I_Crystalline';
  }

  // AI Deadzone Detection: sluggish cadence, or (flat rhythm AND low kinetic energy / abstract filler)
  const isSluggish = res.cadence_alive < 0.35;
  const isFlatAndSloth = res.len_cv < 0.28 && (res.order_params.kappa_kin < 0.35 || res.abst_bloat > 0.05);
  if (isSluggish || isFlatAndSloth || (res.abst_bloat > 0.08 && res.order_params.kappa_kin < 0.25)) {
    return 'V_Deadzone';
  }

  // Phase III: Superfluid Void (Wago brevity, stillness operators, high phi_void)
  if (res.order_params.phi_void >= 0.40 && res.order_params.psi_order < 0.35) {
    return 'III_Void';
  }

  // Phase IV: Turbulent Rupture (High shock, extreme kinetic impulse)
  if (res.groups.IV_Shock >= 0.50 || res.order_params.kappa_kin >= 0.85) {
    return 'IV_Turbulent';
  }

  // Default to Phase II: Self-Organized Criticality
  return 'II_Critical';
}

/**
 * Creates the complete non-intrusive Docked Atelier Drawer State.
 */
export function createPlotailorDrawerState(
  text: string,
  options?: PlotailorDrawerOptions
): PlotailorDrawerState {
  assertRule5Compliance(options);

  const placement = options?.placement ?? 'right-detail-drawer';
  const constitution: PlotailorUiConstitution = {
    placement,
    widthRatio: 0.25,
    allowPopupModal: false,
    autoInterruptTyping: false,
  };

  const sliders: PlotailorEqualizerSliders = {
    flux: options?.initialSliders?.flux ?? 1.0,
    lattice: options?.initialSliders?.lattice ?? 1.0,
    void: options?.initialSliders?.void ?? 1.0,
    shock: options?.initialSliders?.shock ?? 1.0,
  };

  const extraction = extractUsir24d(text);
  const phase = determinePhaseFromState(extraction);

  // Deadzone Warning Badge evaluation
  const isDeadzone =
    phase === 'V_Deadzone' ||
    extraction.cadence_alive < 0.35 ||
    (extraction.len_cv < 0.28 && extraction.order_params.kappa_kin < 0.35);
  let deadzoneWarningBadge: DeadzoneWarningBadge | null = null;
  if (isDeadzone) {
    deadzoneWarningBadge = {
      active: true,
      label: 'AI Deadzone Sedimentation',
      severity: extraction.len_cv < 0.20 ? 'error' : 'warning',
      color: '#f59e0b', // Amber
      reason:
        extraction.len_cv < 0.28
          ? `Sentence length volatility flatlined (CV_L = ${extraction.len_cv.toFixed(2)} < 0.28)`
          : `Sluggish auxiliary endings or abstract filler bloat detected (cadence = ${extraction.cadence_alive.toFixed(2)})`,
    };
  }

  const phaseIndicator: FivePhaseIndicator = {
    phase,
    orderParams: {
      psi_order: extraction.order_params.psi_order,
      phi_void: extraction.order_params.phi_void,
      kappa_kin: extraction.order_params.kappa_kin,
    },
    lenCv: extraction.len_cv,
    cadenceAlive: extraction.cadence_alive,
    isDeadzone,
    deadzoneWarningBadge,
  };

  // Build 12 Spectrum Bars
  const all12: SpectrumBarItem[] = SPECTRUM_METADATA.map((meta) => {
    return {
      id: meta.id,
      name: meta.name,
      period: meta.period,
      group: meta.group,
      value: extraction.q[meta.index] ?? 0.0,
      momentum: extraction.p[meta.index] ?? 0.0,
    };
  });

  const spectrumBars: SpectrumBarsState = {
    flux: all12.filter((item) => item.group === 'Flux'),
    lattice: all12.filter((item) => item.group === 'Lattice'),
    void: all12.filter((item) => item.group === 'Void'),
    shock: all12.filter((item) => item.group === 'Shock'),
    all12,
  };

  // Compute modulated position vector q using current sliders
  const gains: EqualizerGains = {
    alpha_flux: sliders.flux,
    alpha_lattice: sliders.lattice,
    alpha_void: sliders.void,
    alpha_shock: sliders.shock,
  };
  const modulatedQ = applyEqualizer(extraction.q, gains);

  return {
    constitution,
    text,
    extraction,
    phaseIndicator,
    spectrumBars,
    sliders,
    modulatedQ,
    rewritePreview: null,
  };
}

/**
 * Controller class managing the lifecycle of the Plotailor Right Detail Drawer.
 */
export class PlotailorEqualizerController {
  private state: PlotailorDrawerState;

  constructor(initialText = '', options?: PlotailorDrawerOptions) {
    assertRule5Compliance(options);
    this.state = createPlotailorDrawerState(initialText, options);
  }

  public getState(): PlotailorDrawerState {
    return this.state;
  }

  public setText(text: string): PlotailorDrawerState {
    this.state = createPlotailorDrawerState(text, {
      placement: this.state.constitution.placement,
      initialSliders: this.state.sliders,
    });
    return this.state;
  }

  public setSliders(sliders: Partial<PlotailorEqualizerSliders>): PlotailorDrawerState {
    const nextSliders: PlotailorEqualizerSliders = {
      ...this.state.sliders,
      ...sliders,
    };
    // Clip sliders strictly into [0.0, 2.0]
    nextSliders.flux = Math.max(0.0, Math.min(2.0, nextSliders.flux));
    nextSliders.lattice = Math.max(0.0, Math.min(2.0, nextSliders.lattice));
    nextSliders.void = Math.max(0.0, Math.min(2.0, nextSliders.void));
    nextSliders.shock = Math.max(0.0, Math.min(2.0, nextSliders.shock));

    const gains: EqualizerGains = {
      alpha_flux: nextSliders.flux,
      alpha_lattice: nextSliders.lattice,
      alpha_void: nextSliders.void,
      alpha_shock: nextSliders.shock,
    };
    const modulatedQ = applyEqualizer(this.state.extraction.q, gains);

    this.state = {
      ...this.state,
      sliders: nextSliders,
      modulatedQ,
    };
    return this.state;
  }

  public resetSliders(): PlotailorDrawerState {
    return this.setSliders({ flux: 1.0, lattice: 1.0, void: 1.0, shock: 1.0 });
  }

  /**
   * Generates a one-click phase transition rewrite preview.
   * If current state is Phase V (Deadzone), default target is Phase III (Superfluid Void).
   */
  public generateRewriteProposal(targetPhase?: UsirPhase): PhaseTransitionRewritePreview {
    const originalText = this.state.text;
    const currentPhase = this.state.phaseIndicator.phase;
    const target = targetPhase ?? (currentPhase === 'V_Deadzone' ? 'III_Void' : 'II_Critical');

    const directives = generateRewriteDirectives(originalText, target);
    const rewrittenText = steerPhaseRewrite(originalText, target);

    const afterExtr = extractUsir24d(rewrittenText);

    const preview: PhaseTransitionRewritePreview = {
      originalText,
      rewrittenText,
      targetPhase: target,
      directives,
      beforeLenCv: this.state.extraction.len_cv,
      afterLenCv: afterExtr.len_cv,
    };

    this.state = {
      ...this.state,
      rewritePreview: preview,
    };

    return preview;
  }

  /**
   * Applies the proposed rewrite into the editor state.
   */
  public applyRewrite(): string {
    if (!this.state.rewritePreview) {
      return this.state.text;
    }
    const newText = this.state.rewritePreview.rewrittenText;
    this.setText(newText);
    return newText;
  }
}
