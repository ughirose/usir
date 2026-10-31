/**
 * @nishiki/usir-bridges: WorldCraft Harmonic Tension-Release Adapter
 * Maps scene ontology tension curves (Jo-Ha-Kyu / 序破急 progression)
 * to target USIR Tier M particles and verifies trajectory consistency.
 */

import {
  extractUsir24d,
  CANONICAL_PARTICLES,
  mixStyleParticles,
  type CanonicalParticle,
  type Usir24dResult,
  type UsirPhase,
} from '@nishiki/usir-core';
import { determinePhaseFromState } from './plotailorEqualizer.js';

export type JoHaKyuStage = 'jo' | 'ha' | 'kyu';

export interface SceneDescriptor {
  id: string;
  text: string;
  stage?: JoHaKyuStage;
  targetTension?: number; // Expected normalized tension in [0.0, 1.0]
}

export interface SceneHarmonicPoint {
  sceneId: string;
  stage: JoHaKyuStage;
  empiricalTension: number;
  targetTension: number;
  phase: UsirPhase;
  lenCv: number;
  cadenceAlive: number;
  kappaKin: number;
  phiVoid: number;
  shockMomentum: number;
  isDeadzoneSlump: boolean;
}

export interface SequenceEvaluationResult {
  trajectory: SceneHarmonicPoint[];
  isConsistent: boolean;
  deadzoneDetected: boolean;
  maxTension: number;
  minTension: number;
  dynamicRange: number;
  recommendations: string[];
}

export class WorldCraftHarmonicBridge {
  /**
   * Calculates empirical dramatic tension in [0.0, 1.0] from 24D symplectic state:
   * Combines kinetic energy (kappa_kin), shock momentum (G_shock), and inverse stillness (1 - phi_void).
   */
  public calculateEmpiricalTension(extraction: Usir24dResult): number {
    const kappa = Math.min(1.0, extraction.order_params.kappa_kin / 1.2);
    const shock = extraction.groups.IV_Shock;
    const inverseVoid = 1.0 - extraction.order_params.phi_void;

    const rawTension = 0.40 * kappa + 0.35 * shock + 0.25 * inverseVoid;
    return Math.max(0.0, Math.min(1.0, rawTension));
  }

  /**
   * Maps a desired tension value in [0.0, 1.0] and stage (Jo-Ha-Kyu)
   * to a continuous Product-of-Experts blended Canonical Style Particle.
   */
  public mapTensionToParticle(
    tension: number,
    stage: JoHaKyuStage = 'ha'
  ): {
    name: string;
    mu_mix: number[];
    precision_mix: number[][];
    sourceParticles: string[];
    weights: number[];
  } {
    const clampedTension = Math.max(0.0, Math.min(1.0, tension));
    const crystal = CANONICAL_PARTICLES['PORCELAIN_CRYSTAL']!;
    const amber = CANONICAL_PARTICLES['AMBER_CRITICAL']!;
    const voidP = CANONICAL_PARTICLES['OBSIDIAN_VOID']!;
    const lightning = CANONICAL_PARTICLES['LIGHTNING_RUPTURE']!;

    let particles: CanonicalParticle[];
    let weights: number[];

    if (stage === 'jo') {
      // Jo (序): Restrained baseline, formal stability and quietness
      // High weight on Porcelain Crystal & Obsidian Void
      const wCrystal = Math.max(0.1, 0.7 - clampedTension * 0.4);
      const wVoid = Math.max(0.1, 0.4 - clampedTension * 0.2);
      const wAmber = clampedTension * 0.4;
      particles = [crystal, voidP, amber];
      weights = [wCrystal, wVoid, wAmber];
    } else if (stage === 'ha') {
      // Ha (破): Development, rising tension, literary criticality
      // High weight on Amber Critical, with rising Lightning Rupture
      const wAmber = Math.max(0.2, 0.8 - clampedTension * 0.3);
      const wLightning = clampedTension * 0.7;
      const wVoid = (1.0 - clampedTension) * 0.2;
      particles = [amber, lightning, voidP];
      weights = [wAmber, wLightning, wVoid];
    } else {
      // Kyu (急): Climax / Sudden resolution
      // Peak shock injection, transitioning to intense resolution
      if (clampedTension >= 0.7) {
        particles = [lightning, amber];
        weights = [clampedTension, 1.0 - clampedTension];
      } else {
        // Post-climax resolution towards Void
        particles = [voidP, amber, lightning];
        weights = [0.6, 0.3, 0.1];
      }
    }

    // Normalize weights to sum to 1.0
    const sumW = weights.reduce((acc, w) => acc + w, 0);
    const normWeights = weights.map((w) => w / sumW);

    const mix = mixStyleParticles(particles, normWeights);

    return {
      name: `PoE_Harmonic_${stage.toUpperCase()}_T${clampedTension.toFixed(2)}`,
      mu_mix: mix.mu_mix,
      precision_mix: mix.precision_mix,
      sourceParticles: particles.map((p) => p.name),
      weights: normWeights,
    };
  }

  /**
   * Analyzes an individual scene in the sequence.
   */
  public analyzeScene(
    sceneId: string,
    text: string,
    stage: JoHaKyuStage = 'ha',
    targetTension?: number
  ): SceneHarmonicPoint {
    const extraction = extractUsir24d(text);
    const empiricalTension = this.calculateEmpiricalTension(extraction);
    const phase = determinePhaseFromState(extraction);

    const defaultTarget = stage === 'jo' ? 0.25 : stage === 'ha' ? 0.60 : 0.85;
    const tgt = targetTension ?? defaultTarget;

    const isDeadzoneSlump =
      phase === 'V_Deadzone' ||
      extraction.cadence_alive < 0.35 ||
      (extraction.len_cv < 0.28 && extraction.order_params.kappa_kin < 0.35);

    return {
      sceneId,
      stage,
      empiricalTension,
      targetTension: tgt,
      phase,
      lenCv: extraction.len_cv,
      cadenceAlive: extraction.cadence_alive,
      kappaKin: extraction.order_params.kappa_kin,
      phiVoid: extraction.order_params.phi_void,
      shockMomentum: extraction.groups.IV_Shock,
      isDeadzoneSlump,
    };
  }

  /**
   * Evaluates the tension-release trajectory consistency across a multi-scene sequence.
   */
  public evaluateSequence(scenes: SceneDescriptor[]): SequenceEvaluationResult {
    if (scenes.length === 0) {
      return {
        trajectory: [],
        isConsistent: true,
        deadzoneDetected: false,
        maxTension: 0,
        minTension: 0,
        dynamicRange: 0,
        recommendations: ['Sequence is empty.'],
      };
    }

    const trajectory: SceneHarmonicPoint[] = [];
    const recommendations: string[] = [];
    let deadzoneDetected = false;

    for (let i = 0; i < scenes.length; i++) {
      const scene = scenes[i]!;
      // Default stage progression across sequence if not provided
      const defaultStage: JoHaKyuStage =
        i === 0
          ? 'jo'
          : i === scenes.length - 1
            ? 'kyu'
            : 'ha';
      const stage = scene.stage ?? defaultStage;

      const point = this.analyzeScene(scene.id, scene.text, stage, scene.targetTension);
      trajectory.push(point);

      if (point.isDeadzoneSlump) {
        deadzoneDetected = true;
        recommendations.push(
          `Scene "${point.sceneId}" (${point.stage.toUpperCase()}) exhibits AI Deadzone Slump ` +
            `(CV_L = ${point.lenCv.toFixed(2)}, Cadence = ${point.cadenceAlive.toFixed(2)}). ` +
            `Inject LIGHTNING_RUPTURE catalyst to break rhythmic monotony.`
        );
      }
    }

    const tensions = trajectory.map((p) => p.empiricalTension);
    const maxTension = Math.max(...tensions);
    const minTension = Math.min(...tensions);
    const dynamicRange = maxTension - minTension;

    // Check Jo-Ha-Kyu progression consistency:
    // Dynamic range must be >= 0.15 for healthy dramatic tension
    let isConsistent = dynamicRange >= 0.15 && !deadzoneDetected;

    if (dynamicRange < 0.15) {
      recommendations.push(
        `Dramatic dynamic range is too narrow (${dynamicRange.toFixed(2)} < 0.15). ` +
          `Expand the contrast between Jo exposition stillness and Ha/Kyu climactic rupture.`
      );
    }

    // Trajectory direction check: Jo should generally have lower tension than Ha/Kyu peak
    const joPoints = trajectory.filter((p) => p.stage === 'jo');
    const haKyuPoints = trajectory.filter((p) => p.stage === 'ha' || p.stage === 'kyu');
    if (joPoints.length > 0 && haKyuPoints.length > 0) {
      const avgJo = joPoints.reduce((acc, p) => acc + p.empiricalTension, 0) / joPoints.length;
      const maxHaKyu = Math.max(...haKyuPoints.map((p) => p.empiricalTension));
      if (avgJo > maxHaKyu) {
        isConsistent = false;
        recommendations.push(
          `Inverted progression detected: Jo tension (${avgJo.toFixed(2)}) exceeds Ha/Kyu peak (${maxHaKyu.toFixed(2)}).`
        );
      }
    }

    return {
      trajectory,
      isConsistent,
      deadzoneDetected,
      maxTension,
      minTension,
      dynamicRange,
      recommendations,
    };
  }
}
