/**
 * 3-Mode HSMM Multi-Well Director, Momentum Banking & 3,000-Char Longform Generation.
 *
 * Phase B WS-B2 implementation based on Blueprint Section 3:
 * 1. 3 Phase Wells:
 *    - Well 0 (Narrative / Crystal-Void): CV_L = 0.48, phi_void = 0.65, kappa_kin = 0.20
 *    - Well 1 (Dialogue / Polyphony): CV_L = 0.55, Delta_voice >= 0.25, kappa_kin = 0.40
 *    - Well 2 (Climax / Critical-Shock): CV_L = 0.65, kappa_kin = 0.70
 * 2. Transition Matrix T in R^{3x3} with Refractory Period Guard.
 * 3. OPEN-09 Momentum Banking:
 *    - Unreleased momentum accumulation: Delta p_banked <- Delta p_banked + max(0, p_k^* - hat{p}_k)
 *    - Climax burst release: p_climax^* <- p_climax^* + Delta p_banked
 * 4. OPEN-10 Wundt-Berlyne Inverted-U Utility & Refractory Period Gate:
 *    - Blocks consecutive Climax transitions (Purple Prose suppression).
 *    - Forces transition to Well 0 (Narrative / Void - 残心).
 * 5. generateLongform:
 *    - Produces 6-8 paragraphs totaling ~3000 chars without sinking into Deadzone (CV_L < 0.28).
 *    - Maintains healthy CV_L in [0.40, 0.70].
 */

import {
  AMBER_CRITICAL_MU,
  OBSIDIAN_VOID_MU,
  PORCELAIN_CRYSTAL_MU,
} from '../presets/canonicalParticles.js';
import {
  splitSentences,
  stdPop,
  mean,
} from '../sensor/dictionaries.js';
import {
  extractUsir24d,
  type Usir24dResult,
} from '../sensor/symplectic24d.js';
import { type Matrix, type WorldCraftArbitrator } from './worldcraftArbitrator.js';

export interface PhaseWell {
  index: number;
  name: string;
  targetCvL: number;
  targetPhiVoid: number;
  targetKappaKin: number;
  deltaVoice: number;
  targetQ: number[];
  targetP: number[];
}

export const BASE_P_NARRATIVE: readonly number[] = [
  0.15, 0.12, 0.18, 0.14, 0.15, 0.10,
  0.16, 0.12, 0.14, 0.18, 0.11, 0.15,
];

export const BASE_P_DIALOGUE: readonly number[] = [
  0.35, 0.38, 0.28, 0.40, 0.32, 0.36,
  0.42, 0.30, 0.38, 0.34, 0.36, 0.41,
];

export const BASE_P_CLIMAX: readonly number[] = [
  0.68, 0.72, 0.65, 0.75, 0.70, 0.62,
  0.78, 0.69, 0.71, 0.74, 0.66, 0.70,
];

export const DEFAULT_HSMM_TRANSITION_MATRIX: readonly (readonly number[])[] = [
  [0.40, 0.40, 0.20], // Row 0: Narrative
  [0.30, 0.30, 0.40], // Row 1: Dialogue
  [1.00, 0.00, 0.00], // Row 2: Climax -> Narrative (Refractory gate)
];

export function createDefaultWells(dim = 12): Record<number, PhaseWell> {
  const qNarrative = OBSIDIAN_VOID_MU.slice(0, dim);
  const qDialogue = PORCELAIN_CRYSTAL_MU.slice(0, dim);
  const qClimax = AMBER_CRITICAL_MU.slice(0, dim);

  const pNarrative = Array.from(BASE_P_NARRATIVE.slice(0, dim));
  const pDialogue = Array.from(BASE_P_DIALOGUE.slice(0, dim));
  const pClimax = Array.from(BASE_P_CLIMAX.slice(0, dim));

  return {
    0: {
      index: 0,
      name: 'Narrative',
      targetCvL: 0.48,
      targetPhiVoid: 0.65,
      targetKappaKin: 0.20,
      deltaVoice: 0.05,
      targetQ: qNarrative,
      targetP: pNarrative,
    },
    1: {
      index: 1,
      name: 'Dialogue',
      targetCvL: 0.55,
      targetPhiVoid: 0.35,
      targetKappaKin: 0.40,
      deltaVoice: 0.25,
      targetQ: qDialogue,
      targetP: pDialogue,
    },
    2: {
      index: 2,
      name: 'Climax',
      targetCvL: 0.65,
      targetPhiVoid: 0.15,
      targetKappaKin: 0.70,
      deltaVoice: 0.10,
      targetQ: qClimax,
      targetP: pClimax,
    },
  };
}

export function computeWundtBerlyneUtility(arousal: number, aStar = 0.60): number {
  if (!Number.isFinite(arousal) || arousal <= 0 || !Number.isFinite(aStar) || aStar <= 0) return 0.0;
  const ratio = arousal / aStar;
  return Math.E * ratio * Math.exp(-ratio);
}

export function splitSentencesWithFallback(text: string): string[] {
  if (!text) return [];
  const segs = text.split(/(?<=[。！？!?\n」』])/);
  const sents = segs.map((s) => s.trim()).filter((s) => s.length >= 2);
  if (sents.length >= 2) return sents;
  const clauseSegs = text.split(/[、,。！？!?\n」』]+/);
  const clauses = clauseSegs.map((s) => s.trim()).filter((s) => s.length >= 2);
  return clauses.length >= 2 ? clauses : (sents.length > 0 ? sents : (text.trim() ? [text.trim()] : []));
}

export function computeCvL(textOrLengths: string | number[] | string[]): number {
  let lengths: number[];
  if (typeof textOrLengths === 'string') {
    const sents = splitSentencesWithFallback(textOrLengths);
    if (sents.length < 2) return 0.0;
    lengths = sents.map((s) => s.length);
  } else {
    if (textOrLengths.length < 2) return 0.0;
    lengths = textOrLengths.map((x) => (typeof x === 'string' ? x.length : Number(x)));
  }
  const meanLen = mean(lengths);
  if (meanLen <= 1e-6) return 0.0;
  const std = stdPop(lengths);
  return std / meanLen;
}

export interface LongformParagraphRecord {
  paragraphIndex: number;
  wellIndex: number;
  wellName: string;
  text: string;
  charCount: number;
  sentenceCount: number;
  cvL: number;
  targetCvL: number;
  wundtBerlyneUtility: number;
  bankedEnergyBefore: number;
  wasDischarged: boolean;
  dischargedMomentumNorm: number;
  accumulatedDeltaNorm: number;
  refractoryCounterAfter: number;
}

export interface LongformResult {
  fullText: string;
  totalChars: number;
  paragraphCount: number;
  paragraphs: LongformParagraphRecord[];
  wells: number[];
  overallCvL: number;
  inDeadzone: boolean;
  isHealthy: boolean;
  bankedHistory: number[];
  finalBankedEnergy: number;
}

export class HSMMLongformDirector {
  public readonly dim: number;
  public readonly wells: Record<number, PhaseWell>;
  public transitionMatrix: number[][];
  public readonly refractoryCooldown: number;
  public pBanked: number[];
  public refractoryCounter: number;
  /** LCG state when seeded (copied by planWells); null -> Math.random. */
  private rngState: number | null;
  public readonly climaxBaseShock: number;
  public readonly climaxBaseFlux: number;
  public readonly climaxReleaseFrac: number;
  public readonly arbitrator: WorldCraftArbitrator | null;
  public readonly sceneConstraints: Record<number, string[]>;

  constructor(options?: {
    arbitrator?: WorldCraftArbitrator | null;
    sceneConstraints?: Record<number, readonly string[]>;
    dim?: number;
    wells?: Record<number, PhaseWell>;
    transitionMatrix?: number[][];
    refractoryCooldown?: number;
    seed?: number;
    /** Climax base momentum: +shock on Shock [3, 7, 11], -flux on Flux [0, 4, 8] (default 0.25 / 0.20). */
    climaxBaseShock?: number;
    climaxBaseFlux?: number;
    /** Share of the bank released at a Climax that is followed by another Climax (default 0.7). */
    climaxReleaseFrac?: number;
  }) {
    this.climaxBaseShock = options?.climaxBaseShock ?? 0.25;
    this.climaxBaseFlux = options?.climaxBaseFlux ?? 0.2;
    this.climaxReleaseFrac = options?.climaxReleaseFrac ?? 0.7;
    this.dim = options?.dim ?? 12;
    this.wells = options?.wells ?? createDefaultWells(this.dim);
    this.refractoryCooldown = options?.refractoryCooldown ?? 1;

    if (options?.transitionMatrix) {
      this.transitionMatrix = options.transitionMatrix.map((row) => {
        const sum = row.reduce((a, b) => a + b, 0);
        return sum > 0 ? row.map((v) => v / sum) : row.map(() => 1 / row.length);
      });
    } else {
      this.transitionMatrix = DEFAULT_HSMM_TRANSITION_MATRIX.map((row) => [...row]);
    }

    this.pBanked = new Array(this.dim).fill(0.0);
    this.refractoryCounter = 0;
    this.arbitrator = options?.arbitrator ?? null;
    this.sceneConstraints = {};
    for (const [k, modes] of Object.entries(options?.sceneConstraints ?? {})) {
      this.sceneConstraints[Number(k)] = [...modes];
    }

    this.rngState = options?.seed !== undefined ? options.seed >>> 0 : null;
  }

  private rng(): number {
    if (this.rngState === null) return Math.random();
    this.rngState = (Math.imul(1664525, this.rngState) + 1013904223) >>> 0;
    return this.rngState / 4294967296;
  }

  /** Well sequence generateLongform will visit; computed on a copy of the RNG / refractory state (no side effects). */
  public planWells(nParas: number): number[] {
    const savedState = this.rngState;
    const savedRefr = this.refractoryCounter;
    const wells = [0];
    try {
      let cur = 0;
      for (let k = 1; k < nParas; k++) {
        cur = this.stepTransition(cur);
        wells.push(cur);
      }
    } finally {
      this.rngState = savedState;
      this.refractoryCounter = savedRefr;
    }
    return wells;
  }

  /** Fixed Climax base momentum p_climax_base (Shock +, Flux -). */
  public climaxBaseMomentum(): number[] {
    const base = new Array(this.dim).fill(0);
    for (const i of [3, 7, 11]) if (i < this.dim) base[i] = this.climaxBaseShock;
    for (const i of [0, 4, 8]) if (i < this.dim) base[i] = -this.climaxBaseFlux;
    return base;
  }

  public isInRefractory(): boolean {
    return this.refractoryCounter > 0;
  }

  public stepTransition(currentWellIdx: number): number {
    if (currentWellIdx === 2) {
      // Climax just executed -> activate refractory cooldown, force immediate return to Well 0
      this.refractoryCounter = Math.max(1, this.refractoryCooldown);
      return 0;
    }

    const row = [...this.transitionMatrix[currentWellIdx]!];
    if (this.refractoryCounter > 0) {
      this.refractoryCounter -= 1;
      row[2] = 0.0;
      const sum = row.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        for (let j = 0; j < row.length; j++) row[j] = (row[j] ?? 0) / sum;
      } else {
        row[0] = 1.0;
      }
    }

    // Cumulative distribution sampling
    const r = this.rng();
    let cum = 0.0;
    let nextIdx = 0;
    for (let j = 0; j < row.length; j++) {
      cum += row[j]!;
      if (r <= cum || j === row.length - 1) {
        nextIdx = j;
        break;
      }
    }

    return nextIdx;
  }

  public accumulateUnreleasedMomentum(targetP: number[], measuredP: number[]): number[] {
    const delta = new Array(this.dim).fill(0);
    for (let i = 0; i < this.dim; i++) {
      const tp = targetP[i] ?? 0.0;
      const mp = measuredP[i] ?? 0.0;
      const d = Math.max(0.0, tp - mp);
      delta[i] = d;
      this.pBanked[i] = (this.pBanked[i] ?? 0.0) + d;
    }
    return delta;
  }

  /**
   * discharged = releaseFrac * p_banked + p_climax_base, releaseFrac = climaxReleaseFrac when another Climax follows
   * in the paragraph plan (the remaining (1 - releaseFrac) * p_banked is carried forward), else 1.
   * applyClimaxBase = false restores the previous behaviour (discharged = p_banked, full release).
   */
  public dischargeMomentum(
    baseClimaxP: number[],
    laterClimax = false,
    applyClimaxBase = true,
  ): { boostedP: number[]; dischargedDeltaP: number[] } {
    const rf = laterClimax && applyClimaxBase ? this.climaxReleaseFrac : 1.0;
    const base = applyClimaxBase ? this.climaxBaseMomentum() : new Array(this.dim).fill(0);
    const boostedP = new Array(this.dim).fill(0);
    const dischargedDeltaP = new Array(this.dim).fill(0);
    for (let i = 0; i < this.dim; i++) {
      const released = rf * (this.pBanked[i] ?? 0.0);
      dischargedDeltaP[i] = released + base[i]!;
      boostedP[i] = (baseClimaxP[i] ?? 0.0) + dischargedDeltaP[i]!;
      this.pBanked[i] = rf >= 1.0 ? 0.0 : (this.pBanked[i] ?? 0.0) - released;
    }
    return { boostedP, dischargedDeltaP };
  }

  /**
   * WorldCraft null-space arbitration of a paragraph target momentum (Python arbitrate_scene_momentum).
   * The component cut by the constraints is banked: p_banked += max(0, (I - P_null) p).
   */
  public arbitrateSceneMomentum(
    targetP: number[],
    paragraphIndex: number,
    contextText = '',
  ): { safeP: number[]; cutP: number[]; modes: string[] } {
    if (!this.arbitrator) return { safeP: [...targetP], cutP: new Array(this.dim).fill(0), modes: [] };
    const has = Object.prototype.hasOwnProperty.call(this.sceneConstraints, paragraphIndex);
    const modes = has ? [...this.sceneConstraints[paragraphIndex]!] : [];
    const jC = this.arbitrator.compileConstraintJacobian(contextText, has ? modes : null);
    const { safeP, cutP } = this.arbitrateWithJacobian(targetP, jC);
    return { safeP, cutP, modes };
  }

  /** Arbitration with an explicit (general) J_C in R^{k x 12}; banks max(0, cut). */
  public arbitrateWithJacobian(targetP: number[], jacobianC: Matrix): { safeP: number[]; cutP: number[] } {
    if (!this.arbitrator || jacobianC.length === 0) {
      return { safeP: [...targetP], cutP: new Array(this.dim).fill(0) };
    }
    const { safe, cut } = this.arbitrator.arbitrateMomentum(targetP.slice(0, this.dim), jacobianC);
    for (let i = 0; i < this.dim; i++) this.pBanked[i] = (this.pBanked[i] ?? 0.0) + Math.max(0.0, cut[i] ?? 0.0);
    return { safeP: safe, cutP: cut };
  }

  public getBankedEnergy(): number {
    let sumSq = 0.0;
    for (let i = 0; i < this.dim; i++) {
      const v = this.pBanked[i] ?? 0.0;
      sumSq += v * v;
    }
    return 0.5 * sumSq;
  }

  public generateStandaloneParagraph(
    well: PhaseWell,
    targetChars = 420,
    paragraphIdx = 0,
    themePrefix = '',
  ): string {
    let units: string[];
    const prefix = paragraphIdx === 0 && themePrefix ? `${themePrefix}。` : '';
    if (well.index === 0) {
      // Narrative / Void: target CV_L ~ 0.48 (grounding short + contemplative long)
      units = [
        `${prefix}静寂が夜の帳を静かに降ろした。`,
        '遠くの山並みは青黒い輪郭だけをかすかに残し、冷涼な夜気が足元をかすめるように流れていく。',
        '風の音はない。',
        'ただ木々の細い葉先が擦れ合う微かな気配だけが、暗がりの中に白く浮き沈みしていた。',
        '石畳の隙間に潜む古い苔は冷たい露を吸い込み、誰の足跡も刻まれないまま深い闇に沈み込んでいる。',
        '時は緩やかに歩みを止めていた。',
        '遙か昔に旅立った者の記憶のように、かすかな月影が濡れた軒端を青白く照らし出し、沈黙の底へと吸い込まれていく。',
        '息をひそめ、ただ耳を澄ます。',
        '夜明けの気配はまだ遠く、地平の彼方にわずかな藍色の階調が漂うばかりだった。',
        '冷えた大気の中に漂う木々の芳香が、研ぎ澄まされた意識を透明な静止の中へといざなう。',
        '水面は鏡のように張りつめていた。',
        '静けさは失われるのではなく、世界の輪郭そのものを確かめるように深まり続けている。',
      ];
    } else if (well.index === 1) {
      // Dialogue / Polyphony: target CV_L ~ 0.55 (short rapid turns + reflective beats)
      units = [
        '「本当にこれでよいと思うのか」と、彼が低い声で問いかけた。',
        '闇の中から響くその抑揚には、隠しきれない焦燥と微かな祈りが綯い交ぜになっていた。',
        '「引き返す道など、最初から存在しない」',
        '彼女は手元の明かりを覆い隠しながら、乾いた唇を小さく結んだ。',
        '「だが、あの境界を越えれば二度と戻れぬぞ」',
        '冷たい風が二人の間を吹き抜け、古い扉の蝶番をきしませる。',
        '互いの視線が暗がりの中で交錯し、幾重にも重なる沈黙の問いが空気を張り詰めさせた。',
        '「進むほかない」と、彼女は短く告げた。',
        '「足音が近づいてくる」と、彼は背後の暗闇に視線を向けた。',
        '息を殺したまま、二人は握りしめた互いの手のぬくもりだけを唯一の信憑としていた。',
        '「迷うな」',
        'その一言が、夜の厚い壁を切り裂くように二人の胸を打った。',
      ];
    } else {
      // Climax / Critical-Shock: target CV_L ~ 0.65 (punchlines + turbulent surging periods)
      units = [
        '閃光が走った。',
        '張り詰めた大気が一瞬にして爆縮し、天地を裂く重低音が石壁の隙間から全身の骨髄へと突き刺さる。',
        '逃げ場はない。',
        '砕け散る硝子の破片が暗夜の空間を鋭利な乱反射で満たし、凍りついた思考を激しい衝撃の波が根底から洗い流していく。',
        '全てが終わる。',
        '絶叫すらも真空の渦に呑み込まれ、ただ熱い拍動だけが胸の奥底で警鐘のように乱打されていた。',
        '光と影が混ざり合う境界で、張り詰めた運命の糸が音を立てて断ち切られ、新たな混沌が幕を開けた。',
        '一瞬の静止。',
        '崩落の轟音が地底から湧き上がり、確固たる世界が足元から激しく傾いていく。',
        '衝動が奔流となって溢れ出し、もはや誰もその激流を押しとどめることはできなかった。',
      ];
    }

    const offset = (paragraphIdx * 7 + 1) % units.length;
    const chosen: string[] = [];
    let curLen = 0;
    let idx = 0;
    while (curLen < targetChars && idx < units.length * 2) {
      const s = units[(offset + idx) % units.length]!;
      chosen.push(s);
      curLen += s.length;
      idx++;
    }

    return chosen.join('');
  }

  public generateLongform(options?: {
    prompt?: string;
    targetParagraphs?: number;
    targetTotalChars?: number;
    /** Climax base momentum + multi-Climax carry (default false = previous full-release behaviour). */
    climaxBaseMomentum?: boolean;
  }): LongformResult {
    const prompt = options?.prompt ?? '静謐な山寺の夜明け';
    const targetParagraphs = Math.max(6, Math.min(8, options?.targetParagraphs ?? 7));
    const targetTotalChars = options?.targetTotalChars ?? 3000;
    const charsPerPara = Math.max(350, Math.floor(targetTotalChars / targetParagraphs));

    const paragraphs: LongformParagraphRecord[] = [];
    const bankedHistory: number[] = [];
    const allSentences: string[] = [];
    const fullParagraphsText: string[] = [];

    let currentWellIdx = 0;
    const applyCb = options?.climaxBaseMomentum ?? false;
    const plannedWells = applyCb ? this.planWells(targetParagraphs) : [];

    for (let k = 0; k < targetParagraphs; k++) {
      const wellIdx = k === 0 ? 0 : this.stepTransition(currentWellIdx);
      currentWellIdx = wellIdx;

      const well = this.wells[wellIdx]!;
      const energyBefore = this.getBankedEnergy();
      bankedHistory.push(energyBefore);

      let effectiveTargetP: number[];
      let dischargedDeltaP: number[];
      let wasDischarged = false;

      if (wellIdx === 2) {
        const laterClimax = applyCb && plannedWells.slice(k + 1).includes(2);
        const dRes = this.dischargeMomentum(well.targetP, laterClimax, applyCb);
        effectiveTargetP = dRes.boostedP;
        dischargedDeltaP = dRes.dischargedDeltaP;
        wasDischarged = true;
      } else {
        effectiveTargetP = [...well.targetP];
        dischargedDeltaP = new Array(this.dim).fill(0);
      }

      const paraText = this.generateStandaloneParagraph(well, charsPerPara, k, prompt);
      fullParagraphsText.push(paraText);
      const paraSents = splitSentences(paraText);
      allSentences.push(...paraSents);

      const paraCvL = computeCvL(paraSents);

      // Symplectic feature extraction via extractUsir24d
      const feat: Usir24dResult = extractUsir24d(paraText);
      const hatP = feat.p.slice(0, this.dim);

      let accumulatedDelta: number[];
      if (wellIdx !== 2) {
        accumulatedDelta = this.accumulateUnreleasedMomentum(effectiveTargetP, hatP);
      } else {
        accumulatedDelta = new Array(this.dim).fill(0);
      }

      const utility = computeWundtBerlyneUtility(feat.order_params.kappa_kin, 0.60);

      const dNorm = Math.sqrt(dischargedDeltaP.reduce((acc, v) => acc + v * v, 0));
      const aNorm = Math.sqrt(accumulatedDelta.reduce((acc, v) => acc + v * v, 0));

      paragraphs.push({
        paragraphIndex: k,
        wellIndex: wellIdx,
        wellName: well.name,
        text: paraText,
        charCount: paraText.length,
        sentenceCount: paraSents.length,
        cvL: paraCvL,
        targetCvL: well.targetCvL,
        wundtBerlyneUtility: utility,
        bankedEnergyBefore: energyBefore,
        wasDischarged,
        dischargedMomentumNorm: dNorm,
        accumulatedDeltaNorm: aNorm,
        refractoryCounterAfter: this.refractoryCounter,
      });
    }

    const overallCvL = computeCvL(allSentences);
    const totalChars = fullParagraphsText.reduce((acc, p) => acc + p.length, 0);
    const fullText = fullParagraphsText.join('\n\n');

    const inDeadzone = overallCvL < 0.28;
    const isHealthy = overallCvL >= 0.40 && overallCvL <= 0.70;

    return {
      fullText,
      totalChars,
      paragraphCount: paragraphs.length,
      paragraphs,
      wells: paragraphs.map((p) => p.wellIndex),
      overallCvL,
      inDeadzone,
      isHealthy,
      bankedHistory,
      finalBankedEnergy: this.getBankedEnergy(),
    };
  }
}
