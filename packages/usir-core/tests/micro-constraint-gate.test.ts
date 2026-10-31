import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  runMicroConstraintGate,
  detectTautology,
  detectSelectionalMismatch,
  detectRegisterViolation,
  verifyPhaseHardConstraints,
  detectForcedKeywords,
} from '../src/cleanroom/microConstraintGate.js';
import { CLOSED_LOOP_BENCHMARK_CORPUS } from '../src/dynamics/closedLoopCorpus.js';

describe('Step 1: Micro-Constraint Gate & Anti-Goodhart Literary Audit Suite', () => {
  const fixturePath = path.resolve(__dirname, 'fixtures', 'python_reference_40.json');
  const fixture: Array<{ id: string; text: string; phase: string }> = JSON.parse(
    fs.readFileSync(fixturePath, 'utf-8')
  );

  // ---------------------------------------------------------------------------
  // 1. Rejection of 6 Defective Stage 2 Initial Drafts (100% Rejection)
  // ---------------------------------------------------------------------------
  describe('Rejection of 6 Defective Stage 2 Initial Drafts', () => {
    it('catches Defect 1: "青い青葉" + "冷たい砂のように" + "蜂の抜け殻" (Initial DEAD-01 Void draft)', () => {
      const defectiveDraft =
        '傷は塞がらなかった。山あいの宿の縁側に腰を下ろすと、瓦の隙間に乾いた蜜蜂の死骸がただ一つ、雨露に打たれることもなく黒く転がっていた。羽も動かない。昼下がりの日差しが青い青葉の影を落とし、遠い瀬の音だけが、耳の底へ冷たい砂のようにかすかに染み込んでくる。痛みの記憶は薄れた。風が止むと、蜂の抜け殻は乾いた足の爪を瓦の丸みにひっかけたまま、静かに沈黙の底へじっと横たわっていた。';

      const res = runMicroConstraintGate(defectiveDraft, { targetPhase: 'III_Void' });
      expect(res.passed).toBe(false);

      const types = res.violations.map((v) => v.type);
      expect(types).toContain('tautology'); // 青い青葉
      expect(types).toContain('semantic_misuse'); // 蜂の抜け殻
      expect(types).toContain('phase_hard_constraint'); // 冷たい砂のように

      const tokens = res.violations.map((v) => v.token);
      expect(tokens.some((t) => t.includes('青葉'))).toBe(true);
      expect(tokens.some((t) => t.includes('抜け殻'))).toBe(true);
      expect(tokens.some((t) => t.includes('のように'))).toBe(true);
    });

    it('catches Defect 2: "青空へ向かって大きく跳躍するように叫び声を上げた" + abstract fillers "深く"/"強く" (Initial DEAD-01 Critical draft)', () => {
      const defectiveDraft =
        '足が震えた！しかし、血の塊がへばりついた瓦の縁へ這い寄る黒い蜂の針は、午後の鋭い斜光を浴びてぎらりと青黒い凶器の輝きを放った！忽ち風が唸る。生きて動く肉体の激痛と死の冷酷な静止が、引き裂かれた胸の奥深くで猛烈な火花を散らしながら激突したのだ！死ぬものか！男は縁板を強く蹴立て、青空へ向かって大きく跳躍するように叫び声を上げた。';

      const res = runMicroConstraintGate(defectiveDraft, { targetPhase: 'II_Critical' });
      expect(res.passed).toBe(false);

      const types = res.violations.map((v) => v.type);
      expect(types).toContain('forced_keyword'); // 青空へ向かって大きく跳躍するように叫び声を上げた
      expect(types).toContain('phase_hard_constraint'); // 深く, 強く

      const tokens = res.violations.map((v) => v.token);
      expect(tokens.some((t) => t.includes('跳躍'))).toBe(true);
      expect(tokens.some((t) => t.includes('深く') || t.includes('強く'))).toBe(true);
    });

    it('catches Defect 3: "匂いがかすかに立ち上って黒い土を濡らしていた" + intra-sentence tautology (Initial DEAD-05 Void draft)', () => {
      const defectiveDraft =
        '花が咲き乱れていた。白く霞む花びらの群れの真下に立つと、湿った黒い土の底から、冷たい獣の腐乱した骨の匂いがかすかに立ち上って黒い土を濡らしていた。風は凪いでいた。無数の花弁が空を塞ぐ重苦しさの中で、無残に枯れ果てた虫たちの翅が、根方の湿った苔の上に音もなくただ落ちて横たわった。';

      const res = runMicroConstraintGate(defectiveDraft, { targetPhase: 'III_Void' });
      expect(res.passed).toBe(false);

      const types = res.violations.map((v) => v.type);
      expect(types).toContain('semantic_misuse'); // 匂いが...濡らしていた
      expect(types).toContain('tautology'); // 黒い土 repeated in one sentence
    });

    it('catches Defect 4: "土を強く爪立てて" with abstract modifier (Initial DEAD-05 Critical draft)', () => {
      const defectiveDraft =
        '眩暈が頭を撃ち抜いた。しかし、頭上を覆い尽くす桜の絢爛たる白濁色の天幕は、地面の底に埋められた無数の腐肉と血膿を吸い上げて膨れ上がっていた！不意に寒気が走る。腐敗の底から湧き出る妖しい生の嵐に呑み込まれ、男は土を強く爪立てて叫んだ！';

      const res = runMicroConstraintGate(defectiveDraft, { targetPhase: 'II_Critical' });
      expect(res.passed).toBe(false);

      const tokens = res.violations.map((v) => v.token);
      expect(tokens.some((t) => t.includes('強く') || t.includes('爪立て'))).toBe(true);
    });

    it('catches Defect 5: Modern loanword "猛スピードで疾走し去った" (Initial DEAD-06 Critical draft)', () => {
      const defectiveDraft =
        '山あいの夜露が凍りついた。だが、霧に閉ざされた渓谷の茂みから躍り出た巨大な影は、月光に獰猛な牙を剥き出し、鋭い鉤爪で黒い岩肌を削り取った！忽ち友の叫びが響く。咆哮とともに宙を跳躍した巨躯の影は、断崖を滑り落ちて暗黒の森林へと猛スピードで疾走し去った。';

      const res = runMicroConstraintGate(defectiveDraft, {
        targetPhase: 'II_Critical',
        registerMode: 'classical_literary',
      });
      expect(res.passed).toBe(false);

      const types = res.violations.map((v) => v.type);
      expect(types).toContain('register_violation');

      const violation = res.violations.find((v) => v.token.includes('スピード'));
      expect(violation).toBeDefined();
    });

    it('catches Defect 6: Physically absurd train leap "汽車は北の荒野へと怒涛の勢いで跳躍した" (Initial DEAD-07 Critical draft)', () => {
      const defectiveDraft =
        '鉄の車輪が火花を散らした。しかし、暗い隧道を突破した汽車の窓を開け放つと、猛吹雪の白魔が牙を剥いて熱気の車内へ一気に雪崩れ込んだ！汽車は北の荒野へと怒涛の勢いで跳躍した。';

      const res = runMicroConstraintGate(defectiveDraft, { targetPhase: 'II_Critical' });
      expect(res.passed).toBe(false);

      const types = res.violations.map((v) => v.type);
      expect(types).toContain('forced_keyword');

      const tokens = res.violations.map((v) => v.token);
      expect(tokens.some((t) => t.includes('跳躍'))).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Verification of All 16 Refined Stage 2 Rewrites in CLOSED_LOOP_BENCHMARK_CORPUS
  // ---------------------------------------------------------------------------
  describe('Clean Pass Verification across all 16 Refined Stage 2 Rewrites', () => {
    it('verifies 100% (16/16) of CLOSED_LOOP_BENCHMARK_CORPUS pass runMicroConstraintGate with 0 violations', () => {
      expect(CLOSED_LOOP_BENCHMARK_CORPUS.length).toBe(16);

      const failedRecords: { id: string; phase: string; violations: string[] }[] = [];

      for (const record of CLOSED_LOOP_BENCHMARK_CORPUS) {
        const audit = runMicroConstraintGate(record.rewrittenText, {
          targetPhase: record.targetPhase,
          registerMode: 'classical_literary',
        });

        if (!audit.passed) {
          failedRecords.push({
            id: record.id,
            phase: record.targetPhase,
            violations: audit.violations.map((v) => `[${v.type}] ${v.token}: ${v.message}`),
          });
        }
      }

      expect(
        failedRecords,
        `Violations found in refined corpus: ${JSON.stringify(failedRecords, null, 2)}`
      ).toEqual([]);
    });
  });

  // ---------------------------------------------------------------------------
  // 3. 20/20 Masterwork Zero False-Positive & 8/8 Deadzone Full Rejection Audit
  // ---------------------------------------------------------------------------
  describe('20/20 Masterwork Zero False-Positive & 8/8 Deadzone Full Rejection', () => {
    it('achieves 100% (20/20) PASS on all authentic literary masterworks (VOID-01..10 & CRIT-01..10)', () => {
      const voidChunks = fixture.filter((x) => x.id.startsWith('VOID-'));
      const critChunks = fixture.filter((x) => x.id.startsWith('CRIT-'));

      expect(voidChunks.length).toBe(10);
      expect(critChunks.length).toBe(10);

      // Verify VOID-01..10 against Phase III (Void)
      for (const v of voidChunks) {
        const audit = runMicroConstraintGate(v.text, {
          targetPhase: 'III_Void',
          registerMode: 'classical_literary',
        });
        expect(
          audit.passed,
          `Authentic masterwork ${v.id} unexpectedly failed with violations: ${audit.violations.map((x) => x.token).join(', ')}`
        ).toBe(true);
        expect(audit.violations).toHaveLength(0);
      }

      // Verify CRIT-01..10 against Phase II (Critical)
      for (const c of critChunks) {
        const audit = runMicroConstraintGate(c.text, {
          targetPhase: 'II_Critical',
          registerMode: 'classical_literary',
        });
        expect(
          audit.passed,
          `Authentic masterwork ${c.id} unexpectedly failed with violations: ${audit.violations.map((x) => x.token).join(', ')}`
        ).toBe(true);
        expect(audit.violations).toHaveLength(0);
      }
    });

    it('achieves 100% (8/8) REJECT on all AI Deadzone benchmark chunks (DEAD-01..08)', () => {
      const deadChunks = fixture.filter((x) => x.id.startsWith('DEAD-'));
      expect(deadChunks.length).toBe(8);

      for (const d of deadChunks) {
        const audit = runMicroConstraintGate(d.text, {
          targetPhase: 'III_Void',
          registerMode: 'classical_literary',
        });
        expect(
          audit.passed,
          `AI Deadzone passage ${d.id} unexpectedly bypassed the gate!`
        ).toBe(false);
        expect(audit.violations.length).toBeGreaterThanOrEqual(1);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Universal Modality Selectional Restriction Matrix (detectSelectionalMismatch)
  // ---------------------------------------------------------------------------
  describe('Universal Modality Selectional Restriction Matrix', () => {
    it('detects Class 1: Intangible/auditory/olfactory entity wetting matter across multiple novel combinations', () => {
      const cases = [
        '静寂が石畳を濡らしていた。',
        '冷たい視線が衣服を濡らした。',
        '遠くの物音が夜の庭を湿らせる。',
        '芳香が暗い部屋をびしょ濡れにした。',
      ];

      for (const text of cases) {
        const violations = detectSelectionalMismatch(text);
        expect(violations.length).toBeGreaterThanOrEqual(1);
        expect(violations[0]!.type).toBe('semantic_misuse');
      }
    });

    it('detects Class 2: Massive structure or heavy rail vehicle leaping across multiple novel combinations', () => {
      const cases = [
        '機関車は鉄橋の上で飛び跳ねた。',
        '城郭は宙を跳躍した。',
        '重い貨物列車が線路の上を跳ね回った。',
        '巨大な楼門が怒涛の勢いで跳躍した。',
      ];

      for (const text of cases) {
        const violations = detectSelectionalMismatch(text);
        expect(violations.length).toBeGreaterThanOrEqual(1);
        expect(violations[0]!.type).toBe('forced_keyword');
      }
    });

    it('detects Class 3: Adult insect or vertebrate ecdysis (抜け殻) across multiple novel combinations', () => {
      const cases = [
        '道端に猫の抜け殻が転がっていた。',
        '草むらで雀蜂の脱皮殻を見つけた。',
        '庭の隅に狼の抜け殻があった。',
        '古い木の洞に熊の脱殻が残っていた。',
      ];

      for (const text of cases) {
        const violations = detectSelectionalMismatch(text);
        expect(violations.length).toBeGreaterThanOrEqual(1);
        expect(violations[0]!.type).toBe('semantic_misuse');
      }
    });

    it('detects Class 4: Convalescent context cartoon sky leaps', () => {
      const sample = '病み上がりの身体を抱え、青年は青空へ向かって大きく跳躍した。';
      const violations = detectSelectionalMismatch(sample);
      expect(violations.length).toBeGreaterThanOrEqual(1);
      expect(violations[0]!.type).toBe('forced_keyword');
    });
  });

  // ---------------------------------------------------------------------------
  // 5. Isolated Detector Unit Tests
  // ---------------------------------------------------------------------------
  describe('Isolated Detector Capabilities', () => {
    it('detectTautology catches multiple same-root kanji compounds', () => {
      const sample = '白い白雪が降る。黒い黒土の上で、赤い紅蓮の炎が燃える。';
      const violations = detectTautology(sample);
      expect(violations.length).toBeGreaterThanOrEqual(3);
    });

    it('detectRegisterViolation identifies modern loanwords and provides classical suggestions', () => {
      const sample = 'このシステムのプロセスとレベルには高いストレスが存在する。';
      const violations = detectRegisterViolation(sample, 'classical_literary');
      expect(violations.length).toBeGreaterThanOrEqual(4);

      const systemViolation = violations.find((v) => v.token === 'システム');
      expect(systemViolation?.suggestion).toContain('体系');
    });

    it('detectForcedKeywords catches meta Goodhart jargon', () => {
      const sample = 'シンプレクティックな構造とハミルトニアンの臨界相をマハラノビス距離で解く。';
      const violations = detectForcedKeywords(sample);
      expect(violations.length).toBeGreaterThanOrEqual(3);
    });
  });
});
