import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import {
  runClosedLoopPhaseTransition,
  createMockCallbackProvider,
  type LlmProvider,
} from '../src/runtimeLoop.js';
import { CLOSED_LOOP_BENCHMARK_CORPUS } from '@nishiki/usir-core';

describe('Stage 3: Runtime Loop & MCP/CLI Integration Suite', () => {
  const dead01RecordVoid = CLOSED_LOOP_BENCHMARK_CORPUS.find(
    (r) => r.id === 'DEAD-01' && r.targetPhase === 'III_Void'
  )!;
  const dead01RecordCrit = CLOSED_LOOP_BENCHMARK_CORPUS.find(
    (r) => r.id === 'DEAD-01' && r.targetPhase === 'II_Critical'
  )!;

  // ===========================================================================
  // Test 1: Multi-Iteration Self-Healing for III_Void with Micro-Constraint Feedback
  // ===========================================================================
  it('cures micro-constraint defects across iterations for III_Void', async () => {
    // Draft 1 passes macro 24D physics (III_Void, high phi_void, CV_L >= 0.40)
    // but contains 3 deliberate micro-violations: "青い青葉", "冷たい砂のように", and "蜂の抜け殻"
    const defectiveDraft =
      '傷は塞がらなかった。山あいの宿の縁側に腰を下ろすと、瓦の隙間に乾いた蜜蜂の死骸がただ一つ、雨露に打たれることもなく黒く転がっていた。羽も動かない。昼下がりの日差しが青い青葉の影を落とし、遠い瀬の音だけが、耳の底へ冷たい砂のようにかすかに染み込んでくる。痛みの記憶は薄れた。風が止むと、蜂の抜け殻は乾いた足の爪を瓦の丸みにひっかけたまま、静かに沈黙の底へじっと横たわっていた。';

    // Draft 2 is the 100% clean benchmark rewrite from closedLoopCorpus
    const cleanRewrite = dead01RecordVoid.rewrittenText;

    let iteration2PromptReceived = '';

    const mockProvider: LlmProvider = createMockCallbackProvider((prompt, iter) => {
      if (iter === 1) {
        return defectiveDraft;
      }
      iteration2PromptReceived = prompt;
      return cleanRewrite;
    });

    const result = await runClosedLoopPhaseTransition(
      dead01RecordVoid.sourceText,
      'III_Void',
      mockProvider,
      { maxIterations: 3 }
    );

    // Iteration 1 must be rejected by micro-constraint gate despite valid macro physics
    expect(result.history).toHaveLength(2);
    expect(result.history[0]!.microGate.passed).toBe(false);
    expect(result.history[0]!.passedAll).toBe(false);

    // Verify that Iteration 2 prompt received explicit micro-constraint fix instructions
    expect(iteration2PromptReceived).toContain('[MICRO-CONSTRAINT GATE VIOLATIONS TO FIX]');
    expect(iteration2PromptReceived).toContain('青葉');
    expect(iteration2PromptReceived).toContain('抜け殻');
    expect(iteration2PromptReceived).toContain('のように');

    // Iteration 2 must converge cleanly
    expect(result.history[1]!.microGate.passed).toBe(true);
    expect(result.history[1]!.passedAll).toBe(true);
    expect(result.converged).toBe(true);
    expect(result.iterationsUsed).toBe(2);
    expect(result.finalPhase).toBe('III_Void');
    expect(result.finalMetrics.order_params.phi_void).toBeGreaterThanOrEqual(0.55);
    expect(result.finalMicroGate.passed).toBe(true);
    expect(result.finalMicroGate.violations).toHaveLength(0);
  });

  // ===========================================================================
  // Test 2: Multi-Iteration Self-Healing for II_Critical
  // ===========================================================================
  it('cures register and abstract filler defects across iterations for II_Critical', async () => {
    // Draft 1 has low CV_L, modern slang "猛スピードで", and abstract filler "深く"
    const defectiveCritDraft =
      '男は猛スピードで走り出した。胸の奥深くで激痛が走る。雨が激しく門を叩き割る！忽ち冷たい刃が閃光を放ち、石畳を真っ二つに割った！';

    // Draft 2 is the clean critical benchmark rewrite
    const cleanCritRewrite = dead01RecordCrit.rewrittenText;

    let iteration2PromptReceived = '';

    const mockProvider: LlmProvider = createMockCallbackProvider((prompt, iter) => {
      if (iter === 1) {
        return defectiveCritDraft;
      }
      iteration2PromptReceived = prompt;
      return cleanCritRewrite;
    });

    const result = await runClosedLoopPhaseTransition(
      dead01RecordCrit.sourceText,
      'II_Critical',
      mockProvider,
      { maxIterations: 3 }
    );

    expect(result.history).toHaveLength(2);
    expect(result.history[0]!.passedAll).toBe(false);

    // Iteration 2 prompt must instruct fixing the register and abstract violations
    expect(iteration2PromptReceived).toContain('[MICRO-CONSTRAINT GATE VIOLATIONS TO FIX]');
    expect(iteration2PromptReceived).toContain('猛スピード');

    // Converged at iteration 2
    expect(result.converged).toBe(true);
    expect(result.iterationsUsed).toBe(2);
    expect(result.finalPhase).toBe('II_Critical');
    expect(result.finalMetrics.order_params.kappa_kin).toBeGreaterThanOrEqual(0.65);
    expect(result.finalMicroGate.passed).toBe(true);
  });

  // ===========================================================================
  // Test 3: Anti-Oscillation Cumulative Violation Accumulation Across Iterations
  // ===========================================================================
  it('accumulates prior violations across iterations so feedback prevents oscillation', async () => {
    // Iteration 1 draft has violation A: "青い青葉" (tautology)
    const draft1 =
      '傷は塞がらなかった。山あいの宿の縁側に腰を下ろすと、瓦の隙間に乾いた蜜蜂の死骸がただ一つ、雨露に打たれることもなく黒く転がっていた。羽も動かない。昼下がりの日差しが青い青葉の影を落とし、遠い瀬の音だけが、耳の底へ冷たく染み入ってくる。痛みの記憶は薄れた。風が止むと、動かぬ蜂の骸は細い足の爪を瓦の丸みにひっかけたまま、午後の沈黙の底にただじっと横たわっていた。';

    // Iteration 2 draft fixes "青い青葉" (now "青葉") but introduces violation B: "冷たい砂のように" (simile)
    const draft2 =
      '傷は塞がらなかった。山あいの宿の縁側に腰を下ろすと、瓦の隙間に乾いた蜜蜂の死骸がただ一つ、雨露に打たれることもなく黒く転がっていた。羽も動かない。昼下がりの日差しが生い茂る青葉の影を落とし、遠い瀬の音だけが、耳の底へ冷たい砂のようにかすかに染み込んでくる。痛みの記憶は薄れた。風が止むと、動かぬ蜂の骸は細い足の爪を瓦の丸みにひっかけたまま、午後の沈黙の底にただじっと横たわっていた。';

    // Iteration 3 is clean
    const draft3 = dead01RecordVoid.rewrittenText;

    let iteration3PromptReceived = '';

    const mockProvider: LlmProvider = createMockCallbackProvider((prompt, iter) => {
      if (iter === 1) return draft1;
      if (iter === 2) return draft2;
      iteration3PromptReceived = prompt;
      return draft3;
    });

    const result = await runClosedLoopPhaseTransition(
      dead01RecordVoid.sourceText,
      'III_Void',
      mockProvider,
      { maxIterations: 4 }
    );

    // Iteration 1 failed on tautology "青い青葉"
    // Iteration 2 failed on simile "のように"
    // Iteration 3 prompt must accumulate BOTH "青葉" and "のように" to prevent oscillation!
    expect(iteration3PromptReceived).toContain('[MICRO-CONSTRAINT GATE VIOLATIONS TO FIX]');
    expect(iteration3PromptReceived).toContain('青葉');
    expect(iteration3PromptReceived).toContain('のように');
    expect(iteration3PromptReceived).toContain('cumulative violation(s) identified across iterations');

    // Final result converged
    expect(result.converged).toBe(true);
    expect(result.iterationsUsed).toBe(3);
  });

  // ===========================================================================
  // Test 4: CLI Subcommands (extract-genome, trajectory, inspect, diagnose, steer, prompt)
  // ===========================================================================
  describe('CLI subcommands: extract-genome & trajectory', () => {
    const cliPath = path.resolve(__dirname, '../dist/cli.js');
    const tempDir = path.resolve(__dirname, 'temp_cli_test');

    it('extract-genome compiles a strict 4096-byte .usir binary and outputs Clean Room audit', () => {
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
      }
      const outPath = path.join(tempDir, 'test_output_genome.usir');
      if (fs.existsSync(outPath)) {
        fs.unlinkSync(outPath);
      }

      const inputCorpus = JSON.stringify([
        dead01RecordVoid.rewrittenText,
        dead01RecordCrit.rewrittenText,
      ]);

      const cmd = `node "${cliPath}" extract-genome '${inputCorpus.replace(/'/g, "'\\''")}' -o "${outPath}" --json`;
      const stdout = execSync(cmd, { encoding: 'utf-8' });
      const parsed = JSON.parse(stdout);

      // Binary size on disk
      expect(fs.existsSync(outPath)).toBe(true);
      const stat = fs.statSync(outPath);
      expect(stat.size).toBe(4096);

      // JSON stdout verification
      expect(parsed.byteLength).toBe(4096);
      expect(parsed.audit.passed).toBe(true);
      expect(parsed.dominantPhase).toBeDefined();
      expect(parsed.centroid24d).toHaveLength(24);

      // Clean up
      fs.unlinkSync(outPath);
      fs.rmdirSync(tempDir);
    });

    it('extract-genome falls back to sentence-window chunking when given single long text without double newlines', () => {
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
      }
      const outPath = path.join(tempDir, 'single_paragraph_genome.usir');
      if (fs.existsSync(outPath)) {
        fs.unlinkSync(outPath);
      }

      // Single paragraph with 6 sentences without any \n\n
      const singleLongText =
        '傷は塞がらなかった。山あいの宿の縁側に腰を下ろすと、瓦の隙間に乾いた蜜蜂の死骸がただ一つ、雨露に打たれることもなく黒く転がっていた。羽も動かない。昼下がりの日差しが生い茂る青葉の影を落とし、遠い瀬の音だけが、静まり返った耳の底へ冷たく染み入ってくる。痛みの記憶は薄れた。風が止むと、動かぬ蜂の骸は細い足の爪を瓦の丸みにひっかけたまま、午後の沈黙の底にただじっと横たわっていた。';

      const tempInput = path.join(tempDir, 'single_para_input.txt');
      fs.writeFileSync(tempInput, singleLongText, 'utf-8');

      const cmd = `node "${cliPath}" extract-genome "${tempInput}" -o "${outPath}" --json`;
      const stdout = execSync(cmd, { encoding: 'utf-8' });
      const parsed = JSON.parse(stdout);

      expect(fs.existsSync(outPath)).toBe(true);
      expect(fs.statSync(outPath).size).toBe(4096);
      expect(parsed.byteLength).toBe(4096);
      expect(parsed.audit.passed).toBe(true);

      fs.unlinkSync(tempInput);
      fs.unlinkSync(outPath);
      fs.rmdirSync(tempDir);
    });

    it('executes all core CLI subcommands (inspect, diagnose, steer, prompt) cleanly', () => {
      const sampleText = dead01RecordVoid.rewrittenText;

      // 1. inspect
      const inspectOut = execSync(`node "${cliPath}" inspect "${sampleText.slice(0, 80)}" --json`, { encoding: 'utf-8' });
      const inspectJson = JSON.parse(inspectOut);
      expect(inspectJson.z.length).toBe(24);
      expect(inspectJson.clean_room.tierSByteLength).toBe(128);

      // 2. diagnose
      const diagOut = execSync(`node "${cliPath}" diagnose "${sampleText.slice(0, 80)}" --json`, { encoding: 'utf-8' });
      const diagJson = JSON.parse(diagOut);
      expect(diagJson.predictedPhase).toBeDefined();

      // 3. steer
      const steerOut = execSync(`node "${cliPath}" steer "${sampleText.slice(0, 80)}" --target III_Void --json`, { encoding: 'utf-8' });
      const steerJson = JSON.parse(steerOut);
      expect(steerJson.rewrittenText).toBeDefined();
      expect(steerJson.gate4Passed).toBeDefined();

      // 4. prompt
      const promptOut = execSync(`node "${cliPath}" prompt "${sampleText.slice(0, 80)}" --target III_Void --json`, { encoding: 'utf-8' });
      const promptJson = JSON.parse(promptOut);
      expect(promptJson.compiledPrompt).toBeDefined();
      expect(promptJson.targetPhase).toBe('III_Void');
    });

    it('trajectory analyzes 3-mode HSMM wave and isolates Deadzone sinkholes via CLI JSON', () => {
      const fixturePath = path.resolve(
        __dirname,
        '../../usir-core/tests/fixtures/python_reference_40.json'
      );
      const fixtureData: any[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
      const crit01 = fixtureData.find((x) => x.id === 'CRIT-01')!.text;
      const void02 = fixtureData.find((x) => x.id === 'VOID-02')!.text;
      const dead01 = fixtureData.find((x) => x.id === 'DEAD-01')!.text;

      // Hybrid text: Para 0 = Critical, Para 1 = Void, Para 2 = Deadzone Sinkhole
      const hybridDoc = [crit01, void02, dead01].join('\n\n');

      // Write to temporary file to avoid shell escaping issues on Windows
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
      }
      const tempInput = path.join(tempDir, 'trajectory_input.txt');
      fs.writeFileSync(tempInput, hybridDoc, 'utf-8');

      const cmd = `node "${cliPath}" trajectory "${tempInput}" --json`;
      const stdout = execSync(cmd, { encoding: 'utf-8' });
      const parsed = JSON.parse(stdout);

      expect(parsed.windowCount).toBeGreaterThan(0);
      expect(parsed.viterbiPath.length).toBe(parsed.windowCount);
      expect(parsed.modeWeights).toHaveLength(3);
      expect(parsed.hasSinkholes).toBe(true);
      expect(parsed.sinkholeCount).toBe(1);
      expect(parsed.sinkholes[0].paragraphIndex).toBe(2);

      // Clean up
      fs.unlinkSync(tempInput);
      fs.rmdirSync(tempDir);
    });
  });
});
