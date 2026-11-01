import { describe, it, expect } from 'vitest';
import { runLiveOrOfflineIterativeRewrite } from '../src/liveAiConnector.js';
import { PRESALES_SHOWCASE_CASES } from '@nishiki/usir-core';

describe('Workstream 4: Live AI Rewrite Connector & Closed-Loop Refinement', () => {
  const sampleSlop =
    '弊社のクラウド基盤導入について様々な観点から検討することは非常に重要であり、' +
    '年間コストについて最大で42%の削減を達成することが十分に可能であると考えられます。' +
    'さらにSLA 99.99%を保証し、最短6ヶ月での完了を目指します。';

  it('executes 3-iteration closed-loop rewrite in offline deterministic mode with 100% fact retention', async () => {
    const res = await runLiveOrOfflineIterativeRewrite({
      text: sampleSlop,
      targetMode: 'I_Crystalline',
      isLive: false,
      useCaseId: 'case_3_b2b_proposal',
    });

    expect(res).toBeDefined();
    expect(res.isLive).toBe(false);
    expect(res.fallbackUsed).toBe(false);
    expect(res.targetMode).toBe('I_Crystalline');

    // 3 Iterations
    expect(res.iterationLogs.length).toBe(3);
    expect(res.iterationLogs[0]!.iterationName).toContain('Iteration 1: Slop Elimination');
    expect(res.iterationLogs[1]!.iterationName).toContain('Iteration 2: Cadence');
    expect(res.iterationLogs[2]!.iterationName).toContain('Iteration 3: Terminal Kamae');

    // z24 Trajectory contains initial + 3 steps
    expect(res.z24Trajectory.length).toBe(4);
    for (const z of res.z24Trajectory) {
      expect(z.length).toBe(24);
    }

    // Fact Retention Guard preserves 100% of facts
    expect(res.redlineResult.factRetentionRate).toBe(1.0);
    expect(res.finalText).toContain('42%');
    expect(res.finalText).toContain('SLA 99.99%');
    expect(res.finalText).toContain('6ヶ月');

    // Redline Diff generated
    expect(res.redlineResult.redlineHtml).toContain('usir-redline-del');
    expect(res.redlineResult.roiMetrics.slopReductionRatio).toBeGreaterThan(0.7);
  });

  it('supports simulated Live HTTP endpoint and parses response cleanly', async () => {
    let callCount = 0;
    const mockFetch: typeof fetch = async (url, init) => {
      callCount++;
      const body = JSON.parse((init?.body as string) || '{}');
      const rewritten = `[Mock Iteration ${callCount}] 次世代クラウド基盤（42%削減、SLA 99.99%、6ヶ月完了）。`;
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ response: rewritten }),
      } as any;
    };

    const res = await runLiveOrOfflineIterativeRewrite({
      text: sampleSlop,
      targetMode: 'III_Void',
      isLive: true,
      endpoint: 'http://localhost:11434/api/generate',
      model: 'qwen2.5',
      fetchFn: mockFetch,
    });

    expect(res.isLive).toBe(true);
    expect(res.fallbackUsed).toBe(false);
    expect(callCount).toBe(3);
    expect(res.iterationLogs.length).toBe(3);
    expect(res.finalText).toContain('Mock Iteration 3');
  });

  it('gracefully falls back to deterministic offline rewriter when live endpoint is unreachable', async () => {
    const failingFetch: typeof fetch = async () => {
      throw new Error('ECONNREFUSED: Connection refused at http://localhost:11434');
    };

    const res = await runLiveOrOfflineIterativeRewrite({
      text: PRESALES_SHOWCASE_CASES.case_1_literary.beforeText,
      targetMode: 'III_Void',
      isLive: true,
      endpoint: 'http://localhost:11434/api/generate',
      fetchFn: failingFetch,
      useCaseId: 'case_1_literary',
    });

    expect(res.isLive).toBe(false);
    expect(res.fallbackUsed).toBe(true);
    expect(res.iterationLogs.length).toBe(3);
    expect(res.redlineResult.factRetentionRate).toBe(1.0);
    expect(res.finalText.length).toBeGreaterThan(20);
  });
});
