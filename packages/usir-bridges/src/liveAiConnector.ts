/**
 * @nishiki/usir-bridges: Live AI Rewrite Connector & Iterative Closed-Loop Tuning
 * Workstream 4 Implementation: Local Ollama / OpenAI-compatible endpoint integration
 * with 3-Iteration Hamiltonian refinement and instantaneous offline deterministic fallback.
 */

import {
  extractUsir24d,
  runMicroConstraintGate,
  generateRewriteDirectives,
  computeRedlineDiffAndRoi,
  applyDeterministicStyleRewrite,
  stripAiSlopPatterns,
  extractCoreFacts,
  PRESALES_SHOWCASE_CASES,
  type PresalesTargetMode,
  type RedlineDiffResult,
} from '@nishiki/usir-core';

// ============================================================================
// Types & Interfaces
// ============================================================================

export type LiveConnectorTargetMode = PresalesTargetMode;

export interface IterationTelemetry {
  iterationIndex: number; // 1, 2, 3
  iterationName: string;
  inputText: string;
  outputText: string;
  z24: number[];
  cvL: number;
  voidRatio: number;
  slopCount: number;
  gatePassed: boolean;
  violations: string[];
  hamiltonianPrompt?: string;
  durationMs: number;
}

export interface IterativeRewriteOptions {
  text: string;
  targetMode: LiveConnectorTargetMode;
  isLive?: boolean;
  endpoint?: string;
  model?: string;
  apiKey?: string;
  fetchFn?: typeof fetch;
  timeoutMs?: number;
  useCaseId?: string;
}

export interface IterativeRewriteResult {
  isLive: boolean;
  fallbackUsed: boolean;
  targetMode: LiveConnectorTargetMode;
  initialText: string;
  finalText: string;
  iterationLogs: IterationTelemetry[];
  z24Trajectory: number[][]; // [z_initial, z_iter1, z_iter2, z_iter3]
  redlineResult: RedlineDiffResult;
  totalDurationMs: number;
}

// ============================================================================
// Helper: Live HTTP Dispatcher
// ============================================================================

async function callLiveEndpoint(
  endpoint: string,
  model: string,
  systemPrompt: string,
  userPrompt: string,
  apiKey?: string,
  fetchFn?: typeof fetch,
  timeoutMs: number = 8000
): Promise<string> {
  const actualFetch = fetchFn || (typeof globalThis.fetch === 'function' ? globalThis.fetch : null);
  if (!actualFetch) {
    throw new Error('No global fetch function available');
  }

  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const isOpenAi = endpoint.includes('/chat/completions') || endpoint.includes('/v1/');
    let body: string;

    if (isOpenAi) {
      body = JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.3,
      });
    } else {
      // Ollama /api/generate format
      const fullPrompt = `${systemPrompt}\n\n[INPUT TEXT]:\n${userPrompt}\n\n[REWRITTEN OUTPUT]:`;
      body = JSON.stringify({
        model,
        prompt: fullPrompt,
        stream: false,
        options: {
          temperature: 0.3,
        },
      });
    }

    const response = await actualFetch(endpoint, {
      method: 'POST',
      headers,
      body,
      signal: controller ? controller.signal : undefined,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const json = (await response.json()) as any;
    let resultText = '';

    if (isOpenAi) {
      resultText = json.choices?.[0]?.message?.content || '';
    } else {
      // Ollama format returns 'response' field
      resultText = json.response || json.message?.content || '';
    }

    resultText = resultText.trim();
    if (!resultText) {
      throw new Error('Empty response from model');
    }
    return resultText;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

// ============================================================================
// Iterative Closed-Loop Tuning Engine
// ============================================================================

export async function runLiveOrOfflineIterativeRewrite(
  options: IterativeRewriteOptions
): Promise<IterativeRewriteResult> {
  const startTime = Date.now();
  const {
    text,
    targetMode,
    isLive = false,
    endpoint = 'http://localhost:11434/api/generate',
    model = 'llama3',
    apiKey,
    fetchFn,
    timeoutMs = 6000,
    useCaseId,
  } = options;

  const initialMetrics = extractUsir24d(text);
  const zInitial = Array.from(initialMetrics.z_24d);
  const coreFacts = extractCoreFacts(text);

  let currentText = text;
  let fallbackUsed = false;
  const iterationLogs: IterationTelemetry[] = [];
  const zTrajectory: number[][] = [zInitial];

  const iterationDefinitions = [
    {
      index: 1,
      name: 'Iteration 1: Slop Elimination & Fact Lock',
      systemPrompt:
        'You are an expert Japanese editor applying USIR Hamiltonian Slop Elimination & Fact Lock.\n' +
        'Eliminate all vague fillers (〜というテーマ, 様々な観点, 非常に重要, ように思われた, と考えられる, など).\n' +
        'Preserve 100% of concrete facts, numbers, SLA metrics, nouns, and names.\n' +
        'Output ONLY the cleaned Japanese text without commentary or prefixes.',
    },
    {
      index: 2,
      name: 'Iteration 2: Cadence & Void / Soliton Shaping',
      systemPrompt:
        `You are an expert Japanese stylist tuning text into USIR Target Mode: ${targetMode}.\n` +
        'Modulate cadence and sentence length variation.\n' +
        (targetMode === 'III_Void'
          ? 'Break monotonous rhythms with apophatic pauses, short quiet clauses, and spacious breath.'
          : targetMode === 'II_Critical'
          ? 'Inject 1/f kinetic ruptures, vivid physical contrast, sharp momentum bursts, and sensory impact.'
          : targetMode === 'I_Crystalline'
          ? 'Enforce conclusion-first symmetry, crystal-clear declarative clauses, and zero ambiguity.'
          : 'Create extreme contrast between negative space stasis and hyperbolic soliton leap.') +
        '\nOutput ONLY the revised Japanese text.',
    },
    {
      index: 3,
      name: 'Iteration 3: Terminal Kamae & Micro-Constraint Gate',
      systemPrompt:
        `You are an expert Japanese prose master finalizing terminal kamae for USIR Mode: ${targetMode}.\n` +
        'Refine sentence endings, anchors, and ensure absolute compliance with stylistic constraints.\n' +
        'Ensure 100% of facts are intact. Output ONLY the polished final Japanese text.',
    },
  ];

  for (const iterDef of iterationDefinitions) {
    const iterStart = Date.now();
    let iterOutput = '';

    if (isLive && !fallbackUsed) {
      try {
        iterOutput = await callLiveEndpoint(
          endpoint,
          model,
          iterDef.systemPrompt,
          currentText,
          apiKey,
          fetchFn,
          timeoutMs
        );
      } catch (_err) {
        // Fallback to offline deterministic mode on network/timeout error
        fallbackUsed = true;
      }
    }

    if (!isLive || fallbackUsed) {
      // Deterministic offline progression
      if (iterDef.index === 1) {
        // Strip slop and preserve facts
        iterOutput = stripAiSlopPatterns(currentText);
      } else if (iterDef.index === 2) {
        // Apply intermediate structural cadence rewrite
        const partialRewritten = applyDeterministicStyleRewrite(currentText, targetMode);
        iterOutput = partialRewritten;
      } else {
        // Terminal iteration 3: check if built-in preset case matches
        const presetCase = useCaseId ? PRESALES_SHOWCASE_CASES[useCaseId] : null;
        if (presetCase && presetCase.variants[targetMode]) {
          iterOutput = presetCase.variants[targetMode]!.afterText;
        } else {
          iterOutput = applyDeterministicStyleRewrite(text, targetMode);
        }
      }
    }

    // Safety guard: ensure facts are retained in output
    for (const fact of coreFacts) {
      if (!iterOutput.includes(fact)) {
        if (!iterOutput.includes(fact)) {
          // Keep fact present
          iterOutput = iterOutput.trim();
          if (iterOutput.endsWith('。')) {
            iterOutput = iterOutput.slice(0, -1) + `（${fact}）。`;
          } else {
            iterOutput += `（${fact}）。`;
          }
        }
      }
    }

    const iterMetrics = extractUsir24d(iterOutput);
    const zIter = Array.from(iterMetrics.z_24d);
    zTrajectory.push(zIter);

    const q = zIter.slice(0, 12);
    const voidRatio = (q[2]! + q[6]! + q[10]!) / 3.0;
    const gateRes = runMicroConstraintGate(iterOutput, {
      targetPhase: targetMode === 'Genesis' ? 'III_Void' : targetMode,
      registerMode: 'classical_literary',
    });

    const directives = generateRewriteDirectives(iterOutput, targetMode === 'Genesis' ? 'III_Void' : targetMode, iterMetrics);

    const telemetry: IterationTelemetry = {
      iterationIndex: iterDef.index,
      iterationName: iterDef.name,
      inputText: currentText,
      outputText: iterOutput,
      z24: zIter,
      cvL: iterMetrics.len_cv,
      voidRatio,
      slopCount: gateRes.violations.length,
      gatePassed: gateRes.passed,
      violations: gateRes.violations.map((v) => `[${v.type}] ${v.token}: ${v.message}`),
      hamiltonianPrompt: directives.compiledPrompt,
      durationMs: Date.now() - iterStart,
    };

    iterationLogs.push(telemetry);
    currentText = iterOutput;
  }

  const finalText = currentText;
  const redlineResult = computeRedlineDiffAndRoi(text, finalText, targetMode, coreFacts);

  return {
    isLive: isLive && !fallbackUsed,
    fallbackUsed,
    targetMode,
    initialText: text,
    finalText,
    iterationLogs,
    z24Trajectory: zTrajectory,
    redlineResult,
    totalDurationMs: Date.now() - startTime,
  };
}
