/**
 * @nishiki/usir-mcp: Closed-Loop Phase Transition Runtime & Pluggable LLM Providers
 * Orchestrates multi-iteration self-healing generative loops combining macro 24D Hamiltonian
 * physics and micro-constraint clean room gating.
 */

import {
  extractUsir24d,
  classifyPhase,
  generateRewriteDirectives,
  compileIterativeFeedbackPrompt,
  runMicroConstraintGate,
  type UsirPhase,
  type UsirState,
  type MicroConstraintGateResult,
  type MicroViolation,
  type RegisterMode,
} from '@nishiki/usir-core';

// =============================================================================
// LLM PROVIDER INTERFACE & IMPLEMENTATIONS
// =============================================================================

export interface LlmProvider {
  name: string;
  generate(prompt: string, iteration: number): Promise<string>;
}

export interface OllamaProviderOptions {
  baseUrl?: string;
  model?: string;
  temperature?: number;
}

export function createOllamaProvider(options?: OllamaProviderOptions): LlmProvider {
  const baseUrl = options?.baseUrl ?? 'http://localhost:11434';
  const model = options?.model ?? 'llama3';
  const temperature = options?.temperature ?? 0.7;

  return {
    name: `ollama:${model}`,
    async generate(prompt: string, _iteration: number): Promise<string> {
      const url = `${baseUrl.replace(/\/+$/, '')}/api/generate`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
          options: { temperature },
        }),
      });

      if (!res.ok) {
        throw new Error(
          `[OllamaProvider] HTTP ${res.status} ${res.statusText} from ${url}`
        );
      }

      const data = (await res.json()) as { response?: string };
      return data.response ?? '';
    },
  };
}

export interface OpenAiCompatibleProviderOptions {
  baseUrl: string;
  apiKey?: string;
  model: string;
  temperature?: number;
}

export function createOpenAiCompatibleProvider(
  options: OpenAiCompatibleProviderOptions
): LlmProvider {
  const baseUrl = options.baseUrl.replace(/\/+$/, '');
  const model = options.model;
  const apiKey = options.apiKey ?? '';
  const temperature = options.temperature ?? 0.7;

  return {
    name: `openai-compatible:${model}`,
    async generate(prompt: string, _iteration: number): Promise<string> {
      const endpoint = baseUrl.endsWith('/v1')
        ? `${baseUrl}/chat/completions`
        : `${baseUrl}/v1/chat/completions`;

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature,
        }),
      });

      if (!res.ok) {
        throw new Error(
          `[OpenAiProvider] HTTP ${res.status} ${res.statusText} from ${endpoint}`
        );
      }

      const data = (await res.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      return data.choices?.[0]?.message?.content ?? '';
    },
  };
}

export function createMockCallbackProvider(
  handler: (prompt: string, iteration: number) => string | Promise<string>,
  name = 'mock-provider'
): LlmProvider {
  return {
    name,
    async generate(prompt: string, iteration: number): Promise<string> {
      return Promise.resolve(handler(prompt, iteration));
    },
  };
}

// =============================================================================
// CLOSED-LOOP PHASE TRANSITION RUNTIME
// =============================================================================

export interface IterationRecord {
  iteration: number;
  promptUsed: string;
  candidateText: string;
  phase: UsirPhase;
  lenCv: number;
  phiVoid: number;
  kappaKin: number;
  abstractB: number;
  microGate: MicroConstraintGateResult;
  passedAll: boolean;
}

export interface ClosedLoopOptions {
  maxIterations?: number;
  registerMode?: RegisterMode;
}

export interface ClosedLoopResult {
  converged: boolean;
  iterationsUsed: number;
  finalText: string;
  finalPhase: UsirPhase;
  finalMetrics: UsirState;
  finalMicroGate: MicroConstraintGateResult;
  history: IterationRecord[];
}

/**
 * Checks whether a candidate text meets all macro 24D physical bounds and micro-constraints.
 */
function evaluateConvergence(
  metrics: ReturnType<typeof extractUsir24d>,
  phase: UsirPhase,
  targetPhase: UsirPhase,
  microGate: MicroConstraintGateResult
): boolean {
  // 1. Micro-constraint gate must pass with 0 violations
  if (!microGate.passed) {
    return false;
  }

  // 2. Rhythm variation and anti-bloat invariant
  if (metrics.len_cv < 0.4 || metrics.abstract_b_mean > 0.01) {
    return false;
  }

  // 3. Phase-specific physics
  if (targetPhase === 'III_Void') {
    return phase === 'III_Void' && metrics.order_params.phi_void >= 0.55;
  }

  if (targetPhase === 'II_Critical') {
    return phase === 'II_Critical' && metrics.order_params.kappa_kin >= 0.65;
  }

  return phase === targetPhase;
}

/**
 * Executes a multi-iteration closed-loop generative rewrite towards a target thermodynamic phase.
 * If iteration k fails either macro 24D physics or micro constraints, compiles an augmented feedback
 * prompt with explicit violation fixes for iteration k + 1.
 */
export async function runClosedLoopPhaseTransition(
  text: string,
  targetPhase: UsirPhase,
  llmProvider: LlmProvider,
  options?: ClosedLoopOptions
): Promise<ClosedLoopResult> {
  const maxIterations = Math.max(1, options?.maxIterations ?? 3);
  const registerMode = options?.registerMode ?? 'classical_literary';

  const history: IterationRecord[] = [];
  const cumulativeViolations: MicroViolation[] = [];
  const seenViolationKeys = new Set<string>();

  // Initial prompt from Hamiltonian rewrite directives
  const initialDirectives = generateRewriteDirectives(text, targetPhase);
  let currentPrompt = initialDirectives.compiledPrompt;

  let finalText = text;
  let finalMetrics: UsirState = extractUsir24d(text);
  let finalPhase: UsirPhase = classifyPhase(finalMetrics.q.concat(finalMetrics.p)).phase;
  let finalMicroGate: MicroConstraintGateResult = runMicroConstraintGate(text, {
    targetPhase,
    registerMode,
  });
  let converged = false;

  for (let iter = 1; iter <= maxIterations; iter++) {
    // Generate candidate from LLM
    const rawCandidate = await llmProvider.generate(currentPrompt, iter);
    const candidate = rawCandidate.trim();

    // Evaluate macro physics
    const metrics = extractUsir24d(candidate);
    const diag = classifyPhase(metrics.z_24d);

    // Evaluate micro constraints
    const microGate = runMicroConstraintGate(candidate, {
      targetPhase,
      registerMode,
    });

    // Accumulate all prior violations across iterations to prevent oscillation
    for (const v of microGate.violations) {
      const key = `${v.type}:${v.token}`;
      if (!seenViolationKeys.has(key)) {
        seenViolationKeys.add(key);
        cumulativeViolations.push(v);
      }
    }

    const passedAll = evaluateConvergence(metrics, diag.phase, targetPhase, microGate);

    finalText = candidate;
    finalMetrics = {
      q: metrics.q,
      p: metrics.p,
      order_params: metrics.order_params,
      len_cv: metrics.len_cv,
    };
    finalPhase = diag.phase;
    finalMicroGate = microGate;

    history.push({
      iteration: iter,
      promptUsed: currentPrompt,
      candidateText: candidate,
      phase: diag.phase,
      lenCv: metrics.len_cv,
      phiVoid: metrics.order_params.phi_void,
      kappaKin: metrics.order_params.kappa_kin,
      abstractB: metrics.abstract_b_mean,
      microGate,
      passedAll,
    });

    if (passedAll) {
      converged = true;
      break;
    }

    // If not converged and more iterations remain, compile iterative feedback prompt
    if (iter < maxIterations) {
      let nextPrompt = compileIterativeFeedbackPrompt(
        text,
        candidate,
        targetPhase,
        iter
      );

      // Append cumulative micro-constraint gate violations (accumulated across all iterations)
      if (cumulativeViolations.length > 0) {
        const violationLines = cumulativeViolations.map(
          (v) =>
            `- [${v.type.toUpperCase()}] "${v.token}": ${v.message}${
              v.suggestion ? ` (Suggestion: ${v.suggestion})` : ''
            }`
        );

        nextPrompt += `\n\n[MICRO-CONSTRAINT GATE VIOLATIONS TO FIX]
The previous draft was REJECTED because it violated strict literary micro-constraints (${cumulativeViolations.length} cumulative violation(s) identified across iterations).
You MUST physically correct every violation below in this next revision, and ensure NONE of the previously corrected violations are reintroduced:
${violationLines.join('\n')}
Do NOT repeat these words or patterns. Replace them with concrete sensory or classical terms.`;
      }

      currentPrompt = nextPrompt;
    }
  }

  return {
    converged,
    iterationsUsed: history.length,
    finalText,
    finalPhase,
    finalMetrics,
    finalMicroGate,
    history,
  };
}
