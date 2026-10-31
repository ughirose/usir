/**
 * @nishiki/usir-mcp: Standalone MCP Server & CLI Style Audit Gate for USIR.
 */

import { z } from 'zod';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  extractUsir24d,
  computeHamiltonian,
  dirtyExtract,
  emitCleanTierS,
  auditCleanRoomBinary,
  applyEqualizer,
  generateRewriteDirectives,
  compileIterativeFeedbackPrompt,
  steerPhaseRewrite,
  extractTrajectoryWave,
  decodeHsmmModes,
  detectDeadzoneSinkholes,
  compileStyleGenomeToTierM,
  type UsirPhase,
  type EqualizerGains,
} from '@nishiki/usir-core';
import { classifyUsirPhase, getUsirClassifier } from './classifier.js';

export * from './runtimeLoop.js';

// =============================================================================
// PROGRAMMATIC TOOL LOGIC
// =============================================================================

export interface InspectStyle24dInput {
  text: string;
}

export function inspectStyle24d(input: InspectStyle24dInput) {
  const extracted = extractUsir24d(input.text);
  const hamiltonian = computeHamiltonian(extracted.q, extracted.p);

  // Clean room Tier S emission and verification
  const state = dirtyExtract(input.text);
  const tierSBuffer = emitCleanTierS(state, input.text);
  const audit = auditCleanRoomBinary(tierSBuffer, input.text);

  const hexDigest = Array.from(tierSBuffer)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  const base64Digest = Buffer.from(tierSBuffer).toString('base64');

  return {
    q: extracted.q,
    p: extracted.p,
    z: extracted.z_24d,
    order_params: extracted.order_params,
    groups: extracted.groups,
    hamiltonian: {
      kineticEnergy: hamiltonian.kineticEnergy,
      potentialEnergy: hamiltonian.potentialEnergy,
      totalHamiltonian: hamiltonian.totalHamiltonian,
      symplecticGradient: hamiltonian.symplecticGradient,
    },
    clean_room: {
      tierSByteLength: tierSBuffer.byteLength,
      tierSHex: hexDigest,
      tierSBase64: base64Digest,
      auditPassed: audit.passed,
      leaked7Grams: audit.leaked7Grams,
    },
    diagnostics: {
      len_cv: extracted.len_cv,
      cadence_alive: extracted.cadence_alive,
      abstract_b_mean: extracted.abstract_b_mean,
      sluggish_mean: extracted.sluggish_mean,
      concrete_dens: extracted.concrete_dens,
    },
  };
}

export interface DiagnosePhaseInput {
  text: string;
}

export function diagnosePhase(input: DiagnosePhaseInput) {
  const extracted = extractUsir24d(input.text);
  const classifier = getUsirClassifier();
  const { phase: predictedPhase, distance } = classifyUsirPhase(extracted.z_24d);

  // Calculate distances to all 5 phases for transparency
  const allPhaseDistances: Record<string, number> = {};
  const D = extracted.z_24d.length;
  const x_norm = extracted.z_24d.map(
    (v, d) => (v - classifier.mu_tr[d]!) / classifier.sd_tr[d]!
  );

  for (const c of classifier.classes) {
    const centroid = classifier.centroids.get(c)!;
    const diff = x_norm.map((v, d) => v - centroid[d]!);
    let dist = 0.0;
    for (let r = 0; r < D; r++) {
      let rowDot = 0.0;
      const row = classifier.inv_cov[r]!;
      for (let j = 0; j < D; j++) {
        rowDot += row[j]! * diff[j]!;
      }
      dist += diff[r]! * rowDot;
    }
    allPhaseDistances[c] = dist;
  }

  const isDeadzone =
    predictedPhase === 'V_Deadzone' ||
    (extracted.len_cv < 0.28 && extracted.cadence_alive < 0.5);

  let prescription = '';
  if (isDeadzone) {
    prescription =
      'CRITICAL: Text is sedimented in Phase V (AI Deadzone / Thermal Death). Sentence length volatility is flatlined (CV_L < 0.28) and abstract filler bloat has suppressed momentum. Steer immediately to Phase III (Superfluid Void) to cut explanatory filler and recover rhythm, or to Phase II (Critical) for dramatic impulse.';
  } else if (predictedPhase === 'III_Void') {
    prescription =
      'Phase III (Superfluid Void [間]): Excellent subtraction and high Wago simplex stem ratio. Preserves quietude and concrete sensory stillness.';
  } else if (predictedPhase === 'II_Critical') {
    prescription =
      'Phase II (Self-Organized Criticality): Dynamic literary tension with high 1/f rhythm volatility and dramatic synesthetic leaps.';
  } else if (predictedPhase === 'I_Crystalline') {
    prescription =
      'Phase I (Crystalline Order): High logical symmetry, bilateral deduction, and formal axiomatic purity.';
  } else {
    prescription =
      'Phase IV (Turbulent / Chaotic): Excessive ornamentation, purple prose saturation, or hyperactive rupture. Dampen shock operators.';
  }

  return {
    predictedPhase,
    distance,
    allPhaseDistances,
    order_params: extracted.order_params,
    deadzone_diagnostics: {
      isDeadzone,
      len_cv: extracted.len_cv,
      cadence_alive: extracted.cadence_alive,
      abstract_b_mean: extracted.abstract_b_mean,
      sluggish_mean: extracted.sluggish_mean,
    },
    isDeadzone,
    prescription,
  };
}

export interface SteerRewriteInput {
  text: string;
  targetPhase?: UsirPhase;
  gains?: Partial<EqualizerGains>;
}

export function steerRewrite(input: SteerRewriteInput) {
  const targetPhase: UsirPhase = input.targetPhase ?? 'III_Void';

  const beforeExtracted = extractUsir24d(input.text);
  const beforePred = classifyUsirPhase(beforeExtracted.z_24d);
  const originalPhase = beforePred.phase;

  const beforeMetrics = {
    len_cv: beforeExtracted.len_cv,
    phi_void: beforeExtracted.order_params.phi_void,
    abstract_b_mean: beforeExtracted.abstract_b_mean,
    cadence_alive: beforeExtracted.cadence_alive,
  };

  const directives = generateRewriteDirectives(input.text, targetPhase);
  const rewrittenText = steerPhaseRewrite(input.text, targetPhase);

  const afterExtracted = extractUsir24d(rewrittenText);
  const afterPred = classifyUsirPhase(afterExtracted.z_24d);
  const rewrittenPhase = afterPred.phase;

  const afterMetrics = {
    len_cv: afterExtracted.len_cv,
    phi_void: afterExtracted.order_params.phi_void,
    abstract_b_mean: afterExtracted.abstract_b_mean,
    cadence_alive: afterExtracted.cadence_alive,
  };

  const gate4Passed =
    targetPhase === 'III_Void'
      ? afterMetrics.len_cv >= 0.40 &&
        afterMetrics.phi_void >= 0.55 &&
        afterMetrics.abstract_b_mean <= 0.01 &&
        rewrittenPhase === 'III_Void'
      : rewrittenPhase === targetPhase;

  // Optional: compute modulated q state if custom gains provided
  const modulatedQ = input.gains
    ? applyEqualizer(afterExtracted.q, input.gains)
    : afterExtracted.q;

  return {
    originalPhase,
    rewrittenPhase,
    rewrittenText,
    beforeMetrics,
    afterMetrics,
    directives,
    gate4Passed,
    modulatedQ,
  };
}

export interface CompileFeedbackPromptInput {
  originalText: string;
  previousAttempt: string;
  targetPhase?: UsirPhase;
  iteration?: number;
}

export function compileFeedbackPrompt(input: CompileFeedbackPromptInput) {
  const targetPhase = input.targetPhase ?? 'III_Void';
  const iteration = input.iteration ?? 1;
  const prompt = compileIterativeFeedbackPrompt(
    input.originalText,
    input.previousAttempt,
    targetPhase,
    iteration
  );
  return { prompt, targetPhase, iteration };
}

export interface AnalyzeTrajectoryInput {
  text: string;
  windowSize?: number;
  stride?: number;
}

export function analyzeTrajectory(input: AnalyzeTrajectoryInput) {
  const wave = extractTrajectoryWave(input.text, {
    windowSize: input.windowSize,
    stride: input.stride,
  });
  const hsmm = decodeHsmmModes(wave);
  const sinkholes = detectDeadzoneSinkholes(input.text);
  return {
    windowCount: wave.length,
    viterbiPath: hsmm.viterbiPath,
    modeNames: hsmm.modeNames,
    transitionMatrix: hsmm.transitionMatrix,
    modeWeights: hsmm.modeWeights,
    modeCentroids: hsmm.modeCentroids,
    choleskyFactors: hsmm.choleskyFactors,
    sinkholeAnalysis: sinkholes,
  };
}

export interface DistillGenomeInput {
  textOrChunks: string | string[];
  genomeName?: string;
}

export function distillGenome(input: DistillGenomeInput) {
  const chunks: string[] = Array.isArray(input.textOrChunks)
    ? input.textOrChunks
    : input.textOrChunks
        .split(/\n\n+/)
        .map((p) => p.trim())
        .filter((p) => p.length > 0);

  const compiled = compileStyleGenomeToTierM(chunks, {
    genomeName: input.genomeName,
  });
  return {
    byteLength: compiled.binary.byteLength,
    dominantPhase: compiled.dominantPhase,
    centroid24d: compiled.centroid24d,
    audit: compiled.audit,
    tierMBase64: Buffer.from(compiled.binary).toString('base64'),
    tierMHex: Buffer.from(compiled.binary).toString('hex'),
  };
}

// =============================================================================
// PROGRAMMATIC DISPATCHER
// =============================================================================

export function handleToolCall(toolName: string, args: Record<string, any>) {
  switch (toolName) {
    case 'inspect_style_24d': {
      if (typeof args.text !== 'string') {
        throw new Error('inspect_style_24d requires a "text" string property.');
      }
      return inspectStyle24d({ text: args.text });
    }
    case 'diagnose_phase': {
      if (typeof args.text !== 'string') {
        throw new Error('diagnose_phase requires a "text" string property.');
      }
      return diagnosePhase({ text: args.text });
    }
    case 'steer_rewrite': {
      if (typeof args.text !== 'string') {
        throw new Error('steer_rewrite requires a "text" string property.');
      }
      return steerRewrite({
        text: args.text,
        targetPhase: args.targetPhase,
        gains: args.gains,
      });
    }
    case 'analyze_trajectory': {
      if (typeof args.text !== 'string') {
        throw new Error('analyze_trajectory requires a "text" string property.');
      }
      return analyzeTrajectory({
        text: args.text,
        windowSize: args.windowSize,
        stride: args.stride,
      });
    }
    case 'distill_genome': {
      const input = args.textOrChunks ?? args.text;
      if (!input) {
        throw new Error('distill_genome requires "textOrChunks" or "text".');
      }
      return distillGenome({
        textOrChunks: input,
        genomeName: args.genomeName,
      });
    }
    default:
      throw new Error(`Unknown tool name: ${toolName}`);
  }
}

export function getToolDefinitions() {
  return [
    {
      name: 'inspect_style_24d',
      description:
        'Calculates the 24-dimensional symplectic state (q, p), 3 order parameters (psi, phi_void, kappa_kin), Hamiltonian energy, and clean-room Tier S 128B binary digest from text.',
      inputSchema: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Input prose text to analyze' },
        },
        required: ['text'],
      },
    },
    {
      name: 'diagnose_phase',
      description:
        'Classifies text into one of the 5 Landau thermodynamic phases and detects AI Deadzone / Thermal Death flatlines (CV_L < 0.28).',
      inputSchema: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Input text to diagnose' },
        },
        required: ['text'],
      },
    },
    {
      name: 'steer_rewrite',
      description:
        'Applies Hamiltonian 4-group equalization and canonical phase steering to rescue AI Deadzone passages into Phase III (Void) or Phase II (Critical).',
      inputSchema: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Input text to modulate' },
          targetPhase: {
            type: 'string',
            enum: ['I_Crystalline', 'II_Critical', 'III_Void', 'IV_Turbulent', 'V_Deadzone'],
            description: 'Target thermodynamic phase',
          },
          gains: {
            type: 'object',
            properties: {
              flux: { type: 'number' },
              lattice: { type: 'number' },
              void: { type: 'number' },
              shock: { type: 'number' },
            },
            description: 'Custom 4-group gain sliders',
          },
        },
        required: ['text'],
      },
    },
    {
      name: 'analyze_trajectory',
      description:
        'Calculates sliding-window 3-mode HSMM trajectory (Setup, Rupture, Deep Void) and detects Deadzone Sinkholes.',
      inputSchema: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Input text to analyze' },
          windowSize: { type: 'number', description: 'Window size in sentences' },
          stride: { type: 'number', description: 'Window stride in sentences' },
        },
        required: ['text'],
      },
    },
    {
      name: 'distill_genome',
      description:
        'Compiles style corpus text or chunks into a strict 4096-byte Tier M Style Genome binary profile (.usir).',
      inputSchema: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Input corpus text or JSON array' },
          genomeName: { type: 'string', description: 'Optional name identifier' },
        },
        required: ['text'],
      },
    },
  ];
}

// =============================================================================
// MCP SERVER INITIALIZATION
// =============================================================================

export function createUsirMcpServer(): McpServer {
  const server = new McpServer({
    name: 'usir-mcp',
    version: '0.1.0',
  });

  server.tool(
    'inspect_style_24d',
    'Calculates 24D symplectic state, order parameters, Hamiltonian, and Tier S 128B binary digest.',
    {
      text: z.string().describe('Input prose text to inspect'),
    },
    async ({ text }) => {
      const res = inspectStyle24d({ text });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(res, null, 2),
          },
        ],
      };
    }
  );

  server.tool(
    'diagnose_phase',
    'Classifies text into one of 5 Landau phases and returns AI Deadzone diagnostics.',
    {
      text: z.string().describe('Input text to diagnose'),
    },
    async ({ text }) => {
      const res = diagnosePhase({ text });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(res, null, 2),
          },
        ],
      };
    }
  );

  server.tool(
    'steer_rewrite',
    'Performs canonical phase transition rewriting and equalizer modulation.',
    {
      text: z.string().describe('Input text to modulate'),
      targetPhase: z
        .enum(['I_Crystalline', 'II_Critical', 'III_Void', 'IV_Turbulent', 'V_Deadzone'])
        .optional()
        .describe('Target thermodynamic phase'),
      gains: z
        .object({
          flux: z.number().optional(),
          lattice: z.number().optional(),
          void: z.number().optional(),
          shock: z.number().optional(),
        })
        .optional()
        .describe('Custom 4-group equalizer gains'),
    },
    async ({ text, targetPhase, gains }) => {
      const res = steerRewrite({
        text,
        targetPhase: targetPhase as UsirPhase | undefined,
        gains: gains
          ? {
              alpha_flux: gains.flux,
              alpha_lattice: gains.lattice,
              alpha_void: gains.void,
              alpha_shock: gains.shock,
            }
          : undefined,
      });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(res, null, 2),
          },
        ],
      };
    }
  );

  server.tool(
    'analyze_trajectory',
    'Calculates sliding-window 3-mode HSMM trajectory (Setup, Rupture, Deep Void) and detects Deadzone Sinkholes.',
    {
      text: z.string().describe('Input text to analyze'),
      windowSize: z.number().optional().describe('Window size in sentences (default: 3)'),
      stride: z.number().optional().describe('Window stride in sentences (default: 1)'),
    },
    async ({ text, windowSize, stride }) => {
      const res = analyzeTrajectory({ text, windowSize, stride });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(res, null, 2),
          },
        ],
      };
    }
  );

  server.tool(
    'distill_genome',
    'Compiles style corpus text or chunks into a strict 4096-byte Tier M Style Genome binary profile (.usir).',
    {
      text: z.string().describe('Input corpus text (multi-paragraph or JSON array of strings)'),
      genomeName: z.string().optional().describe('Optional name identifier for the genome'),
    },
    async ({ text, genomeName }) => {
      let chunks: string | string[] = text;
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed) && parsed.every((x) => typeof x === 'string')) {
          chunks = parsed;
        }
      } catch {
        // use raw text
      }
      const res = distillGenome({ textOrChunks: chunks, genomeName });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(res, null, 2),
          },
        ],
      };
    }
  );

  return server;
}

export async function startMcpStdio(): Promise<void> {
  const server = createUsirMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('[USIR-MCP] Universal Style Genome MCP Server running on stdio.');
}
