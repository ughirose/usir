#!/usr/bin/env node
/**
 * CLI Entrypoint for @nishiki/usir-mcp: usir-audit
 */

import * as fs from 'fs';
import {
  inspectStyle24d,
  diagnosePhase,
  steerRewrite,
  startMcpStdio,
} from './index.js';
import {
  generateRewriteDirectives,
  extractTrajectoryWave,
  decodeHsmmModes,
  detectDeadzoneSinkholes,
  compileStyleGenomeToTierM,
  splitSentences,
  type UsirPhase,
} from '@nishiki/usir-core';

function printHelp() {
  console.log(`
USIR Style Audit CLI (usir-audit)
Usage:
  usir-audit inspect <file | text> [--json]
  usir-audit diagnose <file | text> [--json]
  usir-audit steer <file | text> [--target <phase>] [--json]
  usir-audit prompt <file | text> [--target <phase>] [--json]
  usir-audit extract-genome <file | text> -o <output.usir> [--json]
  usir-audit trajectory <file | text> [--json]
  usir-audit mcp-stdio

Subcommands:
  inspect         Extracts 24D symplectic state, order parameters, Hamiltonian, and Tier S 128B hash.
  diagnose        Diagnoses Landau 5-phase classification and AI Deadzone / Thermal Death indicators.
  steer           Applies canonical phase transition rewrite (default target: III_Void).
  prompt          Compiles Hamiltonian steering prompt with deficit breakdown.
  extract-genome  Compiles style corpus into a 4096-byte Tier M Style Genome binary profile.
  trajectory      Decodes 3-mode HSMM sliding-window trajectory and isolates Deadzone Sinkholes.
  mcp-stdio       Launches the MCP Server on stdio.

Options:
  -o, --output    Output path for compiled .usir binary profile (extract-genome).
  --target        Target phase for steering (III_Void, II_Critical, I_Crystalline). Default: III_Void.
  --json          Output raw JSON result.
  --help          Show this help message.
`);
}

function resolveText(arg: string): string {
  if (arg === '-' || !arg) {
    return fs.readFileSync(0, 'utf-8');
  }
  if (fs.existsSync(arg)) {
    return fs.readFileSync(arg, 'utf-8');
  }
  return arg;
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    printHelp();
    process.exit(0);
  }

  const subcommand = args[0];
  const isJson = args.includes('--json');

  if (subcommand === 'mcp-stdio') {
    await startMcpStdio();
    return;
  }

  const targetIdx = args.indexOf('--target');
  const targetPhase =
    targetIdx !== -1 && args[targetIdx + 1]
      ? (args[targetIdx + 1] as UsirPhase)
      : 'III_Void';

  const outIdx = args.indexOf('-o') !== -1 ? args.indexOf('-o') : args.indexOf('--output');
  const outputFile = outIdx !== -1 && args[outIdx + 1] ? args[outIdx + 1] : undefined;

  const nonFlagArgs = args.slice(1).filter((a, idx, arr) => {
    if (a.startsWith('-')) return false;
    const prev = arr[idx - 1];
    if (prev === '--target' || prev === '-o' || prev === '--output') return false;
    return true;
  });

  const rawInput = nonFlagArgs[0] ?? '-';
  const text = resolveText(rawInput);

  switch (subcommand) {
    case 'inspect': {
      const res = inspectStyle24d({ text });
      if (isJson) {
        console.log(JSON.stringify(res, null, 2));
      } else {
        console.log('=== USIR 24D SYMPLECTIC INSPECTION ===');
        console.log(
          `Order Params: psi=${res.order_params.psi_order.toFixed(3)}, phi_void=${res.order_params.phi_void.toFixed(3)}, kappa_kin=${res.order_params.kappa_kin.toFixed(3)}`
        );
        console.log(
          `4-Groups: Flux=${res.groups.I_Flux.toFixed(3)}, Lattice=${res.groups.II_Lattice.toFixed(3)}, Void=${res.groups.III_Void.toFixed(3)}, Shock=${res.groups.IV_Shock.toFixed(3)}`
        );
        console.log(
          `Hamiltonian Energy: T(p)=${res.hamiltonian.kineticEnergy.toFixed(3)}, V(q)=${res.hamiltonian.potentialEnergy.toFixed(3)}, Total=${res.hamiltonian.totalHamiltonian.toFixed(3)}`
        );
        console.log(
          `Tier S (128B) SHA-Digest: ${res.clean_room.tierSHex.slice(0, 32)}... (Audit Passed: ${res.clean_room.auditPassed})`
        );
      }
      break;
    }
    case 'diagnose': {
      const res = diagnosePhase({ text });
      if (isJson) {
        console.log(JSON.stringify(res, null, 2));
      } else {
        console.log('=== USIR LANDAU 5-PHASE DIAGNOSIS ===');
        console.log(`Predicted Phase : ${res.predictedPhase}`);
        console.log(`Mahalanobis Dist: ${res.distance.toFixed(4)}`);
        console.log(
          `Deadzone Alert  : ${res.isDeadzone ? 'YES (AI Slop Flatline)' : 'NO (Alive Cadence)'}`
        );
        console.log(
          `Metrics         : CV_L=${res.deadzone_diagnostics.len_cv.toFixed(3)}, AbstBloat=${res.deadzone_diagnostics.abstract_b_mean.toFixed(4)}, CadenceAlive=${res.deadzone_diagnostics.cadence_alive.toFixed(3)}`
        );
        console.log(`Prescription    : ${res.prescription}`);
      }
      break;
    }
    case 'steer': {
      const res = steerRewrite({ text, targetPhase });
      if (isJson) {
        console.log(JSON.stringify(res, null, 2));
      } else {
        console.log('=== USIR PHASE STEERING REWRITE ===');
        console.log(
          `Transition     : ${res.originalPhase} -> ${res.rewrittenPhase}`
        );
        console.log(
          `Before Metrics : CV_L=${res.beforeMetrics.len_cv.toFixed(3)}, phi_void=${res.beforeMetrics.phi_void.toFixed(3)}, AbstBloat=${res.beforeMetrics.abstract_b_mean.toFixed(4)}`
        );
        console.log(
          `After Metrics  : CV_L=${res.afterMetrics.len_cv.toFixed(3)}, phi_void=${res.afterMetrics.phi_void.toFixed(3)}, AbstBloat=${res.afterMetrics.abstract_b_mean.toFixed(4)}`
        );
        console.log(`Gate 4 Passed  : ${res.gate4Passed}`);
        console.log('\n--- Rewritten Text ---\n' + res.rewrittenText);
      }
      break;
    }
    case 'prompt': {
      const directives = generateRewriteDirectives(text, targetPhase);
      if (isJson) {
        console.log(JSON.stringify(directives, null, 2));
      } else {
        console.log(directives.compiledPrompt);
      }
      break;
    }
    case 'extract-genome': {
      let chunks: string[];
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed) && parsed.every((x) => typeof x === 'string')) {
          chunks = parsed;
        } else {
          chunks = text
            .split(/\n\n+/)
            .map((p) => p.trim())
            .filter((p) => p.length > 0);
        }
      } catch {
        chunks = text
          .split(/\n\n+/)
          .map((p) => p.trim())
          .filter((p) => p.length > 0);
      }

      // If single long text without \n\n paragraph breaks produces < 2 chunks,
      // gracefully fall back to sentence-window chunking
      if (chunks.length < 2) {
        const sents = splitSentences(text);
        if (sents.length >= 2) {
          const windowSize = Math.max(2, Math.min(3, Math.ceil(sents.length / 3)));
          const sentenceChunks: string[] = [];
          for (let i = 0; i < sents.length; i += windowSize) {
            const slice = sents.slice(i, i + windowSize);
            sentenceChunks.push(slice.join('。') + '。');
          }
          if (sentenceChunks.length >= 2) {
            chunks = sentenceChunks;
          }
        }
      }

      const outPath = outputFile ?? 'genome.usir';
      const compiled = compileStyleGenomeToTierM(chunks);
      fs.writeFileSync(outPath, compiled.binary);

      const auditPassed = compiled.audit.passed;
      if (isJson) {
        console.log(
          JSON.stringify(
            {
              output: outPath,
              byteLength: compiled.binary.byteLength,
              dominantPhase: compiled.dominantPhase,
              audit: compiled.audit,
              centroid24d: compiled.centroid24d,
            },
            null,
            2
          )
        );
      } else {
        console.log('=== USIR STYLE GENOME EXTRACTION (Tier M) ===');
        console.log(`Output File     : ${outPath} (${compiled.binary.byteLength} bytes)`);
        console.log(`Dominant Phase  : ${compiled.dominantPhase}`);
        console.log(
          `Clean Room Audit: ${auditPassed ? 'PASSED' : 'FAILED'} (7-gram leaks: ${compiled.audit.leaked7Grams.length}, banned words: ${compiled.audit.bannedWordsFound.length})`
        );
        console.log(
          `Centroid 24D    : [${compiled.centroid24d.slice(0, 4).map((v) => v.toFixed(3)).join(', ')}, ...]`
        );
      }
      break;
    }
    case 'trajectory': {
      const wave = extractTrajectoryWave(text);
      const hsmm = decodeHsmmModes(wave);
      const sinkholes = detectDeadzoneSinkholes(text);

      if (isJson) {
        console.log(
          JSON.stringify(
            {
              windowCount: wave.length,
              viterbiPath: hsmm.viterbiPath,
              modeNames: hsmm.modeNames,
              transitionMatrix: hsmm.transitionMatrix,
              modeWeights: hsmm.modeWeights,
              sinkholes: sinkholes.sinkholes,
              hasSinkholes: sinkholes.hasSinkholes,
              sinkholeCount: sinkholes.sinkholeCount,
            },
            null,
            2
          )
        );
      } else {
        console.log('=== USIR 3-MODE HSMM TRAJECTORY & SINKHOLE AUDIT ===');
        console.log(`Windows Analyzed : ${wave.length}`);
        console.log(`Viterbi Path     : [${hsmm.viterbiPath.join(', ')}]`);
        console.log(
          `Mode Weights     : Setup=${hsmm.modeWeights[0]!.toFixed(3)}, Rupture=${hsmm.modeWeights[1]!.toFixed(3)}, Deep Void=${hsmm.modeWeights[2]!.toFixed(3)}`
        );
        console.log(
          `Deadzone Sinkholes: ${sinkholes.hasSinkholes ? `${sinkholes.sinkholeCount} DETECTED` : 'NONE (Clean)'}`
        );
        if (sinkholes.hasSinkholes) {
          for (const s of sinkholes.sinkholes) {
            console.log(
              `  - Paragraph ${s.paragraphIndex} [sentences ${s.sentenceRange[0]}-${s.sentenceRange[1]}]: CV_L=${s.localCvL.toFixed(3)}, kappa=${s.localKappa.toFixed(3)} (${s.textExcerpt})`
            );
          }
        }
      }
      break;
    }
    default: {
      console.error(`Unknown subcommand: ${subcommand}`);
      printHelp();
      process.exit(1);
    }
  }
}

main().catch((err) => {
  console.error('[usir-audit error]', err);
  process.exit(1);
});
