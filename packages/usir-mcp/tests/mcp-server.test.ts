import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  inspectStyle24d,
  diagnosePhase,
  steerRewrite,
  handleToolCall,
  getToolDefinitions,
  createUsirMcpServer,
} from '../src/index.js';

describe('USIR MCP Server & Tool Engine Suite', () => {
  const fixturePath = path.resolve(
    __dirname,
    '../../usir-core/tests/fixtures/python_reference_40.json'
  );
  const fixtureData: any[] = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
  const dead01 = fixtureData.find((r) => r.id === 'DEAD-01')!;
  const void01 = fixtureData.find((r) => r.id === 'VOID-01')!;
  const void03 = fixtureData.find((r) => r.id === 'VOID-03')!;

  it('exposes exactly 5 tool definitions with strict JSON schemas', () => {
    const tools = getToolDefinitions();
    expect(tools.length).toBe(5);
    const names = tools.map((t) => t.name);
    expect(names).toContain('inspect_style_24d');
    expect(names).toContain('diagnose_phase');
    expect(names).toContain('steer_rewrite');
    expect(names).toContain('analyze_trajectory');
    expect(names).toContain('distill_genome');
  });

  it('initializes the McpServer instance cleanly', () => {
    const server = createUsirMcpServer();
    expect(server).toBeDefined();
  });

  describe('Tool 1: inspect_style_24d', () => {
    it('returns complete 24D symplectic state, order params, Hamiltonian, and Tier S digest', () => {
      const res = inspectStyle24d({ text: void03.text });

      expect(res.q.length).toBe(12);
      expect(res.p.length).toBe(12);
      expect(res.z.length).toBe(24);

      expect(res.order_params.phi_void).toBeGreaterThan(0.5);
      expect(res.groups.III_Void).toBeGreaterThan(0.5);

      expect(res.hamiltonian.totalHamiltonian).toBeGreaterThan(0.0);
      expect(res.hamiltonian.symplecticGradient.dq_dt.length).toBe(12);
      expect(res.hamiltonian.symplecticGradient.dp_dt.length).toBe(12);

      // Clean room 128B validation
      expect(res.clean_room.tierSByteLength).toBe(128);
      expect(res.clean_room.auditPassed).toBe(true);
      expect(res.clean_room.leaked7Grams.length).toBe(0);
      expect(res.clean_room.tierSHex.length).toBe(256); // 128 bytes in hex = 256 chars
    });

    it('works via handleToolCall dispatcher', () => {
      const res = handleToolCall('inspect_style_24d', { text: void01.text });
      expect(res.z.length).toBe(24);
    });
  });

  describe('Tool 2: diagnose_phase & Deadzone Detection', () => {
    it('classifies authentic Soseki passage as III_Void', () => {
      const res = diagnosePhase({ text: void03.text });
      expect(res.predictedPhase).toBe('III_Void');
      expect(res.isDeadzone).toBe(false);
      expect(res.prescription).toContain('Phase III (Superfluid Void [間])');
    });

    it('detects AI Deadzone (DEAD-01) with flatlined CV_L', () => {
      const res = diagnosePhase({ text: dead01.text });
      expect(res.predictedPhase).toBe('V_Deadzone');
      expect(res.isDeadzone).toBe(true);
      expect(res.deadzone_diagnostics.len_cv).toBeLessThan(0.28);
      expect(res.prescription).toContain('CRITICAL: Text is sedimented in Phase V');
    });
  });

  describe('Tool 3: steer_rewrite & Gate 4 E2E Phase Transition', () => {
    it('successfully rescues DEAD-01 into Phase III (Void) passing Gate 4 exit criteria', () => {
      const res = steerRewrite({
        text: dead01.text,
        targetPhase: 'III_Void',
      });

      // Initial state
      expect(res.originalPhase).toBe('V_Deadzone');
      expect(res.beforeMetrics.len_cv).toBeLessThan(0.28);
      expect(res.beforeMetrics.phi_void).toBeLessThan(0.15);

      // Gate 4 Exit Criteria:
      // 1. Len CV recovers from 0.13 to >= 0.40
      expect(res.afterMetrics.len_cv).toBeGreaterThanOrEqual(0.40);

      // 2. Void condensation phi_void transitions to >= 0.55
      expect(res.afterMetrics.phi_void).toBeGreaterThanOrEqual(0.55);

      // 3. Abstract filler bloat drops to <= 0.01
      expect(res.afterMetrics.abstract_b_mean).toBeLessThanOrEqual(0.01);

      // 4. Thermodynamic phase transitions to III_Void
      expect(res.rewrittenPhase).toBe('III_Void');

      // 5. Gate 4 assertion flag
      expect(res.gate4Passed).toBe(true);

      // Verify directives were populated
      expect(res.directives.mandatoryOperators).toContain('静か');
      expect(res.directives.purgeSluggishEndings).toContain('のだった');
    });

    it('applies custom equalizer gains cleanly', () => {
      const res = steerRewrite({
        text: dead01.text,
        targetPhase: 'II_Critical',
        gains: { shock: 1.5, flux: 1.2 },
      });
      expect(res.modulatedQ).toBeDefined();
      expect(res.modulatedQ.length).toBe(12);
    });
  });

  describe('CLI usir-audit Execution', () => {
    it('executes diagnose command via child_process returning JSON', async () => {
      const { execSync } = await import('child_process');
      const cliPath = path.resolve(__dirname, '../dist/cli.js');
      const cmd = `node "${cliPath}" diagnose "${void03.text.slice(0, 100)}" --json`;
      const stdout = execSync(cmd, { encoding: 'utf-8' });
      const parsed = JSON.parse(stdout);
      expect(parsed.predictedPhase).toBeDefined();
      expect(parsed.deadzone_diagnostics).toBeDefined();
    });
  });
});
