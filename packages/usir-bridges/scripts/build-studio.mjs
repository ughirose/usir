import * as fs from 'node:fs';
import * as path from 'node:path';
import { buildSync } from '../../../node_modules/.pnpm/esbuild@0.21.5/node_modules/esbuild/lib/main.js';

const repoRoot = 'C:/dev/usir';
const fixturePath = path.join(repoRoot, 'packages/usir-core/tests/fixtures/python_reference_40.json');
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
const getText = (id) => fixture.find((d) => d.id === id).text;

const presets = {
  'VOID-03': { id: 'VOID-03', title: '夏目漱石：草枕 [VOID-03]', expectedPhase: 'III_Void', text: getText('VOID-03') },
  'CRIT-05': { id: 'CRIT-05', title: '梶井基次郎：桜の樹の下には [CRIT-05]', expectedPhase: 'II_Critical', text: getText('CRIT-05') },
  'CRYS-01': { id: 'CRYS-01', title: '民法条文 [CRYS-01]', expectedPhase: 'I_Crystalline', text: getText('CRYS-01') },
  'DEAD-01': { id: 'DEAD-01', title: 'AI模倣スロップ [DEAD-01]', expectedPhase: 'V_Deadzone', text: getText('DEAD-01') },
  'HYBRID-5P': {
    id: 'HYBRID-5P',
    title: '5段落ハイブリッド長文 [中だるみ混入デモ]',
    expectedPhase: 'III_Void',
    text: [getText('VOID-01'), getText('CRIT-01'), getText('DEAD-01'), getText('VOID-02'), getText('CRIT-03')].join('\n\n'),
  },
  'EN-HEMINGWAY-VOID': {
    id: 'EN-HEMINGWAY-VOID',
    title: 'Ernest Hemingway: Big Two-Hearted River [EN-HEMINGWAY-VOID]',
    expectedPhase: 'III_Void',
    text: 'The river was green. Cold water ran over the white stones. Nick looked down through the dark pool and saw the trout. He did not move. The sun was hot on his back. There was no sound in the trees. The wind blew from the hills, cold and clean.',
  },
};

// 1. Bundle USIRCore
const coreBundleResult = buildSync({
  entryPoints: [path.join(repoRoot, 'packages/usir-core/dist/index.js')],
  bundle: true,
  format: 'iife',
  globalName: 'USIRCore',
  platform: 'browser',
  write: false,
});
const coreBundleJs = coreBundleResult.outputFiles[0].text;

// 2. Bundle USIRBridges (Browser safe)
const bridgesBundleResult = buildSync({
  entryPoints: [path.join(repoRoot, 'packages/usir-bridges/dist/browser.js')],
  bundle: true,
  format: 'iife',
  globalName: 'USIRBridges',
  platform: 'browser',
  write: false,
});
const bridgesBundleJs = bridgesBundleResult.outputFiles[0].text;

// 3. Load Template & App JS
const templateHtml = fs.readFileSync(path.join(repoRoot, 'packages/usir-bridges/scripts/studioTemplate.html'), 'utf-8');
const studioAppJs = fs.readFileSync(path.join(repoRoot, 'packages/usir-bridges/scripts/studioApp.js'), 'utf-8');

const scriptInjections = [
  `<script>window.__USIR_PRESETS__ = ${JSON.stringify(presets)};</script>`,
  `<script>${coreBundleJs}</script>`,
  `<script>${bridgesBundleJs}</script>`,
  `<script>${studioAppJs}</script>`,
].join('\n');

const html = templateHtml.replace('<!-- SCRIPT_INJECTIONS -->', () => scriptInjections);

const outDir = path.join(repoRoot, 'packages/usir-bridges/public');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'usir-studio.html'), html, 'utf-8');
console.log('Generated USIR Studio v3.1 successfully (' + html.length + ' bytes).');
