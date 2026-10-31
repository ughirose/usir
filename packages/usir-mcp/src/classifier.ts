import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import {
  fitPhaseClassifier,
  predictPhase,
  type PhaseCentroidsAndCov,
  type UsirPhase,
} from '@nishiki/usir-core';

let cachedClassifier: PhaseCentroidsAndCov | null = null;

export function getUsirClassifier(): PhaseCentroidsAndCov {
  if (cachedClassifier) {
    return cachedClassifier;
  }

  // Resolve python_reference_40.json from fixtures
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const candidatePaths = [
    path.resolve(__dirname, '../../usir-core/tests/fixtures/python_reference_40.json'),
    path.resolve(__dirname, '../node_modules/@nishiki/usir-core/tests/fixtures/python_reference_40.json'),
    path.resolve(process.cwd(), 'packages/usir-core/tests/fixtures/python_reference_40.json'),
    path.resolve(process.cwd(), 'tests/fixtures/python_reference_40.json'),
  ];

  let fixturePath: string | null = null;
  for (const p of candidatePaths) {
    if (fs.existsSync(p)) {
      fixturePath = p;
      break;
    }
  }

  if (!fixturePath) {
    throw new Error(
      `Could not locate python_reference_40.json. Searched paths: ${candidatePaths.join(', ')}`
    );
  }

  const raw = fs.readFileSync(fixturePath, 'utf-8');
  const records: { phase: string; z: number[] }[] = JSON.parse(raw);

  cachedClassifier = fitPhaseClassifier(
    records.map((r) => ({
      phase: r.phase,
      features: r.z,
    }))
  );

  return cachedClassifier;
}

export function classifyUsirPhase(z: number[]): {
  phase: UsirPhase;
  distance: number;
} {
  const classifier = getUsirClassifier();
  return predictPhase(z, classifier);
}
