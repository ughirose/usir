/**
 * @nishiki/usir-core: Universal Style Genome (USIR / Nishiki)
 * Pure TypeScript Symplectic Feature Extraction, Hamiltonian Dynamics & Landau 5-Phase Engine.
 */

// Schema & Binary Serializers
export * from './schema/usirBinarySchema.js';
export * from './schema/syntaxFrame.js';

// Sensor Dictionaries & Primitives
export * from './sensor/dictionaries.js';
export * from './sensor/types.js';
export * from './sensor/microKasane.js';
export * from './sensor/mesoSasae.js';
export * from './sensor/macroKamae.js';
export * from './sensor/symplectic24d.js';
export {
  HTHPolyLexicon,
  type DualSenseEntry,
  type SenseResolutionResult,
  TOTAL_WORDS,
  BYTES_PER_ENTRY,
  EXPECTED_BINARY_SIZE,
  ENTRIES_PER_QUANTILE_BIN,
  lcaDepth64bit,
  jumpDistance64bit,
  packPoly64bit,
  unpackPoly64bit,
  sensoryNibblesToFloats,
} from './sensor/hthLexicon.js';

// Dynamics & Thermodynamics
export * from './dynamics/hamiltonian.js';
export * from './dynamics/landauPhase.js';
export * from './dynamics/equalizer.js';
export * from './dynamics/manifoldGeodesic.js';
export * from './dynamics/bidirectionalBridge.js';
export * from './dynamics/cjkWhitelist.js';

// Clean Room Gate & Micro-Constraint Gate
export * from './cleanroom/threeRoomGate.js';
export * from './cleanroom/microConstraintGate.js';

// Presets & Chemical PoE Mixer (TASK-361)
export * from './presets/canonicalParticles.js';

// Stage 3 Dynamics, HSMM & Closed-Loop
export * from './dynamics/closedLoopCorpus.js';
export * from './dynamics/trajectoryHsmm.js';
export * from './dynamics/hsmmLongformDirector.js';
export * from './dynamics/presalesShowcase.js';

// Stage 3 Style Genome Algebra & Compiler
export * from './algebra/genomeCompiler.js';
export * from './algebra/styleAlgebra.js';

// USIR v2.0 / v3.0 Paper-Grade Architecture
export * from './v2/index.js';


