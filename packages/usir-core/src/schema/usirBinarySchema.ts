/**
 * Deterministic Binary Schema for Universal Style Genome (USIR / Nishiki).
 * Strict Fixed-Length Layouts:
 * - Tier S: 128 Bytes (Real-time 24D Symplectic State & Order Parameters)
 * - Tier M: 4096 Bytes (3-Mode HSMM, Precision Cholesky Factors & Style Profiles)
 */

import { z } from 'zod';

export const USIR_MAGIC = 0x55534952; // ASCII "USIR"
export const TIER_S_BYTE_LENGTH = 128;
export const TIER_M_BYTE_LENGTH = 4096;

// CRC32 Table & Computation (IEEE 802.3 standard)
const CRC32_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  CRC32_TABLE[i] = c >>> 0;
}

export function computeCrc32(buffer: Uint8Array, offset = 0, length = buffer.byteLength - offset): number {
  let crc = 0xffffffff;
  const end = offset + length;
  for (let i = offset; i < end; i++) {
    const byte = buffer[i]!;
    crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ byte) & 0xff]!;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// =============================================================================
// ZOD SCHEMAS
// =============================================================================

export const UsirOrderParamsSchema = z.object({
  psi_order: z.number(),
  phi_void: z.number(),
  kappa_kin: z.number(),
});

export const UsirStateSchema = z.object({
  q: z.array(z.number()).length(12),
  p: z.array(z.number()).length(12),
  order_params: UsirOrderParamsSchema,
  len_cv: z.number().default(0.5),
});

export type UsirState = z.infer<typeof UsirStateSchema>;

export interface DecodedTierS {
  version: number;
  tier: number;
  flags: number;
  q: number[];
  p: number[];
  order_params: {
    psi_order: number;
    phi_void: number;
    kappa_kin: number;
  };
  len_cv: number;
  crc32: number;
}

export interface TierMProfile {
  version?: number;
  flags?: number;
  transition_matrix: number[][]; // 3x3
  weights: number[]; // 3
  centroids: number[][]; // 3 x 24
  cholesky_factors: number[][]; // 3 x 300 lower-triangular elements
}

export interface DecodedTierM {
  version: number;
  tier: number;
  flags: number;
  transition_matrix: number[][];
  weights: number[];
  centroids: number[][];
  cholesky_factors: number[][];
  crc32: number;
}

// =============================================================================
// TIER S: 128 BYTES SERIALIZER / DESERIALIZER
// =============================================================================

/**
 * Encodes a 24D Symplectic state into a strictly 128-byte Uint8Array.
 * Layout:
 * - 0..3: Magic 0x55534952 (uint32 LE)
 * - 4..5: Version (uint16 LE) = 1
 * - 6: Tier (uint8) = 1
 * - 7: Flags (uint8) = 0
 * - 8..11: Reserved (uint32 LE) = 0
 * - 12..59: q (12 x float32 LE) = 48 bytes
 * - 60..107: p (12 x float32 LE) = 48 bytes
 * - 108..111: psi_order (float32 LE) = 4 bytes
 * - 112..115: phi_void (float32 LE) = 4 bytes
 * - 116..119: kappa_kin (float32 LE) = 4 bytes
 * - 120..123: len_cv (float32 LE) = 4 bytes
 * - 124..127: CRC32 of bytes 0..123 (uint32 LE) = 4 bytes
 * Total: 128 bytes.
 */
export function encodeTierS(state: UsirState): Uint8Array {
  const buf = new Uint8Array(TIER_S_BYTE_LENGTH);
  const view = new DataView(buf.buffer);

  // Header
  view.setUint32(0, USIR_MAGIC, true);
  view.setUint16(4, 1, true);
  view.setUint8(6, 1); // Tier S = 1
  view.setUint8(7, 0); // Flags
  view.setUint32(8, 0, true); // Reserved

  // q: 12 floats (offset 12)
  for (let i = 0; i < 12; i++) {
    view.setFloat32(12 + i * 4, state.q[i] ?? 0.0, true);
  }

  // p: 12 floats (offset 60)
  for (let i = 0; i < 12; i++) {
    view.setFloat32(60 + i * 4, state.p[i] ?? 0.0, true);
  }

  // Order params (offset 108)
  view.setFloat32(108, state.order_params.psi_order, true);
  view.setFloat32(112, state.order_params.phi_void, true);
  view.setFloat32(116, state.order_params.kappa_kin, true);

  // len_cv (offset 120)
  view.setFloat32(120, state.len_cv, true);

  // CRC32 Checksum over 0..123
  const crc = computeCrc32(buf, 0, 124);
  view.setUint32(124, crc, true);

  return buf;
}

export function decodeTierS(input: Uint8Array | ArrayBuffer): DecodedTierS {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.byteLength !== TIER_S_BYTE_LENGTH) {
    throw new Error(`Invalid Tier S length: expected ${TIER_S_BYTE_LENGTH} bytes, got ${bytes.byteLength}`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, true);
  if (magic !== USIR_MAGIC) {
    throw new Error(`Invalid USIR magic: expected 0x${USIR_MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
  }

  const version = view.getUint16(4, true);
  const tier = view.getUint8(6);
  const flags = view.getUint8(7);

  const expectedCrc = computeCrc32(bytes, 0, 124);
  const storedCrc = view.getUint32(124, true);
  if (expectedCrc !== storedCrc) {
    throw new Error(`CRC32 mismatch in Tier S: computed 0x${expectedCrc.toString(16)}, stored 0x${storedCrc.toString(16)}`);
  }

  const q: number[] = [];
  for (let i = 0; i < 12; i++) {
    q.push(view.getFloat32(12 + i * 4, true));
  }

  const p: number[] = [];
  for (let i = 0; i < 12; i++) {
    p.push(view.getFloat32(60 + i * 4, true));
  }

  const psi_order = view.getFloat32(108, true);
  const phi_void = view.getFloat32(112, true);
  const kappa_kin = view.getFloat32(116, true);
  const len_cv = view.getFloat32(120, true);

  return {
    version,
    tier,
    flags,
    q,
    p,
    order_params: {
      psi_order,
      phi_void,
      kappa_kin,
    },
    len_cv,
    crc32: storedCrc,
  };
}

// =============================================================================
// TIER M: 4096 BYTES SERIALIZER / DESERIALIZER
// =============================================================================

/**
 * Encodes a 3-Mode HSMM Style Profile into a strictly 4096-byte Uint8Array.
 * Layout:
 * - 0..3: Magic 0x55534952 (uint32 LE)
 * - 4..5: Version (uint16 LE) = 1
 * - 6: Tier (uint8) = 2
 * - 7: Flags (uint8) = 0
 * - 8..11: Reserved (uint32 LE) = 0
 * - 12..47: 3x3 Transition Matrix (9 x float32 LE) = 36 bytes
 * - 48..59: 3 Mode Weights (3 x float32 LE) = 12 bytes
 * - 60..347: 3 Mode Centroids (3 x 24 = 72 x float32 LE) = 288 bytes
 * - 348..3947: 3 Cholesky Factors L_k (3 x 300 = 900 x float32 LE) = 3600 bytes
 * - 3948..4091: Auxiliary Alignment / Padding (144 bytes)
 * - 4092..4095: CRC32 of bytes 0..4091 (uint32 LE) = 4 bytes
 * Total: 4096 bytes.
 */
export function encodeTierM(profile: TierMProfile): Uint8Array {
  const buf = new Uint8Array(TIER_M_BYTE_LENGTH);
  const view = new DataView(buf.buffer);

  // Header
  view.setUint32(0, USIR_MAGIC, true);
  view.setUint16(4, profile.version ?? 1, true);
  view.setUint8(6, 2); // Tier M = 2
  view.setUint8(7, profile.flags ?? 0);
  view.setUint32(8, 0, true);

  // Transition Matrix (3x3 = 9 floats, offset 12)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const val = profile.transition_matrix[r]?.[c] ?? (r === c ? 0.9 : 0.05);
      view.setFloat32(12 + (r * 3 + c) * 4, val, true);
    }
  }

  // Weights (3 floats, offset 48)
  for (let k = 0; k < 3; k++) {
    view.setFloat32(48 + k * 4, profile.weights[k] ?? 1.0 / 3.0, true);
  }

  // Centroids (3 x 24 = 72 floats, offset 60)
  for (let k = 0; k < 3; k++) {
    const centroid = profile.centroids[k] ?? new Array(24).fill(0.5);
    for (let d = 0; d < 24; d++) {
      view.setFloat32(60 + (k * 24 + d) * 4, centroid[d] ?? 0.0, true);
    }
  }

  // Cholesky Factors (3 x 300 = 900 floats, offset 348)
  for (let k = 0; k < 3; k++) {
    const factor = profile.cholesky_factors[k] ?? new Array(300).fill(0.0);
    for (let idx = 0; idx < 300; idx++) {
      view.setFloat32(348 + (k * 300 + idx) * 4, factor[idx] ?? 0.0, true);
    }
  }

  // CRC32 Checksum over 0..4091
  const crc = computeCrc32(buf, 0, 4092);
  view.setUint32(4092, crc, true);

  return buf;
}

export function decodeTierM(input: Uint8Array | ArrayBuffer): DecodedTierM {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.byteLength !== TIER_M_BYTE_LENGTH) {
    throw new Error(`Invalid Tier M length: expected ${TIER_M_BYTE_LENGTH} bytes, got ${bytes.byteLength}`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, true);
  if (magic !== USIR_MAGIC) {
    throw new Error(`Invalid USIR magic in Tier M: 0x${magic.toString(16)}`);
  }

  const version = view.getUint16(4, true);
  const tier = view.getUint8(6);
  const flags = view.getUint8(7);

  const expectedCrc = computeCrc32(bytes, 0, 4092);
  const storedCrc = view.getUint32(4092, true);
  if (expectedCrc !== storedCrc) {
    throw new Error(`CRC32 mismatch in Tier M: computed 0x${expectedCrc.toString(16)}, stored 0x${storedCrc.toString(16)}`);
  }

  const transition_matrix: number[][] = [[], [], []];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      transition_matrix[r]!.push(view.getFloat32(12 + (r * 3 + c) * 4, true));
    }
  }

  const weights: number[] = [];
  for (let k = 0; k < 3; k++) {
    weights.push(view.getFloat32(48 + k * 4, true));
  }

  const centroids: number[][] = [[], [], []];
  for (let k = 0; k < 3; k++) {
    for (let d = 0; d < 24; d++) {
      centroids[k]!.push(view.getFloat32(60 + (k * 24 + d) * 4, true));
    }
  }

  const cholesky_factors: number[][] = [[], [], []];
  for (let k = 0; k < 3; k++) {
    for (let idx = 0; idx < 300; idx++) {
      cholesky_factors[k]!.push(view.getFloat32(348 + (k * 300 + idx) * 4, true));
    }
  }

  return {
    version,
    tier,
    flags,
    transition_matrix,
    weights,
    centroids,
    cholesky_factors,
    crc32: storedCrc,
  };
}
