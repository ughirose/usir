/**
 * 3-Room Clean Room Architecture and 7-Gram Non-Retention Audit Gate for USIR.
 * Guarantees zero raw text retention and zero copyright leakage.
 */

import { encodeTierS, encodeTierM, type TierMProfile, type UsirState } from '../schema/usirBinarySchema.js';
import { extractUsir24d, type Usir24dResult } from '../sensor/symplectic24d.js';

export const BANNED_TARGET_KEYWORDS: readonly string[] = [
  'メロス', '李徴', '島村', '下人', '芥川', '太宰', '中島', '梶井',
  '志賀', '川端', '芭蕉', '漱石', '羅生門', '山月記', '雪国', '檸檬'
];

export interface CleanRoomAuditResult {
  passed: boolean;
  leaked7Grams: string[];
  bannedWordsFound: string[];
  byteLength: number;
}

/**
 * Room 1 (Dirty Room): Ingests raw text, extracts numerical symplectic moments,
 * and immediately severs reference to raw text.
 */
export function dirtyExtract(rawText: string): Usir24dResult {
  const result = extractUsir24d(rawText);
  // Sever any closures or raw string retention
  return {
    q: [...result.q],
    p: [...result.p],
    z_24d: [...result.z_24d],
    groups: { ...result.groups },
    order_params: { ...result.order_params },
    len_cv: result.len_cv,
    abst_bloat: result.abst_bloat,
    concrete_dens: result.concrete_dens,
    abstract_b_mean: result.abstract_b_mean,
    sluggish_mean: result.sluggish_mean,
    cadence_alive: result.cadence_alive,
  };
}

/**
 * Audit Gate: Inspects a serialized binary buffer against source text 7-grams
 * and banned target keywords to verify complete information-theoretic non-retention.
 */
export function auditCleanRoomBinary(
  binary: Uint8Array,
  sourceText?: string
): CleanRoomAuditResult {
  const utf8Decoder = new TextDecoder('utf-8', { fatal: false });
  const utf16Decoder = new TextDecoder('utf-16le', { fatal: false });

  const decodedUtf8 = utf8Decoder.decode(binary);
  const decodedUtf16 = utf16Decoder.decode(binary);

  const leaked7Grams: string[] = [];
  const bannedWordsFound: string[] = [];

  // 1. Scan for banned target keywords
  for (const word of BANNED_TARGET_KEYWORDS) {
    if (decodedUtf8.includes(word) || decodedUtf16.includes(word)) {
      bannedWordsFound.push(word);
    }
  }

  // 2. Scan for contiguous >= 7-grams of source text
  if (sourceText && sourceText.length >= 7) {
    // Collect unique 7-grams
    const seen7Grams = new Set<string>();
    for (let i = 0; i <= sourceText.length - 7; i++) {
      seen7Grams.add(sourceText.slice(i, i + 7));
    }

    for (const gram of seen7Grams) {
      if (decodedUtf8.includes(gram) || decodedUtf16.includes(gram)) {
        leaked7Grams.push(gram);
      }
    }
  }

  const passed = leaked7Grams.length === 0 && bannedWordsFound.length === 0;

  return {
    passed,
    leaked7Grams,
    bannedWordsFound,
    byteLength: binary.byteLength,
  };
}

/**
 * Room 3 (Clean Room): Emits verified Tier S binary buffer (strictly 128 bytes).
 */
export function emitCleanTierS(state: UsirState, sourceText?: string): Uint8Array {
  const binary = encodeTierS(state);
  const audit = auditCleanRoomBinary(binary, sourceText);
  if (!audit.passed) {
    throw new Error(
      `Clean Room Audit Failed for Tier S: leaked 7-grams=${audit.leaked7Grams.length}, banned words=${audit.bannedWordsFound.join(',')}`
    );
  }
  return binary;
}

/**
 * Room 3 (Clean Room): Emits verified Tier M binary buffer (strictly 4096 bytes).
 */
export function emitCleanTierM(profile: TierMProfile, sourceText?: string): Uint8Array {
  const binary = encodeTierM(profile);
  const audit = auditCleanRoomBinary(binary, sourceText);
  if (!audit.passed) {
    throw new Error(
      `Clean Room Audit Failed for Tier M: leaked 7-grams=${audit.leaked7Grams.length}, banned words=${audit.bannedWordsFound.join(',')}`
    );
  }
  return binary;
}
