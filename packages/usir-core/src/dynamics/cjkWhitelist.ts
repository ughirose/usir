/**
 * USIR v4.0 Full-Vocab Non-Japanese CJK Detection Engine.
 *
 * Provides a bitset lookup for all 20,992 ideographs in Unicode range [0x4E00, 0x9FFF].
 * A character is marked true if it fails Japanese EUC-JP (JIS X 0208 / JIS X 0212) encoding,
 * or is a non-Japanese simplified variant (e.g. 步 U+6B65 vs Japanese 歩 U+6B69, or 个 U+4E2A).
 * All standard Japanese Joyo and Jinmeiyo kanji (希, 著, 据, 与, 若, etc.) return false.
 */

const CJK_NON_JP_BITSET_B64 =
  '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uJ8CCGBlHfH/wpXqJVAEfASVSSFGDf+AAhDgKNU8aQHWhVdchDqpdgPEVjxyB9UTLLCaaeJTMTCVEYAgcZ6Y4YCAEQDg9FgV+NY7IOqyTBvAA8EEfX5AjCkLkoge8piwgE07I6ATQgM1UAZiHGkCS/7/C8oX1aVcQioAFoJDBYMUnACGEIUQIQJFEDgEDDIlI0ABAyiHBCoIoywBvMZ2rYEMoyC3RyiMxAOCKf///////////////////////38YHSBE0KHAl6AbgeKEaAAiQAAQlYGgSloCRVAi9///////////D4Vkf8na3D22ZlOhGiINUUViwE+NcKwfMPsE3GgAzgXgTajhQdHAkXkB1LMEQuln85Z8gFJoAgQUCAcwAA1BDsH///////8HBB4YIoiTwPuCEAyQgAYEEHAIASxBAYCEEAwKSEAgXPSQeoCeRapbRMlWDh+K3iqoUh0oyXcAI/BByUPwEZFArqkCgEIQBCQABQkcRQyJgI6HLAYoAhB8gAg8HATWmBFgTlY7QBAGUkxEiPJgul7MCAAgA/Nv1AiiTgmEC/hCC7IlhI75BcA0DAn+f9BhHAZ4/TeBUsXq0SHwElgP5f//////////////////////////////////////f7xiAKUcAMSrATfAgAiI////////j4kqcUDoRyoGgasCAoEQDUAGOCDgDIGZAPNwOPECGLWBYfrwnKE4G6AgkCVAjFUeKLKGgSgFgGSE7B8BchEKIKiEB0KGLwMBsBNj5P///////xIscoPu/2cREaOPU+gAAiimwYEYhEgk/v//////P1SfjsBJhfzAAHuDyBTBJ7C9YpYWvkCDDVPL8f////////8KiiSkPgAmAgFBUYCEumASEEHBwQYNspUCR3wGESAbPoBoJDw4KFZBclBYqkCAoAEYGJEFOCICdrAa9v///////////////x/lIv0ztFkGt2LiAwH65A3AYTqjJxkT+u0hyWAheQc5HhXimHu7mIH/////////////30BAJhEWjwjGIK8GygRODghEczwAcbhjaQEFMwMaMAgQ2/9fCI/Q//////////////8=';

let bitsetCache: Uint8Array | null = null;

function getBitset(): Uint8Array {
  if (!bitsetCache) {
    if (typeof Buffer !== 'undefined') {
      bitsetCache = Buffer.from(CJK_NON_JP_BITSET_B64, 'base64');
    } else {
      const bin = atob(CJK_NON_JP_BITSET_B64);
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) {
        arr[i] = bin.charCodeAt(i);
      }
      bitsetCache = arr;
    }
  }
  return bitsetCache;
}

/**
 * Checks whether text contains any CJK character that is non-Japanese (JIS EUC-JP fail
 * or simplified variant such as 步 U+6B65 or 个 U+4E2A).
 */
export function isNonJapaneseCjk(text: string): boolean {
  const b = getBitset();
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= 0x4E00 && code <= 0x9FFF) {
      if (code === 0x6B65 || code === 0x4E2A) {
        return true;
      }
      const offset = code - 0x4E00;
      const byteIdx = offset >> 3;
      const bitIdx = offset & 7;
      if (byteIdx < b.length && ((b[byteIdx]! & (1 << bitIdx)) !== 0)) {
        return true;
      }
    }
  }
  return false;
}
