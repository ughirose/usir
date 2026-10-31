/**
 * @nishiki/usir-bridges: Narrative-Nano Lock-Free SPSC Ring Buffer Bridge
 * High-speed 128-byte Tier S frame streaming over ArrayBuffer / SharedArrayBuffer.
 * Zero heap allocation in streaming fast path.
 */

import {
  encodeTierS,
  decodeTierS,
  type UsirState,
  type DecodedTierS,
} from '@nishiki/usir-core';

export const NANO_RING_MAGIC = 0x4e414e4f; // ASCII "NANO"
export const NANO_FRAME_BYTE_LENGTH = 128; // Strictly 128 bytes (matching Tier S)
export const NANO_HEADER_BYTE_LENGTH = 64; // Aligned to 64-byte cache line

/**
 * Header layout in Uint32 indices (16 uint32s = 64 bytes):
 * - index 0 (0..3):   Magic (0x4e414e4f)
 * - index 1 (4..7):   Capacity (number of 128B slots)
 * - index 2 (8..11):  Frame size (128)
 * - index 3 (12..15): Write Sequence Counter (uint32)
 * - index 4 (16..19): Read Sequence Counter (uint32)
 * - index 5..15:      Reserved padding
 */
const IDX_MAGIC = 0;
const IDX_CAPACITY = 1;
const IDX_FRAME_SIZE = 2;
const IDX_WRITE_COUNTER = 3;
const IDX_READ_COUNTER = 4;

function loadU32(arr: Uint32Array, idx: number): number {
  try {
    return Atomics.load(arr, idx);
  } catch {
    return arr[idx] ?? 0;
  }
}

function storeU32(arr: Uint32Array, idx: number, val: number): void {
  try {
    Atomics.store(arr, idx, val >>> 0);
  } catch {
    arr[idx] = val >>> 0;
  }
}

export class NanoStyleRingBuffer {
  private readonly buffer: ArrayBufferLike;
  private readonly headerU32: Uint32Array;
  private readonly payloadU8: Uint8Array;
  public readonly capacity: number;
  private readonly internalScratchBuffer: Uint8Array;

  /**
   * Initializes or binds to an existing Narrative-Nano ring buffer.
   * @param bufferOrCapacity An existing ArrayBuffer/SharedArrayBuffer, or capacity in slots (default 32)
   */
  constructor(bufferOrCapacity: ArrayBufferLike | number = 32) {
    if (typeof bufferOrCapacity === 'number') {
      const cap = Math.max(2, bufferOrCapacity);
      const totalBytes = NANO_HEADER_BYTE_LENGTH + cap * NANO_FRAME_BYTE_LENGTH;
      // Default to SharedArrayBuffer if available, otherwise ArrayBuffer
      if (typeof SharedArrayBuffer !== 'undefined') {
        this.buffer = new SharedArrayBuffer(totalBytes);
      } else {
        this.buffer = new ArrayBuffer(totalBytes);
      }
      this.capacity = cap;
      this.headerU32 = new Uint32Array(this.buffer, 0, NANO_HEADER_BYTE_LENGTH / 4);
      this.payloadU8 = new Uint8Array(this.buffer, NANO_HEADER_BYTE_LENGTH);

      // Initialize header
      storeU32(this.headerU32, IDX_MAGIC, NANO_RING_MAGIC);
      storeU32(this.headerU32, IDX_CAPACITY, cap);
      storeU32(this.headerU32, IDX_FRAME_SIZE, NANO_FRAME_BYTE_LENGTH);
      storeU32(this.headerU32, IDX_WRITE_COUNTER, 0);
      storeU32(this.headerU32, IDX_READ_COUNTER, 0);
    } else {
      this.buffer = bufferOrCapacity;
      this.headerU32 = new Uint32Array(this.buffer, 0, NANO_HEADER_BYTE_LENGTH / 4);
      const magic = loadU32(this.headerU32, IDX_MAGIC);
      if (magic !== NANO_RING_MAGIC) {
        throw new Error(`Invalid Nano Ring Buffer magic: expected 0x${NANO_RING_MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
      }
      this.capacity = loadU32(this.headerU32, IDX_CAPACITY);
      this.payloadU8 = new Uint8Array(this.buffer, NANO_HEADER_BYTE_LENGTH);
    }

    this.internalScratchBuffer = new Uint8Array(NANO_FRAME_BYTE_LENGTH);
  }

  /**
   * Returns the underlying ArrayBuffer or SharedArrayBuffer.
   */
  public getRawBuffer(): ArrayBufferLike {
    return this.buffer;
  }

  /**
   * Total number of unread frames currently queued.
   */
  public availableRead(): number {
    const write = loadU32(this.headerU32, IDX_WRITE_COUNTER);
    const read = loadU32(this.headerU32, IDX_READ_COUNTER);
    return (write - read) >>> 0;
  }

  /**
   * Number of available frame slots for writing before buffer becomes full.
   */
  public availableWrite(): number {
    return this.capacity - this.availableRead();
  }

  public isFull(): boolean {
    return this.availableRead() >= this.capacity;
  }

  public isEmpty(): boolean {
    return this.availableRead() === 0;
  }

  /**
   * Clears the ring buffer by setting read and write sequence counters to 0.
   */
  public clear(): void {
    storeU32(this.headerU32, IDX_WRITE_COUNTER, 0);
    storeU32(this.headerU32, IDX_READ_COUNTER, 0);
  }

  /**
   * Writes a 128-byte frame into the ring buffer.
   * Returns true on success, or false if the buffer is full.
   */
  public writeFrame(frame: Uint8Array): boolean {
    if (frame.byteLength < NANO_FRAME_BYTE_LENGTH) {
      throw new Error(
        `Frame too short: expected ${NANO_FRAME_BYTE_LENGTH} bytes, got ${frame.byteLength}`
      );
    }

    const write = loadU32(this.headerU32, IDX_WRITE_COUNTER);
    const read = loadU32(this.headerU32, IDX_READ_COUNTER);

    if (((write - read) >>> 0) >= this.capacity) {
      return false; // Buffer full
    }

    const slot = write % this.capacity;
    const destOffset = slot * NANO_FRAME_BYTE_LENGTH;

    // Zero-heap direct byte transfer
    this.payloadU8.set(frame.subarray(0, NANO_FRAME_BYTE_LENGTH), destOffset);

    // Atomically increment write counter
    storeU32(this.headerU32, IDX_WRITE_COUNTER, (write + 1) >>> 0);
    return true;
  }

  /**
   * Reads the next 128-byte frame from the ring buffer.
   * If outBuffer is provided, writes into outBuffer with ZERO heap allocation.
   * Returns outBuffer (or a new Uint8Array) on success, or null if empty.
   */
  public readFrame(outBuffer?: Uint8Array): Uint8Array | null {
    const write = loadU32(this.headerU32, IDX_WRITE_COUNTER);
    const read = loadU32(this.headerU32, IDX_READ_COUNTER);

    if (write === read) {
      return null; // Buffer empty
    }

    const slot = read % this.capacity;
    const srcOffset = slot * NANO_FRAME_BYTE_LENGTH;

    const out = outBuffer ?? new Uint8Array(NANO_FRAME_BYTE_LENGTH);
    if (out.byteLength < NANO_FRAME_BYTE_LENGTH) {
      throw new Error(`Output buffer too short: expected at least ${NANO_FRAME_BYTE_LENGTH} bytes`);
    }

    out.set(this.payloadU8.subarray(srcOffset, srcOffset + NANO_FRAME_BYTE_LENGTH));

    // Atomically increment read counter
    storeU32(this.headerU32, IDX_READ_COUNTER, (read + 1) >>> 0);
    return out;
  }

  /**
   * High-level helper: encodes a UsirState into Tier S 128B binary and writes to ring buffer.
   */
  public writeTierS(state: UsirState): boolean {
    const tierSBytes = encodeTierS(state);
    return this.writeFrame(tierSBytes);
  }

  /**
   * High-level helper: reads the next Tier S 128B frame and decodes it.
   */
  public readTierS(): DecodedTierS | null {
    const frame = this.readFrame(this.internalScratchBuffer);
    if (!frame) {
      return null;
    }
    return decodeTierS(frame);
  }
}
