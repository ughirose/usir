import { describe, it, expect } from 'vitest';
import {
  generateVoronoiGenesisArtifact,
  synthesizeMusicFromUsir24d,
  encodeCompositionToMidiBytes,
  createMidiDataUri,
  createWebAudioSynthSourceCode,
  playCompositionOnWebAudio,
  encodeVlq,
  computeMusicStats,
} from '../src/multimodalAudioVideo.js';

describe('Workstream 2: Voronoi Genesis Generator & Web Audio / MIDI Synthesizer', () => {
  // ============================================================================
  // 1. VLQ & Basic MIDI Encoding Primitives
  // ============================================================================
  it('correctly encodes Variable-Length Quantities (VLQ) for MIDI delta times', () => {
    expect(encodeVlq(0)).toEqual([0x00]);
    expect(encodeVlq(127)).toEqual([0x7f]);
    expect(encodeVlq(128)).toEqual([0x81, 0x00]);
    expect(encodeVlq(480)).toEqual([0x83, 0x60]);
    expect(encodeVlq(16383)).toEqual([0xff, 0x7f]);
    expect(encodeVlq(16384)).toEqual([0x81, 0x80, 0x00]);
  });

  // ============================================================================
  // 2. Voronoi Genesis Prose & Style Naming Generator
  // ============================================================================
  it('generates distinct Voronoi Genesis artifacts, prose, and naming across different 24D vectors', () => {
    // Vector A: Superfluid Void / Yugen dominant (high Sg-03 Pr, Sg-08 Lp, Sg-11 Ma)
    const zVoid = new Array(24).fill(0.05);
    zVoid[2] = 0.88; // Sg-03 Pr
    zVoid[7] = 0.82; // Sg-08 Lp
    zVoid[10] = 0.95; // Sg-11 Ma
    zVoid[12 + 7] = 0.45; // p_8

    // Vector B: Classical Crystalline / Contrast dominant (high Sg-02 Cr, Sg-06 Sy, Sg-10 At)
    const zCrystal = new Array(24).fill(0.05);
    zCrystal[1] = 0.92; // Sg-02 Cr
    zCrystal[5] = 0.85; // Sg-06 Sy
    zCrystal[9] = 0.89; // Sg-10 At

    // Vector C: Cathartic Rupture / Iki dominant (high Sg-01 Gr, Sg-07 Cs, Sg-12 Ru)
    const zRupture = new Array(24).fill(0.05);
    zRupture[0] = 0.78; // Sg-01 Gr
    zRupture[6] = 0.85; // Sg-07 Cs
    zRupture[11] = 0.96; // Sg-12 Ru
    zRupture[12 + 11] = 0.65; // p_12

    const artVoid = generateVoronoiGenesisArtifact(zVoid, 'ja', 1);
    const artCrystal = generateVoronoiGenesisArtifact(zCrystal, 'ja', 2);
    const artRupture = generateVoronoiGenesisArtifact(zRupture, 'ja', 3);

    // Style ID formatting
    expect(artVoid.styleId).toBe('GENESIS-01');
    expect(artCrystal.styleId).toBe('GENESIS-02');
    expect(artRupture.styleId).toBe('GENESIS-03');

    // Distinct Style Names
    expect(artVoid.styleNameJa).not.toEqual(artCrystal.styleNameJa);
    expect(artVoid.styleNameJa).not.toEqual(artRupture.styleNameJa);
    expect(artCrystal.styleNameJa).not.toEqual(artRupture.styleNameJa);

    expect(artVoid.styleNameEn).not.toEqual(artCrystal.styleNameEn);
    expect(artVoid.styleNameEn).not.toEqual(artRupture.styleNameEn);

    // Check inclusion of uncharted bracket markers
    expect(artVoid.styleNameJa).toMatch(/^【未踏様式 #01: .*[（\(].*[）\)】]$/);
    expect(artVoid.styleNameEn).toMatch(/^\[Uncharted Style #01: .*\]$/);

    // Check dominant elements
    expect(artVoid.dominantElements.length).toBe(3);
    const topVoidIds = artVoid.dominantElements.map((e) => e.id);
    expect(topVoidIds).toContain('Sg-11');
    expect(topVoidIds).toContain('Sg-03');

    const topCrystalIds = artCrystal.dominantElements.map((e) => e.id);
    expect(topCrystalIds).toContain('Sg-02');
    expect(topCrystalIds).toContain('Sg-10');

    const topRuptureIds = artRupture.dominantElements.map((e) => e.id);
    expect(topRuptureIds).toContain('Sg-12');

    // Check molecule profile
    expect(artVoid.moleculeProfile.yugen).toBeGreaterThan(0.0);
    expect(artVoid.moleculeProfile.dominantMolecule).toBeDefined();

    // Check generated prose distinctiveness & non-emptiness
    expect(artVoid.generatedProseJa.length).toBeGreaterThan(30);
    expect(artVoid.generatedProseEn.length).toBeGreaterThan(30);
    expect(artVoid.generatedProseJa).not.toEqual(artCrystal.generatedProseJa);
    expect(artVoid.generatedProseEn).not.toEqual(artCrystal.generatedProseEn);

    // Check Hamiltonian directive
    expect(artVoid.hamiltonianDirective.targetCvL).toBeGreaterThan(0.2);
    expect(artVoid.hamiltonianDirective.targetVoidPauseRatio).toBeGreaterThan(0.4);
    expect(artVoid.hamiltonianDirective.syntaxHopDistance).toBeGreaterThan(1.0);
    expect(artVoid.hamiltonianDirective.terminalAnchor).toBeDefined();
    expect(artVoid.hamiltonianDirective.instructionsJa).toContain('文長変動係数');
    expect(artVoid.hamiltonianDirective.instructionsEn).toContain('Sentence Length');
  });

  // ============================================================================
  // 3. Musical Statistics Across Acoustic Regimes
  // ============================================================================
  it('strictly realizes distinct musical statistics across the three acoustic regimes', () => {
    const dummyZ = new Array(24).fill(0.1);

    // 1. Regime V: AI Deadzone
    const compDeadzone = synthesizeMusicFromUsir24d(dummyZ, 'V_Deadzone', 16.0);
    expect(compDeadzone.mode).toBe('V_Deadzone');
    expect(compDeadzone.bpm).toBe(120);
    expect(compDeadzone.reverbWet).toBe(0.05);

    // Strict requirements: restRatio == 0, velocityStd == 0, ioiVariance == 0
    expect(compDeadzone.stats.restRatio).toBe(0.0);
    expect(compDeadzone.stats.velocityStd).toBe(0.0);
    expect(compDeadzone.stats.meanVelocity).toBe(64.0);
    expect(compDeadzone.stats.ioiVariance).toBe(0.0);
    expect(compDeadzone.stats.maxIntervalLeap).toBeLessThanOrEqual(5);

    // Every melody note must have flat velocity 64 and adjacent notes contiguous in time
    const deadNotes = compDeadzone.tracks[0]!.notes;
    expect(deadNotes.length).toBeGreaterThan(20);
    for (let i = 0; i < deadNotes.length; i++) {
      expect(deadNotes[i]!.velocity).toBe(64);
      if (i > 0) {
        // Contiguous in time: no gap between notes
        const diff = Math.abs(deadNotes[i]!.startTime - (deadNotes[i - 1]!.startTime + deadNotes[i - 1]!.duration));
        expect(diff).toBeLessThan(1e-4);
      }
    }

    // 2. Regime III: Superfluid Void
    const zVoid = new Array(24).fill(0.05);
    zVoid[2] = 0.85;
    zVoid[10] = 0.90;
    const compVoid = synthesizeMusicFromUsir24d(zVoid, 'III_Void', 20.0);
    expect(compVoid.mode).toBe('III_Void');
    expect(compVoid.bpm).toBeGreaterThanOrEqual(54);
    expect(compVoid.bpm).toBeLessThanOrEqual(68);
    expect(compVoid.reverbWet).toBeGreaterThanOrEqual(0.60);

    // Strict requirements: restRatio >= 0.40, pentatonic scale
    expect(compVoid.stats.restRatio).toBeGreaterThanOrEqual(0.40);
    expect(compVoid.stats.velocityStd).toBeGreaterThan(0.0); // Gentle expressive dynamics
    expect(compVoid.stats.meanVelocity).toBeLessThan(60); // Gentle pianissimo/mezzo-piano

    // Pitch classes must belong to Miyako-bushi / In-sen scale [0, 1, 5, 7, 10]
    const voidNotes = compVoid.tracks[0]!.notes;
    expect(voidNotes.length).toBeGreaterThan(5);
    const miyakoBushiPitchClasses = new Set([0, 1, 5, 7, 10]); // C, Db, F, G, Bb
    for (const note of voidNotes) {
      expect(miyakoBushiPitchClasses.has(note.pitch % 12)).toBe(true);
    }

    // 3. Regime II: Self-Organized Critical / Genesis
    const zCrit = new Array(24).fill(0.2);
    zCrit[7] = 0.92; // Sg-08 Lp
    zCrit[11] = 0.88; // Sg-12 Ru
    zCrit[8] = 0.85; // Sg-09 Dy
    zCrit[12 + 7] = 0.65;
    zCrit[12 + 11] = 0.75;
    const compCritical = synthesizeMusicFromUsir24d(zCrit, 'Genesis', 20.0);
    expect(compCritical.mode).toBe('Genesis');
    expect(compCritical.rubatoDepth).toBeGreaterThan(0.25);

    // Strict requirements: wide dramatic leaps >= 8 st, rubato ioiVariance > 0, sharp dynamic accents
    expect(compCritical.stats.maxIntervalLeap).toBeGreaterThanOrEqual(8);
    expect(compCritical.stats.ioiVariance).toBeGreaterThan(0.0);
    expect(compCritical.stats.velocityStd).toBeGreaterThanOrEqual(15.0);
  });

  // ============================================================================
  // 4. Standard MIDI File (.mid) Binary Encoding Verification
  // ============================================================================
  it('encodes valid Standard MIDI File (SMF Format 1) binary with MThd and MTrk chunks', () => {
    const dummyZ = new Array(24).fill(0.2);
    dummyZ[7] = 0.8;
    const comp = synthesizeMusicFromUsir24d(dummyZ, 'II_Critical', 16.0);

    const midiBytes = encodeCompositionToMidiBytes(comp);
    expect(midiBytes instanceof Uint8Array).toBe(true);
    expect(midiBytes.length).toBeGreaterThan(100);

    // 1. Verify Header Chunk: 'MThd'
    expect(midiBytes[0]).toBe(0x4d); // 'M'
    expect(midiBytes[1]).toBe(0x54); // 'T'
    expect(midiBytes[2]).toBe(0x68); // 'h'
    expect(midiBytes[3]).toBe(0x64); // 'd'

    // Header length must be 6
    const headerLen =
      ((midiBytes[4]! << 24) | (midiBytes[5]! << 16) | (midiBytes[6]! << 8) | midiBytes[7]!) >>> 0;
    expect(headerLen).toBe(6);

    // Format must be 1 (multitrack)
    const format = (midiBytes[8]! << 8) | midiBytes[9]!;
    expect(format).toBe(1);

    // Number of tracks: conductor track (1) + instrument tracks
    const nTracks = (midiBytes[10]! << 8) | midiBytes[11]!;
    expect(nTracks).toBe(comp.tracks.length + 1);

    // Division must be 480
    const division = (midiBytes[12]! << 8) | midiBytes[13]!;
    expect(division).toBe(480);

    // 2. Verify Conductor Track: 'MTrk' starts at offset 14
    expect(midiBytes[14]).toBe(0x4d); // 'M'
    expect(midiBytes[15]).toBe(0x54); // 'T'
    expect(midiBytes[16]).toBe(0x72); // 'r'
    expect(midiBytes[17]).toBe(0x6b); // 'k'

    // 3. Scan for required MIDI meta events and note events
    let hasTempoMeta = false;
    let hasNoteOn = false;
    let hasNoteOff = false;
    let hasEndOfTrack = false;

    for (let i = 0; i < midiBytes.length - 3; i++) {
      // Tempo Meta Event: FF 51 03
      if (midiBytes[i] === 0xff && midiBytes[i + 1] === 0x51 && midiBytes[i + 2] === 0x03) {
        hasTempoMeta = true;
      }
      // Note On: 0x90 .. 0x9F
      if ((midiBytes[i]! & 0xf0) === 0x90) {
        hasNoteOn = true;
      }
      // Note Off: 0x80 .. 0x8F
      if ((midiBytes[i]! & 0xf0) === 0x80) {
        hasNoteOff = true;
      }
      // End of Track: FF 2F 00
      if (midiBytes[i] === 0xff && midiBytes[i + 1] === 0x2f && midiBytes[i + 2] === 0x00) {
        hasEndOfTrack = true;
      }
    }

    expect(hasTempoMeta).toBe(true);
    expect(hasNoteOn).toBe(true);
    expect(hasNoteOff).toBe(true);
    expect(hasEndOfTrack).toBe(true);

    // 4. Test Data URI generation
    const dataUri = createMidiDataUri(comp);
    expect(dataUri.startsWith('data:audio/midi;base64,')).toBe(true);
    expect(dataUri.length).toBeGreaterThan(120);
  });

  // ============================================================================
  // 5. Web Audio Synth Player & Source Code Helper
  // ============================================================================
  it('generates valid Web Audio source code and schedules polyphonic notes on mock AudioContext', () => {
    // 1. Source code helper
    const synthCode = createWebAudioSynthSourceCode();
    expect(synthCode).toContain('export function playUsirComposition');
    expect(synthCode).toContain('createConvolver');
    expect(synthCode).toContain('createBiquadFilter');
    expect(synthCode).toContain('linearRampToValueAtTime');

    // 2. Playback on mock AudioContext
    const scheduledNodes: string[] = [];
    const mockCtx = {
      currentTime: 0.0,
      sampleRate: 44100,
      destination: {},
      createGain: () => ({
        gain: {
          value: 1.0,
          setValueAtTime: () => {},
          linearRampToValueAtTime: () => {},
        },
        connect: () => {},
        disconnect: () => {},
      }),
      createBiquadFilter: () => ({
        type: 'lowpass',
        frequency: { setValueAtTime: () => {} },
        connect: () => {},
        disconnect: () => {},
      }),
      createOscillator: () => ({
        type: 'sine',
        frequency: { setValueAtTime: () => {} },
        connect: () => {},
        disconnect: () => {},
        start: () => scheduledNodes.push('start'),
        stop: () => scheduledNodes.push('stop'),
      }),
      createConvolver: () => ({
        buffer: null,
        connect: () => {},
        disconnect: () => {},
      }),
      createBuffer: (_ch: number, len: number, _rate: number) => ({
        getChannelData: (_c: number) => new Float32Array(len),
      }),
    };

    const comp = synthesizeMusicFromUsir24d(new Array(24).fill(0.1), 'V_Deadzone', 4.0);
    const player = playCompositionOnWebAudio(mockCtx, comp);

    expect(typeof player.stop).toBe('function');
    expect(scheduledNodes.filter((n) => n === 'start').length).toBeGreaterThan(10);

    // Call stop and ensure it executes without throw
    expect(() => player.stop()).not.toThrow();
  });

  // ============================================================================
  // 6. Automatic Regime Inference from 24D Vectors
  // ============================================================================
  it('infers the appropriate acoustic regime when mode is omitted', () => {
    // Void-heavy vector
    const zVoid = new Array(24).fill(0.05);
    zVoid[2] = 0.9;
    zVoid[10] = 0.95;
    const compVoid = synthesizeMusicFromUsir24d(zVoid);
    expect(compVoid.mode).toBe('III_Void');

    // Flat uniform vector
    const zDead = new Array(24).fill(0.08);
    const compDead = synthesizeMusicFromUsir24d(zDead);
    expect(compDead.mode).toBe('V_Deadzone');

    // High leap / soliton vector
    const zGenesis = new Array(24).fill(0.1);
    zGenesis[7] = 0.9; // Sg-08 Lp
    const compGen = synthesizeMusicFromUsir24d(zGenesis);
    expect(compGen.mode).toBe('Genesis');
  });
});
