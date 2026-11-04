# USIR / Nishiki Architecture & Implementation Rules (.antigravity/rules.md)

## 0. Handoff (引継ぎ台帳)
作業開始時に `docs/handoff/` の最新MDを読み、決定事項と未完了を引き継ぐ。節目に同フォルダへ追記する（形式は `docs/handoff/README.md`、自動importはしない）。

## 1. Core Architectural Invariants (絶対遵守事項)
1. **Standalone Zero-Coupling**: `packages/usir-core` and `packages/usir-mcp` MUST have ZERO runtime dependencies on `worldcraft`, `plotailor`, or DOM/browser APIs. They must run in pure Node.js, Web Workers, and edge runtimes.
2. **Strict 4-Group Orthogonality (4族の完全直交性)**:
   - **Group I (`Flux`: `Sg-01 Gr`, `Sg-05 Cp`, `Sg-09 Dy`)**: Pure information density, concrete physical stem mass, clause layering, and macro length dynamic range.
   - **Group II (`Lattice`: `Sg-02 Cr`, `Sg-06 Sy`, `Sg-10 At`)**: Pure syntactic regularity, formal deductive markers, and opening-to-closing stem recurrence.
   - **Group III (`Void` [間]: `Sg-03 Pr`, `Sg-07 Cs`, `Sg-11 Ma`)**: Pure Wago 1-char simplex stem porosity, sentence brevity, clean ending cut (absence of sluggish auxiliaries), punctuation quietude, and apophatic stillness operators (`ない / だけ / まま / 静`).
   - **Group IV (`Shock`: `Sg-04 Jt`, `Sg-08 Lp`, `Sg-12 Ru`)**: Pure voiced/geminate consonantal edge, adjacent-sentence semantic Jaccard leap, figurative tropes (`まるで / のように`), and adversative/rhetorical rupture.
   - **Conjugate Momentum (`p` in R^12)**: Strictly temporal derivatives (sentence-to-sentence RMS velocity $\sqrt{\frac{1}{M-1}\sum (\Delta q)^2}$ and sentence-length volatility $CV_L = \sigma_L / \mu_L$). Never copy static $q$ directly into $p$.
3. **Zero Target-Leakage (カンニング語彙の絶対禁止)**:
   - NEVER add author names, character names (e.g., `メロス`, `李徴`, `島村`, `下人`), or work-specific nouns into any feature extractor dictionary. Feature extractors may only use universal morphological, phonological, syntactic, and general sensory character classes.
4. **3-Room Clean Room Gate**:
   - Tier S (`128 bytes`) and Tier M (`4096 bytes`) binary outputs must pass `threeRoomGate.ts`, guaranteeing zero raw text retention and zero contiguous $\ge 7$-character n-gram overlap with source text.
5. **Docked Atelier UI Constitution (When touching `usir-bridges` for Plotailor)**:
   - All USIR visualizers (5-Phase Diagram, 12-Element Waveform, 4-Group Equalizer Sliders) MUST reside inside the **Right DetailDrawer (`25%` width)** or **Bottom Dock**. Creating auto-popping modals during typing is strictly forbidden.
