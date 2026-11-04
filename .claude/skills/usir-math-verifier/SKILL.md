---
name: usir-math-verifier
description: Verify USIR 24-dimensional symplectic instrumentation, order parameters (psi, phi_void, kappa_kin), and LOOCV benchmark reproducibility.
---

# USIR Math Verifier Skill

Use this skill to audit and verify the mathematical and thermodynamic invariants of the Universal Style Genome (USIR):

## Verification Checks
1. **24D Symplectic State $\mathbf{z} = (\mathbf{q}, \mathbf{p})$**:
   - $\mathbf{q} \in [0, 1]^{12}$ represents the 12 elemental style intensities across 4 periodic groups (Flux, Lattice, Void, Shock) and 3 scales (Kasane, Sasae, Kamae).
   - $\mathbf{p} \in \mathbb{R}_{\ge 0}^{12}$ represents temporal sentence-to-sentence derivatives (RMS velocity and sentence length volatility $CV_L$). $\mathbf{p}$ must never simply copy $\mathbf{q}$.
2. **Order Parameters**:
   - $\psi_{\text{order}} \in [-1, 1]$: Macroscopic lattice symmetry vs turbulence.
   - $\phi_{\text{void}} \in [0, 1]$: Superfluid void / apophatic stillness condensation.
   - $\kappa_{\text{kin}} \ge 0.04$: Kinetic momentum and shock activation.
3. **LOOCV Benchmarks**:
   - All 5-phase ($N=40$): $\ge 95.0\%$ (golden: $97.5\%$).
   - Core 3-phase ($N=28$): $\ge 95.0\%$ (golden: $96.4\%$).
   - Ablation inequalities: Full 24D $>$ No Group III (Void), Full 24D $>$ No Group IV (Shock), Full 24D $>$ No Momentum $\mathbf{p}$.
4. **Numerical Equivalence Gate**:
   - $\|\mathbf{z}_{\text{TS}} - \mathbf{z}_{\text{Py}}\|_\infty < 10^{-5}$ across all 40 golden corpus passages.
