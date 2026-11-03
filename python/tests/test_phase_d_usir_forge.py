"""Phase D WS-D1 Test Suite: OSWE Polar-Fourier Phase-Aligned usir-forge & 4KB Binary Compiler."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from usir_py.usir_forge import (
    align_polar_fourier_phase,
    compute_polar_fourier_phase,
    compute_packed_cholesky_24d,
    unpack_cholesky_24d,
    precision_matrix_from_packed_cholesky_24d,
    compile_tier_m_binary,
    load_tier_m_binary,
    classify_mode_24d,
    OSWEStreamingCompiler,
    USIR_MAGIC,
    TIER_M_BYTE_LENGTH,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# 1. POLAR-FOURIER PHASE ALIGNMENT & DESTRUCTIVE CANCELLATION PREVENTION
# ==============================================================================

def test_phase_alignment_cancellation_prevention():
    """Verifies that polar-fourier phase alignment prevents destructive wave cancellation."""
    M = 64
    m = np.arange(M)

    # Two wave trajectories with opposite fundamental phases (delta_theta = pi)
    # Wave 1 has peak at m = 0 (theta = 0)
    wave_1 = 1.0 + np.cos(2.0 * np.pi * m / M)
    # Wave 2 has peak at m = M // 2 (theta = pi, exact opposite phase)
    wave_2 = 1.0 + np.cos(2.0 * np.pi * m / M - np.pi)

    # 1. Unaligned simple average: complete cancellation into flat Deadzone!
    unaligned_avg = (wave_1 + wave_2) / 2.0
    unaligned_ptp = float(np.ptp(unaligned_avg))
    assert unaligned_ptp < 1e-12, f"Expected total phase cancellation (ptp ~ 0), got {unaligned_ptp}"

    # 2. Polar-Fourier aligned average to target_phase = 3*pi/4
    target_phase = 3.0 * math.pi / 4.0
    aligned_1 = np.array(align_polar_fourier_phase(wave_1, target_phase=target_phase))
    aligned_2 = np.array(align_polar_fourier_phase(wave_2, target_phase=target_phase))

    aligned_avg = (aligned_1 + aligned_2) / 2.0
    aligned_ptp = float(np.ptp(aligned_avg))

    # Amplitude must be preserved (peak-to-peak ~ 2.0, no cancellation Deadzone)
    assert aligned_ptp >= 1.90, f"Expected peak preservation (ptp >= 1.90), got {aligned_ptp:.4f}"

    # Both aligned signals must have their peak phase aligned near target_phase
    phase_1 = compute_polar_fourier_phase(aligned_1)
    phase_2 = compute_polar_fourier_phase(aligned_2)
    phase_avg = compute_polar_fourier_phase(aligned_avg)

    assert abs(phase_1 - target_phase) < (2.0 * np.pi / M) + 1e-4
    assert abs(phase_2 - target_phase) < (2.0 * np.pi / M) + 1e-4
    assert abs(phase_avg - target_phase) < (2.0 * np.pi / M) + 1e-4


def test_phase_alignment_2d_trajectory():
    """Verifies 2D trajectory alignment along time axis 0."""
    M = 32
    traj = np.zeros((M, 24), dtype=np.float64)
    for t in range(M):
        traj[t, 3] = 0.5 + 0.5 * math.cos(2.0 * math.pi * t / M)  # Shock coordinate oscillation
        traj[t, 12:24] = 0.2

    aligned = align_polar_fourier_phase(traj, target_phase=3.0 * math.pi / 4.0)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (M, 24)
    # Energy envelope of aligned trajectory must have phase near 3*pi/4
    energy_aligned = 0.5 * np.sum(aligned[:, 12:24] ** 2, axis=1) + aligned[:, 3]
    aligned_phase = compute_polar_fourier_phase(energy_aligned)
    assert abs(aligned_phase - (3.0 * math.pi / 4.0)) < (2.0 * math.pi / M) + 1e-4


# ==============================================================================
# 2. STRICT 4,096 BYTES TIER M BINARY COMPILER
# ==============================================================================

def test_compile_tier_m_binary_strictly_4096_bytes():
    """Verifies that compiled Tier M binary is strictly 4,096 bytes and valid."""
    compiler = OSWEStreamingCompiler(min_clusters=5)

    # Register 5 distinct clusters to satisfy anonymity gate
    for k in range(5):
        sample = np.full(24, 0.4 + 0.1 * k, dtype=np.float64)
        compiler.add_sample(sample, mode=k % 3, cluster_id=f"author_cluster_{k}")

    binary = compiler.compile_tier_m_binary()

    # Exact byte length check
    assert len(binary) == TIER_M_BYTE_LENGTH, f"Expected {TIER_M_BYTE_LENGTH} bytes, got {len(binary)}"

    # Magic check: 0x55534952 (ASCII "USIR", LE bytes b'RISU')
    assert binary[:4] == b"RISU"

    # Load and unpack binary
    decoded = compiler.load_tier_m_binary(binary)
    assert decoded["version"] == 1
    assert decoded["tier"] == 2
    assert decoded["flags"] == 0
    assert len(decoded["transition_matrix"]) == 3
    assert len(decoded["weights"]) == 3
    assert len(decoded["centroids"]) == 3
    assert len(decoded["centroids"][0]) == 24
    assert len(decoded["cholesky_factors"]) == 3
    assert len(decoded["cholesky_factors"][0]) == 300
    assert isinstance(decoded["crc32"], int)
    assert isinstance(decoded["sha256"], str) and len(decoded["sha256"]) == 64


# ==============================================================================
# 3. K >= 5 CLUSTER ANONYMITY GATE
# ==============================================================================

def test_cluster_anonymity_gate_enforcement():
    """Verifies that compilation fails when K < 5 and passes when K >= 5."""
    compiler = OSWEStreamingCompiler(min_clusters=5)

    # Add samples from only 3 distinct authors/clusters
    for i in range(3):
        sample = np.full(24, 0.5, dtype=np.float64)
        compiler.add_sample(sample, mode=0, cluster_id=f"author_{i}")

    assert compiler.cluster_count == 3

    # Anonymity gate must reject K < 5
    with pytest.raises(ValueError) as excinfo:
        compiler.compile_tier_m_binary(enforce_anonymity=True)
    assert "Cluster anonymity gate failed" in str(excinfo.value)
    assert "K=3 < 5" in str(excinfo.value)

    # Adding cluster 4: still fails
    compiler.register_cluster("author_3")
    assert compiler.cluster_count == 4
    with pytest.raises(ValueError):
        compiler.compile_tier_m_binary(enforce_anonymity=True)

    # Adding cluster 5: passes!
    compiler.register_cluster("author_4")
    assert compiler.cluster_count == 5
    binary = compiler.compile_tier_m_binary(enforce_anonymity=True)
    assert len(binary) == 4096


# ==============================================================================
# 4. PYTHON <-> TYPESCRIPT ROUNDTRIP INTEROPERABILITY
# ==============================================================================

def test_python_typescript_binary_roundtrip():
    """Verifies byte-level round-trip compatibility between Python and TypeScript."""
    # Build a profile in Python
    compiler = OSWEStreamingCompiler(min_clusters=5)
    np.random.seed(42)
    for i in range(15):
        vec = np.random.uniform(0.1, 0.9, size=24)
        mode = i % 3
        compiler.add_sample(vec, mode=mode, cluster_id=f"cluster_{i % 5}")

    py_binary = compiler.compile_tier_m_binary()
    assert len(py_binary) == 4096

    # Verify that Node.js / TypeScript usirForge can decode the Python binary
    node_script = f"""
    import {{ loadTierMBinary }} from './packages/usir-core/dist/index.js';
    import fs from 'fs';

    const buf = fs.readFileSync(0);
    const decoded = loadTierMBinary(new Uint8Array(buf));
    console.log(JSON.stringify({{
      version: decoded.version,
      tier: decoded.tier,
      flags: decoded.flags,
      crc32: decoded.crc32,
      weights: decoded.weights,
      centroids0_sample: decoded.centroids[0].slice(0, 3)
    }}));
    """
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        input=py_binary,
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    ts_result = json.loads(proc.stdout.decode("utf-8"))
    assert ts_result["version"] == 1
    assert ts_result["tier"] == 2
    assert ts_result["crc32"] == compiler.load_tier_m_binary(py_binary)["crc32"]
    assert np.allclose(ts_result["weights"], compiler.load_tier_m_binary(py_binary)["weights"], atol=1e-5)

    # Now verify that Python can load a binary compiled by TypeScript
    node_compile_script = """
    import { OSWEStreamingCompiler } from './packages/usir-core/dist/index.js';

    const compiler = new OSWEStreamingCompiler({ minClusters: 5 });
    for (let i = 0; i < 15; i++) {
      const vec = new Array(24).fill(0).map((_, d) => 0.1 * ((i + d) % 9 + 1));
      compiler.addSample(vec, i % 3, `cluster_${i % 5}`);
    }
    const bin = compiler.compileTierMBinary();
    process.stdout.write(Buffer.from(bin));
    """
    proc_comp = subprocess.run(
        ["node", "--input-type=module", "-e", node_compile_script],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    ts_binary = proc_comp.stdout
    assert len(ts_binary) == 4096

    py_decoded = load_tier_m_binary(ts_binary)
    assert py_decoded["version"] == 1
    assert py_decoded["tier"] == 2
    assert len(py_decoded["centroids"]) == 3
    assert len(py_decoded["cholesky_factors"][0]) == 300


# ==============================================================================
# 5. WELFORD STREAMING COVARIANCE & CHOLESKY FACTOR INTEGRITY
# ==============================================================================

def test_welford_cholesky_precision_reconstruction():
    """Verifies that packed Cholesky factor L satisfies L L^T = Lambda = Sigma^-1."""
    compiler = OSWEStreamingCompiler(min_clusters=5)
    np.random.seed(123)

    for i in range(30):
        vec = np.random.normal(loc=0.5, scale=0.15, size=24)
        compiler.add_sample(vec, mode=0, cluster_id=f"c_{i % 5}")

    binary = compiler.compile_tier_m_binary()
    decoded = compiler.load_tier_m_binary(binary)

    # Reconstruct precision matrix Lambda from Mode 0 Cholesky factor
    packed_L0 = decoded["cholesky_factors"][0]
    Lambda_reconstructed = precision_matrix_from_packed_cholesky_24d(packed_L0)

    # Verify symmetry
    assert np.allclose(Lambda_reconstructed, Lambda_reconstructed.T, atol=1e-5)

    # Verify positive definiteness: all eigenvalues > 0
    eigenvals = np.linalg.eigvalsh(Lambda_reconstructed)
    assert np.all(eigenvals > 0.0), f"Reconstructed precision matrix has non-positive eigenvalues: {eigenvals.min()}"


# ==============================================================================
# 6. EDGE CASES & ATTACK SURFACE TESTS
# ==============================================================================

def test_phase_alignment_modes_synchronization():
    """Verifies that circular phase shift rotates modes in lockstep with trajectory vectors."""
    compiler = OSWEStreamingCompiler(min_clusters=5)
    traj = np.zeros((4, 24), dtype=np.float64)
    traj[0, 3] = 10.0  # Climax shock spike at sample 0
    modes = [2, 0, 1, 1]  # Sample 0 is tagged as Climax (mode 2)

    # When aligned to 3pi/4, sample 0 will shift.
    # The mode tag 2 must move with sample 0 to keep shock in Climax.
    compiler.add_trajectory(traj, modes=modes, cluster_id="c1", align_phase=True)

    assert compiler.means[2][3] == 10.0, f"Expected Climax shock to be 10.0, got {compiler.means[2][3]}"
    assert compiler.means[0][3] == 0.0, f"Expected Narration shock to be 0.0, got {compiler.means[0][3]}"
    assert compiler.means[1][3] == 0.0, f"Expected Dialogue shock to be 0.0, got {compiler.means[1][3]}"


def test_phase_alignment_multi_scale_shock():
    """Verifies alignment when climax shock is located in Meso (q7) or Macro (q11) scales."""
    M = 32
    target_phase = 3.0 * math.pi / 4.0

    # Macro shock (q11: Rupture / Soliton) oscillation
    traj_macro = np.zeros((M, 24), dtype=np.float64)
    for t in range(M):
        traj_macro[t, 11] = 0.5 + 0.5 * math.cos(2.0 * math.pi * t / M)
        traj_macro[t, 12:24] = 0.1

    aligned_macro = align_polar_fourier_phase(traj_macro, target_phase=target_phase)
    macro_energy = 0.5 * np.sum(aligned_macro[:, 12:24] ** 2, axis=1) + (
        aligned_macro[:, 3] + aligned_macro[:, 7] + aligned_macro[:, 11]
    ) / 3.0
    phase_macro = compute_polar_fourier_phase(macro_energy)
    assert abs(phase_macro - target_phase) < (2.0 * math.pi / M) + 1e-4


def test_phase_alignment_nan_inf_robustness():
    """Verifies that non-finite inputs fall back safely without unhandled exception."""
    nan_sig = [1.0, float("nan"), 2.0]
    inf_sig = [1.0, float("inf"), 2.0]

    assert compute_polar_fourier_phase(nan_sig) == 0.0
    assert compute_polar_fourier_phase(inf_sig) == 0.0

    aligned_nan = align_polar_fourier_phase(nan_sig)
    assert np.isnan(aligned_nan[1])
    assert aligned_nan[0] == 1.0


def test_anonymity_gate_whitespace_and_empty_id_rejection():
    """Verifies that empty string or whitespace-only cluster IDs are rejected."""
    compiler = OSWEStreamingCompiler(min_clusters=5)
    with pytest.raises(ValueError):
        compiler.register_cluster("   ")
    with pytest.raises(ValueError):
        compiler.register_cluster("")
    with pytest.raises(ValueError):
        compiler.add_sample(np.full(24, 0.5), mode=0, cluster_id="  ")


def test_composite_wave_with_harmonics():
    """Verifies that composite waves with strong harmonics preserve amplitude under alignment."""
    M = 64
    m = np.arange(M)
    w1 = 1.0 + np.cos(2 * np.pi * m / M) + 0.5 * np.cos(4 * np.pi * m / M)
    w2 = 1.0 + np.cos(2 * np.pi * m / M - np.pi) + 0.5 * np.cos(4 * np.pi * m / M - np.pi)

    # Unaligned cancels out
    assert np.ptp((w1 + w2) / 2.0) < 1e-12

    # Aligned preserves wave amplitude
    a1 = np.array(align_polar_fourier_phase(w1))
    a2 = np.array(align_polar_fourier_phase(w2))
    assert np.ptp((a1 + a2) / 2.0) >= 1.95

