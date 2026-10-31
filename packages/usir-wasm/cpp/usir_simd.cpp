/**
 * Universal Style Genome (USIR / Nishiki)
 * WebAssembly SIMD 128-bit Acceleration Kernel (Phase 3 / TASK-363)
 * High-performance 24D Cholesky Distance, Hamiltonian Dynamics & PoE Steering
 */

#include <wasm_simd128.h>
#include <cstddef>
#include <cmath>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#define USIR_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define USIR_EXPORT
#endif

extern "C" {

/**
 * Computes squared Mahalanobis distance in 24 dimensions:
 * || L^T * (z - mu) ||_2^2
 * 
 * @param z         Pointer to 24 floats (current 24D symplectic state)
 * @param mu        Pointer to 24 floats (centroid state)
 * @param L_packed  Pointer to 300 floats (packed lower-triangular Cholesky factor)
 * @return          Squared distance (float)
 */
USIR_EXPORT float usir_cholesky_sq_dist_24d(
    const float* __restrict__ z,
    const float* __restrict__ mu,
    const float* __restrict__ L_packed
) {
    // Step 1: Compute diff = z - mu in 6 SIMD blocks (24 floats / 4 = 6)
    alignas(16) float diff[24];
    for (int g = 0; g < 6; ++g) {
        v128_t vz = wasm_v128_load(z + g * 4);
        v128_t vm = wasm_v128_load(mu + g * 4);
        v128_t vd = wasm_f32x4_sub(vz, vm);
        wasm_v128_store(diff + g * 4, vd);
    }

    // Step 2: Compute y = L^T * diff
    // L is 24x24 lower-triangular, packed row-by-row (i=0..23, j=0..i).
    // y_j = sum_{i=j}^{23} L_{i,j} * diff_i
    alignas(16) float y[24] = {0.0f};
    int L_idx = 0;

    for (int i = 0; i < 24; ++i) {
        const float di = diff[i];
        const v128_t di_splat = wasm_f32x4_splat(di);
        int j = 0;

        // Process 4 elements at a time with SIMD
        for (; j + 4 <= i + 1; j += 4) {
            v128_t y_v = wasm_v128_load(y + j);
            v128_t L_v = wasm_v128_load(L_packed + L_idx);
            y_v = wasm_f32x4_add(y_v, wasm_f32x4_mul(di_splat, L_v));
            wasm_v128_store(y + j, y_v);
            L_idx += 4;
        }

        // Remainder loop
        for (; j <= i; ++j) {
            y[j] += di * L_packed[L_idx++];
        }
    }

    // Step 3: Compute ||y||_2^2 using SIMD dot product
    v128_t sq_sum_v = wasm_f32x4_splat(0.0f);
    for (int g = 0; g < 6; ++g) {
        v128_t y_v = wasm_v128_load(y + g * 4);
        sq_sum_v = wasm_f32x4_add(sq_sum_v, wasm_f32x4_mul(y_v, y_v));
    }

    alignas(16) float sums[4];
    wasm_v128_store(sums, sq_sum_v);
    return sums[0] + sums[1] + sums[2] + sums[3];
}

/**
 * Computes Hamiltonian energy H(q, p) = T(p) + V(q) in 12 coordinates:
 * T(p) = 0.5 * sum_{i=0..11} p_i^2 * mass_inv_i
 * V(q) = 0.5 * sum_{i=0..11} (q_i - 0.5)^2 * stiffness_i
 * 
 * @param q          Pointer to 12 floats (position coordinates)
 * @param p          Pointer to 12 floats (conjugate momentum coordinates)
 * @param mass_inv   Pointer to 12 floats (inverse mass M^-1, or nullptr for default 1.0)
 * @param stiffness  Pointer to 12 floats (potential spring K, or nullptr for default 1.0)
 * @return           Total energy H(q, p)
 */
USIR_EXPORT float usir_hamiltonian_24d(
    const float* __restrict__ q,
    const float* __restrict__ p,
    const float* __restrict__ mass_inv,
    const float* __restrict__ stiffness
) {
    v128_t t_acc = wasm_f32x4_splat(0.0f);
    v128_t v_acc = wasm_f32x4_splat(0.0f);
    const v128_t q_star = wasm_f32x4_splat(0.5f);
    const v128_t one = wasm_f32x4_splat(1.0f);

    for (int g = 0; g < 3; ++g) { // 12 elements / 4 = 3 SIMD iterations
        // Kinetic term: p_i^2 * mass_inv_i
        v128_t vp = wasm_v128_load(p + g * 4);
        v128_t minv = mass_inv ? wasm_v128_load(mass_inv + g * 4) : one;
        v128_t p_sq = wasm_f32x4_mul(vp, vp);
        t_acc = wasm_f32x4_add(t_acc, wasm_f32x4_mul(p_sq, minv));

        // Potential term: (q_i - 0.5)^2 * stiffness_i
        v128_t vq = wasm_v128_load(q + g * 4);
        v128_t diff = wasm_f32x4_sub(vq, q_star);
        v128_t k = stiffness ? wasm_v128_load(stiffness + g * 4) : one;
        v128_t diff_sq = wasm_f32x4_mul(diff, diff);
        v_acc = wasm_f32x4_add(v_acc, wasm_f32x4_mul(diff_sq, k));
    }

    alignas(16) float t_arr[4];
    alignas(16) float v_arr[4];
    wasm_v128_store(t_arr, t_acc);
    wasm_v128_store(v_arr, v_acc);

    const float total_t = t_arr[0] + t_arr[1] + t_arr[2] + t_arr[3];
    const float total_v = v_arr[0] + v_arr[1] + v_arr[2] + v_arr[3];

    return 0.5f * (total_t + total_v);
}

/**
 * Computes Product-of-Experts precision-weighted gradient step for style steering / token logit biasing:
 * out_grad_i = step_size * lambda_diag_i * (target_mu_i - current_z_i)
 * 
 * @param current_z    Pointer to 24 floats (current state)
 * @param target_mu    Pointer to 24 floats (target state centroid)
 * @param lambda_diag  Pointer to 24 floats (diagonal precision weights)
 * @param step_size    Step size scalar
 * @param out_grad     Pointer to 24 floats output gradient
 */
USIR_EXPORT void usir_poe_step_24d(
    const float* __restrict__ current_z,
    const float* __restrict__ target_mu,
    const float* __restrict__ lambda_diag,
    float step_size,
    float* __restrict__ out_grad
) {
    const v128_t step_v = wasm_f32x4_splat(step_size);

    for (int g = 0; g < 6; ++g) { // 24 floats / 4 = 6 SIMD blocks
        v128_t z_v = wasm_v128_load(current_z + g * 4);
        v128_t mu_v = wasm_v128_load(target_mu + g * 4);
        v128_t lam_v = wasm_v128_load(lambda_diag + g * 4);

        v128_t diff = wasm_f32x4_sub(mu_v, z_v);
        v128_t weighted = wasm_f32x4_mul(diff, lam_v);
        v128_t step = wasm_f32x4_mul(weighted, step_v);

        wasm_v128_store(out_grad + g * 4, step);
    }
}

} // extern "C"
