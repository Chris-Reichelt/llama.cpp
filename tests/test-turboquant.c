/*
 * test-turboquant.c
 * ==================
 * Unit test for TurboQuant (TQ3_0, TQ4_0) quantization.
 * Tests:
 *   1. Round-trip accuracy: quantize then dequantize, check MSE
 *   2. Inner product preservation (key property for KV cache)
 *   3. Block size alignment checks
 *   4. Zero block handling
 */

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/* Simple LCG for reproducible test vectors */
static uint32_t lcg_state = 12345;
static float lcg_randf(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    /* Box-Muller for Gaussian */
    static int have_extra = 0;
    static float extra;
    if (have_extra) { have_extra = 0; return extra; }
    float u1 = (lcg_state & 0xFFFFFF) / (float)0x1000000;
    lcg_state = lcg_state * 1664525u + 1013904223u;
    float u2 = (lcg_state & 0xFFFFFF) / (float)0x1000000;
    if (u1 < 1e-6f) u1 = 1e-6f;
    float r = sqrtf(-2.0f * logf(u1));
    extra = r * sinf(2.0f * 3.14159265f * u2);
    have_extra = 1;
    return r * cosf(2.0f * 3.14159265f * u2);
}

#define N_BLOCKS 64
#define BLOCK_SIZE 32

static void test_tq3_roundtrip(void) {
    printf("=== TQ3_0 Round-trip Test ===\n");

    float orig[N_BLOCKS * BLOCK_SIZE];
    float recon[N_BLOCKS * BLOCK_SIZE];
    block_tq3_0 quant[N_BLOCKS];

    /* Fill with Gaussian random values (like KV cache activations) */
    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; i++) {
        orig[i] = lcg_randf() * 2.0f;  /* stddev=2 like typical KV values */
    }

    /* Quantize */
    quantize_row_tq3_0_ref(orig, quant, N_BLOCKS * BLOCK_SIZE);

    /* Dequantize */
    dequantize_row_tq3_0(quant, recon, N_BLOCKS * BLOCK_SIZE);

    /* Compute MSE and relative MSE */
    double mse = 0.0, orig_energy = 0.0;
    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; i++) {
        double diff = orig[i] - recon[i];
        mse += diff * diff;
        orig_energy += orig[i] * orig[i];
    }
    mse /= (N_BLOCKS * BLOCK_SIZE);
    double rel_mse = mse / (orig_energy / (N_BLOCKS * BLOCK_SIZE));

    printf("  MSE:     %.6f\n", mse);
    printf("  RelMSE:  %.4f%% (lower is better)\n", rel_mse * 100.0);
    printf("  Theory:  ~3.0%% expected for 3-bit TurboQuant (from paper: D_mse ≈ 0.03)\n");

    /* Check relative MSE is reasonable (should be < 10%) */
    if (rel_mse < 0.10) {
        printf("  PASS: RelMSE within expected range\n");
    } else {
        printf("  FAIL: RelMSE too high! %.2f%%\n", rel_mse * 100.0);
    }

    /* Check inner product preservation */
    printf("\n--- Inner Product Preservation ---\n");
    double ip_err_sq = 0.0, ip_sq = 0.0;
    int n_pairs = 100;
    lcg_state = 99999;
    for (int p = 0; p < n_pairs; p++) {
        float q[BLOCK_SIZE], k[BLOCK_SIZE], k_recon[BLOCK_SIZE];
        block_tq3_0 kq[1];

        for (int j = 0; j < BLOCK_SIZE; j++) {
            q[j] = lcg_randf();
            k[j] = lcg_randf();
        }

        quantize_row_tq3_0_ref(k, kq, BLOCK_SIZE);
        dequantize_row_tq3_0(kq, k_recon, BLOCK_SIZE);

        double ip_true = 0.0, ip_approx = 0.0;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            ip_true  += q[j] * k[j];
            ip_approx += q[j] * k_recon[j];
        }
        double err = ip_true - ip_approx;
        ip_err_sq += err * err;
        ip_sq += ip_true * ip_true;
    }
    double ip_rel_err = sqrt(ip_err_sq / n_pairs) / sqrt(ip_sq / n_pairs);
    printf("  RMS inner product relative error: %.4f%%\n", ip_rel_err * 100.0);
    if (ip_rel_err < 0.15) {
        printf("  PASS: Inner product error within expected range\n");
    } else {
        printf("  FAIL: Inner product error too high\n");
    }
}

static void test_tq4_roundtrip(void) {
    printf("\n=== TQ4_0 Round-trip Test ===\n");

    float orig[N_BLOCKS * BLOCK_SIZE];
    float recon[N_BLOCKS * BLOCK_SIZE];
    block_tq4_0 quant[N_BLOCKS];

    lcg_state = 54321;
    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; i++) {
        orig[i] = lcg_randf() * 2.0f;
    }

    quantize_row_tq4_0_ref(orig, quant, N_BLOCKS * BLOCK_SIZE);
    dequantize_row_tq4_0(quant, recon, N_BLOCKS * BLOCK_SIZE);

    double mse = 0.0, orig_energy = 0.0;
    for (int i = 0; i < N_BLOCKS * BLOCK_SIZE; i++) {
        double diff = orig[i] - recon[i];
        mse += diff * diff;
        orig_energy += orig[i] * orig[i];
    }
    mse /= (N_BLOCKS * BLOCK_SIZE);
    double rel_mse = mse / (orig_energy / (N_BLOCKS * BLOCK_SIZE));

    printf("  MSE:     %.6f\n", mse);
    printf("  RelMSE:  %.4f%% (lower is better)\n", rel_mse * 100.0);
    printf("  Theory:  ~0.9%% expected for 4-bit TurboQuant (from paper: D_mse ≈ 0.009)\n");

    if (rel_mse < 0.05) {
        printf("  PASS: RelMSE within expected range\n");
    } else {
        printf("  FAIL: RelMSE too high! %.2f%%\n", rel_mse * 100.0);
    }
}

static void test_zero_block(void) {
    printf("\n=== Zero Block Test ===\n");

    float zero[BLOCK_SIZE] = {0};
    float recon[BLOCK_SIZE];
    block_tq3_0 qb3[1];
    block_tq4_0 qb4[1];

    quantize_row_tq3_0_ref(zero, qb3, BLOCK_SIZE);
    dequantize_row_tq3_0(qb3, recon, BLOCK_SIZE);

    int all_zero = 1;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (fabsf(recon[i]) > 1e-6f) { all_zero = 0; break; }
    }
    printf("  TQ3_0 zero block: %s\n", all_zero ? "PASS" : "FAIL");

    quantize_row_tq4_0_ref(zero, qb4, BLOCK_SIZE);
    dequantize_row_tq4_0(qb4, recon, BLOCK_SIZE);

    all_zero = 1;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (fabsf(recon[i]) > 1e-6f) { all_zero = 0; break; }
    }
    printf("  TQ4_0 zero block: %s\n", all_zero ? "PASS" : "FAIL");
}

static void test_block_sizes(void) {
    printf("\n=== Block Size / Struct Size Test ===\n");
    printf("  sizeof(block_tq3_0) = %zu (expected 18)\n", sizeof(block_tq3_0));
    printf("  sizeof(block_tq4_0) = %zu (expected 22)\n", sizeof(block_tq4_0));
    printf("  QKTQ3_0 = %d, QKTQ4_0 = %d\n", QKTQ3_0, QKTQ4_0);
    printf("  TQ3_0 bpw = %.2f (%.0f bits / 32 elements)\n",
           (float)(sizeof(block_tq3_0)*8) / QKTQ3_0,
           (float)(sizeof(block_tq3_0)*8));
    printf("  TQ4_0 bpw = %.2f (%.0f bits / 32 elements)\n",
           (float)(sizeof(block_tq4_0)*8) / QKTQ4_0,
           (float)(sizeof(block_tq4_0)*8));

    if (sizeof(block_tq3_0) == 18 && sizeof(block_tq4_0) == 22) {
        printf("  PASS: struct sizes correct\n");
    } else {
        printf("  FAIL: unexpected struct sizes\n");
    }
}

static void compare_with_q4_0(void) {
    printf("\n=== Comparison: TQ3_0 vs TQ4_0 vs naive ===\n");
    printf("  Type    | bpw  | RelMSE\n");
    printf("  --------|------|-------\n");

    lcg_state = 777;
    int n = N_BLOCKS * BLOCK_SIZE;
    float orig[N_BLOCKS * BLOCK_SIZE];
    float recon[N_BLOCKS * BLOCK_SIZE];
    for (int i = 0; i < n; i++) orig[i] = lcg_randf() * 2.0f;
    double orig_energy = 0.0;
    for (int i = 0; i < n; i++) orig_energy += orig[i] * orig[i];

    /* TQ3_0 */
    block_tq3_0 qb3[N_BLOCKS];
    quantize_row_tq3_0_ref(orig, qb3, n);
    dequantize_row_tq3_0(qb3, recon, n);
    double mse3 = 0.0;
    for (int i = 0; i < n; i++) { double d = orig[i]-recon[i]; mse3 += d*d; }
    printf("  tq3_0   | %.2f | %.3f%%\n",
           (float)(sizeof(block_tq3_0)*8)/QKTQ3_0, mse3/orig_energy*100.0);

    /* TQ4_0 */
    block_tq4_0 qb4[N_BLOCKS];
    quantize_row_tq4_0_ref(orig, qb4, n);
    dequantize_row_tq4_0(qb4, recon, n);
    double mse4 = 0.0;
    for (int i = 0; i < n; i++) { double d = orig[i]-recon[i]; mse4 += d*d; }
    printf("  tq4_0   | %.2f | %.3f%%\n",
           (float)(sizeof(block_tq4_0)*8)/QKTQ4_0, mse4/orig_energy*100.0);

    printf("  --------|------|-------\n");
    printf("  Theory (from paper):\n");
    printf("    3-bit: D_mse ≈ 0.03 * ||x||^2  (~3%% relative MSE for unit vecs)\n");
    printf("    4-bit: D_mse ≈ 0.009 * ||x||^2 (~0.9%% relative MSE for unit vecs)\n");
}

int main(void) {
    printf("TurboQuant Implementation Test\n");
    printf("Based on arXiv:2504.19874 (Daliri et al., 2025)\n");
    printf("Algorithm: PolarQuant = random Hadamard rotation + Lloyd-Max scalar quantization\n\n");

    test_block_sizes();
    test_zero_block();
    test_tq3_roundtrip();
    test_tq4_roundtrip();
    compare_with_q4_0();

    printf("\nAll tests complete.\n");
    return 0;
}
