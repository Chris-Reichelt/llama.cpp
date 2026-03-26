#pragma once

/*
 * TurboQuant: PolarQuant-based KV Cache Quantization
 * ===================================================
 * Implementation of the TurboQuant algorithm from:
 *   "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
 *   (arXiv:2504.19874, Daliri et al., 2025)
 *
 * Key idea:
 *   1. Apply a random rotation (fast Walsh-Hadamard transform + random sign flips)
 *      to the input vector block. This makes each coordinate follow a concentrated
 *      Beta distribution that converges to N(0, 1/d) in high dimensions.
 *   2. Because rotated coordinates are nearly independent, we can apply an optimal
 *      scalar quantizer (Lloyd-Max) to each coordinate independently.
 *   3. The precomputed centroids are optimal for the Gaussian distribution.
 *
 * Block size: 32 (matches q4_0 for compatibility)
 * The fast Walsh-Hadamard transform is O(d log d) and highly SIMD-friendly.
 */

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Precomputed optimal Lloyd-Max codebooks for N(0, 1) distribution
 * ========================================================================
 * These are the optimal centroids for quantizing a standard normal random
 * variable to b bits, found by solving the continuous 1D k-means problem.
 * The boundaries between centroids are midpoints of consecutive centroids.
 *
 * For the actual quantization of block data, centroids are scaled by 1/sqrt(d)
 * where d is the block size, since rotated coordinates ~ N(0, 1/d).
 *
 * Sources: Lloyd-Max quantizer tables for Gaussian distribution.
 */

/* 3-bit (8 levels): optimal centroids for N(0,1) */
static const float tq3_centroids[8] = {
    -1.7479f, -1.0500f, -0.5006f, -0.0000f,
     0.0000f,  0.5006f,  1.0500f,  1.7479f
};
/* Note: The distribution is symmetric, so c[i] = -c[7-i].
 * Boundaries are midpoints: b[i] = (c[i] + c[i+1]) / 2
 * b = {-1.3990, -0.7753, -0.2503, 0.0, 0.2503, 0.7753, 1.3990}
 */

/* 4-bit (16 levels): optimal centroids for N(0,1) */
static const float tq4_centroids[16] = {
    -2.4008f, -1.8438f, -1.4371f, -1.0993f,
    -0.7994f, -0.5224f, -0.2582f,  0.0000f,
     0.0000f,  0.2582f,  0.5224f,  0.7994f,
     1.0993f,  1.4371f,  1.8438f,  2.4008f
};

/* 2-bit (4 levels): optimal centroids for N(0,1) — for QJL residual reference */
static const float tq2_centroids[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

/* ========================================================================
 * Fast Walsh-Hadamard Transform (in-place, unnormalized)
 * ========================================================================
 * The WHT of size d=2^k is the unique orthogonal transform that can be
 * applied in O(d log d) without any multiplications (only add/subtract).
 * Combined with random sign flips, it acts as a random rotation.
 */

static inline void tq_fast_hadamard_transform(float * data, int n) {
    /* In-place Fast Walsh-Hadamard Transform.
     * After this, multiply all elements by 1/sqrt(n) for orthonormal transform.
     * n must be a power of 2. */
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

/* ========================================================================
 * Random sign generation from seed
 * ========================================================================
 * We use a simple xorshift32 PRNG seeded per-block to generate random
 * sign flips (+1/-1) for each coordinate. This is deterministic given
 * the seed, so we can reproduce the same rotation during dequantization.
 */

static inline uint32_t tq_xorshift32(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Apply random sign flips to data array using seed.
 * Each element is multiplied by +1 or -1 based on PRNG bit. */
static inline void tq_apply_random_signs(float * data, int n, uint32_t seed) {
    uint32_t state = seed;
    for (int i = 0; i < n; i++) {
        uint32_t r = tq_xorshift32(&state);
        if (r & 1) {
            data[i] = -data[i];
        }
    }
}

/* ========================================================================
 * PolarQuant rotation: Hadamard * random_signs
 * ========================================================================
 * Full rotation: apply random signs, then Hadamard, then normalize by 1/sqrt(n).
 * This maps any unit vector to approximately uniform on the sphere,
 * making each coordinate ~ N(0, 1/n) for large n.
 */

static inline void tq_rotate_forward(float * data, int n, uint32_t seed) {
    tq_apply_random_signs(data, n, seed);
    tq_fast_hadamard_transform(data, n);
    /* Normalize: WHT is orthogonal when scaled by 1/sqrt(n) */
    float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        data[i] *= inv_sqrt_n;
    }
}

/* Inverse rotation: inverse Hadamard (= Hadamard/n, but since we already
 * divided by sqrt(n) in forward, just do Hadamard then divide by sqrt(n)
 * and undo sign flips). */
static inline void tq_rotate_inverse(float * data, int n, uint32_t seed) {
    float inv_sqrt_n = 1.0f / sqrtf((float)n);
    tq_fast_hadamard_transform(data, n);
    for (int i = 0; i < n; i++) {
        data[i] *= inv_sqrt_n;
    }
    tq_apply_random_signs(data, n, seed);
}

/* ========================================================================
 * Scalar quantization helpers
 * ========================================================================
 * Given a value and a sorted codebook, find the nearest centroid index.
 * For small codebooks (8 or 16 entries), linear search is fast enough
 * and branch-prediction friendly.
 */

static inline int tq_quantize_scalar_3bit(float val, float scale) {
    /* Quantize val (already rotated) to nearest 3-bit centroid.
     * val is in the rotated domain ~ N(0, 1/d), so we scale to N(0,1) space
     * by multiplying by sqrt(d). The codebook is for N(0,1). */
    float x = val * scale;  /* scale = sqrt(block_size) */

    /* Binary search over boundaries (midpoints of consecutive centroids):
     * b = {-1.3990, -0.7753, -0.2503, 0.0, 0.2503, 0.7753, 1.3990} */
    if (x < 0.0f) {
        if (x < -0.7753f) {
            return (x < -1.3990f) ? 0 : 1;
        } else {
            return (x < -0.2503f) ? 2 : 3;
        }
    } else {
        if (x < 0.7753f) {
            return (x < 0.2503f) ? 4 : 5;
        } else {
            return (x < 1.3990f) ? 6 : 7;
        }
    }
}

static inline float tq_dequantize_scalar_3bit(int idx, float inv_scale) {
    /* Convert 3-bit index back to centroid value in rotated domain */
    return tq3_centroids[idx] * inv_scale;  /* inv_scale = 1/sqrt(block_size) */
}

static inline int tq_quantize_scalar_4bit(float val, float scale) {
    float x = val * scale;

    /* 4-bit: 16 levels. Boundaries (midpoints of consecutive centroids):
     * We use a balanced binary search tree for efficiency. */
    if (x < 0.0f) {
        if (x < -0.9494f) {
            if (x < -1.6405f) {
                return (x < -2.1223f) ? 0 : 1;
            } else {
                return (x < -1.2682f) ? 2 : 3;
            }
        } else {
            if (x < -0.3903f) {
                return (x < -0.6609f) ? 4 : 5;
            } else {
                return (x < -0.1291f) ? 6 : 7;
            }
        }
    } else {
        if (x < 0.9494f) {
            if (x < 0.3903f) {
                return (x < 0.1291f) ? 8 : 9;
            } else {
                return (x < 0.6609f) ? 10 : 11;
            }
        } else {
            if (x < 1.6405f) {
                return (x < 1.2682f) ? 12 : 13;
            } else {
                return (x < 2.1223f) ? 14 : 15;
            }
        }
    }
}

static inline float tq_dequantize_scalar_4bit(int idx, float inv_scale) {
    return tq4_centroids[idx] * inv_scale;
}

#ifdef __cplusplus
}
#endif
