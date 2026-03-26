#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

/*
 * TQ3_0 quantize: PolarQuant (WHT + Lloyd-Max 3-bit) on GPU.
 * Runs on a single thread per 32-element block.
 */
static __device__ void quantize_f32_tq3_0_block(const float * __restrict__ x, block_tq3_0 * __restrict__ y) {
    /* TQ3 centroids for N(0,1) */
    const float centroids[8] = {-1.7479f, -1.0500f, -0.5006f, 0.0f, 0.0f, 0.5006f, 1.0500f, 1.7479f};

    float tmp[32];
    float sum_sq = 0.0f;

    for (int j = 0; j < 32; j++) {
        tmp[j] = x[j];
        sum_sq += x[j] * x[j];
    }

    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-10f) {
        memset(y->qs, 0, 12);
        y->seed_lo = 0;
        y->seed_hi = 0;
        return;
    }

    float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) {
        tmp[j] *= inv_norm;
    }

    /* Deterministic seed (must match CPU) */
    /* Note: block index is implicit from pointer offset — we use a simple hash.
     * The seed is stored so dequant can reproduce it. */
    uint64_t ptr_val = (uint64_t)(uintptr_t)y;
    uint32_t seed = (uint32_t)((ptr_val >> 4) * 2654435761u + 0xDEADBEEF);
    y->seed_lo = (uint16_t)(seed & 0xFFFFu);
    y->seed_hi = (uint16_t)(seed >> 16);

    /* Apply random sign flips */
    uint32_t rng_state = seed;
    for (int j = 0; j < 32; j++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        if (rng_state & 1) tmp[j] = -tmp[j];
    }

    /* Walsh-Hadamard Transform (in-place) */
    for (int len = 1; len < 32; len <<= 1) {
        for (int i = 0; i < 32; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                float u = tmp[i + j];
                float v = tmp[i + j + len];
                tmp[i + j]       = u + v;
                tmp[i + j + len] = u - v;
            }
        }
    }

    /* Normalize by 1/sqrt(32) */
    const float inv_sqrt = 1.0f / sqrtf(32.0f);
    const float scale = sqrtf(32.0f);
    for (int j = 0; j < 32; j++) {
        tmp[j] *= inv_sqrt;
    }

    /* Quantize and pack 3-bit indices */
    memset(y->qs, 0, 12);
    for (int j = 0; j < 32; j++) {
        float val = tmp[j] * scale;
        int idx;
        if (val < 0.0f) {
            if (val < -0.7753f) idx = (val < -1.3990f) ? 0 : 1;
            else idx = (val < -0.2503f) ? 2 : 3;
        } else {
            if (val < 0.7753f) idx = (val < 0.2503f) ? 4 : 5;
            else idx = (val < 1.3990f) ? 6 : 7;
        }
        int bit_pos = j * 3;
        int byte_idx = bit_pos / 8;
        int bit_off  = bit_pos % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) {
            y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
        }
    }
}

static __device__ void quantize_f32_tq4_0_block(const float * __restrict__ x, block_tq4_0 * __restrict__ y) {
    const float centroids[16] = {
        -2.4008f, -1.8438f, -1.4371f, -1.0993f,
        -0.7994f, -0.5224f, -0.2582f,  0.0000f,
         0.0000f,  0.2582f,  0.5224f,  0.7994f,
         1.0993f,  1.4371f,  1.8438f,  2.4008f
    };

    float tmp[32];
    float sum_sq = 0.0f;

    for (int j = 0; j < 32; j++) {
        tmp[j] = x[j];
        sum_sq += x[j] * x[j];
    }

    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-10f) {
        memset(y->qs, 0, 16);
        y->seed_lo = 0;
        y->seed_hi = 0;
        return;
    }

    float inv_norm = 1.0f / norm;
    for (int j = 0; j < 32; j++) {
        tmp[j] *= inv_norm;
    }

    uint64_t ptr_val = (uint64_t)(uintptr_t)y;
    uint32_t seed = (uint32_t)((ptr_val >> 4) * 2654435761u + 0xDEADBEEF);
    y->seed_lo = (uint16_t)(seed & 0xFFFFu);
    y->seed_hi = (uint16_t)(seed >> 16);

    uint32_t rng_state = seed;
    for (int j = 0; j < 32; j++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        if (rng_state & 1) tmp[j] = -tmp[j];
    }

    for (int len = 1; len < 32; len <<= 1) {
        for (int i = 0; i < 32; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                float u = tmp[i + j];
                float v = tmp[i + j + len];
                tmp[i + j]       = u + v;
                tmp[i + j + len] = u - v;
            }
        }
    }

    const float inv_sqrt = 1.0f / sqrtf(32.0f);
    const float scale = sqrtf(32.0f);
    for (int j = 0; j < 32; j++) {
        tmp[j] *= inv_sqrt;
    }

    for (int j = 0; j < 16; j++) {
        float v0 = tmp[j * 2] * scale;
        float v1 = tmp[j * 2 + 1] * scale;

        auto quant4 = [](float val) -> int {
            if (val < 0.0f) {
                if (val < -0.9494f) {
                    if (val < -1.6405f) return (val < -2.1223f) ? 0 : 1;
                    else return (val < -1.2682f) ? 2 : 3;
                } else {
                    if (val < -0.3903f) return (val < -0.6609f) ? 4 : 5;
                    else return (val < -0.1291f) ? 6 : 7;
                }
            } else {
                if (val < 0.9494f) {
                    if (val < 0.3903f) return (val < 0.1291f) ? 8 : 9;
                    else return (val < 0.6609f) ? 10 : 11;
                } else {
                    if (val < 1.6405f) return (val < 1.2682f) ? 12 : 13;
                    else return (val < 2.1223f) ? 14 : 15;
                }
            }
        };

        int idx0 = quant4(v0);
        int idx1 = quant4(v1);
        y->qs[j] = (uint8_t)(idx0 | (idx1 << 4));
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
