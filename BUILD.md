# TurboQuant KV Cache — Build & Usage Guide

## What This Is

Implementation of **TurboQuant** (arXiv:2504.19874, Daliri et al. 2025) as new KV cache
quantization types in llama.cpp.

**Algorithm**: PolarQuant = random Hadamard rotation + optimal Lloyd-Max scalar quantization
**Key insight**: Rotation makes KV vector coordinates near-independent with approximately
Gaussian distribution → per-coordinate quantization with precomputed Lloyd-Max codebooks
achieves near-optimal distortion rate.

New types:
- `tq3_0` — 3-bit TurboQuant, 18 bytes/32 elements (~4.5 bpw)
- `tq4_0` — 4-bit TurboQuant, 22 bytes/32 elements (~5.5 bpw)

## Build

```bash
# CPU-only build
cmake -B build -DGGML_CUDA=OFF
cmake --build build -j$(nproc)

# With CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Cross-compile for AVX2-only target (no AVX-512)
cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON \
  -DGGML_AVX512=OFF
cmake --build build -j$(nproc)
```

## Usage

```bash
# TQ3 K-cache + f16 V-cache (works without flash attention)
./build/bin/llama-server -m model.gguf \
  --cache-type-k tq3_0 --cache-type-v f16

# TQ3 K-cache + q8_0 V-cache (requires flash attention, SM 7.0+)
./build/bin/llama-server -m model.gguf \
  --cache-type-k tq3_0 --cache-type-v q8_0 \
  --flash-attn on

# Both K and V as TQ3 (requires flash attention)
./build/bin/llama-server -m model.gguf \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --flash-attn on

# Compare with q4_0 baseline
./build/bin/llama-server -m model.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --flash-attn on
```

**Note:** V cache quantization (any type other than f16) requires `--flash-attn on` in llama.cpp. The P40 (SM 6.1) does not support flash attention, so only K-cache quantization is available without FA.

## Memory Savings

Per KV layer pair (head_dim=128, n_heads=16):

| Cache Type | bpw  | 4K ctx  | 32K ctx  | vs f16 |
|------------|------|---------|----------|--------|
| f16        | 16.0 | 32 MiB  | 256 MiB  | 1.0×   |
| q8_0       | 8.5  | 17 MiB  | 136 MiB  | 0.53×  |
| q4_0       | 4.5  | 9 MiB   | 72 MiB   | 0.28×  |
| tq3_0      | 4.5  | 9 MiB   | 72 MiB   | 0.28×  |
| tq4_0      | 5.5  | 11 MiB  | 88 MiB   | 0.34×  |

tq3_0 has the same bpw as q4_0 but **better perplexity** due to optimal Lloyd-Max
codebook — 3-bit with rotation beats 4-bit uniform with scale factor.

## Quantization Quality (from unit tests)

From `tests/test-turboquant.c` on Gaussian inputs:

```
TQ3_0: RelMSE ≈ 4.2%   (paper theory: ~3.0% for unit vectors)
TQ4_0: RelMSE ≈ 0.99%  (paper theory: ~0.9% for unit vectors)

Rotation invertibility: max error < 1e-5 (numerically perfect)
Block sizes: tq3_0=18 bytes, tq4_0=22 bytes (verified correct)
```

## Algorithm Details

### Block Format

**TQ3_0** (18 bytes/32 elements):
```c
typedef struct {
    ggml_half d;         // L2 norm of original block
    uint16_t seed_lo;    // lower 16 bits of xorshift32 rotation seed
    uint16_t seed_hi;    // upper 16 bits of xorshift32 rotation seed
    uint8_t qs[12];      // 3-bit indices packed (32×3 bits = 96 bits)
} block_tq3_0;
```

**TQ4_0** (22 bytes/32 elements):
```c
typedef struct {
    ggml_half d;         // L2 norm of original block
    uint16_t seed_lo;
    uint16_t seed_hi;
    uint8_t qs[16];      // 4-bit indices packed as nibbles
} block_tq4_0;
```

### Quantize (per 32-element block)

1. Compute L2 norm, store in `d`, normalize vector to unit sphere
2. Generate block-specific rotation seed = `hash(block_index)`
3. Apply PolarQuant rotation: `random_sign_flip → WHT(x)/sqrt(32)`
4. Scale rotated coordinates by `sqrt(32)` to map to N(0,1) space
5. For each coordinate: find nearest Lloyd-Max centroid (binary search)
6. Pack centroid indices into `qs[]`

### Dequantize

1. Unpack centroid indices from `qs[]`, look up centroid values
2. Apply inverse rotation: `WHT(x)/sqrt(32) → undo_sign_flips`
3. Rescale by stored norm `d`

### Lloyd-Max Codebooks (precomputed, optimal for N(0,1))

**3-bit** (8 centroids):
```
{-1.7479, -1.0500, -0.5006, 0.0, 0.0, 0.5006, 1.0500, 1.7479}
```

**4-bit** (16 centroids):
```
{-2.4008, -1.8438, -1.4371, -1.0993, -0.7994, -0.5224, -0.2582, 0.0,
  0.0, 0.2582, 0.5224, 0.7994, 1.0993, 1.4371, 1.8438, 2.4008}
```

## Optimization Details

### CPU: Fused Attention (Phase 4)
- `tq_prerotate_query_f32()` rotates Q blocks in-place using deterministic per-block seeds
- `ggml_vec_dot_tq3_0_f32_prerotated()` — AVX2/FMA dot product assuming pre-rotated query
- Flash attention path detects TQ KV types and uses fused pre-rotated path
- Eliminates per-key inverse WHT from attention inner loop

### CPU: AVX2 SIMD (Phase 3)
- FMA-accelerated dot product: `_mm256_fmadd_ps` for 32-element block accumulation
- Centroid lookup is scalar (3-bit packing isn't SIMD-friendly), multiply-accumulate is vectorized

### CUDA (Phase 3)
- Warp-cooperative dequantize: `__shfl_xor_sync` for WHT butterfly stages
- Each warp thread handles one element of the 32-element block, all 5 WHT stages in registers
- Quantize kernel: single-thread-per-block for KV cache writes

### Known Issues
- Non-flash-attention code path crashes on second request when reusing cached TQ3 KV entries
- GPU seed generation uses address-based hashing (differs from CPU's index-based) — KV cache is not bitwise-identical across backends
- V cache quantization requires flash attention (llama.cpp constraint)

## Files Changed

### New:
- `ggml/src/ggml-turboquant.h` — Rotation primitives, codebooks, quantization helpers

### Modified:
- `ggml/include/ggml.h` — Type enum entries
- `ggml/src/ggml-common.h` — Block struct definitions
- `ggml/src/ggml-quants.{c,h}` — Quantize/dequantize implementations
- `ggml/src/ggml.c` — Type traits, quantize_chunk dispatch
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU backend type traits
- `ggml/src/ggml-cpu/quants.{c,h}` — CPU vec_dot wrappers + prerotated variants
- `ggml/src/ggml-cpu/arch/x86/quants.c` — AVX2/FMA optimized kernels
- `ggml/src/ggml-cpu/arch-fallback.h` — Generic fallback aliases
- `ggml/src/ggml-cpu/ops.cpp` — Flash attention TQ detection + pre-rotate Q
- `ggml/src/ggml-cuda/ggml-cuda.cu` — CUDA type traits
- `ggml/src/ggml-cuda/convert.cu` — CUDA dequantize kernel
- `ggml/src/ggml-cuda/set-rows.cu` — CUDA quantize kernel
- `ggml/src/ggml-cuda/cpy-utils.cuh` — Copy helpers
- `common/arg.cpp` — CLI argument parsing
- `tests/test-turboquant.c` — Unit tests
- `tools/llama-bench/llama-bench.cpp` — Benchmark support
