# TurboQuant KV Cache — Build & Usage Guide

## What This Is

Implementation of **TurboQuant** (arXiv:2504.19874, Daliri et al. 2025) as new KV cache 
quantization types in llama.cpp.

**Algorithm**: PolarQuant = random Hadamard rotation + optimal Lloyd-Max scalar quantization  
**Key insight**: Rotation makes KV vector coordinates near-independent with known Beta 
distribution → simple per-coordinate scalar quantization becomes near-optimal.

New types:
- `tq3_0` — 3-bit TurboQuant, 18 bytes/32 elements (~4.5 bpw)  
- `tq4_0` — 4-bit TurboQuant, 22 bytes/32 elements (~5.5 bpw)

## Hardware

Tested on ncc1701e: i9-7900X (AVX512), 64GB RAM, Tesla P40 24GB

## Build

```bash
cd /home/will/.openclaw/workspace/projects/llama-turboquant

# CPU-only build
mkdir -p build && cd build
cmake .. -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
make -j$(nproc)

# With CUDA (P40)
cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF
make -j$(nproc)
```

## Usage

```bash
# 3-bit KV cache (4.5 bpw) — best quality/size tradeoff
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  --cache-type-k tq3_0 \
  --cache-type-v tq3_0 \
  -c 8192 \
  -p "Your prompt here"

# 4-bit KV cache (5.5 bpw) — near-lossless
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  --cache-type-k tq4_0 \
  --cache-type-v tq4_0 \
  -c 8192 \
  -p "Your prompt here"

# Compare: q4_0 KV cache (baseline)
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -c 8192 \
  -p "Your prompt here"
```

## Benchmark Memory Savings

For a 32K context with Qwen3-32B (128 layers × 1024 KV head dim):
| Cache Type | bpw  | Memory (est.) | Relative |
|------------|------|---------------|----------|
| f16        | 16.0 | 8.0 GB        | 1.0×     |
| q8_0       | 8.5  | 4.25 GB       | 1.9×     |
| q4_0       | 4.5  | 2.25 GB       | 3.6×     |
| tq3_0      | 4.5  | 2.25 GB       | 3.6×     |
| tq4_0      | 5.5  | 2.75 GB       | 2.9×     |

Note: tq3_0 has same bpw as q4_0 but **better theoretical MSE** due to optimal Lloyd-Max 
codebook (3-bit quantization beats 4-bit uniform quantization with scale factor).

## Quantization Quality (from unit tests)

From `tests/test-turboquant.c` on Gaussian inputs (simulating KV cache activations):

```
TQ3_0: RelMSE ≈ 4.2%   (paper theory: ~3.0% for unit vectors)
TQ4_0: RelMSE ≈ 0.99%  (paper theory: ~0.9% for unit vectors)

Rotation invertibility: max error < 1e-5 (numerically perfect)
Block sizes: tq3_0=18 bytes, tq4_0=22 bytes (both verified correct)
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

### Quantize Algorithm (per 32-element block)

1. Compute L2 norm, store in `d`, normalize vector to unit sphere
2. Generate block-specific rotation seed = `hash(block_index)`
3. Apply PolarQuant rotation: `random_sign_flip → WHT(x)/sqrt(32)`
   - Fast Walsh-Hadamard Transform: O(d log d), no multiplications
   - Random sign flips: reproducible from seed via xorshift32
4. Scale rotated coordinates by `sqrt(32)` to map to N(0,1) space
5. For each coordinate: find nearest Lloyd-Max centroid (binary search, 3 or 4 ops)
6. Pack centroid indices into `qs[]`

### Dequantize Algorithm

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

## Files Modified

### New files:
- `ggml/src/ggml-turboquant.h` — Rotation primitives (WHT, xorshift, codebooks)

### Modified files:
- `ggml/include/ggml.h` — Added `GGML_TYPE_TQ3_0=41`, `GGML_TYPE_TQ4_0=42`, `COUNT=43`
- `ggml/src/ggml-common.h` — Block struct definitions for `block_tq3_0`, `block_tq4_0`
- `ggml/src/ggml-quants.h` — Function declarations
- `ggml/src/ggml-quants.c` — Quantize/dequantize implementations + validate_row_data
- `ggml/src/ggml.c` — Type traits registration, quantize_chunk dispatch
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU backend type traits (from_float)
- `ggml/src/ggml-cpu/quants.h` — CPU wrapper declarations
- `ggml/src/ggml-cpu/quants.c` — CPU wrapper implementations
- `ggml/src/ggml-cpu/ops.cpp` — Case labels for quantized type switch statements
- `common/arg.cpp` — Added tq3_0, tq4_0 to `kv_cache_types` list

## Phase 2 (TODO)

- [ ] Specialized `vec_dot` kernel for attention computation in rotated domain
  - Currently: dequantizes to f32 then uses standard attention
  - Target: compute dot products directly in quantized domain (skip dequant)
- [ ] AVX512 SIMD kernel for x86 (ncc1701e: i9-7900X)
- [ ] ARM NEON kernel
- [ ] CUDA kernel for P40
- [ ] Perplexity benchmarks vs q4_0 / q8_0 / f16
- [ ] QJL residual correction (TurboQuant_prod) for unbiased inner product estimation
  - Currently implements TurboQuant_mse (MSE-optimal)
  - For KV cache the MSE-optimal variant may already be sufficient

## Branch

```
git remote: https://github.com/ggml-org/llama.cpp (origin)
branch: feat/turboquant-kv-cache
```
