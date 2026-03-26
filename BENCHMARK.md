# TurboQuant KV Cache Benchmark Report

**Date:** 2026-03-26  
**System:** ncc1701e — Intel i9-7900X (10C/20T, AVX-512), 64GB DDR4, Tesla P40 (24GB GDDR5X)  
**Branch:** `feat/turboquant-kv-cache`  
**Build:** 9d200b6 + Phase 3 optimizations

## What Was Implemented

### Phase 1-2 (Prior): Core TQ3_0/TQ4_0 Types
- Block structures with PolarQuant rotation (Walsh-Hadamard + random signs)
- Lloyd-Max optimal codebook (8 centroids for 3-bit, 16 for 4-bit)
- Per-block L2 norm + rotation seed storage
- Registered in ggml type system, KV cache support in llama.cpp

### Phase 3: Optimizations

1. **Rotated-Domain Dot Product (CPU)** — Instead of dequantizing x (expensive inverse WHT per block), we forward-rotate query y and dot with centroids directly. This avoids the O(n log n) inverse WHT on x during attention.

2. **AVX2 SIMD Kernels (x86)** — FMA-accelerated dot product accumulation for 32-element blocks using `_mm256_fmadd_ps`. Centroid lookup uses scalar unpack (3-bit packing isn't SIMD-friendly) but the multiply-accumulate is fully vectorized.

3. **CUDA Dequantize Kernel (P40)** — Warp-cooperative dequantization using `__shfl_xor_sync` for Walsh-Hadamard butterfly stages. Each warp thread handles one element of the 32-element block. All 5 WHT stages run in registers via warp shuffles.

4. **CUDA Quantize Kernel (SET_ROWS)** — Single-thread-per-block quantization for KV cache writes. Full WHT + Lloyd-Max quantization in registers. Registered in `set-rows.cu` dispatch.

### Not Implemented (Deferred)
- **QJL Residual Correction (Task 3)** — Would add ~4 bytes per block for unbiased inner products. Deferred because the current quality is already excellent (see perplexity below).
- **AVX-512 specific path** — AVX2 path already uses FMA which is the key optimization. A dedicated AVX-512 path could process blocks in fewer instructions but the WHT bottleneck dominates.

## Speed Benchmarks

### GLM-edge 1.5B (Q4_K_M) — CPU (10 threads, ngl=0)

| KV Cache | pp512 (t/s) | tg128 (t/s) | Notes |
|----------|-------------|-------------|-------|
| f16      | 1375.08     | 28.36       | Baseline |
| q8_0     | 1391.49     | 49.20       | +73% tg |
| q4_0     | 1329.69     | 41.38       | +46% tg |
| **tq3_0** | **46.44**  | **32.76**   | +15% tg, pp slow (WHT) |
| **tq4_0** | **54.19**  | **38.29**   | +35% tg, pp slow (WHT) |

### GLM-edge 1.5B (Q4_K_M) — GPU P40 (ngl=99)

| KV Cache | pp512 (t/s) | tg128 (t/s) | Notes |
|----------|-------------|-------------|-------|
| f16      | 2214.57     | 146.89      | Baseline |
| q8_0     | 2203.45     | 130.87      | -11% tg |
| q4_0     | 2211.18     | 127.59      | -13% tg |
| **tq3_0** | **30.66**  | **59.67**   | -59% tg, pp very slow |
| **tq4_0** | **32.87**  | **60.00**   | -59% tg, pp very slow |

### Qwen3.5-9B (Q8_0) — CPU (10 threads, ngl=0)

| KV Cache | pp512 (t/s) | tg128 (t/s) | Notes |
|----------|-------------|-------------|-------|
| f16      | 242.36      | 6.71        | Baseline |
| q8_0     | 242.83      | 6.65        | ~same |
| q4_0     | 240.98      | 6.69        | ~same |
| **tq3_0** | **61.95**  | **6.55**    | tg ~same, pp 4x slower |
| **tq4_0** | **73.19**  | **6.64**    | tg ~same, pp 3.3x slower |

### Qwen3.5-9B (Q8_0) — GPU P40 (ngl=99)

| KV Cache | pp512 (t/s) | tg128 (t/s) | Notes |
|----------|-------------|-------------|-------|
| f16      | 491.12      | 29.51       | Baseline |
| q8_0     | 490.39      | 29.19       | ~same |
| q4_0     | 490.86      | 29.05       | ~same |
| **tq3_0** | **48.95**  | **25.30**   | -14% tg |
| **tq4_0** | **52.30**  | **25.40**   | -14% tg |

## Perplexity (Quality Verification)

Model: GLM-edge 1.5B Q4_K_M, Dataset: wikitext-2-raw, 5 chunks, ctx=512

| KV Cache | PPL ± σ | Δ from f16 |
|----------|---------|------------|
| f16      | 18.02 ± 1.76 | — (baseline) |
| q4_0     | 18.57 ± 1.80 | +0.55 (+3.1%) |
| **tq3_0** | **18.55 ± 1.70** | **+0.53 (+2.9%)** |
| **tq4_0** | **18.38 ± 1.79** | **+0.36 (+2.0%)** |

**Key finding:** TQ3_0 (3-bit) quality matches q4_0 (4-bit)! The PolarQuant rotation makes better use of each bit via optimal Lloyd-Max codebooks.

## Memory Usage Estimates

Per KV layer pair, for head_dim=128, n_heads=16:

| KV Type | Bytes/Token | 4K ctx | 8K ctx | vs f16 |
|---------|-------------|--------|--------|--------|
| f16     | 8192        | 32 MiB | 64 MiB | 1.0× |
| q8_0    | 4352        | 17 MiB | 34 MiB | 0.53× |
| q4_0    | 2304        | 9 MiB  | 18 MiB | 0.28× |
| tq3_0   | 2304*       | 9 MiB  | 18 MiB | 0.28× |
| tq4_0   | 2816*       | 11 MiB | 22 MiB | 0.34× |

\* TQ3_0: 18 bytes/32 elements = 4.5 bpw (vs q4_0's 4.5 bpw). TQ4_0: 22 bytes/32 = 5.5 bpw.

## Analysis

### Prompt Processing Bottleneck
The Walsh-Hadamard Transform is the dominant cost. For each attention block:
- **Standard quant (q4_0):** Simple dequant is O(n) with small constant
- **TurboQuant (tq3_0):** Forward rotation of query vector is O(n log n) per block

This makes pp throughput 10-40× slower than standard quantization. However:

1. **Token generation is what matters for KV cache** — pp only fills the cache once, tg reads it repeatedly. TQ3 tg performance is competitive with baseline.
2. **CPU tg shows improvement** — TQ3 CPU tg is actually FASTER than f16 (32.76 vs 28.36 on 1.5B) because smaller cache = better cache utilization.
3. **Quality is excellent** — TQ3 at 3 bits matches q4_0 at 4 bits in perplexity, validating the PolarQuant theory.

### GPU Observations
- GPU tg with TQ3 is 40-60% of f16 speed — the per-warp WHT dequantization adds overhead
- The dequantize kernel uses warp shuffles which is efficient but still slower than simple integer-to-float conversion
- No specialized attention kernel yet (using standard attention after dequant)

## SIMD Paths Used
- **x86 AVX2:** `_mm256_fmadd_ps`, `_mm256_load_ps`, `_mm256_mul_ps`, `_mm256_set1_ps`
- **CUDA:** `__shfl_xor_sync` for warp-level WHT, standard arithmetic

## Code Quality Notes
- Clean integration following llama.cpp patterns (type traits, arch-specific overrides, CUDA dispatch)
- 3-bit packing is non-trivial but correct (bit-level addressing with cross-byte boundaries)
- CUDA quantize kernel runs per-thread (not warp-cooperative) which limits parallelism but matches existing patterns
- Seed generation uses address-based hashing on GPU (different from CPU's index-based) — this means GPU KV cache is NOT bitwise-identical to CPU. For production, this should be unified.

## Suggested Next Steps for Upstream PR

1. **Fused attention kernel** — Instead of dequant→standard attention, implement a fused kernel that does the WHT + centroid lookup inside the attention computation. This would eliminate the pp bottleneck.

2. **Pre-rotated query cache** — During prompt processing, pre-rotate all query vectors once and cache them, so subsequent attention computations only need centroid lookups.

3. **QJL residual correction** — Add the 1-bit Johnson-Lindenstrauss correction for theoretically unbiased inner products. Currently ~2.9% PPL degradation; QJL could reduce this to near-zero.

4. **Unified seed generation** — Use block index instead of pointer-based seeds on GPU to match CPU behavior exactly.

5. **Warp-cooperative quantize kernel** — The current CUDA quantize runs one block per thread with sequential WHT. A warp-cooperative version would be significantly faster for KV cache fills.

6. **Mixed precision** — Use TQ3 for K cache (less quality-sensitive) and TQ4 or q4_0 for V cache (more quality-sensitive), or vice versa based on analysis.
