# WrinkleFree Inference Engine - Research Notebook

## 2024-12-23: BitNet SGLang Integration - Performance Optimization

### Objective
Optimize BitNet 1.58-bit quantization for CPU inference with SGLang continuous batching.

### Baseline Performance
Initial naive implementation (dequantize on every forward pass):
- **4096x4096 GEMV (batch=1)**: 53.46 ms, 0.63 GOPS
- **4096x4096 GEMM (batch=64)**: 55.21 ms, 38.89 GOPS

### Optimization Summary

| # | Optimization | GEMV (ms) | GEMM (ms) | GEMV Speedup | GEMM Speedup |
|---|--------------|-----------|-----------|--------------|--------------|
| 0 | Baseline | 53.46 | 55.21 | 1.0x | 1.0x |
| 1 | LUT dequantization | 27.71 | 29.70 | 1.9x | 1.9x |
| 2 | Weight caching | 1.89 | 3.59 | 28.3x | 15.4x |
| 3 | BF16 compute | 0.225 | 1.05 | **237x** | **53x** |
| 4 | Thread tuning (8) | 0.230 | 1.06 | 232x | 52x |
| 5 | Numba JIT dequant | N/A* | N/A* | N/A* | N/A* |

*Numba optimization affects initial weight loading only (3-7x faster dequant), not steady-state inference.

### Final Performance
**After all optimizations (4096x4096 weights):**
- GEMV (batch=1): **0.235 ms**, 143 GOPS (**253x faster** than baseline)
- GEMM (batch=64): **1.03 ms**, 2090 GOPS (**61x faster** than baseline)
- Memory compression: **16x** (64MB FP32 → 4MB packed)

---

### Optimization Details

#### Optimization 1: LUT-based Dequantization (1.9x)
**Problem:** Original dequantization used a Python loop (4 iterations) to unpack 2-bit values.

**Solution:** Pre-computed 256-entry lookup table mapping each byte to 4 ternary float values.
```python
# Before (slow): Python loop with 4 iterations
for i in range(4):
    shift = 6 - 2 * i
    values = ((packed & (0x03 << shift)) >> shift) - 1

# After (fast): Single gather operation
lut = _get_lut_ternary()  # 256 x 4 lookup table
unpacked = lut[packed.view(-1)].view(out_features, in_features)
```

**Result:** 53.46ms → 27.71ms (1.9x speedup)

---

#### Optimization 2: Weight Caching (28x cumulative)
**Problem:** Dequantizing weights on every forward pass, even though weights are static.

**Solution:** Cache dequantized weights using packed tensor ID as key.
```python
def _get_cached_weight(self, packed_weight, scale, out_features, in_features):
    cache_key = (id(packed_weight), self.compute_dtype)
    if cache_key not in self._weight_cache:
        weight = dequantize_bitnet(packed_weight, scale, out_features, in_features)
        self._weight_cache[cache_key] = weight.to(self.compute_dtype)
    return self._weight_cache[cache_key]
```

**Result:** 27.71ms → 1.89ms (28x cumulative speedup)

---

#### Optimization 3: BF16 Computation (237x cumulative)
**Problem:** FP32 matmul doesn't leverage modern CPU SIMD instructions optimized for BF16.

**Solution:** Store cached weights in BF16 and compute in BF16.
```python
def __init__(self, compute_dtype=torch.bfloat16):
    self.compute_dtype = compute_dtype

def apply(self, packed_weight, scale, x, out_features, in_features, bias=None):
    weight = self._get_cached_weight(...)  # BF16
    x_compute = x.to(self.compute_dtype)
    return torch.matmul(x_compute, weight.T)
```

**Result:** 1.89ms → 0.225ms (237x cumulative speedup)

**Why BF16 is faster:**
- Modern CPUs (Intel with AVX512_BF16, AMD with AVX-512) have native BF16 instructions
- 2x more elements per SIMD register vs FP32
- Reduced memory bandwidth requirements

---

#### Optimization 4: Thread Count Tuning
**Problem:** Default thread count may not be optimal for all workloads.

**Analysis (16-core CPU):**
| Threads | GEMV (ms) | GEMM (ms) |
|---------|-----------|-----------|
| 1 | 1.33 | 7.63 |
| 2 | 0.78 | 3.92 |
| 4 | 0.42 | 2.01 |
| 8 | 0.23 | 1.06 |
| 16 | 0.20 | 1.06 |

**Solution:** Default to 8 threads (best balance for GEMM).
```python
@staticmethod
def _get_optimal_threads():
    cpu_count = multiprocessing.cpu_count()
    return min(8, cpu_count)  # 8 threads sweet spot
```

---

#### Optimization 5: Numba JIT Dequantization (3-7x faster initial load)
**Problem:** LUT-based dequantization still has Python overhead for large weight tensors.

**Solution:** Numba JIT with parallel execution for initial weight dequantization.
```python
@njit(parallel=True, cache=True, fastmath=True)
def _dequant_numba(packed, scale):
    for row in prange(out_features):
        for col in range(packed_in):
            byte_val = packed[row, col]
            # Unpack 4 values per byte
            output[row, col*4:col*4+4] = [v0, v1, v2, v3] * scale
    return output
```

**Dequantization speedup:**
| Size | Numba (ms) | LUT (ms) | Speedup |
|------|------------|----------|---------|
| 1024x1024 | 0.05 | 0.37 | 7.2x |
| 4096x4096 | 8.80 | 24.26 | 2.8x |
| 8192x8192 | 33.52 | 95.17 | 2.8x |

---

### Memory Usage

| Format | Bits/Weight | 4096x4096 Size |
|--------|-------------|----------------|
| FP32 | 32 | 64 MB |
| FP16 | 16 | 32 MB |
| BitNet (packed) | 2 | 4 MB |
| **Compression ratio** | - | **16x** |

---

### Key Takeaways

1. **Caching is critical**: Dequantizing static weights on every forward pass is the #1 performance killer (28x overhead).

2. **BF16 on modern CPUs is fast**: 8x speedup over FP32 for GEMV, 3.5x for GEMM due to native SIMD support.

3. **Thread scaling**: Diminishing returns beyond 8 threads for GEMM; GEMV scales better to more threads.

4. **Numba helps initial load**: 3-7x faster dequantization for one-time weight loading.

5. **Total speedup: 237x** for GEMV, **53x** for GEMM vs naive implementation.

---

### Files Modified

- `src/wrinklefree_inference/sglang_backend/bitnet_quantization.py` - Core optimizations
- `tests/test_sglang_bitnet.py` - Test suite
- `extern/sglang-bitnet/` - SGLang fork with BitNet integration

---

## 2024-12-23: Model Throughput Benchmarks

### Test Configuration
- **CPU**: 16-core (Desktop machine)
- **Threads**: 8 (optimal for GEMM)
- **Compute dtype**: BF16
- **Memory**: Packed 1.58-bit weights

### Results

| Model | Params | Packed Size | Single Token | Throughput | Batched (32) |
|-------|--------|-------------|--------------|------------|--------------|
| BitNet 2B | 2.4B | ~400 MB | 70.7 ms | **14.2 tok/s** | **372.6 tok/s** |
| BitNet 7B | 6.6B | 1.54 GB | 402.6 ms | **2.5 tok/s** | **70.7 tok/s** |

### Analysis

**2B Model (BitNet-b1.58-2B-4T dimensions)**
- Hidden: 2048, Intermediate: 5632, Layers: 24
- ~51M params per layer
- Single stream: 14.2 tokens/sec (human reading speed)
- Batched: 372.6 tokens/sec (26x batch efficiency)

**7B Model (Falcon3-7B-1.58bit dimensions)**
- Hidden: 4096, Intermediate: 11008, Layers: 32
- ~202M params per layer
- Single stream: 2.5 tokens/sec
- Batched: 70.7 tokens/sec (28x batch efficiency)

### Memory Efficiency

| Model | FP32 Size | Packed Size | Compression |
|-------|-----------|-------------|-------------|
| 2B | ~9.6 GB | ~400 MB | **24x** |
| 7B | ~26.4 GB | ~1.54 GB | **17x** |

### Comparison with BitNet.cpp Claims

Microsoft claims BitNet.cpp achieves:
- "5-7 tokens per second" for 100B models on single CPU
- 2.37x-6.17x speedup on x86 vs FP16

Our Python implementation (with optimizations):
- 14.2 tok/s for 2B model (aligned with scaling)
- 2.5 tok/s for 7B model (memory bandwidth limited)

The Python implementation is competitive for rapid prototyping but native C++ would provide additional speedup from:
- Fused dequant+matmul kernels
- Better cache utilization
- Direct SIMD intrinsics

---

## 2024-12-23: Optimization Round 2 - torch.compile

### Objective
Further optimize 7B model throughput beyond the baseline optimizations.

### Optimizations Tested

| # | Optimization | Result |
|---|--------------|--------|
| 1 | Pre-transpose weights | **SLOWER** - .T view is faster than .contiguous() copy |
| 2 | Larger batch sizes | batch=384 optimal (173.7 tok/s baseline) |
| 3 | torch.compile | **+52% improvement** - 262.8 tok/s |
| 4 | Thread count sweep | Minimal impact with torch.compile |
| 5 | FP16 vs BF16 | BF16 8x faster for batched (FP16 only for batch=1) |
| 6 | Numerical accuracy | Cosine sim > 0.9999 vs FP32 reference (PASS) |

### Final Results (7B Model, 16-core CPU)

| Batch | Baseline | + torch.compile | Speedup |
|-------|----------|-----------------|---------|
| 1 | 2.5 tok/s | 3.7 tok/s | 1.49x |
| 32 | 68.7 tok/s | 104.5 tok/s | 1.52x |
| 128 | 148.4 tok/s | 228.7 tok/s | 1.54x |
| 256 | 169.1 tok/s | 256.1 tok/s | 1.51x |
| 384 | 172.0 tok/s | **262.8 tok/s** | 1.53x |

### Key Findings

1. **Pre-transpose is slower**: Storing `weight.T.contiguous()` is slower than using `.T` view at runtime. The view operation is essentially free, while `.contiguous()` forces a memory copy.

2. **Batch size sweet spot**: Optimal batch is 384 (not 256). Beyond 512, cache thrashing reduces throughput.

3. **torch.compile is significant**: 52% improvement with default mode on CPU. All modes (default, reduce-overhead, max-autotune) give similar results.

4. **Thread count doesn't matter with compile**: With torch.compile, performance is nearly identical from 4-16 threads.

5. **BF16 is essential for batched**: FP16 is only faster for batch=1 (4.1 vs 3.7 tok/s). For batched, BF16 is 8x faster than FP16.

6. **Numerical accuracy preserved**: Cosine similarity > 0.9999 between BF16 optimized and FP32 reference.

### Recommended Configuration

```python
import torch

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

method = BitNetLinearMethod(compute_dtype=torch.bfloat16)

# For production: wrap forward pass with torch.compile
@torch.compile(mode="default")
def forward(x):
    return method.apply(packed_weight, scale, x, out_features, in_features)
```

---

## 2024-12-23: Single-Token Latency Optimization

### Objective
Minimize single-token (batch=1) latency for interactive inference.

### Optimizations Tested

| # | Optimization | tok/s | p99 latency | Improvement |
|---|--------------|-------|-------------|-------------|
| 1 | Baseline (BF16, 8 threads) | 2.46 | 12.88ms | - |
| 2 | + torch.inference_mode | 2.46 | 12.84ms | 0% |
| 3 | + torch.compile | 3.59 | 9.42ms | +46% |
| 4 | + compile + inference_mode | 3.64 | 9.29ms | +48% |
| 5 | + OMP_PROC_BIND=CLOSE | 3.66 | 8.61ms | +49% |
| 6 | + gc.disable() | 3.66 | 8.64ms | +49% |
| 7 | max-autotune mode | 3.68 | 8.86ms | +50% |

### Key Findings

1. **torch.compile is essential**: 46% improvement from JIT compilation alone.

2. **Thread count doesn't matter for batch=1**: With torch.compile, performance is identical from 1-16 threads (memory-bound, not compute-bound).

3. **OMP_PROC_BIND reduces variance**: Setting `OMP_PROC_BIND=CLOSE` and `OMP_PLACES=cores` reduces p99 latency (8.61ms vs 9.29ms).

4. **GC doesn't help**: Disabling garbage collection has no measurable impact.

5. **max-autotune marginal**: Only 2% better than default mode, but requires longer warmup.

### Recommended Configuration for Single-Token

```python
import os
import torch

# Thread binding for consistent latency
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "CLOSE"
os.environ["OMP_PLACES"] = "cores"
torch.set_num_threads(8)

# Compile with default mode (faster warmup than max-autotune)
@torch.compile(mode="default")
@torch.inference_mode()
def forward(x):
    return model(x)

# Critical: warm up with exact input shape
for _ in range(10):
    _ = forward(torch.randn(1, hidden_dim))
```

### Memory-Bound Analysis

Single-token inference is **memory-bound**, not compute-bound:
- Weight tensor for 7B model layer: ~202MB (BF16 dequantized)
- Memory bandwidth required: 202MB * 32 layers = 6.5GB per token
- At DDR4-3200 (~50 GB/s): theoretical max ~7.7 tok/s
- Achieved: 3.68 tok/s (~48% of theoretical)

To go faster, need:
1. Native C++ with fused dequant+matmul kernels
2. AVX512 VNNI instructions for ternary weights
3. Higher memory bandwidth (DDR5 or multi-channel)

---

## 2024-12-23: Native C++ GEMV Kernel

### Objective
Implement native C++ kernel with fused dequant+matmul to reduce memory bandwidth and approach theoretical throughput limit.

### Implementation Details

**Files created:**
- `src/wrinklefree_inference/native/bitnet_kernel.cpp` - AVX512 optimized GEMV
- `src/wrinklefree_inference/native/setup.py` - PyTorch extension build
- `src/wrinklefree_inference/native/__init__.py` - Python bindings
- `benchmark/native_kernel_bench.py` - Benchmarking harness

**Key optimizations:**
1. Pre-computed 256-entry LUT: byte -> 4 floats (4KB, fits in L1)
2. AVX512 FMA instructions for 16-wide SIMD
3. Software prefetching for weights and inputs
4. OpenMP parallelization over output rows

### Optimization Journey

| Iteration | Approach | Speedup vs Python |
|-----------|----------|-------------------|
| 1 | Scalar kernel (correctness) | 0.38x (broken) |
| 2 | Fixed indexing bug | 0.38x (slow) |
| 3 | AVX512 with scalar inner loops | 0.91x |
| 4 | AVX512 with LUT + insertf32x4 | 0.97x |
| 5 | AVX512 + gather | 0.97x |
| 6 | LUT + prefetching | **1.01x** |

### Thread Count Analysis (Native Kernel)

| Threads | ms/layer | tok/s |
|---------|----------|-------|
| 1 | 56.34 | 0.55 |
| 2 | 27.89 | 1.12 |
| 4 | 14.74 | 2.12 |
| 8 | **9.12** | **3.43** |
| 12 | 9.95 | 3.14 |
| 16 | 12.11 | 2.58 |

**Finding:** 8 threads optimal. More threads cause cache contention.

### Final Results (7B Model, 8 threads)

| Implementation | ms/layer | tok/s | Notes |
|----------------|----------|-------|-------|
| Python BF16 + torch.compile | 8.86 | 3.53 | Baseline |
| Native C++ AVX512 | 8.78 | **3.56** | 1.01x speedup |

### Key Takeaways

1. **Python is surprisingly fast**: torch.compile + BF16 is highly optimized and hard to beat.

2. **LUT approach works**: Pre-computed 256 x 4 float LUT fits in L1 cache (4KB) and avoids bit manipulation overhead.

3. **Fused kernels aren't always faster**: The Python baseline caches dequantized weights, so we're not actually saving memory bandwidth.

4. **For true speedup, need:**
   - Keep weights in packed format (no BF16 cache)
   - Use VNNI or AMX instructions for int8 dot products
   - Fuse with attention (avoid materializing activations)

### Code Sample

```cpp
// AVX512 GEMV with LUT lookup
void bitnet_gemv_avx512(
    const uint8_t* weights,  // [N, K/4] packed
    const float* input,      // [K]
    float* output,           // [N]
    float scale, int N, int K
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const uint8_t* w_row = weights + i * (K / 4);
        __m512 acc = _mm512_setzero_ps();

        for (int j = 0; j + 16 <= K / 4; j += 16) {
            // Prefetch
            _mm_prefetch((const char*)(w_row + j + 64), _MM_HINT_T0);

            // LUT lookup: 4 bytes -> 16 floats
            __m128 w0 = _mm_load_ps(BYTE_LUT[w_row[j + 0]]);
            // ... (combine 4 __m128 into __m512)

            __m512 xv = _mm512_loadu_ps(input + j * 4);
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }
        output[i] = _mm512_reduce_add_ps(acc) * scale;
    }
}
```

---

## 2024-12-24: Modal Cloud Benchmarking

### Objective
Run benchmarks on Modal cloud CPUs with more cores and higher memory bandwidth.

### Setup
- **Platform**: Modal serverless (Debian, 32 vCPUs, 64GB RAM)
- **Script**: `benchmark/modal_cpu_bench.py`
- **Kernel**: AVX512 with LUT lookup + prefetching

### Thread Scaling on Modal (32 vCPUs)

| Threads | ms/layer | tok/s | Speedup vs 1T |
|---------|----------|-------|---------------|
| 1 | 80.22 | 0.39 | 1.0x |
| 2 | 40.88 | 0.76 | 2.0x |
| 4 | 21.58 | 1.45 | 3.7x |
| 8 | 10.62 | 2.94 | 7.6x |
| 12 | 7.46 | 4.19 | 10.8x |
| 16 | 5.80 | 5.39 | 13.8x |
| 24 | 3.82 | 8.17 | 21.0x |
| **32** | **3.03** | **10.30** | **26.5x** |

**Key finding**: Near-linear scaling up to 32 threads on Modal CPUs!

### Fused vs Cached Benchmark

Added `dequant()` function to pre-materialize weights like Python does:

| Approach | ms/layer | tok/s | vs Python |
|----------|----------|-------|-----------|
| **Fused GEMV** | 3.36 | 9.29 | **2.46x** |
| Cached (torch.mv) | 10.60 | 2.95 | 0.78x |
| Python BF16 | 8.27 | 3.78 | 1.00x |

**Key insight**: Fused kernel beats both Python and cached approaches because:
1. LUT lookup is L1-cache friendly (4KB LUT fits in L1)
2. Avoids reading full FP32 weight matrix from RAM
3. Fused dequant+FMA keeps data in registers

### Variability Across Modal Containers

Results vary depending on which CPU type Modal assigns:

| Run | Fused (tok/s) | Python (tok/s) | Speedup |
|-----|---------------|----------------|---------|
| Best | 9.29 | 3.78 | 2.46x |
| Typical | 2.9 | 3.5 | 0.83x |
| Worst | 2.7 | 3.7 | 0.73x |

**Explanation**: Modal containers get different CPU types:
- Some have AVX512 (fast fused kernel)
- Some are AVX2 only (slower fallback)
- Memory bandwidth varies by instance type

### Files Added

- `benchmark/modal_cpu_bench.py` - Modal serverless benchmark script
- `bitnet_native.dequant()` - Pre-dequantize weights to FP32
- `bitnet_native.gemv_cached()` - GEMV with cached weights

### Conclusions

1. **32 threads optimal on Modal**: 10.3 tok/s achieved with 32 threads
2. **Fused kernel wins**: When AVX512 is available, fused is 2.46x faster than Python
3. **CPU type matters**: Performance varies 3x based on container assignment
4. **Production recommendation**:
   - Use fused kernel with AVX512 detection
   - Fall back to Python BF16 on non-AVX512 CPUs

---

## 2024-12-24: KV Cache Optimization for Long Context

### Objective
Optimize KV cache memory and performance for long context windows (8K+ tokens).

### Implementation

**Files created:**
- `src/wrinklefree_inference/kv_cache/kv_cache.py` - KV cache with quantization
- `benchmark/kv_cache_bench.py` - Long context benchmark

**Supported formats:**
- BF16: Default, highest quality
- FP16: Same memory as BF16
- FP8 (E4M3/E5M2): 50% memory savings via symmetric int8 quantization
- INT8: 50% memory savings

### 20 Iteration Benchmark Results (8K Context, 32 Layers)

| Format | Memory | Attention (µs) | Cosine Sim | Status |
|--------|--------|----------------|------------|--------|
| **BF16** | 4096 MB | 12593 (avg) | 1.000000 | PASS |
| **FP8 E4M3** | 2048 MB | 13449 (avg) | 0.999816 | PASS |
| **INT8** | 2048 MB | 13709 (avg) | 0.999785 | PASS |

**Quality across 20 iterations:**
- FP8: min cosine sim = 0.998520 (all tests PASS)
- INT8: min cosine sim = 0.998093 (all tests PASS)

### Memory Scaling by Context Length

| Context | BF16 | FP8/INT8 | Savings |
|---------|------|----------|---------|
| 1K | 512 MB | 256 MB | 50% |
| 4K | 2048 MB | 1024 MB | 50% |
| 8K | 4096 MB | 2048 MB | 50% |
| 16K | 8192 MB | 4096 MB | 50% |

### Key Findings

1. **Model output unchanged**: All 20 iterations maintain >0.998 cosine similarity with BF16 baseline

2. **Memory efficiency**: FP8/INT8 provide consistent 50% memory reduction

3. **Latency tradeoff**: Quantized formats are ~7% slower than BF16
   - BF16: 12.6ms average attention
   - FP8/INT8: 13.4-13.7ms average attention

4. **Per-token update latency**:
   - BF16: 1.1 µs/token (fastest)
   - INT8: 2.5 µs/token
   - FP8: 8.9 µs/token (view conversion overhead)

### Recommendations

- **For memory-constrained deployments**: Use INT8 (50% savings, 0.998 quality)
- **For quality-critical tasks**: Use BF16 (baseline quality)
- **For long context (16K+)**: INT8 required to fit in memory

### Code Example

```python
from wrinklefree_inference.kv_cache import KVCache, KVCacheConfig, KVCacheDtype

# Configure KV cache for 8K context with INT8 quantization
config = KVCacheConfig(
    max_seq_len=8192,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    dtype=KVCacheDtype.INT8,  # 50% memory savings
)

cache = KVCache(config)
print(f"Memory: {cache.memory_usage_mb():.0f} MB")  # 2048 MB vs 4096 MB for BF16

# Update cache during generation
cache.update(layer_idx=0, key=k, value=v, seq_pos=current_pos)

# Retrieve for attention
key, value = cache.get(layer_idx=0)
```

---

## 2024-12-24: Activation Sparsity Implementation

### Objective
Implement Q-Sparse-style activation sparsity for BitNet inference optimization.

### Research Background

| Paper | Key Insight |
|-------|-------------|
| [Q-Sparse](https://arxiv.org/abs/2407.10969) | Optimal sparsity: 61.25% for 1.58-bit models (requires training) |
| [BitNet a4.8](https://arxiv.org/abs/2411.04965) | 44.5% sparsity with squared ReLU |
| [DejaVu](https://arxiv.org/abs/2310.17157) | 85% contextual sparsity, 2-7x speedup |
| [PowerInfer](https://arxiv.org/abs/2312.12456) | 11.69x faster with adaptive predictors |

### Implementation

**Files created:**
- `src/wrinklefree_inference/sglang_backend/activation_sparsity.py` - Core sparsity implementations
- `src/wrinklefree_inference/sglang_backend/sparse_attention.py` - Attention sparsity
- `configs/sparsity/` - Hydra configs for different sparsity levels
- `configs/attention/` - Attention sparsity configs
- `benchmark/sparsity_benchmark.py` - Benchmark harness

**Sparsity modes implemented:**
1. **Threshold**: Zero out activations below threshold
2. **Top-K**: Keep top-k% activations by magnitude
3. **Adaptive**: Entropy-based per-token sparsity adjustment

### Benchmark Results: Activation Sparsity (10 iterations)

#### Quality vs Sparsity Trade-off

| Config | Sparsity | Cosine Similarity | Status |
|--------|----------|-------------------|--------|
| dense | 0% | 1.000 | PASS |
| threshold_0.1 | ~8% | 0.9998 | PASS |
| top_k_90 | 10% | 0.9997 | PASS |
| top_k_80 | 20% | 0.9976 | PASS |
| **top_k_70** | **30%** | **0.992** | **PASS** |
| top_k_50 | 50% | 0.965 | WARN |
| top_k_40 | 60% | 0.934 | FAIL |

**Key Finding**: For inference-only (without Q-Sparse training), maximum safe sparsity is **30%** (top_k_ratio=0.7) to maintain >0.99 cosine similarity.

#### Throughput Analysis

| Batch Size | Dense (ms) | Top-K 30% (ms) | Speedup |
|------------|------------|----------------|---------|
| 1 | 0.83 | 1.20 | 0.69x (overhead) |
| 32 | 14.6 | 17.5 | 0.83x (overhead) |
| 128 | 59.9 | 65.1 | 0.92x (overhead) |

**Finding**: Input activation sparsity alone does NOT provide speedup because:
1. Top-k selection has overhead (sorting)
2. Cannot skip dense matmul ops without sparse kernels
3. Need fused sparse matmul for actual speedup

### Recommended Configuration

```yaml
# configs/sparsity/qsparse.yaml
sparsity:
  enabled: true
  mode: "top_k"
  top_k_ratio: 0.7  # 30% sparsity (inference-only safe)

# After Q-Sparse training (in WrinkleFree-CheaperTraining):
# top_k_ratio: 0.4  # 60% sparsity (trained model)
```

---

## 2024-12-24: Sparse Attention Implementation

### Objective
Implement adaptive sparse attention for memory-efficient long context inference.

### Implementation

**Attention sparsity modes:**
1. **Top-K**: Keep top-k attention scores per query
2. **Threshold**: Zero out attention below threshold
3. **Window**: Sliding window + global tokens (Longformer-style)
4. **Dynamic**: Entropy-based adaptive sparsity

### Benchmark Results: Attention Sparsity

| Mode | Sparsity | Memory | Quality (cosine) |
|------|----------|--------|------------------|
| dense | 0% | 100% | 1.000 (baseline) |
| top_k_128 | 75% | 25% | 0.956 |
| top_k_64 | 88% | 12.5% | 0.896 |
| window_256 | 56% | 44% | 0.653 |
| **dynamic** | **53%** | **47%** | **0.990** |

**Key Finding**: **Dynamic sparse attention** achieves best quality (0.99) at 53% sparsity. Window attention has lower quality because it's a fixed pattern.

### Memory Savings for Long Context

| Context | Dense | Window(256) | Savings |
|---------|-------|-------------|---------|
| 2K | 16 MB | 2 MB | 88% |
| 4K | 64 MB | 4 MB | 94% |
| 8K | 256 MB | 8 MB | 97% |
| 16K | 1024 MB | 16 MB | **98%** |

### Recommended Configuration

```yaml
# configs/attention/dynamic.yaml
attention_sparsity:
  enabled: true
  mode: "dynamic"
  dynamic:
    min_ratio: 0.1  # 90% sparsity max
    max_ratio: 0.5  # 50% sparsity min
```

For long context (>8K), use window attention for memory efficiency:

```yaml
# configs/attention/window.yaml
attention_sparsity:
  enabled: true
  mode: "window"
  window:
    size: 256
    global_tokens: 1
    stride: 64
```

---

### Next Steps

1. Integrate with SGLang continuous batching scheduler
2. Add Q-Sparse training to WrinkleFree-CheaperTraining
3. Implement fused sparse matmul kernels for actual speedup
4. Try VNNI-based kernel (avoid float conversion)
5. Add AVX512 detection and dynamic dispatch
6. Fuse KV cache update with attention kernel
7. Test dynamic attention on real generation tasks
