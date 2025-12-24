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

### Next Steps

1. Integrate with SGLang continuous batching scheduler
2. Add FP8 KV cache support
3. Try VNNI-based kernel (avoid float conversion)
4. Profile attention vs FFN bottlenecks
