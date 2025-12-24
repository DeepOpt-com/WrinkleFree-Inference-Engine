"""Native AVX2 BitNet kernel benchmark on Modal.

Builds and benchmarks native SIMD kernels vs Python fallback.
"""

import modal

app = modal.App("bitnet-native-kernel")

# Image with build tools for native kernel compilation
native_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "build-essential",
        "cmake",
        "ninja-build",
        "libomp-dev",
    ])
    .pip_install([
        "torch==2.5.1+cpu",
        "numpy",
        "pybind11",
    ], extra_index_url="https://download.pytorch.org/whl/cpu")
)

# C++ kernel source (AVX2 optimized)
KERNEL_CPP = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

constexpr int QK_I2_S = 128;

// Horizontal sum for AVX2
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(a),
        _mm256_extractf128_si256(a, 1)
    );
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// BitNet GEMV: dot product of packed 2-bit weights and INT8 activations
float bitnet_vec_dot_i2_i8_avx2(
    int n,
    const uint8_t* packed_weights,
    const int8_t* activations
) {
    if (n % QK_I2_S != 0) {
        throw std::invalid_argument("n must be multiple of 128");
    }

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i = 0; i < group32_num; i++) {
        __m256i accu32 = _mm256_setzero_si256();

        for (int j = 0; j < 32; j++) {
            __m256i xq8_3 = _mm256_loadu_si256(
                (const __m256i*)(packed_weights + i * 32 * 32 + j * 32)
            );

            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
            xq8_3 = _mm256_and_si256(xq8_3, mask);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + i * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + i * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + i * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + i * 128 * 32 + j * 128 + 96));

            // Subtract 1 from weights: w_encoded = w + 1, so multiply by (w_encoded - 1) = w
            // maddubs computes sum(a[i] * b[i]) where a is unsigned, b is signed
            // We have w_encoded in [0,1,2], need to compute (w_encoded - 1) * activation
            // Split: w_encoded * activation - 1 * activation = w_encoded * activation - activation
            __m256i ones = _mm256_set1_epi8(1);

            // Compute w_encoded * activation (treating w_encoded as unsigned 0-2)
            __m256i prod0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            __m256i prod1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            __m256i prod2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            __m256i prod3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            // Compute 1 * activation = sum of activations (to subtract)
            __m256i sum0 = _mm256_maddubs_epi16(ones, yq8_0);
            __m256i sum1 = _mm256_maddubs_epi16(ones, yq8_1);
            __m256i sum2 = _mm256_maddubs_epi16(ones, yq8_2);
            __m256i sum3 = _mm256_maddubs_epi16(ones, yq8_3);

            // Result = prod - sum = (w_encoded - 1) * activation = w * activation
            xq8_0 = _mm256_sub_epi16(prod0, sum0);
            xq8_1 = _mm256_sub_epi16(prod1, sum1);
            xq8_2 = _mm256_sub_epi16(prod2, sum2);
            xq8_3 = _mm256_sub_epi16(prod3, sum3);

            accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
            accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }

        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    // Handle remaining blocks
    if (la_num > 0) {
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
            __m256i xq8_3 = _mm256_loadu_si256(
                (const __m256i*)(packed_weights + group32_num * 32 * 32 + j * 32)
            );
            __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask);
            __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask);
            __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask);
            xq8_3 = _mm256_and_si256(xq8_3, mask);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(activations + group32_num * 128 * 32 + j * 128 + 96));

            // Apply same weight offset correction
            __m256i ones = _mm256_set1_epi8(1);
            __m256i prod0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            __m256i prod1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            __m256i prod2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            __m256i prod3 = _mm256_maddubs_epi16(xq8_3, yq8_3);
            __m256i sum0 = _mm256_maddubs_epi16(ones, yq8_0);
            __m256i sum1 = _mm256_maddubs_epi16(ones, yq8_1);
            __m256i sum2 = _mm256_maddubs_epi16(ones, yq8_2);
            __m256i sum3 = _mm256_maddubs_epi16(ones, yq8_3);
            xq8_0 = _mm256_sub_epi16(prod0, sum0);
            xq8_1 = _mm256_sub_epi16(prod1, sum1);
            xq8_2 = _mm256_sub_epi16(prod2, sum2);
            xq8_3 = _mm256_sub_epi16(prod3, sum3);

            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }

    return (float)hsum_i32_8(accu);
}

// Python bindings
py::array_t<float> bitnet_gemv(
    py::array_t<uint8_t> packed_weights,
    py::array_t<int8_t> activations,
    float scale
) {
    auto w = packed_weights.unchecked<2>();
    auto a = activations.unchecked<1>();

    int out_features = w.shape(0);
    int packed_in = w.shape(1);
    int in_features = packed_in * 4;

    if (a.shape(0) != in_features) {
        throw std::runtime_error("Activation size mismatch");
    }

    auto result = py::array_t<float>(out_features);
    auto r = result.mutable_unchecked<1>();

    const uint8_t* w_ptr = w.data(0, 0);
    const int8_t* a_ptr = a.data(0);

    #pragma omp parallel for
    for (int i = 0; i < out_features; i++) {
        float dot = bitnet_vec_dot_i2_i8_avx2(
            in_features,
            w_ptr + i * packed_in,
            a_ptr
        );
        r(i) = dot * scale;
    }

    return result;
}

py::array_t<float> bitnet_gemm(
    py::array_t<uint8_t> packed_weights,
    py::array_t<int8_t> activations,
    float scale
) {
    auto w = packed_weights.unchecked<2>();
    auto a = activations.unchecked<2>();

    int out_features = w.shape(0);
    int packed_in = w.shape(1);
    int in_features = packed_in * 4;
    int batch_size = a.shape(0);

    if (a.shape(1) != in_features) {
        throw std::runtime_error("Activation size mismatch");
    }

    auto result = py::array_t<float>({batch_size, out_features});
    auto r = result.mutable_unchecked<2>();

    const uint8_t* w_ptr = w.data(0, 0);
    const int8_t* a_ptr = a.data(0, 0);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < out_features; i++) {
            float dot = bitnet_vec_dot_i2_i8_avx2(
                in_features,
                w_ptr + i * packed_in,
                a_ptr + b * in_features
            );
            r(b, i) = dot * scale;
        }
    }

    return result;
}

std::string get_simd_info() {
    std::string info = "SIMD: ";
#ifdef __AVX2__
    info += "AVX2 ";
#endif
#ifdef __AVX512F__
    info += "AVX512 ";
#endif
    return info;
}

PYBIND11_MODULE(bitnet_native, m) {
    m.doc() = "Native BitNet AVX2 kernels";
    m.def("gemv", &bitnet_gemv, "BitNet GEMV with AVX2");
    m.def("gemm", &bitnet_gemm, "BitNet GEMM with AVX2");
    m.def("simd_info", &get_simd_info, "Get SIMD capabilities");
}
'''

SETUP_PY = '''
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "bitnet_native",
        ["bitnet_native.cpp"],
        extra_compile_args=["-O3", "-mavx2", "-fopenmp", "-ffast-math"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="bitnet_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
'''


@app.function(
    image=native_image,
    cpu=32.0,
    memory=32768,
    timeout=15 * 60,
)
def benchmark_native_kernel() -> str:
    """Build and benchmark native AVX2 kernel."""
    import subprocess
    import os
    import sys
    import time
    import json

    print("=" * 60)
    print("Native AVX2 BitNet Kernel Benchmark")
    print("=" * 60)

    # Create build directory
    build_dir = "/tmp/bitnet_build"
    os.makedirs(build_dir, exist_ok=True)

    # Write source files
    with open(f"{build_dir}/bitnet_native.cpp", "w") as f:
        f.write(KERNEL_CPP)

    with open(f"{build_dir}/setup.py", "w") as f:
        f.write(SETUP_PY)

    # Build the extension
    print("\n[1/4] Building native kernel...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return json.dumps({"error": "Build failed", "stderr": result.stderr})

    print("Build successful!")

    # Add to path and import
    sys.path.insert(0, build_dir)
    import bitnet_native

    print(f"SIMD: {bitnet_native.simd_info()}")

    # Now benchmark
    import numpy as np

    summary = {"simd": bitnet_native.simd_info()}

    # Test dimensions
    configs = [
        {"out": 1024, "in": 1024},
        {"out": 2048, "in": 2048},
        {"out": 4096, "in": 4096},
    ]

    # ===== Python fallback for comparison =====
    def pack_weights_py(weights):
        out_f, in_f = weights.shape
        packed = np.zeros((out_f, in_f // 4), dtype=np.uint8)
        for i in range(4):
            w = (weights[:, i::4].astype(np.int32) + 1).clip(0, 2)
            packed |= (w.astype(np.uint8) << (i * 2))
        return packed

    def unpack_weights_py(packed):
        out_f = packed.shape[0]
        in_f = packed.shape[1] * 4
        weights = np.zeros((out_f, in_f), dtype=np.float32)
        for i in range(4):
            bits = (packed >> (i * 2)) & 0x03
            weights[:, i::4] = bits.astype(np.float32) - 1.0
        return weights

    def gemv_python(packed, activations, scale):
        weights = unpack_weights_py(packed)
        return np.dot(weights, activations.astype(np.float32)) * scale

    def gemm_python(packed, activations, scale):
        weights = unpack_weights_py(packed)
        return np.dot(activations.astype(np.float32), weights.T) * scale

    # ===== GEMV Benchmark =====
    print("\n[2/4] GEMV Benchmark (single token)...")
    gemv_results = []

    for cfg in configs:
        out_f, in_f = cfg["out"], cfg["in"]

        # Create test data
        weights = np.random.randint(-1, 2, (out_f, in_f)).astype(np.float32)
        packed = pack_weights_py(weights)
        activations = np.random.randn(in_f).astype(np.float32)
        act_i8 = np.clip(activations * 10, -128, 127).astype(np.int8)

        # Warmup
        for _ in range(3):
            _ = bitnet_native.gemv(packed, act_i8, 1.0)

        # Native timing
        native_times = []
        for _ in range(20):
            start = time.perf_counter()
            out_native = bitnet_native.gemv(packed, act_i8, 1.0)
            native_times.append(time.perf_counter() - start)

        # Python timing
        python_times = []
        for _ in range(5):
            start = time.perf_counter()
            out_python = gemv_python(packed, act_i8, 1.0)
            python_times.append(time.perf_counter() - start)

        native_avg = np.mean(native_times) * 1000
        python_avg = np.mean(python_times) * 1000
        speedup = python_avg / native_avg

        # Correctness check
        cosine = np.dot(out_native, out_python) / (np.linalg.norm(out_native) * np.linalg.norm(out_python))

        print(f"  {out_f}x{in_f}: native={native_avg:.3f}ms, python={python_avg:.1f}ms, "
              f"speedup={speedup:.1f}x, cosine={cosine:.6f}")

        gemv_results.append({
            "shape": f"{out_f}x{in_f}",
            "native_ms": native_avg,
            "python_ms": python_avg,
            "speedup": speedup,
            "cosine": cosine,
        })

    summary["gemv"] = gemv_results

    # ===== GEMM Benchmark =====
    print("\n[3/4] GEMM Benchmark (batched)...")
    gemm_results = []

    batch_sizes = [1, 8, 32, 64]
    out_f, in_f = 2048, 2048

    weights = np.random.randint(-1, 2, (out_f, in_f)).astype(np.float32)
    packed = pack_weights_py(weights)

    for batch in batch_sizes:
        activations = np.random.randn(batch, in_f).astype(np.float32)
        act_i8 = np.clip(activations * 10, -128, 127).astype(np.int8)

        # Warmup
        for _ in range(3):
            _ = bitnet_native.gemm(packed, act_i8, 1.0)

        # Native timing
        native_times = []
        for _ in range(20):
            start = time.perf_counter()
            out_native = bitnet_native.gemm(packed, act_i8, 1.0)
            native_times.append(time.perf_counter() - start)

        # Python timing
        python_times = []
        for _ in range(5):
            start = time.perf_counter()
            out_python = gemm_python(packed, act_i8, 1.0)
            python_times.append(time.perf_counter() - start)

        native_avg = np.mean(native_times) * 1000
        python_avg = np.mean(python_times) * 1000
        speedup = python_avg / native_avg
        throughput = batch / (np.mean(native_times))

        print(f"  batch={batch:2d}: native={native_avg:.3f}ms, python={python_avg:.1f}ms, "
              f"speedup={speedup:.1f}x, throughput={throughput:.0f} tok/s")

        gemm_results.append({
            "batch": batch,
            "native_ms": native_avg,
            "python_ms": python_avg,
            "speedup": speedup,
            "throughput": throughput,
        })

    summary["gemm"] = gemm_results

    # ===== Layer Benchmark =====
    print("\n[4/4] Full Layer Benchmark...")

    hidden_dim = 2048
    ffn_dim = 5632
    num_layers = 4

    # Create layer weights
    layers = []
    for _ in range(num_layers):
        layer = {
            "q": pack_weights_py(np.random.randint(-1, 2, (hidden_dim, hidden_dim)).astype(np.float32)),
            "k": pack_weights_py(np.random.randint(-1, 2, (hidden_dim, hidden_dim)).astype(np.float32)),
            "v": pack_weights_py(np.random.randint(-1, 2, (hidden_dim, hidden_dim)).astype(np.float32)),
            "o": pack_weights_py(np.random.randint(-1, 2, (hidden_dim, hidden_dim)).astype(np.float32)),
            "gate": pack_weights_py(np.random.randint(-1, 2, (ffn_dim, hidden_dim)).astype(np.float32)),
            "up": pack_weights_py(np.random.randint(-1, 2, (ffn_dim, hidden_dim)).astype(np.float32)),
            "down": pack_weights_py(np.random.randint(-1, 2, (hidden_dim, ffn_dim)).astype(np.float32)),
        }
        layers.append(layer)

    def quantize(x):
        scale = np.abs(x).max() / 127.0
        return np.clip(x / scale, -128, 127).astype(np.int8), scale

    def forward_layer_native(layer, hidden):
        h_i8, scale = quantize(hidden)
        q = bitnet_native.gemv(layer["q"], h_i8, scale)
        k = bitnet_native.gemv(layer["k"], h_i8, scale)
        v = bitnet_native.gemv(layer["v"], h_i8, scale)
        v_i8, s2 = quantize(v)
        attn = bitnet_native.gemv(layer["o"], v_i8, s2)
        a_i8, s3 = quantize(attn)
        gate = bitnet_native.gemv(layer["gate"], a_i8, s3)
        up = bitnet_native.gemv(layer["up"], a_i8, s3)
        ffn = gate * (1 / (1 + np.exp(-gate))) * up
        f_i8, s4 = quantize(ffn)
        return bitnet_native.gemv(layer["down"], f_i8, s4)

    # Benchmark single token through all layers
    hidden = np.random.randn(hidden_dim).astype(np.float32)

    # Warmup
    for layer in layers:
        hidden = forward_layer_native(layer, hidden)

    # Timing
    times = []
    for _ in range(20):
        hidden = np.random.randn(hidden_dim).astype(np.float32)
        start = time.perf_counter()
        for layer in layers:
            hidden = forward_layer_native(layer, hidden)
        times.append(time.perf_counter() - start)

    layer_avg = np.mean(times) * 1000
    tok_per_sec = 1000.0 / layer_avg

    print(f"  {num_layers} layers: {layer_avg:.2f}ms/token, {tok_per_sec:.1f} tok/s")

    summary["layer_benchmark"] = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "time_ms": layer_avg,
        "tok_per_sec": tok_per_sec,
    }

    # ===== Final Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nSIMD: {summary['simd']}")

    print("\nGEMV Speedup (native vs Python):")
    for r in gemv_results:
        print(f"  {r['shape']}: {r['speedup']:.1f}x faster")

    print("\nGEMM Throughput:")
    for r in gemm_results:
        print(f"  batch={r['batch']:2d}: {r['throughput']:.0f} tok/s")

    print(f"\nLayer Performance ({num_layers} layers, {hidden_dim} hidden):")
    print(f"  {tok_per_sec:.1f} tokens/second")

    return json.dumps(summary)


@app.local_entrypoint()
def main():
    print("Running native kernel benchmark...")
    result = benchmark_native_kernel.remote()
    print("\nBenchmark completed!")
