#!/usr/bin/env python3
"""Benchmark BitNet kernels."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.kernels import benchmark_kernels, NATIVE_KERNELS_AVAILABLE

print(f"Native kernels available: {NATIVE_KERNELS_AVAILABLE}")

if NATIVE_KERNELS_AVAILABLE:
    print("\nGEMV (batch=1):")
    results = benchmark_kernels(in_features=2048, out_features=2048, batch_size=1)
    print(f"  Native: {results['native_ms']:.3f}ms, Torch: {results['torch_ms']:.3f}ms, Speedup: {results['speedup']:.2f}x")

    print("\nGEMM (batch=8):")
    results = benchmark_kernels(in_features=2048, out_features=2048, batch_size=8)
    print(f"  Native: {results['native_ms']:.3f}ms, Torch: {results['torch_ms']:.3f}ms, Speedup: {results['speedup']:.2f}x")

    print("\nGEMM (batch=32):")
    results = benchmark_kernels(in_features=2048, out_features=2048, batch_size=32)
    print(f"  Native: {results['native_ms']:.3f}ms, Torch: {results['torch_ms']:.3f}ms, Speedup: {results['speedup']:.2f}x")

    print("\nLarger dims (4096x4096):")
    results = benchmark_kernels(in_features=4096, out_features=4096, batch_size=1)
    print(f"  Native: {results['native_ms']:.3f}ms, Torch: {results['torch_ms']:.3f}ms, Speedup: {results['speedup']:.2f}x")
else:
    print("Native kernels not available!")
