# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine is a serving layer for 1.58-bit quantized LLMs:
- **Primary Backend**: SGLang-BitNet with native SIMD kernels (AVX2/AVX512)
- **Frontend**: Streamlit chat UI with SSE streaming
- **Deployment**: Via WrinkleFree-Deployer (GCP C3D, H3, RunPod)

## Quick Start

```bash
# Install dependencies
uv sync

# Build sgl-kernel (one-time)
cd extern/sglang-bitnet/sgl-kernel
uv pip install -e . --no-build-isolation
cd ../../..

# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit chat UI
uv run streamlit run demo/serve_sglang.py --server.port 7860

# Test via curl
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Architecture

```
demo/
└── serve_sglang.py                        # Streamlit chat frontend

scripts/
├── launch_sglang_bitnet.sh               # Server launch script
├── benchmark_kernels.py                   # Kernel performance testing
├── validate_kv_cache.py                   # KV cache validation
└── test_repacking.py                      # Weight repacking tests

src/wrinklefree_inference/
├── sglang_backend/                        # SGLang integration utilities
├── kernels/                               # Kernel wrappers
├── kv_cache/                              # KV cache utilities
├── client/                                # API client
└── moe/                                   # MoE support

extern/
├── sglang-bitnet/                         # SGLang with native BitNet kernels
│   ├── python/sglang/srt/models/bitnet.py # BitNet model implementation
│   └── sgl-kernel/                        # Native SIMD kernels (AVX2/AVX512)
└── BitNet/                                # Microsoft BitNet.cpp (reference only)

legacy/                                    # Archived code (see legacy/README.md)
```

## Key Files

| File | Purpose |
|------|---------|
| `demo/serve_sglang.py` | Primary Streamlit chat UI |
| `scripts/launch_sglang_bitnet.sh` | SGLang server launch script |
| `extern/sglang-bitnet/python/sglang/srt/models/bitnet.py` | BitNet model with weight packing |
| `extern/sglang-bitnet/sgl-kernel/` | Native SIMD kernels |

## Common Tasks

### Deploy to cloud
```bash
cd ../WrinkleFree-Deployer
# GCP C3D (production)
sky launch skypilot/inference/gcp_c3d.yaml -y --cluster ie-c3d
# RunPod (development)
sky launch skypilot/inference/runpod_cpu.yaml -y --cluster ie-runpod
```

### Run KV cache validation
```bash
uv run python scripts/validate_kv_cache.py --url http://localhost:30000
```

### Benchmark kernels
```bash
uv run python scripts/benchmark_kernels.py
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Integration tests (requires running server)
INFERENCE_URL=http://localhost:30000 uv run pytest tests/ -v -m integration
```

## sglang-bitnet Setup

**IMPORTANT**: We use a custom fork of SGLang at `extern/sglang-bitnet/`, NOT the upstream sglang package.
This fork includes native SIMD kernels (AVX2/AVX512) for BitNet inference. Do not install sglang from PyPI.

### Building sgl-kernel (CPU-only)

```bash
# Initialize submodule
git submodule update --init extern/sglang-bitnet

# Install CPU-only torch first
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cpu

# Build sgl-kernel
cd extern/sglang-bitnet/sgl-kernel
uv pip install scikit-build-core cmake ninja
uv pip install -e . --no-build-isolation
```

### Verify BitNet Kernels

```python
from sgl_kernel.quantization import bitnet_check_kernel_available
print(bitnet_check_kernel_available())  # Should be True
```

## Notes

- **Custom SGLang fork**: We use `extern/sglang-bitnet/` (custom fork with BitNet kernels), NOT upstream sglang
- **Reference only**: BitNet.cpp at `extern/BitNet/` (do not serve)
- HuggingFace models are automatically packed on-the-fly during loading
- Server uses OpenAI-compatible API (`/v1/chat/completions`)
- Legacy code (BitNet.cpp integration, CLI, benchmarks) is in `legacy/`
