# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine is a serving layer for 1.58-bit quantized LLMs:
- **Primary Backend**: SGLang-BitNet with native SIMD kernels (AVX2/AVX512)
- **Conversion**: HuggingFace → packed weights (on-the-fly)
- **Validation**: KV cache behavior testing
- **Deployment**: Via WrinkleFree-Deployer (GCP C3D, H3, RunPod)

## Quick Start

```bash
# Install dependencies
uv sync

# Build sgl-kernel (one-time)
cd extern/sglang-bitnet/sgl-kernel
uv pip install -e . --no-build-isolation

# Start SGLang server
./scripts/launch_sglang_bitnet.sh

# Start Streamlit chat UI
uv run streamlit run demo/serve_sglang.py --server.port 7860

# Or test via curl
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Architecture

```
# Primary serving stack
extern/sglang-bitnet/
├── python/sglang/srt/models/bitnet.py    # BitNet model with weight packing
├── sgl-kernel/                            # Native SIMD kernels (AVX2/AVX512)
│   ├── csrc/bitnet/                       # C++ kernel implementations
│   └── python/sgl_kernel/quantization/    # Python bindings

# Demo and scripts
demo/
├── serve_sglang.py                        # Primary: SGLang streaming frontend
└── archive/                               # Deprecated implementations

scripts/
└── launch_sglang_bitnet.sh               # Server launch script

# Legacy (reference only)
extern/BitNet/                             # Microsoft BitNet.cpp (reference)
```

## Key Files

| File | Purpose |
|------|---------|
| `demo/serve_sglang.py` | Primary Streamlit chat UI |
| `scripts/launch_sglang_bitnet.sh` | SGLang server launch script |
| `extern/sglang-bitnet/python/sglang/srt/models/bitnet.py` | BitNet model with weight packing |
| `extern/sglang-bitnet/sgl-kernel/` | Native SIMD kernels |
| `extern/BitNet/` | Reference implementation (do not serve) |

## Common Tasks

### Convert a model
```bash
uv run python scripts/convert.py --hf-repo microsoft/BitNet-b1.58-2B-4T -q i2_s
```

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
uv run python scripts/validate_kv_cache.py --url http://localhost:8080
```

## Testing

```bash
# Unit tests only (no server required)
uv run pytest tests/test_conversion.py -v

# Integration tests (requires running server)
INFERENCE_URL=http://localhost:8080 uv run pytest tests/ -v -m integration

# KV cache tests
INFERENCE_URL=http://localhost:8080 uv run pytest tests/test_kv_cache.py -v
```

## Configuration

- **Quantization types**: `i2_s` (CPU optimized), `tl2` (AVX512)
- **Context size**: Default 4096, configurable via `-c` flag
- **Continuous batching**: Enabled by default (`-cb` flag)

## MoE Support

The inference engine supports MoE models via llama.cpp's Mixtral-style expert tensors.

### MoE Components
- `src/wrinklefree_inference/moe/router.py` - TopKRouter, IdentityRouter
- `src/wrinklefree_inference/moe/expert.py` - BitNetMoEFFN with K-of-N routing
- `src/wrinklefree_inference/moe/fake_moe.py` - Convert dense → MoE for testing

### GGUF MoE Tensors
MoE models use these tensor patterns in GGUF:
- `blk.{n}.ffn_gate_inp.weight` - Router logits
- `blk.{n}.ffn_gate_exps.weight` - Packed expert gates (3D tensor)
- `blk.{n}.ffn_up_exps.weight` - Packed expert up projections
- `blk.{n}.ffn_down_exps.weight` - Packed expert down projections

## Integration with WrinkleFree-Eval

The inference engine can be used by WrinkleFree-Eval for optimized evaluation:

```python
# In WrinkleFree-Eval, set BITNET_PATH to use optimized inference
export BITNET_PATH=/path/to/WrinkleFree-Inference-Engine/extern/BitNet

# Or use the inference server
uv run wrinklefree-inference serve -m model.gguf -c 4096 --port 8080

# Eval uses HTTP client
INFERENCE_URL=http://localhost:8080 uv run python scripts/run_eval.py
```

## Streamlit Chat Interface

Run the interactive chat demo:
```bash
# Start SGLang server first
./scripts/launch_sglang_bitnet.sh

# Then start Streamlit
uv run streamlit run demo/serve_sglang.py --server.port 7860 --server.address 0.0.0.0
```

Demo features:
- SSE streaming token generation
- Connects to SGLang backend (OpenAI-compatible API)
- Token/sec stats display
- Server health monitoring

## sglang-bitnet and Native Kernels

The `extern/sglang-bitnet` submodule provides SGLang with native SIMD kernels for BitNet.

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

### Launch Server

```bash
# Use the launch script (recommended)
./scripts/launch_sglang_bitnet.sh

# Or manually
python -m sglang.launch_server \
    --model-path microsoft/bitnet-b1.58-2B-4T \
    --port 30000 --host 0.0.0.0 --device cpu
```

### Kernel Performance

Native SIMD kernels (AVX2/AVX512) provide significant speedups:
- GEMV (batch=1): ~10x faster than torch
- Large dims: ~47x faster

```bash
uv run python scripts/benchmark_kernels.py
```

## Notes

- **Primary backend**: sglang-bitnet at `extern/sglang-bitnet/`
- **Reference only**: BitNet.cpp at `extern/BitNet/` (do not serve)
- HuggingFace models are automatically packed on-the-fly during loading
- Server uses OpenAI-compatible API (`/v1/chat/completions`)
- MoE models use llama.cpp's expert packing format
