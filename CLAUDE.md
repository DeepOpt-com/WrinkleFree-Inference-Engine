# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine is a serving layer for 1.58-bit quantized LLMs:
- **Inference**: BitNet.cpp (extern/BitNet submodule)
- **Conversion**: HuggingFace → GGUF pipeline
- **Validation**: KV cache behavior testing
- **Deployment**: Via WrinkleFree-Deployer (GCP C3D, H3, RunPod)

## Quick Start

```bash
# Install dependencies
uv sync

# Convert a model
uv run wrinklefree-inference convert --hf-repo microsoft/BitNet-b1.58-2B-4T

# Serve the model
uv run wrinklefree-inference serve -m extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# Test generation
curl http://localhost:8080/completion -d '{"prompt": "Hello", "n_predict": 10}'
```

## Architecture

```
src/wrinklefree_inference/
├── server/bitnet_server.py   # BitNetServer - subprocess manager for BitNet.cpp
├── client/bitnet_client.py   # BitNetClient - HTTP client (sync + async)
├── converter/
│   ├── hf_to_gguf.py        # Download from HF, convert to GGUF
│   └── gguf_converter.py    # Convert trained PyTorch models
└── kv_cache/validator.py    # KV cache validation tests
```

## Key Files

| File | Purpose |
|------|---------|
| `extern/BitNet/run_inference_server.py` | BitNet.cpp server script |
| `extern/BitNet/setup_env.py` | Model download + compilation |
| `../WrinkleFree-Deployer/skypilot/inference/` | Deployment configs (GCP, RunPod) |
| `tests/test_kv_cache.py` | KV cache validation tests |

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
uv run streamlit run demo/serve_streamlit.py --server.port 7860 --server.address 0.0.0.0
```

Demo features:
- Streaming token generation
- Top-k sampling with configurable temperature
- Model info display (layers, hidden size, vocab size)
- ~45s model load time on first access

## sglang-bitnet

The `extern/sglang-bitnet` submodule provides SGLang integration for BitNet models.

```bash
# Initialize submodule
git submodule update --init extern/sglang-bitnet

# Install sglang with bitnet support
cd extern/sglang-bitnet
pip install -e "python[all]"

# Launch server (example)
python -m sglang.launch_server --model-path microsoft/BitNet-b1.58-2B-4T
```

Key sglang-bitnet components:
- `python/sglang/srt/` - SGLang runtime
- `benchmark/` - Benchmarking scripts
- `examples/runtime/` - Usage examples

## Notes

- BitNet.cpp is a submodule at `extern/BitNet/` - run `git submodule update --init` if missing
- sglang-bitnet is at `extern/sglang-bitnet/` - fork with BitNet kernel support
- Models are stored in `extern/BitNet/models/<model-name>/`
- GGUF files are ~500MB for 2B model
- Server requires 120s startup time for model loading
- Use `i2_s` quantization for CPU-optimized inference
- MoE models use llama.cpp's expert packing format
