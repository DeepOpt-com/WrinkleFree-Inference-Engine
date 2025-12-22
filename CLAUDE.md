# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WrinkleFree Inference Engine is a serving layer for 1.58-bit quantized LLMs:
- **Inference**: BitNet.cpp (extern/BitNet submodule)
- **Conversion**: HuggingFace → GGUF pipeline
- **Validation**: KV cache behavior testing
- **Deployment**: RunPod via SkyPilot

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
| `skypilot/runpod_cpu.yaml` | RunPod deployment config |
| `tests/test_kv_cache.py` | KV cache validation tests |

## Common Tasks

### Convert a model
```bash
uv run python scripts/convert.py --hf-repo microsoft/BitNet-b1.58-2B-4T -q i2_s
```

### Deploy to RunPod
```bash
cd ../WrinkleFree-Deployer
sky launch ../WrinkleFree-Inference-Engine/skypilot/runpod_cpu.yaml -y --cluster ie-test
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

## Notes

- BitNet.cpp is a submodule at `extern/BitNet/` - run `git submodule update --init` if missing
- Models are stored in `extern/BitNet/models/<model-name>/`
- GGUF files are ~500MB for 2B model
- Server requires 120s startup time for model loading
