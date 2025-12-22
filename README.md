# WrinkleFree Inference Engine

BitNet model inference engine with KV caching optimization. Serves 1.58-bit quantized models using BitNet.cpp for efficient CPU inference.

## Features

- **HuggingFace to GGUF conversion** - Download and convert models automatically
- **BitNet.cpp serving** - Optimized CPU inference with AVX512 support
- **KV cache validation** - Verify caching behavior and performance
- **RunPod deployment** - Ready-to-use SkyPilot configurations

## Quick Start

```bash
# Install dependencies
uv sync

# Convert a model from HuggingFace
uv run wrinklefree-inference convert --hf-repo microsoft/BitNet-b1.58-2B-4T

# Start the inference server
uv run wrinklefree-inference serve --model extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# Generate text
uv run wrinklefree-inference generate --prompt "The future of AI is"

# Validate KV cache
uv run wrinklefree-inference validate
```

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/DeepOpt-com/WrinkleFree-Inference-Engine.git
cd WrinkleFree-Inference-Engine

# Install dependencies
uv sync

# Setup BitNet (compiles inference engine)
uv run python extern/BitNet/setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T -q i2_s
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `convert` | Convert HuggingFace model to GGUF |
| `serve` | Start inference server |
| `generate` | Generate text from prompt |
| `validate` | Run KV cache validation |
| `list-models` | List available GGUF models |
| `benchmark-cost` | Run cost benchmarking |
| `naive-convert` | Naive ternary conversion (for benchmarking) |
| `chat` | Launch Streamlit chat interface |

## RunPod Deployment

```bash
# From WrinkleFree-Deployer directory
cd ../WrinkleFree-Deployer
source .venv/bin/activate

# Launch CPU pod
sky launch ../WrinkleFree-Inference-Engine/skypilot/runpod_cpu.yaml -y --cluster ie-cpu

# Get endpoint
ENDPOINT=$(sky status ie-cpu --endpoint 8080)
curl $ENDPOINT/health

# Run tests
INFERENCE_URL=$ENDPOINT uv run pytest tests/ -v -m integration

# Teardown
sky down ie-cpu -y
```

## API Endpoints

The server exposes these endpoints (OpenAI-compatible):

- `GET /health` - Health check
- `POST /completion` - Text completion
- `POST /tokenize` - Tokenize text
- `POST /detokenize` - Detokenize tokens

### Example Request

```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "n_predict": 50,
    "temperature": 0.7
  }'
```

## KV Cache Validation

The engine includes tools to validate KV caching behavior:

```bash
# Run validation suite
uv run wrinklefree-inference validate --url http://localhost:8080

# Run specific tests
uv run pytest tests/test_kv_cache.py -v
```

Validation checks:
- **Prefix caching** - Same prefix should speed up subsequent requests
- **Context limits** - Graceful handling at context window boundary
- **Continuous batching** - Concurrent requests should succeed

## Cost Benchmarking

Benchmark inference cost efficiency across different hardware configurations:

```bash
# Install benchmark dependencies
uv sync --extra benchmark

# Run local benchmark (requires running server)
uv run wrinklefree-inference benchmark-cost \
    --url http://localhost:8080 \
    --hardware a40 \
    --model bitnet-2b-4t

# Run cloud benchmark on RunPod
sky launch skypilot/benchmark/runpod_a40_benchmark.yaml -y
sky launch skypilot/benchmark/runpod_cpu_64core.yaml -y
```

For naive ternary conversion (low quality, for cost analysis only):

```bash
uv sync --extra convert
uv run wrinklefree-inference naive-convert \
    --model-id meta-llama/Llama-3.1-70B \
    --estimate-only
```

See [docs/cost-benchmark.md](docs/cost-benchmark.md) for detailed methodology.

## Project Structure

```
WrinkleFree-Inference-Engine/
├── src/wrinklefree_inference/
│   ├── server/          # BitNet server wrapper
│   ├── client/          # Python client (sync + async)
│   ├── converter/       # HF to GGUF conversion
│   └── kv_cache/        # KV cache validation
├── benchmark/           # Cost benchmarking module
├── scripts/             # CLI scripts
├── configs/             # Hydra configs
├── skypilot/            # RunPod deployment configs
├── tests/               # pytest test suite
└── extern/BitNet/       # BitNet.cpp submodule
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check src/

# Run type checking
uv run mypy src/
```

## Dependencies

- **BitNet.cpp** - Microsoft's optimized 1.58-bit inference engine
- **WrinkleFree-Deployer** - Cloud deployment infrastructure (optional)

## License

MIT
