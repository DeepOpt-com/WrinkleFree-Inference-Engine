#!/bin/bash
# Launch the Rust sgl-model-gateway with native BitNet inference.
#
# This bypasses the Python gRPC server and calls C++ SIMD kernels directly
# from Rust, eliminating ~49ms of overhead per token.
#
# Usage:
#   ./scripts/launch_rust_gateway.sh [--native] [--port PORT]
#
# Options:
#   --native       Use native C++ inference (default if available)
#   --grpc         Use Python gRPC backend (fallback)
#   --port PORT    Server port (default: 30000)
#   --model PATH   Model path (default: microsoft/bitnet-b1.58-2B-4T)
#
# Performance comparison:
#   - Native inference: ~26 tok/s (matches BitNet.cpp)
#   - gRPC to Python:   ~19 tok/s (49ms overhead)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GATEWAY_DIR="${PROJECT_DIR}/extern/sglang-bitnet/sgl-model-gateway"

# Default configuration
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-microsoft/bitnet-b1.58-2B-4T}"
BACKEND="${BACKEND:-native}"  # native or grpc

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --native)
            BACKEND="native"
            shift
            ;;
        --grpc)
            BACKEND="grpc"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Rust sgl-model-gateway ==="
echo "Backend: $BACKEND"
echo "Model:   $MODEL"
echo "Host:    $HOST"
echo "Port:    $PORT"
echo ""

cd "$GATEWAY_DIR"

# Build with appropriate features
FEATURES=""
if [[ "$BACKEND" == "native" ]]; then
    FEATURES="--features native-inference"
    echo "Building with native inference (C++ SIMD kernels)..."
else
    echo "Building with gRPC backend (Python scheduler)..."
fi

# Build in release mode
cargo build --release $FEATURES

# Set library path for llama.cpp shared libraries
export LD_LIBRARY_PATH="${PROJECT_DIR}/extern/BitNet/build/3rdparty/llama.cpp/src:${PROJECT_DIR}/extern/BitNet/build/3rdparty/llama.cpp/ggml/src:${LD_LIBRARY_PATH:-}"

# Run the gateway
echo ""
echo "Starting gateway..."
echo "API endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo ""

if [[ "$BACKEND" == "native" ]]; then
    # Native mode: use the native_server binary with C++ inference
    # Find the GGUF model file
    GGUF_PATH=""
    if [[ -f "$MODEL" ]]; then
        GGUF_PATH="$MODEL"
    elif [[ -d "$MODEL" ]]; then
        # Look for GGUF in directory
        GGUF_PATH=$(find "$MODEL" -name "*.gguf" | head -1)
    else
        # Check in BitNet models directory
        GGUF_PATH="${PROJECT_DIR}/extern/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    fi

    if [[ ! -f "$GGUF_PATH" ]]; then
        echo "Error: Could not find GGUF model file"
        echo "Looked for: $MODEL"
        echo "Download with: huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir extern/BitNet/models/BitNet-b1.58-2B-4T"
        exit 1
    fi

    echo "Model file: $GGUF_PATH"
    exec cargo run --release $FEATURES --bin native_server -- \
        --host "$HOST" \
        --port "$PORT" \
        --model-path "$GGUF_PATH"
else
    # gRPC mode: connect to Python scheduler
    # User needs to start Python scheduler separately
    echo "Note: Start the Python scheduler first with:"
    echo "  ./scripts/launch_sglang_bitnet.sh"
    echo ""
    exec cargo run --release -- \
        --host "$HOST" \
        --port "$PORT" \
        --grpc-endpoint "http://localhost:50051"
fi
