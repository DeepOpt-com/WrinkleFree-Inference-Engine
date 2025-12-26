#!/bin/bash
# Launch SGLang-BitNet server + Streamlit chat UI
#
# Usage:
#   ./scripts/serve.sh              # Start both server and UI
#   ./scripts/serve.sh --server     # Start only server
#   ./scripts/serve.sh --ui         # Start only UI (assumes server running)
#
# Environment variables:
#   SGLANG_PORT - SGLang server port (default: 30000)
#   STREAMLIT_PORT - Streamlit UI port (default: 7860)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
MODEL="${SGLANG_MODEL:-microsoft/bitnet-b1.58-2B-4T}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-7860}"
HOST="0.0.0.0"

# Parse args
START_SERVER=true
START_UI=true
if [[ "${1:-}" == "--server" ]]; then
    START_UI=false
elif [[ "${1:-}" == "--ui" ]]; then
    START_SERVER=false
fi

cleanup() {
    echo ""
    echo "Shutting down..."
    pkill -f "sglang.launch_server.*$SGLANG_PORT" 2>/dev/null || true
    pkill -f "streamlit.*serve_sglang.py" 2>/dev/null || true
}
trap cleanup EXIT

# Start SGLang server
if $START_SERVER; then
    echo "=== Starting SGLang-BitNet Server ==="
    echo "Model: $MODEL"
    echo "Port:  $SGLANG_PORT"

    # Kill any existing server on this port
    pkill -f "sglang.launch_server.*$SGLANG_PORT" 2>/dev/null || true
    sleep 1

    .venv/bin/python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$SGLANG_PORT" \
        --host "$HOST" \
        --device cpu \
        --dtype bfloat16 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..120}; do
        if curl -s "http://127.0.0.1:$SGLANG_PORT/v1/models" >/dev/null 2>&1; then
            echo "Server ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Error: Server process died"
            exit 1
        fi
        sleep 1
    done
fi

# Start Streamlit UI
if $START_UI; then
    echo ""
    echo "=== Starting Streamlit UI ==="
    echo "URL: http://$HOST:$STREAMLIT_PORT"

    export SGLANG_URL="http://127.0.0.1:$SGLANG_PORT"

    .venv/bin/streamlit run demo/serve_sglang.py \
        --server.port "$STREAMLIT_PORT" \
        --server.address "$HOST" \
        --server.headless true
else
    # Keep server running
    echo ""
    echo "Server running at http://$HOST:$SGLANG_PORT"
    echo "Press Ctrl+C to stop"
    wait
fi
