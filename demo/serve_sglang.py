#!/usr/bin/env python3
"""Streamlit chat interface using SGLang-BitNet backend.

Primary serving interface for BitNet models with native SIMD kernels.
Connects to SGLang server via OpenAI-compatible API with SSE streaming.

Run:
  1. Start SGLang server: ./scripts/launch_sglang_bitnet.sh
  2. Start Streamlit: uv run streamlit run demo/serve_sglang.py --server.port 7860 --server.address 0.0.0.0
"""

import json
import os
import time
from typing import Generator

import requests
import streamlit as st

# SGLang server URL
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")


def check_server() -> dict:
    """Check if SGLang server is running and get model info."""
    try:
        resp = requests.get(f"{SGLANG_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return {"status": "ok", "model": models[0].get("id", "unknown")}
        return {"status": "error", "message": "No models loaded"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": f"Cannot connect to {SGLANG_URL}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_streaming(
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """Stream tokens from SGLang server using SSE."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        with requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")

                # Skip non-data lines
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]  # Remove "data: " prefix

                # Check for stream end
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as e:
        yield f"\n\n**Error:** {e}"


def generate_sync(
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, dict]:
    """Generate response synchronously (non-streaming)."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()
    resp = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    elapsed = time.perf_counter() - start

    data = resp.json()
    choices = data.get("choices", [])
    content = choices[0]["message"]["content"] if choices else ""

    usage = data.get("usage", {})
    stats = {
        "tokens": usage.get("completion_tokens", len(content.split())),
        "elapsed": elapsed,
        "tok_per_sec": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
    }

    return content, stats


# Page config
st.set_page_config(
    page_title="BitNet Chat (SGLang)",
    page_icon="",
    layout="wide",
)

st.title("BitNet-b1.58-2B-4T Chat")
st.caption("SGLang backend | Native SIMD kernels | Streaming generation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max tokens", 32, 512, 256)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
    use_streaming = st.checkbox("Streaming", value=True)

    st.divider()

    # Server status
    server_info = check_server()
    if server_info["status"] == "ok":
        st.success(f"Connected to SGLang")
        st.caption(f"Model: {server_info['model']}")
        st.caption(f"URL: {SGLANG_URL}")
    else:
        st.error(f"Server Error: {server_info['message']}")
        st.info(
            "Start server with:\n```bash\n./scripts/launch_sglang_bitnet.sh\n```"
        )

    st.divider()

    st.caption("Backend: SGLang-BitNet")
    st.caption("Kernels: AVX2/AVX512 SIMD")
    st.caption("Quantization: 1.58-bit ternary")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for API (include system prompt)
    api_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        *st.session_state.messages,
    ]

    # Generate response
    with st.chat_message("assistant"):
        if use_streaming:
            output_placeholder = st.empty()
            stats_placeholder = st.empty()
            full_response = ""
            token_count = 0
            start_time = time.perf_counter()

            for token in generate_streaming(api_messages, max_tokens, temperature):
                full_response += token
                token_count += 1
                output_placeholder.markdown(full_response + "")

            # Final update
            output_placeholder.markdown(full_response)
            elapsed = time.perf_counter() - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0
            stats_placeholder.caption(
                f"Generated {token_count} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)"
            )
        else:
            with st.spinner("Generating..."):
                full_response, stats = generate_sync(api_messages, max_tokens, temperature)
            st.markdown(full_response)
            st.caption(
                f"Generated {stats['tokens']} tokens in {stats['elapsed']:.1f}s "
                f"({stats['tok_per_sec']:.1f} tok/s)"
            )

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
