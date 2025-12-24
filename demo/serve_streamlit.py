#!/usr/bin/env python3
"""Streamlit chat interface for BitNet-b1.58-2B-4T with streaming.

Run: uv run streamlit run demo/serve_streamlit.py --server.port 7860
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import numpy as np
import streamlit as st
from transformers import AutoTokenizer


@st.cache_resource
def load_model():
    """Load model and tokenizer (cached)."""
    from wrinklefree_inference.models.bitnet import load_model as load_bitnet

    with st.spinner("Loading tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BitNet-b1.58-2B-4T",
            trust_remote_code=True,
        )

    with st.spinner("Loading BitNet-2B model (this takes ~45s)..."):
        model = load_bitnet("microsoft/BitNet-b1.58-2B-4T")

    return model, tokenizer


def generate_streaming(model, tokenizer, prompt: str, max_tokens: int, temperature: float):
    """Generate tokens with streaming output."""
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    input_ids = tokenizer.encode(formatted, return_tensors="np")[0]

    # Track generation
    tokens = list(input_ids)
    generated_text = ""
    total_time = 0
    num_generated = 0

    # Streaming placeholder
    output_placeholder = st.empty()
    stats_placeholder = st.empty()

    for _ in range(max_tokens):
        start = time.perf_counter()

        # Forward pass on last token
        last_token = np.array([tokens[-1]])
        logits = model.forward(last_token)

        if logits.ndim > 1:
            logits = logits[-1]

        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Sample
        logits = logits / max(temperature, 0.1)
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()

        # Top-k sampling
        top_k = 40
        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()

        next_token = int(np.random.choice(top_indices, p=top_probs))
        tokens.append(next_token)
        num_generated += 1

        # Decode incrementally
        new_text = tokenizer.decode([next_token], skip_special_tokens=False)

        # Check for EOS
        if next_token == tokenizer.eos_token_id or "<|eot_id|>" in new_text:
            break

        generated_text += new_text

        # Update display
        output_placeholder.markdown(f"**Assistant:** {generated_text}▌")
        tok_per_sec = num_generated / total_time if total_time > 0 else 0
        stats_placeholder.caption(f"Generated {num_generated} tokens @ {tok_per_sec:.1f} tok/s")

    # Final update (remove cursor)
    output_placeholder.markdown(f"**Assistant:** {generated_text}")
    tok_per_sec = num_generated / total_time if total_time > 0 else 0
    stats_placeholder.caption(f"Generated {num_generated} tokens in {total_time:.2f}s ({tok_per_sec:.1f} tok/s)")

    return generated_text


def main():
    st.set_page_config(
        page_title="BitNet-2B Chat",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 BitNet-b1.58-2B-4T Chat")
    st.caption("1.58-bit ternary weights | Native AVX2 kernels | Streaming generation")

    # Load model
    model, tokenizer = load_model()

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        max_tokens = st.slider("Max tokens", 10, 500, 100, step=10)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)

        st.divider()
        st.header("Model Info")
        st.write(f"**Layers:** {model.config.num_hidden_layers}")
        st.write(f"**Hidden:** {model.config.hidden_size}")
        st.write(f"**Vocab:** {model.config.vocab_size:,}")
        st.write(f"**Threads:** {model.kernel.num_threads()}")

        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = generate_streaming(model, tokenizer, prompt, max_tokens, temperature)

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
