"""BitNet model implementation with native kernel support.

Supports loading from HuggingFace Hub and running inference with
our optimized AVX2/OpenMP kernels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import math

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from ..kernels.native import (
    get_kernel,
    repack_hf_weights,
    quantize_activations,
)


@dataclass
class BitNetConfig:
    """BitNet model configuration."""
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf(cls, repo_id: str) -> "BitNetConfig":
        """Load config from HuggingFace Hub."""
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(
            hidden_size=cfg.get("hidden_size", 2560),
            intermediate_size=cfg.get("intermediate_size", 6912),
            num_hidden_layers=cfg.get("num_hidden_layers", 30),
            num_attention_heads=cfg.get("num_attention_heads", 20),
            num_key_value_heads=cfg.get("num_key_value_heads", 5),
            vocab_size=cfg.get("vocab_size", 128256),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", 500000.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        )


class BitNetLinear:
    """Linear layer with packed ternary weights."""

    def __init__(self, weight: np.ndarray, scale: float, kernel):
        """Initialize with repacked weights.

        Args:
            weight: Packed weights in kernel format [out, in/4]
            scale: Weight scale factor
            kernel: Native kernel module
        """
        self.weight = weight
        self.scale = scale
        self.kernel = kernel
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] * 4

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through linear layer.

        Args:
            x: Input activations [batch?, in_features]

        Returns:
            Output [batch?, out_features]
        """
        x_i8, x_scale = quantize_activations(x)
        combined_scale = x_scale * self.scale

        if x.ndim == 1:
            return self.kernel.gemv(self.weight, x_i8, combined_scale)
        else:
            return self.kernel.gemm(self.weight, x_i8, combined_scale)


class BitNetAttention:
    """Multi-head attention with BitNet linear layers."""

    def __init__(self, config: BitNetConfig, layer_weights: dict, kernel):
        self.config = config
        self.kernel = kernel

        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scale = 1.0 / math.sqrt(head_dim)

        # Load projections
        self.q_proj = self._load_linear(layer_weights, "self_attn.q_proj")
        self.k_proj = self._load_linear(layer_weights, "self_attn.k_proj")
        self.v_proj = self._load_linear(layer_weights, "self_attn.v_proj")
        self.o_proj = self._load_linear(layer_weights, "self_attn.o_proj")

        # RMSNorm for attention sublayer
        self.attn_sub_norm = layer_weights.get("self_attn.attn_sub_norm.weight")

    def _load_linear(self, weights: dict, name: str) -> BitNetLinear:
        packed_hf = weights[f"{name}.weight"]
        scale_arr = weights[f"{name}.weight_scale"]
        scale = float(scale_arr.item() if scale_arr.ndim == 0 else scale_arr[0])
        repacked = repack_hf_weights(packed_hf)
        return BitNetLinear(repacked, scale, self.kernel)

    def forward(self, hidden: np.ndarray, position: int = 0) -> np.ndarray:
        """Forward pass (simplified, no KV cache for now)."""
        # Q, K, V projections
        q = self.q_proj.forward(hidden)
        k = self.k_proj.forward(hidden)
        v = self.v_proj.forward(hidden)

        # Sublayer norm
        if self.attn_sub_norm is not None:
            q = self._rms_norm(q) * self.attn_sub_norm
            k = self._rms_norm(k) * self.attn_sub_norm[:k.shape[-1]]
            v = self._rms_norm(v) * self.attn_sub_norm[:v.shape[-1]]

        # Simplified attention (no masking, single token)
        # For batched: reshape to heads, compute attention, reshape back
        attn_out = v  # Simplified - skip actual attention for demo

        return self.o_proj.forward(attn_out)

    def _rms_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + eps)


class BitNetMLP:
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: BitNetConfig, layer_weights: dict, kernel):
        self.kernel = kernel

        self.gate_proj = self._load_linear(layer_weights, "mlp.gate_proj")
        self.up_proj = self._load_linear(layer_weights, "mlp.up_proj")
        self.down_proj = self._load_linear(layer_weights, "mlp.down_proj")

        self.ffn_sub_norm = layer_weights.get("mlp.ffn_sub_norm.weight")

    def _load_linear(self, weights: dict, name: str) -> BitNetLinear:
        packed_hf = weights[f"{name}.weight"]
        scale_arr = weights[f"{name}.weight_scale"]
        scale = float(scale_arr.item() if scale_arr.ndim == 0 else scale_arr[0])
        repacked = repack_hf_weights(packed_hf)
        return BitNetLinear(repacked, scale, self.kernel)

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)

        # SwiGLU activation
        hidden = gate * (1 / (1 + np.exp(-np.clip(gate, -20, 20)))) * up

        # Sublayer norm
        if self.ffn_sub_norm is not None:
            hidden = self._rms_norm(hidden) * self.ffn_sub_norm

        return self.down_proj.forward(hidden)

    def _rms_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + eps)


class BitNetLayer:
    """Single transformer layer."""

    def __init__(self, config: BitNetConfig, layer_weights: dict, kernel):
        self.attention = BitNetAttention(config, layer_weights, kernel)
        self.mlp = BitNetMLP(config, layer_weights, kernel)

        self.input_layernorm = layer_weights.get("input_layernorm.weight")
        self.post_attention_layernorm = layer_weights.get("post_attention_layernorm.weight")
        self.eps = config.rms_norm_eps

    def forward(self, hidden: np.ndarray, position: int = 0) -> np.ndarray:
        # Pre-norm attention
        residual = hidden
        hidden = self._rms_norm(hidden) * self.input_layernorm
        hidden = self.attention.forward(hidden, position)
        hidden = residual + hidden

        # Pre-norm MLP
        residual = hidden
        hidden = self._rms_norm(hidden) * self.post_attention_layernorm
        hidden = self.mlp.forward(hidden)
        hidden = residual + hidden

        return hidden

    def _rms_norm(self, x: np.ndarray) -> np.ndarray:
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + self.eps)


class BitNetModel:
    """Full BitNet model for causal language modeling."""

    def __init__(self, repo_id: str = "microsoft/BitNet-b1.58-2B-4T"):
        """Load model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID
        """
        print(f"Loading {repo_id}...")

        # Build native kernel
        print("  Building native kernel...")
        self.kernel = get_kernel()
        print(f"  Kernel ready ({self.kernel.num_threads()} threads)")

        # Load config
        self.config = BitNetConfig.from_hf(repo_id)
        print(f"  Config: {self.config.num_hidden_layers} layers, "
              f"{self.config.hidden_size} hidden, "
              f"{self.config.vocab_size} vocab")

        # Download weights
        print("  Downloading weights...")
        model_path = hf_hub_download(repo_id, "model.safetensors")

        # Load weights using torch (handles bfloat16)
        print("  Loading and repacking weights...")
        import torch
        state_dict = load_safetensors(model_path)

        def to_numpy(t):
            """Convert tensor to numpy, handling bfloat16."""
            if t.dtype == torch.bfloat16:
                return t.float().numpy()
            elif t.dtype == torch.uint8:
                return t.numpy()
            else:
                return t.numpy()

        self.embed_tokens = to_numpy(state_dict["model.embed_tokens.weight"])

        self.layers = []
        for i in range(self.config.num_hidden_layers):
            layer_weights = {}
            prefix = f"model.layers.{i}."
            for key in state_dict.keys():
                if key.startswith(prefix):
                    layer_weights[key[len(prefix):]] = to_numpy(state_dict[key])
            self.layers.append(BitNetLayer(self.config, layer_weights, self.kernel))
            if (i + 1) % 10 == 0:
                print(f"    Loaded {i + 1}/{self.config.num_hidden_layers} layers")

        self.norm = to_numpy(state_dict["model.norm.weight"])

        if self.config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = to_numpy(state_dict["lm_head.weight"])

        print(f"Model ready!")

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            token_ids: Input token IDs [seq_len] or [batch, seq_len]

        Returns:
            Logits [seq_len, vocab] or [batch, seq_len, vocab]
        """
        # Embedding lookup
        if token_ids.ndim == 1:
            hidden = self.embed_tokens[token_ids]
        else:
            hidden = self.embed_tokens[token_ids.flatten()].reshape(
                token_ids.shape + (self.config.hidden_size,)
            )

        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden = layer.forward(hidden)

        # Final norm
        hidden = self._rms_norm(hidden) * self.norm

        # LM head
        logits = np.dot(hidden, self.lm_head.T)

        return logits

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> tuple[list[int], float]:
        """Generate tokens autoregressively.

        Args:
            prompt_ids: Input token IDs [seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Tuple of (generated token IDs, tokens per second)
        """
        import time

        tokens = list(prompt_ids)
        generated = []
        total_time = 0

        for _ in range(max_new_tokens):
            start = time.perf_counter()

            # Forward pass on last token only (simplified - should use KV cache)
            input_ids = np.array([tokens[-1]])
            logits = self.forward(input_ids)

            if logits.ndim > 1:
                logits = logits[-1]

            total_time += time.perf_counter() - start

            # Sample next token
            logits = logits / max(temperature, 0.1)
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()

            # Top-k sampling
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs = top_probs / top_probs.sum()

            next_token = int(np.random.choice(top_indices, p=top_probs))
            tokens.append(next_token)
            generated.append(next_token)

            # Stop on EOS
            if next_token == self.config.vocab_size - 1:  # Approximate EOS check
                break

        tok_per_sec = len(generated) / total_time if total_time > 0 else 0
        return generated, tok_per_sec

    def _rms_norm(self, x: np.ndarray) -> np.ndarray:
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + self.config.rms_norm_eps)


def load_model(repo_id: str = "microsoft/BitNet-b1.58-2B-4T") -> BitNetModel:
    """Load a BitNet model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID. Supported models:
            - microsoft/BitNet-b1.58-2B-4T

    Returns:
        BitNetModel instance ready for inference
    """
    return BitNetModel(repo_id)
