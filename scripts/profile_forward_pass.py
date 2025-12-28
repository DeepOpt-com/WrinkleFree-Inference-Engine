#!/usr/bin/env python
"""Profile a realistic forward pass to understand actual overhead.

This simulates what happens during a single token decode step.
"""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import torch.nn.functional as F
import time
import json
import os

from sgl_kernel.quantization.bitnet import bitnet_gemv, quantize_activations_i8
from sglang.srt.models.bitnet import _unpack_ternary_weights, _pack_ternary_weights

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("REALISTIC FORWARD PASS PROFILER")
print("="*70)

# Model config
n_layers = 30
hidden_dim = 2560
num_heads = 20
num_kv_heads = 4  # GQA
head_dim = 128
mlp_hidden = 6912
seq_len = 50  # Context length

print(f"\nModel: {n_layers}L, {hidden_dim}H, {num_heads}Q/{num_kv_heads}KV heads")
print(f"Context: {seq_len} tokens")

# Pre-allocate all tensors (like sglang does)
print("\nPre-allocating tensors...")

# Input activation
x = torch.randn(1, hidden_dim, dtype=torch.bfloat16)

# Packed weights (one layer's worth)
q_weight = torch.randint(0, 256, (hidden_dim, hidden_dim // 4), dtype=torch.uint8)
k_weight = torch.randint(0, 256, (num_kv_heads * head_dim, hidden_dim // 4), dtype=torch.uint8)
v_weight = torch.randint(0, 256, (num_kv_heads * head_dim, hidden_dim // 4), dtype=torch.uint8)
o_weight = torch.randint(0, 256, (hidden_dim, hidden_dim // 4), dtype=torch.uint8)
gate_weight = torch.randint(0, 256, (mlp_hidden, hidden_dim // 4), dtype=torch.uint8)
up_weight = torch.randint(0, 256, (mlp_hidden, hidden_dim // 4), dtype=torch.uint8)
down_weight = torch.randint(0, 256, (hidden_dim, mlp_hidden // 4), dtype=torch.uint8)

# KV cache (pre-allocated flat buffer)
max_tokens = 4096
k_cache = torch.randn(max_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)
v_cache = torch.randn(max_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)

# Token indices for this request
token_indices = torch.arange(seq_len, dtype=torch.int64)

# RMS norm weights
rms_weight = torch.ones(hidden_dim, dtype=torch.bfloat16)

# Output buffer (reused)
output = torch.empty_like(x)

n_runs = 100

print("\n" + "="*70)
print("TIMING INDIVIDUAL OPERATIONS IN CONTEXT")
print("="*70)

times = {}

# 1. RMS Norm
def rms_norm(x, weight, eps=1e-5):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(x.dtype)) * weight

for _ in range(10):
    _ = rms_norm(x, rms_weight)
start = time.perf_counter()
for _ in range(n_runs):
    normed = rms_norm(x, rms_weight)
times['rms_norm'] = (time.perf_counter() - start) / n_runs * 1000

# 2. Quantization
for _ in range(10):
    _ = quantize_activations_i8(x.float())
start = time.perf_counter()
for _ in range(n_runs):
    x_int8, scale = quantize_activations_i8(normed.float())
times['quantize'] = (time.perf_counter() - start) / n_runs * 1000

# 3. Q/K/V projections
x_int8_1d = x_int8.squeeze(0)
for _ in range(10):
    _ = bitnet_gemv(q_weight, x_int8_1d, 1.0)
start = time.perf_counter()
for _ in range(n_runs):
    q = bitnet_gemv(q_weight, x_int8_1d, 1.0)
times['q_gemv'] = (time.perf_counter() - start) / n_runs * 1000

start = time.perf_counter()
for _ in range(n_runs):
    k = bitnet_gemv(k_weight, x_int8_1d, 1.0)
times['k_gemv'] = (time.perf_counter() - start) / n_runs * 1000

start = time.perf_counter()
for _ in range(n_runs):
    v = bitnet_gemv(v_weight, x_int8_1d, 1.0)
times['v_gemv'] = (time.perf_counter() - start) / n_runs * 1000

# 4. KV cache gather
for _ in range(10):
    _ = k_cache[token_indices]
start = time.perf_counter()
for _ in range(n_runs):
    k_gathered = k_cache[token_indices]
    v_gathered = v_cache[token_indices]
times['kv_gather'] = (time.perf_counter() - start) / n_runs * 1000

# 5. Reshape Q for attention
q_test = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
for _ in range(10):
    _ = q_test.view(1, num_heads, 1, head_dim)
start = time.perf_counter()
for _ in range(n_runs):
    q_reshaped = q_test.view(1, num_heads, 1, head_dim)
times['q_reshape'] = (time.perf_counter() - start) / n_runs * 1000

# 6. K/V reshape with GQA expansion
k_test = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)
v_test = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)

for _ in range(10):
    _ = k_test.transpose(0, 1).unsqueeze(0)
start = time.perf_counter()
for _ in range(n_runs):
    # [seq, kv_heads, dim] -> [1, kv_heads, seq, dim]
    k_for_attn = k_test.transpose(0, 1).unsqueeze(0)
    v_for_attn = v_test.transpose(0, 1).unsqueeze(0)
times['kv_reshape'] = (time.perf_counter() - start) / n_runs * 1000

# 7. SDPA with GQA
q_attn = torch.randn(1, num_heads, 1, head_dim, dtype=torch.bfloat16)
k_attn = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
v_attn = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

for _ in range(10):
    _ = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, enable_gqa=True)
start = time.perf_counter()
for _ in range(n_runs):
    attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, enable_gqa=True)
times['sdpa'] = (time.perf_counter() - start) / n_runs * 1000

# 8. O projection (need to quantize attn output)
attn_flat = attn_out.view(1, -1)
for _ in range(10):
    attn_int8, attn_scale = quantize_activations_i8(attn_flat.float())
start = time.perf_counter()
for _ in range(n_runs):
    attn_int8, attn_scale = quantize_activations_i8(attn_flat.float())
times['o_quantize'] = (time.perf_counter() - start) / n_runs * 1000

attn_int8_1d = attn_int8.squeeze(0)
for _ in range(10):
    _ = bitnet_gemv(o_weight, attn_int8_1d, 1.0)
start = time.perf_counter()
for _ in range(n_runs):
    o = bitnet_gemv(o_weight, attn_int8_1d, 1.0)
times['o_gemv'] = (time.perf_counter() - start) / n_runs * 1000

# 9. Residual add
o_bf16 = o.view(1, -1).to(torch.bfloat16)
for _ in range(10):
    _ = x + o_bf16
start = time.perf_counter()
for _ in range(n_runs):
    x2 = x + o_bf16
times['residual1'] = (time.perf_counter() - start) / n_runs * 1000

# 10. MLP: RMS norm
start = time.perf_counter()
for _ in range(n_runs):
    mlp_normed = rms_norm(x2, rms_weight)
times['mlp_rms_norm'] = (time.perf_counter() - start) / n_runs * 1000

# 11. MLP: quantize
start = time.perf_counter()
for _ in range(n_runs):
    mlp_int8, mlp_scale = quantize_activations_i8(mlp_normed.float())
times['mlp_quantize'] = (time.perf_counter() - start) / n_runs * 1000

# 12. Gate/Up projections
mlp_int8_1d = mlp_int8.squeeze(0)
for _ in range(10):
    _ = bitnet_gemv(gate_weight, mlp_int8_1d, 1.0)
start = time.perf_counter()
for _ in range(n_runs):
    gate = bitnet_gemv(gate_weight, mlp_int8_1d, 1.0)
times['gate_gemv'] = (time.perf_counter() - start) / n_runs * 1000

start = time.perf_counter()
for _ in range(n_runs):
    up = bitnet_gemv(up_weight, mlp_int8_1d, 1.0)
times['up_gemv'] = (time.perf_counter() - start) / n_runs * 1000

# 13. SiLU activation and multiply
gate_bf16 = gate.to(torch.bfloat16)
up_bf16 = up.to(torch.bfloat16)
for _ in range(10):
    _ = F.silu(gate_bf16) * up_bf16
start = time.perf_counter()
for _ in range(n_runs):
    hidden = F.silu(gate_bf16) * up_bf16
times['silu_mul'] = (time.perf_counter() - start) / n_runs * 1000

# 14. Down projection
hidden_int8, hidden_scale = quantize_activations_i8(hidden.float())
for _ in range(10):
    _ = bitnet_gemv(down_weight, hidden_int8, 1.0)
start = time.perf_counter()
for _ in range(n_runs):
    down = bitnet_gemv(down_weight, hidden_int8, 1.0)
times['down_gemv'] = (time.perf_counter() - start) / n_runs * 1000

# 15. Final residual
down_bf16 = down.view(1, -1).to(torch.bfloat16)
start = time.perf_counter()
for _ in range(n_runs):
    out = x2 + down_bf16
times['residual2'] = (time.perf_counter() - start) / n_runs * 1000

print("\n" + "="*70)
print("PER-LAYER BREAKDOWN")
print("="*70)

total_layer = sum(times.values())
print(f"\n{'Operation':<20} {'Time (ms)':<12} {'% Layer':<10}")
print("-" * 45)
for name, t in sorted(times.items(), key=lambda x: -x[1]):
    print(f"{name:<20} {t:.4f}       {t/total_layer*100:>5.1f}%")
print("-" * 45)
print(f"{'LAYER TOTAL':<20} {total_layer:.4f}ms")

# Calculate per-token time
per_token = total_layer * n_layers
theoretical_tps = 1000 / per_token

print(f"\n" + "="*70)
print("PER-TOKEN ANALYSIS")
print("="*70)
print(f"""
Per-layer time:      {total_layer:.4f}ms
30 layers:           {per_token:.2f}ms
Theoretical TPS:     {theoretical_tps:.1f} tok/s

Measured sglang:     62ms → 16 tok/s
Our simulation:      {per_token:.1f}ms → {theoretical_tps:.0f} tok/s

Difference (sglang overhead): {62 - per_token:.1f}ms
""")

print("="*70)
print("WHERE IS THE 62ms GOING?")
print("="*70)
print(f"""
Our simulation:              {per_token:>6.1f}ms  (realistic forward pass)
HTTP server overhead:        ~10ms     (estimated: parsing, SSE, etc.)
Scheduler overhead:          ~5ms      (radix tree, batching)
Token sampling:              ~2ms      (logits processing, sampling)
Detokenization:              ~1ms      (token→text)
Memory pool management:      ~5ms      (allocation tracking)
GIL/async overhead:          ~5ms      (Python async, threading)
Unknown overhead:            {62 - per_token - 28:.1f}ms
-----------------------------------------
Total sglang measured:       ~62ms
""")
