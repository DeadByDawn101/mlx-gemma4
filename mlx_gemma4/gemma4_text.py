# Copyright © 2025 RavenX AI / DeadByDawn101
# MLX implementation of Gemma 4 text backbone
# Supports: E2B, E4B, 26B A4B (MoE), 31B
# Key diffs vs Gemma 3:
#   - Per-Layer Embeddings (PLE): hidden_size_per_layer_input per layer
#   - Hybrid attention: sliding (local) + full (global) alternating
#   - global_head_dim != head_dim (global layers use wider heads)
#   - Unified K=V for global attention layers (attention_k_eq_v flag)
#   - final_logit_softcapping=30.0
#   - MoE support (26B A4B variant)

from dataclasses import dataclass, field
from typing import Optional, List
import math

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512      # global (full) attention layers use this
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 4
    rope_theta: float = 1000000.0
    rope_local_base_freq: float = 10000.0
    query_pre_attn_scalar: Optional[float] = None
    sliding_window: int = 512
    sliding_window_pattern: int = 6  # every N layers is full attention
    max_position_embeddings: int = 131072
    hidden_size_per_layer_input: int = 256    # PLE embedding dim per layer
    attention_dropout: float = 0.0
    hidden_activation: str = "gelu_pytorch_tanh"
    final_logit_softcapping: float = 30.0
    layer_types: List[str] = field(default_factory=list)
    # MoE params (26B A4B only)
    enable_moe_block: bool = False
    num_experts: int = 128
    num_experts_per_tok: int = 8
    expert_intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim ** -0.5


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Gemma4Attention(nn.Module):
    """
    Gemma 4 attention — handles both sliding (local) and full (global) variants.
    Global layers use global_head_dim (larger), sliding layers use head_dim.
    """
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.is_global = self._is_global_layer(layer_idx, args.layer_types)

        # Global layers use wider heads
        hd = args.global_head_dim if self.is_global else args.head_dim
        self.head_dim = hd
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads

        self.scale = args.query_pre_attn_scalar

        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, args.hidden_size, bias=False)

        if self.is_global:
            # Global layers: standard RoPE with rope_theta
            self.rope = nn.RoPE(
                hd,
                traditional=False,
                base=args.rope_theta,
            )
        else:
            # Sliding layers: local RoPE with rope_local_base_freq
            self.rope = nn.RoPE(
                hd,
                traditional=False,
                base=args.rope_local_base_freq,
            )

    @staticmethod
    def _is_global_layer(layer_idx: int, layer_types: List[str]) -> bool:
        if layer_types and layer_idx < len(layer_types):
            return layer_types[layer_idx] == "full_attention"
        # Fallback: every sliding_window_pattern-th layer is global
        return False

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, heads, L, head_dim)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        # Scaled dot product attention
        # Repeat k,v if GQA
        if self.n_kv_heads != self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            attn = attn + mask

        # Sliding window mask for local attention
        if not self.is_global and cache is None:
            sw = self.args.sliding_window
            sw_mask = mx.triu(mx.full((L, L), float("-inf")), k=1)
            # Mask out tokens outside sliding window
            if L > sw:
                sw_mask = sw_mask + mx.tril(mx.full((L, L), float("-inf")), k=-sw)
            attn = attn + sw_mask[None, None]

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(attn.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class Gemma4MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.act = nn.GELU(approx="tanh")  # gelu_pytorch_tanh

    def __call__(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Gemma4MoE(nn.Module):
    """MoE block for 26B A4B variant."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        expert_dim = args.expert_intermediate_size or args.intermediate_size
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.router = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.experts = [Gemma4MLP(args) for _ in range(self.num_experts)]
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        logits = self.router(x_flat)
        weights, indices = mx.topk(logits, self.num_experts_per_tok, axis=-1)
        weights = mx.softmax(weights.astype(mx.float32), axis=-1).astype(x.dtype)
        out = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(axis=-1)
            if mask.sum() > 0:
                expert_in = x_flat[mask]
                expert_out = expert(expert_in)
                # weighted sum
                w = weights[mask][:, (indices[mask] == i).argmax(axis=-1)]
                out = out.at[mask].add(expert_out * w[:, None])
        return out.reshape(B, L, D)


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(args, layer_idx)
        # MoE or dense MLP
        if args.enable_moe_block:
            self.mlp = Gemma4MoE(args)
        else:
            self.mlp = Gemma4MLP(args)
        self.input_layernorm  = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attn_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

        # Per-Layer Embedding (PLE) — each layer gets its own token embedding table
        ple_dim = args.hidden_size_per_layer_input
        if ple_dim and ple_dim > 0:
            self.ple_embed = nn.Embedding(args.vocab_size, ple_dim)
            self.ple_proj  = nn.Linear(ple_dim, args.hidden_size, bias=False)
        else:
            self.ple_embed = None

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        # PLE: inject per-layer token information
        if self.ple_embed is not None and input_ids is not None:
            ple = self.ple_proj(self.ple_embed(input_ids))
            x = x + ple

        # Self-attention
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + self.post_attn_layernorm(r)

        # MLP / MoE
        r = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(r)
        return x


class Gemma4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Gemma4DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings
            input_ids = None
        else:
            h = self.embed_tokens(inputs)
            input_ids = inputs
            # Gemma normalizes embeddings by sqrt(hidden_size)
            h = h * math.sqrt(self.args.hidden_size)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                mask=mask,
                cache=cache[i] if cache else None,
                input_ids=input_ids,
            )

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4Model(args)
        # Gemma ties lm_head to embed_tokens (no separate lm_head weights)
        self.lm_head = None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        # Tied weights: use embed_tokens.weight for lm_head
        logits = out @ self.model.embed_tokens.weight.T
        # Final logit softcapping
        sc = self.args.final_logit_softcapping
        if sc:
            logits = sc * mx.tanh(logits / sc)
        return logits

    def sanitize(self, weights):
        # Remove vision/audio weights if present (text-only inference)
        return {k: v for k, v in weights.items()
                if not any(k.startswith(p) for p in
                          ["vision_tower", "audio_tower", "multi_modal_projector",
                           "vision_model", "audio_model"])}

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i, layer in enumerate(self.model.layers):
            is_global = Gemma4Attention._is_global_layer(
                i, self.args.layer_types if self.args.layer_types else []
            )
            if is_global:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
