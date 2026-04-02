# mlx-gemma4 🖤

**First MLX implementation of Google Gemma 4** — built by [RavenX AI](https://github.com/DeadByDawn101).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-deadbydawn101-yellow)](https://huggingface.co/deadbydawn101)

Supports text-only inference for all Gemma 4 variants on Apple Silicon:

| Model | Params (effective) | Context | Modalities | VRAM |
|-------|--------------------|---------|------------|------|
| E2B | 2.3B | 128K | Text, Image, **Audio** | ~3GB |
| E4B | 4.5B | 128K | Text, Image, **Audio** | ~6GB |
| 26B A4B (MoE) | **3.8B active** | 256K | Text, Image | ~5GB |
| 31B | 30.7B | 256K | Text, Image | ~18GB |

## What's New in Gemma 4 (vs Gemma 3)

| Feature | Gemma 3 | Gemma 4 |
|---------|---------|---------|
| **Per-Layer Embeddings (PLE)** | ❌ | ✅ Each layer gets its own token embedding |
| **Global head dim** | Same as local | ✅ `global_head_dim=512` (2× local) |
| **Hybrid attention** | Sliding only | ✅ Sliding (local) + Full (global) alternating |
| **Context window** | 128K | 256K (26B/31B) |
| **Final logit softcap** | ❌ | ✅ `tanh(x/30) × 30` |
| **MoE architecture** | ❌ | ✅ 26B A4B: 128 experts, 8 active |
| **Audio** | ❌ | ✅ E2B/E4B: 30s audio, ASR + translation |
| **Native function calling** | ❌ | ✅ Structured tool use for agents |

## TurboQuant KV Cache Integration

mlx-gemma4 ships with **Gemma-4 aware KV cache compression** via [TurboQuant-MLX](https://github.com/DeadByDawn101/turboquant-mlx).

### Why Gemma-4 Needs a Smarter Cache Strategy

Gemma-4's hybrid attention creates two distinct KV cache behaviors:
- **Sliding (local) layers** — KV bounded by `sliding_window=512`. Already memory-efficient.
- **Global (full) layers** — KV grows unboundedly with context length. **This is where compression matters most.**

Our strategy:
```
Global layers  → TurboQuantKVCache (PolarQuant + QJL residual, 4.6× compression)
Sliding layers → RotatingKVCache   (window-bounded, minimal overhead)
```

Global layers also have `global_head_dim=512` (2× the local `head_dim=256`), making them **4× larger** per token in KV storage. TurboQuant cuts that back down.

### Usage

```python
from mlx_lm import load, generate
from mlx_gemma4.turboquant_cache import patch_gemma4_model

# Load any Gemma-4 variant
model, tokenizer = load("deadbydawn101/gemma-4-E2B-heretic-mlx-4bit")

# Patch with TurboQuant-aware cache (installs TurboQuant compression on global layers)
patch_gemma4_model(model)

# Generate with 4.6× KV compression on global attention layers
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain Solana's parallel transaction processing."}],
    tokenize=False, add_generation_prompt=True
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=500))
```

### Memory Savings

For a 128K token context:

| Cache Type | Global Layer KV | Total (35 layers) |
|------------|-----------------|-------------------|
| Standard | ~2GB | ~14GB |
| TurboQuant (4-bit) | ~450MB | ~3.1GB |
| **Savings** | **4.4×** | **4.5×** |

## Install

```bash
git clone https://github.com/DeadByDawn101/mlx-gemma4
cd mlx-gemma4
pip install mlx mlx-lm numpy

# Optional: TurboQuant for KV compression
git clone https://github.com/DeadByDawn101/turboquant-mlx
pip install -e turboquant-mlx
```

## Convert & Publish

Convert any Gemma-4 checkpoint (including abliterated variants) to MLX:

```bash
# Heretic E2B (abliterated) → 4-bit MLX
python convert_gemma4.py \
    --model p-e-w/gemma-4-E2B-it-heretic-ara \
    --bits 4 \
    --publish YourHFUsername/gemma-4-E2B-heretic-mlx-4bit

# Standard E4B → 4-bit MLX  
python convert_gemma4.py \
    --model google/gemma-4-E4B-it \
    --bits 4 \
    --publish YourHFUsername/gemma-4-E4B-mlx-4bit

# 26B MoE → 4-bit MLX (runs at 4B speed)
python convert_gemma4.py \
    --model google/gemma-4-26b-a4b-it \
    --bits 4 \
    --publish YourHFUsername/gemma-4-26B-MoE-mlx-4bit
```

## Pre-converted Models (HuggingFace)

| Model | HF Repo | Size |
|-------|---------|------|
| Gemma-4-E2B-heretic (abliterated) | [deadbydawn101/gemma-4-E2B-heretic-mlx-4bit](https://huggingface.co/deadbydawn101/gemma-4-E2B-heretic-mlx-4bit) | ~3GB |
| Gemma-4-E4B | [deadbydawn101/gemma-4-E4B-mlx-4bit](https://huggingface.co/deadbydawn101/gemma-4-E4B-mlx-4bit) | ~5GB |

Pull via Ollama:
```bash
ollama pull hf.co/deadbydawn101/gemma-4-E2B-heretic-mlx-4bit
ollama run hf.co/deadbydawn101/gemma-4-E2B-heretic-mlx-4bit
```

## Architecture Notes

### Per-Layer Embeddings (PLE)
Gemma-4's most novel addition. Each decoder layer gets its own small embedding table (`hidden_size_per_layer_input=256`) that injects per-layer token representations. This gives each layer fresh, layer-specific token information rather than relying solely on the propagated hidden state.

```python
# In each layer:
ple = self.ple_proj(self.ple_embed(input_ids))  # Layer-specific token info
x = x + ple  # Injected before attention
```

### Hybrid Attention Pattern
```
Layer 0: sliding   (window=512)
Layer 1: sliding
Layer 2: sliding
Layer 3: sliding
Layer 4: FULL      ← global, head_dim=512, unbounded KV
Layer 5: sliding
...
```
Global layers occur roughly every 5 layers (7 global out of 35 for E2B).

### Attention-Aware Eviction (Research Direction)
The `AttentionAwareCache` class implements importance-score-based eviction inspired by memory consolidation research:

```
importance = base_weight × recency_decay × attention_boost
```

Tokens frequently attended to are kept longer. This mirrors how the brain (and Claude Code's autoDream memory system) decides what to preserve vs. compress — high-reference items survive, low-reference items decay.

## Benchmarks

*Coming soon — running on M3 Ultra 96GB + M4 Max 128GB via Thunderbolt 5.*

## Related Projects

- [TurboQuant-MLX](https://github.com/DeadByDawn101/turboquant-mlx) — KV cache compression for Apple Silicon
- [star-platinum-cluster](https://github.com/DeadByDawn101/star-platinum-cluster) — Multi-node MLX inference cluster
- [OpenClaude](https://github.com/DeadByDawn101/openclaude) — Claude Code tools + local models

## Built By

[RavenX AI](https://github.com/DeadByDawn101) — empire building with gothic intelligence 🖤

> *"Not a tool. An empire."*

---

Apache 2.0 — fork it, ship it, improve it.
