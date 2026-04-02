# mlx-gemma4 🖤

**First MLX implementation of Google Gemma 4** — built by RavenX AI.

Supports text-only inference for all Gemma 4 variants:
- `E2B` (2.3B effective — phones/edge)  
- `E4B` (4.5B effective — laptops, audio+vision)
- `26B A4B` (MoE, 4B active — fast desktop inference)
- `31B` (dense — workstation/M3 Ultra)

## Key Architectural Features

Gemma 4 adds these on top of Gemma 3:

| Feature | Gemma 3 | Gemma 4 |
|---------|---------|---------|
| Per-Layer Embeddings | ❌ | ✅ `hidden_size_per_layer_input=256` |
| Global head dim | Same as local | ✅ `global_head_dim=512` (2x) |
| Hybrid attention | Sliding only | ✅ Sliding + Full alternating |
| Context window | 128K | 256K (medium models) |
| Final logit softcap | ❌ | ✅ `30.0` |
| Audio support | ❌ | ✅ E2B/E4B variants |

## Convert & Use

```bash
# Convert heretic (abliterated) E2B to MLX 4-bit
python convert_gemma4.py \
    --model p-e-w/gemma-4-E2B-it-heretic-ara \
    --bits 4 \
    --publish DeadByDawn101/gemma-4-E2B-heretic-mlx-4bit

# Convert standard E2B
python convert_gemma4.py \
    --model google/gemma-4-E2B-it \
    --bits 4 \
    --publish DeadByDawn101/gemma-4-E2B-mlx-4bit
```

## Inference

```python
from mlx_lm import load, generate

model, tokenizer = load("DeadByDawn101/gemma-4-E2B-heretic-mlx-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is Bitcoin dominance?"}],
    tokenize=False, add_generation_prompt=True
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

## Built By

[RavenX AI](https://github.com/DeadByDawn101) — empire building with gothic intelligence 🖤
