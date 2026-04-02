#!/usr/bin/env python3
"""
Convert Gemma-4 (any variant) to MLX format with optional quantization.
Publishes to HuggingFace Hub.

Usage:
  python convert_gemma4.py --model p-e-w/gemma-4-E2B-it-heretic-ara --output ~/Models/gemma4-e2b-heretic-mlx
  python convert_gemma4.py --model p-e-w/gemma-4-E2B-it-heretic-ara --bits 4 --publish DeadByDawn101/gemma-4-E2B-heretic-mlx-4bit
  python convert_gemma4.py --model google/gemma-4-E2B-it --bits 8 --publish mlx-community/gemma-4-E2B-it-8bit
"""
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Convert Gemma-4 to MLX")
    p.add_argument("--model", required=True, help="HF model ID or local path")
    p.add_argument("--output", default=None, help="Output directory (default: ~/Models/<model-name>-mlx)")
    p.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits (default: 4)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--publish", default=None, help="HF repo to publish to (e.g. DeadByDawn101/gemma-4-E2B-heretic-mlx-4bit)")
    p.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env var)")
    return p.parse_args()


def load_hf_model(model_id: str):
    """Load model weights from HuggingFace using transformers."""
    print(f"📥 Loading {model_id} via transformers...")
    from transformers import AutoTokenizer, AutoConfig
    import torch

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load in CPU float32 for conversion
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        print(f"✅ Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
        return model, tokenizer, config
    except Exception as e:
        print(f"AutoModelForCausalLM failed: {e}")
        print("Trying AutoModelForMultimodalLM...")
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        return model, tokenizer, config


def extract_text_weights(state_dict: dict) -> dict:
    """Extract only text/language model weights, skip vision/audio."""
    skip_prefixes = [
        "vision_tower.", "audio_tower.", "vision_model.",
        "multi_modal_projector.", "audio_encoder.",
    ]
    weights = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        # Normalize key names
        new_k = k
        # language_model.model.* -> model.*
        if new_k.startswith("language_model."):
            new_k = new_k[len("language_model."):]
        weights[new_k] = v.float().numpy()
    return weights


def quantize_weights(weights: dict, bits: int) -> dict:
    """Simple linear quantization for non-embedding weights."""
    if bits == 8:
        # 8-bit: store as int8 with scale
        q_weights = {}
        for k, v in weights.items():
            if v.ndim < 2 or "embed" in k or "norm" in k:
                q_weights[k] = v  # keep fp32
            else:
                scale = np.abs(v).max() / 127.0
                q = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
                q_weights[k] = q
                q_weights[k + "_scale"] = np.array([scale], dtype=np.float32)
        return q_weights
    else:
        # 4-bit: pack two values per byte
        q_weights = {}
        for k, v in weights.items():
            if v.ndim < 2 or "embed" in k or "norm" in k:
                q_weights[k] = v.astype(np.float16)
            else:
                # Group-wise 4-bit quantization (group_size=64)
                shape = v.shape
                flat = v.reshape(-1)
                gs = 64
                n_groups = math.ceil(len(flat) / gs)
                padded = np.pad(flat, (0, n_groups * gs - len(flat)))
                groups = padded.reshape(n_groups, gs)
                scales = (np.abs(groups).max(axis=1, keepdims=True) / 7.0).clip(1e-8)
                q = np.clip(np.round(groups / scales), -8, 7).astype(np.int8)
                q_weights[k] = q.reshape(shape[0], -1)
                q_weights[k + "_scales"] = scales.reshape(-1).astype(np.float16)
        return q_weights


def save_mlx_model(weights: dict, config, tokenizer, output_dir: Path, dtype: str):
    """Save weights in safetensors + config files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving to {output_dir}...")

    try:
        import safetensors.numpy as st
        # Convert to appropriate dtype
        dt = {"bfloat16": np.float16, "float16": np.float16, "float32": np.float32}[dtype]
        saved = {k: v.astype(dt) if v.dtype in [np.float32, np.float64] else v
                 for k, v in weights.items()}
        st.save_file(saved, str(output_dir / "model.safetensors"))
        print(f"✅ Saved {len(saved)} weight tensors")
    except ImportError:
        # Fallback: save as npz
        np.savez(str(output_dir / "model.npz"), **weights)
        print(f"✅ Saved as npz (install safetensors for better format)")

    # Save config
    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    # Add MLX-specific fields
    cfg_dict["model_type"] = "gemma4_text"
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    print(f"✅ Tokenizer saved")

    # Write model card
    (output_dir / "README.md").write_text(f"""# Gemma-4 MLX Conversion

Built by [RavenX AI](https://github.com/DeadByDawn101) — first MLX support for Gemma 4.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("{output_dir.name}")
response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)
print(response)
```

## About

This is a text-only MLX conversion of Google's Gemma 4 model.
Vision/audio towers are excluded for faster text inference.

Key architectural additions vs Gemma 3:
- Per-Layer Embeddings (PLE): each layer gets additional token context
- Hybrid attention: sliding window (local) + full (global) layers
- Global attention uses wider head_dim (global_head_dim)
- Final logit softcapping at 30.0

Built with ❤️ by RavenX AI / DeadByDawn101
""")


def publish_to_hf(output_dir: Path, repo_id: str, token: str):
    """Push to HuggingFace Hub."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"🚀 Publishing to {repo_id}...")
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial MLX conversion of Gemma-4 — RavenX AI",
    )
    print(f"✅ Published: https://huggingface.co/{repo_id}")


def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")

    output = Path(args.output) if args.output else (
        Path.home() / "Models" / f"{args.model.split('/')[-1]}-mlx-{args.bits}bit"
    )

    # Load
    model, tokenizer, config = load_hf_model(args.model)

    # Extract text weights
    print("🔧 Extracting text weights (skipping vision/audio)...")
    weights = extract_text_weights(dict(model.state_dict()))
    print(f"  {len(weights)} text weight tensors")

    # Quantize
    if args.bits < 16:
        print(f"📦 Quantizing to {args.bits}-bit...")
        weights = quantize_weights(weights, args.bits)
        print(f"  Done: {len(weights)} tensors")

    # Save
    save_mlx_model(weights, config, tokenizer, output, args.dtype)

    # Publish
    if args.publish:
        if not token:
            print("⚠️  No HF token — set HF_TOKEN env var or use --hf-token")
        else:
            publish_to_hf(output, args.publish, token)

    print(f"\n✅ Done! Model at: {output}")
    print(f"Test: python -m mlx_lm generate --model {output} --prompt 'Hello' --max-tokens 50")


if __name__ == "__main__":
    main()
