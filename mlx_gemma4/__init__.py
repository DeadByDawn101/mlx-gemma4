"""
mlx-gemma4 — MLX implementation of Google Gemma 4
Supports text-only inference for E2B, E4B, 26B A4B (MoE), 31B variants.

Built by RavenX AI / DeadByDawn101
GitHub: https://github.com/DeadByDawn101/mlx-gemma4
"""

from .gemma4_text import Model, ModelArgs

__version__ = "0.1.0"
__all__ = ["Model", "ModelArgs"]
