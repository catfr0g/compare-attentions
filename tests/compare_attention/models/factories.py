"""Маленький GPT-2 для тестов."""

from compare_attention.models.config import GPT2Config

_BASE = {
    "vocab_size": 32,
    "block_size": 16,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 8,
    "dropout": 0.0,
    "bias": True,
    "attention": "classic",
}


def gpt2_config(**overrides: object) -> GPT2Config:
    params = {**_BASE, **overrides}
    return GPT2Config(**params)
