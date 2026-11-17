"""Сборка механизма внимания по конфигу."""

from torch import nn

from compare_attention.models.classic import ClassicAttention
from compare_attention.models.config import GPT2Config


def build_attention(config: GPT2Config, layer_idx: int = 0) -> nn.Module:
    if config.attention == "classic":
        return ClassicAttention(config)
    # Optional CUDA backend is loaded only when selected; model math is unchanged.
    from compare_attention.models.deltanet import DeltaNet  # pylint: disable=import-outside-toplevel

    return DeltaNet(config, layer_idx)
