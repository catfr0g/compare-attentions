"""DeltaNet в слоте внимания GPT-2.

Слой берётся из flash-linear-attention и снаружи отдаёт один тензор
той же формы, что и классическое внимание.

Yang et al., 2024, https://arxiv.org/abs/2406.06484
"""

from fla.layers.delta_net import DeltaNet as LibraryDeltaNet
from torch import Tensor, nn

from compare_attention.models.config import GPT2Config


class DeltaNet(nn.Module):
    """Адаптер библиотечного DeltaNet: [B, T, n_embd] -> [B, T, n_embd]."""

    def __init__(self, config: GPT2Config, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer = LibraryDeltaNet(
            d_model=config.n_embd,
            num_heads=config.n_head,
            layer_idx=layer_idx,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden, _, _ = self.layer(x)
        return self.resid_dropout(hidden)
