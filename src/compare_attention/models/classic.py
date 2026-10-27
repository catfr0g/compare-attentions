"""Классическое внимание GPT-2.

Слой — ``torch.nn.MultiheadAttention``. Снаружи один тензор той же формы,
что и у DeltaNet.
"""

from torch import Tensor, nn

from compare_attention.models.config import GPT2Config


class ClassicAttention(nn.Module):
    """Адаптер MultiheadAttention: [B, T, n_embd] -> [B, T, n_embd]."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.layer = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(length, device=x.device)
        hidden, _ = self.layer(x, x, x, attn_mask=causal, need_weights=False, is_causal=True)
        return self.resid_dropout(hidden)
