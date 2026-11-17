"""Causal SDPA с исходными параметрами и инициализацией MultiheadAttention."""

from torch import Tensor, nn
from torch.nn import functional as F

from compare_attention.models.config import GPT2Config


class ClassicAttention(nn.Module):
    """Прямой QKV/SDPA; MHA сохранён как совместимый контейнер параметров."""

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
        layer = self.layer
        if (
            not layer._qkv_same_embed_dim  # pylint: disable=protected-access
            or layer.bias_k is not None
            or layer.bias_v is not None
            or layer.add_zero_attn
            or not layer.batch_first
        ):
            raise ValueError("Direct SDPA requires the repository self-attention contract")
        batch, length, width = x.shape
        heads = layer.num_heads
        dimension = width // heads
        q, k, v = F.linear(x, layer.in_proj_weight, layer.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(batch, length, heads, dimension).transpose(1, 2)
        k = k.reshape(batch, length, heads, dimension).transpose(1, 2)
        v = v.reshape(batch, length, heads, dimension).transpose(1, 2)
        dropout = layer.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        output = output.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.resid_dropout(layer.out_proj(output))
