"""GPT-2 с подменяемым блоком внимания.

Pre-norm, GELU-MLP и абсолютные позиционные эмбеддинги — общие.
В каждом блоке внимание либо классическое, либо DeltaNet.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compare_attention.models.attention import build_attention
from compare_attention.models.config import GPT2Config

_INIT_STD = 0.02


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config, index) for index in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def forward(self, idx: Tensor) -> Tensor:
        time = idx.size(1)
        if time > self.config.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.config.block_size}")
        positions = torch.arange(time, device=idx.device)
        hidden = self.drop(self.wte(idx) + self.wpe(positions))
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.ln_f(hidden))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None:
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=_INIT_STD)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)

    def _scale_residual_projections(self) -> None:
        std = _INIT_STD / math.sqrt(2 * self.config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("c_proj.weight", "o_proj.weight", "out_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=std)


class Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = build_attention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))
