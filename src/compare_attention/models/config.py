"""Размер GPT-2 и выбор механизма внимания."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AttentionName = Literal["classic", "deltanet"]


class GPT2Config(BaseModel):
    """Каркас GPT-2. ``attention`` подменяет только блок внимания."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = Field(default=8192, ge=1)
    block_size: int = Field(default=512, ge=1)
    n_layer: int = Field(default=12, ge=1)
    n_head: int = Field(default=12, ge=1)
    n_embd: int = Field(default=768, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    bias: bool = True
    attention: AttentionName = "classic"

    @model_validator(mode="after")
    def _heads_divide_width(self) -> GPT2Config:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self
