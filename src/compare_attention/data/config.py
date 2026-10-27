"""Параметры kv_retrieval."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IGNORE_INDEX = -1
FORMAT_VERSION = 2
DATASET_NAME = "kv_retrieval"
SOURCE = "https://github.com/On-Point-RND/transformer-architecture-generalization"
META_NAME = "meta.json"
MAX_REJECTS = 10_000
PROGRESS_EVERY = 10_000


class KVRetrievalConfig(BaseModel):
    """Параметры одной подготовленной выборки kv_retrieval."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = Field(default=8192, ge=1)
    input_seq_len: int = Field(default=512, ge=2)
    num_kv_pairs: int = Field(default=64, ge=1)
    power_a: float = Field(default=0.01, gt=0)
    n_train: int = Field(default=100_000, ge=1)
    n_val: int = Field(default=1_000, ge=1)
    n_test: int = Field(default=1_000, ge=1)
    seed: int = Field(default=1337, ge=0)

    @field_validator("power_a")
    @classmethod
    def _finite_power(cls, power_a: float) -> float:
        if not math.isfinite(power_a):
            raise ValueError("power_a must be positive and finite")
        return power_a

    @model_validator(mode="after")
    def _fits_sequence(self) -> KVRetrievalConfig:
        if self.input_seq_len % 2:
            raise ValueError(f"input_seq_len must be even, got {self.input_seq_len}")
        if self.vocab_size <= self.input_seq_len:
            raise ValueError("vocab_size must exceed input_seq_len")
        if 4 * self.num_kv_pairs > self.input_seq_len:
            raise ValueError(
                "input_seq_len must be at least 4 * num_kv_pairs "
                "(K pairs plus room for K distinct query slots)",
            )
        return self
