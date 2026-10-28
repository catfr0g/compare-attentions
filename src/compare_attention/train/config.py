"""Бюджет сравнения и параметры одного запуска."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.data.store import load_config
from compare_attention.models.config import GPT2Config

_FROZEN = ConfigDict(frozen=True)
# flash-linear-attention длиннее этого порога считает чанковым ядром, оно не принимает float32.
DELTA_CHUNKED_ABOVE = 64


class ComparisonAxis(StrEnum):
    """Как уравнивать архитектуры.

    Сейчас обучает только ``train_flops``. ``layer_flops`` и ``parameters``
    пока не отбирают сетку.
    """

    TRAIN_FLOPS = "train_flops"
    LAYER_FLOPS = "layer_flops"
    PARAMETERS = "parameters"


class Schedule(BaseModel):
    """Сколько считать и каким шагом."""

    model_config = _FROZEN

    train_flops: int = Field(gt=0, description="Бюджет FLOPs на обучение, включая обратный проход.")
    axis: ComparisonAxis = Field(
        default=ComparisonAxis.TRAIN_FLOPS,
        description="Ось сравнения. Пока выполняется только train_flops.",
    )
    batch_size: int = Field(default=8, ge=1)
    seed: int = Field(default=1337, ge=0)
    dtype: Literal["float32", "bfloat16"] = "float32"
    compile: bool = Field(default=True, description="Собрать модель через torch.compile.")

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32


class Optim(BaseModel):
    """AdamW."""

    model_config = _FROZEN

    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)


class TrainConfig(BaseModel):
    """Один запуск: данные, сетка, бюджет."""

    model_config = _FROZEN

    data: Path
    model: GPT2Config
    schedule: Schedule
    optim: Optim = Field(default_factory=Optim)
    dataset: KVRetrievalConfig

    @model_validator(mode="before")
    @classmethod
    def _read_dataset(cls, raw: object) -> object:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        payload["dataset"] = load_config(payload["data"])
        return payload

    @model_validator(mode="after")
    def _supported_axis(self) -> TrainConfig:
        if self.schedule.axis is not ComparisonAxis.TRAIN_FLOPS:
            raise NotImplementedError(f"axis {self.schedule.axis.value} is not implemented")
        return self

    @model_validator(mode="after")
    def _vocab_matches_dataset(self) -> TrainConfig:
        if self.model.vocab_size != self.dataset.vocab_size:
            raise ValueError("vocab_size must match the dataset")
        return self

    @model_validator(mode="after")
    def _block_covers_sequence(self) -> TrainConfig:
        if self.model.block_size < self.dataset.input_seq_len:
            raise ValueError("block_size must cover the dataset sequence")
        return self

    @model_validator(mode="after")
    def _batch_fits_training_split(self) -> TrainConfig:
        if self.schedule.batch_size > self.dataset.n_train:
            raise ValueError("batch_size must not exceed the training split")
        return self

    @model_validator(mode="after")
    def _deltanet_long_sequence_is_bfloat16(self) -> TrainConfig:
        long_deltanet = (
            self.model.attention == "deltanet" and self.dataset.input_seq_len > DELTA_CHUNKED_ABOVE
        )
        if long_deltanet and self.schedule.dtype == "float32":
            raise ValueError("deltanet sequences longer than 64 require bfloat16")
        return self
