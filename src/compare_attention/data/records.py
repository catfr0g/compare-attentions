"""Выборки kv_retrieval: промпты, метки и результат записи."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from compare_attention.data.config import KVRetrievalConfig

_FROZEN = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class TokenSplit(BaseModel):
    """Входы и попозиционные метки одного разбиения."""

    model_config = _FROZEN

    inputs: np.ndarray
    labels: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenSplit):
            return NotImplemented
        same_inputs = np.array_equal(self.inputs, other.inputs)
        same_labels = np.array_equal(self.labels, other.labels)
        return same_inputs and same_labels


class KVRetrievalDataset(BaseModel):
    """Обучающая, валидационная и тестовая выборки одного сида."""

    model_config = _FROZEN

    config: KVRetrievalConfig
    train: TokenSplit
    validation: TokenSplit
    test: TokenSplit


class PrepareResult(BaseModel):
    """Итог записи датасета: каталог и признак того, что файлы были созданы заново."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path
    written: bool
