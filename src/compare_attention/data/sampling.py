"""Генерация примеров kv_retrieval (MQAR).

Процедура совпадает с ``tasks/kv_retrieval.py`` из
https://github.com/On-Point-RND/transformer-architecture-generalization
(Zoology, Arora et al., 2023). Последовательность начинается с K пар
ключ–значение. Затем каждый ключ встречается ещё раз, смещение выбирается
по степенному закону ``power_a``. Метка стоит в той же позиции, что и
ключ-запрос, и равна парному значению. На остальных позициях метка ``-1``.

Ключи берутся из нижней половины словаря, кроме токена 0. Значения — из
верхней. Сначала снимается валидация, затем тест, затем обучение, поэтому
размер обучающей выборки не сдвигает val и test.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, SkipValidation

from compare_attention.data.config import (
    DATASET_NAME,
    IGNORE_INDEX,
    MAX_REJECTS,
    PROGRESS_EVERY,
    KVRetrievalConfig,
)
from compare_attention.data.records import KVRetrievalDataset, TokenSplit

logger = logging.getLogger(__name__)


def gap_probabilities(input_seq_len: int, num_kv_pairs: int, power_a: float) -> np.ndarray:
    """Веса смещений запроса: ``power_a * i ** (power_a - 1)``, затем нормировка."""
    space = (input_seq_len - 2 * num_kv_pairs) // 2
    indices = np.arange(1, space + 1, dtype=np.float64)
    weights = power_a * indices ** (power_a - 1.0)
    return weights / weights.sum()


def _write_pairs(prompt: np.ndarray, keys: np.ndarray, pair_values: np.ndarray) -> None:
    for offset, key in enumerate(keys):
        prompt[offset * 2] = key
        prompt[offset * 2 + 1] = pair_values[offset]


def _write_queries(
    prompt: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
    pair_values: np.ndarray,
    gaps: np.ndarray,
) -> None:
    context = len(keys) * 2
    for key, pair_value, gap in zip(keys, pair_values, gaps, strict=True):
        position = context + 2 * int(gap)
        prompt[position] = key
        labels[position] = pair_value


def sample_example(
    rng: np.random.Generator,
    config: KVRetrievalConfig,
    gap_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Один пример: промпт и метки длины ``input_seq_len``."""
    num_pairs = config.num_kv_pairs
    key_options = np.arange(1, config.vocab_size // 2)
    value_options = np.arange(config.vocab_size // 2, config.vocab_size)
    keys = rng.choice(key_options, size=num_pairs, replace=False)
    pair_values = rng.choice(value_options, size=num_pairs, replace=False)
    gaps = rng.choice(len(gap_probs), size=num_pairs, replace=False, p=gap_probs)
    prompt = rng.integers(0, config.vocab_size, size=config.input_seq_len, dtype=np.int64)
    labels = np.full(config.input_seq_len, IGNORE_INDEX, dtype=np.int64)
    _write_pairs(prompt, keys, pair_values)
    _write_queries(prompt, labels, keys, pair_values, gaps)
    return prompt, labels


def _note_duplicate(rejects: int, max_rejects: int, split_name: str) -> int:
    updated = rejects + 1
    if updated >= max_rejects:
        raise ValueError(
            f"{DATASET_NAME} drew {max_rejects} duplicate prompts in a row "
            f"while building the {split_name} split",
        )
    return updated


class _Batch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inputs: np.ndarray
    labels: np.ndarray
    taken: SkipValidation[set[bytes]]
    filled: int = 0
    rejects: int = 0


def _remember(
    batch: _Batch,
    prompt: np.ndarray,
    label: np.ndarray,
    key: bytes,
    split_name: str,
) -> None:
    batch.taken.add(key)
    batch.inputs[batch.filled] = prompt
    batch.labels[batch.filled] = label
    batch.filled += 1
    batch.rejects = 0
    milestone = batch.filled % PROGRESS_EVERY == 0
    finished = batch.filled == len(batch.inputs)
    if milestone or finished:
        logger.info("%s %s: %s/%s", DATASET_NAME, split_name, batch.filled, len(batch.inputs))


def _take_one(
    sample: Any,
    rng: np.random.Generator,
    config: KVRetrievalConfig,
    gap_probs: np.ndarray,
    batch: _Batch,
    max_rejects: int,
    split_name: str,
) -> None:
    prompt, label = sample(rng, config, gap_probs)
    key = prompt.tobytes()
    if key in batch.taken:
        batch.rejects = _note_duplicate(batch.rejects, max_rejects, split_name)
        return
    _remember(batch, prompt, label, key, split_name)


def generate_split(
    rng: np.random.Generator,
    count: int,
    config: KVRetrievalConfig,
    gap_probs: np.ndarray,
    taken: set[bytes],
    sample: Any = sample_example,
    max_rejects: int = MAX_REJECTS,
    split_name: str = "split",
) -> tuple[np.ndarray, np.ndarray]:
    """``count`` примеров, чьи промпты ещё не входят в ``taken``."""
    batch = _Batch(
        inputs=np.empty((count, config.input_seq_len), dtype=np.int64),
        labels=np.empty((count, config.input_seq_len), dtype=np.int64),
        taken=taken,
    )
    while batch.filled < count:
        _take_one(sample, rng, config, gap_probs, batch, max_rejects, split_name)
    return batch.inputs, batch.labels


def _draw_split(
    rng: np.random.Generator,
    count: int,
    config: KVRetrievalConfig,
    gap_probs: np.ndarray,
    taken: set[bytes],
    split_name: str,
) -> TokenSplit:
    inputs, labels = generate_split(rng, count, config, gap_probs, taken, split_name=split_name)
    return TokenSplit(inputs=inputs, labels=labels)


def generate_dataset(config: KVRetrievalConfig) -> KVRetrievalDataset:
    """Валидация, затем тест, затем обучение, из одного генератора ``config.seed``."""
    rng = np.random.default_rng(config.seed)
    probs = gap_probabilities(config.input_seq_len, config.num_kv_pairs, config.power_a)
    taken: set[bytes] = set()
    validation = _draw_split(rng, config.n_val, config, probs, taken, "val")
    test = _draw_split(rng, config.n_test, config, probs, taken, "test")
    train = _draw_split(rng, config.n_train, config, probs, taken, "train")
    return KVRetrievalDataset(config=config, train=train, validation=validation, test=test)
