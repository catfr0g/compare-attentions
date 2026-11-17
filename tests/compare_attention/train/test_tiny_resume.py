"""Character metadata and padded tails retain the existing exact CPU resume contract."""

import hashlib
import random

import numpy as np
import pytest
import torch

from compare_attention.data.tiny_shakespeare import prepare_tiny_shakespeare
from compare_attention.models.config import GPT2Config
from compare_attention.train import loop
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.config import Schedule, TrainConfig
from compare_attention.train.flops import estimate_step
from tests.compare_attention.train.test_resume_equivalence import (
    _assert_nested_equal,
    _InterruptedBackup,
    _next_random_values,
    _RandomBackup,
    _SnapshotInterrupt,
)


def test_character_lm_full_resume_cpu(tmp_path, monkeypatch):
    # pylint: disable=too-many-locals
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    torch.set_num_threads(1)
    raw = b"ab\nc" * 101
    source = tmp_path / "input.txt"
    source.write_bytes(raw)
    data = tmp_path / "data"
    prepare_tiny_shakespeare(
        source, data, context=8, expected_sha256=hashlib.sha256(raw).hexdigest()
    )
    config = TrainConfig(
        data=data,
        model=GPT2Config(vocab_size=4, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.2),
        schedule=Schedule(
            train_flops=1, batch_size=2, compile=False, checkpoint_every_steps=2, val_every_steps=2
        ),
    )
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"train_flops": estimate_step(config).total * 6}
            )
        }
    )
    random.seed(123)
    np.random.seed(123)
    reference = tmp_path / "full"
    loop.train(config, _RandomBackup(reference, False))
    expected_next = _next_random_values(False)
    random.seed(123)
    np.random.seed(123)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(_SnapshotInterrupt):
        loop.train(config, _InterruptedBackup(interrupted, False))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    resumed = tmp_path / "resumed"
    loop.train(
        config, _RandomBackup(resumed, False), resume_from=interrupted / "checkpoints/latest.json"
    )
    actual_next = _next_random_values(False)
    expected = load_checkpoint(reference / "checkpoints/latest.json")
    actual = load_checkpoint(resumed / "checkpoints/latest.json")
    for field in (
        "model",
        "optimizer",
        "precision",
        "rng",
        "cursor",
        "losses",
        "validation_history",
        "early_stopping",
        "step",
        "flops",
    ):
        _assert_nested_equal(expected[field], actual[field], field)
    _assert_nested_equal(expected_next, actual_next, "next_rng")
