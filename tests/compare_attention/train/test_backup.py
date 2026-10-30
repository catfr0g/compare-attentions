# pylint: disable=redefined-outer-name
"""Локальные артефакты запуска: metrics.jsonl и чекпоинт модели."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.backup import BackupTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


def _rows(result_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (result_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _plain_model() -> GPT2:
    return GPT2(gpt2_config(n_layer=1))


def test_every_step_lands_in_the_journal(prepared: Path, tmp_path: Path) -> None:
    backup = BackupTracker(tmp_path)
    backup.start(train_config(prepared))

    backup.log_loss(0.5, 128)
    backup.log_loss(0.25, 256)

    assert _rows(tmp_path) == [
        {"type": "step", "loss": 0.5, "flops": 128},
        {"type": "step", "loss": 0.25, "flops": 256},
    ]


def test_the_result_lands_in_the_journal(prepared: Path, tmp_path: Path) -> None:
    config = train_config(prepared)
    backup = BackupTracker(tmp_path)
    backup.start(config)

    backup.finish(train_result(config, steps=2), _plain_model())

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["type"] == "result"
    assert rows[0]["steps"] == 2
    assert rows[0]["train_flops"] == 1
    assert rows[0]["quality"]["val_accuracy"] == 0.5
    assert rows[0]["config"]["schedule"]["train_flops"] == 1


def test_the_checkpoint_loads_back_into_a_model(prepared: Path, tmp_path: Path) -> None:
    backup = BackupTracker(tmp_path)
    backup.start(train_config(prepared))
    backup.finish(train_result(train_config(prepared)), _plain_model())

    state = torch.load(tmp_path / "model.pt", weights_only=True)
    rebuilt = GPT2(gpt2_config(n_layer=1))
    rebuilt.load_state_dict(state)
