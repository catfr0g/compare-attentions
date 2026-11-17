"""Real offline ClearML serialization and optional LR fanout through nested trackers."""

import pytest
import torch
from clearml.backend_api.session.session import Session
from clearml.backend_interface.base import InterfaceBase

from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.tracker import ClearMLTracker, DisabledTracker, GroupTracker
from tests.compare_attention.train.test_lr_training import metric_rows, tiny_config
from tests.integration.test_clearml_saving import _jsonl, _scalar_tuples, _task_params


@pytest.mark.integration
def test_real_trainer_serializes_lr_in_offline_clearml(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    cache = tmp_path / "cache"
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    monkeypatch.setenv("CLEARML_CACHE_DIR", str(cache))
    monkeypatch.setattr(Session, "_offline_mode", True)
    monkeypatch.setattr(InterfaceBase, "_offline_mode", True)
    config = tiny_config(tmp_path, steps=4)
    clearml = ClearMLTracker()
    ids = []
    original_start = clearml.start

    def start(config):
        original_start(config)
        ids.append(clearml.task_id)

    monkeypatch.setattr(clearml, "start", start)
    output = tmp_path / "output"
    loop.train(
        config, GroupTracker(DisabledTracker(), GroupTracker(clearml, BackupTracker(output)))
    )
    folder = cache / "offline" / ids[0]
    scalars = _scalar_tuples(_jsonl(folder / "metrics.jsonl"))
    expected = {
        (row["learning_rate"], row["flops"])
        for row in metric_rows(output)
        if row["type"] == "learning_rate"
    }
    assert expected
    actual = {
        (value, axis)
        for title, series, value, axis in scalars
        if title == "train" and series == "learning_rate"
    }
    assert actual == expected
    assert _task_params(folder)["optim/lr_schedule"] == "warmup_cosine"
