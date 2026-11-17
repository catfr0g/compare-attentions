"""A genuine GradScaler overflow must not publish attempted LR/FLOPs progress."""

# Inject overflow at the actual training update boundary, not a fake scaler.
# pylint: disable=protected-access

import pytest
import torch

from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.flops import estimate_step
from tests.compare_attention.train.test_lr_training import metric_rows, tiny_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA GradScaler")
def test_scheduled_overflow_aborts_without_advancing_successful_lr_history(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"dtype": "float16", "amp_max_scale": 128.0}
            )
        }
    )
    original = loop._step
    attempts = 0

    def poison_second_gradient(model, optimizer, inputs, labels):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return original(model, optimizer, inputs, labels)

    monkeypatch.setattr(loop, "_step", poison_second_gradient)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="overflow.*skipped"):
        loop.train(config, BackupTracker(output))
    rows = metric_rows(output)
    assert [r["flops"] for r in rows if r["type"] == "step"] == [estimate_step(config).total]
    assert [r["flops"] for r in rows if r["type"] == "learning_rate"] == [
        estimate_step(config).total
    ]
    state = load_checkpoint(output / "checkpoints/latest.json")
    assert state["step"] == 1
    assert state["precision"]["optimizer_steps"] == 1
    assert not (output / "model.pt").exists()
