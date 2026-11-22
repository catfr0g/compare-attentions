"""Real CPU native save and fake child process: no credentials/network/model updates."""

from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_attention.train.backup import BackupTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_native_cpu_save_child_isolation_rng_and_scoped_restore(tmp_path, monkeypatch):
    module = importlib.import_module("colab_checkpoint_backup")

    original = BackupTracker.save_checkpoint
    calls = []
    result = tmp_path / "results"
    tracker = BackupTracker(result)
    monkeypatch.setenv("HF_TOKEN", "forbidden-parent-token")
    monkeypatch.setenv("CLEARML_API_ACCESS_KEY", "forbidden-parent-clearml")
    monkeypatch.setenv("KAGGLE_KEY", "forbidden-parent-kaggle")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("PYTHONPATH", "forbidden-pythonpath")

    def child(command, **kwargs):
        assert (result / "checkpoints/step_00000125.pt").is_file()
        assert "forbidden" not in str(command) + str(kwargs)
        assert not any(k.startswith(("CLEARML_", "KAGGLE_")) for k in kwargs["env"])
        assert "HF_TOKEN" not in kwargs["env"]
        assert "PYTHONPATH" not in kwargs["env"] and "PYTHONOPTIMIZE" not in kwargs["env"]
        assert kwargs["check"] and kwargs["stdout"] == module.subprocess.DEVNULL
        assert kwargs["stderr"] == module.subprocess.DEVNULL
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", child)
    before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        dict(os.environ),
    )
    with module.checkpoint_backups(tmp_path / "private.env", tmp_path / "package"):
        tracker.save_checkpoint(
            {"step": 125, "optimizer": {}, "rng": torch.get_rng_state()}, 125, 2
        )
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state(), dict(os.environ))
    assert before[0] == after[0] and np.array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2]) and before[3] == after[3]
    assert BackupTracker.save_checkpoint is original
    assert len(calls) == 1 and calls[0][:3] == [
        sys.executable,
        str(Path(module.__file__).resolve()),
        "publish",
    ]

    def fail(*_args, **_kwargs):
        raise RuntimeError("private-token-must-not-escape")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Checkpoint backup failed") as error:
        with module.checkpoint_backups(tmp_path / "private.env", tmp_path / "package"):
            tracker.save_checkpoint({"step": 250}, 250, 2)
    assert "private-token" not in str(error.value)
    assert (result / "checkpoints/step_00000250.pt").is_file()
    assert BackupTracker.save_checkpoint is original


def test_native_failure_never_launches_child(tmp_path, monkeypatch):
    module = importlib.import_module("colab_checkpoint_backup")

    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    with module.checkpoint_backups(tmp_path / "private.env", tmp_path):
        with pytest.raises(ValueError):
            BackupTracker(tmp_path / "r").save_checkpoint({}, -1, 2)
    assert not calls


def test_frozen_loop_excludes_synchronous_child_io_from_train_elapsed(tmp_path, monkeypatch):
    """Exercise native timing boundary without constructing/training any model."""
    module = importlib.import_module("colab_checkpoint_backup")
    loop = importlib.import_module("compare_attention.train.loop")
    clock = [0.0]
    result = tmp_path / "result"
    result.mkdir()
    tracker = BackupTracker(result)
    control = SimpleNamespace(
        flops=0,
        cursor=0,
        losses=[],
        elapsed=0.0,
        stop_reason="budget",
        validation=lambda *_a, **_k: False,
    )
    config = SimpleNamespace(
        optim=SimpleNamespace(lr_schedule="constant"),
        schedule=SimpleNamespace(
            train_flops=2, batch_size=1, log_every_steps=100, val_every_steps=100
        ),
    )

    def tracked_step(*args):
        clock[0] += 1.0
        return 0.5, args[-1] + 1

    def child(*_a, **_k):
        clock[0] += 10.0

    def checkpoint(**_kwargs):
        step = len(control.losses)
        tracker.save_checkpoint({"step": step}, step, 2)

    control.checkpoint = checkpoint
    monkeypatch.setattr(loop.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(loop, "_tracked_step", tracked_step)
    monkeypatch.setattr(loop, "_evaluate", lambda *_a: SimpleNamespace())
    monkeypatch.setattr(loop, "_log_progress", lambda *_a: None)
    monkeypatch.setattr(loop, "next_learning_rate", lambda *_a: 0.001)
    monkeypatch.setattr(module.subprocess, "run", child)
    with module.checkpoint_backups(tmp_path / "private.env", tmp_path / "package"):
        elapsed = loop._fit(  # pylint: disable=protected-access
            config, None, None, (None, None, None, None), tracker, 1, control
        )[0]
    assert elapsed == 2.0 and clock[0] == 22.0
    assert control.losses == [0.5, 0.5] and control.flops == 2
