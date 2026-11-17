# pylint: disable=redefined-outer-name
"""Локальные артефакты запуска: metrics.jsonl и чекпоинт модели."""

import hashlib
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


def test_periodic_checkpoint_publishes_verified_latest_and_retains_only_owned(tmp_path):
    backup = BackupTracker(tmp_path)
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    unrelated = folder / "notes.pt"
    unrelated.write_bytes(b"keep")
    for step in (1, 2, 3):
        assert (
            backup.save_checkpoint({"step": step, "weight": torch.tensor([step])}, step, 2) is None
        )
    latest = json.loads((folder / "latest.json").read_text())
    assert latest == {
        "step": 3,
        "path": "step_00000003.pt",
        "sha256": hashlib.sha256((folder / "step_00000003.pt").read_bytes()).hexdigest(),
    }
    assert not (folder / "step_00000001.pt").exists()
    assert (folder / "step_00000002.pt").exists()
    assert unrelated.read_bytes() == b"keep"
    assert torch.load(folder / latest["path"], weights_only=True)["weight"].item() == 3


@pytest.mark.parametrize("failure", ["save", "payload_replace", "latest_replace"])
def test_interrupted_save_preserves_previous_latest_and_files(tmp_path, monkeypatch, failure):
    backup = BackupTracker(tmp_path)
    backup.save_checkpoint({"step": 1}, 1, 1)
    folder = tmp_path / "checkpoints"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    latest = (folder / "latest.json").read_bytes()
    original_save = torch.save
    original_replace = Path.replace

    def save(*args, **kwargs):
        if failure == "save":
            args[1].write(b"interrupted")
            raise OSError("interrupted")
        return original_save(*args, **kwargs)

    def replace(self, target):
        if (failure == "payload_replace" and Path(target).suffix == ".pt") or (
            failure == "latest_replace" and Path(target).name == "latest.json"
        ):
            raise OSError("interrupted")
        return original_replace(self, target)

    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="interrupted"):
        backup.save_checkpoint({"step": 2}, 2, 1)
    assert (folder / "latest.json").read_bytes() == latest
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert not list(folder.glob("*.tmp"))


def test_retention_survives_tracker_restart_without_deleting_foreign_named_files(tmp_path):
    backup = BackupTracker(tmp_path)
    backup.save_checkpoint({"step": 1}, 1, 2)
    folder = tmp_path / "checkpoints"
    foreign = folder / "step_00000000.pt"
    torch.save({"step": 0}, foreign)
    BackupTracker(tmp_path).save_checkpoint({"step": 2}, 2, 1)
    assert not (folder / "step_00000001.pt").exists()
    assert foreign.exists()


def test_checkpoint_rejects_rollback_step(tmp_path):
    backup = BackupTracker(tmp_path)
    backup.save_checkpoint({"step": 2}, 2, 1)
    with pytest.raises(ValueError, match="step"):
        backup.save_checkpoint({"step": 1}, 1, 1)


def test_latest_prevents_rollback_even_if_ownership_manifest_is_missing(tmp_path):
    backup = BackupTracker(tmp_path)
    backup.save_checkpoint({"step": 2}, 2, 1)
    (tmp_path / "checkpoints" / ".owned.json").unlink()
    with pytest.raises(ValueError, match="step"):
        backup.save_checkpoint({"step": 1}, 1, 1)


def test_prune_failure_does_not_misreport_committed_save_as_failed(tmp_path, monkeypatch, caplog):
    backup = BackupTracker(tmp_path)
    backup.save_checkpoint({"step": 1}, 1, 1)
    original = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == "step_00000001.pt":
            raise OSError("file busy")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    backup.save_checkpoint({"step": 2}, 2, 1)
    latest = json.loads((tmp_path / "checkpoints" / "latest.json").read_text())
    assert latest["step"] == 2
    assert "retention" in caplog.text
