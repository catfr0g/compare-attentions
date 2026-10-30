# pylint: disable=redefined-outer-name,too-few-public-methods
"""Трекер обучения."""

from pathlib import Path

import pytest

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import TrainConfig
from compare_attention.train.result import Quality, Speed
from compare_attention.train.tracker import (
    ClearMLTracker,
    DisabledTracker,
    GroupTracker,
    _init_task,
    select,
)
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result

PROJECT_NAME = "compare-attention"


class RecordingLogger:
    """Записывает скаляры вместо отправки в ClearML."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, str, float, int]] = []

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.scalars.append((title, series, value, iteration))


class RecordingTask:
    """Подменяет задачу ClearML и запоминает всё, что в неё уходит."""

    id = "task-id-1"

    def __init__(self, init_params: dict[str, str]) -> None:
        self.init_params = init_params
        self.connected: object | None = None
        self.logger = RecordingLogger()
        self.closed = False

    def connect(self, payload: object) -> None:
        self.connected = payload

    def get_logger(self) -> RecordingLogger:
        return self.logger

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def created_tasks() -> list[RecordingTask]:
    return []


@pytest.fixture
def task_factory(created_tasks: list[RecordingTask]):
    def factory(_config: TrainConfig) -> RecordingTask:
        task = RecordingTask({"project_name": PROJECT_NAME})
        created_tasks.append(task)
        return task

    return factory


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


def _result(config: TrainConfig, steps: int = 3):
    return train_result(
        config,
        steps=steps,
        quality=Quality(train_loss=0.5, val_loss=0.4, val_accuracy=0.75),
        speed=Speed(train_tokens_per_second=100.0, peak_gpu_bytes=None),
    )


def test_default_factory_names_the_task(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_init(**kwargs: object) -> RecordingTask:
        calls.append(kwargs)
        return RecordingTask({key: str(value) for key, value in kwargs.items()})

    config = train_config(prepared)
    monkeypatch.setattr("compare_attention.train.tracker.Task.init", fake_init)

    _init_task(config)

    assert len(calls) == 1
    assert calls[0]["project_name"] == PROJECT_NAME
    assert calls[0]["task_name"] == f"{config.model.attention} flops={config.schedule.train_flops}"


def test_clearml_tracker_exposes_the_task_id(prepared: Path, task_factory) -> None:
    tracker = ClearMLTracker(task_factory=task_factory)

    tracker.start(train_config(prepared))

    assert tracker.task_id == "task-id-1"


def test_clearml_tracker_connects_the_config(
    prepared: Path, task_factory, created_tasks: list[RecordingTask]
) -> None:
    tracker = ClearMLTracker(task_factory=task_factory)

    tracker.start(train_config(prepared))

    connected = created_tasks[0].connected
    assert connected["schedule"]["train_flops"] == 1
    assert connected["model"]["attention"] == "classic"
    assert isinstance(connected["data"], str)


def test_clearml_tracker_reports_per_step_loss(
    prepared: Path, task_factory, created_tasks: list[RecordingTask]
) -> None:
    tracker = ClearMLTracker(task_factory=task_factory)
    tracker.start(train_config(prepared))

    tracker.log_loss(0.5, 128)
    tracker.log_loss(0.25, 256)

    scalars = created_tasks[0].logger.scalars
    assert scalars == [
        ("train", "loss", 0.5, 128),
        ("train", "loss", 0.25, 256),
    ]


def test_clearml_tracker_reports_the_summary(
    prepared: Path, task_factory, created_tasks: list[RecordingTask]
) -> None:
    config = train_config(prepared)
    tracker = ClearMLTracker(task_factory=task_factory)
    tracker.start(config)
    task = created_tasks[0]

    tracker.finish(_result(config, steps=3), GPT2(gpt2_config(n_layer=1)))

    seen = {
        (title, series): (value, iteration)
        for title, series, value, iteration in task.logger.scalars
    }
    assert seen[("result", "steps")] == (3, 3)
    assert seen[("result", "train_loss")] == (0.5, 3)
    assert seen[("result", "val_loss")] == (0.4, 3)
    assert seen[("result", "val_accuracy")] == (0.75, 3)
    assert seen[("result", "train_tokens_per_second")] == (100.0, 3)
    assert ("result", "peak_gpu_bytes") not in seen


def test_clearml_tracker_closes_the_task(
    prepared: Path, task_factory, created_tasks: list[RecordingTask]
) -> None:
    config = train_config(prepared)
    tracker = ClearMLTracker(task_factory=task_factory)
    tracker.start(config)
    task = created_tasks[0]

    tracker.finish(_result(config, steps=3), GPT2(gpt2_config(n_layer=1)))

    assert task.closed


def test_select_maps_the_kinds() -> None:
    assert isinstance(select("clearml"), ClearMLTracker)
    assert isinstance(select("disabled"), DisabledTracker)


def test_select_rejects_the_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        select("tensorboard")


def test_group_tracker_drives_all_trackers(
    prepared: Path,
    task_factory,
    created_tasks: list[RecordingTask],
    tmp_path: Path,
) -> None:
    config = train_config(prepared)
    backup = BackupTracker(tmp_path)
    group = GroupTracker(ClearMLTracker(task_factory=task_factory), backup)
    model = GPT2(gpt2_config(n_layer=1))

    group.start(config)
    group.log_loss(0.5, 128)
    group.finish(train_result(config), model)

    assert created_tasks[0].closed
    rows = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert (tmp_path / "model.pt").is_file()
