"""Отражаем ход обучения в ClearML и сводим несколько трекеров вместе."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from clearml import Task

from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import TrainConfig
from compare_attention.train.identity import record_identity
from compare_attention.train.result import TrainResult

_PROJECT = "compare-attention"


class ScalarLogger(Protocol):  # pylint: disable=too-few-public-methods
    """Минимум логгера ClearML, нужный шагам обучения."""

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None: ...


class TrackedTask(Protocol):
    """Минимум задачи ClearML, нужный трекеру."""

    def connect(self, payload: object) -> None: ...
    def get_logger(self) -> ScalarLogger: ...
    def close(self) -> None: ...


TaskFactory = Callable[[TrainConfig], TrackedTask]


def _init_task(config: TrainConfig) -> TrackedTask:
    """Завести задачу, имя которой показывает модель и бюджет."""
    name = f"{config.model.attention} flops={config.schedule.train_flops}"
    return Task.init(
        project_name=_PROJECT,
        task_name=name,
        task_type=Task.TaskTypes.training,
    )


class DisabledTracker:
    """Трекер-пустышка для тестов и запусков без облака."""

    def start(self, _config: TrainConfig) -> None:
        pass

    def log_loss(self, _loss: float, _flops: int) -> None:
        pass

    def log_validation(self, _loss: float, _accuracy: float, _flops: int) -> None:
        pass

    def finish(self, _result: TrainResult, _model: torch.nn.Module) -> None:
        pass


class ClearMLTracker:
    """Пишет конфиг, потерьной прогресс и итог в задачу ClearML."""

    def __init__(self, task_factory: TaskFactory | None = None) -> None:
        self._make_task = task_factory or _init_task
        self._task: TrackedTask | None = None
        self._logger: ScalarLogger | None = None

    @property
    def task_id(self) -> str | None:
        """Идентификатор задачи, чтобы найти её на сервере после закрытия."""
        if self._task is None:
            return None
        return getattr(self._task, "id", None)

    def start(self, config: TrainConfig) -> None:
        """Создать задачу и прицепить к ней конфиг запуска."""
        self._task = self._make_task(config)
        self._task.connect(config.model_dump(mode="json"))
        self._logger = self._task.get_logger()

    def log_loss(self, loss: float, flops: int) -> None:
        """Отметить потерю шага; по оси x идёт количество FLOPs."""
        if self._task is None or self._logger is None:
            raise RuntimeError("start() must be called first")
        self._logger.report_scalar(title="train", series="loss", value=loss, iteration=flops)

    def log_learning_rate(self, learning_rate: float, flops: int) -> None:
        """Use the same exact successful-update FLOPs axis as train loss."""
        if self._task is None or self._logger is None:
            raise RuntimeError("start() must be called first")
        self._logger.report_scalar(
            title="train", series="learning_rate", value=learning_rate, iteration=flops
        )

    def log_validation(self, loss: float, accuracy: float, flops: int) -> None:
        """Качество на validation по той же оси FLOPs, что и train loss."""
        if self._task is None or self._logger is None:
            raise RuntimeError("start() must be called first")
        self._logger.report_scalar(title="validation", series="loss", value=loss, iteration=flops)
        self._logger.report_scalar(
            title="validation", series="accuracy", value=accuracy, iteration=flops
        )

    def finish(self, result: TrainResult, _model: torch.nn.Module) -> None:
        """Отметить итог и закрыть задачу."""
        if self._task is None or self._logger is None:
            raise RuntimeError("start() must be called first")
        serieses = {
            "steps": float(result.steps),
            "train_flops": float(result.train_flops),
            "requested_train_flops": float(result.config.schedule.train_flops),
            "early_stopped": float(result.stop_reason == "early_stopping"),
            "budget_exhausted": float(result.train_flops >= result.config.schedule.train_flops),
            "train_loss": result.quality.train_loss,
            "val_loss": result.quality.val_loss,
            "val_accuracy": result.quality.val_accuracy,
            "train_tokens_per_second": result.speed.train_tokens_per_second,
            "peak_gpu_bytes": result.speed.peak_gpu_bytes,
        }
        for name, value in serieses.items():
            if value is None:
                continue
            self._logger.report_scalar(
                title="result", series=name, value=value, iteration=result.steps
            )
        self._task.close()
        self._task = None
        self._logger = None


type Tracker = DisabledTracker | ClearMLTracker | BackupTracker | GroupTracker


class GroupTracker:
    """Направляет события обучения сразу нескольким трекерам."""

    def __init__(self, *trackers: Tracker) -> None:
        self._trackers = trackers

    def start(self, config: TrainConfig) -> None:
        for tracker in self._trackers:
            tracker.start(config)

    def validate_resume_destination(self) -> None:
        """Preflight every nested persistent sink before any tracker starts."""
        for tracker in self._trackers:
            validate_resume_destination(tracker)

    def save_checkpoint(self, payload: dict, step: int, keep_last: int) -> None:
        """Forward to optional sinks; fail closed if no sink accepts checkpoints."""
        sinks = [getattr(tracker, "save_checkpoint", None) for tracker in self._trackers]
        sinks = [sink for sink in sinks if callable(sink)]
        if not sinks:
            raise RuntimeError("checkpoint enabled but no accepting checkpoint sink")
        for sink in sinks:
            sink(payload, step, keep_last)

    def record_identity(self, identity: dict[str, object]) -> None:
        """Forward optional provenance events without requiring legacy trackers to change."""
        for tracker in self._trackers:
            record_identity(tracker, identity)

    def log_loss(self, loss: float, flops: int) -> None:
        for tracker in self._trackers:
            tracker.log_loss(loss, flops)

    def log_learning_rate(self, learning_rate: float, flops: int) -> None:
        """Fan out optional LR callbacks, retaining legacy/custom tracker support."""
        for tracker in self._trackers:
            callback = getattr(tracker, "log_learning_rate", None)
            if callable(callback):
                callback(learning_rate, flops)

    def log_validation(self, loss: float, accuracy: float, flops: int) -> None:
        for tracker in self._trackers:
            tracker.log_validation(loss, accuracy, flops)

    def finish(self, result: TrainResult, model: torch.nn.Module) -> None:
        for tracker in self._trackers:
            tracker.finish(result, model)


def validate_resume_destination(tracker: Tracker) -> None:
    """Run an optional sink preflight without initializing trackers or touching RNG."""
    validate = getattr(tracker, "validate_resume_destination", None)
    if callable(validate):
        validate()


def select(kind: str) -> Tracker:
    """Выбрать трекер по имени из CLI."""
    kinds: dict[str, type[DisabledTracker] | type[ClearMLTracker]] = {
        "clearml": ClearMLTracker,
        "disabled": DisabledTracker,
    }
    if kind not in kinds:
        raise ValueError(f"unknown tracker kind {kind!r}")
    return kinds[kind]()
