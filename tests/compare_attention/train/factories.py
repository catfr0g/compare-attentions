"""Маленькие конфиги и результаты обучения для тестов."""

from pathlib import Path

from compare_attention.models.config import GPT2Config
from compare_attention.train.config import Optim, Schedule, TrainConfig
from compare_attention.train.result import Quality, Speed, TrainResult
from tests.compare_attention.models.factories import gpt2_config

_BASE_SCHEDULE = {
    "train_flops": 1,
    "batch_size": 2,
    "seed": 0,
    "compile": False,
}


def train_config(
    path: Path,
    *,
    model: GPT2Config | None = None,
    **schedule_overrides: object,
) -> TrainConfig:
    """Конфиг одного крошечного запуска."""
    schedule = {**_BASE_SCHEDULE, **schedule_overrides}
    return TrainConfig(
        data=path,
        model=model or gpt2_config(n_layer=1),
        schedule=Schedule(**schedule),
        optim=Optim(learning_rate=1e-2, weight_decay=0.0),
    )


def train_result(
    config: TrainConfig,
    *,
    steps: int = 1,
    quality: Quality | None = None,
    speed: Speed | None = None,
) -> TrainResult:
    """Итог крошечного запуска в одну ступень."""
    return TrainResult(
        steps=steps,
        train_flops=1,
        quality=quality or Quality(train_loss=1.0, val_loss=1.0, val_accuracy=0.5),
        speed=speed or Speed(train_tokens_per_second=1.0, peak_gpu_bytes=None),
        config=config,
    )
