"""Итог одного запуска: качество и скорость обучения."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from compare_attention.train.config import TrainConfig

_FROZEN = ConfigDict(frozen=True)


class Quality(BaseModel):
    """Loss обучения и качество на validation."""

    model_config = _FROZEN

    train_loss: float
    val_loss: float
    val_accuracy: float


class Speed(BaseModel):
    """Скорость учебных шагов и пиковая память GPU."""

    model_config = _FROZEN

    train_tokens_per_second: float
    peak_gpu_bytes: int | None


class TrainResult(BaseModel):
    """Сколько шагов уложилось в бюджет и чем это закончилось."""

    model_config = _FROZEN

    steps: int
    train_flops: int
    quality: Quality
    speed: Speed
    config: TrainConfig
    stop_reason: Literal["budget_exhausted", "early_stopping"] = "budget_exhausted"
    early_stopping: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)

    def __str__(self) -> str:
        lines = (
            f"steps={self.steps}",
            f"stop_reason={self.stop_reason}",
            f"train_flops={self.train_flops}",
            f"train_loss={self.quality.train_loss:.4f}",
            f"val_loss={self.quality.val_loss:.4f}",
            f"val_acc={self.quality.val_accuracy:.3f}",
            f"train_tok/s={self.speed.train_tokens_per_second:.0f}",
            f"peak_gpu_bytes={self.speed.peak_gpu_bytes}",
        )
        return " ".join(lines)
