"""Локальная страховка запуска: metrics.jsonl и чекпоинт модели рядом с ним."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from compare_attention.models.gpt2 import GPT2
from compare_attention.train.config import TrainConfig
from compare_attention.train.result import TrainResult


class BackupTracker:
    """Пишет потери по шагам, итог и веса модели в каталог результата."""

    def __init__(self, result_dir: Path) -> None:
        self._result_dir = result_dir

    def start(self, _config: TrainConfig) -> None:
        """Готовим каталог результата."""
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def log_loss(self, loss: float, flops: int) -> None:
        """Каждый шаг обучения отражается строкой в metrics.jsonl."""
        row = {"type": "step", "loss": loss, "flops": flops}
        with (self._result_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self, result: TrainResult, model: GPT2) -> None:
        """Итог запуска и веса модели остаются рядом с метриками."""
        row = {"type": "result", **result.model_dump(mode="json")}
        with (self._result_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        torch.save(model.state_dict(), self._result_dir / "model.pt")
