"""Цикл обучения до бюджета FLOPs и базовые метрики запуска."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

from compare_attention.data.config import IGNORE_INDEX
from compare_attention.data.store import load_split
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.config import TrainConfig
from compare_attention.train.result import Quality, Speed, TrainResult

logger = logging.getLogger(__name__)


def train(config: TrainConfig) -> TrainResult:
    """Учить, пока сумма FLOPs шагов не достигнет бюджета."""
    _seed(config.schedule.seed)
    train_x, train_y = _load(config.data, "train")
    val_x, val_y = _load(config.data, "val")
    model = GPT2(config.model)
    dtype = config.schedule.torch_dtype()
    device = _device()
    model.to(device=device, dtype=dtype)
    compiled = _compile(model, config.schedule.compile)
    moved = tuple(tensor.to(device) for tensor in (train_x, train_y, val_x, val_y))
    result = _run(config, compiled, moved, device)
    logger.info("%s", result)
    return result


def _run(
    config: TrainConfig,
    model: torch.nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
) -> TrainResult:
    train_x, train_y, val_x, val_y = tensors
    optimizer = torch.optim.AdamW(
        _unique_parameters(model),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elapsed, losses, spent = _fit(config, model, optimizer, train_x, train_y)
    quality = _quality(model, val_x, val_y, config.schedule.batch_size, losses)
    speed = _measured_speed(train_x, config, device, elapsed, len(losses))
    return TrainResult(
        steps=len(losses),
        train_flops=spent,
        quality=quality,
        speed=speed,
        config=config,
    )


def _measured_speed(
    inputs: Tensor,
    config: TrainConfig,
    device: torch.device,
    elapsed: float,
    steps: int,
) -> Speed:
    return Speed(
        train_tokens_per_second=_tokens_per_second(
            steps,
            config.schedule.batch_size,
            int(inputs.size(1)),
            elapsed,
        ),
        peak_gpu_bytes=_peak(device),
    )


def _fit(
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    labels: Tensor,
) -> tuple[float, list[float], int]:
    started = time.perf_counter()
    losses: list[float] = []
    cursor = 0
    counter = FlopCounterMode(display=False)
    progress = tqdm(
        total=config.schedule.train_flops,
        desc="train",
        unit="FLOP",
        unit_scale=True,
        file=sys.stderr,
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    with counter, progress:
        while counter.get_total_flops() < config.schedule.train_flops:
            batch_x, batch_y, cursor = _rows(inputs, labels, config.schedule.batch_size, cursor)
            before = counter.get_total_flops()
            losses.append(_step(model, optimizer, batch_x, batch_y))
            step_flops = counter.get_total_flops() - before
            if step_flops <= 0:
                raise RuntimeError("training step recorded no flops")
            progress.update(step_flops)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started, losses, counter.get_total_flops()


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    labels: Tensor,
) -> float:
    model.train()
    loss = _loss(model(inputs), labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _quality(
    model: torch.nn.Module,
    inputs: Tensor,
    labels: Tensor,
    batch_size: int,
    step_losses: list[float],
) -> Quality:
    val_loss, accuracy = _validate(model, inputs, labels, batch_size)
    return Quality(
        train_loss=sum(step_losses) / len(step_losses),
        val_loss=val_loss,
        val_accuracy=accuracy,
    )


def _validate(
    model: torch.nn.Module,
    inputs: Tensor,
    labels: Tensor,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    supervised = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            batch_x = inputs[start : start + batch_size]
            batch_y = labels[start : start + batch_size]
            logits = model(batch_x)
            loss_sum += float(_loss(logits, batch_y, reduction="sum"))
            mask = batch_y != IGNORE_INDEX
            supervised += int(mask.sum())
            correct += int((logits.argmax(dim=-1).eq(batch_y) & mask).sum())
    val_loss = loss_sum / supervised if supervised else 0.0
    accuracy = correct / supervised if supervised else 0.0
    return val_loss, accuracy


def _loss(logits: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


def _rows(inputs: Tensor, labels: Tensor, batch: int, start: int) -> tuple[Tensor, Tensor, int]:
    count = inputs.size(0)
    index = (torch.arange(batch, device=inputs.device) + start) % count
    return inputs.index_select(0, index), labels.index_select(0, index), (start + batch) % count


def _load(path: Path, split: str) -> tuple[Tensor, Tensor]:
    loaded = load_split(path, split)
    return torch.from_numpy(loaded.inputs), torch.from_numpy(loaded.labels)


def _compile(model: GPT2, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    return torch.compile(model)


def _unique_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    chosen: list[torch.nn.Parameter] = []
    for param in model.parameters():
        identity = param.data_ptr()
        if identity in seen:
            continue
        seen.add(identity)
        chosen.append(param)
    return chosen


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peak(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def _tokens_per_second(steps: int, batch: int, seq: int, elapsed: float) -> float:
    return steps * batch * seq / elapsed
