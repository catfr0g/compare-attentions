"""Цикл обучения до бюджета FLOPs и базовые метрики запуска."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from compare_attention.data.config import IGNORE_INDEX
from compare_attention.data.store import load_split
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.compiler import compilation_phase, compile_model
from compare_attention.train.config import TrainConfig
from compare_attention.train.control import TrainingControl
from compare_attention.train.distributed import (
    current,
    finalize_evidence,
    initialize_evidence,
    local_call,
    run_training,
)
from compare_attention.train.distributed import step as distributed_step
from compare_attention.train.flops import estimate_step
from compare_attention.train.identity import (
    initialized_identity,
    parameter_fingerprint,
    record_identity,
)
from compare_attention.train.lr_schedule import log_learning_rate, next_learning_rate
from compare_attention.train.precision import Precision, autocast, current_precision
from compare_attention.train.result import Quality, Speed, TrainResult
from compare_attention.train.tracker import DisabledTracker, Tracker, validate_resume_destination

logger = logging.getLogger(__name__)


def train(
    config: TrainConfig,
    tracker: Tracker | None = None,
    *,
    resume_from: Path | None = None,
    distributed: bool = False,
) -> TrainResult:
    """Use the same trainer with explicitly enabled single-node torchrun execution."""
    return run_training(config, tracker, resume_from, distributed, _train)


def _train(
    config: TrainConfig, tracker: Tracker | None = None, *, resume_from: Path | None = None
) -> TrainResult:
    """Учить, пока сумма FLOPs шагов не достигнет бюджета."""
    # pylint: disable=too-many-locals
    mirror = tracker if tracker is not None else DisabledTracker()
    if resume_from is not None:
        validate_resume_destination(mirror)
    mirror.start(config)
    train_x, train_y = local_call(lambda: _load(config.data, "train"), "load_train")
    val_x, val_y = local_call(lambda: _load(config.data, "val"), "load_val")
    local_call(
        lambda: _validate_loaded_splits(config, (train_x, train_y), (val_x, val_y)),
        "validate_splits",
    )
    # Task.init() reseeds torch; apply the requested seed after every tracker starts.
    _seed(config.schedule.seed)
    effective_seed = torch.initial_seed()
    model = local_call(lambda: GPT2(config.model), "construct_model")
    step_cost = estimate_step(config)
    if step_cost.total <= 0:
        raise ValueError("analytical training step FLOPs must be positive")
    identity = initialized_identity(config, model, effective_seed)
    identity["recorded_flops"] = step_cost.to_dict()
    identity["accounting"] = step_cost.to_dict()
    if current() is not None:
        initialize_evidence(config, identity)
    device = _device()
    precision = None
    if config.schedule.dtype == "float16":
        precision = (
            Precision(device, max_scale=config.schedule.amp_max_scale)
            if config.schedule.amp_max_scale is not None
            else Precision(device)
        )
    if precision is not None:
        identity["precision"] = precision.metadata()
    record_identity(mirror, identity)
    logger.info(
        "train start attention=%s device=%s dtype=%s compile=%s "
        "train_samples=%d val_samples=%d batch_size=%d train_flops=%d",
        config.model.attention,
        device,
        config.schedule.torch_dtype(),
        config.schedule.compile,
        train_x.size(0),
        val_x.size(0),
        config.schedule.batch_size,
        config.schedule.train_flops,
    )
    parameter_dtype = torch.float32 if precision is not None else config.schedule.torch_dtype()
    local_call(lambda: model.to(device=device, dtype=parameter_dtype), "move_model")
    wrapped = current().wrap(model) if current() is not None else model
    compiled = local_call(
        lambda: _compile(wrapped, config.schedule.compile, identity["compiler"]), "compile_model"
    )
    moved = local_call(
        lambda: tuple(tensor.to(device) for tensor in (train_x, train_y, val_x, val_y)),
        "move_splits",
    )
    # Dynamo config exposes patch dynamically; force compilation errors to propagate.
    try:
        with (
            torch._dynamo.config.patch(suppress_errors=False),  # pylint: disable=protected-access,no-member
            precision.activate() if precision is not None else nullcontext(),
        ):
            result = _run(
                config, compiled, moved, device, mirror, step_cost.total, identity, resume_from
            )
            if precision is not None and (
                precision.optimizer_steps != result.steps or precision.skipped_optimizer_steps
            ):
                raise RuntimeError("CUDA AMP optimizer update count does not match completed steps")
    except Exception as error:
        identity.update(status="failed", failure=str(error))
        if precision is not None:
            identity["precision"] = precision.metadata()
        record_identity(mirror, identity)
        raise
    if precision is not None:
        identity["precision"] = precision.metadata()
    if (
        config.schedule.compile
        and not identity["compiler"]["successful_training_backend_invocations"]
    ):
        raise RuntimeError("compile requested but no successful Inductor training backend return")
    status = "early_stopped" if result.stop_reason == "early_stopping" else "completed"
    identity.update(
        status=status,
        stop_reason=result.stop_reason,
        final_parameters_sha256=parameter_fingerprint(model),
    )
    if current() is not None:
        finalize_evidence(identity)
    result = result.model_copy(update={"provenance": identity.copy()})
    mirror.finish(result, model)
    record_identity(mirror, identity)
    logger.info("%s", result)
    return result


def _validate_loaded_splits(
    config: TrainConfig, train_pair: tuple[Tensor, Tensor], val_pair: tuple[Tensor, Tensor]
) -> None:
    """Reject arrays whose actual shape or dtype differs from the priced batch contract."""
    for split, pair in (("train", train_pair), ("val", val_pair)):
        expected = (getattr(config.dataset, f"n_{split}"), config.dataset.input_seq_len)
        for kind, tensor in zip(("inputs", "labels"), pair, strict=True):
            if tuple(tensor.shape) != expected or tensor.dtype != torch.int64:
                raise ValueError(
                    f"loaded {split} {kind} shape/dtype mismatch: "
                    f"expected {expected}/int64, got {tuple(tensor.shape)}/{tensor.dtype}"
                )


def _run(
    config: TrainConfig,
    model: torch.nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    device: torch.device,
    tracker: Tracker,
    step_flops: int,
    identity: dict | None = None,
    resume_from: Path | None = None,
) -> TrainResult:
    # The run binds measured speed, stopping and checkpoint provenance together.
    # pylint: disable=too-many-locals
    train_x = tensors[0]
    optimizer = torch.optim.AdamW(
        _unique_parameters(model),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    control = TrainingControl(config, model, optimizer, tracker, identity or {}, resume_from)
    elapsed, losses, spent, quality = _fit(
        config, model, optimizer, tensors, tracker, step_flops, control
    )
    speed = _measured_speed(train_x, config, device, elapsed, len(losses))
    return TrainResult(
        steps=len(losses),
        train_flops=spent,
        quality=quality,
        speed=speed,
        config=config,
        stop_reason=control.stop_reason,
        early_stopping=control.stopper.state_dict(),
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
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    tracker: Tracker,
    step_flops: int,
    control: TrainingControl,
) -> tuple[float, list[float], int, Quality]:
    quality = None
    progress = tqdm(
        total=config.schedule.train_flops,
        initial=control.flops,
        desc="train",
        unit="FLOP",
        unit_scale=True,
        file=sys.stderr,
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    with progress:
        while control.flops < config.schedule.train_flops:
            started = time.perf_counter()
            learning_rate = next_learning_rate(
                config.optim, config.schedule.train_flops, control.flops, step_flops
            )
            if config.optim.lr_schedule != "constant":
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            loss, control.cursor = _tracked_step(
                model,
                optimizer,
                tensors[0],
                tensors[1],
                config.schedule.batch_size,
                control.cursor,
            )
            control.flops += step_flops
            control.losses.append(loss)
            tracker.log_loss(loss, control.flops)
            log_learning_rate(tracker, config.optim, learning_rate, control.flops)
            progress.set_postfix(loss=f"{loss:.4f}", refresh=False)
            progress.update(step_flops)
            control.elapsed += time.perf_counter() - started
            step = len(control.losses)
            if (
                step == 1
                or step % config.schedule.log_every_steps == 0
                or (control.flops >= config.schedule.train_flops)
            ):
                _log_progress(
                    config,
                    tensors[0],
                    control.losses,
                    control.flops,
                    control.elapsed,
                    learning_rate,
                )
            if step % config.schedule.val_every_steps == 0 and (
                control.flops < config.schedule.train_flops
            ):
                quality = _evaluate(config, model, tensors, control.losses, tracker, control.flops)
                if control.validation(quality):
                    break
            if control.flops < config.schedule.train_flops:
                control.checkpoint()
    if control.stop_reason != "early_stopping":
        quality = _evaluate(config, model, tensors, control.losses, tracker, control.flops)
        control.validation(quality, final_budget=True)
    if quality is None:
        raise RuntimeError("Training ended without final validation")
    control.checkpoint(final=True)
    return control.elapsed, control.losses, control.flops, quality


def _log_progress(
    config: TrainConfig,
    inputs: Tensor,
    losses: list[float],
    flops: int,
    elapsed: float,
    learning_rate: float | None = None,
) -> None:
    budget = config.schedule.train_flops
    speed = _tokens_per_second(
        len(losses), config.schedule.batch_size, int(inputs.size(1)), elapsed
    )
    eta = max(0, budget - flops) * elapsed / flops
    logger.info(
        "train step=%d flops=%d/%d (%.1f%%) loss=%.4f lr=%.3g tok/s=%.0f elapsed=%.1fs eta=%.1fs",
        len(losses),
        flops,
        budget,
        min(flops / budget, 1.0) * 100,
        losses[-1],
        config.optim.learning_rate if learning_rate is None else learning_rate,
        speed,
        elapsed,
        eta,
    )


def _tracked_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    labels: Tensor,
    batch: int,
    cursor: int,
) -> tuple[float, int]:
    """Один шаг обучения без инструментирования или калибровки."""
    batch_x, batch_y, cursor = _rows(inputs, labels, batch, cursor)
    if (run := current()) is not None:
        size = run.batch(batch)
        start = run.rank * size
        batch_x, batch_y = batch_x[start : start + size], batch_y[start : start + size]
    return _step(model, optimizer, batch_x, batch_y), cursor


def _evaluate(
    config: TrainConfig,
    model: torch.nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    losses: list[float],
    tracker: Tracker,
    flops: int,
) -> Quality:
    logger.info(
        "validation start step=%d flops=%d samples=%d", len(losses), flops, tensors[2].size(0)
    )
    quality = local_call(
        lambda: _quality(model, tensors[2], tensors[3], config.schedule.batch_size, losses),
        "validation",
    )
    if (run := current()) is not None:
        quality = run.validation_observation(quality)
    tracker.log_validation(quality.val_loss, quality.val_accuracy, flops)
    logger.info(
        "validation step=%d flops=%d val_loss=%.4f val_accuracy=%.4f",
        len(losses),
        flops,
        quality.val_loss,
        quality.val_accuracy,
    )
    return quality


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    labels: Tensor,
) -> float:
    if current() is not None:
        return distributed_step(model, optimizer, inputs, labels, _loss)
    model.train()
    with compilation_phase("training"), autocast():
        loss = _loss(model(inputs), labels)
    optimizer.zero_grad(set_to_none=True)
    precision = current_precision()
    if precision is None:
        loss.backward()
        optimizer.step()
    else:
        precision.backward_step(loss, optimizer)
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
    with torch.no_grad(), compilation_phase("validation"), autocast():
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


def _compile(
    model: GPT2, enabled: bool, evidence: dict[str, object] | None = None
) -> torch.nn.Module:
    return compile_model(model, enabled, evidence if evidence is not None else {})


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
    if current() is not None:
        return current().device
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
