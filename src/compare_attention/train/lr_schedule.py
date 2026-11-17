"""Stateless LR on successful-update FLOPs; learning_rate is always the peak."""

import logging
import math

from compare_attention.train.config import Optim


def next_learning_rate(optim: Optim, budget: int, spent: int, step_cost: int) -> float:
    """Price the next update at its clamped END progress, without advancing state."""
    if optim.lr_schedule == "constant":
        return optim.learning_rate
    progress = min((spent + step_cost) / budget, 1.0)
    warmup = optim.warmup_fraction
    if warmup > 0 and progress <= warmup:
        return optim.learning_rate * progress / warmup
    phase = (progress - warmup) / (1 - warmup)
    ratio = optim.min_lr_ratio + (1 - optim.min_lr_ratio) * (1 + math.cos(math.pi * phase)) / 2
    return optim.learning_rate * ratio


def log_learning_rate(tracker, optim: Optim, learning_rate: float, flops: int) -> None:
    """Optional callback; constant runs retain their historical metrics contract."""
    callback = getattr(tracker, "log_learning_rate", None)
    if optim.lr_schedule == "constant":
        return
    if callable(callback):
        callback(learning_rate, flops)
    logging.getLogger(__name__).info("learning_rate flops=%d lr=%.17g", flops, learning_rate)
