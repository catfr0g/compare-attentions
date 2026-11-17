"""Training-only progress, opt-in validation stopping and checkpoint boundaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from compare_attention.train.checkpoint import (
    capture_state,
    load_checkpoint,
    restore_rng,
    restore_state,
)
from compare_attention.train.config import TrainConfig
from compare_attention.train.distributed import current, local_call
from compare_attention.train.distributed_checkpoint import capture, single_view
from compare_attention.train.lr_schedule import log_learning_rate, next_learning_rate
from compare_attention.train.precision import current_precision
from compare_attention.train.result import Quality
from compare_attention.train.stopping import EarlyStopping

logger = logging.getLogger(__name__)


class TrainingControl:  # pylint: disable=too-many-instance-attributes
    """Keep checkpoint/validation I/O outside the measured optimizer-step interval."""

    def __init__(self, config: TrainConfig, model, optimizer, tracker, identity, resume_from=None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.tracker = tracker
        self.identity = identity
        self.losses: list[float] = []
        self.cursor: int = 0
        self.flops: int = 0
        self.elapsed: float = 0.0
        self.validations: list[dict] = []
        self.stop_reason: Literal["budget_exhausted", "early_stopping"] = "budget_exhausted"
        schedule = config.schedule
        self.stopper = EarlyStopping(
            schedule.early_stopping_patience,
            schedule.early_stopping_min_delta,
            schedule.early_stopping_min_steps,
        )
        if schedule.checkpoint_every_steps and not callable(
            getattr(tracker, "save_checkpoint", None)
        ):
            raise ValueError("checkpoint_every_steps requires a persistent checkpoint tracker")
        if resume_from is not None:
            self._resume(Path(resume_from))

    def _resume(self, path: Path) -> None:
        payload = local_call(lambda: load_checkpoint(path), "load_checkpoint")
        if (run := current()) is not None:
            payload = single_view(payload, run.rank)
        progress = local_call(
            lambda: restore_state(
                payload, self.model, self.optimizer, self.config, self.identity, current_precision()
            ),
            "restore_checkpoint",
        )
        self.losses = list(progress["losses"])
        self.cursor = progress["cursor"]
        self.flops = progress["flops"]
        self.elapsed = progress["elapsed"]
        self.validations = list(progress.get("validation_history", []))
        self.stopper.load_state_dict(progress["early_stopping"])
        if 0 < self.stopper.patience <= self.stopper.bad_validations:
            raise ValueError("Cannot resume a terminal early-stopped checkpoint")
        if self.flops >= self.config.schedule.train_flops:
            raise ValueError("Cannot resume an already exhausted training budget")
        # Re-emit recorded history into the NEW result directory/tracker, not the old journal.
        by_flops = {row["flops"]: row for row in self.validations}
        cost = self.identity["accounting"]["total"]
        for step, loss in enumerate(self.losses, 1):
            self.tracker.log_loss(loss, step * cost)
            learning_rate = next_learning_rate(
                self.config.optim, self.config.schedule.train_flops, (step - 1) * cost, cost
            )
            log_learning_rate(self.tracker, self.config.optim, learning_rate, step * cost)
            if (validation := by_flops.get(step * cost)) is not None:
                self.tracker.log_validation(
                    validation["loss"], validation["accuracy"], validation["flops"]
                )
        logger.info("resume step=%d flops=%d checkpoint=%s", len(self.losses), self.flops, path)
        restore_rng(payload["rng"])  # Tracker replay must not perturb the resumed trajectory.

    def validation(self, quality: Quality, *, final_budget: bool = False) -> bool:
        """Record the observed validation once, and decide whether to end exploration."""
        self.validations.append(
            {
                "type": "validation",
                "loss": quality.val_loss,
                "accuracy": quality.val_accuracy,
                "flops": self.flops,
            }
        )
        stopped = self.stopper.observe(quality.val_loss, len(self.losses))
        if stopped and not final_budget:
            self.stop_reason = "early_stopping"
            logger.info(
                "early stopping step=%d flops=%d budget=%d state=%s",
                len(self.losses),
                self.flops,
                self.config.schedule.train_flops,
                self.stopper.state_dict(),
            )
            return True
        return False

    def checkpoint(self, *, final: bool = False) -> None:
        """Save full state only at successful update boundaries; never count this as training."""
        interval = self.config.schedule.checkpoint_every_steps
        step = len(self.losses)
        if not interval or (not final and step % interval):
            return
        payload = capture_state(
            self.model,
            self.optimizer,
            self.config,
            step=step,
            cursor=self.cursor,
            flops=self.flops,
            losses=self.losses,
            elapsed=self.elapsed,
            early_stopping=self.stopper.state_dict(),
            identity=self.identity,
            precision=current_precision(),
            validation_history=self.validations,
        )
        if current() is not None:
            payload = capture(payload, self.config)
        self.tracker.save_checkpoint(payload, step, self.config.schedule.checkpoint_keep_last)
        logger.info("checkpoint saved step=%d flops=%d", step, self.flops)
