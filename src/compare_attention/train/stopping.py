"""Validation-event early stopping with an optional optimizer-step warmup."""

import math


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


class EarlyStopping:
    """Track loss improvements; patience zero disables stopping."""

    def __init__(self, patience: int = 0, min_delta: float = 0.001, min_steps: int = 2000):
        self.patience = _nonnegative_int(patience, "patience")
        self.min_steps = _nonnegative_int(min_steps, "min_steps")
        self.min_delta = _finite_number(min_delta, "min_delta")
        if min_delta < 0:
            raise ValueError("min_delta must be nonnegative")

        self.best_loss: float | None = None
        self.best_step: int | None = None
        self.bad_validations = 0
        self.last_step: int | None = None

    def observe(self, loss: float, step: int) -> bool:
        """Record one validation and return whether training should stop.

        Warmup updates the global best but never accumulates bad events. The
        first eligible event compares against that warmup best (it does not
        reset the baseline); with no prior events it establishes the baseline.
        """
        loss = _finite_number(loss, "loss")
        step = _nonnegative_int(step, "step")
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("validation step must be strictly increasing")
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_step = step
            self.bad_validations = 0
        elif step >= self.min_steps:
            self.bad_validations += 1
        self.last_step = step
        return 0 < self.patience <= self.bad_validations and step >= self.min_steps

    def state_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible state including its settings."""
        return {
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "bad_validations": self.bad_validations,
            "last_step": self.last_step,
            "settings": {
                "patience": self.patience,
                "min_delta": self.min_delta,
                "min_steps": self.min_steps,
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore atomically; reject incompatible settings and impossible timelines."""
        if not isinstance(state, dict) or set(state) != set(self.state_dict()):
            raise ValueError("invalid early-stopping state schema")
        settings = state["settings"]
        if not isinstance(settings, dict) or set(settings) != {
            "patience",
            "min_delta",
            "min_steps",
        }:
            raise ValueError("invalid early-stopping settings")
        candidate = EarlyStopping(**settings)
        if candidate.state_dict()["settings"] != self.state_dict()["settings"]:
            raise ValueError("early-stopping settings do not match")
        best_loss, best_step = state["best_loss"], state["best_step"]
        last_step, bad = state["last_step"], state["bad_validations"]
        bad = _nonnegative_int(bad, "bad_validations")
        if last_step is None:
            if best_loss is not None or best_step is not None or bad != 0:
                raise ValueError("inconsistent unobserved early-stopping state")
            best_loss, best_step = None, None
        else:
            last_step = _nonnegative_int(last_step, "last_step")
            best_step = _nonnegative_int(best_step, "best_step")
            best_loss = _finite_number(best_loss, "best_loss")
            if best_step > last_step:
                raise ValueError("best_step cannot exceed last_step")
            eligible_slots = max(0, last_step - max(best_step, self.min_steps - 1))
            if bad > eligible_slots or (
                last_step >= self.min_steps and best_step < last_step and bad == 0
            ):
                raise ValueError("inconsistent bad-validation count")
        self.best_loss, self.best_step = best_loss, best_step
        self.last_step, self.bad_validations = last_step, bad
