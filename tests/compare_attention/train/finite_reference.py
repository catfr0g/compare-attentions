"""CUDA FP16 AMP policy: FP32 masters, shared initial scale 128, no warm-up.

Both attention families use the same explicit scale. Autocast covers only
forward/loss/evaluation; backward and optimizer updates run outside autocast.
A skipped update aborts the run: its forward/backward work cannot be reported
as a completed optimizer step or a successfully met training FLOP budget.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch
from torch import Tensor

INITIAL_SCALE = 128.0
_ACTIVE: ContextVar["Precision | None"] = ContextVar("training_precision", default=None)


class Precision:
    """One run's scaler and actual optimizer-update evidence."""

    def __init__(self, device: torch.device, max_scale: float | None = None) -> None:
        if device.type != "cuda":
            raise ValueError("float16 training requires CUDA AMP with FP32 master parameters")
        if max_scale is not None and (
            isinstance(max_scale, bool)
            or not isinstance(max_scale, (int, float))
            or not math.isfinite(max_scale)
            or max_scale < INITIAL_SCALE
        ):
            raise ValueError(f"max_scale must be finite and >= {INITIAL_SCALE}")
        self.max_scale = float(max_scale) if max_scale is not None else None
        self.scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
        self.optimizer_steps = 0
        self.skipped_optimizer_steps = 0

    def autocast(self):
        """Create a fresh forward-only autocast context."""
        return torch.autocast("cuda", dtype=torch.float16)

    def backward_step(
        self, loss: Tensor, optimizer: torch.optim.Optimizer, *, synchronize=None
    ) -> None:
        """Unscale gradients and count only an optimizer.step that really ran."""
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        if synchronize is not None:
            finite = torch.isfinite(loss).item() and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            synchronize(finite, "nonfinite scaled gradients; optimizer update aborted")
        if not torch.isfinite(loss) and all(
            torch.isfinite(parameter.grad).all()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ):
            # GradScaler checks gradients, not loss values; reject detached NaN/Inf too.
            self.scaler.update()
            self.skipped_optimizer_steps += 1
            raise RuntimeError(
                "CUDA AMP nonfinite loss: optimizer update skipped; training aborted"
            )
        performed = 0

        def counted(_optimizer, _args, _kwargs):
            nonlocal performed
            performed += 1

        handle = optimizer.register_step_post_hook(counted)
        try:
            self.scaler.step(optimizer)
            self.scaler.update()
        finally:
            handle.remove()
        self.optimizer_steps += performed
        if performed != 1:
            self.skipped_optimizer_steps += 1
            raise RuntimeError("CUDA AMP overflow: optimizer update skipped; training aborted")
        if self.max_scale is not None and self.scaler.get_scale() > self.max_scale:
            # Manual update changes only the scale, preserving the growth tracker.
            # Apply only after success: never grow a reduced scale or hide a skip.
            self.scaler.update(new_scale=self.max_scale)

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible evidence, detached from mutable counters."""
        details = {
            "mode": "amp_float16",
            "parameter_dtype": "float32",
            "autocast_dtype": "float16",
            "grad_scaler_enabled": self.scaler.is_enabled(),
            "optimizer_steps": self.optimizer_steps,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "initial_scale": INITIAL_SCALE,
            "final_scale": float(self.scaler.get_scale()),
        }
        if self.max_scale is not None:
            # Keep legacy uncapped provenance byte-for-byte schema compatible.
            details.update(
                max_scale=self.max_scale,
                growth_factor=self.scaler.get_growth_factor(),
                backoff_factor=self.scaler.get_backoff_factor(),
                growth_interval=self.scaler.get_growth_interval(),
            )
        return details

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Scope the policy without changing existing loop helper signatures."""
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)


def current_precision() -> Precision | None:
    """Return the active FP16 run; legacy float32/BF16 runs have none."""
    return _ACTIVE.get()


def autocast():
    """Autocast only for an explicitly active FP16 run."""
    precision = current_precision()
    return precision.autocast() if precision is not None else nullcontext()
