"""Fail-closed optimizer validation; keep checkpoint's two-argument API stable."""

# Schema checks deliberately reject bool-as-int.
# pylint: disable=unidiomatic-typecheck,too-many-branches

import math

import torch

from compare_attention.train.config import Optim
from compare_attention.train.lr_schedule import next_learning_rate


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def expected_checkpoint_lr(payload) -> float | None:
    """Derive last-used LR only for an explicitly enabled, bound schedule."""
    raw = payload.get("config", {}).get("optim", {})
    if raw.get("lr_schedule", "constant") == "constant":
        return None
    optim = Optim.model_validate(raw)
    config = payload["config"]
    identity = payload["identity"]
    if identity.get("config") != config:
        raise ValueError("checkpoint identity/config invalid")
    cost = identity.get("accounting", {}).get("total")
    step, flops = payload["step"], payload["flops"]
    budget = config["schedule"]["train_flops"]
    if (
        not _integer(cost, 1)
        or not _integer(step)
        or not _integer(flops)
        or not _integer(budget, 1)
        or flops != step * cost
    ):
        raise ValueError("checkpoint accounting budget mismatch")
    if step == 0:
        return optim.learning_rate
    if flops - cost >= budget:
        raise ValueError("checkpoint budget inconsistent with step")
    return next_learning_rate(optim, budget, flops - cost, cost)


def _validate_optimizer(payload, optimizer):  # pylint: disable=too-many-locals
    """Check parameter mapping/state tensors before PyTorch can silently cast them."""
    saved = payload["optimizer"]
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer schema invalid")
    if len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer groups mismatch")
    scheduled_lr = expected_checkpoint_lr(payload)
    parameters = {}
    for previous, current in zip(saved["param_groups"], optimizer.param_groups, strict=True):
        if len(previous["params"]) != len(current["params"]):
            raise ValueError("checkpoint optimizer parameter count mismatch")
        options = {key: value for key, value in previous.items() if key != "params"}
        expected = {key: value for key, value in current.items() if key != "params"}
        if scheduled_lr is not None:
            # Replay may run on another OS/libm; allow only numerical rounding in LR.
            actual_lr = options.get("lr")
            if (
                not isinstance(actual_lr, (int, float))
                or not _finite(actual_lr)
                or not math.isclose(actual_lr, scheduled_lr, rel_tol=1e-12, abs_tol=1e-15)
            ):
                raise ValueError("checkpoint optimizer settings mismatch")
            expected["lr"] = actual_lr
        if options != expected:
            raise ValueError("checkpoint optimizer settings mismatch")
        for index, parameter in zip(previous["params"], current["params"], strict=True):
            if not _integer(index) or index in parameters:
                raise ValueError("checkpoint optimizer parameter index invalid")
            parameters[index] = parameter
    if not isinstance(saved["state"], dict) or not saved["state"].keys() <= parameters.keys():
        raise ValueError("checkpoint optimizer unknown parameter")
    for index, state in saved["state"].items():
        if not isinstance(state, dict):
            raise ValueError("checkpoint optimizer state invalid")
        if isinstance(optimizer, torch.optim.AdamW):
            keys = {"step", "exp_avg", "exp_avg_sq"}
            group = next(group for group in saved["param_groups"] if index in group["params"])
            if group.get("amsgrad", False):
                keys.add("max_exp_avg_sq")
            if set(state) != keys:
                raise ValueError("checkpoint optimizer AdamW state keys invalid")
        for name, value in state.items():
            if name == "step":
                scalar = (
                    value.item()
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                    else value
                )
                if not _finite(scalar) or scalar != payload["step"]:
                    raise ValueError("checkpoint optimizer step inconsistent")
            elif (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.shape != parameters[index].shape
                or value.dtype != parameters[index].dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("checkpoint optimizer tensor shape/dtype invalid")
