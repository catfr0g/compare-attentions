"""Validate raw LR telemetry before exposing a legacy non-LR history view."""

# Exact JSON number types reject bool-as-int; keep the evidence schema together.
# pylint: disable=unidiomatic-typecheck,too-many-boolean-expressions

import math

from compare_attention.train.config import Optim
from compare_attention.train.lr_schedule import next_learning_rate


def validated_metric_history(rows: list[dict], config: dict, *, step_cost: int) -> list[dict]:
    """Replay actual updates against the DECLARED budget, including shortened runs.

    JSON preserves the logged float, but independent Linux/Windows libm cosine
    evaluations need not be bitwise equal. The 1e-12 relative / 1e-15 absolute
    bounds allow last-ULP portability without accepting materially wrong policy.
    Original rows are never modified; only validated LR rows are projected out.
    """
    raw_optim = config.get("optim", {})
    optim = Optim.model_validate(raw_optim, strict=True)
    scheduled = optim.lr_schedule != "constant"
    if scheduled and not Optim.model_fields.keys() <= raw_optim.keys():
        raise ValueError("Scheduled LR requires complete optimizer configuration")
    budget = config["schedule"]["train_flops"]
    if type(step_cost) is not int or step_cost <= 0 or type(budget) is not int or budget <= 0:
        raise ValueError("LR replay requires positive integer budget and step cost")
    view = []
    seen = 0
    pending = False
    for row in rows:
        kind = row.get("type")
        if pending and kind != "learning_rate":
            raise ValueError("Missing learning_rate immediately after successful step")
        if kind == "learning_rate":
            if not scheduled or not pending:
                raise ValueError("Unexpected or duplicate learning_rate record")
            actual = row.get("learning_rate")
            flops = row.get("flops")
            if (
                row.keys() != {"type", "learning_rate", "flops"}
                or type(actual) is not float
                or not math.isfinite(actual)
                or type(flops) is not int
                or flops <= 0
                or flops != seen * step_cost
            ):
                raise ValueError("Malformed learning_rate record")
            expected = next_learning_rate(optim, budget, (seen - 1) * step_cost, step_cost)
            if not math.isfinite(expected) or not math.isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("Recorded learning_rate differs from declared policy")
            pending = False
            continue
        if kind not in {"step", "validation", "result"}:
            raise ValueError("Unexpected metric record")
        if kind == "step":
            seen += 1
            if type(row.get("flops")) is not int or row["flops"] != seen * step_cost:
                raise ValueError("Out-of-order step FLOPs in LR history")
            pending = scheduled
        view.append(row)
    if pending:
        raise ValueError("Missing final learning_rate record")
    return view
