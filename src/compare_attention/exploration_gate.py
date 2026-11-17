"""Fail-closed artifact verification for exploratory (not fixed-budget) runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.flop_gate import _verify_quality, verify_fp16_precision
from compare_attention.lr_gate import validated_metric_history
from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.config import Optim, Schedule, TrainConfig
from compare_attention.train.flops import estimate_step
from compare_attention.train.identity import parameter_fingerprint, verify_fp16_checkpoint
from compare_attention.train.stopping import EarlyStopping


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _config(raw: dict) -> TrainConfig:
    """Reconstruct saved metadata without opening its remote data directory."""
    components = {
        "model": GPT2Config,
        "schedule": Schedule,
        "optim": Optim,
        "dataset": KVRetrievalConfig,
    }
    if not isinstance(raw.get("data"), str) or not raw["data"]:
        raise ValueError("Missing data identity")
    values = {}
    for name, kind in components.items():
        payload = raw.get(name)
        required = kind.model_fields.keys()
        if (
            name == "optim"
            and isinstance(payload, dict)
            and payload.get("lr_schedule", "constant") == "constant"
        ):
            required = required - {"lr_schedule", "warmup_fraction", "min_lr_ratio"}
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise ValueError(f"Incomplete {name} configuration")
        values[name] = kind.model_validate_json(json.dumps(payload), strict=True)
    return TrainConfig.model_construct(data=Path(raw["data"]), **values)


def _verify_contract(identity: dict, result: dict) -> TrainConfig:
    parsed = _config(result["config"])
    if identity.get("schema_version") != 2:
        raise ValueError("Exploration requires complete identity schema 2")
    for key in ("initial_parameters_sha256", "final_parameters_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("Incomplete parameter fingerprints")
    if any(
        identity.get(key) != parsed.schedule.seed or isinstance(identity.get(key), bool)
        for key in ("requested_seed", "effective_seed")
    ) or (identity.get("attention") != parsed.model.attention):
        raise ValueError("Identity seed/model mismatch")
    cost = json.loads(json.dumps(estimate_step(parsed).to_dict()))
    if identity.get("accounting") != cost:
        raise ValueError("Accounting differs from the complete configured cost model")
    compiler = identity.get("compiler", {})
    if compiler.get("requested") is not parsed.schedule.compile or (
        parsed.schedule.compile
        and (
            compiler.get("backend_invocation_verified") is not True
            or any(
                not _positive_int(compiler.get(key))
                for key in (
                    "backend_invocations",
                    "training_backend_invocations",
                    "successful_backend_invocations",
                    "successful_training_backend_invocations",
                )
            )
        )
    ):
        raise ValueError("Missing successful training compiler return")
    return parsed


def _history(rows: list[dict], result: dict, cost: int) -> None:
    steps = result["steps"]
    if not _positive_int(steps) or not _positive_int(result["train_flops"]):
        raise ValueError("Actual steps/FLOPs must be positive integers")
    updates = [row for row in rows if row.get("type") == "step"]
    if (
        len(updates) != steps
        or result["train_flops"] != steps * cost
        or any(
            not _positive_int(row.get("flops")) or row["flops"] != index * cost
            for index, row in enumerate(updates, 1)
        )
    ):
        raise ValueError("Actual update count or cumulative FLOPs mismatch")
    metrics = [row["loss"] for row in updates]
    metrics += [
        row[key] for row in rows if row.get("type") == "validation" for key in ("loss", "accuracy")
    ]
    metrics += list(result["quality"].values())
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in metrics
    ):
        raise ValueError("Every loss/quality metric must be a finite number")
    _verify_quality(rows, result, step_flops=cost, steps=steps)
    seen = 0
    for row in rows[:-1]:
        if row.get("type") == "step":
            seen += 1
        elif row.get("type") != "validation" or row["flops"] != seen * cost:
            raise ValueError("Unexpected or out-of-order metric record")
    if not math.isclose(
        result["quality"]["train_loss"],
        sum(row["loss"] for row in updates) / steps,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Final training loss differs from actual history")


def _termination(rows: list[dict], result: dict, config: TrainConfig, cost: int) -> None:
    schedule = config.schedule
    reason = result.get("stop_reason")
    status = {"budget_exhausted": "completed", "early_stopping": "early_stopped"}.get(reason)
    if (
        status is None
        or result["provenance"].get("status") != status
        or result["provenance"].get("stop_reason") != reason
    ):
        raise ValueError("Explicit status/stop reason mismatch")
    spent, budget = result["train_flops"], schedule.train_flops
    if reason == "early_stopping":
        if (
            schedule.early_stopping_patience <= 0
            or not 0 < spent < budget
            or result["steps"] % schedule.val_every_steps
        ):
            raise ValueError("Early stopping requires explicit patience and below-budget work")
    elif not budget <= spent < budget + cost:
        raise ValueError("Budget not exhausted or overshoot exceeds one update")
    stopper = EarlyStopping(
        schedule.early_stopping_patience,
        schedule.early_stopping_min_delta,
        schedule.early_stopping_min_steps,
    )
    first_stop = None
    for row in rows:
        if row.get("type") != "validation":
            continue
        step = row["flops"] // cost
        stopped = stopper.observe(row["loss"], step)
        if stopped and row["flops"] < budget and first_stop is None:
            first_stop = step
    if (reason == "early_stopping" and first_stop != result["steps"]) or (
        reason == "budget_exhausted" and first_stop is not None
    ):
        raise ValueError("Validation replay does not justify the first stopping point")
    if "early_stopping" in result:
        expected_state = stopper.state_dict()
        stopper.load_state_dict(result["early_stopping"])
        if result["early_stopping"] != expected_state:
            raise ValueError("Forged early stopping controller state")


def _verify_bf16_checkpoint(directory: Path, identity: dict, config: TrainConfig) -> None:
    """Bind native BF16 state without load_state_dict casts or tied-alias overwrite."""
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = GPT2(config.model).bfloat16()
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("BF16 checkpoint keys differ from configured model")
    for name in expected:
        reference = expected[name]
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.dtype != reference.dtype
            or value.shape != reference.shape
        ):
            raise ValueError("BF16 checkpoint dtype/shape mismatch")
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        canonical = aliases.setdefault(id(parameter), name)
        value = state[name].contiguous().reshape(-1).view(torch.uint8)
        reference = state[canonical].contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(value, reference):
            raise ValueError("BF16 checkpoint conflicting tied aliases")
    model.load_state_dict(state, strict=True)
    actual = parameter_fingerprint(model)
    if actual != identity.get("final_parameters_sha256") or actual == identity.get(
        "initial_parameters_sha256"
    ):
        raise ValueError("BF16 checkpoint fingerprint mismatch or unchanged weights")


def _verify_run(
    directory: Path, *, expected_config: dict | None = None, expected_gpu: str | None = None
) -> dict[str, Any]:
    """Verify actual artifacts; a passing early run is not fixed-budget completion."""
    directory = Path(directory)
    identity = json.loads((directory / "run_identity.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = [row for row in rows if row.get("type") == "result"]
    if len(results) != 1 or rows[-1] != results[0]:
        raise ValueError("Expected exactly one final result")
    result = results[0]
    config = result["config"]
    if identity.get("config") != config or result.get("provenance") != identity:
        raise ValueError("Result configuration/provenance differs from identity")
    if expected_config is not None and expected_config != config:
        raise ValueError("Unexpected configuration")
    if (
        expected_gpu is not None
        and (identity.get("runtime", {}).get("gpu") or {}).get("name") != expected_gpu
    ):
        raise ValueError("Unexpected GPU")
    parsed = _verify_contract(identity, result)
    steps = result["steps"]
    cost = identity["accounting"]["total"]
    rows = validated_metric_history(rows, config, step_cost=cost)
    _history(rows, result, cost)
    _termination(rows, result, parsed, cost)
    if config["schedule"]["dtype"] == "float16":
        verify_fp16_precision(identity, steps=steps)
    if parsed.schedule.dtype == "bfloat16":
        _verify_bf16_checkpoint(directory, identity, parsed)
    else:
        verify_fp16_checkpoint(directory, identity)
    return {
        "passed": True,
        "stop_reason": result["stop_reason"],
        "budget_exhausted": result["train_flops"] >= config["schedule"]["train_flops"],
        "steps": steps,
        "train_flops": result["train_flops"],
        "quality": result["quality"],
        "directory": str(directory.resolve()),
    }


def verify_exploration_run(
    directory: Path, *, expected_config: dict | None = None, expected_gpu: str | None = None
) -> dict[str, Any]:
    """Verify exploratory artifacts, raising ValueError on malformed evidence.

    expected_config is the complete serialized TrainConfig (including dataset),
    not the experiment YAML/kv mapping. Early completion never proves a fixed budget.
    """
    try:
        return _verify_run(
            Path(directory), expected_config=expected_config, expected_gpu=expected_gpu
        )
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Malformed exploration artifacts") from error
