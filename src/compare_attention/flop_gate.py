"""Fail-closed check of cloud accounting/compiler smoke and numeric burn-in."""

# Independent artifact gates deliberately repeat evidence checks.
# pylint: disable=duplicate-code

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, TypeGuard

from compare_attention.lr_gate import validated_metric_history
from compare_attention.train.identity import verify_fp16_checkpoint


def _verify_identity(
    directory: Path, *, attention: str, expected: dict[str, Any], expected_gpu: str = "Tesla T4"
) -> dict[str, Any]:
    """Check versioned accounting and actual compiler/runtime evidence."""
    identity = json.loads((directory / "run_identity.json").read_text(encoding="utf-8"))
    accounting = identity.get("accounting")
    if not expected.get("policy") or accounting != json.loads(json.dumps(expected)):
        raise ValueError("Cloud accounting differs from the verified cost model")
    if expected["total"] <= 0 or expected["forward"] + expected["backward"] != expected["total"]:
        raise ValueError("Invalid expected accounting total")
    if (
        identity.get("schema_version") != 2
        or identity.get("status") != "completed"
        or identity.get("attention") != attention
    ):
        raise ValueError("Incomplete or wrong-architecture identity schema")
    if identity.get("requested_seed") != 42 or identity.get("effective_seed") != 42:
        raise ValueError("Smoke requested/effective seed must be 42")
    compiler = identity.get("compiler", {})
    if (
        compiler.get("requested") is not True
        or any(
            compiler.get(key, 0) <= 0
            for key in (
                "backend_invocations",
                "training_backend_invocations",
                "successful_backend_invocations",
                "successful_training_backend_invocations",
            )
        )
        or compiler.get("backend_invocation_verified") is not True
    ):
        raise ValueError("The training compiler backend did not demonstrably return successfully")
    gpu = identity.get("runtime", {}).get("gpu") or {}
    if gpu.get("name") != expected_gpu:
        raise ValueError(f"The smoke did not run on the expected GPU {expected_gpu}")
    return identity


def verify_fp16_precision(identity: dict[str, Any], *, steps: int) -> None:
    """Require scaled FP16 autocast with FP32 masters and no skipped updates."""
    fingerprints = [
        identity.get(key) for key in ("initial_parameters_sha256", "final_parameters_sha256")
    ]
    if (
        any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
            for value in fingerprints
        )
        or fingerprints[0].lower() == fingerprints[1].lower()
    ):
        raise ValueError("FP16 parameters fingerprints must be valid and distinct")
    precision = identity.get("precision")
    if not isinstance(precision, dict) or any(
        precision.get(key) != value
        for key, value in {
            "mode": "amp_float16",
            "parameter_dtype": "float32",
            "autocast_dtype": "float16",
        }.items()
    ):
        raise ValueError("Missing or incompatible FP16 precision metadata")
    if precision.get("grad_scaler_enabled") is not True or any(
        not isinstance(precision.get(key), int)
        or isinstance(precision[key], bool)
        or precision[key] != value
        for key, value in {"optimizer_steps": steps, "skipped_optimizer_steps": 0}.items()
    ):
        raise ValueError("FP16 precision requires actual optimizer updates without skips")
    for key in ("initial_scale", "final_scale"):
        scale = precision.get(key)
        if (
            not isinstance(scale, (float, int))
            or isinstance(scale, bool)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("FP16 precision scale must be finite and positive")

    cap = identity.get("config", {}).get("schedule", {}).get("amp_max_scale")
    if cap is not None:
        _verify_scale_cap(precision, cap)


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _verify_scale_cap(precision: dict, cap: object) -> None:
    if not _finite_number(cap) or cap < 128:
        raise ValueError("Requested FP16 scale cap is invalid")
    if (
        precision.get("max_scale") != cap
        or isinstance(precision.get("max_scale"), bool)
        or precision["initial_scale"] > cap
        or precision["final_scale"] > cap
    ):
        raise ValueError("Requested FP16 scale cap lacks matching runtime evidence")
    growth = precision.get("growth_factor")
    backoff = precision.get("backoff_factor")
    interval = precision.get("growth_interval")
    valid_interval = isinstance(interval, int) and not isinstance(interval, bool) and interval > 0
    if (
        not _finite_number(growth)
        or not _finite_number(backoff)
        or growth <= 1
        or not 0 < backoff < 1
        or not valid_interval
    ):
        raise ValueError("FP16 scale growth metadata is invalid")


def _verify_quality(rows: list[dict], result: dict, *, step_flops: int, steps: int) -> None:
    """Require every loss finite and exactly the configured validation cadence."""
    validations = [row for row in rows if row.get("type") == "validation"]
    cadence = result["config"]["schedule"].get("val_every_steps", 100)
    if not isinstance(cadence, int) or isinstance(cadence, bool) or cadence <= 0:
        raise ValueError("Validation cadence must be a positive non-bool integer")
    validation_flops = [index * step_flops for index in range(cadence, steps, cadence)]
    if [row["flops"] for row in validations] != [*validation_flops, steps * step_flops]:
        raise ValueError("Expected cadence validations and exactly one final validation")
    quality = result["quality"]
    if (
        validations[-1]["loss"] != quality["val_loss"]
        or validations[-1]["accuracy"] != quality["val_accuracy"]
    ):
        raise ValueError("Inconsistent final validation")
    if not all(
        math.isfinite(value)
        for value in [
            *(row["loss"] for row in rows if row.get("type") == "step"),
            *(row[key] for row in validations for key in ("loss", "accuracy")),
            *quality.values(),
        ]
    ):
        raise ValueError("Non-finite smoke loss or quality")


def verify_flops_smoke(
    directory: Path,
    *,
    attention: str,
    expected: dict[str, Any],
    expected_gpu: str = "Tesla T4",
    expected_dtype: str | None = None,
    expected_steps: int = 1,
) -> dict[str, Any]:
    """Verify an exact update horizon (one step by default) before full runs."""
    if (
        not isinstance(expected_steps, int)
        or isinstance(expected_steps, bool)
        or expected_steps <= 0
    ):
        raise ValueError("expected_steps must be a positive non-bool integer")
    identity = _verify_identity(
        directory, attention=attention, expected=expected, expected_gpu=expected_gpu
    )
    compiler = identity["compiler"]
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = validated_metric_history(rows, rows[-1]["config"], step_cost=expected["total"])
    steps = [row for row in rows if row.get("type") == "step"]
    results = [row for row in rows if row.get("type") == "result"]
    if (
        len(steps) != expected_steps
        or len(results) != 1
        or not isinstance(results[0].get("steps"), int)
        or isinstance(results[0]["steps"], bool)
        or results[0]["steps"] != expected_steps
    ):
        horizon = "one training step" if expected_steps == 1 else f"{expected_steps} training steps"
        raise ValueError(f"Expected exactly {horizon} and final result")
    result = results[0]
    if result.get("provenance") != identity:
        raise ValueError("Smoke result provenance differs from completed identity")
    budget = expected["total"] * expected_steps
    if any(row["flops"] != index * expected["total"] for index, row in enumerate(steps, 1)) or (
        result["train_flops"] != budget
    ):
        raise ValueError("Exact-step accounting budget does not match emitted FLOPs")
    config = result["config"]
    if expected_dtype is not None and config["schedule"].get("dtype") != expected_dtype:
        raise ValueError("Unexpected dtype in saved configuration")
    if config["schedule"].get("dtype") == "float16":
        if identity.get("config") != config:
            raise ValueError("Identity and result configuration differ")
        verify_fp16_precision(identity, steps=expected_steps)
        verify_fp16_checkpoint(directory, identity)
    if config["schedule"]["train_flops"] != budget:
        raise ValueError("The configured accounting budget does not match expected steps")
    if config["schedule"]["seed"] != 42 or config["model"]["attention"] != attention:
        raise ValueError("Unexpected seed/architecture in the saved configuration")
    if config["schedule"].get("compile") is not True:
        raise ValueError("The configured compiler must be enabled")
    _verify_quality(rows, result, step_flops=expected["total"], steps=expected_steps)
    return {
        "passed": True,
        "attention": attention,
        "steps": len(steps),
        "optimizer_steps": identity.get("precision", {}).get("optimizer_steps", len(steps)),
        "policy": expected["policy"],
        "step_flops": expected["total"],
        "backend_invocations": compiler["backend_invocations"],
        "directory": str(directory.resolve()),
    }
