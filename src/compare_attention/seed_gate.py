"""Validate an independent pilot seed pair before submitting the remaining experiments."""

# Keep raw JSON decoding visible at each independent public artifact boundary.
# pylint: disable=duplicate-code

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from compare_attention.flop_gate import verify_fp16_precision
from compare_attention.lr_gate import validated_metric_history
from compare_attention.train.identity import verify_fp16_checkpoint


def checkpoint_parameters_sha256(path: Path) -> str:
    """Hash tensor contents, independent of checkpoint serialization or metadata."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Missing checkpoint state dict: {path}")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_v2_accounting(result: dict[str, Any], identity: dict[str, Any]) -> None:
    if result.get("provenance") != identity:
        raise ValueError("Result provenance differs from completed run identity")
    cost = identity.get("accounting", {})
    total = cost.get("total", 0)
    if not cost.get("policy") or total <= 0 or result["train_flops"] != result["steps"] * total:
        raise ValueError("Versioned accounting does not match completed training steps")
    compiler = identity.get("compiler", {})
    requested = result["config"]["schedule"].get("compile", False)
    if compiler.get("requested") != requested or (
        requested
        and (
            compiler.get("training_backend_invocations", 0) <= 0
            or compiler.get("successful_training_backend_invocations", 0) <= 0
            or compiler.get("backend_invocation_verified") is not True
        )
    ):
        raise ValueError("Missing training compiler evidence for versioned accounting")
    if result["config"]["schedule"].get("dtype") == "float16":
        if identity.get("config") != result["config"]:
            raise ValueError("FP16 result configuration differs from identity")
        verify_fp16_precision(identity, steps=result["steps"])


def _verify_final_validation(rows: list[dict], result: dict, directory: Path) -> None:
    validation = [
        row
        for row in rows
        if row.get("type") == "validation" and row["flops"] == result["train_flops"]
    ]
    if len(validation) != 1:
        raise ValueError(f"Expected exactly one final validation for {directory}")
    final = validation[0]
    if (
        final["loss"] != result["quality"]["val_loss"]
        or final["accuracy"] != result["quality"]["val_accuracy"]
    ):
        raise ValueError(f"Inconsistent final validation for {directory}")


# Exact step types reject bools; keep the complete single-run acceptance boundary.
# pylint: disable-next=too-many-branches
def _read_run(directory: Path, seed: int, attention: str) -> dict[str, Any]:
    identity = json.loads((directory / "run_identity.json").read_text(encoding="utf-8"))
    if identity.get("schema_version") not in {1, 2} or identity.get("status") != "completed":
        raise ValueError(f"Expected completed identity schema version 1 or 2 for {directory}")
    if identity.get("requested_seed") != seed or identity.get("effective_seed") != seed:
        raise ValueError(f"Requested/effective seed mismatch for {directory}")
    initial = identity.get("initial_parameters_sha256", "")
    if not isinstance(initial, str) or len(initial) != 64:
        raise ValueError(f"Invalid initial parameters fingerprint for {directory}")
    try:
        bytes.fromhex(initial)
    except ValueError as exc:
        raise ValueError(f"Invalid initial parameters fingerprint for {directory}") from exc
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = [row for row in rows if row.get("type") == "result"]
    if len(results) != 1 or rows[-1].get("type") != "result":
        raise ValueError(f"Expected exactly one final result for {directory}")
    result = results[0]
    if result["config"]["schedule"].get("dtype") == "float16" and identity["schema_version"] != 2:
        raise ValueError("FP16 requires identity schema version 2")
    if identity["schema_version"] == 2:
        _verify_v2_accounting(result, identity)
    config = result["config"]
    if config["schedule"]["seed"] != seed or config["model"]["attention"] != attention:
        raise ValueError(f"Unexpected seed/attention configuration for {directory}")
    # pylint: disable-next=unidiomatic-typecheck
    if type(result.get("steps")) is not int or result["steps"] <= 0:
        raise ValueError("Invalid completed step count")
    cost = (
        identity["accounting"]["total"]
        if identity["schema_version"] == 2
        else result["train_flops"] // result["steps"]
    )
    rows = validated_metric_history(rows, config, step_cost=cost)
    steps = [row for row in rows if row.get("type") == "step"]
    if len(steps) != result["steps"] or len(steps) < 2:
        raise ValueError(f"Incomplete step history for {directory}")
    if result["train_flops"] < config["schedule"]["train_flops"]:
        raise ValueError(f"Training budget was not reached for {directory}")
    if steps[-1]["flops"] != result["train_flops"]:
        raise ValueError(f"Final step budget does not match result for {directory}")
    values = [row["loss"] for row in steps] + list(result["quality"].values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite quality/history for {directory}")
    _verify_final_validation(rows, result, directory)
    checkpoint_hash = (
        verify_fp16_checkpoint(directory, identity)
        if config["schedule"].get("dtype") == "float16"
        else checkpoint_parameters_sha256(directory / "model.pt")
    )
    return {
        "config": config,
        "identity": identity,
        "losses": [row["loss"] for row in steps],
        "checkpoint_parameters_sha256": checkpoint_hash,
        "directory": str(directory.resolve()),
    }


def verify_seed_pair(
    first: Path,
    second: Path,
    *,
    seeds: tuple[int, int] = (42, 6767),
    attention: str = "classic",
) -> dict[str, Any]:
    """Fail closed unless pilots prove independent initialization and training."""
    if seeds[0] == seeds[1]:
        raise ValueError("Pilot seeds must differ")
    runs = [
        _read_run(path, seed, attention) for path, seed in zip((first, second), seeds, strict=True)
    ]
    normalized = []
    for run in runs:
        config = deepcopy(run["config"])
        config["schedule"]["seed"] = 0
        normalized.append(config)
    if normalized[0] != normalized[1]:
        raise ValueError("Pilot configuration differs beyond the training seed")
    runtimes = [run["identity"].get("runtime") for run in runs]
    if not all(runtimes) or runtimes[0] != runtimes[1]:
        raise ValueError("Pilot runtime versions or hardware differ")
    if runs[0]["identity"].get("accounting") != runs[1]["identity"].get("accounting"):
        raise ValueError("Pilot accounting policies or step costs differ")
    initial = [run["identity"]["initial_parameters_sha256"] for run in runs]
    if initial[0] == initial[1]:
        raise ValueError("Pilot initial parameters are identical")
    final = [run["checkpoint_parameters_sha256"] for run in runs]
    if final[0] == final[1]:
        raise ValueError("Pilot final parameters are identical")
    if runs[0]["losses"] == runs[1]["losses"]:
        raise ValueError("Pilot loss trajectories are identical")
    return {
        "passed": True,
        "attention": attention,
        "seeds": list(seeds),
        "initial_parameters_sha256": initial,
        "final_parameters_sha256": final,
        "directories": [run["directory"] for run in runs],
    }
