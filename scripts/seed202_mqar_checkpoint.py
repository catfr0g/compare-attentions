"""Read-only authentication and structural audit of the original MQAR FULL."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

FULL_SHA = "e2289ced84240f1f81dfcaa254741457a7863ae9a40db6ab68f9cc021e904542"
FULL_BYTES = 27_196_049
STEP = 980
COST = 463_856_467_968
TOTAL = 10_780


def require(condition, code):
    if not condition:
        raise ValueError(code)


def descriptor(payload):
    """Reference metadata must originate only in the SHA-authenticated original."""
    from compare_attention.train.checkpoint import _digest

    return {
        "shared": {
            name: _digest({name: payload[name]}) for name in payload if name != "integrity_sha256"
        },
        "model": {
            name: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
            for name, value in payload["model"].items()
        },
        "optimizer_parameters": payload["optimizer_parameters"],
        "model_structure": payload["model_structure"],
    }


def validate_structure(payload, expected):
    """Check genuine native tensor state without creating or restoring any model."""
    import torch

    from compare_attention.train.distributed_checkpoint import validate

    validate(payload)
    require(payload["step"] == STEP, "ORIGINAL_STEP")
    require(payload["flops"] == STEP * COST, "ORIGINAL_FLOPS")
    require(payload["cursor"] == STEP * 64 % 100_000, "ORIGINAL_CURSOR")
    require(len(payload["losses"]) == STEP, "ORIGINAL_HISTORY")
    require(
        payload["validation_history"]
        == [
            {
                "type": "validation",
                "loss": 8.331951721191405,
                "accuracy": 0.00015625,
                "flops": STEP * COST,
            }
        ],
        "ORIGINAL_VALIDATION",
    )
    require(
        payload["distributed"]["topology"]
        == {
            "world_size": 2,
            "global_batch_size": 64,
            "local_batch_size": 32,
            "sharding": "ordered_cyclic_contiguous_v1",
            "device_type": "cuda",
        },
        "ORIGINAL_TOPOLOGY",
    )
    require(
        payload["precision"]
        == {
            "optimizer_steps": STEP,
            "skipped_optimizer_steps": 0,
            "scaler": {
                "scale": 128.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": STEP,
            },
        },
        "ORIGINAL_PRECISION",
    )
    require(payload["optimizer_class"] == "torch.optim.adamw.AdamW", "ORIGINAL_ADAMW")
    optimizer = payload["optimizer"]
    require(set(optimizer) == {"state", "param_groups"}, "OPTIMIZER_SCHEMA")
    require(
        len(optimizer["param_groups"]) == len(payload["optimizer_parameters"]) == 1,
        "OPTIMIZER_GROUPS",
    )
    group = optimizer["param_groups"][0]
    names = dict(zip(group["params"], payload["optimizer_parameters"][0], strict=True))
    require(len(names) == len(set(names.values())) == 52, "OPTIMIZER_ORDER")
    require(set(optimizer["state"]) == set(names), "OPTIMIZER_STATE_CLOSURE")
    require(
        group["lr"] == 0.001
        and group["weight_decay"] == 0.1
        and group["betas"] == (0.9, 0.999)
        and group["eps"] == 1e-8,
        "ORIGINAL_OPTIMIZER_SETTINGS",
    )
    for index, state in optimizer["state"].items():
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "ADAMW_STATE_SCHEMA")
        require(state["step"].numel() == 1 and state["step"].item() == STEP, "ADAMW_STEP")
        reference = payload["model"][names[index]]
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            require(
                value.shape == reference.shape and value.dtype == reference.dtype == torch.float32,
                "ADAMW_TENSOR_SCHEMA",
            )
    require(set(payload["model"]) - set(names.values()) == {"lm_head.weight"}, "MODEL_NAMES")
    require(
        torch.equal(payload["model"]["lm_head.weight"], payload["model"]["wte.weight"]),
        "TIED_WEIGHTS",
    )

    def finite(value):
        if isinstance(value, torch.Tensor):
            require(
                value.device.type == "cpu" and value.layout == torch.strided, "READ_ONLY_CPU_TENSOR"
            )
            require(bool(torch.isfinite(value).all()), "FINITE_TENSOR")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                finite(child)
        elif type(value) is float:
            require(math.isfinite(value), "FINITE_SCALAR")

    finite(payload)
    require(descriptor(payload) == expected, "AUTHENTIC_ORIGINAL_STRUCTURE")
    return {
        "scope": "AUTHENTIC_FULL_METADATA_NO_TARGET_RESTORE",
        "step": STEP,
        "next_step": STEP + 1,
        "remaining_updates": TOTAL - STEP,
        "optimizer_states": len(names),
        "actual_target_restore": "PENDING_CLOUD_ONLY",
    }


def validate_original(path):
    """Authenticate whole bytes BEFORE weights-only CPU deserialization."""
    import torch

    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size == FULL_BYTES, "ORIGINAL_FULL_SIZE")
    with path.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == FULL_SHA, "ORIGINAL_FULL_SHA256"
        )
        stream.seek(0)
        payload = torch.load(stream, weights_only=True, map_location="cpu")
    return validate_structure(payload, descriptor(payload))
