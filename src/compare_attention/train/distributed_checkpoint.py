"""Typed DDP checkpoint envelope; legacy single-device validators remain strict."""

from typing import TypedDict

import torch

from compare_attention.train import checkpoint as legacy
from compare_attention.train.compiler import compiler_metadata
from compare_attention.train.distributed import current

SHARDING = "ordered_cyclic_contiguous_v1"


def _validate_compiler(identity, config, step):
    compiler = identity.get("compiler")
    requested = config["schedule"]["compile"]
    reference = compiler_metadata(requested)
    counters = {key for key, value in reference.items() if type(value) is int}
    if not isinstance(compiler, dict) or compiler.keys() != reference.keys():
        raise ValueError("DDP checkpoint compiler schema invalid")
    for key, expected in reference.items():
        actual = compiler[key]
        if key in counters:
            if type(actual) is not int or actual < 0:
                raise ValueError("DDP checkpoint compiler counter invalid")
        elif key != "backend_invocation_verified" and actual != expected:
            raise ValueError("DDP checkpoint compiler identity mismatch")
    successful = compiler["successful_backend_invocations"]
    if (
        type(compiler["backend_invocation_verified"]) is not bool
        or compiler["backend_invocation_verified"] != (successful > 0)
        or successful > compiler["backend_invocations"]
        or (requested and step > 0 and not compiler["successful_training_backend_invocations"])
    ):
        raise ValueError("DDP checkpoint compiler verification invalid")


class Topology(TypedDict):
    world_size: int
    global_batch_size: int
    local_batch_size: int
    sharding: str
    device_type: str


def _validate_rng(row, spec):
    """CPU structural audit is not CUDA RNG restoration/execution qualification."""
    legacy._validate_rng(row["rng"], validate_cuda=False)  # pylint: disable=protected-access
    rng, gpu = row["rng"], row["identity"].get("runtime", {}).get("gpu")
    if spec["device_type"] == "cpu":
        if rng["cuda"] != [] or rng["cuda_devices"] != [] or gpu is not None:
            raise ValueError("DDP checkpoint CPU RNG topology mismatch")
        return
    if (
        not isinstance(gpu, dict)
        or type(gpu.get("index")) is not int
        or gpu["index"] != row["rank"]
        or not isinstance(gpu.get("name"), str)
        or not gpu["name"]
        or not isinstance(gpu.get("compute_capability"), list)
        or len(gpu["compute_capability"]) != 2
        or any(type(value) is not int or value < 0 for value in gpu["compute_capability"])
        or not isinstance(rng["cuda"], list)
        or len(rng["cuda"]) != 1
        or rng["cuda_devices"] != [[gpu["name"], gpu["compute_capability"]]]
    ):
        raise ValueError("DDP checkpoint selected CUDA RNG topology mismatch")
    value = rng["cuda"][0]
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or not value.numel()
    ):
        raise ValueError("DDP checkpoint selected CUDA RNG state invalid")


def topology(config) -> Topology:
    run = current()
    assert run is not None
    return {
        "world_size": run.world_size,
        "global_batch_size": config.schedule.batch_size,
        "local_batch_size": run.batch(config.schedule.batch_size),
        "sharding": SHARDING,
        "device_type": run.device.type,
    }


def single_view(payload, rank):
    """Project only after validating the typed envelope; never a legacy gate downgrade."""
    result = {key: value for key, value in payload.items() if key != "distributed"}
    result["schema_version"] = 1
    row = payload["distributed"]["ranks"][rank]
    result["rng"] = row["rng"]
    result["identity"] = row["identity"]
    result["integrity_sha256"] = legacy._digest(result)  # pylint: disable=protected-access
    return result


def validate(payload):
    """Reject missing/extra ranks, mismatched topology and tampered shared state."""
    # pylint: disable=protected-access
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 2
    ):
        raise ValueError("DDP checkpoint schema mismatch")
    distributed = payload.get("distributed")
    if not isinstance(distributed, dict) or set(distributed) != {"topology", "ranks"}:
        raise ValueError("DDP checkpoint distributed schema mismatch")
    spec = distributed["topology"]
    fields = {"world_size", "global_batch_size", "local_batch_size", "sharding", "device_type"}
    if (
        not isinstance(spec, dict)
        or set(spec) != fields
        or type(spec["world_size"]) is not int
        or spec["world_size"] != 2
        or any(
            type(spec[key]) is not int or spec[key] <= 0
            for key in ("global_batch_size", "local_batch_size")
        )
        or spec["global_batch_size"] != spec["world_size"] * spec["local_batch_size"]
        or spec["sharding"] != SHARDING
        or spec["device_type"] not in {"cpu", "cuda"}
    ):
        raise ValueError("DDP checkpoint topology invalid")
    rows = distributed["ranks"]
    if not isinstance(rows, list) or len(rows) != spec["world_size"]:
        raise ValueError("DDP checkpoint missing ranks")
    initialization = (
        "schema_version",
        "requested_seed",
        "effective_seed",
        "attention",
        "initial_parameters_sha256",
        "fingerprint_format",
        "initial_parameters_stage",
    )
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"rank", "rng", "identity"}
            or type(row["rank"]) is not int
            or row["rank"] != rank
        ):
            raise ValueError("DDP checkpoint rank schema invalid")
        view = single_view(payload, rank)
        legacy._validate_progress(view)
        _validate_rng(row, spec)
        if row["identity"].get("config") != payload["config"]:
            raise ValueError("DDP checkpoint rank identity/config mismatch")
        if (
            any(row["identity"].get(key) != rows[0]["identity"].get(key) for key in initialization)
            or row["identity"].get("schema_version") != 3
            or row["identity"].get("requested_seed") != payload["config"]["schedule"]["seed"]
            or row["identity"].get("effective_seed") != payload["config"]["schedule"]["seed"]
        ):
            raise ValueError("DDP checkpoint rank initialization identity mismatch")
        _validate_compiler(row["identity"], payload["config"], payload["step"])
    if payload["identity"] != rows[0]["identity"] or legacy._digest(
        {"rng": payload["rng"]}
    ) != legacy._digest({"rng": rows[0]["rng"]}):
        raise ValueError("DDP checkpoint canonical rank-zero state mismatch")
    if spec["global_batch_size"] != payload["config"]["schedule"]["batch_size"]:
        raise ValueError("DDP checkpoint global batch mismatch")
    if payload["integrity_sha256"] != legacy._digest(payload):
        raise ValueError("DDP checkpoint integrity checksum mismatch")


def capture(payload, config):
    """Gather small per-rank state; assert shared optimizer/scaler/tensors agree."""
    # pylint: disable=protected-access
    run = current()
    assert run is not None
    shared = {
        key: payload[key]
        for key in (
            "model",
            "optimizer",
            "precision",
            "step",
            "cursor",
            "flops",
            "losses",
            "early_stopping",
            "validation_history",
        )
    }
    digests = run.gather(legacy._digest(shared))
    if len(set(digests)) != 1:
        raise RuntimeError("DDP shared model/optimizer/scaler/progress diverged")
    rows = run.gather({"rank": run.rank, "rng": payload["rng"], "identity": payload["identity"]})
    result = dict(payload)
    # Shared rank-zero fields are canonical; rank-specific identity/RNG are explicit.
    result["rng"], result["identity"] = rows[0]["rng"], rows[0]["identity"]
    result["schema_version"] = 2
    result["distributed"] = {"topology": topology(config), "ranks": rows}
    result["integrity_sha256"] = legacy._digest(result)
    validate(result)
    return result


def preflight(path, config):
    """Fail closed before tracker setup, writes, model construction or history replay."""
    payload = legacy.load_checkpoint(path)
    validate(payload)
    if payload["distributed"]["topology"] != topology(config):
        raise ValueError("DDP checkpoint topology mismatch")
    if payload["config"] != config.model_dump(mode="json"):
        raise ValueError("DDP checkpoint config mismatch")
    stopping = payload["early_stopping"]
    if 0 < config.schedule.early_stopping_patience <= stopping["bad_validations"]:
        raise ValueError("Cannot resume a terminal early-stopped checkpoint")
    if payload["flops"] >= config.schedule.train_flops:
        raise ValueError("Cannot resume an already exhausted training budget")
    return payload
