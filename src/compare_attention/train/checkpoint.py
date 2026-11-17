"""Versioned, CPU-owned training state; RNG restoration is deliberately last."""

# Explicit fail-closed schemas reject bool-as-int and keep related invariants together.
# pylint: disable=unidiomatic-typecheck,too-many-boolean-expressions,too-many-branches

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compare_attention.train.checkpoint_optimizer import _validate_optimizer
from compare_attention.train.compiler import compiler_metadata
from compare_attention.train.distributed import current


def _digest(payload: dict) -> str:
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            visit(["tensor", str(tensor.dtype), list(tensor.shape)])
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict{")
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                visit(key)
                visit(value[key])
            digest.update(b"}")
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + b"[")
            for item in value:
                visit(item)
            digest.update(b"]")
        else:
            digest.update((type(value).__name__ + ":" + repr(value) + ";").encode())

    visit({key: value for key, value in payload.items() if key != "integrity_sha256"})
    return digest.hexdigest()


def _identity(identity: dict) -> dict:
    fields = set(
        (
            "schema_version requested_seed effective_seed attention config "
            "initial_parameters_sha256 fingerprint_format initial_parameters_stage"
        ).split()
    )
    chosen = {key: value for key, value in identity.items() if key in fields}
    runtime_fields = set("python platform torch cuda cudnn fla fla_core triton clearml gpu".split())
    runtime = identity.get("runtime", {})
    chosen["runtime"] = {key: value for key, value in runtime.items() if key in runtime_fields}
    if isinstance(chosen["runtime"].get("gpu"), dict):
        chosen["runtime"]["gpu"] = {
            key: value
            for key, value in chosen["runtime"]["gpu"].items()
            if key in {"name", "index", "compute_capability", "total_memory_bytes"}
        }
    if "compiler" in identity:
        chosen["compiler"] = {
            key: value
            for key, value in identity["compiler"].items()
            if key in compiler_metadata(False)
        }
    if "accounting" in identity:
        keys = {"policy", "total", "forward", "backward", "breakdown", "metadata"}
        chosen["accounting"] = {
            key: value for key, value in identity["accounting"].items() if key in keys
        }
    return chosen


def _cpu(value) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported checkpoint value: {type(value).__name__}")


def _unwrap(model):
    while True:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod  # pylint: disable=protected-access
        else:
            return model


def _model_types(model):
    return [
        (name, f"{type(module).__module__}.{type(module).__qualname__}")
        for name, module in _unwrap(model).named_modules()
    ]


def _optimizer_names(model, optimizer):
    names = {id(parameter): name for name, parameter in _unwrap(model).named_parameters()}
    try:
        return [
            [names[id(parameter)] for parameter in group["params"]]
            for group in optimizer.param_groups
        ]
    except KeyError as error:
        raise ValueError("optimizer parameter is not in the model") from error


def _rng_devices():
    if not torch.cuda.is_available():
        return []
    if (run := current()) is not None:
        return [run.local_rank]
    return list(range(torch.cuda.device_count()))


def _cuda_devices():
    return [
        [torch.cuda.get_device_name(index), list(torch.cuda.get_device_capability(index))]
        for index in _rng_devices()
    ]


def capture_state(  # pylint: disable=too-many-arguments
    model,
    optimizer,
    config,
    *,
    step,
    cursor,
    flops,
    losses,
    elapsed,
    early_stopping,
    identity,
    precision,
    validation_history=None,
) -> dict:
    """Capture independent CPU copies at a completed optimizer-step boundary."""
    numpy_state = np.random.get_state()
    payload = _cpu(
        {
            "schema_version": 1,
            "config": config.model_dump(mode="json"),
            "identity": _identity(identity),
            "model": _unwrap(model).state_dict(),
            "model_structure": _model_types(model),
            "optimizer": optimizer.state_dict(),
            "optimizer_parameters": _optimizer_names(model, optimizer),
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "precision": None
            if precision is None
            else {
                "scaler": precision.scaler.state_dict(),
                "optimizer_steps": precision.optimizer_steps,
                "skipped_optimizer_steps": precision.skipped_optimizer_steps,
            },
            "rng": {
                "python": random.getstate(),
                "numpy": [
                    numpy_state[0],
                    torch.tensor(numpy_state[1].astype(np.int64)),
                    *numpy_state[2:],
                ],
                "torch": torch.get_rng_state(),
                "cuda": [torch.cuda.get_rng_state(index) for index in _rng_devices()],
                "cuda_devices": _cuda_devices(),
            },
            "step": step,
            "cursor": cursor,
            "flops": flops,
            "losses": losses,
            "elapsed": elapsed,
            "early_stopping": early_stopping,
            "validation_history": validation_history if validation_history is not None else [],
        }
    )
    payload["integrity_sha256"] = _digest(payload)
    return payload


def load_checkpoint(path) -> dict:
    """Safely load a .pt payload, or latest.json with relative-path/SHA256 checks."""
    path = Path(path)
    manifest = None
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"step", "path", "sha256"}
            or not _integer(manifest["step"])
            or manifest["path"] != f"step_{manifest['step']:08d}.pt"
            or not isinstance(manifest["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", manifest["sha256"]) is None
        ):
            raise ValueError("checkpoint manifest path/schema invalid")
        path = path.parent / manifest["path"]
        if path.is_symlink():
            raise ValueError("checkpoint manifest path must not be a symlink")
    with path.open("rb") as stream:
        if manifest is not None:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
                raise ValueError("checkpoint manifest checksum mismatch")
            stream.seek(0)
        payload = torch.load(stream, weights_only=True, map_location="cpu")
    if isinstance(payload, dict) and payload.get("schema_version") == 2:
        from compare_attention.train.distributed_checkpoint import validate

        validate(payload)
    else:
        _validate_progress(payload)
    if payload["integrity_sha256"] != _digest(payload):
        raise ValueError("checkpoint integrity checksum mismatch")
    if manifest is not None and manifest["step"] != payload["step"]:
        raise ValueError("checkpoint manifest step mismatch")
    return payload


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_progress(payload):
    required = set(
        (
            "schema_version config identity model model_structure optimizer optimizer_class "
            "optimizer_parameters "
            "precision rng "
            "step cursor flops losses elapsed early_stopping validation_history "
            "integrity_sha256"
        ).split()
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("checkpoint schema mismatch")
    step, flops, cursor = payload["step"], payload["flops"], payload["cursor"]
    if not all(_integer(value) for value in (step, flops, cursor)):
        raise ValueError("checkpoint step/cursor/budget invalid")
    if not isinstance(payload["losses"], list) or len(payload["losses"]) != step:
        raise ValueError("checkpoint step/loss history mismatch")
    if not all(_finite(loss) for loss in payload["losses"]):
        raise ValueError("checkpoint nonfinite loss")
    if not _finite(payload["elapsed"]) or payload["elapsed"] < 0:
        raise ValueError("checkpoint elapsed invalid")
    if not isinstance(payload["early_stopping"], dict):
        raise ValueError("checkpoint early stopping state invalid")
    history = payload["validation_history"]
    if not isinstance(history, list):
        raise ValueError("checkpoint validation history invalid")
    previous = -1
    for row in history:
        if (
            not isinstance(row, dict)
            or not _integer(row.get("flops"), 1)
            or not previous < row["flops"] <= flops
            or not _finite(row.get("loss"))
            or not _finite(row.get("accuracy"))
            or not 0 <= row["accuracy"] <= 1
            or ("step" in row and (not _integer(row["step"], 1) or row["step"] > step))
        ):
            raise ValueError("checkpoint validation history inconsistent")
        previous = row["flops"]
    try:
        config = payload["config"]
        budget = config["schedule"]["train_flops"]
        batch = config["schedule"]["batch_size"]
        count = config["dataset"]["n_train"]
        if not all(_integer(value, 1) for value in (budget, batch, count)):
            raise ValueError("checkpoint config budget/cursor invalid")
        if cursor != (step * batch) % count:
            raise ValueError("checkpoint cursor inconsistent with step")
        if (step == 0 and flops != 0) or (
            step > 0 and (flops <= 0 or flops % step or flops - flops // step >= budget)
        ):
            raise ValueError("checkpoint budget inconsistent with step")
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint config missing budget/cursor") from error


def _validate_rng(state, *, validate_cuda=True):
    try:
        if set(state) != {"python", "numpy", "torch", "cuda", "cuda_devices"}:
            raise ValueError("missing RNG")
        random.Random().setstate(state["python"])
        numpy_state = state["numpy"]
        if (
            len(numpy_state) != 5
            or numpy_state[0] != "MT19937"
            or not isinstance(numpy_state[1], torch.Tensor)
            or numpy_state[1].dtype != torch.int64
            or numpy_state[1].shape != (624,)
            or not bool(((numpy_state[1] >= 0) & (numpy_state[1] <= 2**32 - 1)).all())
        ):
            raise ValueError("invalid NumPy RNG")
        np.random.RandomState().set_state(_numpy_rng(numpy_state))  # pylint: disable=no-member
        torch.Generator(device="cpu").set_state(state["torch"])
        if not validate_cuda:
            return
        devices = _rng_devices()
        if (
            not isinstance(state["cuda"], list)
            or len(state["cuda"]) != len(devices)
            or state["cuda_devices"] != _cuda_devices()
        ):
            raise ValueError("CUDA RNG device count mismatch")
        for index, value in zip(devices, state["cuda"], strict=True):
            torch.Generator(device=f"cuda:{index}").set_state(value)
    except (KeyError, TypeError, RuntimeError, IndexError) as error:
        raise ValueError("checkpoint RNG invalid") from error


def _numpy_rng(state):
    return (state[0], state[1].numpy().astype(np.uint32), *state[2:])


def _validate_model(state, model):
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("checkpoint model keys mismatch")
    for name, reference in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"checkpoint model shape/dtype mismatch: {name}")
    aliases = {}
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(
        model.named_buffers(remove_duplicate=False)
    ):
        canonical = aliases.setdefault(id(tensor), name)
        left = state[name].cpu().contiguous().reshape(-1).view(torch.uint8)
        right = state[canonical].cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left, right):
            raise ValueError(f"checkpoint model tied alias mismatch: {name}")


def _validate(payload, model, optimizer, config, identity, precision):
    _validate_progress(payload)
    config_dict = config.model_dump(mode="json")
    if payload["config"] != config_dict:
        raise ValueError("checkpoint config mismatch")
    for candidate in (payload["identity"], identity):
        if (
            not isinstance(candidate, dict)
            or candidate.get("config") != config_dict
            or not isinstance(candidate.get("runtime"), dict)
            or not isinstance(candidate.get("initial_parameters_sha256"), str)
        ):
            raise ValueError("checkpoint identity/config invalid")
    for key in (
        "config",
        "runtime",
        "initial_parameters_sha256",
        "requested_seed",
        "effective_seed",
        "attention",
        "fingerprint_format",
        "initial_parameters_stage",
        "accounting",
    ):
        if payload["identity"].get(key) != identity.get(key):
            raise ValueError(f"checkpoint identity mismatch: {key}")
    accounting = payload["identity"].get("accounting")
    if accounting is not None and (
        not _integer(accounting.get("total"), 1)
        or payload["flops"] != payload["step"] * accounting["total"]
    ):
        raise ValueError("checkpoint accounting budget mismatch")
    if payload["model_structure"] != _model_types(model):
        raise ValueError("checkpoint model structure mismatch")
    _validate_model(payload["model"], model)
    if payload["optimizer_parameters"] != _optimizer_names(model, optimizer):
        raise ValueError("checkpoint optimizer parameter order mismatch")
    optimizer_class = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    if payload["optimizer_class"] != optimizer_class:
        raise ValueError("checkpoint optimizer class mismatch")
    try:
        _validate_optimizer(payload, optimizer)
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("checkpoint optimizer invalid") from error
    if (payload["precision"] is None) != (precision is None):
        raise ValueError("checkpoint precision mismatch")
    if precision is not None:
        saved = payload["precision"]
        if (
            not isinstance(saved, dict)
            or set(saved) != {"scaler", "optimizer_steps", "skipped_optimizer_steps"}
            or not _integer(saved["optimizer_steps"])
            or saved["optimizer_steps"] != payload["step"]
            or type(saved["skipped_optimizer_steps"]) is not int
            or saved["skipped_optimizer_steps"] != 0
        ):
            raise ValueError("checkpoint precision counters inconsistent")
        scaler = saved["scaler"]
        expected_keys = {
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        }
        if (
            not isinstance(scaler, dict)
            or set(scaler) != expected_keys
            or not all(_finite(scaler[key]) for key in ("scale", "growth_factor", "backoff_factor"))
            or scaler["scale"] <= 0
            or scaler["growth_factor"] <= 1
            or not 0 < scaler["backoff_factor"] < 1
            or not _integer(scaler["growth_interval"], 1)
            or not _integer(scaler["_growth_tracker"])
            or scaler["_growth_tracker"] >= scaler["growth_interval"]
        ):
            raise ValueError("checkpoint scaler state invalid")
        try:
            copy.deepcopy(precision.scaler).load_state_dict(scaler)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError("checkpoint scaler invalid") from error
    _validate_rng(payload["rng"])
    if payload["integrity_sha256"] != _digest(payload):
        raise ValueError("checkpoint integrity checksum mismatch")


def restore_state(payload, model, optimizer, config, identity, precision) -> dict:
    """Validate before mutation, restore training state, then RNG, and return progress."""
    model = _unwrap(model)
    _validate(payload, model, optimizer, config, identity, precision)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(_cpu(payload["optimizer"]))
    if precision is not None:
        precision.scaler.load_state_dict(payload["precision"]["scaler"])
        precision.optimizer_steps = payload["precision"]["optimizer_steps"]
        precision.skipped_optimizer_steps = payload["precision"]["skipped_optimizer_steps"]
    keys = "step cursor flops losses elapsed early_stopping validation_history".split()
    progress = _cpu({key: payload[key] for key in keys})
    restore_rng(payload["rng"])
    return progress


def restore_rng(state: dict) -> None:
    """Reapply snapshot RNG after optional tracker replay; validate before mutation."""
    _validate_rng(state)
    random.setstate(state["python"])
    np.random.set_state(_numpy_rng(state["numpy"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        for index, value in zip(_rng_devices(), state["cuda"], strict=True):
            torch.cuda.set_rng_state(value, index)
