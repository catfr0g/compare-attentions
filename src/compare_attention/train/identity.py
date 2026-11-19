"""Local, secret-free evidence of initialized weights and runtime provenance."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.compiler import compiler_metadata
from compare_attention.train.config import TrainConfig


def parameter_fingerprint(model: torch.nn.Module) -> str:
    """Hash sorted unique named parameters: name, dtype, shape and exact tensor bytes.

    CPU copies do not draw RNG or modify weights, gradients or optimizer state.
    Byte views support BF16 without NumPy's BF16 conversion limitations.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        digest.update(header.encode("utf-8") + b"\n")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_fp16_checkpoint(directory: Path, identity: dict[str, Any]) -> str:
    """Bind FP16 provenance to strict FP32 checkpoint contents, without RNG effects.

    Reconstruct only the declared repository architecture on CPU; validate before
    load_state_dict can silently cast or overwrite conflicting tied aliases.
    """
    raw_config = identity.get("config", {}).get("model", {})
    if not isinstance(raw_config, dict) or not GPT2Config.model_fields.keys() <= raw_config.keys():
        raise ValueError("FP16 checkpoint requires complete model configuration")
    config = GPT2Config.model_validate(raw_config)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("FP16 checkpoint requires a nonempty state dict")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = GPT2(config).float()
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("FP16 checkpoint keys differ from configured model")
    for name, reference in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.dtype != reference.dtype
            or value.shape != reference.shape
        ):
            raise ValueError(f"FP16 checkpoint dtype/shape mismatch: {name}")
    aliases: dict[int, str] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        canonical = aliases.setdefault(id(parameter), name)
        value = state[name].contiguous().reshape(-1).view(torch.uint8)
        reference = state[canonical].contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(value, reference):
            raise ValueError(f"FP16 checkpoint conflicting tied aliases: {name}")
    model.load_state_dict(state, strict=True)
    actual = parameter_fingerprint(model)
    if actual != str(identity.get("final_parameters_sha256", "")).lower():
        raise ValueError("FP16 checkpoint final parameters fingerprint differs from provenance")
    if actual == str(identity.get("initial_parameters_sha256", "")).lower():
        raise ValueError("FP16 checkpoint parameters did not update from initialization")
    return actual


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, object]:
    """Allowlisted versions/hardware only: never environment, URLs or credentials."""
    gpu = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "index": index,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "fla": _version("flash-linear-attention"),
        "fla_core": _version("fla-core"),
        "triton": _version("triton") or _version("triton-windows"),
        "clearml": _version("clearml"),
        "gpu": gpu,
    }


def initialized_identity(
    config: TrainConfig, model: torch.nn.Module, effective_seed: int
) -> dict[str, Any]:
    """Identity before dtype/device conversion, compile or optimizer construction."""
    return {
        "schema_version": 2,
        "status": "initialized",
        "requested_seed": config.schedule.seed,
        "effective_seed": effective_seed,
        "attention": config.model.attention,
        "initial_parameters_sha256": parameter_fingerprint(model),
        "final_parameters_sha256": None,
        "fingerprint_format": "sha256-sorted-unique-named-parameters-json-header-raw-bytes-v1",
        "initial_parameters_stage": "initialized_cpu_before_dtype_conversion",
        "runtime": runtime_metadata(),
        "config": config.model_dump(mode="json"),
        "compiler": compiler_metadata(config.schedule.compile),
    }


def record_identity(tracker: object, identity: dict[str, object]) -> None:
    """Optional tracker event preserves existing third-party tracker interfaces."""
    recorder = getattr(tracker, "record_identity", None)
    if recorder is not None:
        from compare_attention.train.lifecycle import failure_phase

        with failure_phase("identity_write"):
            recorder(identity)
