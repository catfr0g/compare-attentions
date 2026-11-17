"""Read and validate evaluation weights without resuming any training state."""

import json
from pathlib import Path

import torch
import yaml

from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.checkpoint import _validate_model, load_checkpoint
from compare_attention.train.identity import parameter_fingerprint


def _model_config(mapping):
    if not isinstance(mapping, dict) or set(mapping) != set(GPT2Config.model_fields):
        raise ValueError("complete checkpoint model config is required")
    return GPT2Config.model_validate(mapping)


# Multiple supported formats share one explicit, fail-closed loading boundary.
# pylint: disable-next=too-many-branches
def load_evaluation_model(checkpoint, config=None):
    """Accept FP32 state dict + identity/YAML, or an integrity-checked full checkpoint.

    Full checkpoints supply weights/config only: optimizer, scaler and RNG are never
    restored. Shape/dtype/finite/tied-alias checks precede load_state_dict.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    pointer = None
    model_config = None
    if checkpoint.suffix == ".json":
        payload = load_checkpoint(checkpoint)
        pointer = checkpoint
        checkpoint = checkpoint.parent / json.loads(checkpoint.read_text(encoding="utf-8"))["path"]
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    full = isinstance(payload, dict) and "schema_version" in payload and "model" in payload
    identity = None
    if full:
        payload = load_checkpoint(pointer or checkpoint)
        model_config = _model_config(payload["config"]["model"])
        identity = payload["identity"]
        if not isinstance(identity, dict) or identity.get("config") != payload["config"]:
            raise ValueError("checkpoint identity/config mismatch")
        state = payload["model"]
    else:
        state = payload
        identity_path = checkpoint.parent / "run_identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            model_config = _model_config(identity.get("config", {}).get("model"))
            if identity.get("attention") != model_config.attention:
                raise ValueError("checkpoint attention/config mismatch")
        elif config is None:
            raise ValueError("state dict requires run_identity.json or --config YAML")
    if config is not None:
        explicit = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        if not isinstance(explicit, dict):
            raise ValueError("YAML config must contain a complete model mapping")
        requested = _model_config(explicit.get("model"))
        if identity is not None and requested != model_config:
            raise ValueError("explicit config conflicts with checkpoint config")
        model_config = requested
    if model_config is None:
        raise ValueError("checkpoint model config missing")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = GPT2(model_config).float()
    _validate_model(state, model)
    model.load_state_dict(state, strict=True)
    fingerprint = parameter_fingerprint(model)
    if identity is not None and not full:
        if identity.get("final_parameters_sha256") != fingerprint:
            raise ValueError("checkpoint final parameters fingerprint mismatch")
    return model, {
        "path": checkpoint.resolve(),
        "pointer": pointer,
        "provenance": identity,
        "parameters_sha256": fingerprint,
        "format": "training_checkpoint" if full else "state_dict",
        "step": payload["step"] if full else None,
    }
