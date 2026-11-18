"""Exact cloud FULL code replay with model-free constructor transport only."""

import builtins
import contextlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace


def metadata_full(function):
    """Keep its code object and all native predicates; replace host constructors."""
    from seed202_pilot import mqar_tensor_schema

    blueprint = json.loads(
        Path(mqar_tensor_schema.__file__).with_name("mqar-tensor-schema.json").read_bytes()
    )
    active = {}

    class ModelMetadata:
        def __init__(self, config):
            assert config.model_dump(mode="json") == active["payload"]["config"]["model"]

        def float(self):
            return self

        def parameters(self):
            return ()

        def load_state_dict(self, state, strict):
            assert state is active["payload"]["model"] and strict is True
            mqar_tensor_schema.validate_model_optimizer(active["payload"])

    def validate_model(state, _model):
        assert state is active["payload"]["model"]
        mqar_tensor_schema.validate_model_optimizer(active["payload"])

    cp = SimpleNamespace(
        _validate_model=validate_model,
        _model_types=lambda _model: [tuple(row) for row in blueprint["model_structure"]],
        _optimizer_names=lambda *_args: blueprint["optimizer_parameters"],
    )
    torch_metadata = SimpleNamespace(
        random=SimpleNamespace(fork_rng=lambda **_kwargs: contextlib.nullcontext()),
        device=lambda *_args: contextlib.nullcontext(),
        optim=SimpleNamespace(AdamW=lambda *_args, **_kwargs: object()),
    )

    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            if name == "torch" and not fromlist:
                return torch_metadata
            if name == "compare_attention.models.gpt2" and fromlist == ("GPT2",):
                return SimpleNamespace(GPT2=ModelMetadata)
            if name == "compare_attention.train" and fromlist == ("checkpoint",):
                return SimpleNamespace(checkpoint=cp)
            if name == "compare_attention.train.checkpoint_optimizer" and fromlist == (
                "_validate_optimizer",
            ):
                return SimpleNamespace(
                    _validate_optimizer=lambda *_args: mqar_tensor_schema.validate_model_optimizer(
                        active["payload"]
                    )
                )
            if name == "compare_attention.train.identity" and fromlist == (
                "parameter_fingerprint",
            ):
                return SimpleNamespace(
                    parameter_fingerprint=lambda _model: (
                        mqar_tensor_schema.validate_model_optimizer(active["payload"])
                    )
                )
        return builtins.__import__(name, globals, locals, fromlist, level)

    scope = {**function.__globals__, "__builtins__": {**vars(builtins), "__import__": imports}}
    exact = FunctionType(function.__code__, scope, function.__name__, function.__defaults__)
    exact.__kwdefaults__ = function.__kwdefaults__
    assert exact.__code__ is function.__code__  # pylint: disable=no-member

    def validate(payload, case, **kwargs):
        active["payload"] = payload
        try:
            return exact(payload, case, **kwargs)
        finally:
            active.clear()

    return validate
