"""Requested FP16 scale caps must be evidenced in runtime metadata."""

from copy import deepcopy

import pytest

from compare_attention.flop_gate import verify_fp16_precision


def identity():
    return {
        "initial_parameters_sha256": "a" * 64,
        "final_parameters_sha256": "b" * 64,
        "config": {"schedule": {"amp_max_scale": 128.0}},
        "precision": {
            "mode": "amp_float16",
            "parameter_dtype": "float32",
            "autocast_dtype": "float16",
            "grad_scaler_enabled": True,
            "optimizer_steps": 2,
            "skipped_optimizer_steps": 0,
            "initial_scale": 128.0,
            "final_scale": 128.0,
            "max_scale": 128.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
    }


@pytest.mark.parametrize(
    "tamper", ["missing", "different", "exceeded", "bad_growth", "bool_interval"]
)
def test_requested_scale_cap_requires_valid_runtime_evidence(tamper):
    value = deepcopy(identity())
    precision = value["precision"]
    if tamper == "missing":
        precision.pop("max_scale")
    elif tamper == "different":
        precision["max_scale"] = 256.0
    elif tamper == "exceeded":
        precision["final_scale"] = 256.0
    elif tamper == "bad_growth":
        precision["growth_factor"] = float("nan")
    else:
        precision["growth_interval"] = True
    with pytest.raises(ValueError, match="scale|growth"):
        verify_fp16_precision(value, steps=2)


def test_requested_cap_valid_metadata_is_accepted():
    verify_fp16_precision(identity(), steps=2)
