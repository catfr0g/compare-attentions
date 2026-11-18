"""Model-free tensor and optimizer schema predicates for host native replay."""

import json
from pathlib import Path


def validate_model_optimizer(payload):
    """Use the original SHA-bound schema, preserving strict native shape/order/options."""
    import torch
    from seed202_pilot.cloud_contract import canonical

    from compare_attention.train.identity import parameter_fingerprint

    blueprint = json.loads(Path(__file__).with_name("mqar-tensor-schema.json").read_bytes())

    def require(value, code):
        if not value:
            raise ValueError(code)

    require(payload["config"]["model"]["attention"] == "classic", "MQAR_SCHEMA_FAMILY")
    expected_structure = [tuple(row) for row in blueprint["model_structure"]]
    require(payload["model_structure"] == expected_structure, "FULL_MODEL_STRUCTURE")
    require(
        payload["optimizer_parameters"] == blueprint["optimizer_parameters"], "FULL_OPTIMIZER_ORDER"
    )
    require(payload["optimizer_class"] == "torch.optim.adamw.AdamW", "FULL_OPTIMIZER_CLASS")
    state = payload["model"]
    require(set(state) == set(blueprint["model"]), "FULL_MODEL_NAMES")
    for name, spec in blueprint["model"].items():
        value = state[name]
        require(
            isinstance(value, torch.Tensor)
            and value.layout == torch.strided
            and list(value.shape) == spec["shape"]
            and str(value.dtype) == spec["dtype"]
            and bool(torch.isfinite(value).all()),
            "FULL_MODEL_TENSOR_SCHEMA",
        )
    require(torch.equal(state["wte.weight"], state["lm_head.weight"]), "FULL_MODEL_TIED_ALIAS")
    optimizer = payload["optimizer"]
    require(
        type(optimizer) is dict and set(optimizer) == {"state", "param_groups"},
        "FULL_OPTIMIZER_SCHEMA",
    )
    require(len(optimizer["param_groups"]) == 1, "FULL_OPTIMIZER_GROUPS")
    group = optimizer["param_groups"][0]
    expected_group = dict(blueprint["optimizer_group"])
    expected_group["betas"] = tuple(expected_group["betas"])
    require(
        group == expected_group and canonical(group) == canonical(expected_group),
        "FULL_OPTIMIZER_OPTIONS_AND_IDS",
    )
    names = dict(zip(group["params"], blueprint["optimizer_parameters"][0], strict=True))
    require(set(optimizer["state"]) == set(names), "FULL_OPTIMIZER_INCOMPLETE")
    for identifier, saved in optimizer["state"].items():
        require(
            type(saved) is dict and set(saved) == {"step", "exp_avg", "exp_avg_sq"},
            "FULL_OPTIMIZER_STATE_KEYS",
        )
        step = saved["step"]
        scalar = step.item() if isinstance(step, torch.Tensor) and step.numel() == 1 else step
        require(type(scalar) in (int, float) and scalar == payload["step"], "FULL_ADAM_STEP")
        for name in ("exp_avg", "exp_avg_sq"):
            value = saved[name]
            reference = state[names[identifier]]
            require(
                isinstance(value, torch.Tensor)
                and value.layout == torch.strided
                and value.shape == reference.shape
                and value.dtype == reference.dtype
                and bool(torch.isfinite(value).all()),
                "FULL_ADAM_TENSOR_SCHEMA",
            )

    class SavedTensorView:
        def named_parameters(self):
            return [(name, state[name]) for name in blueprint["optimizer_parameters"][0]]

    final = parameter_fingerprint(SavedTensorView())
    require(final != payload["identity"]["initial_parameters_sha256"], "FULL_UNCHANGED_WEIGHTS")
    return final
