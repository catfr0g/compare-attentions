"""Exact authenticated pure saved-tensor inspector; task identity supplied separately."""

from mqar_final_tail_native import FINAL, canonical, require


def inspect_full(payload, result, case=None):
    """Inspect real saved unique parameters and own-rank compiler/runtime evidence."""
    import torch
    from quality_inherit import native_checks

    from compare_attention.train.distributed_checkpoint import validate
    from compare_attention.train.identity import parameter_fingerprint

    validate(payload)
    checks = native_checks()
    checks.finite_full_state(payload)  # pylint: disable=no-member  # authenticated AST leaf
    identity, step = result["provenance"], payload["step"]
    require(payload["config"] == identity["config"] == result["config"], "FULL_RESULT_CONFIG")
    if case:
        require(canonical(payload["config"]) == canonical(case["config"]), "FULL_POLICY_CONFIG")
        require(
            canonical(identity["accounting"]) == canonical(case["accounting"]), "FULL_ACCOUNTING"
        )
        require(
            step == case["updates"] and payload["flops"] == case["actual_flops"], "FULL_HORIZON"
        )
    require(
        step == result["steps"] and payload["flops"] == result["train_flops"], "FULL_RESULT_BUDGET"
    )
    require(
        payload["cursor"] == (step * 64) % payload["config"]["dataset"]["n_train"], "FULL_CURSOR"
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
        "FULL_TOPOLOGY",
    )
    ranks, wires = payload["distributed"]["ranks"], identity["distributed"]["ranks"]
    require(
        [r["rank"] for r in ranks] == [0, 1] and [r["rank"] for r in wires] == [0, 1], "EXACT_RANKS"
    )
    for row, wire in zip(ranks, wires, strict=True):
        rank, original = row["rank"], row["identity"]
        require(type(wire["rank"]) is int and type(rank) is int, "TYPED_RANKS")
        checks.target_runtime(original["runtime"], rank)  # pylint: disable=no-member
        checks.compiler_evidence(original["compiler"])  # pylint: disable=no-member
        checks.compiler_evidence(wire["compiler"])  # pylint: disable=no-member
        checks.precision_evidence(wire["precision"], step)  # pylint: disable=no-member
        require(wire["compiler"] == original["compiler"], "RANK_COMPILER_JOIN")
        require(original["config"] == payload["config"], "RANK_CONFIG")
        require(
            canonical(original["accounting"]) == canonical(identity["accounting"]),
            "RANK_ACCOUNTING",
        )
        require(wire["effective_seed"] == original["effective_seed"] == 202, "RANK_SEED")
        require(
            wire["initial_parameters_sha256"] == original["initial_parameters_sha256"],
            "RANK_INITIAL",
        )
        require(
            wire["final_parameters_sha256"] == FINAL
            and wire["status"] == "completed"
            and wire["stop_reason"] == "budget_exhausted",
            "RANK_TERMINATION",
        )
    require(
        payload["precision"]
        == {
            "optimizer_steps": step,
            "skipped_optimizer_steps": 0,
            "scaler": {
                "scale": 128.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": step % 2000,
            },
        },
        "FULL_SCALER_COUNTERS",
    )
    names, weights, saved = payload["optimizer_parameters"], payload["model"], payload["optimizer"]
    require(len(names) == 1 and len(names[0]) == len(set(names[0])), "UNIQUE_PARAMETERS")
    require(set(weights) - set(names[0]) == {"lm_head.weight"}, "PARAMETER_KEYS")
    require(torch.equal(weights["lm_head.weight"], weights["wte.weight"]), "TIED_PARAMETERS")
    require(
        all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in weights.values()),
        "MODEL_FP32_FINITE",
    )
    require(payload["optimizer_class"] == "torch.optim.adamw.AdamW", "OPTIMIZER_CLASS")
    expected = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.1,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
        "params": list(range(len(names[0]))),
    }
    require(
        set(saved) == {"param_groups", "state"} and saved["param_groups"] == [expected],
        "OPTIMIZER_OPTIONS",
    )
    require(set(saved["state"]) == set(expected["params"]), "OPTIMIZER_COMPLETENESS")
    for index, name in enumerate(names[0]):
        state = saved["state"][index]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "OPTIMIZER_SCHEMA")
        require(state["step"].numel() == 1 and state["step"].item() == step, "OPTIMIZER_STEP")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            require(
                value.shape == weights[name].shape
                and value.dtype == torch.float32
                and value.layout == torch.strided
                and bool(torch.isfinite(value).all()),
                "OPTIMIZER_TENSOR",
            )

    class SavedParameters:
        """Native fingerprint protocol applied only to genuine saved unique tensors."""

        def named_parameters(self):
            return [(name, weights[name]) for name in names[0]]

    require(
        parameter_fingerprint(SavedParameters()) == identity["final_parameters_sha256"] == FINAL,
        "FINAL_PARAMETERS",
    )
    require(FINAL != identity["initial_parameters_sha256"], "UNCHANGED_PARAMETERS")
    return len(names[0])
