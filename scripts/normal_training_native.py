"""NORMAL FULL structural replay; no saved model/optimizer/RNG restoration or forward."""

import json
from pathlib import Path

from normal_training_plan import canonical, require


def validate_precision(payload, config, step):
    """Checkpoint scaler state has a different schema from final identity metadata."""
    precision = payload["precision"]
    require(
        type(precision) is dict
        and set(precision) == {"scaler", "optimizer_steps", "skipped_optimizer_steps"}
        and type(precision["optimizer_steps"]) is int
        and precision["optimizer_steps"] == step
        and type(precision["skipped_optimizer_steps"]) is int
        and precision["skipped_optimizer_steps"] == 0,
        "NORMAL_FULL_PRECISION_COUNTERS",
    )
    require(
        config["schedule"]["amp_max_scale"] == 128.0
        and canonical(precision["scaler"])
        == canonical(
            {
                "scale": 128.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": step % 2000,
            }
        ),
        "NORMAL_FULL_SCALER",
    )


def validate_full(payload, case, *, step):
    """Direct callback validator with recipe-derived seed/task/topology, no old default."""
    import torch
    from quality_inherit import native_checks

    from compare_attention.models.config import GPT2Config
    from compare_attention.models.gpt2 import GPT2
    from compare_attention.train import checkpoint as cp
    from compare_attention.train.distributed_checkpoint import validate
    from compare_attention.train.identity import parameter_fingerprint
    from compare_attention.train.stopping import EarlyStopping

    validate(payload)
    config = case["config"]
    require(canonical(payload["config"]) == canonical(config), "NORMAL_FULL_CONFIG")
    require(
        type(step) is int and payload["step"] == step and 0 < step <= case["updates"],
        "NORMAL_FULL_HORIZON",
    )
    require(
        payload["flops"] == step * case["step_flops"]
        and payload["cursor"] == step * case["global_batch"] % config["dataset"]["n_train"],
        "NORMAL_FULL_BUDGET_CURSOR",
    )
    topology = {
        "world_size": 2,
        "global_batch_size": case["global_batch"],
        "local_batch_size": case["local_batch"],
        "sharding": "ordered_cyclic_contiguous_v1",
        "device_type": "cpu" if case["engineering_only"] else "cuda",
    }
    require(payload["distributed"]["topology"] == topology, "NORMAL_FULL_TOPOLOGY")
    checks = native_checks()
    # Authenticated AST-loaded helper, not a statically declared module member.
    checks.finite_full_state(payload)  # pylint: disable=no-member
    stopper = EarlyStopping(
        config["schedule"]["early_stopping_patience"],
        config["schedule"]["early_stopping_min_delta"],
        config["schedule"]["early_stopping_min_steps"],
    )
    history = payload["validation_history"]
    require(
        [r["flops"] for r in history]
        == [s * case["step_flops"] for s in case["validation_steps"] if s <= step],
        "NORMAL_FULL_VAL_GRID",
    )
    for row in history:
        require(
            set(row) == {"type", "loss", "accuracy", "flops"}
            and row["type"] == "validation"
            and row["loss"] >= 0
            and 0 <= row["accuracy"] <= 1,
            "NORMAL_FULL_VAL_SCHEMA",
        )
        stopper.observe(row["loss"], row["flops"] // case["step_flops"])
    require(
        stopper.state_dict() == payload["early_stopping"]
        and len(payload["losses"]) == step
        and all(v >= 0 for v in payload["losses"]),
        "NORMAL_FULL_HISTORY",
    )
    for rank, row in enumerate(payload["distributed"]["ranks"]):
        identity = row["identity"]
        require(
            identity["requested_seed"] == identity["effective_seed"] == case["seed"]
            and identity["attention"] == case["family"]
            and canonical(identity["accounting"]) == canonical(case["accounting"]),
            "NORMAL_FULL_DECLARED_IDENTITY",
        )
        if not case["engineering_only"]:
            checks.target_runtime(identity["runtime"], rank)  # pylint: disable=no-member
            checks.compiler_evidence(identity["compiler"])  # pylint: disable=no-member
    # Meta construction establishes independent names/shapes/types without weights,
    # allocation, optimizer construction, forward, seed calls or RNG restoration.
    with torch.device("meta"):
        model = GPT2(GPT2Config.model_validate(config["model"]))
    reference = model.state_dict()
    state = payload["model"]
    require(
        set(state) == set(reference) and payload["model_structure"] == cp._model_types(model),
        "NORMAL_FULL_MODEL_SCHEMA",
    )  # pylint: disable=protected-access
    for name, value in state.items():
        require(
            isinstance(value, torch.Tensor)
            and value.layout == torch.strided
            and value.shape == reference[name].shape
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all()),
            "NORMAL_FULL_MODEL_TENSOR",
        )
    require(torch.equal(state["wte.weight"], state["lm_head.weight"]), "NORMAL_FULL_TIED_WEIGHTS")
    names = [name for name, _ in model.named_parameters()]
    require(
        payload["optimizer_parameters"] == [names]
        and payload["optimizer_class"] == "torch.optim.adamw.AdamW",
        "NORMAL_FULL_OPTIMIZER_ORDER",
    )
    optimizer = payload["optimizer"]
    require(
        set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
        "NORMAL_FULL_OPTIMIZER_SCHEMA",
    )
    group = optimizer["param_groups"][0]
    require(
        group["params"] == list(range(len(names)))
        and set(optimizer["state"]) == set(group["params"]),
        "NORMAL_FULL_OPTIMIZER_IDS",
    )
    from compare_attention.train.config import Optim
    from compare_attention.train.lr_schedule import next_learning_rate

    expected_lr = next_learning_rate(
        Optim(**config["optim"]),
        config["schedule"]["train_flops"],
        (step - 1) * case["step_flops"],
        case["step_flops"],
    )
    require(
        group["lr"] == expected_lr
        and group["weight_decay"] == config["optim"]["weight_decay"]
        and tuple(group["betas"]) == (0.9, 0.999)
        and group["eps"] == 1e-8
        and group["amsgrad"] is False
        and group["maximize"] is False,
        "NORMAL_FULL_OPTIMIZER_OPTIONS",
    )
    for identifier, saved in optimizer["state"].items():
        require(
            set(saved) == {"step", "exp_avg", "exp_avg_sq"} and saved["step"].item() == step,
            "NORMAL_FULL_ADAM_COUNTER",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            value = saved[key]
            require(
                value.shape == state[names[identifier]].shape
                and value.dtype == torch.float32
                and bool(torch.isfinite(value).all()),
                "NORMAL_FULL_ADAM_MOMENTS",
            )
    if case["engineering_only"]:
        require(payload["precision"] is None, "NORMAL_ENGINEERING_PRECISION")
    else:
        validate_precision(payload, config, step)

    class TensorView:
        def named_parameters(self):
            return [(name, state[name]) for name in names]

    final = parameter_fingerprint(TensorView())
    require(final != payload["identity"]["initial_parameters_sha256"], "NORMAL_FULL_NO_UPDATES")
    return {
        "updates": step,
        "final_sha256": final,
        "initial_sha256": payload["identity"]["initial_parameters_sha256"],
        "world_size": 2,
    }


def verify_native(root, case):
    """Read actual final/retained checkpoints and all native science events."""
    import torch
    from quality_recovery import snapshot_inventory

    from compare_attention.train.checkpoint import load_checkpoint

    root = Path(root)
    identity = json.loads((root / "run_identity.json").read_bytes())
    events = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    result = events[-1]
    require(
        result["type"] == "result"
        and result["provenance"] == identity
        and identity["status"] == "completed"
        and identity["stop_reason"] == "budget_exhausted"
        and result["steps"] == case["updates"]
        and result["train_flops"] == case["actual_flops"],
        "NORMAL_NATIVE_RESULT",
    )
    require(
        canonical(identity["config"]) == canonical(case["config"])
        and canonical(identity["accounting"]) == canonical(case["accounting"]),
        "NORMAL_NATIVE_CONFIG",
    )
    losses = [row for row in events if row["type"] == "step"]
    vals = [row for row in events if row["type"] == "validation"]
    rates = [row for row in events if row["type"] == "learning_rate"]
    require(
        len(events) == 1 + len(losses) + len(vals) + len(rates)
        and len(losses) == case["updates"]
        and [r["flops"] for r in losses]
        == [s * case["step_flops"] for s in range(1, case["updates"] + 1)],
        "NORMAL_NATIVE_TRACE",
    )
    require(
        [r["flops"] for r in vals] == [s * case["step_flops"] for s in case["validation_steps"]],
        "NORMAL_NATIVE_VAL",
    )
    require(
        canonical(result["config"]) == canonical(case["config"])
        and result["quality"]
        == {
            "train_loss": sum(row["loss"] for row in losses) / len(losses),
            "val_loss": vals[-1]["loss"],
            "val_accuracy": vals[-1]["accuracy"],
        },
        "NORMAL_NATIVE_QUALITY",
    )
    inventory = snapshot_inventory(root / "checkpoints", case["checkpoint_steps"])
    for row in inventory["retained"]:
        payload = load_checkpoint(root / "checkpoints" / row["path"])
        proof = validate_full(payload, case, step=row["step"])
        require(
            payload["losses"] == [r["loss"] for r in losses[: row["step"]]]
            and payload["validation_history"]
            == [r for r in vals if r["flops"] <= payload["flops"]],
            "NORMAL_NATIVE_FULL_PREFIX",
        )
    require(
        proof["updates"] == case["updates"]
        and proof["initial_sha256"] == identity["initial_parameters_sha256"]
        and proof["final_sha256"] == identity["final_parameters_sha256"],
        "NORMAL_NATIVE_FINAL_IDENTITY",
    )
    weights = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    require(
        set(weights) == set(payload["model"])
        and all(torch.equal(value, payload["model"][name]) for name, value in weights.items()),
        "NORMAL_NATIVE_LAST_WEIGHTS",
    )
    for rank, row in enumerate(identity["distributed"]["ranks"]):
        require(
            row["rank"] == row["local_rank"] == rank
            and row["status"] == "completed"
            and row["step_flops"] == case["rank_step_flops"]
            and row["final_parameters_sha256"] == proof["final_sha256"],
            "NORMAL_NATIVE_RANKS",
        )
    return {
        **proof,
        "retained_steps": [row["step"] for row in inventory["retained"]],
        "structural_only": True,
    }
