"""Independent real-weight artifacts for exploratory-run verification."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.exploration_gate import verify_exploration_run
from compare_attention.flop_gate import verify_flops_smoke
from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.seed_gate import verify_seed_pair
from compare_attention.train.config import Optim, Schedule, TrainConfig
from compare_attention.train.flops import estimate_step
from compare_attention.train.identity import parameter_fingerprint
from compare_attention.train.stopping import EarlyStopping


def artifact(directory: Path, *, early=False, steps=None, cadence=2, dtype="float16"):
    """Construct legal recorded evidence, not a numerical training claim."""
    directory.mkdir()
    steps = steps or (4 if early else 5)
    config = TrainConfig.model_construct(
        data=Path("/data"),
        model=GPT2Config(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8),
        schedule=Schedule(
            train_flops=1,
            batch_size=1,
            seed=42,
            dtype=dtype,
            compile=True,
            val_every_steps=cadence,
            early_stopping_patience=1 if early else 0,
            early_stopping_min_steps=0,
        ),
        optim=Optim(),
        dataset=KVRetrievalConfig(
            vocab_size=8, input_seq_len=4, num_kv_pairs=1, n_train=2, n_val=1, n_test=1
        ),
    )
    cost = estimate_step(config).to_dict()
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"train_flops": cost["total"] * (10 if early else steps) - 1}
            )
        }
    )
    with torch.random.fork_rng(devices=[]):
        model = GPT2(config.model)
    initial = parameter_fingerprint(model)
    if dtype == "bfloat16":
        model.bfloat16()
    with torch.no_grad():
        model.wte.weight.add_(0.01)
    torch.save(model.state_dict(), directory / "model.pt")
    identity = {
        "schema_version": 2,
        "status": "early_stopped" if early else "completed",
        "stop_reason": "early_stopping" if early else "budget_exhausted",
        "requested_seed": 42,
        "effective_seed": 42,
        "attention": "classic",
        "initial_parameters_sha256": initial,
        "final_parameters_sha256": parameter_fingerprint(model),
        "runtime": {"gpu": {"name": "Tesla T4"}},
        "config": config.model_dump(mode="json"),
        "accounting": cost,
        "compiler": {
            "requested": True,
            "backend_invocations": 2,
            "training_backend_invocations": 1,
            "successful_backend_invocations": 2,
            "successful_training_backend_invocations": 1,
            "backend_invocation_verified": True,
        },
        "precision": {
            "mode": "amp_float16",
            "parameter_dtype": "float32",
            "autocast_dtype": "float16",
            "grad_scaler_enabled": True,
            "optimizer_steps": steps,
            "skipped_optimizer_steps": 0,
            "initial_scale": 65536.0,
            "final_scale": 65536.0,
        },
    }
    rows = []
    for index in range(1, steps + 1):
        rows.append({"type": "step", "loss": 2.0, "flops": index * cost["total"]})
        if index % cadence == 0 or index == steps:
            rows.append(
                {"type": "validation", "loss": 1.9, "accuracy": 0.1, "flops": index * cost["total"]}
            )
    rows.append(
        {
            "type": "result",
            "steps": steps,
            "train_flops": steps * cost["total"],
            "config": identity["config"],
            "provenance": identity,
            "stop_reason": identity["stop_reason"],
            "quality": {"train_loss": 2.0, "val_loss": 1.9, "val_accuracy": 0.1},
        }
    )
    persist(directory, rows)
    return rows


def persist(directory, rows):
    (directory / "run_identity.json").write_text(json.dumps(rows[-1]["provenance"]))
    (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "reason",
        "disabled",
        "too_soon",
        "late_stop",
        "forged_state",
        "budget_claim",
        "compiler",
        "seed",
        "cost",
        "step_count",
        "step_cost",
        "float_steps",
    ],
)
def test_early_metadata_cannot_forge_legitimate_stop(tmp_path, change):

    directory = tmp_path / "run"
    rows = artifact(directory, early=True, steps=6 if change == "late_stop" else None)
    result = rows[-1]
    identity = result["provenance"]
    schedule = result["config"]["schedule"]
    if change == "status":
        identity["status"] = "completed"
    elif change == "reason":
        result["stop_reason"] = "budget_exhausted"
    elif change == "disabled":
        schedule["early_stopping_patience"] = 0
    elif change == "too_soon":
        schedule["early_stopping_patience"] = 2
    elif change == "forged_state":
        result["early_stopping"] = {"best_loss": 0.0}
    elif change == "budget_claim":
        result["stop_reason"] = identity["stop_reason"] = "budget_exhausted"
        identity["status"] = "completed"
    elif change == "compiler":
        identity["compiler"]["successful_training_backend_invocations"] = 0
    elif change == "seed":
        identity["effective_seed"] = 123
    elif change == "cost":
        identity["accounting"]["total"] += 1
    elif change == "step_count":
        del rows[0]
    elif change == "step_cost":
        rows[0]["flops"] += 1
    elif change == "float_steps":
        result["steps"] = 4.0
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory)


def test_legal_early_stop_reports_actual_work_and_replayed_state(tmp_path):

    directory = tmp_path / "run"
    rows = artifact(directory, early=True)
    stopper = EarlyStopping(1, 0.001, 0)
    stopper.observe(1.9, 2)
    stopper.observe(1.9, 4)
    rows[-1]["early_stopping"] = stopper.state_dict()
    persist(directory, rows)
    report = verify_exploration_run(directory)
    assert report["passed"] and not report["budget_exhausted"]
    assert report["stop_reason"] == "early_stopping"
    assert report["steps"] == 4


@pytest.mark.parametrize(
    "change",
    [
        "off_cadence",
        "bool_patience",
        "bool_seed",
        "bool_loss",
        "unknown_record",
        "missing_model",
        "missing_dataset",
        "overshoot",
        "provenance",
        "missing_identity_reason",
    ],
)
def test_malformed_evidence_fails_closed(tmp_path, change):

    directory = tmp_path / "run"
    rows = artifact(directory, early=True, steps=3 if change == "off_cadence" else None)
    identity = rows[-1]["provenance"]
    if change == "bool_patience":
        rows[-1]["config"]["schedule"]["early_stopping_patience"] = True
    elif change == "bool_seed":
        rows[-1]["config"]["schedule"]["seed"] = True
        identity["requested_seed"] = identity["effective_seed"] = 1
    elif change == "bool_loss":
        rows[0]["loss"] = True
        rows[-1]["quality"]["train_loss"] = 1.75
    elif change == "unknown_record":
        rows.insert(0, {"type": "unrecorded_update"})
    elif change == "missing_model":
        del rows[-1]["config"]["model"]["n_head"]
    elif change == "missing_dataset":
        del rows[-1]["config"]["dataset"]["seed"]
    elif change == "overshoot":
        rows[-1]["config"]["schedule"]["train_flops"] = 1
        rows[-1]["stop_reason"] = identity["stop_reason"] = "budget_exhausted"
        identity["status"] = "completed"
    elif change == "missing_identity_reason":
        del identity["stop_reason"]
    elif change == "provenance":
        persist(directory, rows)
        identity = json.loads((directory / "run_identity.json").read_text())
        identity["effective_seed"] = 0
        (directory / "run_identity.json").write_text(json.dumps(identity))
        with pytest.raises(ValueError):
            verify_exploration_run(directory)
        return
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory)


@pytest.mark.parametrize(
    "change",
    [
        "early_loss",
        "validation_loss",
        "validation_accuracy",
        "quality",
        "missing_validation",
        "duplicate_validation",
        "missing_result",
        "duplicate_result",
        "trailing_record",
        "out_of_order",
        "checkpoint",
        "half_checkpoint",
        "forged_hash",
        "missing_fingerprint",
        "wrong_gpu",
        "expected_config",
    ],
)
def test_quality_checkpoint_and_identity_evidence_are_independently_verified(tmp_path, change):
    # Explicit independent artifact corruptions keep the rejection matrix inspectable.
    # pylint: disable=too-many-branches,too-many-statements

    directory = tmp_path / "run"
    rows = artifact(directory, early=True)
    kwargs = {}
    if change == "early_loss":
        rows[0]["loss"] = float("nan")
    elif change == "validation_loss":
        rows[2]["loss"] = float("inf")
    elif change == "validation_accuracy":
        rows[2]["accuracy"] = float("nan")
    elif change == "quality":
        rows[-1]["quality"]["val_loss"] = 3.0
    elif change == "missing_validation":
        del rows[2]
    elif change == "duplicate_validation":
        rows.insert(-1, rows[-2].copy())
    elif change == "missing_result":
        persist(directory, rows)
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:-1]))
        with pytest.raises(ValueError):
            verify_exploration_run(directory)
        return
    elif change == "duplicate_result":
        rows.insert(-1, rows[-1].copy())
    elif change == "trailing_record":
        persist(directory, rows)
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write('\n{"type":"step","loss":2.0,"flops":100}')
        with pytest.raises(ValueError):
            verify_exploration_run(directory)
        return
    elif change == "out_of_order":
        rows[1], rows[2] = rows[2], rows[1]
    elif change in {"checkpoint", "half_checkpoint"}:
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        if change == "checkpoint":
            state["wte.weight"].add_(1)
        else:
            state = {key: value.half() for key, value in state.items()}
        torch.save(state, directory / "model.pt")
    elif change == "forged_hash":
        rows[-1]["provenance"]["final_parameters_sha256"] = "b" * 64
    elif change == "missing_fingerprint":
        del rows[-1]["provenance"]["initial_parameters_sha256"]
    elif change == "wrong_gpu":
        kwargs["expected_gpu"] = "NVIDIA L4"
    elif change == "expected_config":
        expected = deepcopy(rows[-1]["config"])
        expected["dataset"]["seed"] += 1
        kwargs["expected_config"] = expected
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory, **kwargs)


def test_exploration_does_not_relax_fixed_budget_gates(tmp_path):

    directory = tmp_path / "run"
    rows = artifact(directory, early=True)
    with pytest.raises(ValueError, match="identity"):
        verify_flops_smoke(
            directory,
            attention="classic",
            expected=rows[-1]["provenance"]["accounting"],
            expected_steps=4,
        )
    with pytest.raises(ValueError, match="completed"):
        verify_seed_pair(directory, directory)


@pytest.mark.parametrize("field", ["loss", "flops"])
def test_missing_metric_fields_fail_with_validation_error(tmp_path, field):

    directory = tmp_path / "run"
    rows = artifact(directory, early=True)
    del rows[0][field]
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory)


def test_controller_state_boolean_counter_cannot_equal_integer_metadata(tmp_path):
    directory = tmp_path / "run"
    rows = artifact(directory, early=True)
    stopper = EarlyStopping(1, 0.001, 0)
    stopper.observe(1.9, 2)
    stopper.observe(1.9, 4)
    state = stopper.state_dict()
    state["bad_validations"] = True
    rows[-1]["early_stopping"] = state
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory)


@pytest.mark.parametrize("steps,cadence", [(1, 100), (4, 2), (5, 2)])
def test_completed_validation_boundaries_and_final_controller_state(tmp_path, steps, cadence):
    directory = tmp_path / "run"
    rows = artifact(directory, steps=steps, cadence=cadence)
    stopper = EarlyStopping(0, 0.001, 0)
    cost = rows[-1]["provenance"]["accounting"]["total"]
    for row in rows:
        if row["type"] == "validation":
            stopper.observe(row["loss"], row["flops"] // cost)
    rows[-1]["early_stopping"] = stopper.state_dict()
    persist(directory, rows)
    assert verify_exploration_run(directory)["budget_exhausted"]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_non_fp16_checkpoints_are_bound_without_requiring_amp(tmp_path, dtype):
    directory = tmp_path / "run"
    rows = artifact(directory, early=True, dtype=dtype)
    del rows[-1]["provenance"]["precision"]
    persist(directory, rows)
    assert verify_exploration_run(directory)["passed"]
    initial = rows[-1]["provenance"].pop("initial_parameters_sha256")
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory)
    rows[-1]["provenance"]["initial_parameters_sha256"] = initial
    persist(directory, rows)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    state["wte.weight"].add_(1)
    torch.save(state, directory / "model.pt")
    with pytest.raises(ValueError):
        verify_exploration_run(directory)


def test_completed_budget_report_uses_actual_cost_and_safe_checkpoint(tmp_path):

    directory = tmp_path / "run"
    rows = artifact(directory)
    report = verify_exploration_run(
        directory, expected_config=rows[-1]["config"], expected_gpu="Tesla T4"
    )
    assert report["passed"] and report["budget_exhausted"]
    assert report["stop_reason"] == "budget_exhausted"
    assert report["steps"] == 5
    assert report["train_flops"] == rows[-1]["train_flops"]
    assert report["quality"] == rows[-1]["quality"]
    assert report["directory"] == str(directory.resolve())
