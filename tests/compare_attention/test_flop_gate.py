"""Cloud FLOP smoke must pass before a full comparison is submitted."""

# Independent artifact fixtures intentionally exercise each public gate separately.
# pylint: disable=duplicate-code

import json
from pathlib import Path

import pytest
import torch

from compare_attention.flop_gate import verify_flops_smoke
from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.identity import parameter_fingerprint


def _smoke(directory: Path, gpu: str = "Tesla T4") -> dict:
    directory.mkdir()
    estimate = {"policy": "algorithmic_matmul_conv_v2", "forward": 30, "backward": 70, "total": 100}
    identity = {
        "schema_version": 2,
        "status": "completed",
        "requested_seed": 42,
        "effective_seed": 42,
        "attention": "classic",
        "accounting": estimate,
        "compiler": {
            "requested": True,
            "backend_invocations": 2,
            "training_backend_invocations": 1,
            "validation_backend_invocations": 1,
            "successful_backend_invocations": 2,
            "successful_training_backend_invocations": 1,
            "successful_validation_backend_invocations": 1,
            "backend_invocation_verified": True,
        },
        "runtime": {"gpu": {"name": gpu}},
    }
    rows = [
        {"type": "step", "loss": 2.0, "flops": 100},
        {"type": "validation", "loss": 1.9, "accuracy": 0.01, "flops": 100},
        {
            "type": "result",
            "steps": 1,
            "train_flops": 100,
            "quality": {"train_loss": 2.0, "val_loss": 1.9, "val_accuracy": 0.01},
            "config": {
                "model": {"attention": "classic"},
                "schedule": {"seed": 42, "train_flops": 100, "compile": True},
            },
        },
    ]
    rows[-1]["provenance"] = identity
    (directory / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (directory / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    return estimate


def _fp16_smoke(directory: Path, **updates) -> dict:
    estimate = _smoke(directory)
    identity_path = directory / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["config"]["schedule"]["dtype"] = "float16"
    identity["config"] = rows[-1]["config"]
    model_config = GPT2Config(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8)
    rows[-1]["config"]["model"] = model_config.model_dump(mode="json")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
        model = GPT2(model_config)
    identity["initial_parameters_sha256"] = parameter_fingerprint(model)
    with torch.no_grad():
        model.wte.weight.add_(0.01)
    identity["final_parameters_sha256"] = parameter_fingerprint(model)
    torch.save(model.state_dict(), directory / "model.pt")
    identity["precision"] = {
        "mode": "amp_float16",
        "parameter_dtype": "float32",
        "autocast_dtype": "float16",
        "grad_scaler_enabled": True,
        "optimizer_steps": 1,
        "skipped_optimizer_steps": 0,
        "initial_scale": 65536.0,
        "final_scale": 65536.0,
    }
    identity["precision"].update(updates)
    rows[-1]["provenance"] = identity
    identity_path.write_text(json.dumps(identity))
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    return estimate


@pytest.mark.parametrize(
    "change", ["forged", "tampered", "unupdated", "half", "keys", "shape", "alias"]
)
def test_fp16_forged_final_fingerprint_is_rejected(tmp_path: Path, change: str) -> None:
    directory = tmp_path / "smoke"
    expected = _fp16_smoke(directory)
    path = directory / "run_identity.json"
    identity = json.loads(path.read_text())
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if change == "forged":
        identity["final_parameters_sha256"] = "b" * 64
    elif change == "tampered":
        state["wte.weight"].add_(1)
    elif change == "unupdated":
        config = GPT2Config.model_validate(identity["config"]["model"])
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            state = GPT2(config).state_dict()
    elif change == "half":
        state = {key: value.half() for key, value in state.items()}
    elif change == "keys":
        del state["wpe.weight"]
    elif change == "shape":
        state["wpe.weight"] = state["wpe.weight"][:1]
    else:
        state["lm_head.weight"] = state["lm_head.weight"].clone().add_(1)
    torch.save(state, directory / "model.pt")
    path.write_text(json.dumps(identity))
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["provenance"] = identity
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="checkpoint"):
        verify_flops_smoke(directory, attention="classic", expected=expected)


def test_fp16_scaled_one_actual_update_passes(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    expected = _fp16_smoke(directory)
    before_rng = torch.get_rng_state().clone()
    assert verify_flops_smoke(
        directory, attention="classic", expected=expected, expected_dtype="float16"
    )["passed"]
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "updates",
    [
        {"parameter_dtype": "float16"},
        {"grad_scaler_enabled": False},
        {"autocast_dtype": "bfloat16"},
        {"mode": "legacy"},
        {"optimizer_steps": 0},
        {"optimizer_steps": True},
        {"skipped_optimizer_steps": 1},
        {"skipped_optimizer_steps": False},
        {"initial_scale": 0.0},
        {"final_scale": float("inf")},
        {"final_scale": float("nan")},
        {"final_scale": -1.0},
    ],
)
def test_fp16_invalid_precision_cannot_pass(tmp_path: Path, updates: dict) -> None:
    directory = tmp_path / "smoke"
    expected = _fp16_smoke(directory, **updates)
    with pytest.raises(ValueError, match="precision"):
        verify_flops_smoke(
            directory, attention="classic", expected=expected, expected_dtype="float16"
        )


@pytest.mark.parametrize("change", ["missing_precision", "wrong_dtype", "config_mismatch"])
def test_fp16_requires_matched_config_and_precision(tmp_path: Path, change: str) -> None:
    directory = tmp_path / "smoke"
    expected = _fp16_smoke(directory)
    identity_path = directory / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    if change == "missing_precision":
        del identity["precision"]
    elif change == "wrong_dtype":
        identity["config"]["schedule"]["dtype"] = "bfloat16"
        rows[-1]["config"] = identity["config"]
    else:
        identity["config"]["schedule"]["batch_size"] = 16
    rows[-1]["provenance"] = identity
    identity_path.write_text(json.dumps(identity))
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="precision|dtype|configuration"):
        verify_flops_smoke(
            directory, attention="classic", expected=expected, expected_dtype="float16"
        )


@pytest.mark.parametrize("value", [None, "", "a" * 64, "z" * 64, "b" * 63])
def test_fp16_requires_distinct_valid_parameter_hashes(tmp_path: Path, value) -> None:
    directory = tmp_path / "smoke"
    expected = _fp16_smoke(directory)
    path = directory / "run_identity.json"
    identity = json.loads(path.read_text())
    identity["final_parameters_sha256"] = (
        identity["initial_parameters_sha256"] if value == "a" * 64 else value
    )
    path.write_text(json.dumps(identity))
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["provenance"] = identity
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="parameters"):
        verify_flops_smoke(
            directory, attention="classic", expected=expected, expected_dtype="float16"
        )


def test_exact_one_step_new_policy_and_real_backend_pass(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory)
    result = verify_flops_smoke(directory, attention="classic", expected=estimate)
    assert result["passed"] is True
    assert result["steps"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("backend_invocations", 0), ("requested", False), ("training_backend_invocations", 0)],
)
def test_compile_flag_alone_cannot_pass(tmp_path: Path, field: str, value) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory)
    path = directory / "run_identity.json"
    identity = json.loads(path.read_text())
    identity["compiler"][field] = value
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="compiler"):
        verify_flops_smoke(directory, attention="classic", expected=estimate)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "successful_backend_invocations": 0,
            "successful_training_backend_invocations": 0,
            "successful_validation_backend_invocations": 0,
            "backend_invocation_verified": False,
        },
        {  # validation-only success
            "successful_backend_invocations": 1,
            "successful_training_backend_invocations": 0,
        },
        {"backend_invocation_verified": False},
        {"backend_invocation_verified": 1},  # must be the explicit boolean
        {"successful_backend_invocations": 0},
        {"successful_training_backend_invocations": None},  # absent legacy evidence
    ],
)
def test_attempts_or_validation_success_cannot_pass(tmp_path: Path, updates: dict) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory)
    path = directory / "run_identity.json"
    identity = json.loads(path.read_text())
    identity["compiler"].update(updates)
    if identity["compiler"]["successful_training_backend_invocations"] is None:
        del identity["compiler"]["successful_training_backend_invocations"]
    path.write_text(json.dumps(identity), encoding="utf-8")
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["provenance"] = identity
    metrics.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="compiler"):
        verify_flops_smoke(directory, attention="classic", expected=estimate)


@pytest.mark.parametrize("gpu", ["NVIDIA L4", "Tesla T4", "Nvidia L4"])
def test_expected_gpu_requires_exact_name(tmp_path: Path, gpu: str) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory, gpu=gpu)
    if gpu == "NVIDIA L4":
        result = verify_flops_smoke(
            directory, attention="classic", expected=estimate, expected_gpu="NVIDIA L4"
        )
        assert result["passed"] is True
    else:
        with pytest.raises(ValueError, match="expected.*NVIDIA L4"):
            verify_flops_smoke(
                directory, attention="classic", expected=estimate, expected_gpu="NVIDIA L4"
            )


def test_default_gpu_still_requires_t4(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory, gpu="NVIDIA L4")
    with pytest.raises(ValueError, match="expected"):
        verify_flops_smoke(directory, attention="classic", expected=estimate)


def test_wrong_cost_does_not_launch_full_series(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory)
    estimate = {**estimate, "total": 101}
    with pytest.raises(ValueError, match="accounting"):
        verify_flops_smoke(directory, attention="classic", expected=estimate)


def test_smoke_provenance_cannot_disagree(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory)
    path = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["provenance"]["effective_seed"] = 1337
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        verify_flops_smoke(directory, attention="classic", expected=estimate)


def _burnin(directory: Path, *, steps: int = 32, cadence: int | None = None) -> dict:
    expected = _fp16_smoke(directory, optimizer_steps=steps)
    identity_path = directory / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    path = directory / "metrics.jsonl"
    result = json.loads(path.read_text().splitlines()[-1])
    result["steps"] = steps
    result["train_flops"] = expected["total"] * steps
    result["config"]["schedule"]["train_flops"] = result["train_flops"]
    if cadence is not None:
        result["config"]["schedule"]["val_every_steps"] = cadence
    identity["config"] = result["config"]
    result["provenance"] = identity
    rows = []
    for index in range(1, steps + 1):
        flops = index * expected["total"]
        rows.append({"type": "step", "loss": 2.0, "flops": flops})
        if index % (cadence or 100) == 0 or index == steps:
            rows.append({"type": "validation", "loss": 1.9, "accuracy": 0.01, "flops": flops})
    identity_path.write_text(json.dumps(identity))
    path.write_text("\n".join(json.dumps(row) for row in [*rows, result]))
    return expected


@pytest.mark.parametrize("cadence,steps", [(None, 32), (8, 32), (10, 32), (32, 32), (None, 128)])
def test_exact_fp16_burnin_reports_actual_updates(
    tmp_path: Path, cadence: int | None, steps: int
) -> None:
    directory = tmp_path / "smoke"
    expected = _burnin(directory, cadence=cadence, steps=steps)
    report = verify_flops_smoke(
        directory,
        attention="classic",
        expected=expected,
        expected_steps=steps,
        expected_dtype="float16",
    )
    assert report["passed"] is True
    assert report["steps"] == report["optimizer_steps"] == steps
    assert report["step_flops"] == expected["total"]


def _corrupt_burnin_result(directory: Path, result: dict, change: str) -> None:
    identity = result["provenance"]
    if change == "result_steps":
        result["steps"] = 31
    elif change == "float_result_steps":
        result["steps"] = 32.0
    elif change == "result_budget":
        result["train_flops"] -= 100
    elif change == "config_budget":
        result["config"]["schedule"]["train_flops"] -= 100
    elif change in {"updates", "bool_updates", "skips"}:
        key = "skipped_optimizer_steps" if change == "skips" else "optimizer_steps"
        identity["precision"][key] = True if change == "bool_updates" else 1
    elif change == "checkpoint":
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        state["wte.weight"].add_(1)
        torch.save(state, directory / "model.pt")
    elif change.endswith("cadence"):
        value = {"zero": 0, "bool": True, "float": 8.0, "negative": -8}
        result["config"]["schedule"]["val_every_steps"] = value[change.split("_")[0]]
    else:
        identity["effective_seed"] = 1337


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 32.0, "32", None])
def test_expected_steps_must_be_positive_integer(tmp_path: Path, value) -> None:
    directory = tmp_path / "smoke"
    expected = _smoke(directory)
    with pytest.raises(ValueError, match="expected_steps"):
        verify_flops_smoke(directory, attention="classic", expected=expected, expected_steps=value)


@pytest.mark.parametrize(
    "change",
    [
        "early_loss",
        "early_validation_loss",
        "early_validation_accuracy",
        "missing_final",
        "duplicate_final",
        "missing_interval",
        "extra_interval",
        "wrong_final_quality",
        "step_hole",
        "step_count",
        "result_steps",
        "float_result_steps",
        "result_budget",
        "config_budget",
        "updates",
        "bool_updates",
        "skips",
        "provenance",
        "checkpoint",
        "zero_cadence",
        "bool_cadence",
        "float_cadence",
        "negative_cadence",
    ],
)
def test_burnin_rejects_incomplete_or_invalid_artifacts(tmp_path: Path, change: str) -> None:
    directory = tmp_path / "smoke"
    expected = _burnin(directory, cadence=8)
    path = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = rows[-1]
    identity = result["provenance"]
    if change == "early_loss":
        rows[1]["loss"] = float("inf")
    elif change.startswith("early_validation_"):
        rows[8][change.removeprefix("early_validation_")] = float("nan")
    elif change == "missing_final":
        del rows[-2]
    elif change == "duplicate_final":
        rows.insert(-1, rows[-2].copy())
    elif change == "missing_interval":
        del rows[8]
    elif change == "extra_interval":
        rows.insert(2, {"type": "validation", "flops": 200, "loss": 1.9, "accuracy": 0.01})
    elif change == "wrong_final_quality":
        rows[-2]["loss"] = 1.8
    elif change == "step_hole":
        rows[1]["flops"] = 300
    elif change == "step_count":
        del rows[1]
    else:
        _corrupt_burnin_result(directory, result, change)
    if change != "provenance":
        identity["config"] = result["config"]
        (directory / "run_identity.json").write_text(json.dumps(identity))
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        verify_flops_smoke(directory, attention="classic", expected=expected, expected_steps=32)


def test_default_gate_rejects_complete_two_step_burnin(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    expected = _burnin(directory, steps=2)
    with pytest.raises(ValueError, match="one training step"):
        verify_flops_smoke(directory, attention="classic", expected=expected)


def test_hidden_second_step_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "smoke"
    estimate = _smoke(directory)
    path = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.insert(1, {"type": "step", "loss": 1.95, "flops": 200})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="one training step"):
        verify_flops_smoke(directory, attention="classic", expected=estimate)
