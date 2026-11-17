"""A cloud seed pair must prove independent initialization before the sweep continues."""

# Independent artifact fixtures intentionally exercise each public gate separately.
# pylint: disable=duplicate-code

import json
from pathlib import Path

import pytest
import torch

from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.seed_gate import verify_seed_pair
from compare_attention.train.identity import parameter_fingerprint


def _run(root: Path, seed: int, schema_version: int = 1) -> Path:
    root.mkdir()
    identity = {
        "schema_version": schema_version,
        "status": "completed",
        "requested_seed": seed,
        "effective_seed": seed,
        "initial_parameters_sha256": f"{seed:064x}",
        "runtime": {"torch": "test", "gpu": "test"},
    }
    result = {
        "type": "result",
        "steps": 2,
        "train_flops": 100,
        "quality": {"train_loss": 1.0, "val_loss": 1.1, "val_accuracy": 0.1},
        "config": {
            "model": {"attention": "classic"},
            "schedule": {"seed": seed, "batch_size": 8, "train_flops": 100},
            "dataset": {"seed": 1337},
        },
    }
    if schema_version == 2:
        identity["accounting"] = {"policy": "algorithmic_matmul_conv_v2", "total": 50}
        identity["compiler"] = {
            "requested": True,
            "training_backend_invocations": 1,
            "successful_training_backend_invocations": 1,
            "backend_invocation_verified": True,
        }
        result["config"]["schedule"]["compile"] = True
        result["provenance"] = identity
    rows = [
        {"type": "step", "loss": 2.0, "flops": 50},
        {"type": "step", "loss": 1.0 + seed / 10000, "flops": 100},
        {"type": "validation", "loss": 1.1, "accuracy": 0.1, "flops": 100},
        result,
    ]
    (root / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (root / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    torch.save({"weight": torch.tensor([float(seed)])}, root / "model.pt")
    return root


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_schema_one_only_allows_legacy_precisions(tmp_path: Path, dtype: str) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    for directory in (first, second):
        path = directory / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["config"]["schedule"]["dtype"] = dtype
        path.write_text("\n".join(json.dumps(row) for row in rows))
    if dtype == "float16":
        with pytest.raises(ValueError, match="FP16.*schema"):
            verify_seed_pair(first, second)
    else:
        assert verify_seed_pair(first, second)["passed"]


def test_independent_complete_pair_passes(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    report = verify_seed_pair(first, second)
    assert report["passed"] is True
    assert report["seeds"] == [42, 6767]


def test_schema_two_preserves_verified_accounting_and_provenance(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", 42, schema_version=2)
    second = _run(tmp_path / "second", 6767, schema_version=2)
    assert verify_seed_pair(first, second)["passed"] is True


@pytest.mark.parametrize(
    "updates",
    [{"successful_training_backend_invocations": 0}, {"backend_invocation_verified": False}],
)
def test_v2_compile_attempts_cannot_pass(tmp_path: Path, updates: dict) -> None:
    first = _run(tmp_path / "first", 42, schema_version=2)
    second = _run(tmp_path / "second", 6767, schema_version=2)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text())
    identity["compiler"].update(updates)
    path.write_text(json.dumps(identity))
    metrics = second / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["provenance"] = identity
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="compiler"):
        verify_seed_pair(first, second)


def _fp16_run(root: Path, seed: int, **updates) -> Path:
    directory = _run(root, seed, schema_version=2)
    path = directory / "run_identity.json"
    identity = json.loads(path.read_text())
    metrics = directory / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["config"]["schedule"]["dtype"] = "float16"
    identity["config"] = rows[-1]["config"]
    model_config = GPT2Config(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8)
    rows[-1]["config"]["model"] = model_config.model_dump(mode="json")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        model = GPT2(model_config)
    identity["initial_parameters_sha256"] = parameter_fingerprint(model)
    with torch.no_grad():
        model.wte.weight.add_(0.01)
    identity["final_parameters_sha256"] = parameter_fingerprint(model)
    torch.save(model.state_dict(), directory / "model.pt")
    identity["precision"] = {
        "optimizer_steps": 2,
        "skipped_optimizer_steps": 0,
        "initial_scale": 65536.0,
        "final_scale": float(seed),
        "mode": "amp_float16",
        "grad_scaler_enabled": True,
        "parameter_dtype": "float32",
        "autocast_dtype": "float16",
    }
    identity["precision"].update(updates)
    rows[-1]["provenance"] = identity
    path.write_text(json.dumps(identity))
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    return directory


@pytest.mark.parametrize(
    "change", ["forged", "tampered", "unupdated", "half", "keys", "shape", "alias"]
)
def test_fp16_forged_final_fingerprint_is_rejected(tmp_path: Path, change: str) -> None:
    first = _fp16_run(tmp_path / "first", 42)
    second = _fp16_run(tmp_path / "second", 6767)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text())
    directory = second
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if change == "forged":
        identity["final_parameters_sha256"] = "b" * 64
    elif change == "tampered":
        state["wte.weight"].add_(1)
    elif change == "unupdated":
        config = GPT2Config.model_validate(identity["config"]["model"])
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(6767).get_state())
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
    metrics = second / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["provenance"] = identity
    metrics.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="checkpoint"):
        verify_seed_pair(first, second)


def test_fp16_pilots_allow_different_final_scales(tmp_path: Path) -> None:
    first = _fp16_run(tmp_path / "first", 42)
    second = _fp16_run(tmp_path / "second", 6767)
    before_rng = torch.get_rng_state().clone()
    assert verify_seed_pair(first, second)["passed"]
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "updates",
    [
        {"optimizer_steps": 1},
        {"skipped_optimizer_steps": 1},
        {"final_scale": 0.0},
        {"grad_scaler_enabled": False},
        {"parameter_dtype": "float16"},
        {"mode": "legacy"},
    ],
)
def test_fp16_pilots_require_legitimate_updates(tmp_path: Path, updates: dict) -> None:
    first = _fp16_run(tmp_path / "first", 42)
    second = _fp16_run(tmp_path / "second", 6767, **updates)
    with pytest.raises(ValueError, match="precision"):
        verify_seed_pair(first, second)


def test_v2_result_provenance_must_match_identity(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", 42, schema_version=2)
    second = _run(tmp_path / "second", 6767, schema_version=2)
    path = second / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["provenance"]["effective_seed"] = 1337
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        verify_seed_pair(first, second)


def test_v2_emitted_budget_must_equal_step_cost_times_steps(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", 42, schema_version=2)
    second = _run(tmp_path / "second", 6767, schema_version=2)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text())
    identity["accounting"]["total"] = 51
    path.write_text(json.dumps(identity), encoding="utf-8")
    metrics = second / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[-1]["provenance"] = identity
    metrics.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="accounting"):
        verify_seed_pair(first, second)


def test_incomplete_identity_is_rejected(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["status"] = "initialized"
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="completed identity"):
        verify_seed_pair(first, second)


def test_runtime_mismatch_is_rejected(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["runtime"]["torch"] = "different"
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        verify_seed_pair(first, second)


def test_tracker_reseed_is_rejected(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["effective_seed"] = 1337
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="effective seed"):
        verify_seed_pair(first, second)


def test_identical_initial_parameters_are_rejected(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    path = second / "run_identity.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["initial_parameters_sha256"] = f"{42:064x}"
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="initial parameters"):
        verify_seed_pair(first, second)


def test_checkpoint_tensor_values_not_serialization_prove_difference(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    torch.save({"weight": torch.tensor([42.0])}, second / "model.pt")
    with pytest.raises(ValueError, match="final parameters"):
        verify_seed_pair(first, second)


def test_identical_loss_trajectories_are_rejected(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    first_rows = [json.loads(line) for line in (first / "metrics.jsonl").read_text().splitlines()]
    path = second / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row, source in zip(rows, first_rows, strict=True):
        if row["type"] == "step":
            row["loss"] = source["loss"]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="loss trajectories"):
        verify_seed_pair(first, second)


def test_other_hyperparameters_cannot_change(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    path = second / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["config"]["schedule"]["batch_size"] = 16
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        verify_seed_pair(first, second)


def test_missing_final_validation_is_rejected(tmp_path: Path) -> None:
    first, second = _run(tmp_path / "first", 42), _run(tmp_path / "second", 6767)
    path = second / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows if row["type"] != "validation"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="final validation"):
        verify_seed_pair(first, second)
