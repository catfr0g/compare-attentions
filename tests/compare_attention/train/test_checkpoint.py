"""Resumable snapshots: real tensors, deterministic continuation and strict rejection."""

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_attention.train import checkpoint
from compare_attention.train.backup import BackupTracker
from compare_attention.train.identity import parameter_fingerprint
from compare_attention.train.precision import Precision


def setup_run():
    """A tiny CPU workload with the same config/identity interface as training."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    config_dict = {"schedule": {"train_flops": 1000, "batch_size": 2}, "dataset": {"n_train": 8}}
    config = SimpleNamespace(model_dump=lambda **_: copy.deepcopy(config_dict))
    identity = {
        "config": config_dict,
        "runtime": {"torch": str(torch.__version__)},
        "initial_parameters_sha256": parameter_fingerprint(model),
    }
    return model, optimizer, config, identity


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def snapshot(model, optimizer, config, identity, **kwargs):
    return checkpoint.capture_state(
        model,
        optimizer,
        config,
        step=1,
        cursor=2,
        flops=10,
        losses=[0.5],
        elapsed=1.5,
        early_stopping={"stopped": True},
        identity=identity,
        precision=None,
        **kwargs,
    )


def test_real_roundtrip_resumes_rng_optimizer_and_copies(tmp_path):
    model, optimizer, config, identity = setup_run()
    update(model, optimizer)
    payload = snapshot(model, optimizer, config, identity)
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    expected_loss = update(model, optimizer)
    expected_state = copy.deepcopy(model.state_dict())
    expected_random = (random.random(), np.random.random(), torch.rand(2))
    loaded = checkpoint.load_checkpoint(path)
    restored = checkpoint.restore_state(loaded, model, optimizer, config, identity, None)
    assert restored["early_stopping"] == {"stopped": True}
    assert update(model, optimizer) == expected_loss
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_state[name])
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    assert not torch.equal(payload["model"]["0.weight"], model.state_dict()["0.weight"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 999),
        ("step", 2),
        ("cursor", 3),
        ("flops", 0),
        ("elapsed", float("nan")),
        ("rng", {}),
        ("losses", [float("inf")]),
        ("identity", {}),
        ("config", {}),
        ("optimizer_class", "forged"),
    ],
)
def test_restore_rejects_inconsistent_state_before_mutating_live_run(field, value):
    model, optimizer, config, identity = setup_run()
    update(model, optimizer)
    payload = snapshot(model, optimizer, config, identity)
    payload[field] = value
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


@pytest.mark.parametrize("field", ["config", "runtime", "initial_parameters_sha256"])
def test_restore_rejects_different_requested_identity(field):
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    other = copy.deepcopy(identity)
    other[field] = "different"
    with pytest.raises(ValueError, match="identity|config|runtime"):
        checkpoint.restore_state(payload, model, optimizer, config, other, None)


def test_restore_rejects_silent_model_dtype_cast():
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    model.double()
    with pytest.raises(ValueError, match="dtype|model"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_validation_history_and_public_rng_replay_are_independent():
    model, optimizer, config, identity = setup_run()
    history = [{"step": 1, "flops": 10, "loss": 0.4, "accuracy": 0.5}]
    payload = snapshot(model, optimizer, config, identity, validation_history=history)
    history[0]["loss"] = 999
    expected = torch.rand(2)
    progress = checkpoint.restore_state(payload, model, optimizer, config, identity, None)
    assert progress["validation_history"][0]["loss"] == 0.4
    torch.rand(5)
    checkpoint.restore_rng(payload["rng"])
    assert torch.equal(torch.rand(2), expected)


def test_latest_manifest_verifies_digest_and_prevents_path_escape(tmp_path):
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    BackupTracker(tmp_path).save_checkpoint(payload, 1, 2)
    latest = tmp_path / "checkpoints" / "latest.json"
    assert checkpoint.load_checkpoint(latest)["step"] == 1
    manifest = json.loads(latest.read_text())
    manifest["sha256"] = "0" * 64
    latest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.load_checkpoint(latest)
    manifest["path"] = "../escape.pt"
    latest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path|manifest"):
        checkpoint.load_checkpoint(latest)


def test_identity_is_allowlisted_and_detached():
    model, optimizer, config, identity = setup_run()
    identity["token"] = "never persist"
    identity["runtime"]["secret"] = "never persist"
    identity["compiler"] = {"requested": False, "backend_invocations": 1, "token": "secret"}
    payload = snapshot(model, optimizer, config, identity)
    identity["compiler"]["backend_invocations"] = 2
    assert "token" not in payload["identity"]
    assert "secret" not in payload["identity"]["runtime"]
    assert payload["identity"]["compiler"] == {"requested": False, "backend_invocations": 1}


def test_checkpoint_rejects_tensor_value_corruption():
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    payload["model"]["0.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="integrity|checksum"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def _reseal(payload):
    payload["integrity_sha256"] = checkpoint._digest(payload)  # pylint: disable=protected-access


@pytest.mark.parametrize("kind", ["shape", "key", "alias", "signed_zero"])
def test_rejects_model_structure_and_tied_alias_forgery(kind):
    model, optimizer, config, identity = setup_run()
    model[1] = torch.nn.Linear(3, 3)
    model[1].weight = model[0].weight
    payload = snapshot(model, optimizer, config, identity)
    if kind == "shape":
        payload["model"]["0.weight"] = torch.zeros(1)
    elif kind == "key":
        del payload["model"]["0.bias"]
    elif kind == "alias":
        payload["model"]["1.weight"][0, 0] += 1
    else:
        payload["model"]["0.weight"].zero_()
        payload["model"]["1.weight"].zero_()
        payload["model"]["1.weight"][0, 0] = -0.0
    _reseal(payload)
    with pytest.raises(ValueError, match="model|alias"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


@pytest.mark.parametrize("kind", ["shape", "dtype", "counter", "unknown_parameter"])
def test_rejects_optimizer_tensor_forgery(kind):
    model, optimizer, config, identity = setup_run()
    update(model, optimizer)
    payload = snapshot(model, optimizer, config, identity)
    state = next(iter(payload["optimizer"]["state"].values()))
    if kind == "shape":
        state["exp_avg"] = torch.zeros(1)
    elif kind == "dtype":
        state["exp_avg"] = state["exp_avg"].double()
    elif kind == "counter":
        state["step"].fill_(99)
    else:
        payload["optimizer"]["state"][999] = state
    _reseal(payload)
    with pytest.raises(ValueError, match="optimizer"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_real_cpu_scaler_and_counters_resume_bitwise():
    model, optimizer, config, identity = setup_run()
    policy = SimpleNamespace(
        scaler=torch.amp.GradScaler("cpu", init_scale=128),
        optimizer_steps=0,
        skipped_optimizer_steps=0,
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(2, 3)).square().mean()
        policy.scaler.scale(loss).backward()
        policy.scaler.step(optimizer)
        policy.scaler.update()
        policy.optimizer_steps += 1
        return loss.item()

    step()
    payload = checkpoint.capture_state(
        model,
        optimizer,
        config,
        step=1,
        cursor=2,
        flops=10,
        losses=[0.5],
        elapsed=1.5,
        early_stopping={},
        identity=identity,
        precision=policy,
    )
    expected = step()
    expected_state = copy.deepcopy(model.state_dict())
    checkpoint.restore_state(payload, model, optimizer, config, identity, policy)
    assert policy.optimizer_steps == 1
    assert policy.skipped_optimizer_steps == 0
    assert policy.scaler.get_scale() == 128
    assert step() == expected
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected_state[name])


@pytest.mark.parametrize("scale", [float("nan"), 0.0, -1.0])
def test_rejects_invalid_scaler_scale(scale):
    model, optimizer, config, identity = setup_run()
    policy = SimpleNamespace(
        scaler=torch.amp.GradScaler("cpu", init_scale=128),
        optimizer_steps=1,
        skipped_optimizer_steps=0,
    )
    payload = checkpoint.capture_state(
        model,
        optimizer,
        config,
        step=1,
        cursor=2,
        flops=10,
        losses=[0.5],
        elapsed=1.5,
        early_stopping={},
        identity=identity,
        precision=policy,
    )
    payload["precision"]["scaler"]["scale"] = scale
    _reseal(payload)
    with pytest.raises(ValueError, match="scaler|scale"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, policy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA device")
def test_real_tiny_cuda_fp16_checkpoint_roundtrip(tmp_path):  # pylint: disable=too-many-locals
    model, _, config, identity = setup_run()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    policy = Precision(torch.device("cuda"))

    def step():
        optimizer.zero_grad(set_to_none=True)
        with policy.autocast():
            loss = model(torch.randn(2, 3, device="cuda")).float().square().mean()
        policy.backward_step(loss, optimizer)
        return loss.item()

    first = step()
    payload = checkpoint.capture_state(
        model,
        optimizer,
        config,
        step=1,
        cursor=2,
        flops=10,
        losses=[first],
        elapsed=1.5,
        early_stopping={},
        identity=identity,
        precision=policy,
    )
    path = tmp_path / "tiny-cuda.pt"
    torch.save(payload, path)
    expected = step()
    expected_state = copy.deepcopy(model.state_dict())
    checkpoint.restore_state(
        checkpoint.load_checkpoint(path), model, optimizer, config, identity, policy
    )
    assert policy.optimizer_steps == 1
    assert policy.skipped_optimizer_steps == 0
    assert step() == expected
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected_state[name])
    for parameter, state in optimizer.state.items():
        assert state["exp_avg"].device == parameter.device
        assert state["exp_avg_sq"].device == parameter.device


def test_compiled_wrapper_uses_original_model_keys():
    model, optimizer, config, identity = setup_run()
    compiled = torch.compile(model, backend="eager")
    payload = snapshot(compiled, optimizer, config, identity)
    assert set(payload["model"]) == set(model.state_dict())
    checkpoint.restore_state(payload, compiled, optimizer, config, identity, None)


@pytest.mark.parametrize(
    "state", [{}, {"python": (), "numpy": [], "torch": torch.zeros(1), "cuda": []}]
)
def test_public_rng_rejects_invalid_state_without_mutation(state):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        checkpoint.restore_rng(state)
    assert torch.equal(torch.get_rng_state(), before)


def test_validation_history_rejects_impossible_future_event():
    model, optimizer, config, identity = setup_run()
    payload = snapshot(
        model,
        optimizer,
        config,
        identity,
        validation_history=[{"step": 2, "flops": 20, "loss": 0.5, "accuracy": 0.5}],
    )
    with pytest.raises(ValueError, match="validation"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_accounting_identity_is_preserved_and_binds_budget():
    model, optimizer, config, identity = setup_run()
    identity["accounting"] = {
        "policy": "test",
        "total": 10,
        "forward": 3,
        "backward": 7,
        "breakdown": {},
        "metadata": {},
    }
    payload = snapshot(model, optimizer, config, identity)
    assert payload["identity"]["accounting"] == identity["accounting"]
    payload["flops"] = 20
    _reseal(payload)
    with pytest.raises(ValueError, match="budget|accounting"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_optimizer_parameter_order_cannot_be_silently_reassigned():
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="optimizer"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_cuda_rng_requires_same_device_descriptors():
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    assert "cuda_devices" in payload["rng"]
    payload["rng"]["cuda_devices"] = ["forged"]
    _reseal(payload)
    with pytest.raises(ValueError, match="CUDA RNG"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_restored_optimizer_does_not_mutate_snapshot_tensor_storage():
    model, optimizer, config, identity = setup_run()
    update(model, optimizer)
    payload = snapshot(model, optimizer, config, identity)
    before = copy.deepcopy(payload["optimizer"])
    checkpoint.restore_state(payload, model, optimizer, config, identity, None)
    update(model, optimizer)
    for index, state in before["state"].items():
        for key, value in state.items():
            assert torch.equal(payload["optimizer"]["state"][index][key], value)


@pytest.mark.parametrize("kind", ["model_nonfinite", "optimizer_missing_moment", "schema_bool"])
def test_resealed_payload_still_requires_valid_training_state(kind):
    model, optimizer, config, identity = setup_run()
    update(model, optimizer)
    payload = snapshot(model, optimizer, config, identity)
    if kind == "model_nonfinite":
        payload["model"]["0.weight"][0, 0] = float("nan")
    elif kind == "optimizer_missing_moment":
        del next(iter(payload["optimizer"]["state"].values()))["exp_avg"]
    else:
        payload["schema_version"] = True
    _reseal(payload)
    with pytest.raises(ValueError):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)


def test_different_model_computation_with_identical_weight_shapes_is_rejected():
    model, optimizer, config, identity = setup_run()
    payload = snapshot(model, optimizer, config, identity)
    model[1] = torch.nn.Identity()
    with pytest.raises(ValueError, match="model"):
        checkpoint.restore_state(payload, model, optimizer, config, identity, None)
