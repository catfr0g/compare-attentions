"""Fixed analytical budgets must not execute calibration or dispatcher counters."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.exc import SkipFrame

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train import compiler, loop
from compare_attention.train.backup import BackupTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.train.factories import train_config


@pytest.mark.parametrize(
    "split_name,kind,mutation",
    [("train", 0, "length"), ("train", 1, "length"), ("val", 0, "rows"), ("val", 1, "dtype")],
)
def test_rejects_loaded_shape_before_model_or_cost_estimation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_name: str, kind: int, mutation: str
) -> None:
    prepare_kv_retrieval(data_config(), tmp_path / "data")
    config = train_config(tmp_path / "data", train_flops=1)
    load = loop._load  # pylint: disable=protected-access

    def malformed(path, split):
        pair = list(load(path, split))
        if split == split_name:
            tensor = pair[kind]
            if mutation == "length":
                pair[kind] = tensor[:, :-1]
            elif mutation == "rows":
                pair[kind] = tensor[:-1]
            else:
                pair[kind] = tensor.float()
        return tuple(pair)

    def forbidden(*_args, **_kwargs):
        pytest.fail("model/cost construction ran before validating loaded tensor shapes")

    monkeypatch.setattr(loop, "_load", malformed)
    monkeypatch.setattr(loop, "GPT2", forbidden)
    monkeypatch.setattr(loop, "estimate_step", forbidden)
    with pytest.raises(ValueError, match="loaded .* mismatch"):
        loop.train(config)


@pytest.mark.parametrize("validation_succeeds", [False, True])
def test_train_rejects_skipped_training_even_if_validation_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_succeeds: bool
) -> None:
    prepare_kv_retrieval(data_config(), tmp_path / "data")
    phases = []
    successful_phases = []

    def backend(graph, _inputs):
        phase = compiler._PHASE.get()  # pylint: disable=protected-access
        phases.append(phase)
        if phase == "validation" and validation_succeeds:
            successful_phases.append(phase)
            return graph.forward
        raise SkipFrame("unsupported BF16")

    validate = loop._validate  # pylint: disable=protected-access

    def fresh_validation(*args):
        # SkipFrame caches the skipped frame; force a distinct validation compile attempt.
        torch._dynamo.reset()  # pylint: disable=protected-access
        return validate(*args)

    monkeypatch.setattr(compiler, "lookup_backend", lambda _name: backend)
    monkeypatch.setattr(loop, "_validate", fresh_validation)
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = train_config(tmp_path / "data", compile=True)
    torch._dynamo.reset()  # pylint: disable=protected-access
    try:
        with pytest.raises(RuntimeError, match="successful.*training"):
            loop.train(config, BackupTracker(tmp_path / "result"))
    finally:
        torch._dynamo.reset()  # pylint: disable=protected-access
    assert "training" in phases
    assert "validation" in phases
    assert bool(successful_phases) is validation_succeeds
    assert all(phase == "validation" for phase in successful_phases)
    identity = json.loads((tmp_path / "result" / "run_identity.json").read_text())
    assert identity["status"] == "initialized"
    rows = (tmp_path / "result" / "metrics.jsonl").read_text().splitlines()
    assert not any(json.loads(row)["type"] == "result" for row in rows)


@pytest.mark.parametrize(("budget", "steps"), [(1, 1), (100, 1), (101, 2), (201, 3)])
def test_fixed_budget_has_no_hidden_steps_or_dynamic_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int, steps: int
) -> None:
    prepare_kv_retrieval(data_config(), tmp_path / "data")
    estimates = []
    actual_steps = []
    original_step = loop._step  # pylint: disable=protected-access

    def estimate(config):
        estimates.append(config)
        return SimpleNamespace(total=100, to_dict=lambda: {"policy": "test-v1", "total": 100})

    def step(*args):
        actual_steps.append(1)
        return original_step(*args)

    def forbidden(*_args, **_kwargs):
        pytest.fail("FlopCounterMode entered production training")

    monkeypatch.setattr(loop, "estimate_step", estimate, raising=False)
    monkeypatch.setattr(loop, "_step", step)
    monkeypatch.setattr(torch.utils.flop_counter, "FlopCounterMode", forbidden)
    monkeypatch.setattr(loop, "FlopCounterMode", forbidden, raising=False)
    config = train_config(tmp_path / "data", train_flops=budget, val_every_steps=1)
    result = loop.train(config, BackupTracker(tmp_path / "result"))
    assert estimates == [config]
    assert len(actual_steps) == result.steps == steps
    assert result.train_flops == steps * 100
    assert 0 <= result.train_flops - budget < 100
    assert result.provenance["recorded_flops"]["policy"] == "test-v1"
