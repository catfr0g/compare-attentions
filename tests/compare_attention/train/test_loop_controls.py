"""Real tiny-loop tests for exploratory early stopping and restart contracts."""

import json

import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.exploration_gate import verify_exploration_run
from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.flops import estimate_step
from compare_attention.train.tracker import DisabledTracker, GroupTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.train.factories import train_config


def test_early_stop_saves_final_weights_without_claiming_completed_budget(tmp_path, monkeypatch):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    cost = estimate_step(train_config(data)).total
    config = train_config(
        data,
        train_flops=cost * 10,
        val_every_steps=1,
        early_stopping_patience=2,
        early_stopping_min_steps=0,
    )
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loop, "_validate", lambda *_: (1.0, 0.0))
    output = tmp_path / "output"
    result = loop.train(config, BackupTracker(output))
    assert result.steps == 3
    assert result.train_flops == cost * 3 < config.schedule.train_flops
    assert result.stop_reason == "early_stopping"
    assert result.provenance["status"] == "early_stopped"
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    validations = [row for row in rows if row["type"] == "validation"]
    assert len(validations) == 3
    assert [r["flops"] for r in validations] == [cost, 2 * cost, 3 * cost]
    assert rows[-1]["stop_reason"] == "early_stopping"
    assert (output / "model.pt").is_file()
    assert result.quality.val_loss == 1.0
    verified = verify_exploration_run(output, expected_config=config.model_dump(mode="json"))
    assert verified["passed"] and verified["stop_reason"] == "early_stopping"
    assert verified["budget_exhausted"] is False


def test_disabled_stopping_runs_to_budget_with_flat_validation(tmp_path, monkeypatch):
    prepare_kv_retrieval(data_config(), tmp_path)
    cost = estimate_step(train_config(tmp_path)).total
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loop, "_validate", lambda *_: (1.0, 0.0))
    result = loop.train(train_config(tmp_path, train_flops=cost * 4, val_every_steps=1))
    assert result.steps == 4
    assert result.stop_reason == "budget_exhausted"
    assert result.provenance["status"] == "completed"


def test_budget_boundary_has_one_validation_and_budget_takes_precedence(tmp_path, monkeypatch):
    prepare_kv_retrieval(data_config(), tmp_path)
    cost = estimate_step(train_config(tmp_path)).total
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loop, "_validate", lambda *_: (1.0, 0.0))
    out = tmp_path / "output"
    result = loop.train(
        train_config(
            tmp_path,
            train_flops=cost * 3,
            val_every_steps=1,
            early_stopping_patience=2,
            early_stopping_min_steps=0,
        ),
        BackupTracker(out),
    )
    assert result.stop_reason == "budget_exhausted"
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["type"] == "validation"]) == 3


def test_interrupted_training_resumes_exact_weights_and_complete_history(tmp_path, monkeypatch):
    # Keep the real interruption seam and both trajectories visible in this integration proof.
    # pylint: disable=too-many-locals,protected-access
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    cost = estimate_step(train_config(data)).total
    config = train_config(
        data,
        train_flops=cost * 6,
        val_every_steps=2,
        checkpoint_every_steps=2,
        checkpoint_keep_last=2,
    )
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))

    class RngBackup(BackupTracker):
        def log_loss(self, loss, flops):
            torch.rand(1)  # Real trackers may consume RNG while replaying history.
            super().log_loss(loss, flops)

    reference = tmp_path / "reference"
    full = loop.train(config, RngBackup(reference))
    expected_next_random = torch.rand(4)
    original = loop._tracked_step
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption after checkpoint")
        return original(*args)

    monkeypatch.setattr(loop, "_tracked_step", interrupted)
    partial = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        loop.train(config, RngBackup(partial))
    checkpoint = partial / "checkpoints/step_00000002.pt"
    assert checkpoint.is_file()
    monkeypatch.setattr(loop, "_tracked_step", original)
    resumed_dir = tmp_path / "resumed"
    resumed = loop.train(config, RngBackup(resumed_dir), resume_from=checkpoint)
    assert torch.equal(torch.rand(4), expected_next_random)
    assert resumed.steps == full.steps == 6
    assert resumed.train_flops == full.train_flops
    expected = torch.load(reference / "model.pt", weights_only=True)
    actual = torch.load(resumed_dir / "model.pt", weights_only=True)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)

    def histories(directory):
        return [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
            if json.loads(line)["type"] != "result"
        ]

    assert histories(reference) == histories(resumed_dir)
    assert resumed.early_stopping == full.early_stopping
    assert len(list((resumed_dir / "checkpoints").glob("step_*.pt"))) == 2


@pytest.mark.parametrize("grouped", [False, True])
def test_api_resume_rejects_used_sink_before_any_tracker_side_effect(
    tmp_path, monkeypatch, grouped
):
    # The real optimizer interruption boundary must remain visible in this regression.
    # pylint: disable=protected-access
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    cost = estimate_step(train_config(data)).total
    config = train_config(data, train_flops=cost * 6, val_every_steps=2, checkpoint_every_steps=2)
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    original = loop._tracked_step
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("interrupted at saved step two")
        return original(*args)

    output = tmp_path / "output"
    monkeypatch.setattr(loop, "_tracked_step", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        loop.train(config, BackupTracker(output))
    monkeypatch.setattr(loop, "_tracked_step", original)
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    class StartMustNotRun(DisabledTracker):
        def start(self, _config):
            pytest.fail("resume guard must precede every tracker start")

    sink = BackupTracker(output)
    tracker = GroupTracker(StartMustNotRun(), GroupTracker(sink)) if grouped else sink
    with pytest.raises(ValueError, match="new result directory"):
        loop.train(config, tracker, resume_from=output / "checkpoints/latest.json")
    after = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    assert after == before


def test_resume_rejects_a_terminal_early_stopped_checkpoint(tmp_path, monkeypatch):
    data = tmp_path / "data"
    prepare_kv_retrieval(data_config(), data)
    cost = estimate_step(train_config(data)).total
    config = train_config(
        data,
        train_flops=cost * 10,
        val_every_steps=1,
        early_stopping_patience=1,
        early_stopping_min_steps=0,
        checkpoint_every_steps=1,
    )
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loop, "_validate", lambda *_: (1.0, 0.0))
    original = tmp_path / "original"
    result = loop.train(config, BackupTracker(original))
    assert result.stop_reason == "early_stopping"
    with pytest.raises(ValueError, match="early.stopped|terminal"):
        loop.train(
            config,
            BackupTracker(tmp_path / "resumed"),
            resume_from=original / "checkpoints/latest.json",
        )
