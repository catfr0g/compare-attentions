"""Raw scheduled artifacts must survive public and orchestrator acceptance."""

# Explicit corruption matrix keeps each rejected evidence mutation inspectable.
# pylint: disable=protected-access,too-many-branches,too-many-statements,duplicate-code
import hashlib
import json
import math
from functools import partial

import pytest
import torch

from compare_attention import kaggle_series
from compare_attention.exploration_gate import verify_exploration_run
from compare_attention.flop_gate import verify_flops_smoke
from compare_attention.seed_gate import verify_seed_pair
from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.config import Optim
from compare_attention.train.lr_schedule import next_learning_rate
from tests.compare_attention.test_exploration_gate import artifact, persist
from tests.compare_attention.test_flop_gate import _burnin
from tests.compare_attention.test_seed_gate import _run
from tests.compare_attention.train.test_lr_training import tiny_config


def scheduled(rows, cost):
    """Add policy-priced telemetry to independent gate fixtures."""
    config = rows[-1]["config"]
    config["optim"] = Optim(lr_schedule="warmup_cosine", warmup_fraction=0.5).model_dump()
    optim = Optim.model_validate(config["optim"])
    identity = rows[-1].get("provenance", {})
    if "config" in identity:
        identity["config"] = config
    view = []
    for row in rows:
        view.append(row)
        if row["type"] == "step":
            view.append(
                {
                    "type": "learning_rate",
                    "learning_rate": next_learning_rate(
                        optim, config["schedule"]["train_flops"], row["flops"] - cost, cost
                    ),
                    "flops": row["flops"],
                }
            )
    return view


def read_rows(directory):
    return [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("steps", [1, 32])
def test_scheduled_exact_step_smoke(tmp_path, steps):
    directory = tmp_path / "smoke"
    expected = _burnin(directory, steps=steps)
    rows = scheduled(read_rows(directory), expected["total"])
    persist(directory, rows)
    assert verify_flops_smoke(
        directory, attention="classic", expected=expected, expected_steps=steps
    )["passed"]


@pytest.mark.parametrize("early", [False, True])
def test_scheduled_full_and_legitimate_early_prefix(tmp_path, early):
    directory = tmp_path / "run"
    rows = artifact(directory, early=early)
    rows = scheduled(rows, rows[-1]["provenance"]["accounting"]["total"])
    persist(directory, rows)
    assert verify_exploration_run(directory)["budget_exhausted"] is not early


def test_scheduled_seed_pair(tmp_path):
    pair = [_run(tmp_path / str(seed), seed, schema_version=2) for seed in (42, 6767)]
    for directory in pair:
        persist(directory, scheduled(read_rows(directory), 50))
    assert verify_seed_pair(*pair)["passed"]


@pytest.mark.parametrize("gate", ["exploration", "smoke", "seed"])
@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "before_step",
        "after_validation",
        "wrong_flops",
        "float_flops",
        "bool_flops",
        "negative_flops",
        "wrong_lr",
        "nan",
        "inf",
        "bool_lr",
        "int_lr",
        "extra_key",
        "missing_key",
        "constant",
        "unknown",
        "reordered_steps",
        "wrong_policy",
        "incomplete_optim",
    ],
)
def test_public_gates_reject_corrupted_raw_lr_history(tmp_path, gate, change):
    directory = tmp_path / "run"
    if gate == "exploration":
        rows = artifact(directory)
        cost = rows[-1]["provenance"]["accounting"]["total"]
        verify = partial(verify_exploration_run, directory)
    elif gate == "smoke":
        expected = _burnin(directory, steps=32, cadence=2)
        cost = expected["total"]
        verify = partial(
            verify_flops_smoke, directory, attention="classic", expected=expected, expected_steps=32
        )
        rows = read_rows(directory)
    else:
        other = _run(tmp_path / "other", 6767, schema_version=2)
        persist(other, scheduled(read_rows(other), 50))
        _run(directory, 42, schema_version=2)
        rows = read_rows(directory)
        cost = 50
        verify = partial(verify_seed_pair, directory, other)
    rows = scheduled(rows, cost)
    lr = rows[1]
    if change == "missing":
        del rows[1]
    elif change == "duplicate":
        rows.insert(2, lr.copy())
    elif change == "before_step":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "after_validation":
        rows.remove(lr)
        rows.insert(-1, lr)
    elif change == "wrong_flops":
        lr["flops"] += 1
    elif change == "float_flops":
        lr["flops"] = float(lr["flops"])
    elif change == "bool_flops":
        lr["flops"] = True
    elif change == "negative_flops":
        lr["flops"] = -cost
    elif change == "wrong_lr":
        lr["learning_rate"] *= 1.000001
    elif change in {"nan", "inf"}:
        lr["learning_rate"] = float(change)
    elif change == "bool_lr":
        lr["learning_rate"] = True
    elif change == "int_lr":
        lr["learning_rate"] = 0
    elif change == "extra_key":
        lr["extra"] = 0
    elif change == "missing_key":
        del lr["learning_rate"]
    elif change == "constant":
        rows[-1]["config"]["optim"]["lr_schedule"] = "constant"
    elif change == "unknown":
        rows.insert(2, {"type": "unknown"})
    elif change == "reordered_steps":
        rows[0], rows[2] = rows[2], rows[0]
    elif change == "incomplete_optim":
        del rows[-1]["config"]["optim"]["min_lr_ratio"]
    elif change == "wrong_policy":
        rows[-1]["config"]["optim"]["warmup_fraction"] = 0.0
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify()


def test_cross_platform_last_ulp_lr_is_accepted(tmp_path):
    directory = tmp_path / "run"
    rows = artifact(directory)
    rows = scheduled(rows, rows[-1]["provenance"]["accounting"]["total"])
    rows[1]["learning_rate"] = math.nextafter(rows[1]["learning_rate"], math.inf)
    persist(directory, rows)
    assert verify_exploration_run(directory)["passed"]


def test_legacy_constant_config_defaults_only_new_optim_fields(tmp_path):
    directory = tmp_path / "run"
    rows = artifact(directory)
    for key in ("lr_schedule", "warmup_fraction", "min_lr_ratio"):
        del rows[-1]["config"]["optim"][key]
    persist(directory, rows)
    assert verify_exploration_run(directory)["passed"]
    del rows[-1]["config"]["optim"]["learning_rate"]
    persist(directory, rows)
    with pytest.raises(ValueError):
        verify_exploration_run(directory)


@pytest.mark.parametrize("scenario", ["completed", "early", "resumed"])
def test_real_scheduled_cpu_artifact_passes_raw_gate_and_receipt_seam(
    tmp_path, monkeypatch, scenario
):
    """Only redirect dataset IO; training, persistence and acceptance remain real."""
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    config = tiny_config(tmp_path, steps=3)
    if scenario == "early":
        config = config.model_copy(
            update={
                "schedule": config.schedule.model_copy(
                    update={
                        "early_stopping_patience": 1,
                        "early_stopping_min_steps": 0,
                        "early_stopping_min_delta": 100.0,
                    }
                )
            }
        )
    data = config.data
    original_load = loop._load
    monkeypatch.setattr(loop, "_load", lambda _, split: original_load(data, split))
    config = config.model_copy(update={"data": "/kaggle/input/datasets/owner/tiny"})
    output = tmp_path / "download"
    directory = output / "scheduled"
    resume_from = None
    if scenario == "resumed":
        previous = tmp_path / "previous"
        loop.train(config, BackupTracker(previous))
        resume_from = previous / "checkpoints/step_00000001.pt"
    loop.train(config, BackupTracker(directory), resume_from=resume_from)
    files = {
        f"scheduled/{name}": {
            "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            "bytes": (directory / name).stat().st_size,
        }
        for name in ("model.pt", "metrics.jsonl", "run_identity.json")
    }
    state = {
        "owner": "owner",
        "data_id": "owner/tiny",
        "identity": {
            "fingerprint": "f" * 64,
            "configs": {"tiny": config.model_dump(mode="json")},
            "data": {"spec": config.dataset.model_dump(mode="json")},
        },
        "runs": {"tiny": {"slug": "scheduled"}},
    }
    receipt = {
        "identity": {
            "series": "f" * 64,
            "config": "tiny",
            "id": "owner/scheduled",
            "slug": "scheduled",
        },
        "files": files,
    }
    (output / "result-manifest.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_exploration_run(directory)["passed"]
    report = kaggle_series._verify_result(output, state, "tiny")
    assert report["steps"] == (2 if scenario == "early" else 3)
    assert report["stop_reason"] == (
        "early_stopping" if scenario == "early" else "budget_exhausted"
    )
    assert files == {
        f"scheduled/{name}": {
            "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            "bytes": (directory / name).stat().st_size,
        }
        for name in ("model.pt", "metrics.jsonl", "run_identity.json")
    }
