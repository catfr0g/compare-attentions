"""Failure-boundary probes exercised inside a real torchrun group."""

import ast
import copy
import json
import re
from functools import partial

import torch

from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import _digest, load_checkpoint
from compare_attention.train.distributed import distributed_run
from compare_attention.train.lifecycle import primary_failure
from compare_attention.train.tracker import DisabledTracker, GroupTracker


def assert_phase_failure(error, phase, exception_class, basename, function, raw_message):
    """Require the real sanitized collective envelope, not an arbitrary RuntimeError."""
    assert type(error) is RuntimeError
    message = str(error)
    prefix = "DDP phase failed: "
    assert message.startswith(prefix)
    assert raw_message not in message
    rows = ast.literal_eval(message[len(prefix) :])
    assert isinstance(rows, list) and len(rows) == 2
    reports = [row for row in rows if row is not None]
    assert reports
    for report in reports:
        assert set(report) == {"schema", "phase", "exception_class", "frames"}
        assert report["schema"] == "training-lifecycle-failure-v1"
        assert report["phase"] == phase
        assert report["exception_class"] == exception_class
        assert isinstance(report["frames"], list) and report["frames"]
        for frame in report["frames"]:
            assert set(frame) == {"basename", "function", "line"}
            assert re.fullmatch(r"[A-Za-z_0-9.-]+\.py", frame["basename"])
            assert re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.<>]*", frame["function"])
            assert type(frame["line"]) is int and frame["line"] > 0
        origin = report["frames"][-1]
        assert (origin["basename"], origin["function"]) == (basename, function)
    failure = primary_failure(error, "training")
    assert failure == reports[0]
    return failure


class MustNotStart(DisabledTracker):
    def start(self, _config):
        raise AssertionError("tracker started before rejected resume")


class FailingWriter(BackupTracker):
    def log_loss(self, loss, flops):
        raise RuntimeError("intentional writer failure")


def failures(root, config, mode):
    """One rank's nonfinite data/gradient or writer failure aborts every rank."""
    # pylint: disable=protected-access,too-many-locals
    rank = int(__import__("os").environ["RANK"])
    original_loss, original_load, original_model = loop._loss, loop._load, loop.GPT2
    if mode == "nan":

        def loss(*args, **kwargs):
            value = original_loss(*args, **kwargs)
            return value * float("nan") if rank == 1 else value

        loop._loss = loss
    elif mode in {"zero", "load"}:

        def load(path, split):
            if mode == "load" and rank == 1:
                raise ValueError("intentional rank-local load failure")
            inputs, labels = original_load(path, split)
            return inputs, torch.full_like(labels, -1) if split == "train" else labels

        loop._load = load
    elif mode == "gradient":

        def model(model_config):
            instance = original_model(model_config)
            if rank == 1:
                next(instance.parameters()).register_hook(lambda value: value * float("nan"))
            return instance

        loop.GPT2 = model
    sink = (FailingWriter if mode == "writer" else BackupTracker)(root / mode)
    try:
        with distributed_run(True) as run:
            try:
                loop.train(config, sink, distributed=True)
            except RuntimeError as error:
                message = str(error)
                expected = {
                    "nan": "nonfinite loss",
                    "gradient": "nonfinite gradients",
                    "zero": "no supervised targets",
                    "writer": "writer failure",
                    "load": "rank-local load failure",
                }[mode]
                if mode in {"writer", "load"}:
                    assert_phase_failure(
                        error,
                        "log_loss" if mode == "writer" else "load_train",
                        "RuntimeError" if mode == "writer" else "ValueError",
                        "ddp_guards.py",
                        "log_loss" if mode == "writer" else "load",
                        expected,
                    )
                else:
                    assert expected in message, message
                assert len(set(run.gather(message))) == 1
            else:
                raise AssertionError("failed rank was accepted")
    finally:
        loop._loss, loop._load, loop.GPT2 = original_loss, original_load, original_model
    assert not torch.distributed.is_initialized()
    (root / f"{mode}-failed-{rank}.txt").write_text(message)


def resume_guards(root, config):
    with distributed_run(True) as run:
        loop.train(config, BackupTracker(root / "baseline"), distributed=True)
        base = load_checkpoint(root / "baseline/checkpoints/step_00000003.pt")
        terminal = load_checkpoint(root / "baseline/checkpoints/latest.json")
        for case in ("used", "terminal", "topology", "missing", "rng"):
            payload = copy.deepcopy(terminal if case == "terminal" else base)
            if case == "topology":
                payload["distributed"]["topology"]["local_batch_size"] += 1
            elif case == "missing":
                payload["distributed"]["ranks"].pop()
            elif case == "rng":
                payload["distributed"]["ranks"][1]["rng"].pop("python")
            payload["integrity_sha256"] = _digest(payload)
            path = root / f"{case}.pt"
            run.call(partial(torch.save, payload, path), primary=True, phase="fixture")
            sink_path = root / ("baseline" if case == "used" and run.rank == 1 else f"sink-{case}")
            tracker = GroupTracker(MustNotStart(), GroupTracker(BackupTracker(sink_path)))
            try:
                loop.train(config, tracker, distributed=True, resume_from=path)
            except RuntimeError as error:
                message = str(error)
                assert "tracker started" not in message, message
                expected = {
                    "used": "new result directory",
                    "terminal": "exhausted",
                    "topology": "topology",
                    "missing": "missing ranks",
                    "rng": "RNG",
                }[case]
                basename, function = {
                    "used": ("backup.py", "validate_resume_destination"),
                    "terminal": ("distributed_checkpoint.py", "preflight"),
                    "topology": ("distributed_checkpoint.py", "validate"),
                    "missing": ("distributed_checkpoint.py", "validate"),
                    "rng": ("checkpoint.py", "_validate_rng"),
                }[case]
                assert_phase_failure(
                    error,
                    "resume_destination" if case == "used" else "resume_preflight",
                    "ValueError",
                    basename,
                    function,
                    expected,
                )
                assert len(set(run.gather(message))) == 1
            else:
                raise AssertionError("invalid checkpoint/sink was accepted")
            assert not (root / f"sink-{case}").exists()
        (root / f"guards-{run.rank}.json").write_text(json.dumps({"cases": 5}))


def stopping(root, config, mode):
    # Real updates, controlled validation observations; stopping math is unchanged.
    original = loop._validate  # pylint: disable=protected-access
    loop._validate = lambda *_args: (1.0, 0.0)  # pylint: disable=protected-access
    try:
        with distributed_run(True) as run:
            result = loop.train(config, BackupTracker(root / "stopped"), distributed=True)
            assert result.steps == 3
            assert result.stop_reason == (
                "early_stopping" if mode == "early" else "budget_exhausted"
            )
            assert len({item["stop_reason"] for item in run.gather(result.model_dump())}) == 1
            if mode == "early":
                try:
                    loop.train(
                        config,
                        MustNotStart(),
                        distributed=True,
                        resume_from=root / "stopped/checkpoints/latest.json",
                    )
                except RuntimeError as error:
                    assert_phase_failure(
                        error,
                        "resume_preflight",
                        "ValueError",
                        "distributed_checkpoint.py",
                        "preflight",
                        "terminal early-stopped",
                    )
                    assert len(set(run.gather(str(error)))) == 1
                else:
                    raise AssertionError("early-stopped checkpoint resumed")
    finally:
        loop._validate = original  # pylint: disable=protected-access
