"""Few-step fresh synthetic model -> actual tracker -> shared completion gate."""

import copy
import json

import pytest
import torch
import training_completion as completion

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train.backup import BackupTracker
from compare_attention.train.checkpoint import load_checkpoint
from compare_attention.train.flops import estimate_step
from compare_attention.train.loop import train
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.train.factories import train_config
from tests.compare_attention.train.test_distributed import assert_equal


class ScalarTask:
    """Synthetic transport fixture; trainer and scalar producer are real."""

    id = "synthetic-lifecycle-task"

    def __init__(self, fail=None):
        self.events = []
        self.order = []
        self.fail = fail
        self.parameters = {}

    def boundary(self, name):
        self.order.append(name)
        if self.fail == name:
            raise LookupError("PRIVATE_MESSAGE_Bearer_secret")

    def connect(self, config):
        def domain(value):
            if isinstance(value, dict):
                return {key: domain(item) for key, item in value.items()}
            return "" if value is None else str(value)

        self.parameters = domain(config)

    def get_logger(self):
        return self

    def report_scalar(self, title, series, value, iteration):
        if title == "result":
            self.boundary("result_log")
        self.events.append({"metric": title, "variant": series, "value": value, "iter": iteration})

    def flush(self, *, wait_for_uploads):
        assert wait_for_uploads
        self.boundary("flush")

    def close(self):
        self.boundary("close")

    def readback(self, task_id):
        assert task_id == self.id
        self.boundary("readback")
        events = copy.deepcopy(self.events)
        if self.fail == "validation":
            events[0]["value"] = float("nan")
        parameters = copy.deepcopy(self.parameters)
        if self.fail == "parameters":
            parameters["schedule"]["seed"] = "999"
        return {
            "status": "completed",
            "events": events,
            "total": len(events),
            "task_id": self.id,
            "parameters": parameters,
        }


def setup(tmp_path, *, policy="constant", seed=202, task="mqar"):
    data = tmp_path / "data"
    if task == "sorting":
        from compare_attention.cli.prepare_sorting import main
        from compare_attention.models.config import GPT2Config

        assert main(["--output", str(data), "--n-train", "4", "--n-val", "3", "--n-test", "2"]) == 0
        model = GPT2Config(
            vocab_size=8, block_size=128, n_layer=1, n_head=2, n_embd=16, dropout=0.2
        )
    else:
        prepare_kv_retrieval(data_config(), data)
        model = None
    config = train_config(data, model=model, seed=seed, checkpoint_every_steps=2, val_every_steps=2)
    return config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={"train_flops": estimate_step(config).total * 3}
            ),
            "optim": config.optim.model_copy(update={"lr_schedule": policy}),
        }
    )


def tracker(out, task):
    service = completion.TrainingServiceTracker(
        {},
        "engineering",
        updates=3,
        validations=2,
        task_factory=lambda _: task,
        readback=task.readback,
    )
    return completion.CompletionTracker(BackupTracker(out), service)


@pytest.mark.parametrize("seed", [202, 303, 404])
@pytest.mark.parametrize("policy", ["constant", "warmup_cosine"])
@pytest.mark.parametrize("task_kind", ["mqar", "sorting"])
def test_public_normal_producer_reaches_completion_with_unchanged_state(
    tmp_path, seed, policy, task_kind
):
    config = setup(tmp_path, seed=seed, policy=policy, task=task_kind)
    baseline = tmp_path / "baseline"
    # Execute the exact authenticated pre-fix trainer on fresh synthetic data,
    # not another run through the candidate's own _train implementation.
    import hashlib
    import types
    from pathlib import Path

    old_path = Path(
        "C:/Users/Michael/AppData/Local/hermes/cache/scratch/mqar202-repair-work-v1/wheel/compare_attention/train/loop.py"
    )
    old_raw = old_path.read_bytes()
    assert (
        hashlib.sha256(old_raw).hexdigest()
        == "ced8503c14d5e1f5872ed86f505484af645720ba6c15490c7d7f7013f090b40f"
    )
    original_loop = types.ModuleType("authenticated_original_loop")
    exec(compile(old_raw, "authenticated_original_loop.py", "exec"), vars(original_loop))
    vars(original_loop)["train"](config, BackupTracker(baseline))
    expected_rng = torch.get_rng_state().clone()
    task = ScalarTask()
    sink = tracker(tmp_path / "service", task)
    result = train(config, sink)
    assert result.steps == 3
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert task.order[-3:] == ["flush", "close", "readback"]
    for field in (
        "model",
        "optimizer",
        "precision",
        "rng",
        "losses",
        "validation_history",
        "cursor",
        "step",
        "flops",
    ):
        left = load_checkpoint(baseline / "checkpoints/latest.json")
        right = load_checkpoint(tmp_path / "service/checkpoints/latest.json")
        assert_equal(left[field], right[field])
    assert sink.receipt["scientific_event_count"] == len(task.events)
    raw = json.loads((tmp_path / "service/training-unsampled-readback.json").read_bytes())
    assert raw["events"] == task.events
    assert (
        json.loads((tmp_path / "service/run_identity.json").read_bytes())["status"] == "completed"
    )


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("result_log", "clearml_finish"),
        ("flush", "clearml_flush"),
        ("close", "clearml_close"),
        ("readback", "clearml_readback"),
        ("validation", "clearml_validate_receipt"),
        ("parameters", "clearml_validate_receipt"),
        ("identity", "identity_write"),
    ],
)
def test_real_trainer_original_failure_phase_is_persisted(tmp_path, failure, expected):
    config = setup(tmp_path)
    task = ScalarTask(fail=failure)
    out = tmp_path / "failed"
    sink = tracker(out, task)
    if failure == "identity":
        original = sink.backup.record_identity

        def record(identity):
            if identity["status"] == "completed":
                raise LookupError("PRIVATE_MESSAGE_identity")
            original(identity)

        sink.backup.record_identity = record
    exception = ValueError if failure in {"validation", "parameters"} else LookupError
    with pytest.raises(exception):
        train(config, sink)
    path = out / "lifecycle-failure.json"
    assert path.is_file(), "no original failure persisted"
    report = json.loads(path.read_bytes())
    assert report["phase"] == expected
    assert report["exception_class"] == exception.__name__
    assert "PRIVATE_MESSAGE" not in path.read_text()
    assert "Bearer" not in path.read_text()
    assert report["frames"][-1]["basename"].endswith(".py")
