# pylint: disable=redefined-outer-name
"""Обучение до бюджета FLOPs."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compare_attention.data.config import IGNORE_INDEX
from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.train import loop
from compare_attention.train.config import ComparisonAxis, Optim, Schedule, TrainConfig
from compare_attention.train.loop import train
from compare_attention.train.result import TrainResult
from compare_attention.train.tracker import DisabledTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config


class RecordingTracker(DisabledTracker):
    """Запоминает вызовы трекера."""

    def __init__(self) -> None:
        self.started: list[TrainConfig] = []
        self.losses: list[float] = []
        self.finished: list[TrainResult] = []
        self.validations: list[tuple[int, float, float, int]] = []

    def start(self, config: TrainConfig) -> None:
        self.started.append(config)

    def log_loss(self, loss: float, _flops: int) -> None:
        self.losses.append(loss)

    def log_validation(self, loss: float, accuracy: float, flops: int) -> None:
        self.validations.append((len(self.losses), loss, accuracy, flops))

    def finish(self, result: TrainResult, _model: torch.nn.Module) -> None:
        self.finished.append(result)


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


@pytest.fixture
def trained(prepared: Path):
    return train(train_config(prepared))


def test_train_loss_is_finite(trained) -> None:
    assert torch.isfinite(torch.tensor(trained.quality.train_loss))


def test_validation_accuracy_is_a_fraction(trained) -> None:
    assert 0.0 <= trained.quality.val_accuracy <= 1.0


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_validation_weights_only_supervised_tokens(batch_size: int) -> None:
    logits = torch.zeros(3, 2, 3)
    labels = torch.tensor([[IGNORE_INDEX, 0], [1, 0], [1, IGNORE_INDEX]])

    loss, accuracy = loop._validate(  # pylint: disable=protected-access
        torch.nn.Identity(), logits, labels, batch_size
    )

    assert loss == pytest.approx(float(torch.log(torch.tensor(3.0))))
    assert accuracy == 0.5


def test_train_speed_is_positive(trained) -> None:
    assert trained.speed.train_tokens_per_second > 0


def test_peak_gpu_memory_follows_the_device(trained) -> None:
    if torch.cuda.is_available():
        assert trained.speed.peak_gpu_bytes > 0
        return
    assert trained.speed.peak_gpu_bytes is None


def test_tiny_budget_records_one_step(prepared: Path) -> None:
    result = train(train_config(prepared))

    assert result.steps == 1


def test_vocab_must_match_the_dataset(prepared: Path) -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        train_config(prepared, model=gpt2_config(vocab_size=64, n_layer=1, dropout=0.0))


def test_block_must_cover_the_sequence(prepared: Path) -> None:
    with pytest.raises(ValueError, match="block_size"):
        train_config(prepared, model=gpt2_config(block_size=8, n_layer=1, dropout=0.0))


def test_batch_must_fit_the_training_split(prepared: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        train_config(prepared, batch_size=8)


@pytest.mark.parametrize("axis", [ComparisonAxis.LAYER_FLOPS, ComparisonAxis.PARAMETERS])
def test_other_axes_are_not_trained_yet(prepared: Path, axis) -> None:
    with pytest.raises(NotImplementedError, match=axis.value):
        train_config(prepared, axis=axis)


@pytest.fixture
def isolated_dynamo_cache():
    """Keep unrelated compiler tests from exhausting this test's guard budget."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    try:
        yield
    finally:
        torch._dynamo.reset()  # pylint: disable=protected-access


@pytest.mark.requires_cpu_inductor
@pytest.mark.usefixtures("isolated_dynamo_cache")
def test_compiled_model_reaches_a_finite_loss(prepared: Path) -> None:
    step_flops = train(train_config(prepared)).train_flops
    spy = RecordingTracker()
    result = train(
        train_config(prepared, compile=True, train_flops=step_flops * 3, val_every_steps=1),
        tracker=spy,
    )

    assert torch.isfinite(torch.tensor(result.quality.train_loss))
    assert result.steps == 3
    assert len(spy.validations) == 3


def test_tracker_sees_the_start_of_the_run(prepared: Path) -> None:
    spy = RecordingTracker()

    result = train(train_config(prepared), tracker=spy)

    assert spy.started == [result.config]


def test_tracker_sees_every_step_loss(prepared: Path) -> None:
    spy = RecordingTracker()

    result = train(train_config(prepared), tracker=spy)

    assert result.steps >= 1
    assert len(spy.losses) == result.steps


def test_tracker_sees_the_final_result(prepared: Path) -> None:
    spy = RecordingTracker()

    result = train(train_config(prepared), tracker=spy)

    assert spy.finished == [result]


@pytest.mark.parametrize(
    ("interval", "expected_steps"), [(1, [1, 2, 3]), (2, [2, 3]), (3, [3]), (10, [3])]
)
def test_validation_is_recorded_during_training_and_at_the_end(
    prepared: Path, interval: int, expected_steps: list[int]
) -> None:
    step_flops = train(train_config(prepared)).train_flops
    spy = RecordingTracker()

    result = train(
        train_config(prepared, train_flops=step_flops * 3, val_every_steps=interval), tracker=spy
    )

    assert result.steps == 3
    assert result.train_flops == step_flops * 3
    assert [step for step, *_ in spy.validations] == expected_steps
    assert [flops for *_, flops in spy.validations] == [
        step_flops * step for step in expected_steps
    ]
    _, loss, accuracy, _ = spy.validations[-1]
    assert loss == result.quality.val_loss
    assert accuracy == result.quality.val_accuracy


def test_progress_is_logged_without_a_terminal(
    prepared: Path, caplog: pytest.LogCaptureFixture
) -> None:
    step_flops = train(train_config(prepared)).train_flops
    caplog.set_level(logging.INFO, logger="compare_attention.train.loop")

    result = train(
        train_config(prepared, train_flops=step_flops * 3, log_every_steps=2, val_every_steps=2)
    )

    messages = [record.getMessage() for record in caplog.records]
    progress = [message for message in messages if message.startswith("train step=")]
    assert len(progress) == 3
    for step, message in enumerate(progress, start=1):
        assert f"step={step} " in message
        assert f"flops={step_flops * step}/{result.config.schedule.train_flops}" in message
        assert "loss=" in message
        assert "lr=" in message
        assert "tok/s=" in message
        assert "eta=" in message
    validation = [message for message in messages if message.startswith("validation step=")]
    assert len(validation) == 2
    assert "step=2 " in validation[0]
    assert "val_loss=" in validation[0]
    assert "val_accuracy=" in validation[0]


def test_validation_time_is_not_counted_as_training_time(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step_flops = train(train_config(prepared)).train_flops
    clock = SimpleNamespace(now=0.0)
    original_step = loop._tracked_step  # pylint: disable=protected-access
    original_validate = loop._validate  # pylint: disable=protected-access

    def timed_step(*args):
        clock.now += 1.0
        return original_step(*args)

    def slow_validation(*args):
        clock.now += 1000.0
        return original_validate(*args)

    monkeypatch.setattr(loop, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    monkeypatch.setattr(loop, "_tracked_step", timed_step)
    monkeypatch.setattr(loop, "_validate", slow_validation)

    result = train(train_config(prepared, train_flops=step_flops * 3, val_every_steps=1))

    tokens_per_step = result.config.schedule.batch_size * result.config.dataset.input_seq_len
    assert result.speed.train_tokens_per_second == tokens_per_step


@pytest.mark.parametrize("field", ["log_every_steps", "val_every_steps"])
@pytest.mark.parametrize("value", [0, -1])
def test_reporting_intervals_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        Schedule.model_validate({"train_flops": 1, field: value})


def test_long_deltanet_in_float32_is_rejected(tmp_path: Path) -> None:
    spec = data_config(
        vocab_size=128,
        input_seq_len=80,
        num_kv_pairs=2,
        n_train=2,
        n_val=1,
        n_test=1,
    )
    prepare_kv_retrieval(spec, tmp_path)

    with pytest.raises(ValueError, match="bfloat16"):
        TrainConfig(
            data=tmp_path,
            model=gpt2_config(vocab_size=128, block_size=80, n_layer=1, attention="deltanet"),
            schedule=Schedule(train_flops=1, batch_size=1, seed=0),
            optim=Optim(learning_rate=1e-2, weight_decay=0.0),
        )
