# pylint: disable=redefined-outer-name
"""Обучение до бюджета FLOPs."""

from pathlib import Path

import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
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

    def start(self, config: TrainConfig) -> None:
        self.started.append(config)

    def log_loss(self, loss: float, _flops: int) -> None:
        self.losses.append(loss)

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


def test_compiled_model_reaches_a_finite_loss(prepared: Path) -> None:
    result = train(train_config(prepared, compile=True))

    assert torch.isfinite(torch.tensor(result.quality.train_loss))


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
