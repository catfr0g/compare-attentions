# pylint: disable=redefined-outer-name
"""Обучение до бюджета FLOPs."""

from pathlib import Path

import pytest
import torch

from compare_attention.data.store import load_config, prepare_kv_retrieval
from compare_attention.train.config import ComparisonAxis, Optim, Schedule, TrainConfig
from compare_attention.train.loop import train
from compare_attention.train.result import Quality, Speed, TrainResult
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


def _run(
    path: Path,
    train_flops: int = 1,
    axis: ComparisonAxis = ComparisonAxis.TRAIN_FLOPS,
    batch_size: int = 2,
    compile_model: bool = False,
    model=None,
) -> TrainConfig:
    chosen = gpt2_config(n_layer=1, dropout=0.0) if model is None else model
    return TrainConfig(
        data=path,
        model=chosen,
        schedule=Schedule(
            train_flops=train_flops,
            axis=axis,
            batch_size=batch_size,
            seed=0,
            compile=compile_model,
        ),
        optim=Optim(learning_rate=1e-2, weight_decay=0.0),
    )


@pytest.fixture
def trained(prepared: Path):
    return train(_run(prepared))


def test_dataset_config_is_part_of_the_run(prepared: Path) -> None:
    assert _run(prepared).dataset == load_config(prepared)


def test_identical_results_are_equal(prepared: Path) -> None:
    config = _run(prepared, train_flops=1)
    result = TrainResult(
        steps=1,
        train_flops=1,
        quality=Quality(train_loss=1.0, val_loss=1.0, val_accuracy=0.5),
        speed=Speed(train_tokens_per_second=1.0, peak_gpu_bytes=None),
        config=config,
    )

    assert result == result.model_copy()


def test_different_validation_accuracy_is_not_equal(prepared: Path) -> None:
    config = _run(prepared, train_flops=1)
    result = TrainResult(
        steps=1,
        train_flops=1,
        quality=Quality(train_loss=1.0, val_loss=1.0, val_accuracy=0.5),
        speed=Speed(train_tokens_per_second=1.0, peak_gpu_bytes=None),
        config=config,
    )
    changed = result.quality.model_copy(update={"val_accuracy": 0.25})

    assert result != result.model_copy(update={"quality": changed})


def test_training_stops_when_recorded_flops_reach_the_budget(trained) -> None:
    assert trained.train_flops >= trained.config.schedule.train_flops


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
    result = train(_run(prepared, train_flops=1))

    assert result.steps == 1


def test_vocab_must_match_the_dataset(prepared: Path) -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        _run(prepared, train_flops=1, model=gpt2_config(vocab_size=64, n_layer=1, dropout=0.0))


def test_block_must_cover_the_sequence(prepared: Path) -> None:
    with pytest.raises(ValueError, match="block_size"):
        _run(prepared, train_flops=1, model=gpt2_config(block_size=8, n_layer=1, dropout=0.0))


def test_batch_must_fit_the_training_split(prepared: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _run(prepared, train_flops=1, batch_size=8)


@pytest.mark.parametrize("axis", [ComparisonAxis.LAYER_FLOPS, ComparisonAxis.PARAMETERS])
def test_other_axes_are_not_trained_yet(prepared: Path, axis: ComparisonAxis) -> None:
    with pytest.raises(NotImplementedError, match=axis.value):
        _run(prepared, train_flops=1, axis=axis)


def test_compiled_model_reaches_a_finite_loss(prepared: Path) -> None:
    result = train(_run(prepared, compile_model=True))

    assert torch.isfinite(torch.tensor(result.quality.train_loss))


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
