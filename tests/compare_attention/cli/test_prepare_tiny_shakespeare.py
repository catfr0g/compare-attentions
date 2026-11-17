"""Public preparation parser and fail-closed character-data selection."""

import importlib
from pathlib import Path

import pytest

from compare_attention.cli.train import TrainArgs, _ensure_dataset, parse_train_args
from compare_attention.train.config import Schedule


def test_missing_character_data_cannot_generate_kv(tmp_path):
    args = TrainArgs(
        data=tmp_path / "missing",
        schedule=Schedule(train_flops=1),
        dataset_task="tiny_shakespeare_char",
    )
    with pytest.raises(FileNotFoundError, match="prepare_tiny_shakespeare"):
        _ensure_dataset(args)
    assert not args.data.exists()


def test_public_prepare_parser():
    cli = importlib.import_module("compare_attention.cli.prepare_tiny_shakespeare")
    args = cli.parse_args(["--output", "new", "--source", "input.txt"])
    assert args.output == Path("new")
    assert args.source == Path("input.txt")
    assert args.context == 512


def test_legacy_cli_serialization_does_not_add_task_field(tmp_path):
    legacy = TrainArgs(data=tmp_path, schedule=Schedule(train_flops=1))
    assert "dataset_task" not in legacy.model_dump(mode="json")
    tiny = legacy.model_copy(update={"dataset_task": "tiny_shakespeare_char"})
    assert tiny.model_dump(mode="json")["dataset_task"] == "tiny_shakespeare_char"


def test_task_selector_survives_yaml_parser(tmp_path):
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "data: missing\ndataset_task: tiny_shakespeare_char\nschedule:\n  train_flops: 1\n"
    )
    assert parse_train_args(["--config", str(config)]).dataset_task == "tiny_shakespeare_char"
