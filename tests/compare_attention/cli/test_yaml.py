# pylint: disable=redefined-outer-name
"""Конфигурация экспериментов через YAML."""

import sys
import textwrap
from pathlib import Path

import pytest

from compare_attention.cli.train import load_experiment, parse_train_args


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    experiment = textwrap.dedent(
        """
        data: data/kv_retrieval
        schedule:
          train_flops: 7
        optim:
          weight_decay: 0.05
        model:
          n_layer: 3
          attention: deltanet
        kv:
          input_seq_len: 16
          num_kv_pairs: 2
          n_train: 4
        data: data/kv_retrieval
        tracker: disabled
        """
    )
    path = tmp_path / "exp.yaml"
    path.write_text(experiment, encoding="utf-8")
    return path


def test_yaml_fills_the_defaults(yaml_path: Path) -> None:
    args = load_experiment(yaml_path)

    assert args.schedule.train_flops == 7
    assert args.schedule.batch_size == 8
    assert args.optim.weight_decay == 0.05
    assert args.model.attention == "deltanet"
    assert args.model.n_layer == 3
    assert args.kv.input_seq_len == 16
    assert args.kv.num_kv_pairs == 2
    assert args.kv.n_train == 4
    assert args.data == Path("data/kv_retrieval")
    assert args.tracker == "disabled"


def test_cli_flags_over_the_yaml(yaml_path: Path) -> None:
    args = parse_train_args(["--config", str(yaml_path), "--train-flops", "9"])

    assert args.schedule.train_flops == 9


def test_yaml_values_stay_without_cli_flags(yaml_path: Path) -> None:
    args = parse_train_args(["--config", str(yaml_path)])

    assert args.schedule.train_flops == 7
    assert args.model.attention == "deltanet"


def test_none_argv_falls_back_to_sys_argv(yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(yaml_path)])

    args = parse_train_args(None)

    assert args.schedule.train_flops == 7
    assert args.tracker == "disabled"


def test_parse_without_config_requires_the_data_flag() -> None:
    args = parse_train_args(["--data", "data/kv_retrieval", "--train-flops", "1"])

    assert args.config is None
