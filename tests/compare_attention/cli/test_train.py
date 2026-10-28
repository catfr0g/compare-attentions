"""Команда обучения."""

from pathlib import Path

import tyro

from compare_attention.cli.train import TrainArgs, main
from compare_attention.data.store import prepare_kv_retrieval
from tests.compare_attention.data.factories import config as data_config

ROOT = Path(__file__).resolve().parents[3]


def test_train_command_is_registered() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'train = "compare_attention.cli.train:main"' in project


def test_cli_accepts_a_flop_budget(tmp_path: Path) -> None:
    prepare_kv_retrieval(data_config(), tmp_path)

    args = tyro.cli(
        TrainArgs,
        args=["--data", str(tmp_path), "--train-flops", "1", "--batch-size", "2"],
    )

    assert args.schedule.train_flops == 1


def test_cli_trains_until_the_budget(tmp_path: Path) -> None:
    prepare_kv_retrieval(data_config(), tmp_path)

    code = main(
        [
            "--data",
            str(tmp_path),
            "--train-flops",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--n-embd",
            "8",
            "--vocab-size",
            "32",
            "--block-size",
            "16",
            "--dropout",
            "0",
            "--seed",
            "0",
            "--no-compile",
        ]
    )

    assert code == 0
