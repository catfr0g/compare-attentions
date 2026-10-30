"""Команда обучения: полный прогон."""

import json
from pathlib import Path
from typing import Any

import pytest

from compare_attention.cli.train import main
from compare_attention.data.store import prepare_kv_retrieval
from tests.compare_attention.cli.factories import train_flags
from tests.compare_attention.data.factories import config as data_config


@pytest.fixture(autouse=True)
def clean_hf_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ограждает прогон от репозиториев, выбранных в окружении."""
    monkeypatch.delenv("HF_REPO", raising=False)


def _run(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    result_dir = tmp_path / "res"
    code = main(train_flags(tmp_path, tracker="disabled", result_dir=result_dir))
    assert code == 0
    return result_dir


def test_train_command_writes_a_log_to_the_result_dir(tmp_path: Path) -> None:
    result_dir = _run(tmp_path)

    assert (result_dir / "train.log").exists()


def test_train_command_leaves_a_checkpoint(tmp_path: Path) -> None:
    result_dir = _run(tmp_path)

    assert (result_dir / "model.pt").is_file()


def test_train_command_leaves_the_training_journal(tmp_path: Path) -> None:
    result_dir = _run(tmp_path)

    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in (result_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows
    assert rows[0]["type"] == "step"
    assert rows[-1]["type"] == "result"


def test_train_command_succeeds_with_the_default_result_dir(tmp_path: Path) -> None:
    prepare_kv_retrieval(data_config(), tmp_path)

    code = main(train_flags(tmp_path, tracker="disabled"))

    assert code == 0
