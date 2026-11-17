"""Логирование запуска не зависит от настройки корневого логгера."""

import logging
from contextlib import nullcontext
from pathlib import Path

import pytest

from compare_attention.cli.train import _configure_logging


@pytest.mark.parametrize("fail", [False, True])
def test_logging_preserves_existing_handlers_and_cleans_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fail: bool
) -> None:
    root_handlers = list(logging.getLogger().handlers)
    package_logger = logging.getLogger("compare_attention")
    package_handlers = list(package_logger.handlers)
    previous_level = package_logger.level
    expected_error = pytest.raises(RuntimeError, match="training failed") if fail else nullcontext()

    with expected_error, _configure_logging(tmp_path / "train.log"):
        logging.getLogger("compare_attention.train.loop").info("training progress")
        if fail:
            raise RuntimeError("training failed")

    assert "training progress" in (tmp_path / "train.log").read_text(encoding="utf-8")
    assert "training progress" in capsys.readouterr().out
    assert logging.getLogger().handlers == root_handlers
    assert package_logger.handlers == package_handlers
    assert package_logger.level == previous_level
