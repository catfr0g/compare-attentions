"""CLI staged-серии без реальных ключей и cloud writes."""

from pathlib import Path

import pytest

from compare_attention import kaggle_series_cli
from tests.compare_attention.test_kaggle_artifacts import make_inputs
from tests.compare_attention.test_kaggle_series import Runner


def test_stage_cli_never_loads_credentials(tmp_path, monkeypatch, capsys):
    configs, data = make_inputs(tmp_path)

    def forbidden():
        pytest.fail("staging must not load credentials")

    monkeypatch.setattr(kaggle_series_cli, "authenticate", forbidden)
    runner = Runner()
    assert (
        kaggle_series_cli.main(
            [
                "stage",
                "--root",
                str(tmp_path),
                "--owner",
                "user",
                "--data",
                str(data),
                "--cache",
                str(tmp_path / "staged"),
                "--configs",
                *map(str, configs),
            ],
            run=runner,
        )
        == 0
    )
    assert Path(capsys.readouterr().out.strip()).is_file()
    assert not any("kaggle" in command[0] for command in runner.commands)


def test_help_never_loads_credentials(monkeypatch):
    monkeypatch.setattr(kaggle_series_cli, "authenticate", lambda: pytest.fail("credential read"))
    with pytest.raises(SystemExit) as caught:
        kaggle_series_cli.main(["--help"])
    assert caught.value.code == 0
