"""Keep CLI end-to-end tests inside their own files and simulated services."""

from pathlib import Path

import pytest

from compare_attention.cli import train as train_cli


@pytest.fixture(autouse=True)
def isolated_cli_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never discover developer secrets or perform a real upload from tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_REPO", "")
    monkeypatch.setattr(
        train_cli,
        "find_dotenv",
        lambda **_kwargs: str(tmp_path / ".env") if (tmp_path / ".env").is_file() else "",
    )

    def forbidden_upload(*_args, **_kwargs):
        raise AssertionError("CLI tests must explicitly replace the remote uploader")

    monkeypatch.setattr(train_cli, "upload_results", forbidden_upload)
