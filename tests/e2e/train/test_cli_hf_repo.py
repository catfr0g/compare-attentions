# pylint: disable=redefined-outer-name
"""Правила выбора репозитория Hugging Face для команды обучения."""

from pathlib import Path

import pytest

from compare_attention.cli.train import main
from compare_attention.data.store import prepare_kv_retrieval
from tests.compare_attention.cli.factories import train_flags
from tests.compare_attention.data.factories import config as data_config


class RecordingUpload:  # pylint: disable=too-few-public-methods
    """Подменяет выгрузку в Hugging Face и запоминает вызовы."""

    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls

    def __call__(self, result_dir: Path, repo_id: str) -> str:
        self.calls.append({"result_dir": result_dir, "repo_id": repo_id})
        return f"https://fake/{repo_id}"


@pytest.fixture
def calls() -> list[dict]:
    return []


@pytest.fixture
def upload(calls: list[dict], monkeypatch: pytest.MonkeyPatch) -> RecordingUpload:
    recorder = RecordingUpload(calls)
    monkeypatch.setattr("compare_attention.cli.train.upload_results", recorder)
    return recorder


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


def test_the_repo_from_the_environment_is_used(
    prepared: Path, tmp_path: Path, upload: RecordingUpload, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_REPO", "user/from-env")
    result_dir = tmp_path / "res"

    code = main(train_flags(prepared, tracker="disabled", result_dir=result_dir))

    assert code == 0
    assert upload.calls == [{"result_dir": result_dir, "repo_id": "user/from-env"}]


def test_the_flag_overrides_the_environment(
    prepared: Path, tmp_path: Path, upload: RecordingUpload, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_REPO", "user/from-env")
    result_dir = tmp_path / "res"

    code = main(
        train_flags(prepared, tracker="disabled", result_dir=result_dir, hf_repo="user/from-flag")
    )

    assert code == 0
    assert upload.calls[0]["repo_id"] == "user/from-flag"


def test_the_dotenv_file_feeds_the_environment(
    prepared: Path, tmp_path: Path, upload: RecordingUpload, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_REPO", raising=False)
    (tmp_path / ".env").write_text("HF_REPO=user/from-file\n", encoding="utf-8")
    result_dir = tmp_path / "res"

    code = main(train_flags(prepared, tracker="disabled", result_dir=result_dir))

    assert code == 0
    assert upload.calls == [{"result_dir": result_dir, "repo_id": "user/from-file"}]
