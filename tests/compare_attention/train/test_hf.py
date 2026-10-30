# pylint: disable=redefined-outer-name
"""Выгрузка результатов запуска в приватный репозиторий Hugging Face."""

from collections.abc import Callable
from pathlib import Path

import pytest

from compare_attention.train.hf import upload_results

REPO = "user/compare-attention"


class RecordingApi:
    """Подменяет HfApi и запоминает вызовы."""

    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls

    def create_repo(self, repo_id: str, private: bool = False, exist_ok: bool = False) -> None:
        self.calls.append(
            {"kind": "create_repo", "repo_id": repo_id, "private": private, "exist_ok": exist_ok}
        )

    def upload_folder(self, **kwargs) -> None:
        self.calls.append({"kind": "upload_folder", **kwargs})


@pytest.fixture
def calls() -> list[dict]:
    return []


@pytest.fixture
def api_factory(calls: list[dict]) -> Callable[[], RecordingApi]:
    def factory() -> RecordingApi:
        return RecordingApi(calls)

    return factory


def _uploaded(
    tmp_path: Path, api_factory: Callable[[], RecordingApi]
) -> tuple[Path, str, list[dict]]:
    target = tmp_path / "runs-01"
    target.mkdir()
    (target / "model.pt").write_bytes(b"checkpoint")

    url = upload_results(target, REPO, api_factory)
    return target, url, calls


def test_upload_results_creates_a_private_repo(
    tmp_path: Path, api_factory: Callable[[], RecordingApi], calls: list[dict]
) -> None:
    _uploaded(tmp_path, api_factory)

    assert calls[0] == {
        "kind": "create_repo",
        "repo_id": REPO,
        "private": True,
        "exist_ok": True,
    }


def test_upload_results_puts_the_run_in_a_subfolder(
    tmp_path: Path, api_factory: Callable[[], RecordingApi], calls: list[dict]
) -> None:
    target, _url, _calls = _uploaded(tmp_path, api_factory)
    upload = next(call for call in calls if call["kind"] == "upload_folder")

    assert upload["repo_id"] == REPO
    assert upload["repo_type"] == "model"
    assert upload["folder_path"] == target
    assert upload["path_in_repo"] == f"runs/{target.name}"


def test_upload_results_returns_the_repo_url(
    tmp_path: Path, api_factory: Callable[[], RecordingApi]
) -> None:
    _target, url, _calls = _uploaded(tmp_path, api_factory)

    assert url == f"https://huggingface.co/{REPO}/tree/main/runs/runs-01"
