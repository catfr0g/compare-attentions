"""Сливаем локальные артефакты запуска в приватный репозиторий Hugging Face."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi


def upload_results(
    result_dir: Path, repo_id: str, api_factory: Callable[[], HfApi] | None = None
) -> str:
    """Каталог запуска уезжает в репозиторий `runs/<имя каталога>`."""
    api = (api_factory or HfApi)()
    token = os.getenv("HF_TOKEN")
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=result_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"runs/{result_dir.name}",
        token=token,
        commit_message=f"Добавить результат запуска {result_dir.name}",
    )
    return f"https://huggingface.co/{repo_id}/tree/main/runs/{result_dir.name}"
