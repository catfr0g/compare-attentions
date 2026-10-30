"""Kernel-шаблон для генерации исходника ядра Kaggle."""

SECRETS = (
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
    "CLEARML_API_HOST",
    "CLEARML_WEB_HOST",
    "CLEARML_FILES_HOST",
    "HF_TOKEN",
    "HF_REPO",
)


def secrets_block() -> str:
    """Многострочный кортеж для встройки в kernel-код."""
    rows = ",\n    ".join(f'"{key}"' for key in SECRETS)
    return "(\n    " + rows + ",\n)"


_KERNEL_TEMPLATE = '''"""Обучение compare-attention на Kaggle (сгенерировано train_kaggle.py)."""

import glob
import os
import subprocess
import sys
from pathlib import Path

_SECRETS = {secrets}

_PREPARE = {prepare}
_TRAIN = {train}


def _secrets() -> None:
    """Env-файлы dataset -> окружение (Kaggle Secrets как fallback)."""
    for env_file in glob.glob("/kaggle/input/**/*.env", recursive=True):
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
    except Exception:
        return
    for name in _SECRETS:
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ.setdefault(name, value)


def main() -> None:
    _secrets()
    wheels = glob.glob({wheel_glob!r}, recursive=True)
    pinned = glob.glob("/kaggle/input/**/requirements.txt", recursive=True)
    if pinned:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--root-user-action=ignore",
                "-r",
                pinned[0],
            ]
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--root-user-action=ignore",
            wheels[0],
        ]
    )
    subprocess.check_call(
        [sys.executable, "-m", "compare_attention.cli.prepare_kv_retrieval", *(_PREPARE)]
    )
    subprocess.check_call([sys.executable, "-m", "compare_attention.cli.train", *(_TRAIN)])


main()
'''
