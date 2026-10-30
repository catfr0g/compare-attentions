"""Через docker compose поднимаем настоящий сервер ClearML (для e2e)."""

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
from testcontainers.compose import DockerCompose

HERE = Path(__file__).parent

_FIXED_ACCESS_KEY = "EGRTCO8JMSIGI6S39GTP43NFWXDQOW"
_FIXED_SECRET_KEY = "LPEJbGJ6bK4tujQcmrD3i1dbMBDdwUwelVa-LG0K0FFmY9bzH_H0Sw"

_ENV_VARS = (
    "CLEARML_API_HOST",
    "CLEARML_WEB_HOST",
    "CLEARML_FILES_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
    "CLEARML_API_VERIFY_HOST",
)


def _docker_is_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _wait_for_apiserver(address: str, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{address}/debug.ping", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("clearml apiserver did not come up")


@pytest.fixture(scope="session")
def clearml_server() -> Iterator[str]:
    """Копировать переменные окружения на свой сервер и вернуть его адрес."""
    if not _docker_is_ready():
        pytest.skip("docker is not available")
    compose = DockerCompose(
        HERE,
        compose_file_name="docker-compose-clearml.yml",
        wait=False,
        services=["mongo", "redis", "elasticsearch", "apiserver"],
    )
    saved = {key: os.environ.get(key) for key in _ENV_VARS}
    try:
        compose.start()
        host = compose.get_service_host("apiserver", 8008)
        port = compose.get_service_port("apiserver", 8008)
        address = f"http://{host}:{port}"
        _wait_for_apiserver(address)
        os.environ.update(
            {
                "CLEARML_API_HOST": address,
                "CLEARML_WEB_HOST": address,
                "CLEARML_FILES_HOST": address,
                "CLEARML_API_ACCESS_KEY": _FIXED_ACCESS_KEY,
                "CLEARML_API_SECRET_KEY": _FIXED_SECRET_KEY,
                "CLEARML_API_VERIFY_HOST": "false",
            }
        )
        yield address
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        compose.stop()
