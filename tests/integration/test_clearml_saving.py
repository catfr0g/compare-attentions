# pylint: disable=redefined-outer-name
"""Сохранение в ClearML: официальный offline-режим SDK вместо сервера."""

import json
from pathlib import Path
from typing import Any

import pytest
from clearml.backend_api.session.session import Session
from clearml.backend_interface.base import InterfaceBase

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.result import Quality, Speed
from compare_attention.train.tracker import ClearMLTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result

pytestmark = pytest.mark.integration


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


@pytest.fixture
def offline_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Включает offline-режим SDK и направляет его выгрузку во временный кэш."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    monkeypatch.setenv("CLEARML_CACHE_DIR", str(cache))
    monkeypatch.setattr(Session, "_offline_mode", True)
    monkeypatch.setattr(InterfaceBase, "_offline_mode", True)
    return cache


def _run(config, task_id: str) -> None:
    tracker = ClearMLTracker()
    tracker.start(config)
    task_id.append(tracker.task_id)
    tracker.log_loss(0.5, 128)
    tracker.log_loss(0.25, 256)
    tracker.finish(
        train_result(
            config,
            steps=2,
            quality=Quality(train_loss=0.375, val_loss=0.4, val_accuracy=0.25),
            speed=Speed(train_tokens_per_second=100.0, peak_gpu_bytes=None),
        ),
        GPT2(gpt2_config(n_layer=1)),
    )


@pytest.fixture
def session(prepared: Path, offline_cache: Path) -> tuple[Path, str]:
    ids: list[str] = []
    _run(train_config(prepared), ids)
    folder = offline_cache / "offline" / ids[0]
    assert folder.exists()
    return folder, ids[0]


def test_the_config_lands_in_the_offline_session(session: tuple[Path, str]) -> None:
    folder, _task_id = session

    params = _task_params(folder)

    assert params.get("schedule/train_flops") == "1"
    assert params.get("model/attention") == "classic"


def test_the_scalars_lands_in_the_offline_session(session: tuple[Path, str]) -> None:
    folder, _task_id = session
    scalar_events = _jsonl(folder / "metrics.jsonl")

    scalars = _scalar_tuples(scalar_events)

    assert ("train", "loss", 0.5, 128) in scalars
    assert ("train", "loss", 0.25, 256) in scalars
    assert ("result", "train_loss", 0.375, 2) in scalars


def test_the_console_lands_in_the_offline_session(
    session: tuple[Path, str], offline_cache: Path
) -> None:
    folder, task_id = session
    logs = _jsonl(folder / "log.jsonl")

    assert any(event.get("msg") for event in logs)
    assert (offline_cache / "offline" / f"{task_id}.zip").is_file()


def _task_params(folder: Path) -> dict[str, str]:
    payload: dict[str, Any] = json.loads((folder / "task.json").read_text(encoding="utf-8"))
    sections = payload.get("hyperparams", {})
    flattened = [parameter for parameters in sections.values() for parameter in parameters.values()]
    return {parameter["name"]: parameter["value"] for parameter in flattened}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        events.extend(json.loads(line))
    return events


def _scalar_tuples(events: list[dict[str, Any]]) -> set[tuple[str, str, float, int]]:
    scalars = [event for event in events if event.get("type") == "training_stats_scalar"]
    return {(event["metric"], event["variant"], event["value"], event["iter"]) for event in scalars}
