# pylint: disable=redefined-outer-name
"""Сохранение в настоящий сервер ClearML, поднятый docker compose."""

from pathlib import Path

import pytest
from clearml import Task

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.tracker import ClearMLTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result

pytestmark = [pytest.mark.docker, pytest.mark.usefixtures("clearml_server")]


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_kv_retrieval(data_config(), tmp_path)
    return tmp_path


def test_the_whole_run_reaches_the_server(prepared: Path) -> None:
    config = train_config(prepared)
    tracker = ClearMLTracker()

    tracker.start(config)
    task_id = tracker.task_id
    tracker.log_loss(0.5, 128)
    tracker.log_loss(0.25, 256)
    tracker.finish(train_result(config, steps=2), GPT2(gpt2_config(n_layer=1)))

    task = Task.get_task(task_id=task_id)
    params = task.get_parameters()
    assert any(key.endswith("train_flops") and value == "1" for key, value in params.items())

    reported = task.get_reported_scalars()
    values = [step["value"] for step in reported.get("train", {}).get("loss", [])]
    assert sorted(values) == [0.25, 0.5]
