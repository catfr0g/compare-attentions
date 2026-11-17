"""Result tracking explicitly distinguishes early stop from budget completion."""

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.tracker import ClearMLTracker
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result
from tests.compare_attention.train.test_tracker import RecordingTask


def test_clearml_reports_early_stop_and_requested_vs_actual_budget(tmp_path):
    prepare_kv_retrieval(data_config(), tmp_path)
    config = train_config(tmp_path, train_flops=10)
    result = train_result(config, steps=2).model_copy(
        update={"train_flops": 2, "stop_reason": "early_stopping"}
    )
    task = RecordingTask({"project_name": "test"})
    tracker = ClearMLTracker(lambda _: task)
    tracker.start(config)
    tracker.finish(result, GPT2(gpt2_config(n_layer=1)))
    scalars = {(title, series): value for title, series, value, _ in task.logger.scalars}
    assert scalars["result", "early_stopped"] == 1.0
    assert scalars["result", "budget_exhausted"] == 0.0
    assert scalars["result", "requested_train_flops"] == 10.0
    assert scalars["result", "train_flops"] == 2.0
