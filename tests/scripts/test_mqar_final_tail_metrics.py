"""Synthetic logits only: these values are not scientific model scores."""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_same_position_tail_records_and_independent_ce():
    spec = importlib.util.find_spec("mqar_final_tail_metrics")
    assert spec is not None, "MQAR task-correct record consumer is missing"
    metrics = importlib.import_module("mqar_final_tail_metrics")
    labels = np.array([[-1, 1, -1, 2], [0, -1, 1, -1], [-1, 2, 0, -1]])
    # Deliberately wrong one query; mask and chronological positions matter.
    predictions = [[1, 2], [0, 2], [2, 0]]
    records = [
        {
            "row": i,
            "positions": np.flatnonzero(y != -1).tolist(),
            "targets": y[y != -1].tolist(),
            "predictions": predictions[i],
            "nll": [math.log(3)] * 2,
        }
        for i, y in enumerate(labels)
    ]
    got = metrics.close_records(records, labels)
    assert got == {
        "rows": 3,
        "supervised_count": 6,
        "correct": 5,
        "accuracy": 5 / 6,
        "loss": math.log(3),
        "whole_example_correct": 2,
        "whole_example_exact_match": 2 / 3,
    }
    for mutate in (
        lambda r: r.pop(),
        lambda r: r[0].update(row=True),
        lambda r: r[1]["targets"].__setitem__(0, 2),
        lambda r: r[2]["nll"].__setitem__(0, float("nan")),
        lambda r: r[0]["positions"].reverse(),
        lambda r: r[0].update(oracle=True),
    ):
        import copy

        broken = copy.deepcopy(records)
        mutate(broken)
        with pytest.raises(ValueError):
            metrics.close_records(broken, labels)


def test_query_count_and_labels_not_inferred_from_task_name():
    spec = importlib.util.find_spec("mqar_final_tail_metrics")
    assert spec is not None, "MQAR task-correct record consumer is missing"
    metrics = importlib.import_module("mqar_final_tail_metrics")
    labels = np.array([[0, -1, -1], [1, 2, -1]])
    records = [
        {"row": 0, "positions": [0], "targets": [0], "predictions": [0], "nll": [0.2]},
        {
            "row": 1,
            "positions": [0, 1],
            "targets": [1, 2],
            "predictions": [0, 2],
            "nll": [0.4, 0.6],
        },
    ]
    got = metrics.close_records(records, labels)
    assert got["supervised_count"] == 3 and got["loss"] == pytest.approx(0.4)
