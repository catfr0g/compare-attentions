"""Independent pinned upstream oracle and RNG replay, no upstream training imports."""

# The reference replay deliberately exposes every compared record and derived metric.
# pylint: disable=too-many-locals

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from compare_attention.data.selective_count import prepare_selective_count

UPSTREAM = Path(__file__).resolve().parents[3] / "local/reports/four-task-benchmark/upstream"


def reference_task():
    if not UPSTREAM.is_dir():
        pytest.skip("local pinned source audit not present")
    pins = json.loads((UPSTREAM / "pins.json").read_text())["task-reference"]
    package = types.ModuleType("_selective_reference")
    package.__path__ = []
    sys.modules["_selective_reference"] = package
    for module in ("base", "selective_count"):
        raw = (UPSTREAM / f"task-reference-tasks-{module}.py").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pins["files"][f"tasks/{module}.py"]["sha256"]
        name = "_selective_reference." + module
        spec = importlib.util.spec_from_file_location(
            name, UPSTREAM / f"task-reference-tasks-{module}.py"
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules["_selective_reference.selective_count"].SelectiveCountTask


def test_literal_rng_collation_and_metrics(tmp_path):
    task_type = reference_task()
    path = tmp_path / "count"
    config = prepare_selective_count(path, context=512, n_train=35, n_val=3, n_test=5)
    seen = set()
    for split, seed in config.split_seeds.items():
        task = task_type(n_symbols=32, seq_len=512, seed=seed)
        x = np.load(path / f"{split}_inputs.npy")
        y = np.load(path / f"{split}_labels.npy")
        items = [task._sample_one() for _ in range(len(x))]  # pylint: disable=protected-access
        rx, ry = task.collate(items, block_size=512)
        np.testing.assert_array_equal(x, rx)
        np.testing.assert_array_equal(y, ry)
        for row in x:
            key = hashlib.sha256(row.tobytes()).digest()
            assert key not in seen
            seen.add(key)
        p = np.zeros_like(y)
        source_metrics = task.metrics(p, y)
        true = y[:, -1] - config.class_low
        assert source_metrics["nmae"] == np.mean(true) / config.metric_normalizer
        assert source_metrics["acc"] == 0
    # Source supports 512 explicitly; PAD is reserved, not a sampled symbol.
    assert config.vocab_size == 546
    assert config.metric_normalizer == 17
    assert config.class_low + config.max_count == 545
