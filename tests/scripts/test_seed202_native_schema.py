"""Replay the genuine MQAR config through the externally repaired recipe."""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

WORK = Path(os.environ["TMPDIR"]) / "mqar202-repair-work-v1"
SOURCE = Path(os.environ.get("MQAR_REPAIR_SOURCE", WORK / "parent/source/src"))
EXECUTION = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/"
    "remaining-three-after-tail-v1/qualified-launch-bridge-v1/execution-classic-20261118-v1"
)


@pytest.fixture(scope="module")
def recipe():
    """Use the actual archived function and its real frozen dependencies."""
    sys.path.insert(0, str(SOURCE))
    module = importlib.import_module("seed202_pilot.cloud_native")
    assert Path(module.__file__).resolve().is_relative_to(SOURCE.resolve())
    yield module.validate_recipe
    sys.path.remove(str(SOURCE))


@pytest.fixture
def genuine():
    """Read native first-save identity, never construct a model or restore RNG."""
    paths = list((EXECUTION / "failed-download-v1").rglob("out/run_identity.json"))
    assert len(paths) == 1
    identity = json.loads(paths[0].read_bytes())
    assert identity["status"] == "failed"
    config = identity["config"]
    assert "task" not in config["dataset"]
    return config


def case(family="classic"):
    """Complete task-policy envelope, independent of the downloaded config."""
    return {
        "id": f"mqar-k64-{family}-seed202",
        "task": "mqar-k64",
        "family": family,
        "seed": 202,
        "requested_flops": 5_000_000_000_000_000,
        "global_batch": 64,
        "local_batch": 32,
        "world_size": 2,
        "test_split_permitted": False,
    }


def test_genuine_first_save_native_kv_schema_is_valid(recipe, genuine):
    """The lawful absent dataset.task must not raise the proven KeyError."""
    before = copy.deepcopy(genuine)
    recipe(genuine, case())
    assert genuine == before


@pytest.mark.parametrize("family", ["classic", "deltanet"])
def test_both_mqar_families_use_the_frozen_declared_task(recipe, genuine, family):
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.matrix import config_payload

    expected = config_payload(require_authorized_case(case(family)["id"]))
    config = {**genuine, **{k: expected[k] for k in ("model", "schedule", "optim")}}
    recipe(config, case(family))
    recipe(json.loads(json.dumps(config)), json.loads(json.dumps(case(family))))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("dataset", "task", "kv_retrieval"),
        ("dataset", "seed", 202),
        ("dataset", "num_kv_pairs", 32),
        ("dataset", "vocab_size", 4096),
        ("dataset", "input_seq_len", 256),
        ("dataset", "n_train", 1000),
        ("dataset", "power_a", 0.02),
        ("model", "n_layer", 7),
        ("schedule", "seed", 101),
        ("schedule", "dtype", "float32"),
        ("schedule", "train_flops", 5000000000000001),
    ],
)
def test_wrong_scientific_values_rejected(recipe, genuine, section, key, value):
    genuine[section][key] = value
    with pytest.raises((ValueError, TypeError)):
        recipe(genuine, case())


@pytest.mark.parametrize("key", ["num_kv_pairs", "vocab_size", "input_seq_len", "seed"])
def test_missing_required_native_kv_field_rejected(recipe, genuine, key):
    del genuine["dataset"][key]
    with pytest.raises((ValueError, TypeError)):
        recipe(genuine, case())


@pytest.mark.parametrize("key,value", [("task", "sorting"), ("family", "deltanet"), ("seed", 101)])
def test_wrong_declared_case_rejected(recipe, genuine, key, value):
    wrong = {**case(), key: value}
    with pytest.raises(ValueError):
        recipe(genuine, wrong)


@pytest.mark.parametrize("family", ["deltanet"])
def test_sorting_predicate_and_scientific_recipe_unchanged(recipe, family):
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.matrix import config_payload

    spec = require_authorized_case(f"sorting-{family}-seed202")
    expected = config_payload(spec)
    config = {
        **{key: expected[key] for key in ("model", "schedule", "optim")},
        "dataset": {"task": "sorting"},
    }
    bound = {**case(family), "id": spec.case_id, "task": "sorting"}
    recipe(config, bound)
    config["dataset"]["task"] = "kv_retrieval"
    with pytest.raises(ValueError):
        recipe(config, bound)


def test_repair_requires_exact_authenticated_source():
    from seed202_native_schema import repair_native

    source = WORK / "parent/source/src/seed202_pilot/cloud_native.py"
    raw = source.read_bytes()
    repaired = repair_native(raw)
    assert b"and task_matches(config, spec)," in repaired
    assert b'config["dataset"]["task"]' not in repaired
    with pytest.raises(ValueError, match="EXACT_V7_NATIVE_SOURCE_REQUIRED"):
        repair_native(raw + b"\n")
