"""Normal service completion: genuine response, never a scientific model replay."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import training_completion as completion

ROOT = Path(
    "E:/HermesArtifacts/compare-attention/sorting-mqar-seed202-pilot-v1/"
    "mqar-classic-resume-repair-v1/execution-resume-classic03"
)
EVENT_SHA = "ba1aa87445d644052512a8dac85c6639ed5c330f24fd73fe36c40b87341d5e1c"


@pytest.fixture
def genuine():
    raw = (ROOT / "diagnosis-v1/clearml-actual-safe-all-events.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EVENT_SHA
    value = json.loads(raw)
    (metrics,) = (ROOT / "failed-download-v1").rglob("out/metrics.jsonl")
    config = json.loads(metrics.read_text().splitlines()[-1])["config"]
    return value, config


def validate(value, config):
    return completion.validate_unsampled_readback(
        value, task_id=value["task_id"], updates=10780, validations=11, config=config
    )


def test_staged_normal_worker_and_service_consume_same_root_fix(tmp_path, genuine):
    import importlib.util

    from build_normal_training_lifecycle import stage_source

    archive = ROOT.parent / "mount-postreview-real03/sorting-mqar-seed202-pilot-v1.zip"
    source = stage_source(archive, tmp_path / "normal-source")
    spec = importlib.util.spec_from_file_location(
        "staged_normal_completion", source / "seed202_pilot/cloud_completion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value, config = genuine
    receipt = module.validate_unsampled_readback(
        value, task_id=value["task_id"], updates=10780, validations=11, config=config
    )
    assert receipt == validate(value, config)
    consumer = (source / "seed202_pilot/cloud_case.py").read_text()
    assert "from seed202_pilot.cloud_completion import" in consumer
    assert "validate_unsampled_readback(" in consumer


def test_genuine_completed_response_retains_all_resource_rows(genuine):
    value, config = genuine
    before = copy.deepcopy(value)
    receipt = validate(value, config)
    assert value == before
    assert receipt["event_count"] == 11252
    assert receipt["scientific_event_count"] == 10812
    assert receipt["resource_event_count"] == 440
    assert receipt["required_counts"]["train/loss"] == 10780
    assert receipt["expected_learning_rate_events"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "nan",
        "bool_value",
        "bool_iteration",
        "duplicate",
        "order",
        "foreign_task",
        "unknown_app",
        "fake_machine",
        "fake_gpu",
        "fake_title",
        "missing",
    ],
)
def test_rejects_invalid_rows_without_discarding_them(genuine, mutation):
    value, config = genuine
    losses = [row for row in value["events"] if row["metric"] == "train"]
    if mutation == "nan":
        losses[0]["value"] = float("nan")
    elif mutation == "bool_value":
        losses[0]["value"] = True
    elif mutation == "bool_iteration":
        losses[0]["iter"] = True
    elif mutation == "duplicate":
        losses[1]["iter"] = losses[0]["iter"]
    elif mutation == "order":
        losses[1]["iter"], losses[0]["iter"] = losses[0]["iter"], losses[1]["iter"]
    elif mutation == "foreign_task":
        losses[0]["task"] = "foreign-task"
    elif mutation == "missing":
        value["events"].remove(losses[0])
    else:
        metric, variant = {
            "unknown_app": ("train", "secret_extra"),
            "fake_machine": (":monitor:machine", "loss"),
            "fake_gpu": (":monitor:gpu", "gpu_0_loss"),
            "fake_title": (":monitor:machine:rank1", "cpu_usage"),
        }[mutation]
        value["events"].append({"metric": metric, "variant": variant, "iter": 1, "value": 1.0})
    value["total"] = len(value["events"])
    with pytest.raises(ValueError):
        validate(value, config)


def test_resource_schema_includes_all_producer_series_not_observed_count(genuine):
    value, config = genuine
    value["events"] = [row for row in value["events"] if not row["metric"].startswith(":monitor:")]
    value["total"] = len(value["events"])
    assert validate(value, config)["resource_event_count"] == 0
    machine = [
        "cpu_usage",
        "cpu_temperature",
        "memory_used_gb",
        "memory_free_gb",
        "disk_free_percent",
        "network_tx_mbs",
        "network_rx_mbs",
        "io_read_mbs",
        "io_write_mbs",
    ]
    gpu = ["temperature", "utilization", "mem_used_gb", "mem_usage", "mem_free_gb", "power_draw"]
    for title, names in (
        (":monitor:machine", machine),
        (":monitor:gpu", [f"gpu_7_{name}" for name in gpu]),
    ):
        value["events"] += [{"metric": title, "variant": n, "iter": 1, "value": 0.5} for n in names]
    value["total"] = len(value["events"])
    assert validate(value, config)["resource_event_count"] == len(machine) + len(gpu)


def tiny_response(config, policy, seed):
    from compare_attention.train.config import Optim
    from compare_attention.train.lr_schedule import next_learning_rate

    config = copy.deepcopy(config)
    config["schedule"].update(seed=seed, train_flops=30, val_every_steps=2)
    config["optim"]["lr_schedule"] = policy
    optim = Optim(**config["optim"])
    events = []
    for index in range(3):
        flops = (index + 1) * 10
        events.append({"metric": "train", "variant": "loss", "iter": flops, "value": 1.0})
        if policy != "constant":
            events.append(
                {
                    "metric": "train",
                    "variant": "learning_rate",
                    "iter": flops,
                    "value": next_learning_rate(optim, 30, index * 10, 10),
                }
            )
        if flops in (20, 30):
            for metric in ("loss", "accuracy"):
                events.append(
                    {"metric": "validation", "variant": metric, "iter": flops, "value": 0.5}
                )
    for name, number in {
        "steps": 3,
        "train_flops": 30,
        "requested_train_flops": 30,
        "early_stopped": 0,
        "budget_exhausted": 1,
        "train_loss": 1.0,
        "val_loss": 0.5,
        "val_accuracy": 0.5,
        "train_tokens_per_second": 10,
        "peak_gpu_bytes": 100,
    }.items():
        events.append({"metric": "result", "variant": name, "iter": 3, "value": number})
    return {
        "events": events,
        "total": len(events),
        "status": "completed",
        "task_id": "synthetic",
    }, config


@pytest.mark.parametrize("seed", [202, 303, 404])
@pytest.mark.parametrize("policy", ["constant", "warmup_cosine"])
def test_constant_and_scheduled_producer_contract_not_seed_specific(genuine, seed, policy):
    _, config = genuine
    value, config = tiny_response(config, policy, seed)
    receipt = completion.validate_unsampled_readback(
        value, task_id="synthetic", updates=3, validations=2, config=config
    )
    assert receipt["expected_learning_rate_events"] == (0 if policy == "constant" else 3)


@pytest.mark.parametrize("mutation", ["lr", "axis", "result", "schedule"])
def test_rejects_internally_counted_but_wrong_scientific_trace(genuine, mutation):
    _, config = genuine
    value, config = tiny_response(config, "warmup_cosine", 404)
    if mutation == "schedule":
        config["optim"]["lr_schedule"] = "invented"
    else:
        series = {"lr": "learning_rate", "axis": "loss", "result": "steps"}[mutation]
        row = next(r for r in value["events"] if r["variant"] == series)
        row["iter" if mutation == "axis" else "value"] += 1
    with pytest.raises(ValueError):
        completion.validate_unsampled_readback(
            value, task_id="synthetic", updates=3, validations=2, config=config
        )
