"""Synthetic CPU data/schema tests, never CUDA/DDP scientific qualification."""

import json
import math
from pathlib import Path

import pytest
from normal_training_plan import derive, digest

from compare_attention.data.tiny_shakespeare import prepare_tiny_shakespeare


def inputs(tmp_path):
    source = tmp_path / "synthetic.txt"
    source.write_text("abcd" * 40, encoding="utf-8")
    data = tmp_path / "data"
    prepare_tiny_shakespeare(source, data, context=8, expected_sha256=digest(source))
    package = tmp_path / "package"
    package.mkdir()
    (package / "normal-source-identity.json").write_text(
        json.dumps({"schema": "normal-source-identity-v1", "files": {}})
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        json.dumps(
            {
                "data": str(data),
                "dataset_task": "tiny_shakespeare_char",
                "distributed": True,
                "tracker": "clearml",
                "model": {
                    "vocab_size": 4,
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 8,
                    "attention": "classic",
                },
                "schedule": {
                    "train_flops": 1,
                    "batch_size": 4,
                    "seed": 202,
                    "dtype": "float32",
                    "compile": False,
                    "val_every_steps": 1,
                    "checkpoint_every_steps": 1,
                    "checkpoint_keep_last": 2,
                    "log_every_steps": 1,
                    "early_stopping_patience": 0,
                },
            }
        )
    )
    plan = tmp_path / "plan.json"
    selected = {
        "seed": 202,
        "task": "tiny_shakespeare_char",
        "family": "classic",
        "recipe_sha256": digest(recipe),
        "runtime_sha256": "0" * 64,
        "data_files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in data.iterdir()
        },
    }
    plan.write_text(
        json.dumps(
            {
                "schema": "normal-training-plan-v1",
                "purpose": "fresh_normal_training",
                "engineering_only": True,
                "source_identity_sha256": digest(package / "normal-source-identity.json"),
                "cases": {"synthetic-char": selected},
            }
        )
    )
    return package, plan, "synthetic-char", recipe, data


def test_shared_derive_routes_real_character_schema_and_global_pricing(tmp_path):
    case = derive(*inputs(tmp_path))
    assert case["task"] == "tiny_shakespeare_char"
    assert case["config"]["dataset"]["label_alignment"] == "same_position_next_character"
    assert "query_positions" not in case["config"]["dataset"]
    assert case["global_batch"] == 4 and case["local_batch"] == 2
    assert case["step_flops"] == 2 * case["rank_step_flops"]
    assert case["updates"] == 1


def test_tiny_plan_requires_raw_input_pin_even_when_present_on_disk(tmp_path):
    args = inputs(tmp_path)
    plan = json.loads(args[1].read_bytes())
    del plan["cases"][args[2]]["data_files"]["input.txt"]
    args[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="NORMAL_FULL_DATA_PINS"):
        derive(*args)


def cached_fixture(tmp_path):
    args = inputs(tmp_path)
    case = derive(*args)
    root = tmp_path / "result"
    (root / "out").mkdir(parents=True)
    identity = {
        "config": case["config"],
        "status": "completed",
        "stop_reason": "budget_exhausted",
        "final_parameters_sha256": "a" * 64,
    }
    (root / "out/run_identity.json").write_text(json.dumps(identity))
    (root / "out/model.pt").write_bytes(b"SYNTHETIC_NOT_TENSOR_OR_TRAINED_WEIGHTS")
    rows = [
        {"type": "validation", "loss": 1.25, "accuracy": 0.25, "flops": case["actual_flops"]},
        {
            "type": "result",
            "steps": case["updates"],
            "train_flops": case["actual_flops"],
            "config": case["config"],
            "provenance": identity,
            "quality": {"val_loss": 1.25, "val_accuracy": 0.25},
        },
    ]
    (root / "out/metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    native = {"updates": case["updates"], "final_sha256": "a" * 64, "world_size": 2}
    return root, case, args[-1], native


def test_native_lm_quality_uses_last_validation_without_forward(tmp_path):
    from normal_training_lm import cached_quality

    root, case, data, native = cached_fixture(tmp_path)
    report = cached_quality(root, case, data, native)
    assert report["split"] == "val" and report["test_scored"] is False
    assert report["character_cross_entropy"] == 1.25
    assert report["perplexity"] == math.exp(1.25)
    assert (
        report["supervised_targets"]
        == case["config"]["dataset"]["splits"]["val"]["supervised_targets"]
    )
    assert report["step"] == case["updates"]
    assert report["final_parameters_sha256"] == native["final_sha256"]


@pytest.mark.parametrize("failure", ["nan", "wrong_step", "wrong_weights", "wrong_labels"])
def test_native_lm_quality_rejects_unbound_cached_values(tmp_path, failure):
    from normal_training_lm import cached_quality

    root, case, data, native = cached_fixture(tmp_path)
    path = root / "out/metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "nan":
        rows[0]["loss"] = float("nan")
    elif failure == "wrong_step":
        rows[0]["flops"] -= 1
    elif failure == "wrong_weights":
        native["final_sha256"] = "b" * 64
    else:
        (data / "val_labels.npy").write_bytes(b"SYNTHETIC_CORRUPT")
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        cached_quality(root, case, data, native)


def test_shared_task_producer_and_host_accept_actual_cached_lm_report(tmp_path, monkeypatch):
    import normal_training_native
    from unified_normal_host import task_quality
    from unified_normal_task import native_character_task

    inherited = Path(
        "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/"
        "sorting-finish-import-fix29/static-preflight-reorder42/next-mqar-delta404-45/"
        "candidate/payload/normal-package/inherited/operator"
    )
    if not inherited.is_dir():
        pytest.skip("saved shared recovery implementation unavailable")
    monkeypatch.syspath_prepend(str(inherited))

    root, case, data, native = cached_fixture(tmp_path)
    case["engineering_only"] = False
    package = tmp_path / "package"
    (package / "compare_attention-0.1.0-py3-none-any.whl").write_bytes(b"SYNTHETIC_WHEEL")
    calls = []

    def synthetic_native(path, selected):
        calls.append((path, selected))
        return native

    monkeypatch.setattr(normal_training_native, "verify_native", synthetic_native)
    native_character_task(root, case, data)
    accepted = task_quality(package, root, case, data)
    assert accepted["status"] == "ACCEPTED" and len(calls) == 2
    assert accepted["next_launch_authorized"] is False
    report = root / "task-val.json"
    saved = json.loads(report.read_bytes())
    saved["character_cross_entropy"] = 0.1
    report.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="NORMAL_LM_CACHED_REPORT_JOIN"):
        task_quality(package, root, case, data)


def test_tiny_request_builder_freezes_current_wheel_raw_data_and_real_entry(tmp_path, monkeypatch):
    from build_tiny_training_request import build

    args = inputs(tmp_path)
    inherited = Path(
        "E:/HermesArtifacts/compare-attention/remaining-seeds-sorting-mqar-v1/"
        "sorting-finish-import-fix29/static-preflight-reorder42/next-mqar-delta404-45/"
        "candidate/payload/normal-package"
    )
    if not inherited.is_dir():
        pytest.skip("authenticated shared lifecycle package unavailable")

    # An existing real current wheel, not a synthetic scientific payload.
    wheel = Path(
        "E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1/"
        "preparation02/wheel/compare_attention-0.1.0-py3-none-any.whl"
    )
    if not wheel.is_file():
        pytest.skip("current wheel unavailable")
    request = build(
        inherited,
        Path(__file__).parents[2],
        wheel,
        args[3],
        args[-1],
        tmp_path / "request",
        "TinyClassic202",
        "0" * 64,
    )
    saved = json.loads((request / "request.json").read_bytes())
    assert saved["launch_authority"] is False
    assert (request / "data/tiny/input.txt").read_bytes() == (args[-1] / "input.txt").read_bytes()
    assert (request / "normal-package" / wheel.name).read_bytes() == wheel.read_bytes()
    assert "--nproc-per-node=2" in (request / "normal-package/setup.py").read_text()
    assert saved["entrypoint_argv"][0:2] == ["uv", "run"]
    import importlib.util
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location(
        "tiny_emitted_setup", request / "normal-package/setup.py"
    )
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    actual_popen = subprocess.Popen
    captured = []

    def stdlib_child(argv, **kwargs):
        captured.append((argv, kwargs["env"]))
        # Capture the real torchrun/process/env seam, then execute only stdlib.
        # This is NOT a CUDA/DDP or native callback qualification.
        return actual_popen(
            [*argv[:9], "-c", "import os;assert 'HF_TOKEN' not in os.environ"], **kwargs
        )

    monkeypatch.setattr(setup.subprocess, "Popen", stdlib_child)
    monkeypatch.setenv("HF_TOKEN", "SYNTHETIC_DO_NOT_PROPAGATE")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    worker_args = saved["entrypoint_argv"][saved["entrypoint_argv"].index("--") + 1 :]
    assert (
        setup.main(
            ["--uv", saved["entrypoint_argv"][8], "--python", sys.executable, "--", *worker_args]
        )
        == 0
    )
    assert "--nproc-per-node=2" in captured[0][0]
    assert "--no-python" in captured[0][0]
    assert "HF_TOKEN" not in captured[0][1] and "CUDA_VISIBLE_DEVICES" not in captured[0][1]
    assert "normal-package" in captured[0][1]["PYTHONPATH"]
    with pytest.raises(ValueError, match="CREATE_ONLY"):
        build(
            inherited,
            Path(__file__).parents[2],
            wheel,
            args[3],
            args[-1],
            request,
            "TinyClassic202",
            "0" * 64,
        )
    # Separate synthetic CPU qualification: real two-rank worker + SDK transport
    # fixtures + native FULL callbacks. Never the real corpus/config or cloud.
    monkeypatch.setattr(setup.subprocess, "Popen", actual_popen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.delenv("HF_TOKEN")
    package = request / "normal-package"
    plan = json.loads(args[1].read_bytes())
    plan["source_identity_sha256"] = digest(package / "normal-source-identity.json")
    synthetic_plan = tmp_path / "synthetic-engineering-plan.json"
    synthetic_plan.write_text(json.dumps(plan))
    private = tmp_path / "synthetic-only.env"
    private.write_text(
        "\n".join(
            [
                "CLEARML_API_HOST=https://fixture-api.invalid",
                "CLEARML_WEB_HOST=https://fixture-web.invalid",
                "CLEARML_FILES_HOST=https://fixture-files.invalid",
                "CLEARML_API_ACCESS_KEY=fixture-access",
                "CLEARML_API_SECRET_KEY=fixture-secret",
                "HF_TOKEN=hf_FIXTURE_ONLY",
                "HF_REPO=fixture/repo",
            ]
        )
    )
    from unified_normal_task import native_character_task

    from tests.scripts.test_normal_training_closure import saved_phase

    synthetic = (args[2], args[3], synthetic_plan, private, args[-1])
    output = tmp_path / "synthetic-cpu-run"
    for phase in ("worker", "finish", "accept"):
        saved_phase(package, synthetic, output, phase)
    monkeypatch.syspath_prepend(str(package / "inherited/operator"))
    monkeypatch.syspath_prepend(str(package / "src"))
    case = derive(package, synthetic_plan, args[2], args[3], args[-1])
    report = native_character_task(output, case, args[-1])
    assert report["step"] == 1 and math.isfinite(report["character_cross_entropy"])
    assert report["additional_forward_executed"] is False
    accepted = json.loads((tmp_path / "synthetic-cpu-run-normal-accepted.json").read_bytes())
    assert accepted["scope"] == "ENGINEERING_CPU_SDK_FIXTURE_ONLY"
    assert len(accepted["immutable_publications"]) == 2
