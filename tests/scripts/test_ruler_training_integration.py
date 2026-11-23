"""Synthetic-only RULER wrapper tests; no scientific corpus execution."""

import json

import pytest
from normal_training_plan import derive, digest

from compare_attention.data.tracking import prepare_tracking


def inputs(tmp_path):
    data = tmp_path / "data"
    prepare_tracking(
        data, task="ruler_variable_tracking", n_train=4, n_val=3, n_test=2, pad_to_128=True
    )
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
                "dataset_task": "ruler_variable_tracking",
                "distributed": True,
                "tracker": "clearml",
                "model": {
                    "vocab_size": 116,
                    "block_size": 128,
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
        "task": "ruler_variable_tracking",
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
                "engineering_only": False,
                "source_identity_sha256": digest(package / "normal-source-identity.json"),
                "cases": {"RulerClassic202": selected},
            }
        )
    )
    return package, plan, "RulerClassic202", recipe, data


def test_ruler_derive_uses_native_schema_and_global_pricing(tmp_path):
    case = derive(*inputs(tmp_path))
    assert case["task"] == case["config"]["dataset"]["name"] == "ruler_variable_tracking"
    assert case["config"]["dataset"]["input_seq_len"] == 128
    assert case["config"]["dataset"]["prompt_length"] == 114
    assert case["global_batch"] == 2 * case["local_batch"] == 4
    assert case["step_flops"] == 2 * case["rank_step_flops"]
    assert case["checkpoint_steps"] == [1]


@pytest.mark.parametrize("missing", ["test_inputs.npy", "test_labels.npy"])
def test_ruler_derive_requires_all_six_arrays(tmp_path, missing):
    args = inputs(tmp_path)
    value = json.loads(args[1].read_bytes())
    del value["cases"][args[2]]["data_files"][missing]
    args[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="NORMAL_FULL_DATA_PINS"):
        derive(*args)


def test_observer_matches_unchanged_native_and_records_actual_feedback(tmp_path):
    import numpy as np
    import torch
    from normal_training_tracking import observe_generation, score_predictions

    from compare_attention.data.tracking import RulerConfig
    from compare_attention.tracking_evaluation import generated_metrics
    from tests.compare_attention.test_tracking_evaluation import FeedbackModel

    data = inputs(tmp_path)[-1]
    config = RulerConfig.model_validate_json((data / "meta.json").read_bytes())
    x, y = (np.load(data / f"val_{kind}.npy") for kind in ("inputs", "labels"))
    plain, observed = FeedbackModel(config.vocab_size), FeedbackModel(config.vocab_size)
    state = torch.random.get_rng_state().clone()
    expected = generated_metrics(plain, x, y, config, batch_size=2)
    actual, records = observe_generation(observed, x, y, config, batch_size=2)
    assert actual == expected == score_predictions(x, y, config, records)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(plain.calls) == len(observed.calls) == 10
    assert not observed._forward_hooks
    assert [c.shape[1] for c in observed.calls] == [114, 115, 116, 117, 118] * 2
    assert [r["row"] for r in records] == [0, 1, 2]
    for a, b in zip(plain.calls, observed.calls, strict=True):
        assert torch.equal(a, b)
    assert actual["generated_chain_exact_match"] == 0
    assert actual["generated_valid_variable_predictions"] < 15
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[:, config.prompt_length :] = 0
    changed_y[:, config.prompt_length - 1 :] = 0
    _, again = observe_generation(
        FeedbackModel(config.vocab_size), changed_x, changed_y, config, batch_size=2
    )
    assert again == records


def test_observer_removes_hook_on_native_nonfinite_failure(tmp_path):
    import numpy as np
    from normal_training_tracking import observe_generation

    from compare_attention.data.tracking import RulerConfig
    from tests.compare_attention.test_tracking_evaluation import FeedbackModel

    class Nonfinite(FeedbackModel):
        def forward(self, x):
            return super().forward(x) * float("nan")

    data = inputs(tmp_path)[-1]
    config = RulerConfig.model_validate_json((data / "meta.json").read_bytes())
    x, y = (np.load(data / f"val_{kind}.npy") for kind in ("inputs", "labels"))
    model = Nonfinite(config.vocab_size)
    with pytest.raises(ValueError, match="^nonfinite generated logits$"):
        observe_generation(model, x, y, config, batch_size=2)
    assert not model._forward_hooks


def test_cached_report_binds_last_iterate_and_pure_records(tmp_path):
    import copy

    import numpy as np
    from normal_training_tracking import cached_report, observe_generation

    from compare_attention.data.tracking import RulerConfig
    from tests.compare_attention.test_tracking_evaluation import FeedbackModel

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
    (root / "out/model.pt").write_bytes(b"SYNTHETIC_NOT_WEIGHTS")
    native = {"updates": case["updates"], "final_sha256": "a" * 64}
    config = RulerConfig.model_validate(case["config"]["dataset"], strict=True)
    x, y = (np.load(args[-1] / f"val_{kind}.npy") for kind in ("inputs", "labels"))
    metrics, records = observe_generation(FeedbackModel(config.vocab_size), x, y, config)
    report = cached_report(root, case, args[-1], native, records)
    assert report["metrics"] == metrics
    assert report["checkpoint"]["parameters_sha256"] != report["checkpoint"]["weights_file_sha256"]
    assert report["selection"] == "last_full_budget_iterate" and report["test_scored"] is False
    assert report == json.loads(json.dumps(report, allow_nan=False))
    for mutation in ("duplicate", "truncate", "bool", "bad_vocab"):
        bad = copy.deepcopy(records)
        if mutation == "duplicate":
            bad[1]["row"] = 0
        elif mutation == "truncate":
            bad.pop()
        else:
            bad[0]["prediction"][0] = True if mutation == "bool" else config.vocab_size
        with pytest.raises(ValueError, match="NORMAL_RULER_RECORD"):
            cached_report(root, case, args[-1], native, bad)
    changed = copy.deepcopy(case)
    changed["updates"] += 1
    with pytest.raises(ValueError, match="NORMAL_RULER_LAST_ITERATE_JOIN"):
        cached_report(root, changed, args[-1], native, records)


def test_host_accepts_only_whole_bound_cache_without_evaluation(tmp_path, monkeypatch):
    import hashlib
    import zipfile

    import normal_training_native
    from normal_training_tracking import cached_report
    from unified_normal_host import inference_readback, task_quality

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
    (root / "out/model.pt").write_bytes(b"SYNTHETIC_NOT_WEIGHTS")
    native = {"updates": case["updates"], "final_sha256": "a" * 64}
    monkeypatch.setattr(normal_training_native, "verify_native", lambda *_: native)
    records = [{"row": i, "prediction": [0] * 5} for i in range(3)]
    report = cached_report(root, case, args[-1], native, records)
    (root / "task-val.json").write_text(json.dumps(report))
    wheel = args[0] / "compare_attention-0.1.0-py3-none-any.whl"
    # This synthetic source fixture is not represented as a real scientific wheel.
    names = [
        "compare_attention.evaluation_checkpoint",
        "compare_attention.tracking_evaluation",
        "compare_attention.models.classic",
    ]
    with zipfile.ZipFile(wheel, "w") as saved:
        for name in names:
            saved.writestr(name.replace(".", "/") + ".py", b"SYNTHETIC")
    source = {
        "no_training_method_replacement": True,
        "wheel_sha256": digest(wheel),
        "imported_modules": {
            name: {
                "member": name.replace(".", "/") + ".py",
                "sha256": hashlib.sha256(b"SYNTHETIC").hexdigest(),
            }
            for name in names
        },
    }
    (root / "task-source.json").write_text(json.dumps(source))
    import compare_attention.evaluation_checkpoint as evaluation

    monkeypatch.setattr(
        evaluation, "load_evaluation_model", lambda *_: pytest.fail("HOST attempted model load")
    )
    assert task_quality(args[0], root, case, args[-1])["status"] == "ACCEPTED"
    assert inference_readback(root, case, {})["scope"] == (
        "NATIVE_CLOUD_GENERATION_NO_SEPARATE_INFERENCE_SERVICE"
    )
    assert inference_readback(root, case, {})["diagnostic"]["status"] == "not_required"
    (root / "task-val.json").write_text('{"split":"test",' + json.dumps(report)[1:])
    with pytest.raises(ValueError, match="NORMAL_RULER_JSON_DUPLICATE"):
        task_quality(args[0], root, case, args[-1])
    report["metrics"]["generated_chain_exact_match"] = 1.0
    (root / "task-val.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="NORMAL_RULER_CACHED_REPORT_JOIN"):
        task_quality(args[0], root, case, args[-1])


def test_cloud_task_refuses_host_before_any_model_load(tmp_path, monkeypatch):
    from normal_training_tracking import cloud_task

    import compare_attention.evaluation_checkpoint as evaluation

    monkeypatch.setattr(
        evaluation,
        "load_evaluation_model",
        lambda *_: pytest.fail("CPU attempted scientific model load"),
    )
    with pytest.raises(ValueError, match="NORMAL_RULER_CLOUD_ONLY"):
        cloud_task(tmp_path, tmp_path, {}, tmp_path)


def test_actual_ruler_builder_emits_current_science_and_transitive_imports(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from build_ruler_training_request import build

    base = Path("E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1")
    parent = base / "provider04/first-classic202-candidate03/payload/normal-package"
    wheel = base / "preparation02/wheel/compare_attention-0.1.0-py3-none-any.whl"
    assert parent.is_dir() and wheel.is_file(), "authenticated local fixtures required"
    args = inputs(tmp_path)
    output = build(
        parent,
        Path(__file__).parents[2],
        wheel,
        args[3],
        args[-1],
        tmp_path / "request",
        "RulerClassic202",
        "0" * 64,
    )
    saved = json.loads((output / "request.json").read_bytes())
    assert saved["status"] == "UNBOUND_PARENT_REVIEW_REQUIRED"
    assert saved["launch_authority"] is False
    assert (output / "normal-package" / wheel.name).read_bytes() == wheel.read_bytes()
    assert {p.name for p in (output / "data/ruler").iterdir()} == {
        "meta.json",
        *(f"{s}_{k}.npy" for s in ("train", "val", "test") for k in ("inputs", "labels")),
    }
    code = """import sys,runpy
from pathlib import Path
p=Path(sys.argv[1]);sys.path[:0]=[str(p),str(p/'src'),str(p/'inherited/operator')]
import normal_training_tracking as helper
import normal_training_worker as worker
assert Path(helper.__file__).parent==p/'src'
assert Path(worker.__file__).parent==p/'src'
runpy.run_path(str(p/'worker.py'),run_name='emitted_import_only')
runpy.run_path(str(p.parent/'normal-task.py'),run_name='emitted_import_only')
print('EMITTED_IMPORT_CLOSURE_OK')
"""
    child = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            "-I",
            "-B",
            "-c",
            code,
            str(output / "normal-package"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "EMITTED_IMPORT_CLOSURE_OK"


def test_native_cached_links_follow_wrong_predicted_chain_not_label_adjacency(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import torch
    from normal_training_tracking import observe_generation

    from compare_attention.data.tracking import RulerConfig, oracle

    data = inputs(tmp_path)[-1]
    config = RulerConfig.model_validate_json((data / "meta.json").read_bytes())
    x, y = (np.load(data / f"val_{kind}.npy") for kind in ("inputs", "labels"))
    wrong = []
    for prompt in x[:, : config.prompt_length]:
        truth, details = oracle(prompt, config)
        edges = details["edges"]
        start = next(
            edges[node] for node in edges if node not in edges.values() and edges[node] != truth[0]
        )
        chain = [start]
        for _ in range(4):
            chain.append(edges[chain[-1]])
        wrong.append(chain)

    class WrongChain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(attention="classic", vocab_size=116, block_size=128)

        def forward(self, prefix):
            logits = torch.zeros((*prefix.shape, 116))
            position = prefix.shape[1] - config.prompt_length
            for row, chain in enumerate(wrong):
                logits[row, -1, chain[position]] = 1
            return logits

    metrics, records = observe_generation(WrongChain(), x, y, config)
    assert metrics["generated_chain_exact_match"] == 0
    assert metrics["generated_correct_positions"] == 0
    assert metrics["generated_links_correct"] == metrics["generated_links_total"] == 12
    assert metrics["generated_valid_variable_predictions"] == 15
    assert [row["prediction"] for row in records] == wrong
