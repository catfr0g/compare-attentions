"""Public preparation, trainer parser and pricing integration; CPU collection only."""

import numpy as np
import pytest

from compare_attention.data.tracking import prepare_tracking


@pytest.mark.parametrize("task", ["ruler_variable_tracking", "state_based_recall"])
def test_public_prepare_train_pricing(tmp_path, task):
    from compare_attention.cli.prepare_tracking import main
    from compare_attention.cli.train import _ensure_dataset, parse_train_args
    from compare_attention.data.store import load_config
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step

    path = tmp_path / task
    assert (
        main(
            [
                "--output",
                str(path),
                "--task",
                task,
                "--n-train",
                "35",
                "--n-val",
                "3",
                "--n-test",
                "5",
            ]
        )
        == 0
    )
    config = load_config(path)
    args = parse_train_args(
        [
            "--data",
            str(path),
            "--dataset-task",
            task,
            "--train-flops",
            "100000000000000",
            "--batch-size",
            "32",
            "--dtype",
            "float16",
            "--vocab-size",
            str(config.vocab_size),
            "--n-layer",
            "4",
            "--n-head",
            "4",
            "--n-embd",
            "128",
        ]
    )
    _ensure_dataset(args)
    payload = {key: getattr(args, key) for key in ("data", "schedule", "optim", "model")}
    training = TrainConfig(**payload)
    assert type(training.dataset) is type(config)
    assert estimate_step(training).metadata["shape"]["sequence_length"] == config.input_seq_len
    assert np.load(path / "test_labels.npy").shape[0] == 5
    with pytest.raises(FileExistsError):
        main(["--output", str(path), "--task", task])


def test_declared_wrong_task_rejected(tmp_path):
    from compare_attention.cli.train import _ensure_dataset, parse_train_args

    path = tmp_path / "data"
    prepare_tracking(path, task="state_based_recall", n_train=2, n_val=1, n_test=1)
    args = parse_train_args(
        ["--data", str(path), "--dataset-task", "ruler_variable_tracking", "--train-flops", "1"]
    )
    with pytest.raises(ValueError, match="task"):
        _ensure_dataset(args)


@pytest.mark.parametrize("task", ["ruler_variable_tracking", "state_based_recall"])
def test_real_saved_weights_public_eval_and_snapshot(tmp_path, monkeypatch, task):
    import json

    import torch

    from compare_attention.cli import train as train_cli
    from compare_attention.cli.evaluate import main as eval_main
    from compare_attention.evaluation import evaluate_checkpoint

    monkeypatch.setattr(train_cli, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.delenv("HF_REPO", raising=False)
    path, result = tmp_path / "data", tmp_path / "trained"
    c = prepare_tracking(path, task=task, n_train=2, n_val=1, n_test=3)
    assert (
        train_cli.main(
            [
                "--data",
                str(path),
                "--dataset-task",
                task,
                "--train-flops",
                "1",
                "--batch-size",
                "2",
                "--seed",
                "42",
                "--dtype",
                "float32",
                "--no-compile",
                "--tracker",
                "disabled",
                "--checkpoint-every-steps",
                "1",
                "--vocab-size",
                str(c.vocab_size),
                "--n-layer",
                "1",
                "--n-head",
                "1",
                "--n-embd",
                "8",
                "--result-dir",
                str(result),
            ]
        )
        == 0
    )

    def forbidden(*_a, **_k):
        raise AssertionError("evaluation must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    output = tmp_path / "eval.json"
    assert (
        eval_main(
            [
                "--checkpoint",
                str(result),
                "--data",
                str(path),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--batch-size",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["split"] == "test" and report["generated_rows"] == 3
    assert report["teacher_forced_supervised_count"] == 3 * c.answer_length
    assert report["data"]["config"] == json.loads(c.model_dump_json())
    pointer = result / "checkpoints/latest.json"
    snapshot_report = evaluate_checkpoint(
        pointer, path, device="cpu", dtype="float32", batch_size=2
    )
    assert (
        snapshot_report["checkpoint"]["parameters_sha256"]
        == report["checkpoint"]["parameters_sha256"]
    )
    assert snapshot_report["generated_position_accuracy"] == report["generated_position_accuracy"]


def test_full_proposal_config_matrix_parser():
    import json
    from pathlib import Path

    from compare_attention.cli.train import parse_train_args
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step

    root = Path(__file__).resolve().parents[3]
    report = (
        root / "local/reports/four-task-benchmark/variable-tracking-prep/budget-alternatives.json"
    )
    rows = json.loads(report.read_text())
    assert len(rows) == 12
    for row in rows:
        path = (
            root
            / "local/experiments/variable-tracking-kaggle-v1/configs"
            / row["task"]
            / (f"{row['attention']}-{row['requested_train_flops']}.yaml")
        )
        args = parse_train_args(["--config", str(path)])
        config = TrainConfig(
            **{key: getattr(args, key) for key in ("data", "schedule", "optim", "model")}
        )
        step = estimate_step(config).total
        assert step == row["step_flops"]
        assert (config.schedule.train_flops + step - 1) // step == row["steps"]


def test_selected_budget_parser_and_exact_estimator():
    import json
    from pathlib import Path

    from compare_attention.cli.train import parse_train_args
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step

    root = Path(__file__).resolve().parents[3]
    report = root / "local/reports/four-task-benchmark/variable-tracking-prep"
    selected = json.loads((report / "selected-budget-verified.json").read_text())
    assert len(selected["runs"]) == 4 and not selected["launch_approved"]
    for row in selected["runs"]:
        args = parse_train_args(["--config", str(root / row["config"])])
        c = TrainConfig(**{k: getattr(args, k) for k in ("data", "schedule", "optim", "model")})
        step = estimate_step(c).total
        assert step == row["step_flops"]
        assert (c.schedule.train_flops + step - 1) // step == row["steps"]
        assert c.schedule.early_stopping_patience == 0


@pytest.mark.parametrize("task", ["ruler_variable_tracking", "state_based_recall"])
def test_real_module_main_sanitized_subprocess(tmp_path, task):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    env = {
        k: os.environ[k]
        for k in (
            "SYSTEMROOT",
            "WINDIR",
            "PATH",
            "HOME",
            "USERPROFILE",
            "USERNAME",
            "USER",
            "LOGNAME",
        )
        if k in os.environ
    }
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_DOTENV_DISABLED="1",
        PYTHONPATH=str(root / "src"),
        HF_REPO="",
        CLEARML_OFFLINE_MODE="1",
        TEMP=str(tmp_path),
        TMPDIR=str(tmp_path),
        TMP=str(tmp_path),
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
    )

    def run(module, *args):
        completed = subprocess.run(
            [sys.executable, "-B", "-m", module, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    data, trained, output = tmp_path / "data", tmp_path / "trained", tmp_path / "eval.json"
    run(
        "compare_attention.cli.prepare_tracking",
        "--output",
        str(data),
        "--task",
        task,
        "--n-train",
        "2",
        "--n-val",
        "1",
        "--n-test",
        "3",
    )
    vocab = "116" if task.startswith("ruler") else "43"
    run(
        "compare_attention.cli.train",
        "--data",
        str(data),
        "--dataset-task",
        task,
        "--train-flops",
        "1",
        "--batch-size",
        "2",
        "--dtype",
        "float32",
        "--no-compile",
        "--tracker",
        "disabled",
        "--vocab-size",
        vocab,
        "--n-layer",
        "1",
        "--n-head",
        "1",
        "--n-embd",
        "8",
        "--result-dir",
        str(trained),
    )
    run(
        "compare_attention.cli.evaluate",
        "--checkpoint",
        str(trained),
        "--data",
        str(data),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--batch-size",
        "2",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text())
    assert report["split"] == "test" and report["generated_rows"] == 3
    assert report["device"] == "cpu"
