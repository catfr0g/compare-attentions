"""Public sorting preparation, trainer accounting and primary evaluator integration."""

import importlib.util

import numpy as np


def test_public_sorting_prepare_train_and_evaluation(tmp_path):
    assert importlib.util.find_spec("compare_attention.cli.prepare_sorting") is not None
    from compare_attention.cli.prepare_sorting import main
    from compare_attention.cli.train import _ensure_dataset, parse_train_args
    from compare_attention.evaluation import evaluate_model
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step
    from tests.compare_attention.test_sorting_evaluation import EOSModel

    data = tmp_path / "data"
    assert main(["--output", str(data), "--n-train", "32", "--n-val", "1", "--n-test", "3"]) == 0
    args = parse_train_args(
        [
            "--data",
            str(data),
            "--dataset-task",
            "sorting",
            "--train-flops",
            "1000000000000000",
            "--batch-size",
            "32",
            "--vocab-size",
            "8",
            "--n-layer",
            "4",
            "--n-head",
            "4",
            "--n-embd",
            "128",
            "--dtype",
            "float16",
        ]
    )
    _ensure_dataset(args)
    c = TrainConfig(**{k: getattr(args, k) for k in ("data", "schedule", "optim", "model")})
    assert estimate_step(c).metadata["shape"]["sequence_length"] == 128
    x, y = (np.load(data / f"test_{k}.npy") for k in ("inputs", "labels"))
    result = evaluate_model(EOSModel(), x, y, task_config=c.dataset, batch_size=2)
    assert result["primary_metrics"] == ["generated_answer_exact_match"]
    assert result["teacher_forced_supervised_count"] == 120
    assert len(result["generated_records"]) == 3


def test_actual_cpu_two_update_public_main(tmp_path, monkeypatch):
    import json

    import torch

    from compare_attention.cli import train as train_cli
    from compare_attention.cli.evaluate import main as eval_main
    from compare_attention.cli.prepare_sorting import main as prepare_main
    from compare_attention.train.config import TrainConfig
    from compare_attention.train.flops import estimate_step

    assert not torch.cuda.is_available()
    monkeypatch.setattr(train_cli, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.delenv("HF_REPO", raising=False)
    data, trained, output = tmp_path / "data", tmp_path / "trained", tmp_path / "test.json"
    assert (
        prepare_main(["--output", str(data), "--n-train", "4", "--n-val", "1", "--n-test", "3"])
        == 0
    )
    args = [
        "--data",
        str(data),
        "--dataset-task",
        "sorting",
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
        "8",
        "--n-layer",
        "1",
        "--n-head",
        "1",
        "--n-embd",
        "8",
        "--result-dir",
        str(trained),
    ]
    parsed = train_cli.parse_train_args(args)
    config = TrainConfig(**{k: getattr(parsed, k) for k in ("data", "schedule", "optim", "model")})
    step = estimate_step(config).total
    args[args.index("--train-flops") + 1] = str(2 * step)
    assert train_cli.main(args) == 0
    assert (
        eval_main(
            [
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
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["generated_rows"] == 3 and report["teacher_forced_supervised_count"] == 120
    assert report["data"]["config"]["input_seq_len"] == 128
    assert len(report["generated_records"]) == 3
    identity = json.loads((trained / "run_identity.json").read_text())
    assert identity["status"] == "completed"


def test_delta_non64_tail_still_rejected():
    import pytest

    from compare_attention.train.flops_delta import delta_attention_cost

    with pytest.raises(ValueError, match="chunk shape"):
        delta_attention_cost(32, 79, 128, 4)
