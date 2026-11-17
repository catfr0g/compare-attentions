"""Public task preparation, parser, pricing and count evaluation contracts."""

# Public-entrypoint imports are deferred to preserve the missing-feature RED proof.
# pylint: disable=import-outside-toplevel

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compare_attention.cli.train import parse_train_args
from compare_attention.data.selective_count import prepare_selective_count
from compare_attention.evaluation import evaluate_model
from compare_attention.train.config import TrainConfig
from compare_attention.train.flops import estimate_step


def test_public_prepare_parser_and_pricing(tmp_path):
    from compare_attention.cli.prepare_selective_count import main
    from compare_attention.data.store import load_config

    path = tmp_path / "count"
    assert (
        main(
            [
                "--output",
                str(path),
                "--context",
                "8",
                "--n-train",
                "32",
                "--n-val",
                "3",
                "--n-test",
                "5",
            ]
        )
        == 0
    )
    args = parse_train_args(
        [
            "--data",
            str(path),
            "--dataset-task",
            "selective_count",
            "--train-flops",
            "1000",
            "--vocab-size",
            "42",
            "--batch-size",
            "32",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "8",
            "--no-compile",
        ]
    )
    payload = args.model_dump()
    for key in ("config", "kv", "dataset_task", "result_dir", "resume_from", "tracker", "hf_repo"):
        payload.pop(key, None)
    config = TrainConfig(**payload)
    assert config.dataset == load_config(path)
    assert estimate_step(config).total > 0


def test_full_argmax_clipping_and_tail(tmp_path):
    config = prepare_selective_count(tmp_path / "data", context=4, n_train=32, n_val=2, n_test=3)
    x = np.array([[1, 2, 2, 2], [2, 1, 1, 1], [1, 2, 1, 2]], dtype=np.int64)
    y = np.full_like(x, -1)
    y[:, -1] = [36, 36, 35]

    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(attention="classic", block_size=4, vocab_size=38)

        def forward(self, inputs):
            logits = torch.zeros((*inputs.shape, 38))
            predictions = torch.where(inputs[:, 0] == 1, 0, 37)
            logits[torch.arange(len(inputs)), -1, predictions] = torch.where(
                inputs[:, 0] == 1, 10.0, 1.0
            )
            return logits

    metrics = evaluate_model(Fixed(), x, y, batch_size=2, task_config=config)
    assert metrics["correct"] == 0
    assert metrics["supervised_count"] == 3
    # raw decoded counts -33,4,-33; clipped 0,4,0 vs true 3,3,2
    assert metrics["count_mae"] == 2
    assert metrics["count_nmae"] == 2 / 1.125
    assert metrics["count_raw_mae"] == (36 + 1 + 35) / 3
    assert metrics["count_outside_class_range"] == 2
    assert "bits_per_character" not in metrics
    expected_loss = (
        torch.nn.functional.cross_entropy(
            Fixed()(torch.from_numpy(x)).flatten(0, 1),
            torch.from_numpy(y).flatten(),
            ignore_index=-1,
            reduction="sum",
        )
        / 3
    )
    assert metrics["loss"] == pytest.approx(float(expected_loss))


def test_delta_cpu_is_explicitly_unsupported(tmp_path):
    from compare_attention.data.store import load_split
    from compare_attention.models.config import GPT2Config
    from compare_attention.models.gpt2 import GPT2

    data = tmp_path / "count"
    config = prepare_selective_count(data, context=8, n_train=32, n_val=3, n_test=5)
    arrays = load_split(data, "test")
    model = GPT2(
        GPT2Config(
            vocab_size=config.vocab_size,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            attention="deltanet",
        )
    )
    with pytest.raises(ValueError, match="DeltaNet requires CUDA"):
        evaluate_model(model, arrays.inputs, arrays.labels, task_config=config)


def test_frozen_candidate_matrix():
    from pathlib import Path

    from compare_attention.cli.train import load_experiment

    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "experiments/selective_count/v1").glob("*/seed_42.yaml"))
    assert len(paths) == 2
    payloads = []
    for path in paths:
        args = parse_train_args(["--config", str(path)])
        assert args == load_experiment(path).model_copy(update={"config": path})
        assert args.dataset_task == "selective_count"
        assert args.model.vocab_size == 546
        assert args.model.block_size == 512
        assert (args.model.n_layer, args.model.n_embd, args.model.n_head) == (4, 128, 4)
        assert args.schedule.batch_size == 32
        assert args.schedule.seed == 42
        assert args.schedule.dtype == "float16"
        assert args.schedule.compile
        assert args.schedule.amp_max_scale == 128
        assert args.schedule.early_stopping_patience == 0
        assert args.optim.learning_rate == 0.001
        assert args.optim.weight_decay == 0.1
        assert args.optim.lr_schedule == "constant"
        raw = args.model_dump(mode="json")
        raw.pop("config")
        raw["model"].pop("attention")
        payloads.append(raw)
    assert payloads[0] == payloads[1]
