"""Public permutation CLI uses dense labels and prices actual length, not context cap."""

# pylint: disable=import-outside-toplevel
import pytest

from compare_attention.cli.train import parse_train_args
from compare_attention.train.config import TrainConfig
from compare_attention.train.flops import estimate_step


def test_prepare_parser_and_actual_length_pricing(tmp_path):
    from compare_attention.cli.prepare_permutation import main

    data = tmp_path / "c5"
    assert (
        main(
            [
                "--output",
                str(data),
                "--group",
                "c5",
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
            str(data),
            "--dataset-task",
            "permutation",
            "--train-flops",
            "100000000000000",
            "--vocab-size",
            "121",
            "--block-size",
            "512",
            "--n-layer",
            "4",
            "--n-head",
            "4",
            "--n-embd",
            "128",
            "--batch-size",
            "32",
            "--seed",
            "42",
            "--dtype",
            "float16",
        ]
    )
    payload = args.model_dump()
    for key in ("config", "kv", "dataset_task", "result_dir", "resume_from", "tracker", "hf_repo"):
        payload.pop(key, None)
    config = TrainConfig(**payload)
    assert config.model.block_size == 512
    assert config.dataset.input_seq_len == 32
    assert config.dataset.allowed_token_ids == [1, 34, 65, 91, 97]
    cost = estimate_step(config)
    assert cost.metadata["shape"]["sequence_length"] == 32
    assert cost.total > 0
    alternate = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"input_seq_len": 512})}
    )
    assert estimate_step(alternate).total > cost.total * 16
    with pytest.raises(ValueError, match="block_size"):
        TrainConfig(**(payload | {"model": args.model.model_copy(update={"block_size": 16})}))


def test_candidate_matrix_complete_and_matched():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "experiments/permutation/v1").glob("*/*/seed_42.yaml"))
    assert len(paths) == 4
    pairs = {}
    for path in paths:
        args = parse_train_args(["--config", str(path)])
        assert args.dataset_task == "permutation"
        assert args.model.block_size == 512
        assert args.model.vocab_size == 121
        assert (args.model.n_layer, args.model.n_embd, args.model.n_head) == (4, 128, 4)
        assert args.schedule.batch_size == 32
        assert args.schedule.seed == 42
        assert args.schedule.dtype == "float16"
        assert args.schedule.compile
        assert args.schedule.amp_max_scale == 128
        assert args.schedule.early_stopping_patience == 0
        assert args.optim.learning_rate == 0.001
        assert args.optim.lr_schedule == "constant"
        assert args.optim.weight_decay == 0.1
        raw = args.model_dump(mode="json", exclude={"config"})
        raw["model"].pop("attention")
        pairs.setdefault(path.parents[1].name, []).append(raw)
    assert pairs.keys() == {"c5", "s5"}
    assert all(values[0] == values[1] for values in pairs.values())
