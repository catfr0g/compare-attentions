"""Pair matrix must change attention only and preserve approved controls."""

from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment, parse_train_args
from compare_attention.train.config import TrainConfig
from compare_attention.train.flops import estimate_step

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "experiments/tiny_shakespeare/v1"


def test_exact_pair_matrix_and_real_parser():
    files = sorted(MATRIX.glob("*/seed_42.yaml"))
    assert len(files) == 2
    payloads = []
    for file in files:
        raw = yaml.safe_load(file.read_text())
        family = raw["model"].pop("attention")
        assert family == file.parent.name
        assert raw["dataset_task"] == "tiny_shakespeare_char"
        assert raw["schedule"]["batch_size"] == 32
        assert raw["schedule"]["seed"] == 42
        assert raw["schedule"]["train_flops"] == 10**16
        assert raw["schedule"]["early_stopping_patience"] == 0
        assert raw["schedule"]["amp_max_scale"] == 128
        assert raw["optim"]["learning_rate"] == 0.001
        assert raw["optim"]["lr_schedule"] == "constant"
        assert raw["tracker"] == "clearml"
        assert load_experiment(file) == parse_train_args(["--config", str(file)]).model_copy(
            update={"config": None}
        )
        payloads.append(raw)
    assert payloads[0] == payloads[1]


def test_production_contract_from_parsed_configs(tmp_path):
    import hashlib  # pylint: disable=import-outside-toplevel

    from compare_attention.data.tiny_shakespeare import prepare_tiny_shakespeare  # pylint: disable=import-outside-toplevel

    source = tmp_path / "input.txt"
    raw = ("".join(chr(i) for i in range(32, 97)) * 512).encode()
    source.write_bytes(raw)
    data = tmp_path / "prepared"
    prepare_tiny_shakespeare(source, data, expected_sha256=hashlib.sha256(raw).hexdigest())
    files = sorted(MATRIX.glob("*/seed_42.yaml"))
    assert len(files) == 2
    for file in files:
        args = load_experiment(file)
        config = TrainConfig(data=data, model=args.model, schedule=args.schedule, optim=args.optim)
        assert config.dataset.vocab_size == 65
        assert config.dataset.input_seq_len == 512
        assert estimate_step(config).total > 0
