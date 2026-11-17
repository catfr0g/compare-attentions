"""Public training/config/cost compatibility for the character-LM adapter."""

import hashlib
import importlib
import json
import math

import pytest

from compare_attention.cli.train import parse_train_args
from compare_attention.models.config import GPT2Config
from compare_attention.train.config import Schedule, TrainConfig
from compare_attention.train.flops import estimate_step


def character_data(tmp_path):
    tiny = importlib.import_module("compare_attention.data.tiny_shakespeare")
    raw = b"ab\nc" * 100
    source = tmp_path / "raw.txt"
    source.write_bytes(raw)
    data = tmp_path / "data"
    tiny.prepare_tiny_shakespeare(
        source, data, context=8, expected_sha256=hashlib.sha256(raw).hexdigest()
    )
    return data


def test_real_train_config_cost_and_same_position_scoring(tmp_path, monkeypatch):
    import torch  # pylint: disable=import-outside-toplevel

    from compare_attention.cli.train import main  # pylint: disable=import-outside-toplevel
    from compare_attention.evaluation import evaluate_checkpoint  # pylint: disable=import-outside-toplevel
    from compare_attention.train import loop  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    monkeypatch.setattr(loop, "_device", lambda: torch.device("cpu"))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("HF_REPO", raising=False)
    data = character_data(tmp_path)
    config = TrainConfig(
        data=data,
        model=GPT2Config(vocab_size=4, block_size=8, n_layer=1, n_head=2, n_embd=16),
        schedule=Schedule(train_flops=1, batch_size=2, compile=False, checkpoint_every_steps=1),
    )
    cost = estimate_step(config)
    assert cost.total > 0
    assert config.dataset.name == "tiny_shakespeare_char"
    output = tmp_path / "training"
    yaml_path = tmp_path / "classic.yaml"
    payload = config.model_dump(mode="json", exclude={"dataset"})
    payload.update(tracker="disabled", result_dir=str(output))
    yaml_path.write_text(json.dumps(payload))
    parsed = parse_train_args(["--config", str(yaml_path)])
    assert parsed.model == config.model
    assert main(["--config", str(yaml_path)]) == 0
    report = evaluate_checkpoint(
        output / "model.pt", data, device="cpu", dtype="float32", batch_size=2
    )
    assert report["task"] == "character_language_modeling"
    assert report["supervised_count"] == 19
    assert report["bits_per_character"] == pytest.approx(report["loss"] / math.log(2))
    assert list((output / "checkpoints").glob("*.pt"))


@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_production_shape_cost_uses_character_metadata(tmp_path, attention):
    data = character_data(tmp_path)
    config = TrainConfig(
        data=data,
        model=GPT2Config(
            vocab_size=4, block_size=512, n_layer=4, n_head=4, n_embd=128, attention=attention
        ),
        schedule=Schedule(train_flops=10**16, batch_size=32, dtype="float16", amp_max_scale=128),
    )
    assert estimate_step(config).metadata["shape"]["vocab_size"] == 4
