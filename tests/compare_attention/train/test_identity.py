"""Hashing exact parameters is read-only, including BF16 and tied weights."""

from pathlib import Path

import pytest
import torch

from compare_attention.models.config import AttentionName, GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.identity import (
    parameter_fingerprint,
    runtime_metadata,
    verify_fp16_checkpoint,
)


@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_checkpoint_canonical_hash_and_cpu_rng(
    tmp_path: Path, attention: AttentionName, monkeypatch
) -> None:
    config = GPT2Config(
        vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8, attention=attention
    )
    model = GPT2(config)
    identity = {
        "config": {"model": config.model_dump(mode="json")},
        "initial_parameters_sha256": "a" * 64,
        "final_parameters_sha256": parameter_fingerprint(model),
    }
    torch.save(model.state_dict(), tmp_path / "model.pt")
    before_rng = torch.get_rng_state().clone()

    def no_cuda(*_args, **_kwargs):
        raise AssertionError("checkpoint validation must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    monkeypatch.setattr(torch.cuda, "device_count", no_cuda)
    assert verify_fp16_checkpoint(tmp_path, identity) == parameter_fingerprint(model)
    assert torch.equal(before_rng, torch.get_rng_state())
    identity["initial_parameters_sha256"] = identity["final_parameters_sha256"]
    with pytest.raises(ValueError, match="did not update"):
        verify_fp16_checkpoint(tmp_path, identity)
    assert torch.equal(before_rng, torch.get_rng_state())


def test_checkpoint_aliases_must_be_bitwise_equal(tmp_path: Path) -> None:
    config = GPT2Config(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8)
    model = GPT2(config)
    with torch.no_grad():
        model.wte.weight.zero_()
    identity = {
        "config": {"model": config.model_dump(mode="json")},
        "initial_parameters_sha256": "a" * 64,
        "final_parameters_sha256": parameter_fingerprint(model),
    }
    state = model.state_dict()
    state["lm_head.weight"] = -state["wte.weight"].clone()
    torch.save(state, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="tied aliases"):
        verify_fp16_checkpoint(tmp_path, identity)


def test_fingerprint_is_value_sensitive_and_preserves_training_state() -> None:
    model = torch.nn.Linear(4, 3).to(dtype=torch.bfloat16)
    model(torch.ones(2, 4, dtype=torch.bfloat16)).sum().backward()
    before_rng = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    before_weights = {name: value.clone() for name, value in model.state_dict().items()}
    before_gradients = [parameter.grad.clone() for parameter in model.parameters()]
    digest = parameter_fingerprint(model)
    runtime_metadata()
    assert digest == parameter_fingerprint(model)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_cuda, torch.cuda.get_rng_state_all(), strict=True)
    )
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before_weights.items()
    )
    assert all(
        torch.equal(gradient, parameter.grad)
        for gradient, parameter in zip(before_gradients, model.parameters(), strict=True)
    )
    with torch.no_grad():
        model.weight[0, 0].add_(1)
    assert digest != parameter_fingerprint(model)
