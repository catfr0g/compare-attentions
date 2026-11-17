"""Contract tests for the versioned algorithmic cost, without a training probe."""

import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from compare_attention.data.config import KVRetrievalConfig
from compare_attention.models.config import GPT2Config
from compare_attention.train.config import ComparisonAxis, Schedule, TrainConfig


def cost_config(**model_overrides: object) -> TrainConfig:
    """Validated shapes; construct only skips the dataset filesystem lookup."""
    model = GPT2Config(
        **{
            "vocab_size": 17,
            "block_size": 8,
            "n_embd": 4,
            "n_head": 2,
            "n_layer": 1,
            **model_overrides,
        }
    )
    dataset = KVRetrievalConfig(vocab_size=model.vocab_size, input_seq_len=4, num_kv_pairs=1)
    return TrainConfig.model_construct(
        data=Path("unused"),
        model=model,
        dataset=dataset,
        schedule=Schedule(train_flops=1, batch_size=2, compile=False),
    )


def test_classic_step_has_explicit_forward_backward_breakdown() -> None:
    assert importlib.util.find_spec("compare_attention.train.flops") is not None
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    cost = estimate_step(cost_config())
    assert cost.policy == "algorithmic_matmul_conv_v2"
    assert cost.breakdown == {
        "attention.qkv": {"forward": 768, "backward": 1536},
        "attention.output": {"forward": 256, "backward": 512},
        "attention.qk": {"forward": 256, "backward": 512},
        "attention.av": {"forward": 256, "backward": 512},
        "mlp.expand": {"forward": 1024, "backward": 2048},
        "mlp.contract": {"forward": 1024, "backward": 2048},
        "lm_head": {"forward": 1088, "backward": 2176},
    }
    assert cost.forward == sum(term["forward"] for term in cost.breakdown.values())
    assert cost.backward == sum(term["backward"] for term in cost.breakdown.values())
    assert cost.total == cost.forward + cost.backward


def test_json_contract_is_detached_and_records_conventions() -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    cost = estimate_step(cost_config())
    encoded = cost.to_dict()
    assert json.loads(json.dumps(encoded)) == encoded
    assert set(encoded) == {"policy", "forward", "backward", "total", "breakdown", "metadata"}
    assert encoded["metadata"]["schema_version"] == 1
    assert encoded["metadata"]["shape"] == {
        "batch_size": 2,
        "sequence_length": 4,
        "hidden_size": 4,
        "num_heads": 2,
        "num_layers": 1,
        "vocab_size": 17,
    }
    assert encoded["metadata"]["causal_convention"] == "dense_square"
    assert encoded["metadata"]["padding_convention"] == "logical_tokens_no_kernel_padding"
    assert encoded["metadata"]["mac_flops"] == 2
    assert encoded["metadata"]["attention"] == "classic"
    assert encoded["metadata"]["recompute"] == "none"
    assert "optimizer" in encoded["metadata"]["excluded"]
    encoded["breakdown"]["lm_head"]["forward"] = -1
    assert cost.breakdown["lm_head"]["forward"] > 0
    with pytest.raises(FrozenInstanceError):
        cost.total = 0


@pytest.mark.parametrize(
    "section,updates",
    [
        ("model", {"attention": "unknown"}),
        ("model", {"n_embd": 5}),
        ("model", {"n_layer": 0}),
        ("model", {"block_size": 2}),
        ("model", {"vocab_size": 19}),
        ("model", {"dropout": float("nan")}),
        ("schedule", {"batch_size": 0}),
        ("schedule", {"batch_size": 100_001}),
        ("schedule", {"axis": ComparisonAxis.PARAMETERS}),
        ("schedule", {"dtype": "float64"}),
        ("dataset", {"input_seq_len": 0}),
    ],
)
def test_invalid_or_unsupported_configuration_fails_closed(section, updates) -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    config = cost_config()
    invalid = config.model_copy(
        update={section: getattr(config, section).model_copy(update=updates)}
    )
    with pytest.raises((ValueError, NotImplementedError)):
        estimate_step(invalid)


def test_delta_attention_is_delegated_once_per_layer_not_tripled() -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel
    from compare_attention.train.flops_delta import delta_attention_cost  # pylint: disable=import-outside-toplevel

    config = cost_config(attention="deltanet", n_layer=2, n_embd=32)
    cost = estimate_step(config)
    attention = delta_attention_cost(2, 4, 32, 2)
    expected = {
        f"attention.{name}": {key: 2 * value for key, value in term.items()}
        for name, term in attention.items()
    }
    actual = {name: term for name, term in cost.breakdown.items() if name.startswith("attention.")}
    assert actual == expected
    assert cost.backward == sum(term["backward"] for term in cost.breakdown.values())
    assert cost.metadata["recompute"] == "fla_default_backward_recompute"
    assert cost.metadata["causal_convention"] == "fla_causal_recurrence_or_chunk64"
    assert cost.metadata["architecture"] == "repository_gpt2_v1"
    assert cost.metadata["included"] == ["dense_matmul", "convolution", "delta_core_contractions"]


def test_cost_is_pure_and_independent_of_training_runtime(monkeypatch) -> None:
    from compare_attention.models.gpt2 import GPT2  # pylint: disable=import-outside-toplevel
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cost estimation must not execute a model/optimizer")

    monkeypatch.setattr(GPT2, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(torch, "randn", forbidden)
    config = cost_config()
    rng = torch.random.get_rng_state().clone()
    cost = estimate_step(config)
    other = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"block_size": 128, "bias": False, "dropout": 0.5}
            ),
            "schedule": config.schedule.model_copy(update={"compile": True, "dtype": "bfloat16"}),
        }
    )
    assert estimate_step(other) == cost
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("width,heads,layers,vocab", [(4, 1, 1, 17), (6, 3, 2, 23)])
def test_classic_matches_independent_dense_reference(width, heads, layers, vocab) -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    config = cost_config(n_embd=width, n_head=heads, n_layer=layers, vocab_size=vocab)

    # Independently enumerate every scalar MAC of each matrix product, not the production helpers.
    def mm(left_rows, inner, right_columns):
        return sum(2 for _ in range(left_rows) for _ in range(inner) for _ in range(right_columns))

    per_layer = 3 * mm(8, width, width) + mm(8, width, width)
    per_layer += 2 * heads * (mm(4, width // heads, 4) + mm(4, 4, width // heads))
    per_layer += mm(8, width, 4 * width) + mm(8, 4 * width, width)
    expected = layers * per_layer + mm(8, width, vocab)
    cost = estimate_step(config)
    assert cost.forward == expected
    # Every included classic matmul has two trainable operands. Bias sums and embedding
    # scatter-adds do not turn into invented GEMMs even though the head weight is tied.
    assert cost.backward == 2 * expected


@pytest.mark.parametrize("bias", [False, True])
def test_complete_classic_gpt_matches_registered_pytorch_counter(bias) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel  # pylint: disable=import-outside-toplevel
    from torch.utils.flop_counter import FlopCounterMode  # pylint: disable=import-outside-toplevel

    from compare_attention.models.gpt2 import GPT2  # pylint: disable=import-outside-toplevel
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    config = cost_config(bias=bias, n_layer=2)
    with torch.random.fork_rng(devices=[]):
        model = GPT2(config.model).train()
        indices = torch.arange(8).reshape(2, 4)
        with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
            output = model(indices)
            forward = counter.get_total_flops()
            output.sum().backward()
        total = counter.get_total_flops()
    cost = estimate_step(config)
    assert forward == cost.forward
    assert total - forward == cost.backward
    assert total == cost.total


def test_production_classic_bounds_and_exact_total() -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    config = TrainConfig.model_construct(
        data=Path("unused"),
        model=GPT2Config(),
        dataset=KVRetrievalConfig(),
        schedule=Schedule(train_flops=1, batch_size=8, dtype="bfloat16"),
    )
    cost = estimate_step(config)
    expected_forward = 12 * (24 * 8 * 512 * 768**2 + 4 * 8 * 512**2 * 768)
    expected_forward += 2 * 8 * 512 * 768 * 8192
    assert cost.forward == expected_forward
    assert cost.backward == 2 * expected_forward
    assert cost.total == 3 * expected_forward
    assert 2_000_000_000_000 < cost.total < 3_000_000_000_000


def test_delegated_shortconv_matches_registered_counter_including_recompute() -> None:
    from torch.utils.flop_counter import FlopCounterMode  # pylint: disable=import-outside-toplevel

    from compare_attention.train.flops_delta import delta_attention_cost  # pylint: disable=import-outside-toplevel

    # One depthwise causal conv: explicitly left-padded input, T logical outputs.
    # Dense MAC convention counts boundary multiplies against zeros, not an optimized kernel.
    # Count one channel without groups, then sum 32 independent channels × 3 streams.
    # PyTorch's registered grouped weight-gradient formula ignores _groups; its
    # aggregate depthwise backward is not a valid oracle for this comparison.
    inputs = torch.nn.functional.pad(torch.ones(2, 1, 4), (3, 0)).requires_grad_()
    weight = torch.ones(1, 1, 4, requires_grad=True)
    with FlopCounterMode(display=False) as counter:
        output = torch.nn.functional.conv1d(inputs, weight)
        forward = counter.get_total_flops()
        output.sum().backward()
        torch.nn.functional.conv1d(inputs.detach(), weight.detach())
    costs = delta_attention_cost(2, 4, 32, 2)["short_convolution"]
    assert costs["forward"] == 3 * 32 * forward
    assert costs["backward"] == 3 * 32 * (counter.get_total_flops() - forward)


def test_configuration_subclasses_are_not_silently_counted() -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    class UnsupportedGPT(GPT2Config):
        extra_dense_layer: bool = True

    config = cost_config()
    unsupported = UnsupportedGPT(**config.model.model_dump())
    with pytest.raises(NotImplementedError, match="unsupported configuration type"):
        estimate_step(config.model_copy(update={"model": unsupported}))


def test_production_delta_bound_and_recompute_is_not_global_three_x() -> None:
    from compare_attention.train.flops import estimate_step  # pylint: disable=import-outside-toplevel

    config = TrainConfig.model_construct(
        data=Path("unused"),
        model=GPT2Config(attention="deltanet"),
        dataset=KVRetrievalConfig(),
        schedule=Schedule(train_flops=1, batch_size=8, dtype="bfloat16"),
    )
    cost = estimate_step(config)
    assert 2_000_000_000_000 < cost.total < 3_000_000_000_000
    assert cost.total == cost.forward + cost.backward
    assert cost.total != 3 * cost.forward
    assert cost.breakdown["attention.chunk_recompute_wy"]["backward"] > 0
    assert cost.breakdown["attention.chunk_recompute_state"]["backward"] > 0
    assert cost.breakdown["lm_head"] == {
        "forward": 2 * 8 * 512 * 768 * 8192,
        "backward": 4 * 8 * 512 * 768 * 8192,
    }
