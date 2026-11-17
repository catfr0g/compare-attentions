"""Independent shape ledgers for the versioned DeltaNet algorithmic metric."""

# Named tensors keep the mathematical VJP comparison reviewable.
# pylint: disable=too-many-locals

import importlib
import importlib.util

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from tests.compare_attention.train.delta_reference import (
    Ledger,
    chunk_backward,
    chunk_reference,
    recurrence,
    recurrent_schedule,
)


def estimator():
    name = "compare_attention.train.flops_delta"
    assert importlib.util.find_spec(name) is not None, "DeltaNet estimator is not implemented"
    return importlib.import_module(name).delta_attention_cost


def test_default_adapter_projections_and_conv_recomputation():
    costs = estimator()(8, 512, 768, 12)
    tokens = 8 * 512
    for name in ("q_projection", "k_projection", "v_projection", "output_projection"):
        assert costs[name] == {
            "forward": 2 * tokens * 768 * 768,
            "backward": 4 * tokens * 768 * 768,
        }
    assert costs["beta_projection"] == {
        "forward": 2 * tokens * 768 * 12,
        "backward": 4 * tokens * 768 * 12,
    }
    # FLA's Triton SiLU backward recomputes the convolution preactivation.
    assert costs["short_convolution"] == {
        "forward": 3 * 2 * tokens * 768 * 4,
        "backward": 3 * 6 * tokens * 768 * 4,
    }
    assert costs["qk_l2norm"] == {"forward": 0, "backward": 0}
    assert costs["output_rmsnorm"] == {"forward": 0, "backward": 0}


@pytest.mark.parametrize("dim", [16, 64])
def test_chunk_count_matches_executed_reference_contractions(dim):
    torch.manual_seed(919)
    q, k, v = [torch.randn(128, dim, dtype=torch.float64) for _ in range(3)]
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.rand(128, 1, dtype=torch.float64)
    inputs = [x.requires_grad_() for x in (q, k, v, beta)]
    ledger = Ledger()
    output, saved = chunk_reference(*inputs, ledger)
    torch.testing.assert_close(output, recurrence(*inputs), atol=1e-11, rtol=1e-10)
    do = torch.randn_like(output)
    expected_grads = torch.autograd.grad(output, inputs, do)
    manual_grads = chunk_backward(saved, do, ledger)
    for actual, expected in zip(manual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    # Three independent sequences, two heads. Per-head width is dim.
    costs = estimator()(3, 128, dim * 2, 2)
    for name, phases in ledger.costs.items():
        assert costs[name] == {phase: 6 * value for phase, value in phases.items()}
    assert costs["chunk_recompute_state"]["forward"] == 0
    assert costs["chunk_wy_grad"]["backward"] > 0


@pytest.mark.parametrize(
    "shape",
    [
        (0, 512, 768, 12),
        (8, 0, 768, 12),
        (8, 512, 767, 12),
        (8, 65, 768, 12),
        (8, 129, 768, 12),
        (8, 512, 768, 0),
        (8, 512, 768, True),
        (8, 512.0, 768, 12),
        (8, 512, 3072, 12),
        (8, 512, 120, 12),
    ],
)
def test_unsupported_shapes_fail_closed(shape):
    with pytest.raises(ValueError, match="unsupported DeltaNet"):
        estimator()(*shape)


@pytest.mark.parametrize("length", [1, 17, 64])
def test_short_sequence_uses_actual_recurrent_two_pass_backward(length):
    torch.manual_seed(42)
    q, k, v = [torch.randn(length, 16, dtype=torch.float64) for _ in range(3)]
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.rand(length, 1, dtype=torch.float64)
    inputs = [x.requires_grad_() for x in (q, k, v, beta)]
    expected = recurrence(*inputs)
    do = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, inputs, do)
    ledger = Ledger()
    actual, grads = recurrent_schedule(*inputs, do, ledger)
    torch.testing.assert_close(actual, expected)
    for actual_grad, target in zip(grads, expected_grad, strict=True):
        torch.testing.assert_close(actual_grad, target, atol=1e-11, rtol=1e-10)
    costs = estimator()(1, length, 16, 1)
    assert not any(key.startswith("chunk_") for key in costs)
    for name, phases in ledger.costs.items():
        assert costs[name] == phases


def test_unknown_fla_release_fails_closed(monkeypatch):
    module = importlib.import_module("compare_attention.train.flops_delta")
    monkeypatch.setattr(module, "version", lambda _name: "0.7.0", raising=False)
    with pytest.raises(ValueError, match="FLA 0.6.0"):
        estimator()(8, 512, 768, 12)


def test_nondefault_conv_backend_fails_closed(monkeypatch):
    monkeypatch.setenv("FLA_CONV_BACKEND", "cuda")
    with pytest.raises(ValueError, match="Triton"):
        estimator()(8, 512, 768, 12)


def test_backend_dispatch_override_fails_closed(monkeypatch):
    monkeypatch.setenv("FLA_TILELANG", "1")
    with pytest.raises(ValueError, match="backend"):
        estimator()(8, 512, 768, 12)


def test_installed_alternate_backend_fails_closed(monkeypatch):
    module = importlib.import_module("compare_attention.train.flops_delta")
    monkeypatch.setattr(
        module, "find_spec", lambda name: object() if name == "tilelang" else None, raising=False
    )
    with pytest.raises(ValueError, match="backend"):
        estimator()(8, 512, 768, 12)


@pytest.mark.parametrize(
    "name,outputs",
    [
        ("q_projection", 32),
        ("k_projection", 32),
        ("v_projection", 32),
        ("output_projection", 32),
        ("beta_projection", 2),
    ],
)
def test_projection_counts_match_native_pytorch_autograd(name, outputs):
    inputs = torch.randn(3, 128, 32, requires_grad=True)
    layer = torch.nn.Linear(32, outputs, bias=False)
    with FlopCounterMode(display=False) as counter:
        layer(inputs).sum().backward()
    costs = estimator()(3, 128, 32, 2)
    assert sum(costs[name].values()) == counter.get_total_flops()


def test_convolution_schedule_matches_explicit_channel_matmuls_plus_recompute():
    inputs = torch.randn(3, 32, 128, requires_grad=True)
    weight = torch.randn(32, 4, requires_grad=True)
    # Explicit left padding gives exactly T outputs. Decompose depthwise conv
    # into channel matmuls: Torch's grouped-convolution backward flop formula
    # does not divide the weight-gradient channel cross-product by groups.
    windows = torch.nn.functional.pad(inputs, (3, 0)).unfold(-1, 4, 1)

    def convolution():
        return torch.stack(
            [windows[:, channel].reshape(-1, 4) @ weight[channel, :, None] for channel in range(32)]
        )

    with FlopCounterMode(display=False) as counter:
        torch.nn.functional.silu(convolution()).sum().backward()
        with torch.no_grad():
            convolution()
    row = estimator()(3, 128, 32, 2)["short_convolution"]
    assert sum(row.values()) == 3 * counter.get_total_flops()


def test_batch_sequence_and_width_scaling_are_algebraic():
    base = estimator()(8, 512, 768, 12)
    for shape in [(16, 512, 768, 12), (8, 1024, 768, 12)]:
        doubled = estimator()(*shape)
        assert doubled == {
            key: {phase: 2 * value for phase, value in row.items()} for key, row in base.items()
        }
    width = estimator()(8, 512, 1536, 12)
    assert width["q_projection"]["forward"] == 4 * base["q_projection"]["forward"]
    assert width["chunk_state"]["forward"] == 4 * base["chunk_state"]["forward"]
    assert width["chunk_gram"]["forward"] == 2 * base["chunk_gram"]["forward"]
    assert width["chunk_inverse"] == base["chunk_inverse"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_metric_is_dtype_compile_seed_independent_and_does_not_consume_rng(dtype, compiled):
    baseline = estimator()(8, 512, 768, 12)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        # Torch's config patch is a dynamic module API not visible to pylint.
        # pylint: disable=protected-access,no-member
        with torch._dynamo.config.patch(disable=not compiled):
            for seed in (42, 6767, 696969):
                torch.manual_seed(seed)
                before = torch.get_rng_state().clone()
                assert estimator()(8, 512, 768, 12) == baseline
                assert torch.equal(torch.get_rng_state(), before)
    finally:
        torch.set_default_dtype(original)
