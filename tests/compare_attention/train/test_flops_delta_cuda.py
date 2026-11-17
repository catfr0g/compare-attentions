"""Local CUDA evidence for the supported real FLA 0.6.0 branches."""

# Lazy FLA imports let CPU-only test collection skip CUDA cleanly.
# pylint: disable=too-many-locals,import-outside-toplevel

import pytest
import torch

from compare_attention.train.flops_delta import delta_attention_cost
from tests.compare_attention.train.delta_reference import Ledger, chunk_reference, recurrence

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + FLA")


def _assert_relative_error(actual, expected, limit=0.04):
    actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
    assert torch.isfinite(actual).all()
    relative_rms = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rms < limit, f"relative RMS error {relative_rms.item()} >= {limit}"


@pytest.mark.parametrize("length", [17, 64, 128])
def test_real_fla_cuda_outputs_and_gradients_match_reference(length):
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

    torch.manual_seed(919)
    inputs = [torch.randn(1, length, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    inputs.append(torch.rand(1, length, 1, device="cuda", dtype=torch.bfloat16))
    inputs = [value.requires_grad_() for value in inputs]
    function = fused_recurrent_delta_rule if length <= 64 else chunk_delta_rule
    output, _ = function(*inputs, use_qk_l2norm_in_kernel=True)
    do = torch.randn_like(output)
    gradients = torch.autograd.grad(output, inputs, do)
    reference_inputs = [
        value.detach().double().cpu().reshape(length, -1).requires_grad_() for value in inputs
    ]
    q, k, v, beta = reference_inputs
    q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    if length <= 64:
        reference = recurrence(q, k, v, beta)
    else:
        reference, _ = chunk_reference(q, k, v, beta, Ledger())
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, do.double().cpu().reshape(length, -1)
    )
    _assert_relative_error(output.reshape(length, -1), reference)
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        _assert_relative_error(actual.reshape(length, -1), expected)


def test_production_default_adapter_cuda_finite_forward_and_backward():
    from fla.layers.delta_net import DeltaNet

    torch.manual_seed(6767)
    layer = DeltaNet(d_model=768, num_heads=12).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(8, 512, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output, _, _ = layer(inputs)
    output.float().square().mean().backward()
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert all(
        conv.backend == "triton" for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)
    )
    # Estimation itself has no CUDA/model-execution dependency.
    assert sum(row["backward"] for row in delta_attention_cost(8, 512, 768, 12).values()) > 0
