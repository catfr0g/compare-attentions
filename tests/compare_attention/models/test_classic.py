"""Classic attention keeps MHA parameters but executes direct native SDPA."""

from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from compare_attention.models.classic import ClassicAttention
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.identity import parameter_fingerprint
from tests.compare_attention.models.factories import gpt2_config


def test_forward_bypasses_multiheadattention_wrapper() -> None:
    model = ClassicAttention(gpt2_config(n_embd=32, n_head=4))
    values = torch.randn(2, 16, 32)
    with patch.object(model.layer, "forward", side_effect=AssertionError("MHA wrapper called")):
        output = model(values)
    assert output.shape == values.shape
    assert torch.isfinite(output).all()


def _legacy_forward(self: ClassicAttention, x: Tensor) -> Tensor:
    causal = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
    hidden, _ = self.layer(x, x, x, attn_mask=causal, need_weights=False, is_causal=True)
    return self.resid_dropout(hidden)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_native_mha_outputs_and_all_parameter_vjps(bias: bool, training: bool) -> None:
    torch.manual_seed(42)
    model = ClassicAttention(gpt2_config(n_embd=32, n_head=4, bias=bias)).train(training)
    values = torch.randn(2, 16, 32, requires_grad=True)
    targets = [values, *model.parameters()]
    expected = _legacy_forward(model, values)
    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, targets, grad_outputs=cotangent)
    actual = model(values)
    actual_gradients = torch.autograd.grad(actual, targets, grad_outputs=cotangent)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6)
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=3e-6)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_model_fingerprint_ties_logits_and_vjps_preserved() -> None:
    # pylint: disable=too-many-locals
    torch.manual_seed(42)
    model = GPT2(gpt2_config(n_embd=32, n_head=4, vocab_size=64))
    fingerprint = parameter_fingerprint(model)
    names = [name for name, _ in model.named_parameters()]
    parameters = list(model.parameters())
    tokens = torch.randint(0, 64, (2, 16))
    with patch.object(ClassicAttention, "forward", _legacy_forward):
        expected = model(tokens)
        expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
    actual = model(tokens)
    actual_gradients = torch.autograd.grad(actual.square().mean(), parameters)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6)
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=3e-6)
    assert parameter_fingerprint(model) == fingerprint
    assert [name for name, _ in model.named_parameters()] == names
    assert model.lm_head.weight is model.wte.weight
    assert all(parameter.dtype == torch.float32 for parameter in parameters)


def test_future_tokens_do_not_change_earlier_outputs() -> None:
    model = ClassicAttention(gpt2_config(n_embd=32, n_head=4)).eval()
    values = torch.randn(2, 16, 32)
    changed = values.clone()
    changed[:, 8:] += 5.0
    with torch.no_grad():
        first, second = model(values), model(changed)
    torch.testing.assert_close(first[:, :8], second[:, :8], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("training", [False, True])
def test_attention_dropout_obeys_training_mode(training: bool) -> None:
    model = ClassicAttention(gpt2_config(n_embd=32, n_head=4, dropout=0.25)).train(training)
    original = torch.nn.functional.scaled_dot_product_attention
    with patch(
        "compare_attention.models.classic.F.scaled_dot_product_attention", wraps=original
    ) as op:
        model(torch.randn(2, 16, 32))
    assert op.call_args.kwargs["dropout_p"] == (0.25 if training else 0.0)
    assert op.call_args.kwargs["is_causal"] is True


def test_unsupported_attention_variant_fails_closed() -> None:
    model = ClassicAttention(gpt2_config(n_embd=32, n_head=4))
    model.layer.add_zero_attn = True
    with pytest.raises(ValueError, match="self-attention contract"):
        model(torch.randn(2, 16, 32))
