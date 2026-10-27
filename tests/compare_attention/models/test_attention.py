# pylint: disable=redefined-outer-name
"""Оба механизма внимания занимают один и тот же слот."""

import pytest
import torch

from compare_attention.models.attention import build_attention
from compare_attention.models.classic import ClassicAttention
from compare_attention.models.deltanet import DeltaNet
from tests.compare_attention.models.factories import gpt2_config

_KINDS = ("classic", "deltanet")


def _on_device(module: torch.nn.Module) -> torch.nn.Module:
    if not isinstance(module, DeltaNet):
        return module
    if not torch.cuda.is_available():
        pytest.skip("ядра DeltaNet требуют CUDA")
    return module.cuda()


@pytest.fixture(params=_KINDS)
def attention(request: pytest.FixtureRequest) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_attention(gpt2_config(attention=request.param)).eval()


@pytest.mark.parametrize(
    ("kind", "cls"),
    [("classic", ClassicAttention), ("deltanet", DeltaNet)],
)
def test_config_selects_the_named_mechanism(kind: str, cls: type[torch.nn.Module]) -> None:
    module = build_attention(gpt2_config(attention=kind))

    assert isinstance(module, cls)


def test_attention_preserves_shape(attention: torch.nn.Module) -> None:
    attention = _on_device(attention)
    x = torch.randn(2, 6, 8, device=next(attention.parameters()).device)

    y = attention(x)

    assert y.shape == x.shape


def test_future_tokens_do_not_change_past_outputs(attention: torch.nn.Module) -> None:
    torch.manual_seed(0)
    attention = _on_device(attention)
    x = torch.randn(2, 6, 8, device=next(attention.parameters()).device)
    changed = x.clone()
    changed[:, -1] += 1

    past = attention(x)[:, :-1]
    shifted = attention(changed)[:, :-1]

    assert torch.allclose(past, shifted, atol=1e-5)


def test_past_token_changes_the_current_output(attention: torch.nn.Module) -> None:
    torch.manual_seed(0)
    attention = _on_device(attention)
    x = torch.randn(2, 6, 8, device=next(attention.parameters()).device)
    changed = x.clone()
    changed[:, 0] += 1

    current = attention(x)[:, -1]
    other = attention(changed)[:, -1]

    assert not torch.allclose(current, other)
