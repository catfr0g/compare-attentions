"""GPT-2: эмбеддинги, pre-norm блоки и сменное внимание."""

import pytest
import torch

from compare_attention.models.gpt2 import GPT2
from tests.compare_attention.models.factories import gpt2_config

_KINDS = ("classic", "deltanet")
_OUTPUT_PROJECTIONS = (
    "c_proj.weight",
    "c_proj.bias",
    "o_proj.weight",
    "o_proj.bias",
    "out_proj.weight",
    "out_proj.bias",
)


def _on_device(model: GPT2, attention: str) -> GPT2:
    if attention != "deltanet":
        return model
    if not torch.cuda.is_available():
        pytest.skip("ядра DeltaNet требуют CUDA")
    return model.cuda()


@pytest.mark.parametrize("attention", _KINDS)
def test_logits_have_batch_time_and_vocab(attention: str) -> None:
    model = _on_device(GPT2(gpt2_config(attention=attention)).eval(), attention)
    idx = torch.randint(0, 32, (2, 6), device=next(model.parameters()).device)

    logits = model(idx)

    assert logits.shape == (2, 6, 32)


@pytest.mark.parametrize("attention", _KINDS)
def test_future_token_does_not_change_earlier_logits(attention: str) -> None:
    torch.manual_seed(0)
    model = _on_device(GPT2(gpt2_config(attention=attention)).eval(), attention)
    idx = torch.randint(0, 32, (2, 6), device=next(model.parameters()).device)
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 32

    earlier = model(idx)[:, :-1]
    shifted = model(changed)[:, :-1]

    assert torch.allclose(earlier, shifted, atol=1e-5)


@pytest.mark.parametrize("attention", _KINDS)
def test_zero_output_projections_leave_the_embedding(attention: str) -> None:
    model = _on_device(GPT2(gpt2_config(attention=attention)).eval(), attention)
    for name, param in model.named_parameters():
        if name.endswith(_OUTPUT_PROJECTIONS):
            param.data.zero_()
    idx = torch.randint(0, 32, (2, 6), device=next(model.parameters()).device)
    positions = torch.arange(idx.size(1), device=idx.device)

    logits = model(idx)

    expected = model.lm_head(model.ln_f(model.wte(idx) + model.wpe(positions)))
    assert torch.allclose(logits, expected, atol=1e-5)


def test_token_embedding_is_tied_with_the_output_head() -> None:
    model = GPT2(gpt2_config())

    assert model.lm_head.weight is model.wte.weight


def test_sequence_longer_than_the_block_is_rejected() -> None:
    model = GPT2(gpt2_config(block_size=4))

    with pytest.raises(ValueError, match="block_size"):
        model(torch.randint(0, 32, (1, 5)))


@pytest.mark.parametrize("attention", _KINDS)
def test_loss_reaches_the_token_embedding(attention: str) -> None:
    model = _on_device(GPT2(gpt2_config(attention=attention)), attention)
    logits = model(torch.randint(0, 32, (2, 6), device=next(model.parameters()).device))

    logits.sum().backward()

    assert torch.isfinite(model.wte.weight.grad).all()
