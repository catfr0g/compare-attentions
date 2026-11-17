"""Inference-only evaluation against real tiny repository models."""

import importlib
import importlib.util

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from compare_attention.models.config import GPT2Config
from compare_attention.models.gpt2 import GPT2
from compare_attention.train.identity import parameter_fingerprint


def evaluation():
    assert importlib.util.find_spec("compare_attention.evaluation"), "evaluation feature missing"
    return importlib.import_module("compare_attention.evaluation")


def tiny_model():
    return GPT2(GPT2Config(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8))


def test_same_position_weighted_masked_metrics_tail_without_updates(monkeypatch):
    model = tiny_model()
    inputs = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]], dtype=np.int64)
    labels = np.array([[-1, 2, -1, -1], [3, 2, 1, 0], [-1, -1, -1, 3]], dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(inputs))
        expected = (
            F.cross_entropy(
                logits.flatten(0, 1),
                torch.from_numpy(labels).flatten(),
                ignore_index=-1,
                reduction="sum",
            ).item()
            / 6
        )
        correct = ((logits.argmax(-1).numpy() == labels) & (labels != -1)).sum()
    before = parameter_fingerprint(model)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", lambda *_a, **_k: pytest.fail("optimizer"))
    observed = []
    model.register_forward_pre_hook(
        lambda *_: observed.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
    )
    result = evaluation().evaluate_model(model, inputs, labels, batch_size=2)
    assert result["loss"] == pytest.approx(expected, abs=1e-6)
    assert result["correct"] == correct
    assert result["supervised_count"] == 6
    assert result["accuracy"] == correct / 6
    assert result["accuracy_percent"] == 100 * correct / 6
    assert result["rows"] == 3
    assert observed == [(True, False), (True, False)]
    assert parameter_fingerprint(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "fault",
    [
        "input_dtype",
        "label_dtype",
        "shape",
        "input_range",
        "label_range",
        "empty",
        "batch",
        "nonfinite",
    ],
)
def test_invalid_evaluation_inputs_fail_clearly(fault):
    model = tiny_model()
    inputs = np.array([[1, 2, 3, 4]], dtype=np.int64)
    labels = np.array([[-1, 2, -1, 3]], dtype=np.int64)
    batch = 2
    if fault == "input_dtype":
        inputs = inputs.astype(np.int32)
    elif fault == "label_dtype":
        labels = labels.astype(np.float32)
    elif fault == "shape":
        labels = labels[:, :2]
    elif fault == "input_range":
        inputs[0, 0] = 16
    elif fault == "label_range":
        labels[0, 0] = -2
    elif fault == "empty":
        labels[:] = -1
    elif fault == "batch":
        batch = 0
    else:
        model.register_forward_hook(lambda _m, _a, output: output * float("nan"))
    with pytest.raises(ValueError):
        evaluation().evaluate_model(model, inputs, labels, batch_size=batch)


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", "float16"), ("cpu", "bfloat16"), ("invalid", "float32"), ("cpu", "invalid")],
)
def test_unsupported_precision_is_not_silently_changed(device, dtype):
    model = tiny_model()
    inputs = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with pytest.raises(ValueError):
        evaluation().evaluate_model(model, inputs, inputs, device=device, dtype=dtype)


def test_finite_logits_with_overflowing_loss_are_rejected():
    model = tiny_model()
    inputs = np.ones((1, 4), dtype=np.int64)

    def extremes(_model, _args, logits):
        result = torch.full_like(logits, -2e38)
        result[..., 0] = 2e38
        return result

    model.register_forward_hook(extremes)
    with pytest.raises(ValueError, match="nonfinite.*loss"):
        evaluation().evaluate_model(model, inputs, inputs)
