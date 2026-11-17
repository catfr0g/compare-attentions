"""Numerical and overflow evidence from real CUDA AMP and production FLA."""

import copy
import json

import pytest
import torch

from compare_attention.data.store import prepare_kv_retrieval
from compare_attention.models.attention import build_attention
from compare_attention.models.gpt2 import GPT2
from compare_attention.train import loop
from compare_attention.train.backup import BackupTracker
from compare_attention.train.compiler import compile_model
from compare_attention.train.precision import Precision, current_precision
from tests.compare_attention.data.factories import config as data_config
from tests.compare_attention.models.factories import gpt2_config
from tests.compare_attention.train.factories import train_config, train_result

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA AMP")


@pytest.fixture(autouse=True)
def isolated_compiler_state():
    """Unrelated compiled models must not exhaust this test's Dynamo guard budget."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_fp16_matches_fp32_small_model_loss_gradients_and_updates(attention) -> None:
    torch.manual_seed(42)
    config = gpt2_config(
        vocab_size=256,
        block_size=32,
        n_layer=1,
        n_embd=32,
        n_head=2,
        attention=attention,
        dropout=0.0,
    )
    reference = GPT2(config).cuda()
    mixed = copy.deepcopy(reference)
    inputs = torch.randint(0, 256, (2, 32), device="cuda")
    labels = torch.randint(0, 256, inputs.shape, device="cuda")
    baseline = torch.optim.SGD(reference.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(mixed.parameters(), lr=0.01)
    expected = loop._step(reference, baseline, inputs, labels)  # pylint: disable=protected-access
    precision = Precision(torch.device("cuda"))
    with precision.activate():
        actual = loop._step(mixed, optimizer, inputs, labels)  # pylint: disable=protected-access
        val_loss, _ = loop._validate(mixed, inputs, labels, 2)  # pylint: disable=protected-access
    assert actual == pytest.approx(expected, abs=0.01, rel=0.005)
    assert val_loss == pytest.approx(actual, abs=0.05)
    for first, second in zip(reference.parameters(), mixed.parameters(), strict=True):
        assert second.dtype == torch.float32
        assert second.grad is not None and torch.isfinite(second.grad).all()
        torch.testing.assert_close(second.grad, first.grad, atol=0.003, rtol=0.05)
        torch.testing.assert_close(second, first, atol=0.0001, rtol=0.01)
    assert precision.optimizer_steps == 1
    assert current_precision() is None


@pytest.mark.parametrize("attention", ["classic", "deltanet"])
def test_compiled_amp_train_validation_train_has_finite_meaningful_gradients(attention) -> None:

    torch.manual_seed(42)
    model = GPT2(
        gpt2_config(
            vocab_size=256, block_size=128, n_layer=1, n_embd=32, n_head=2, attention=attention
        )
    ).cuda()
    evidence = {}
    compiled = compile_model(model, True, evidence)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    inputs = torch.randint(0, 256, (2, 128), device="cuda")
    labels = torch.randint(0, 256, inputs.shape, device="cuda")
    precision = Precision(torch.device("cuda"))
    with precision.activate():
        for _ in range(2):
            before = [p.detach().clone() for p in model.parameters()]
            loss = loop._step(compiled, optimizer, inputs, labels)  # pylint: disable=protected-access
            assert torch.isfinite(torch.tensor(loss))
            parameters = list(model.parameters())
            assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in parameters)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            assert any(p.grad.abs().sum() > 0 for p in parameters)
            assert any(not torch.equal(a, b) for a, b in zip(before, parameters, strict=True))
            quality = loop._validate(compiled, inputs, labels, 2)  # pylint: disable=protected-access
            assert all(torch.isfinite(torch.tensor(value)) for value in quality)
    assert precision.optimizer_steps == 2 and precision.skipped_optimizer_steps == 0
    assert evidence["successful_training_backend_invocations"] > 0
    assert evidence["successful_validation_backend_invocations"] > 0
    assert evidence["backend_invocation_verified"] is True


def test_overflow_is_persisted_and_never_emits_a_completed_result(tmp_path, monkeypatch) -> None:
    prepare_kv_retrieval(data_config(), tmp_path / "data")
    config = train_config(tmp_path / "data", dtype="float16")
    original = loop._loss  # pylint: disable=protected-access

    def overflow(*args, **kwargs):
        return original(*args, **kwargs) * float("inf")

    monkeypatch.setattr(loop, "_loss", overflow)
    with pytest.raises(RuntimeError, match="overflow.*skipped"):
        loop.train(config, BackupTracker(tmp_path / "result"))
    identity = json.loads((tmp_path / "result" / "run_identity.json").read_text())
    assert identity["status"] == "failed"
    precision = identity["precision"]
    assert precision["optimizer_steps"] == 0
    assert precision["skipped_optimizer_steps"] == 1
    assert precision["initial_scale"] == 128.0
    assert precision["final_scale"] == 64.0
    assert "overflow" in identity["failure"]
    assert not (tmp_path / "result" / "model.pt").exists()
    events = tmp_path / "result" / "metrics.jsonl"
    assert not events.exists() or not events.read_text().strip()
    assert current_precision() is None


def test_nonfinite_loss_with_finite_derivatives_cannot_update_parameters(
    tmp_path, monkeypatch
) -> None:
    prepare_kv_retrieval(data_config(), tmp_path / "data")
    config = train_config(tmp_path / "data", dtype="float16")
    original = loop._loss  # pylint: disable=protected-access

    def nonfinite(*args, **kwargs):
        # Adding a detached NaN leaves finite derivatives: GradScaler alone misses it.
        return original(*args, **kwargs) + float("nan")

    monkeypatch.setattr(loop, "_loss", nonfinite)
    with pytest.raises(RuntimeError, match="nonfinite loss"):
        loop.train(config, BackupTracker(tmp_path / "result"))
    identity = json.loads((tmp_path / "result" / "run_identity.json").read_text())
    assert identity["status"] == "failed"
    assert identity["precision"]["optimizer_steps"] == 0
    assert not (tmp_path / "result" / "model.pt").exists()


def test_fp16_rejects_result_without_matching_real_updates(tmp_path, monkeypatch) -> None:

    prepare_kv_retrieval(data_config(), tmp_path / "data")
    config = train_config(tmp_path / "data", dtype="float16")
    monkeypatch.setattr(loop, "_run", lambda config, *_args: train_result(config))
    with pytest.raises(RuntimeError, match="optimizer.*count"):
        loop.train(config, BackupTracker(tmp_path / "result"))
    identity = json.loads((tmp_path / "result" / "run_identity.json").read_text())
    assert identity["status"] == "failed"
    assert identity["precision"]["optimizer_steps"] == 0
    assert not (tmp_path / "result" / "model.pt").exists()


def test_production_delta_adapter_fp16_backward_has_finite_meaningful_gradients() -> None:
    torch.manual_seed(42)
    model = build_attention(gpt2_config(n_embd=768, n_head=12, attention="deltanet"), 0).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    inputs = torch.randn(8, 512, 768, device="cuda", requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    precision = Precision(torch.device("cuda"))
    with precision.autocast():
        output = model(inputs)
        loss = output.float().square().mean()
    precision.backward_step(loss, optimizer)
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    parameters = list(model.parameters())
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in parameters)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert all(p.grad.abs().sum() > 0 for p in parameters)
    assert any(
        not torch.equal(before, after) for before, after in zip(initial, parameters, strict=True)
    )
    assert precision.optimizer_steps == 1 and precision.skipped_optimizer_steps == 0
