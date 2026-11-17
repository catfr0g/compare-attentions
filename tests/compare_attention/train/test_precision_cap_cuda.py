"""Tiny real CUDA checks of scale capping and scaler-state continuation."""

import copy

import pytest
import torch

from compare_attention.train.precision import Precision

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA AMP")


def _step(model, optimizer, precision):
    optimizer.zero_grad(set_to_none=True)
    with precision.autocast():
        output = model(torch.full((2, 4), 0.25, device="cuda"))
        assert output.dtype == torch.float16
        loss = output.float().square().mean()
    before = model.weight.detach().clone()
    precision.backward_step(loss, optimizer)
    assert not torch.equal(before, model.weight)
    assert torch.isfinite(loss)
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    assert model.weight.dtype == torch.float32


@pytest.mark.parametrize("growth_interval", [1, 3])
def test_repeated_growth_is_capped_with_real_updates_and_restore_parity(growth_interval):
    model = torch.nn.Linear(4, 2, bias=False).cuda()
    with torch.no_grad():
        model.weight.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    precision = Precision(torch.device("cuda"), max_scale=128)
    precision.scaler.set_growth_interval(growth_interval)
    for step in range(1, 5):
        _step(model, optimizer, precision)
        assert precision.scaler.get_scale() == 128.0
        assert precision.scaler.state_dict()["_growth_tracker"] == step % growth_interval
    restored_model = copy.deepcopy(model)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored = Precision(torch.device("cuda"), max_scale=128)
    restored.scaler.load_state_dict(precision.scaler.state_dict())
    restored.optimizer_steps = precision.optimizer_steps
    restored.skipped_optimizer_steps = precision.skipped_optimizer_steps
    assert restored.scaler.state_dict() == precision.scaler.state_dict()
    for step in range(5, 9):
        _step(model, optimizer, precision)
        _step(restored_model, restored_optimizer, restored)
        assert precision.scaler.get_scale() == restored.scaler.get_scale() == 128.0
        assert precision.scaler.state_dict()["_growth_tracker"] == step % growth_interval
        assert restored.scaler.state_dict() == precision.scaler.state_dict()
        assert torch.equal(restored_model.weight, model.weight)
    assert precision.optimizer_steps == restored.optimizer_steps == 8
    assert precision.skipped_optimizer_steps == restored.skipped_optimizer_steps == 0


def test_cap_after_restore_preserves_nonzero_growth_tracker():
    model = torch.nn.Linear(4, 2, bias=False).cuda()
    with torch.no_grad():
        model.weight.fill_(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    precision = Precision(torch.device("cuda"), max_scale=128)
    state = precision.scaler.state_dict()
    state.update(scale=256.0, growth_interval=10, _growth_tracker=1)
    precision.scaler.load_state_dict(state)
    assert precision.scaler.get_scale() == 256.0
    _step(model, optimizer, precision)
    assert precision.scaler.get_scale() == 128.0
    assert precision.scaler.state_dict()["_growth_tracker"] == 2


def test_capped_real_overflow_still_aborts_and_preserves_backoff():
    parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    precision = Precision(torch.device("cuda"), max_scale=128)
    with pytest.raises(RuntimeError, match="overflow.*skipped"):
        precision.backward_step(parameter.square().sum() * float("inf"), optimizer)
    assert parameter.item() == 1.0
    assert precision.scaler.get_scale() == 64.0
    assert precision.optimizer_steps == 0
    assert precision.skipped_optimizer_steps == 1
