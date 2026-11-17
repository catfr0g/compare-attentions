"""Scale-cap policy on CPU fakes, with real optimizer-update counting."""

import pytest
import torch

from compare_attention.train.precision import INITIAL_SCALE, Precision


@pytest.mark.parametrize("cap", [0, 127, -1, float("nan"), float("inf"), True, "128"])
def test_invalid_scale_cap_rejected_before_scaler_construction(monkeypatch, cap) -> None:
    def unexpected_scaler(*_args, **_kwargs):
        pytest.fail("invalid cap must not construct a CUDA scaler")

    monkeypatch.setattr(torch.amp, "GradScaler", unexpected_scaler)
    with pytest.raises(ValueError, match="max_scale"):
        Precision(torch.device("cuda"), max_scale=cap)


class FakeScaler:
    """Simulate scaler growth/backoff without requiring CUDA on policy tests."""

    def __init__(self, *, scale=INITIAL_SCALE, next_scale=256.0, skip=False):
        self.current_scale = scale
        self.next_scale = next_scale
        self.skip = skip
        self.updates = []

    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer):
        pass

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self, new_scale=None):
        self.updates.append(new_scale)
        self.current_scale = self.next_scale if new_scale is None else new_scale

    def get_scale(self):
        return self.current_scale

    def is_enabled(self):
        return True


@pytest.mark.parametrize(
    "cap, expected_scale, updates",
    [
        (None, 256.0, [None]),
        (128.0, 128.0, [None, 128.0]),
        (256.0, 256.0, [None]),
        (512.0, 256.0, [None]),
    ],
)
def test_cap_only_limits_successful_growth(monkeypatch, cap, expected_scale, updates) -> None:
    scaler = FakeScaler()
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: scaler)
    precision = Precision(torch.device("cuda"), max_scale=cap)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    precision.backward_step(parameter.square(), optimizer)
    assert parameter.item() == pytest.approx(0.8)
    assert precision.optimizer_steps == 1
    assert precision.skipped_optimizer_steps == 0
    assert scaler.get_scale() == expected_scale
    assert scaler.updates == updates


@pytest.mark.parametrize("skip", [False, True])
def test_cap_never_masks_backoff_or_skipped_optimizer_step(monkeypatch, skip) -> None:
    scaler = FakeScaler(next_scale=64.0, skip=skip)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: scaler)
    precision = Precision(torch.device("cuda"), max_scale=128.0)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    if skip:
        with pytest.raises(RuntimeError, match="overflow.*skipped"):
            precision.backward_step(parameter.square(), optimizer)
    else:
        precision.backward_step(parameter.square(), optimizer)
    assert parameter.item() == pytest.approx(1.0 if skip else 0.8)
    assert precision.optimizer_steps == (0 if skip else 1)
    assert precision.skipped_optimizer_steps == int(skip)
    assert scaler.get_scale() == 64.0
    assert scaler.updates == [None]
    # The policy's post-step hook must not leak to subsequent optimizer use.
    optimizer.step()
    assert precision.optimizer_steps == (0 if skip else 1)


@pytest.mark.parametrize("loss_value", [float("nan"), float("inf")])
def test_capped_nonfinite_loss_with_finite_gradients_still_aborts(monkeypatch, loss_value):
    scaler = FakeScaler()
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: scaler)
    precision = Precision(torch.device("cuda"), max_scale=128)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(RuntimeError, match="nonfinite loss"):
        precision.backward_step(parameter.square() + loss_value, optimizer)
    assert parameter.item() == 1.0
    assert precision.optimizer_steps == 0
    assert precision.skipped_optimizer_steps == 1
    assert scaler.updates == [None]


def test_default_scaler_constructor_and_metadata_remain_compatible(monkeypatch) -> None:
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeScaler()

    monkeypatch.setattr(torch.amp, "GradScaler", factory)
    precision = Precision(torch.device("cuda"))
    assert calls == [(("cuda",), {"init_scale": 128.0})]
    assert precision.metadata() == {
        "mode": "amp_float16",
        "parameter_dtype": "float32",
        "autocast_dtype": "float16",
        "grad_scaler_enabled": True,
        "optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "initial_scale": 128.0,
        "final_scale": 128.0,
    }


def test_capped_metadata_records_configured_policy(monkeypatch) -> None:
    # Use a real CPU scaler so its policy accessors have production semantics.
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: scaler)
    precision = Precision(torch.device("cuda"), max_scale=128)
    metadata = precision.metadata()
    assert metadata["max_scale"] == 128.0
    assert metadata["growth_factor"] == 2.0
    assert metadata["backoff_factor"] == 0.5
    assert metadata["growth_interval"] == 2000
