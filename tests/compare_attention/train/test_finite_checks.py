"""CPU transport/AMP-policy proofs, not CUDA execution or speed evidence."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from compare_attention.train.precision import Precision, gradients_finite
from tests.compare_attention.train import finite_reference
from tests.compare_attention.train.test_distributed import assert_equal


class ScalarReads(TorchDispatchMode):
    """Count actual scalar extraction operators, including implicit bool reads."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture
def cpu_precision(monkeypatch):
    """Only redirect CUDA construction; execute the real public CPU GradScaler."""
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=2)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: scaler)
    return Precision(torch.device("cuda"))


@pytest.mark.parametrize("count", [1, 52])
def test_ddp_amp_guard_extracts_one_scalar_independent_of_parameter_count(cpu_precision, count):
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(count)]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    observations = []
    reads = ScalarReads()

    def synchronize(finite, message):
        observations.append((finite, message, reads.count))
        assert finite is True
        assert reads.count == 1

    loss = sum(parameter.square().sum() for parameter in parameters)
    with reads:
        cpu_precision.backward_step(loss, optimizer, synchronize=synchronize)
    assert observations == [(True, "nonfinite scaled gradients; optimizer update aborted", 1)]
    assert cpu_precision.optimizer_steps == 1
    assert cpu_precision.skipped_optimizer_steps == 0


@pytest.mark.parametrize("count", [1, 52])
def test_ddp_unscaled_step_has_constant_scalar_reads(monkeypatch, count):
    from compare_attention.train import distributed

    class SumModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(2)) for _ in range(count)]
            )

        def forward(self, _inputs):
            return sum(value.square().sum() for value in self.values)

    run = distributed.DistributedRun(0, 2, 0, torch.device("cpu"))
    monkeypatch.setattr(distributed, "current", lambda: run)
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "gather", lambda value: [value, value])
    monkeypatch.setattr(
        distributed.dist, "all_gather", lambda outputs, value: [out.copy_(value) for out in outputs]
    )
    model = SumModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reads = ScalarReads()
    with reads:
        loss = distributed.step(
            model, optimizer, torch.ones(2), torch.ones(2), lambda output, *_a, **_kw: output
        )
    assert loss == count
    assert reads.count == 7


def make_policy(monkeypatch, policy_class=Precision, cap=None):
    # CUDA-only production constructor is redirected, not production scaler methods.
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2)
    with monkeypatch.context() as patch:
        patch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: scaler)
        policy = policy_class(torch.device("cuda"), max_scale=cap)
    return policy


def require(finite, message):
    if not finite:
        raise RuntimeError(f"DDP {message}")


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_every_nonfinite_gradient_aborts_before_optimizer_without_scaler_changes(
    monkeypatch, position, bad
):
    for policy_class in (finite_reference.Precision, Precision):
        policy = make_policy(monkeypatch, policy_class)
        parameters = [torch.nn.Parameter(torch.ones(3)) for _ in range(3)]
        optimizer = torch.optim.AdamW(parameters, lr=0.1)
        parameters[position].register_hook(lambda gradient: torch.full_like(gradient, bad))
        before = policy.scaler.state_dict()
        with pytest.raises(
            RuntimeError, match="^DDP nonfinite scaled gradients; optimizer update aborted$"
        ):
            policy.backward_step(
                sum(parameter.square().sum() for parameter in parameters),
                optimizer,
                synchronize=require,
            )
        assert all(torch.equal(parameter, torch.ones(3)) for parameter in parameters)
        assert optimizer.state_dict()["state"] == {}
        assert policy.scaler.state_dict() == before
        assert policy.optimizer_steps == policy.skipped_optimizer_steps == 0
        assert not optimizer._optimizer_step_post_hooks  # pylint: disable=protected-access


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("synchronized", [False, True])
def test_nonfinite_detached_loss_matches_v6_error_counters_and_scale(
    monkeypatch, bad, synchronized
):
    results = []
    for policy_class in (finite_reference.Precision, Precision):
        policy = make_policy(monkeypatch, policy_class)
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        with pytest.raises(RuntimeError) as failure:
            policy.backward_step(
                parameter.square().sum() + bad,
                optimizer,
                synchronize=require if synchronized else None,
            )
        results.append(
            (
                str(failure.value),
                policy.metadata(),
                policy.scaler.state_dict(),
                optimizer.state_dict(),
            )
        )
        assert torch.equal(parameter, torch.ones(3))
        assert torch.isfinite(parameter.grad).all()
        assert optimizer.state_dict()["state"] == {}
    assert_equal(results[0], results[1])


@pytest.mark.parametrize("kind", ["none", "empty", "noncontiguous", "groups"])
def test_finite_aggregation_preserves_gradient_storage_and_vacuous_checks(kind):
    parameters = [torch.nn.Parameter(torch.ones(2, 3)) for _ in range(3)]
    optimizer = torch.optim.SGD([{"params": parameters[:1]}, {"params": parameters[1:]}], lr=0.1)
    if kind == "empty":
        parameters.append(torch.nn.Parameter(torch.empty(0)))
        optimizer.add_param_group({"params": parameters[-1:]})
    if kind != "none":
        for parameter in parameters:
            parameter.grad = (
                torch.ones(3, 2).t() if kind == "noncontiguous" else torch.ones_like(parameter)
            )
    before = [(p.grad, p.grad.clone() if p.grad is not None else None) for p in parameters]
    reads = ScalarReads()
    with reads:
        assert gradients_finite(optimizer) is True
    assert reads.count == (0 if kind == "none" else 1)
    for parameter, (gradient, values) in zip(parameters, before, strict=True):
        assert parameter.grad is gradient
        if gradient is not None:
            assert torch.equal(gradient, values)
    assert gradients_finite(optimizer, loss=torch.tensor(float("nan"))) is False


@pytest.mark.parametrize("synchronized", [False, True])
@pytest.mark.parametrize("cap", [None, 128.0])
def test_finite_amp_trajectory_is_bitwise_v6_including_optimizer_scaler_and_rng(
    monkeypatch, synchronized, cap
):
    results = []
    for policy_class in (finite_reference.Precision, Precision):
        torch.manual_seed(733)
        policy = make_policy(monkeypatch, policy_class, cap)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 2)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        history, gradients = [], []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(2, 3)).square().sum()
            history.append(loss.item())
            policy.backward_step(loss, optimizer, synchronize=require if synchronized else None)
            gradients.append([p.grad.clone() for p in model.parameters()])
        results.append(
            (
                model.state_dict(),
                optimizer.state_dict(),
                policy.scaler.state_dict(),
                policy.metadata(),
                history,
                gradients,
                torch.get_rng_state(),
                torch.rand(3),
            )
        )
    assert_equal(results[0], results[1])


@pytest.mark.parametrize("synchronized", [False, True])
def test_all_missing_gradients_preserve_gradscaler_assertion(monkeypatch, synchronized):
    results = []
    for policy_class in (finite_reference.Precision, Precision):
        policy = make_policy(monkeypatch, policy_class)
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        with pytest.raises(AssertionError) as failure:
            policy.backward_step(
                torch.tensor(1.0, requires_grad=True),
                optimizer,
                synchronize=require if synchronized else None,
            )
        results.append((str(failure.value), policy.metadata(), optimizer.state_dict()))
    assert_equal(results[0], results[1])


def test_real_overflow_skip_matches_v6_without_synchronize(monkeypatch):
    results = []
    for policy_class in (finite_reference.Precision, Precision):
        policy = make_policy(monkeypatch, policy_class)
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.register_hook(lambda gradient: gradient * float("inf"))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        with pytest.raises(RuntimeError, match="CUDA AMP overflow: optimizer update skipped"):
            policy.backward_step(parameter.square().sum(), optimizer)
        assert torch.equal(parameter, torch.ones(3))
        assert policy.optimizer_steps == 0
        assert policy.skipped_optimizer_steps == 1
        assert policy.scaler.get_scale() == 64.0
        results.append((policy.metadata(), policy.scaler.state_dict(), optimizer.state_dict()))
    assert_equal(results[0], results[1])


def test_real_two_rank_amp_v6_differential_resume_and_nonfinite_abort(tmp_path):
    import json

    from compare_attention.train.flops import estimate_step
    from tests.compare_attention.train.test_distributed import fixture, launch

    config = fixture(tmp_path, dropout=0.2)
    config = config.model_copy(
        update={
            "schedule": config.schedule.model_copy(
                update={
                    "dtype": "float16",
                    "checkpoint_every_steps": 2,
                    "checkpoint_keep_last": 2,
                    "train_flops": estimate_step(config).total * 7,
                }
            )
        }
    )
    (tmp_path / "config.json").write_text(config.model_dump_json())
    launch(tmp_path, str(tmp_path), module="tests.compare_attention.train.finite_checks_worker")
    for rank in range(2):
        proof = json.loads((tmp_path / f"finite-proof-{rank}.json").read_text())
        assert proof["updates"] == 7
        assert proof["next_rng_equal"] is proof["group_destroyed"] is True
        assert len(proof["equal_fields"]) == 11
        assert [row["optimizer_steps"] for row in proof["failures"]] == [0, 0, 0]
