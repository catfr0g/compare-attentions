"""Opt-in torchrun runtime; the global ordered stream stays in the existing loop."""

import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from compare_attention.train.result import Quality

_ACTIVE: ContextVar["DistributedRun | None"] = ContextVar("distributed_training", default=None)

# Wire IDs, not hashes: zero requests exact object diagnostics on every rank.
# All calls send a header so a hot/unknown phase mismatch cannot split protocols.
_HOT_PHASE_IDS = {"forward": 1, "scaled_optimizer_step": 2, "optimizer_step": 3, "log_loss": 4}


def _windows_gloo(store, rank, world_size, timeout):
    # Windows UV transport can choose IPv6 from a dual-stack interface and hang
    # before the process-group timeout. This single-node CPU path binds IPv4.
    options = dist.ProcessGroupGloo._Options()  # pylint: disable=protected-access
    options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    options._timeout = timeout
    return dist.ProcessGroupGloo(store, rank, world_size, options)


def current():
    """Return this run's distributed context, or None for the unchanged legacy path."""
    return _ACTIVE.get()


def local_call(function, phase):
    """Keep legacy calls direct; synchronize rank-local setup/validation failures."""
    run = current()
    return function() if run is None else run.call(function, phase=phase)


def run_training(config, tracker, resume_from, enabled, runner):
    """Operational wrapper, never an independent training implementation."""
    from compare_attention.train.distributed_checkpoint import preflight
    from compare_attention.train.lifecycle import observe_failures
    from compare_attention.train.tracker import DisabledTracker, validate_resume_destination

    with distributed_run(enabled) as run:
        mirror = tracker if tracker is not None else DisabledTracker()
        if run is None and resume_from is not None:
            validate_resume_destination(mirror)
        if run is not None:
            run.call(lambda: run.batch(config.schedule.batch_size), phase="global_batch")
            contracts = run.gather((config.model_dump(mode="json"), str(resume_from)))
            run.require(
                all(contract == contracts[0] for contract in contracts),
                "config/resume contract differs across ranks",
            )
            if resume_from is not None:
                run.call(lambda: validate_resume_destination(mirror), phase="resume_destination")
                run.call(lambda: preflight(resume_from, config), phase="resume_preflight")
            mirror = PrimaryTracker(mirror, run)
        with observe_failures(tracker, rank=(run.rank if run else None)):
            return runner(config, mirror, resume_from=resume_from)


class DistributedRun:
    """Bound collective waits and isolate all external writers on rank zero."""

    def __init__(self, rank, world_size, local_rank, device):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device

    def gather(self, value) -> list[Any]:
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def call(self, function: Callable[[], Any], *, primary=False, phase=None) -> Any:
        """Exchange phase failures before any peer enters its next collective."""
        from compare_attention.train.lifecycle import failure_phase, primary_failure

        value, failure = None, None
        try:
            if not primary or self.rank == 0:
                with failure_phase(phase or function.__name__):
                    value = function()
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Rank coordination is also a console boundary. Never transmit text
            # or chained exceptions: SDK errors can contain credentials and URLs.
            failure = primary_failure(error, phase or function.__name__)
        name = phase or function.__name__
        phase_id = _HOT_PHASE_IDS.get(name, 0)
        header = torch.tensor(
            [phase_id, int(failure is not None)], dtype=torch.int64, device=self.device
        )
        headers = [torch.empty_like(header) for _ in range(self.world_size)]
        dist.all_gather(headers, header)
        # One device-to-host boundary, including NCCL's device-resident headers.
        agreed = torch.stack(headers).cpu().tolist()
        if phase_id and all(row == [phase_id, 0] for row in agreed):
            return value
        rows = self.gather((name, failure))
        if len({row[0] for row in rows}) != 1:
            raise RuntimeError(f"DDP phase order mismatch: {[row[0] for row in rows]}")
        failures = [row[1] for row in rows]
        if any(failures):
            error = RuntimeError(f"DDP phase failed: {failures}")
            error._training_primary_failure = next(row for row in failures if row)  # pylint: disable=protected-access
            raise error from None
        return value

    def validation_observation(self, quality: Quality) -> Quality:
        """Agree on rank zero's full-split result before logging or stopping.

        Every rank evaluates the same complete split, not disjoint shards. Keep
        its supervised-token weighting; averaging rank means would redefine it.
        Call only after collective validation success, never to repair history.
        """
        values = [quality]
        dist.broadcast_object_list(values, src=0, device=self.device)
        return values[0]

    def require(self, condition, message):
        flag = torch.tensor(int(bool(condition)), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            raise RuntimeError(f"DDP {message}")

    def wrap(self, model):
        return DistributedDataParallel(
            model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    def batch(self, global_batch):
        if global_batch < self.world_size or global_batch % self.world_size:
            raise ValueError("DDP global batch must be divisible by world_size")
        return global_batch // self.world_size


@contextmanager
def distributed_run(enabled):
    """Own a single-node process group only when explicitly requested."""
    if not enabled:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and current() is None:
            raise ValueError("torchrun requires explicit distributed=True / --distributed")
        yield current()
        return
    if current() is not None:
        yield current()
        return
    try:
        rank, world, local = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    except (KeyError, ValueError) as error:
        raise ValueError("DDP requires torchrun with RANK/WORLD_SIZE/LOCAL_RANK") from error
    if world != 2 or not 0 <= rank < world or local != rank:
        raise ValueError("DDP requires a single node with world_size=2 and local_rank=rank")
    if dist.is_initialized():
        raise ValueError("DDP trainer requires ownership of the process group")
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"
    if os.name == "nt" and device.type == "cpu":
        backend = "gloo_ipv4"
        if backend not in dist.Backend.backend_list:
            dist.Backend.register_backend(backend, _windows_gloo, devices=["cpu"])
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=60),
    )
    token = _ACTIVE.set(DistributedRun(rank, world, local, device))
    try:
        yield current()
    finally:
        _ACTIVE.reset(token)
        dist.destroy_process_group()


class PrimaryTracker:
    """Collective callback facade: no duplicate tasks, metrics, snapshots or uploads."""

    def __init__(self, tracker, run):
        self.tracker = tracker
        self.run = run

    def _emit(self, name, *args):
        def invoke():
            callback = getattr(self.tracker, name, None)
            if callable(callback):
                return callback(*args)
            if name == "save_checkpoint":
                raise ValueError("checkpoint enabled but no accepting checkpoint sink")
            return None

        return self.run.call(invoke, primary=True, phase=name)

    def start(self, config):
        self._emit("start", config)

    def validate_resume_destination(self):
        self._emit("validate_resume_destination")

    def record_identity(self, identity):
        self._emit("record_identity", identity)

    def log_loss(self, loss, flops):
        self._emit("log_loss", loss, flops)

    def log_learning_rate(self, learning_rate, flops):
        self._emit("log_learning_rate", learning_rate, flops)

    def log_validation(self, loss, accuracy, flops):
        self._emit("log_validation", loss, accuracy, flops)

    def save_checkpoint(self, payload, step, keep_last):
        self._emit("save_checkpoint", payload, step, keep_last)

    def finish(self, result, model):
        self._emit("finish", result, model)


def initialize_evidence(config, identity):
    """Price real rank-local shapes and bind actual matched initialization."""
    from compare_attention.train.flops import estimate_step

    run = current()
    assert run is not None
    local_batch = run.batch(config.schedule.batch_size)
    local = config.model_copy(
        update={"schedule": config.schedule.model_copy(update={"batch_size": local_batch})}
    )
    cost = estimate_step(local).to_dict()
    rows = run.gather(
        {
            "rank": run.rank,
            "local_rank": run.local_rank,
            "step_flops": cost["total"],
            "accounting": cost,
            "effective_seed": identity["effective_seed"],
            "initial_parameters_sha256": identity["initial_parameters_sha256"],
        }
    )
    run.require(
        len({row["initial_parameters_sha256"] for row in rows}) == 1,
        "initial parameters differ across ranks",
    )
    run.require(
        sum(row["step_flops"] for row in rows) == identity["accounting"]["total"],
        "global FLOPs do not equal sum of local priced shapes",
    )
    identity["schema_version"] = 3  # Explicitly outside all legacy single-device gates.
    identity["distributed"] = {
        "world_size": run.world_size,
        "global_batch_size": config.schedule.batch_size,
        "local_batch_size": local_batch,
        "global_step_flops": sum(row["step_flops"] for row in rows),
        "sharding": "ordered_cyclic_contiguous_v1",
        "ranks": rows,
    }


def finalize_evidence(identity):
    """Verify each rank's actual compile/precision/final tensors before publishing."""
    run = current()
    assert run is not None
    rows = run.gather(
        {
            "rank": run.rank,
            "compiler": identity["compiler"].copy(),
            "precision": identity.get("precision"),
            "final_parameters_sha256": identity["final_parameters_sha256"],
            "status": identity["status"],
            "stop_reason": identity["stop_reason"],
        }
    )
    run.require(
        len({row["final_parameters_sha256"] for row in rows}) == 1,
        "final parameters differ across ranks",
    )
    run.require(
        len({(row["status"], row["stop_reason"]) for row in rows}) == 1,
        "rank stopping decisions differ",
    )
    for row, evidence in zip(identity["distributed"]["ranks"], rows, strict=True):
        row.update(evidence)


def step(model, optimizer, inputs, labels, loss_function):
    """Weight gradients by global supervised targets, not by rank-local means."""
    from compare_attention.train.compiler import compilation_phase
    from compare_attention.train.precision import autocast, current_precision, gradients_finite

    run = current()
    model.train()
    assert run is not None
    with compilation_phase("training"), autocast():
        loss = run.call(
            lambda: loss_function(model(inputs), labels, reduction="sum"), phase="forward"
        )
    count = (labels != -1).sum()
    dist.all_reduce(count)
    run.require(count.item() > 0, "global batch has no supervised targets")
    run.require(torch.isfinite(loss).item(), "nonfinite loss; update aborted")
    total = loss.detach().clone()
    dist.all_reduce(total)
    optimizer.zero_grad(set_to_none=True)
    weighted = loss * (run.world_size / count)
    if (precision := current_precision()) is not None:
        run.call(
            lambda: precision.backward_step(weighted, optimizer, synchronize=run.require),
            phase="scaled_optimizer_step",
        )
    else:
        weighted.backward()
        run.require(gradients_finite(optimizer), "nonfinite gradients; update aborted")
        run.call(optimizer.step, phase="optimizer_step")
    return (total / count).item()
