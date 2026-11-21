"""Preserve native backup and identity; ClearML is auxiliary diagnostic telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter


def failure_phase(name):
    """Lazy normal-lifecycle hook; pure legacy readback needs no scientific imports."""
    from compare_attention.train.lifecycle import failure_phase as guard

    return guard(name)


# ClearML utilities/resource_monitor.py: ResourceMonitor._machine_stats and
# _get_gpu_stats. Single-node titles only; never a ':monitor:' prefix exemption.
_MACHINE_SERIES = frozenset(
    {
        "cpu_usage",
        "cpu_temperature",
        "memory_used_gb",
        "memory_free_gb",
        "disk_free_percent",
        "network_tx_mbs",
        "network_rx_mbs",
        "io_read_mbs",
        "io_write_mbs",
    }
)
_GPU_SERIES = re.compile(
    r"gpu_(?:0|[1-9][0-9]*)_(?:temperature|utilization|mem_used_gb|mem_usage|mem_free_gb|power_draw)"
)


def resource_event(row) -> bool:
    """Recognize only exact producer namespaces and declared scalar series."""
    metric, variant = row.get("metric"), row.get("variant")
    return (metric == ":monitor:machine" and variant in _MACHINE_SERIES) or (
        metric == ":monitor:gpu"
        and isinstance(variant, str)
        and _GPU_SERIES.fullmatch(variant) is not None
    )


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _constant_config_contract(config, updates=0) -> dict[str, object]:
    # Compatibility name for authenticated callers; policy is request-derived,
    # never seed/task hardcoded. No training or model import for constant traces.
    value = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    _require(type(value) is dict, "CLEARML_CONFIG_CONTRACT")
    try:
        normalized = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("CLEARML_CONFIG_CONTRACT") from error
    optim = normalized.get("optim")
    _require(type(optim) is dict, "CLEARML_CONFIG_CONTRACT")
    policy = optim.get("lr_schedule")
    _require(policy in {"constant", "warmup_cosine"}, "CLEARML_LR_POLICY")
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "lr_policy": policy,
        "expected_learning_rate_events": 0 if policy == "constant" else updates,
    }


def clearml_domain(value):
    """The pinned SDK's parameters readback uses string leaves, including nulls."""
    if isinstance(value, dict):
        return {key: clearml_domain(item) for key, item in value.items()}
    return "" if value is None else str(value)


def validate_unsampled_readback(
    value, *, task_id: str, updates: int, validations: int, config, include_peak_gpu_bytes=True
) -> dict[str, object]:
    """Require the complete unsampled scalar history from a fresh SDK read."""
    policy = _constant_config_contract(config, updates)
    config = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    if "parameters" in value:
        _require(value["parameters"] == clearml_domain(config), "CLEARML_CONFIG_READBACK")
    _require(value["status"] == "completed", "CLEARML_COMPLETION")
    _require(value.get("task_id", task_id) == task_id, "CLEARML_TASK_ID")
    events = value["events"]
    _require(
        type(value["total"]) is int and value["total"] == len(events), "CLEARML_UNSAMPLED_HISTORY"
    )
    previous = {}
    for row in events:
        _require(type(row) is dict, "CLEARML_EVENT_SCHEMA")
        key = (row.get("metric"), row.get("variant"))
        _require(all(type(item) is str for item in key), "CLEARML_EVENT_SCHEMA")
        _require(row.get("task", task_id) == task_id, "CLEARML_TASK_ID")
        _require(
            type(row.get("value")) in (int, float) and math.isfinite(row["value"]),
            "CLEARML_EVENT_VALUE",
        )
        iteration = row.get("iter")
        _require(type(iteration) is int and iteration >= 0, "CLEARML_EVENT_ITERATION")
        if not resource_event(row):
            _require(iteration > previous.get(key, -1), "CLEARML_EVENT_ORDER")
            previous[key] = iteration
    counts = Counter((row.get("metric"), row.get("variant")) for row in events)
    required = {
        ("train", "loss"): updates,
        ("train", "learning_rate"): policy["expected_learning_rate_events"],
        ("validation", "loss"): validations,
        ("validation", "accuracy"): validations,
    }
    for name in (
        "steps",
        "train_flops",
        "requested_train_flops",
        "early_stopped",
        "budget_exhausted",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "train_tokens_per_second",
        "peak_gpu_bytes",
    ):
        required[("result", name)] = 1
    if not include_peak_gpu_bytes:
        required[("result", "peak_gpu_bytes")] = 0
    _require(
        all(counts[key] == count for key, count in required.items()),
        "CLEARML_COMPLETE_EVENT_COUNTS",
    )
    resources = sum(resource_event(row) for row in events)
    scientific = len(events) - resources
    _require(sum(required.values()) == scientific, "CLEARML_COMPLETE_EVENT_COUNTS")
    _scientific_trace(events, config, updates)
    return {
        "task_id": task_id,
        "status": "completed",
        "event_count": len(events),
        "scientific_event_count": scientific,
        "resource_event_count": resources,
        "required_counts": {f"{key[0]}/{key[1]}": count for key, count in required.items()},
        **policy,
        "unsampled_complete_history": True,
        "flush_before_close": True,
    }


def _scientific_trace(events, config, updates):
    """Validate order/accounting/LR against the requested numerical policy."""

    def rows(title, series):
        return [r for r in events if (r["metric"], r["variant"]) == (title, series)]

    losses = rows("train", "loss")
    _require(
        type(updates) is int and updates > 0 and len(losses) == updates, "CLEARML_UPDATE_COUNT"
    )
    step = losses[0]["iter"]
    _require(
        step > 0 and [r["iter"] for r in losses] == [step * i for i in range(1, updates + 1)],
        "CLEARML_FLOPS_AXIS",
    )
    result = {r["variant"]: r for r in events if r["metric"] == "result"}
    _require(all(r["iter"] == updates for r in result.values()), "CLEARML_RESULT_AXIS")
    budget = config["schedule"]["train_flops"]
    _require(type(budget) is int and budget > 0, "CLEARML_CONFIG_CONTRACT")
    for name, expected in {
        "steps": updates,
        "train_flops": step * updates,
        "requested_train_flops": budget,
        "budget_exhausted": float(step * updates >= budget),
    }.items():
        _require(result[name]["value"] == expected, "CLEARML_RESULT_ACCOUNTING")
    exhausted = step * updates >= budget
    _require(not exhausted or step * (updates - 1) < budget, "CLEARML_UPDATE_BUDGET")
    _require(result["early_stopped"]["value"] == float(not exhausted), "CLEARML_STOP_REASON")
    cadence = config["schedule"]["val_every_steps"]
    _require(type(cadence) is int and cadence > 0, "CLEARML_CONFIG_CONTRACT")
    val_axis = [step * i for i in range(cadence, updates, cadence)] + [step * updates]
    for name in ("loss", "accuracy"):
        _require(
            [r["iter"] for r in rows("validation", name)] == val_axis, "CLEARML_VALIDATION_AXIS"
        )
    for name in ("loss", "accuracy"):
        _require(
            result["val_" + name]["value"] == rows("validation", name)[-1]["value"],
            "CLEARML_FINAL_QUALITY",
        )
    mean_loss = math.fsum(r["value"] for r in losses) / updates
    _require(
        math.isclose(result["train_loss"]["value"], mean_loss, rel_tol=1e-12, abs_tol=1e-15),
        "CLEARML_TRAIN_QUALITY",
    )
    if config["optim"]["lr_schedule"] != "constant":
        from compare_attention.train.config import Optim
        from compare_attention.train.lr_schedule import next_learning_rate

        optim = Optim(**config["optim"])
        rates = rows("train", "learning_rate")
        _require([r["iter"] for r in rates] == [r["iter"] for r in losses], "CLEARML_LR_AXIS")
        for i, row in enumerate(rates):
            expected = next_learning_rate(optim, budget, i * step, step)
            _require(
                math.isclose(row["value"], expected, rel_tol=1e-12, abs_tol=1e-15),
                "CLEARML_LR_TRACE",
            )


class _FlushTask:
    """Intercept the frozen tracker's close without changing its scalar logic."""

    def __init__(self, task, readback, persist=None) -> None:
        self._task, self._readback, self._persist = task, readback, persist
        self.receipt = None

    def __getattr__(self, name):
        return getattr(self._task, name)

    def close(self) -> None:
        with failure_phase("clearml_flush"):
            self._task.flush(wait_for_uploads=True)
        identifier = self._task.id
        with failure_phase("clearml_close"):
            self._task.close()
        with failure_phase("clearml_readback"):
            self.receipt = self._readback(identifier)
        if self._persist is not None:
            with failure_phase("clearml_raw_receipt_write"):
                self._persist(self.receipt)


class TrainingServiceTracker:
    """Use the frozen ClearMLTracker with an explicit unsampled close barrier."""

    def __init__(
        self,
        values,
        name: str,
        *,
        updates: int,
        validations: int,
        task_factory=None,
        readback=None,
        invoke=None,
    ) -> None:
        self.values, self.name = values, name
        self.updates, self.validations = updates, validations
        self._task_factory, self._readback = task_factory, readback
        self._invoke, self._proxy, self.delegate = invoke, None, None
        self.task_id, self.receipt = None, None
        self._config = None
        self.persist_readback = None

    def _call(self, phase, action):
        original = action

        def observed():
            with failure_phase(phase):
                return original()

        action = observed
        if self._invoke is not None:
            return self._invoke(phase, action)
        if not self.values:  # CPU engineering fixture: no service environment.
            return action()
        from quality_inherit import leaf

        services = leaf("services")

        def operation():
            with services.service_env(self.values, "track"):
                return action()

        return services.safe(phase, operation)

    @staticmethod
    def _production_task(name):
        from clearml import Task

        return Task.init(
            project_name="compare-attention",
            task_name=name,
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
            auto_connect_streams=False,
        )

    @staticmethod
    def _production_readback(task_id: str) -> dict[str, object]:
        from clearml import Task
        from clearml.backend_api.services import events as api_events

        observed = Task.get_task(task_id=task_id)
        observed.reload()
        status = str(observed.get_status()).lower().split(".")[-1]
        events, scroll_id, total = [], None, None
        pages, cursors, seen = [], set(), set()
        while True:
            request = api_events.GetTaskEventsRequest(
                task=task_id,
                order="asc",
                scroll_id=scroll_id,
                batch_size=10_000,
                event_type="training_stats_scalar",
            )
            result = Task._get_default_session().send(request)  # pylint: disable=protected-access
            # Inspect raw wire JSON, not DTO-coerced totals (2.0/True -> int).
            raw = result.response_data
            _require(type(raw) is dict, "CLEARML_PAGE_SCHEMA")
            page, reported = raw.get("events"), raw.get("total")
            _require(
                type(page) is list and type(reported) is int and reported >= 0,
                "CLEARML_PAGE_SCHEMA",
            )
            if total is None:
                total = reported
            _require(total == reported, "CLEARML_PAGE_TOTAL_DRIFT")
            pages.append(raw)
            for row in page:
                _require(type(row) is dict, "CLEARML_EVENT_SCHEMA")
                key = json.dumps(row, sort_keys=True, allow_nan=False)
                _require(key not in seen, "CLEARML_PAGE_DUPLICATE")
                seen.add(key)
                events.append(row)
            _require(len(events) <= total, "CLEARML_PAGE_OVERSHOOT")
            if len(events) == total:
                break
            _require(bool(page), "CLEARML_PAGE_INCOMPLETE")
            next_cursor = raw.get("scroll_id")
            _require(
                type(next_cursor) is str and bool(next_cursor) and next_cursor not in cursors,
                "CLEARML_PAGE_CURSOR",
            )
            cursors.add(next_cursor)
            scroll_id = next_cursor
        return {
            "status": status,
            "task_id": task_id,
            "events": events,
            "total": total,
            "pages": pages,
            "parameters": observed.get_parameters_as_dict()["General"],
        }

    def start(self, config) -> None:
        _constant_config_contract(config)
        self._config = config

        def operation():
            from compare_attention.train.tracker import ClearMLTracker

            def factory(current):
                task = (
                    self._task_factory(current)
                    if self._task_factory is not None
                    else self._production_task(self.name)
                )
                self.task_id = task.id
                reader = self._readback or self._production_readback
                self._proxy = _FlushTask(task, reader, self.persist_readback)
                return self._proxy

            self.delegate = ClearMLTracker(task_factory=factory)
            self.delegate.start(config)

        self._call("clearml_start", operation)

    def log_loss(self, loss, flops) -> None:
        self._call("clearml_loss", lambda: self.delegate.log_loss(loss, flops))

    def log_learning_rate(self, learning_rate, flops) -> None:
        self._call(
            "clearml_learning_rate",
            lambda: self.delegate.log_learning_rate(learning_rate, flops),
        )

    def log_validation(self, loss, accuracy, flops) -> None:
        self._call(
            "clearml_validation",
            lambda: self.delegate.log_validation(loss, accuracy, flops),
        )

    def _validate_receipt(self, value, result) -> dict[str, object]:
        _require(self._config is not None and "parameters" in value, "CLEARML_CONFIG_CONTRACT")
        return validate_unsampled_readback(
            value,
            task_id=self.task_id,
            updates=self.updates,
            validations=self.validations,
            config=self._config,
            include_peak_gpu_bytes=result.speed.peak_gpu_bytes is not None,
        )

    def finish(self, result, model) -> None:
        self._call("clearml_finish", lambda: self.delegate.finish(result, model))
        with failure_phase("clearml_validate_receipt"):
            _require(
                self._proxy is not None and self._proxy.receipt is not None, "CLEARML_READBACK"
            )
            self.receipt = self._validate_receipt(self._proxy.receipt, result)


class CompletionTracker:
    """Keep the public tracker API while enforcing the final durable ordering."""

    def __init__(self, backup, service) -> None:
        self.backup, self.service = backup, service
        self.service.persist_readback = backup.record_scalar_readback
        self.task_id, self.receipt = None, None
        self.diagnostics = []

    def _service(self, method, *args):
        from training_diagnostics import observe

        from compare_attention.train.lifecycle import observe_failures

        def operation():
            # Auxiliary SDK failures must not consume the native primary-failure slot.
            with observe_failures(None):
                return getattr(self.service, method)(*args)

        observed = observe(method, operation)
        if observed["diagnostic"]["status"] == "warning":
            if not any(row["diagnostic"]["phase"] == method for row in self.diagnostics):
                self.diagnostics.append(observed)
        return observed

    def start(self, config) -> None:
        self.backup.start(config)
        self._service("start", config)
        self.task_id = self.service.task_id

    def record_failure(self, report) -> None:
        self.backup.record_failure(report)

    def record_identity(self, identity) -> None:
        with failure_phase("identity_write"):
            self.backup.record_identity(identity)

    def save_checkpoint(self, payload, step, keep_last) -> None:
        self.backup.save_checkpoint(payload, step, keep_last)

    def log_loss(self, loss, flops) -> None:
        self.backup.log_loss(loss, flops)
        self._service("log_loss", loss, flops)

    def log_learning_rate(self, learning_rate, flops) -> None:
        callback = getattr(self.backup, "log_learning_rate", None)
        if callable(callback):
            callback(learning_rate, flops)
        self._service("log_learning_rate", learning_rate, flops)

    def log_validation(self, loss, accuracy, flops) -> None:
        self.backup.log_validation(loss, accuracy, flops)
        self._service("log_validation", loss, accuracy, flops)

    def finish(self, result, model) -> None:
        self.backup.finish(result, model)
        self._service("finish", result, model)
        self.receipt = self.service.receipt
        self.record_identity(result.provenance)
