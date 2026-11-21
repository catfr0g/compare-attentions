"""Hard native/private-HF acceptance with auxiliary, nonblocking ClearML diagnostics."""

import hashlib
import json
import zipfile
from itertools import groupby
from pathlib import Path

from normal_training_native import verify_native
from normal_training_plan import canonical, create_json, digest, require, source_identity


def verify_sources(package, root, case):
    """Both actual source/rank envelopes must bind the NEW wheel and exact recipe."""
    identity = source_identity(package)
    require(
        case["source_identity_sha256"] == digest(Path(package) / "normal-source-identity.json"),
        "NORMAL_SOURCE_PLAN",
    )
    with zipfile.ZipFile(Path(package) / "compare_attention-0.1.0-py3-none-any.whl") as wheel:
        pins = {name: hashlib.sha256(wheel.read(name)).hexdigest() for name in wheel.namelist()}
    final = json.loads((Path(root) / "out/run_identity.json").read_bytes())
    for rank in (0, 1):
        record = json.loads((Path(root) / f"source-rank-{rank}.json").read_bytes())
        require(
            set(record) == {"wheel_sha256", "imported_modules", "no_training_method_replacement"}
            and record["wheel_sha256"] == identity["wheel_sha256"]
            and record["no_training_method_replacement"] is True,
            "NORMAL_SOURCE_RECORD",
        )
        modules = record["imported_modules"]
        require(
            {
                "compare_attention.train.loop",
                "compare_attention.train.distributed_checkpoint",
                "compare_attention.train.lifecycle",
            }
            <= set(modules),
            "NORMAL_SOURCE_REQUIRED_MODULES",
        )
        for name, row in modules.items():
            require(
                name == "compare_attention" or name.startswith("compare_attention."),
                "NORMAL_SOURCE_NAMESPACE",
            )
            require(row["sha256"] == pins[row["member"]], "NORMAL_SOURCE_IMPORTED_PIN")
        evidence = json.loads((Path(root) / f"rank-{rank}.json").read_bytes())
        value = evidence["identity"]
        require(
            evidence["rank"] == rank
            and evidence["engineering_only"] == case["engineering_only"]
            and value["config"] == case["config"]
            and value["requested_seed"] == value["effective_seed"] == case["seed"]
            and value["status"] == "completed"
            and value["final_parameters_sha256"] == final["final_parameters_sha256"],
            "NORMAL_RANK_ENVELOPE",
        )
        if not case["engineering_only"]:
            from quality_inherit import native_checks

            checks = native_checks()
            checks.target_runtime(value["runtime"], rank)  # pylint: disable=no-member
            checks.execution_runtime_evidence(evidence["execution_runtime"])  # pylint: disable=no-member
            require(
                hashlib.sha256(canonical(evidence["execution_runtime"]).encode()).hexdigest()
                == case["runtime_sha256"],
                "NORMAL_RANK_EXACT_RUNTIME",
            )
    return {"wheel_sha256": identity["wheel_sha256"], "source_rank_records": 2}


def verify_native_service(root, raw):
    """Join every scientific scalar value/axis to the real native producer history."""
    from training_completion import resource_event

    rows = [
        json.loads(line) for line in (Path(root) / "out/metrics.jsonl").read_text().splitlines()
    ]
    expected = []
    for row in rows[:-1]:
        if row["type"] == "step":
            expected.append(("train", "loss", row["flops"], row["loss"]))
        elif row["type"] == "learning_rate":
            expected.append(("train", "learning_rate", row["flops"], row["learning_rate"]))
        elif row["type"] == "validation":
            for name in ("loss", "accuracy"):
                expected.append(("validation", name, row["flops"], row[name]))
        else:
            require(False, "NORMAL_NATIVE_SERVICE_VALUES")
    result = rows[-1]
    budget = result["config"]["schedule"]["train_flops"]
    values = {
        "steps": float(result["steps"]),
        "train_flops": float(result["train_flops"]),
        "requested_train_flops": float(budget),
        "early_stopped": float(result["stop_reason"] == "early_stopping"),
        "budget_exhausted": float(result["train_flops"] >= budget),
        **result["quality"],
        "train_tokens_per_second": result["speed"]["train_tokens_per_second"],
        "peak_gpu_bytes": result["speed"]["peak_gpu_bytes"],
    }
    expected.extend(
        ("result", name, result["steps"], value)
        for name, value in values.items()
        if value is not None
    )
    actual = [
        (row["metric"], row["variant"], row["iter"], row["value"])
        for row in raw["events"]
        if not resource_event(row)
    ]
    require(sorted(actual) == sorted(expected), "NORMAL_NATIVE_SERVICE_VALUES")


def _event_comparison(events):
    """Ignore only order within contiguous equal-iteration groups, never raw evidence.

    The SDK requests ascending events without a tie-breaker. Keep group order,
    every full row (including timestamps/resources), and duplicate multiplicity.
    This is a comparison view, not a replacement for unsampled validation.
    """
    return [
        (axis, sorted(rows, key=canonical)) for axis, rows in groupby(events, lambda r: r["iter"])
    ]


def training_readback(root, case, values):
    """Observe tracking independently; service failure cannot reject native science."""
    from training_diagnostics import observe

    state = {"receipt": None, "raw": None, "fresh_sdk_readback": False}
    return observe(
        "normal_host_training_readback", lambda: _readback(root, case, values, state), state
    )


def _readback(root, case, values, state):
    """Strict diagnostic predicate; raw evidence survives even a failed predicate."""
    from quality_inherit import leaf
    from training_completion import TrainingServiceTracker, validate_unsampled_readback

    root = Path(root)
    envelope = json.loads((root / "training-task.json").read_bytes())
    identity = json.loads((root / "out/run_identity.json").read_bytes())
    require(
        set(envelope) == {"schema", "case", "task_id", "name", "identity", "receipt"}
        and envelope["schema"] == "normal-training-return-v1"
        and envelope["case"] == case
        and envelope["identity"] == identity
        and identity["status"] == "completed",
        "NORMAL_RETURN_ENVELOPE",
    )
    raw = json.loads((root / "out/training-unsampled-readback.json").read_bytes())
    include_peak = envelope["receipt"]["required_counts"]["result/peak_gpu_bytes"] == 1
    expected = validate_unsampled_readback(
        raw,
        task_id=envelope["task_id"],
        updates=case["updates"],
        validations=len(case["validation_steps"]),
        config=case["config"],
        include_peak_gpu_bytes=include_peak,
    )
    require(expected == envelope["receipt"] and "parameters" in raw, "NORMAL_RETURN_RECEIPT")
    services = leaf("services")

    def operation():
        with services.service_env(values, "track"):
            return TrainingServiceTracker._production_readback(envelope["task_id"])  # pylint: disable=protected-access

    fresh = services.safe("normal_host_training_readback", operation)
    state.update(raw=fresh, fresh_sdk_readback=True)
    from training_diagnostics import validate_pages

    validate_pages(fresh)
    receipt = validate_unsampled_readback(
        fresh,
        task_id=envelope["task_id"],
        updates=case["updates"],
        validations=len(case["validation_steps"]),
        config=case["config"],
        include_peak_gpu_bytes=include_peak,
    )
    require("parameters" in fresh, "NORMAL_FRESH_PARAMETERS")
    verify_native_service(root, fresh)
    return {
        "receipt": receipt,
        "raw": fresh,
        "fresh_sdk_readback": True,
        "diagnostic": {"status": "verified", "authority": "AUXILIARY_ONLY"},
    }


def _training_comparison(value):
    """Change only raw.events in a shallow comparison view; all metadata stays exact."""
    return {**value, "raw": {**value["raw"], "events": _event_comparison(value["raw"]["events"])}}


def accept(package, root, case, values):
    """Host whole NORMAL-lifecycle acceptance; engineering is never GPU/quality acceptance."""
    from quality_services import HFTransport

    root = Path(root)
    source = verify_sources(package, root, case)
    native = verify_native(root / "out", case)
    require(
        native == json.loads((root / "native-proof.json").read_bytes()), "NORMAL_HOST_NATIVE_DRIFT"
    )
    training = training_readback(root, case, values)
    # Tracking snapshots remain diagnostic; native and private publication stay hard.
    publication = json.loads((root.parent / (root.name + "-publication.json")).read_bytes())
    require(
        publication["case"] == case and publication["schema"] == "normal-training-publication-v1",
        "NORMAL_FINAL_ENVELOPE",
    )
    final = publication["final"]
    transport = HFTransport(values)
    periodic = json.loads((root / "periodic-readback.json").read_bytes())["publications"]
    require([r["step"] for r in periodic] == case["checkpoint_steps"], "NORMAL_HOST_PERIODIC_GRID")
    require(
        len({r["revision"] for r in periodic}) == len(periodic), "NORMAL_HOST_PERIODIC_REVISIONS"
    )
    remote_pins = []
    for row in [*periodic, final]:
        require(row["state"] == "verified", "NORMAL_HOST_PUBLICATION_STATUS")
        observed = transport.read(row, row["revision"])
        require(
            observed["private"] is True
            and observed["revision"] == row["revision"]
            and set(observed["files"]) == set(row["files"]),
            "NORMAL_HOST_HF_LISTING",
        )
        for name, raw in observed["files"].items():
            require(
                type(raw) is bytes
                and len(raw) == row["files"][name]["bytes"]
                and hashlib.sha256(raw).hexdigest() == row["files"][name]["sha256"],
                "NORMAL_HOST_HF_BYTES",
            )
            if row is final:
                require(raw == (root / name).read_bytes(), "NORMAL_HOST_FINAL_LOCAL_BYTES")
        remote_pins.append({"revision": row["revision"], "files": row["files"]})
    expected_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(set(final["files"]) == expected_files, "NORMAL_HOST_FINAL_FULL_INVENTORY")
    accepted = {
        "schema": "normal-training-lifecycle-acceptance-v1",
        "case": case,
        "scope": "ENGINEERING_CPU_SDK_FIXTURE_ONLY"
        if case["engineering_only"]
        else "TARGET_NORMAL_LIFECYCLE",
        "status": "ACCEPTED_NORMAL_LIFECYCLE",
        "source": source,
        "native": native,
        "training": training["receipt"],
        "immutable_publications": remote_pins,
        "task_quality_acceptance": "SEPARATE_TASK_SPECIFIC_ACCEPTANCE_REQUIRED",
        "next_launch_authorized": False,
    }
    create_json(root.parent / (root.name + "-normal-accepted.json"), accepted)
    return accepted


def next_witness(accepted, task_quality, future_decision):
    """A lifecycle receipt alone cannot admit a next scientific case/allocation."""
    require(
        accepted["scope"] == "TARGET_NORMAL_LIFECYCLE"
        and accepted["status"] == "ACCEPTED_NORMAL_LIFECYCLE",
        "NORMAL_TARGET_ACCEPTANCE_REQUIRED",
    )
    require(
        task_quality["status"] == "ACCEPTED"
        and task_quality["case"] == accepted["case"]
        and task_quality["final_sha256"] == accepted["native"]["final_sha256"],
        "NORMAL_SEPARATE_TASK_QUALITY",
    )
    require(
        future_decision["approved"] is True
        and future_decision["previous_acceptance_sha256"]
        == hashlib.sha256(canonical(accepted).encode()).hexdigest(),
        "NORMAL_SEPARATE_NEXT_DECISION",
    )
    return {
        "scope": "NEXT_DECISION_WITNESS_ONLY_NO_LAUNCH",
        "previous_case": accepted["case"]["id"],
        "decision": future_decision,
    }
