"""Ordinary NORMAL whole-object replay from request-pinned immutable observations."""

import hashlib
from pathlib import Path, PurePosixPath

from unified_normal_contract import canonical, read, require


def inference_observation(root, case, observed):
    """Revalidate a complete unsampled stored observation, without API or booleans alone."""
    from seed202_pilot.cloud_services import _scalars

    root = Path(root)
    envelope = read(root / "task-clearml-readback.json")
    report = read(root / "task-val.json")
    identity = read(root / "out/run_identity.json")
    require(
        set(envelope) == {"task_id", "status", "artifact_sha256", "fresh_sdk_readback"}
        and envelope["status"] == "completed"
        and envelope["fresh_sdk_readback"] is True,
        "UNIFIED_ORDINARY_INFERENCE_ENVELOPE",
    )
    raw = observed["raw"]
    expected = _scalars(report)
    require(
        type(raw) is dict
        and type(raw.get("total")) is int
        and type(raw.get("events")) is list
        and raw["total"] == len(raw["events"]) == len(expected),
        "UNIFIED_ORDINARY_INFERENCE_TOTAL",
    )
    require(
        all(
            type(row["iter"]) is int and type(row["value"]) in (int, float) for row in raw["events"]
        ),
        "UNIFIED_ORDINARY_INFERENCE_TYPES",
    )
    require(
        sorted((r["metric"], r["variant"], r["iter"], r["value"]) for r in raw["events"])
        == sorted(("quality", name, 0, value) for name, value in expected.items()),
        "UNIFIED_ORDINARY_INFERENCE_EVENTS",
    )
    binding = {
        "final_parameters_sha256": identity["final_parameters_sha256"],
        "config_sha256": hashlib.sha256(canonical(case["config"]).encode()).hexdigest(),
        "task": report["task"],
        "split": "val",
    }
    require(
        hashlib.sha256((root / "task-val.json").read_bytes()).hexdigest()
        == envelope["artifact_sha256"],
        "UNIFIED_ORDINARY_INFERENCE_ARTIFACT",
    )
    expected_observation = {
        **envelope,
        "metrics": expected,
        "binding": binding,
        "raw": raw,
        "fresh_sdk_readback": True,
        "remote_mutations": 0,
    }
    require(
        canonical(observed) == canonical(expected_observation), "UNIFIED_ORDINARY_INFERENCE_WHOLE"
    )
    return expected_observation


def replay(spec, recorded, inputs):
    """No credentials/API calls, saved model restore, optimizer or RNG restoration."""
    require(
        set(spec) == {"profile", "source_request", "result", "data", "snapshot", "index"}
        and spec["profile"] == "REPAIRED_NORMAL_V1",
        "UNIFIED_ORDINARY_PROFILE_SCHEMA",
    )
    candidate = Path(spec["source_request"]).parents[2]
    from seed202_pilot.cli import _candidate
    from unified_normal_source import operator_profile

    _candidate(candidate)
    requested = read(spec["source_request"])
    deployments = read(candidate / "PLAN-UNAPPROVED.json")["deployments"]
    deployment = next(row for row in deployments if row["case_id"] == requested["run"]["id"])
    require(
        operator_profile(candidate, deployment)["normal_authority_bound"],
        "UNIFIED_ORDINARY_SOURCE_AUTHORITY",
    )
    payload = candidate / "payload"
    package = payload / "normal-package"
    root, snapshot, data = (Path(spec[k]) for k in ("result", "snapshot", "data"))
    for directory in (payload, package, root, root.parent, snapshot, payload / "source/src", data):
        require(str(directory.resolve()) in inputs["roots"], "UNIFIED_ORDINARY_INPUT_ROOT")
        require(
            all(str(p.resolve()) in inputs["files"] for p in directory.rglob("*") if p.is_file()),
            "UNIFIED_ORDINARY_INPUT_CLOSURE",
        )
    require(
        all(
            str((candidate / name).resolve()) in inputs["files"]
            for name in (
                "candidate.json",
                "PLAN-UNAPPROVED.json",
                "sorting-mqar-seed202-pilot-v1.zip",
            )
        ),
        "UNIFIED_ORDINARY_CANDIDATE_PINS",
    )
    require(
        str(Path(spec["source_request"]).resolve()) in inputs["files"],
        "UNIFIED_ORDINARY_REQUEST_PIN",
    )
    from seed202_pilot.storage import inventory
    from unified_normal_contract import digest

    require(str(Path(spec["index"]).resolve()) in inputs["files"], "UNIFIED_ORDINARY_INDEX_PIN")
    index = read(spec["index"])
    require(
        root.name == "normal-" + deployment["nonce"]
        and inventory(root.parent) == index["inventory"],
        "UNIFIED_ORDINARY_DOWNLOAD_BINDING",
    )
    before, after = index["provider_before"], index["provider_after"]
    meta = read(payload / "deploy" / deployment["case_id"] / "kernel/kernel-metadata.json")
    fields = {
        "target",
        "version",
        "head",
        "private",
        "source_sha256",
        "requested_versioned_inputs",
        "numeric_id",
        "typed_status",
        "observed_bare_dto_inputs",
    }
    require(
        set(before) == fields
        and canonical(before) == canonical(after)
        and before["target"] == deployment["kernel_target"]
        and before["typed_status"] == "COMPLETE"
        and before["private"] is True
        and type(before["numeric_id"]) is int
        and before["numeric_id"] > 0
        and type(before["head"]) is int
        and before["head"] == 1
        and type(before["version"]) is int
        and before["version"] == 1
        and before["source_sha256"]
        == digest(payload / "deploy" / deployment["case_id"] / "kernel/kernel.py")
        and before["requested_versioned_inputs"] == meta["dataset_sources"]
        and len(before["observed_bare_dto_inputs"]) == len(meta["dataset_sources"])
        and set(before["observed_bare_dto_inputs"])
        == {v.rsplit("/", 1)[0] for v in meta["dataset_sources"]},
        "UNIFIED_ORDINARY_PROVIDER_OBSERVATIONS",
    )
    from unified_normal_host import declared_data_domains, sources, task_quality

    with sources(package):
        from normal_training_acceptance import verify_native_service, verify_sources
        from normal_training_native import validate_full, verify_native
        from normal_training_plan import derive
        from training_completion import validate_unsampled_readback

        from compare_attention.train.checkpoint import load_checkpoint

        case = derive(
            package,
            payload / "unified/normal-plan.json",
            requested["run"]["id"],
            payload / "configs" / (requested["run"]["id"] + ".yaml"),
            data,
        )
        require(case["engineering_only"] is False, "UNIFIED_ORDINARY_TARGET_ONLY")
        for index, domain in enumerate(declared_data_domains(case)):
            selected = {**case, "config": {**case["config"], "data": domain}}
            try:
                native = verify_native(root / "out", selected)
            except ValueError as error:
                if error.args != ("NORMAL_NATIVE_CONFIG",) or index + 1 == len(
                    declared_data_domains(case)
                ):
                    raise
            else:
                case = selected
                break
        source = verify_sources(package, root, case)
        task = task_quality(package, root, case, data)
        require(
            recorded["case"] == case
            and recorded["scope"] == "TARGET_NORMAL_FULL_TASK_LIFECYCLE"
            and recorded["status"] == "ACCEPTED_UNIFIED_NORMAL",
            "UNIFIED_ORDINARY_TARGET_SCOPE",
        )
        raw = read(root / "out/training-unsampled-readback.json")
        producer = read(root / "training-task.json")
        training = recorded["training"]
        require(
            set(training) == {"receipt", "raw", "fresh_sdk_readback"}
            and training["fresh_sdk_readback"] is True
            and set(producer) == {"schema", "case", "task_id", "name", "identity", "receipt"}
            and producer["schema"] == "normal-training-return-v1"
            and producer["case"] == case
            and producer["identity"] == read(root / "out/run_identity.json"),
            "UNIFIED_ORDINARY_PRODUCER_ENVELOPE",
        )
        require(read(root / "native-proof.json") == native, "UNIFIED_ORDINARY_NATIVE_PROOF")
        include_peak = producer["receipt"]["required_counts"]["result/peak_gpu_bytes"] == 1
        for observation in (raw, training["raw"]):
            receipt = validate_unsampled_readback(
                observation,
                task_id=producer["task_id"],
                updates=case["updates"],
                validations=len(case["validation_steps"]),
                config=case["config"],
                include_peak_gpu_bytes=include_peak,
            )
            require(
                receipt == producer["receipt"] == training["receipt"],
                "UNIFIED_ORDINARY_TRAINING_RECEIPT",
            )
            verify_native_service(root, observation)
        require(
            training["raw"]["events"] == raw["events"]
            and training["raw"]["parameters"] == raw["parameters"],
            "UNIFIED_ORDINARY_TRAINING_RAW_JOIN",
        )
        manifest = read(snapshot / "snapshot.json")
        require(
            set(manifest) == {"case", "publications", "files"} and manifest["case"] == case,
            "UNIFIED_ORDINARY_SNAPSHOT_CASE",
        )
        periodic = read(root / "periodic-readback.json")["publications"]
        final_envelope = read(root.parent / (root.name + "-publication.json"))
        require(
            final_envelope["schema"] == "normal-training-publication-v1"
            and final_envelope["case"] == case
            and [row["step"] for row in periodic] == case["checkpoint_steps"],
            "UNIFIED_ORDINARY_PUBLICATION_GRID",
        )
        rows = [*periodic, final_envelope["final"]]
        require(
            all(row["state"] == "verified" for row in rows)
            and manifest["publications"]
            == [{"revision": r["revision"], "files": r["files"]} for r in rows]
            and manifest["files"]
            == {
                str(PurePosixPath(r["revision"]) / name): pin
                for r in rows
                for name, pin in r["files"].items()
            },
            "UNIFIED_ORDINARY_PUBLICATION_BINDING",
        )
        require(
            {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()}
            == set(manifest["files"]) | {"snapshot.json"},
            "UNIFIED_ORDINARY_SNAPSHOT_INVENTORY",
        )
        for name, pin in manifest["files"].items():
            member = PurePosixPath(name)
            require(
                not member.is_absolute()
                and ".." not in member.parts
                and "\\" not in name
                and ":" not in name,
                "UNIFIED_ORDINARY_SNAPSHOT_PATH",
            )
            path = snapshot / name
            require(
                path.stat().st_size == pin["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"],
                "UNIFIED_ORDINARY_SNAPSHOT_BYTES",
            )
            if "checkpoints" in path.parts and path.suffix == ".pt":
                full = load_checkpoint(path)
                require(full["step"] in case["checkpoint_steps"], "UNIFIED_ORDINARY_FULL_GRID")
                validate_full(full, case, step=full["step"])
        final = final_envelope["final"]
        require(
            set(final["files"])
            == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()},
            "UNIFIED_ORDINARY_FINAL_INVENTORY",
        )
        for name, pin in final["files"].items():
            require(
                (root / name).stat().st_size == pin["bytes"]
                and hashlib.sha256((root / name).read_bytes()).hexdigest() == pin["sha256"],
                "UNIFIED_ORDINARY_FINAL_BYTES",
            )
        publication = {
            "publications": manifest["publications"],
            "fresh_sdk_readback": True,
            "remote_mutations": 0,
            "snapshot": str(snapshot.resolve()),
            "files": manifest["files"],
        }
        lifecycle = {
            "schema": "normal-training-lifecycle-acceptance-v1",
            "case": case,
            "scope": "TARGET_NORMAL_LIFECYCLE",
            "status": "ACCEPTED_NORMAL_LIFECYCLE",
            "source": source,
            "native": native,
            "training": training["receipt"],
            "immutable_publications": publication["publications"],
            "task_quality_acceptance": "SEPARATE_TASK_SPECIFIC_ACCEPTANCE_REQUIRED",
            "next_launch_authorized": False,
        }
        inference = inference_observation(root, case, recorded["inference"])
        expected = {
            "schema": "unified-normal-whole-acceptance-v1",
            "case": case,
            "scope": "TARGET_NORMAL_FULL_TASK_LIFECYCLE",
            "source": source,
            "native": native,
            "training": training,
            "task": task,
            "inference": inference,
            "publication": publication,
            "remote_mutations": 0,
            "next_launch_authorized": False,
            "lifecycle": lifecycle,
            "status": "ACCEPTED_UNIFIED_NORMAL",
            "whole_lifecycle_sha256": hashlib.sha256(canonical(lifecycle).encode()).hexdigest(),
        }
        require(canonical(recorded) == canonical(expected), "UNIFIED_ORDINARY_WHOLE_OBJECT")
    return expected
