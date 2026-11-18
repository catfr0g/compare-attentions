"""ENGINEERING ONLY completion bytes from actual native/service producers."""

import importlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


def close(
    root,
    case,
    identity,
    quality,
    snapshot,
    rows,
    config,
    fixture,
    candidate,
    worker_args,
    monkeypatch,
):
    import numpy as np
    from quality_inherit import leaf
    from seed202_mqar_clearml_fixture import ClearMLFixture
    from seed202_pilot.cloud_case import normal_return_envelope
    from seed202_pilot.cloud_completion import TrainingServiceTracker
    from seed202_pilot.cloud_evaluate import mqar_report_from_predictions
    from seed202_pilot.contract import require_authorized_case
    from seed202_pilot.host_services import validate_training_receipts
    from seed202_pilot.mqar_authority import pin
    from seed202_pilot.mqar_continuation import provenance
    from seed202_pilot.mqar_restore_gate import FIELDS
    from seed202_pilot.storage import canonical_bytes, create_json, inventory

    from compare_attention.train.result import Quality, Speed, TrainResult

    spec = require_authorized_case(case["id"])
    wheel = candidate / "payload/inherited/training-wheel.whl"
    importlib.import_module("compare_attention.train.loop")
    importlib.import_module("compare_attention.train.distributed_checkpoint")

    wheel_root = Path(sys.modules["compare_attention"].__file__).resolve().parent.parent
    with zipfile.ZipFile(wheel) as archive:
        pins = {
            name: pin(archive.read(name))["sha256"]
            for name in archive.namelist()
            if name.startswith("compare_attention/")
        }
    imported = leaf("paired_worker").imported(wheel_root, pins)
    for rank in (0, 1):
        source_path = root / f"source-rank-{rank}.json"
        if not source_path.exists():
            create_json(
                source_path,
                {
                    "wheel_sha256": pin(wheel.read_bytes())["sha256"],
                    "imported_modules": imported,
                    "no_training_method_replacement": True,
                },
            )
        receipt_path = root / f"restore-equality-rank-{rank}.json"
        receipt = {
            "scope": "ACTUAL_TARGET_NATIVE_RESTORE_EQUALITY_BEFORE_981",
            "rank": rank,
            "step": 980,
            "next_step": 981,
            "fields": list(FIELDS),
            "native_restore_unchanged": True,
            "rng_checked_after_history_replay": True,
        }
        if not receipt_path.exists():
            create_json(receipt_path, receipt)
        else:
            assert json.loads(receipt_path.read_bytes()) == receipt
    create_json(root / "resume-provenance.json", provenance())
    deploy = candidate / "payload/deploy" / case["id"]
    request = next(
        row
        for row in json.loads((candidate / "PLAN-UNAPPROVED.json").read_bytes())["deployments"]
        if row["case_id"] == case["id"]
    )
    (root / "mounted-authority").mkdir()
    for name in request["mounted_authority_files"]:
        shutil.copyfile(deploy / "resume-input" / name, root / "mounted-authority" / name)
    clearml_fixture = ClearMLFixture(monkeypatch, root.parent / "ENGINEERING_ONLY_clearml")
    service = TrainingServiceTracker(
        {},
        "ENGINEERING_ONLY_mqar_full",
        updates=10780,
        validations=11,
    )
    service.start(config)
    metrics = [json.loads(line) for line in (root / "out/metrics.jsonl").read_text().splitlines()]
    for event in metrics[:-1]:
        if event["type"] == "step":
            service.log_loss(event["loss"], event["flops"])
        else:
            service.log_validation(event["loss"], event["accuracy"], event["flops"])
    result = TrainResult(
        steps=10780,
        train_flops=case["actual_flops"],
        quality=Quality(**quality),
        speed=Speed(train_tokens_per_second=1.0, peak_gpu_bytes=1),
        config=config,
        provenance=identity,
        stop_reason="budget_exhausted",
    )
    service.finish(result, None)
    create_json(root / "training-task.json", normal_return_envelope(case["id"], result, service))
    create_json(root / "training-readback.json", {**service.receipt, "fresh_sdk_readback": True})
    assert validate_training_receipts(root) == service.receipt
    assert len(clearml_fixture.pages) == 2
    # Read the genuine immutable source dataset; do not duplicate arrays or alter pins.
    data = Path(
        "E:/HermesArtifacts/compare-attention/fp16-k64-small/"
        "272c2da7a9ef619711e4e92a1a98400ac0ed64b08f481d13752093d450820975/data"
    )
    labels = np.load(data / "val_labels.npy", mmap_mode="r")
    report = mqar_report_from_predictions(
        np.where(labels >= 0, labels, 0), labels, spec, identity, evidence_scope="TARGET_GPU_RESULT"
    )
    # Execute the actual finish consumer. Only target-only evaluation and SDK task
    # transport are synthetic; native/history/source/service/publication code runs.
    for name in ("native-proof.json", "training-readback.json", "periodic-readback.json"):
        path = root / name
        shutil.copyfile(path, root.parent / ("ENGINEERING_ONLY_before_finish_" + name))
        path.unlink()
    from seed202_pilot import cloud_evaluate, cloud_native, mqar_host_native
    from seed202_pilot.inherit import read_parent_member
    from seed202_pilot.mqar_package_guard import install, restore
    from seed202_pilot.mqar_resume_case import finish
    from seed202_pilot.science import _NATIVE, _module

    bridge = json.loads((candidate / "payload/qualified-launch/bridge.json").read_bytes())
    worker_args.science, worker_args.mqar = Path(bridge["science"]), data
    previous = install(worker_args.science)
    prior_native = sys.modules.get("quality_native_v2")
    try:
        sys.modules["quality_native_v2"] = _module(
            "quality_native_v2",
            read_parent_member(candidate / "payload/inherited/parent-candidate03.zip.bin", _NATIVE),
        )
        values = {"HF_TOKEN": "hf_ENGINEERING_ONLY", "HF_REPO": "engineering/mqar"}
        values.update(
            {
                name: "ENGINEERING_ONLY"
                for name in (
                    "CLEARML_API_ACCESS_KEY",
                    "CLEARML_API_SECRET_KEY",
                    "CLEARML_API_HOST",
                    "CLEARML_WEB_HOST",
                    "CLEARML_FILES_HOST",
                )
            }
        )
        with monkeypatch.context() as boundary:
            import quality_inherit

            original_leaf = quality_inherit.leaf
            services_module = original_leaf("services")
            boundary.setattr(services_module, "credentials", lambda *_args, **_kwargs: values)
            boundary.setattr(
                quality_inherit,
                "leaf",
                lambda name: services_module if name == "services" else original_leaf(name),
            )
            boundary.setattr(cloud_native, "bind_native", mqar_host_native.bind_native)
            boundary.setattr(cloud_evaluate, "evaluate_target", lambda *_args: report)
            finish(worker_args, case)
    finally:
        restore(previous)
        if prior_native is None:
            sys.modules.pop("quality_native_v2", None)
        else:
            sys.modules["quality_native_v2"] = prior_native
    receipt_path = root.parent / (root.name + "-worker-accepted.json")
    publication = json.loads(receipt_path.read_bytes())["final_publication"]
    assert len(clearml_fixture.pages) == 4
    assert (
        json.loads((root / "task-clearml-readback.json").read_bytes())["fresh_sdk_readback"] is True
    )
    assert validate_training_receipts(root)["event_count"] == 10812
    for name in publication["files"]:
        target = snapshot / "final" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fixture.files[publication["prefix"] + "/" + name])
    (snapshot / "snapshot.json").write_bytes(
        canonical_bytes(
            {"periodic": {row["revision"]: row for row in rows}, "final_publication": publication}
        )
        + b"\n"
    )
    services = root.parent.parent.parent / "ENGINEERING_ONLY_services.json"
    create_json(
        services,
        {
            "schema_version": 2,
            "scope": "FRESH_REMOTE_SERVICE_SNAPSHOT",
            "snapshot": str(snapshot),
            "snapshot_inventory": inventory(snapshot),
            "periodic_revisions": 11,
            "final_all_bytes": True,
            "clearml": {"status": "completed", "engineering_fixture": True},
        },
    )
    return data, services, labels, clearml_fixture
