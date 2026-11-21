"""Saved NORMAL worker: public parser/train, real native callbacks, closed services."""

import hashlib
import json
import os
import sys
from pathlib import Path

from normal_launch_diagnostics import NormalArgumentParser, normal_context, normal_entry
from normal_training_native import validate_full, verify_native
from normal_training_plan import (
    canonical,
    create_json,
    derive,
    digest,
    execution_authority,
    require,
)


def imported_sources(package):
    """Read back actual imported scientific members, not a declared source list."""
    import zipfile

    package = Path(package)
    wheel = package / "compare_attention-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as saved:
        pins = {name: hashlib.sha256(saved.read(name)).hexdigest() for name in saved.namelist()}
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name == "compare_attention" or name.startswith("compare_attention."):
            path = Path(module.__file__)
            member = name.replace(".", "/") + (
                "/__init__.py" if path.name == "__init__.py" else ".py"
            )
            actual = digest(path)
            require(actual == pins[member], "NORMAL_IMPORTED_SOURCE_BYTES")
            modules[name] = {"member": member, "sha256": actual}
    require(
        {"compare_attention.train.loop", "compare_attention.train.distributed_checkpoint"}
        <= set(modules),
        "NORMAL_IMPORTED_TRAINING_MODULES",
    )
    return {
        "wheel_sha256": digest(wheel),
        "imported_modules": modules,
        "no_training_method_replacement": True,
    }


def worker(args, case, values):
    from quality_inherit import leaf
    from quality_recovery import RecoveryPublisher
    from quality_services import HFTransport
    from quality_tracker import QualityTracker
    from training_completion import CompletionTracker, TrainingServiceTracker

    from compare_attention.train.config import TrainConfig
    from compare_attention.train.loop import train
    from compare_attention.train.tracker import DisabledTracker

    rank = int(os.environ["RANK"])
    require(
        int(os.environ["WORLD_SIZE"]) == 2
        and int(os.environ["LOCAL_RANK"]) == rank
        and rank in (0, 1),
        "NORMAL_DDP_RANKS",
    )
    if case["engineering_only"]:
        from compare_attention.train.identity import runtime_metadata

        execution = {"scope": "CPU_ENGINEERING_ONLY", **runtime_metadata()}
    else:
        execution = leaf("runtime").runtime_attestation()
        require(
            hashlib.sha256(canonical(execution).encode()).hexdigest() == case["runtime_sha256"],
            "NORMAL_EXACT_TARGET_RUNTIME",
        )
    root = args.output
    if rank == 0:
        require(not (root / "out").exists(), "NORMAL_CREATE_ONLY_RESULT")
        root.mkdir(parents=True, exist_ok=True)
        publisher = RecoveryPublisher(
            root / "periodic.jsonl", args.prefix + "/periodic", HFTransport(values)
        )
        # Explicit current validator: never use the inherited captured default.
        backup = QualityTracker(root / "out", case, publisher, validate=validate_full)

        class Service(TrainingServiceTracker):
            @staticmethod
            def _production_task(name):
                from clearml import Task

                hooks = (
                    "matplotlib",
                    "tensorflow",
                    "tensorboard",
                    "pytorch",
                    "xgboost",
                    "scikit",
                    "fastai",
                    "lightgbm",
                    "hydra",
                    "detect_repository",
                    "tfdefines",
                    "joblib",
                    "megengine",
                    "catboost",
                    "gradio",
                )
                return Task.init(
                    project_name="compare-attention",
                    task_name=name,
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False,
                    auto_connect_frameworks=dict.fromkeys(hooks, False),
                    auto_connect_arg_parser=False,
                    auto_connect_streams=False,
                )

        service = Service(
            values,
            args.prefix.replace("/", "-"),
            updates=case["updates"],
            validations=len(case["validation_steps"]),
        )
        tracker = CompletionTracker(backup, service)
    else:
        tracker = DisabledTracker()
    result = train(TrainConfig.model_validate(case["config"]), tracker, distributed=True)
    create_json(root / f"source-rank-{rank}.json", imported_sources(args.package))
    create_json(
        root / f"rank-{rank}.json",
        {
            "rank": rank,
            "identity": result.provenance,
            "execution_runtime": execution,
            "engineering_only": case["engineering_only"],
        },
    )
    if rank == 0:
        require(
            result.provenance["status"] == "completed" and result.stop_reason == "budget_exhausted",
            "NORMAL_RETURN_IDENTITY",
        )
        create_json(
            root / "training-task.json",
            {
                "schema": "normal-training-return-v1",
                "case": case,
                "task_id": service.task_id,
                "name": service.name,
                "identity": result.provenance,
                "receipt": service.receipt,
            },
        )
        publisher.complete(case["checkpoint_steps"])
        create_json(root / "training-diagnostics.json", {"warnings": tracker.diagnostics})


def finish(args, case, values):
    from normal_training_acceptance import training_readback, verify_sources
    from quality_recovery import RecoveryPublisher
    from quality_services import HFTransport

    root = args.output
    verify_sources(args.package, root, case)
    proof = verify_native(root / "out", case)
    if not case["engineering_only"]:
        from unified_normal_host import task_quality

        task_quality(args.package, root, case, args.data)
    create_json(root / "native-proof.json", proof)
    fresh = training_readback(root, case, values)
    create_json(root / "training-fresh-readback.json", fresh)
    transport = HFTransport(values)
    periodic = RecoveryPublisher(
        root / "periodic.jsonl", args.prefix + "/periodic", transport
    ).complete(case["checkpoint_steps"])
    for row in periodic:
        observed = transport.read(row, row["revision"])
        require(
            observed["private"] is True
            and observed["revision"] == row["revision"]
            and set(observed["files"]) == set(row["files"]),
            "NORMAL_PERIODIC_READBACK",
        )
        for name, raw in observed["files"].items():
            require(
                len(raw) == row["files"][name]["bytes"]
                and hashlib.sha256(raw).hexdigest() == row["files"][name]["sha256"],
                "NORMAL_PERIODIC_ALL_BYTES",
            )
    create_json(root / "periodic-readback.json", {"publications": periodic})
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    final = RecoveryPublisher(
        root.parent / (root.name + "-final.jsonl"), args.prefix + "/final", transport
    )
    receipt = final.publish(case["updates"], files)
    create_json(
        root.parent / (root.name + "-publication.json"),
        {"schema": "normal-training-publication-v1", "case": case, "final": receipt},
    )


def _execute(args, case, values):
    if args.phase == "worker":
        worker(args, case, values)
    elif args.phase == "finish":
        finish(args, case, values)
    else:
        from normal_training_acceptance import accept

        accept(args.package, args.output, case, values)
    return 0


def main(argv=None):
    """No default seed/case/source, no resume mode and no automatic following run."""
    parser = NormalArgumentParser(allow_abbrev=False)
    for name in ("package", "plan", "recipe", "data", "private", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("case", "prefix"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--phase", choices=("worker", "finish", "accept"), required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args(argv)
    case = derive(args.package, args.plan, args.case, args.recipe, args.data)
    if case["engineering_only"]:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
            "NORMAL_CPU_ENGINEERING_ENV",
        )
    elif args.phase != "accept":
        execution_authority(case, args.review, args.decision)
        from quality_inherit import leaf

        leaf("runtime").guard()
    require(args.prefix.startswith("normal-training/"), "NORMAL_DISTINCT_NAMESPACE")
    from quality_inherit import leaf

    services = leaf("services")
    rank = int(os.environ.get("RANK", "0"))
    values = services.credentials(args.private, rank=0) if rank == 0 else None
    if normal_context() is not None:
        return normal_entry(lambda: _execute(args, case, values), args.phase)
    try:
        _execute(args, case, values)
    except Exception as error:  # pylint: disable=broad-exception-caught
        from compare_attention.train.lifecycle import primary_failure

        path = args.output / (
            "lifecycle-failure.json" if rank == 0 else f"lifecycle-failure-rank-{rank}.json"
        )
        # Prefer the primary recorded before DDP/service conversion; no raw text.
        original = (
            args.output
            / "out"
            / ("lifecycle-failure.json" if rank == 0 else f"lifecycle-failure-rank-{rank}.json")
        )
        report = (
            json.loads(original.read_bytes())
            if original.is_file()
            else primary_failure(error, args.phase)
        )
        try:
            if not path.exists():
                create_json(path, report)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        print(canonical(report), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
