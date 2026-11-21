"""One target-only task evaluation between NORMAL return and final publication."""

import hashlib
import json
import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from normal_launch_diagnostics import NormalArgumentParser, normal_cli


def publish_tracking(args, case, report):
    """Publish auxiliary inference telemetry without gating saved native quality."""
    from training_diagnostics import observe

    def operation():
        from quality_inherit import leaf

        services = leaf("services")
        values = services.credentials(args.private, rank=0)
        from seed202_pilot.cloud_services import publish_task

        envelope = json.loads((args.output / "training-task.json").read_bytes())
        service = services.ServiceTracker(values, envelope["name"])
        provenance = json.loads((args.output / "out/run_identity.json").read_bytes())
        return publish_task(
            service, report, case["config"], provenance, args.output / "task-val.json"
        )

    observed = observe("normal_task_tracking", operation)
    if observed["diagnostic"]["status"] == "verified":
        return {key: value for key, value in observed.items() if key != "diagnostic"}
    return observed


def main(argv=None):
    parser = NormalArgumentParser(allow_abbrev=False)
    for name in ("package", "plan", "recipe", "data", "private", "output", "review", "decision"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--phase", choices=("task",), required=True)
    args = parser.parse_args(argv)
    sys.path[:0] = [
        str(args.package / "src"),
        str(args.package / "inherited/operator"),
        str(args.package / "inherited/operator/inherited"),
        str(args.package / "inherited/tools"),
    ]
    from normal_training_plan import (
        create_json,
        derive,
        execution_authority,
        require,
        source_identity,
    )

    identity = source_identity(args.package)
    require(
        not any(
            n == "compare_attention" or n.startswith("compare_attention.") for n in sys.modules
        ),
        "UNIFIED_TASK_FRESH_SOURCE_REQUIRED",
    )
    wheel = args.package / "compare_attention-0.1.0-py3-none-any.whl"
    require(
        hashlib.sha256(wheel.read_bytes()).hexdigest() == identity["wheel_sha256"],
        "UNIFIED_TASK_WHEEL",
    )
    root = Path(os.environ["TMPDIR"]) / ("normal-task-wheel-" + uuid.uuid4().hex)
    with zipfile.ZipFile(wheel) as saved:
        names = saved.namelist()
        require(len(names) == len(set(names)), "UNIFIED_TASK_UNIQUE_WHEEL")
        for name in names:
            part = PurePosixPath(name)
            require(
                not part.is_absolute()
                and ".." not in part.parts
                and "\\" not in name
                and ":" not in name,
                "UNIFIED_TASK_SAFE_WHEEL",
            )
        saved.extractall(root)
    try:
        sys.path.insert(0, str(root))
        case = derive(args.package, args.plan, args.case, args.recipe, args.data)
        if case["engineering_only"]:
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "UNIFIED_TASK_ENGINEERING_CPU")
            create_json(
                args.output / "task-engineering.json",
                {
                    "scope": "SYNTHETIC_NORMAL_LIFECYCLE_NOT_TASK_QUALITY",
                    "case": case,
                    "task_evaluation_executed": False,
                    "scientific_acceptance": False,
                    "next_launch_authorized": False,
                },
            )
            return 0
        execution_authority(case, args.review, args.decision)
        from quality_inherit import leaf

        leaf("runtime").guard()
        sys.path.insert(1, str(args.package / "inherited-source"))
        from normal_training_native import verify_native
        from normal_training_worker import imported_sources
        from seed202_pilot.cloud_evaluate import evaluate_target
        from seed202_pilot.contract import require_authorized_case

        proof = verify_native(args.output / "out", case)
        spec = require_authorized_case(case["id"])
        report = evaluate_target(args.output / "out/model.pt", args.data, spec)
        require(
            report["checkpoint"]["parameters_sha256"] == proof["final_sha256"],
            "UNIFIED_TASK_NORMAL_WEIGHTS_JOIN",
        )
        create_json(args.output / "task-val.json", report)
        create_json(args.output / "task-source.json", imported_sources(args.package))
        readback = publish_tracking(args, case, report)
        create_json(args.output / "task-clearml-readback.json", readback)
    finally:
        shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(normal_cli(main, "task"))
