"""Pre-authentication command-wide barrier and held review/decision byte leases."""

import contextlib
import json
from pathlib import Path

from seed202_pilot.host_resume import exact, require

MODE = "MQAR_CLASSIC_STEP980_V1"


@contextlib.contextmanager
def command_scope(args, approval):
    from seed202_pilot.host_actions import acknowledged
    from seed202_pilot.host_actions import command_scope as inherited_scope
    from seed202_pilot.prepared_files import lease

    plan = approval["plan"] if approval else json.loads(Path(args.plan).read_bytes())
    if plan["deployment"].get("mode") != MODE:
        with inherited_scope(args, approval):
            yield
        return
    require(args.plan is not None and args.review is not None, "EXPLICIT_REVIEWED_PLAN_FILES")
    files = [args.plan, args.review]
    for name in ("approval", "preparation", "lifetime"):
        path = getattr(args, name, None)
        if path is not None:
            files.append(path)
    with lease(files):
        require(exact(json.loads(Path(args.plan).read_bytes()), plan), "HELD_PLAN_BYTES")
        review = json.loads(Path(args.review).read_bytes())
        if approval:
            require(
                exact(json.loads(Path(args.approval).read_bytes()), approval),
                "HELD_EXECUTION_DECISION_BYTES",
            )
            require(exact(review, approval["review"]), "HELD_INDEPENDENT_REVIEW_BYTES")
        with inherited_scope(args, approval):
            for action in ("publish", "publish-resume", "submit"):
                journal = Path(plan["journals"][action])
                if not (journal / "intent.json").exists():
                    continue
                authority = json.loads((journal / "approval.json").read_bytes())["source_authority"]
                require(exact(authority["plan"], plan), "COMMAND_WIDE_ORIGINAL_AUTHORITY")
                if not acknowledged(authority, action):
                    require(
                        args.command == "reconcile" and args.action == action,
                        "COMMAND_WIDE_PENDING_UNKNOWN_GET_ONLY",
                    )
            yield
