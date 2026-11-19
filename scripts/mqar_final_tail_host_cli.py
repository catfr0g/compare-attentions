"""Public tail-only host entry: frozen plan -> two-stage proof -> seal -> audited SDK.

No default command mutates. External review and execution decision must already exist.
"""

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from mqar_final_tail_host_contract import activate, bind, exact, validate_plan  # noqa: E402
from mqar_final_tail_native import require  # noqa: E402
from mqar_final_tail_runner import create_json  # noqa: E402


def load(path):
    return json.loads(Path(path).read_bytes())


def _provider(args):
    import importlib.metadata

    require(
        importlib.metadata.version("kaggle") == "2.2.5"
        and importlib.metadata.version("kagglesdk") == "0.1.38",
        "TAIL_SDK_VERSION",
    )
    from seed202_pilot.provider import inherited_provider_class

    return inherited_provider_class(args.parent_archive)(args.auth_env, "zimike")


def _services(args, plan):
    from mqar_final_tail_host_services import absence

    return absence(args.private, args.science_dir, plan["nonce"])


@contextlib.contextmanager
def preview_lock(plan, action):
    path = Path(plan["journals"][action]) / "single-writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield path
    finally:
        path.unlink()


def _execute(args):
    if args.command == "check":
        from mqar_final_tail_package import check

        check(args.candidate)
        return {"status": "EXACT_BYTES_NOT_APPROVAL"}
    if args.command == "bind":
        value = bind(args.candidate, args.journal_root, args.decision, args.data)
        create_json(args.output, value)
        return {"status": "TAIL_PLAN_NOT_REVIEWED_NOT_AUTHORIZED"}
    plan = validate_plan(load(args.plan))
    activate(plan)
    if args.command == "seal":
        from mqar_final_tail_host_seal import seal

        create_json(args.output, seal(plan, load(args.review), load(args.proof)))
        return {"status": "TAIL_ACTION_SEALED"}
    if args.command in {"status", "collect", "services", "accept"}:
        from mqar_final_tail_host_observe import execute_observation

        return execute_observation(args, plan, _provider)
    from mqar_final_tail_host_actions import command_scope, perform, reconcile
    from mqar_final_tail_host_prepare import prepare, session
    from mqar_final_tail_host_seal import validate, validate_review

    action = args.command if args.command in {"publish", "submit"} else args.action
    require(action in {"publish", "submit"}, "TAIL_EXPLICIT_ACTION")
    review = load(args.review)
    validate_review(plan, review)
    approval = None
    if args.command in {"publish", "submit"}:
        approval = validate(load(args.approval), revalidate=False)
        require(
            exact(approval["plan"], plan)
            and exact(approval["review"], review)
            and approval["proof"]["action"] == action,
            "TAIL_APPROVAL_ACTION",
        )
    with command_scope(plan, action, reconcile=args.command == "reconcile"):
        if args.command == "reconcile":
            reconcile(plan, action, _provider(args), args.parent_archive)
        elif args.command == "prepare":
            prepare(
                plan,
                action,
                review,
                _provider(args),
                args.science_archive,
                args.parent_archive,
                args.output,
                args.science_dir,
            )
        else:
            receipt = load(args.receipt)
            require(
                str(Path(args.science_dir).resolve()) == receipt["science_directory"],
                "TAIL_IMPORT_ROOT_ARGUMENT",
            )
            require(
                str(Path(args.parent_archive).resolve()) == receipt["parent_archive"],
                "TAIL_PARENT_ARGUMENT",
            )
            with session(plan, receipt, action, review) as wrapper:
                provider = _provider(args)
                if args.command == "probe":
                    from mqar_final_tail_host_live import probe

                    with preview_lock(plan, action) as lock:
                        create_json(
                            args.output,
                            probe(
                                plan,
                                action,
                                wrapper,
                                provider,
                                args.private,
                                load(args.lifetime),
                                lambda: _services(args, plan),
                                lock,
                            ),
                        )
                else:
                    perform(
                        approval,
                        wrapper,
                        provider,
                        args.private,
                        load(args.lifetime),
                        lambda: _services(args, plan),
                    )
    return {"status": "TAIL_HOST_ACTION_FINISHED", "action": action, "command": args.command}


def execute(args):
    """Authenticate and lease every actual import root before provider/credential loading."""
    if args.command in {"check", "bind", "seal"}:
        return _execute(args)
    plan = validate_plan(load(args.plan))
    activate(plan)
    from mqar_final_tail_host_prepare import paths
    from seed202_pilot.prepared_files import lease

    files = paths(plan, args.science_archive, args.parent_archive, args.science_dir)
    with lease(files) as local:
        require(
            all(
                local[str(Path(plan["data_directory"]) / n)]["sha256"] == p["sha256"]
                for n, p in plan["data_files"].items()
            ),
            "ALL_DATA_ALIASES_BEFORE_CREDENTIALS",
        )
        return _execute(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "command",
        choices=(
            "check",
            "bind",
            "prepare",
            "probe",
            "seal",
            "publish",
            "submit",
            "reconcile",
            "status",
            "collect",
            "services",
            "accept",
        ),
    )
    for name in (
        "candidate",
        "journal-root",
        "decision",
        "plan",
        "review",
        "receipt",
        "proof",
        "approval",
        "output",
        "science-archive",
        "parent-archive",
        "auth-env",
        "private",
        "science-dir",
        "lifetime",
        "destination",
        "index",
        "service-proof",
        "native",
        "result",
        "policy-review",
        "data",
        "work",
        "input",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--action", choices=("publish", "submit"))
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            result = execute(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        # No exception values, SDK streams, signed URLs or credential derivatives.
        print(
            json.dumps(
                {
                    "status": "TAIL_HOST_STOPPED",
                    "exception_class": type(error).__name__
                    if type(error).__name__
                    in {
                        "ValueError",
                        "RuntimeError",
                        "OSError",
                        "FileNotFoundError",
                        "FileExistsError",
                        "PermissionError",
                        "KeyError",
                        "TypeError",
                    }
                    else "Exception",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
