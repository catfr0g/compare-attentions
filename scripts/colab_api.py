"""Explicit Colab host controller; allocation/deletion require --execute."""

import argparse
import json
import os
import re
from pathlib import Path

import colab_api_auth as auth
import colab_api_client as client
import colab_api_runtime as runtime

DEFAULT_TOKEN = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share")))
    / "compare-attention/colab-api/token.json"
)
JOURNAL_COMMANDS = {"create", "status", "reconcile", "stop", "delete"}


def validate(args):
    for enabled, commands in (
        (args.management, {"auth-login"}),
        (args.execute, {"create", "stop", "delete"}),
        (args.journal, JOURNAL_COMMANDS),
        (args.runtime, {"get", "operation"}),
        (args.no_browser or args.overwrite, {"auth-login"}),
    ):
        if enabled and args.command not in commands:
            raise client.ApiError("INVALID_ARGUMENT")
    if (args.wait_timeout is not None or args.poll_interval is not None) and args.command not in (
        "status",
        "reconcile",
    ):
        raise client.ApiError("INVALID_ARGUMENT")
    runtime.validate_poll(
        args.wait_timeout if args.wait_timeout is not None else 0,
        args.poll_interval if args.poll_interval is not None else 2,
    )
    if args.management and args.token_file.resolve() == DEFAULT_TOKEN.resolve():
        raise client.ApiError(
            "SEPARATE_MANAGEMENT_TOKEN_REQUIRED",
            action="Use --token-file with management-token.json; preserve readonly token",
        )
    if args.command in JOURNAL_COMMANDS and not args.journal:
        raise client.ApiError("JOURNAL_REQUIRED")
    if args.command in ("status", "reconcile", "stop", "delete"):
        runtime.journal.load(args.journal)
    if args.command == "operation":
        client.operation_name(args.runtime)


def run(args):
    validate(args)
    if args.command == "auth-login":
        return auth.login(
            args.client_file,
            args.token_file,
            overwrite=args.overwrite,
            no_browser=args.no_browser,
            management=args.management,
        )
    if args.command == "auth-status":
        result = auth.status(args.token_file)
        if not result["scope_valid"]:
            raise client.ApiError(
                "SCOPE_REQUIRED", action="Repeat auth-login granting readonly scope"
            )
        return result
    if args.command == "get" and (
        not args.runtime or not re.fullmatch(r"runtimes/[a-zA-Z0-9_-]+", args.runtime)
    ):
        raise client.ApiError("INVALID_RUNTIME_NAME", action="Use get runtimes/ID")
    if args.command == "create" and not args.execute:
        return runtime.create(args.journal, None)
    if args.command in ("stop", "delete") and not args.execute:
        return runtime.stop(args.journal, None)
    if args.command in ("create", "stop", "delete"):
        token = auth.access_token(args.token_file, args.client_file, required_scope=auth.MANAGE)
        action = runtime.create if args.command == "create" else runtime.stop
        return action(args.journal, token, execute=True)
    return read_command(args, auth.access_token(args.token_file, args.client_file))


def read_command(args, token):
    if args.command in ("status", "reconcile"):
        return runtime.reconcile(
            args.journal,
            token,
            wait_timeout=args.wait_timeout if args.wait_timeout is not None else 0,
            poll_interval=args.poll_interval if args.poll_interval is not None else 2,
        )
    if args.command == "operation":
        state = {"operation": args.runtime}
        raw = client.operation(args.runtime, token)
        runtime.apply_operation(state, raw)
        result = {"name": state["operation"], "done": raw.get("done", False)}
        if "failure" in state:
            result["failure"] = state["failure"]
        return result
    if args.command in ("specs", "t4-eligible"):
        rows = client.specs(
            client.get(client.BASE + "runtimespecs", token), t4_only=args.command == "t4-eligible"
        )
        return {"runtimeSpecs": rows}
    if args.command == "get":
        return client.runtime(client.get(client.BASE + args.runtime, token))
    rows = client.get(client.BASE + "runtimes", token).get("runtimes", [])
    if not isinstance(rows, list):
        raise client.ApiError("INVALID_RESPONSE")
    return {"runtimes": [client.runtime(row) for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN,
    )
    parser.add_argument("--client-file", type=Path)
    parser.add_argument(
        "--no-browser", action="store_true", help="Print login URL; automatic callback"
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace login token")
    parser.add_argument(
        "--management",
        action="store_true",
        help="auth-login only: full Colab scope, separate token required",
    )
    parser.add_argument(
        "--journal", type=Path, help="Durable secret-free runtime management journal"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Explicitly create or delete ONE journal runtime"
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        help="status/reconcile only: bounded GET polling, 0..300 seconds",
    )
    parser.add_argument(
        "--poll-interval", type=float, help="status/reconcile only: 1..30 seconds (default 2)"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="t4-eligible",
        choices=[
            "auth-status",
            "auth-login",
            "specs",
            "t4-eligible",
            "runtimes",
            "get",
            "operation",
            *sorted(JOURNAL_COMMANDS),
        ],
    )
    parser.add_argument("runtime", nargs="?", help="get only: runtimes/ID")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args)))
        return 0
    except client.ApiError as exc:
        print(json.dumps(exc.payload))
        return 2
    except (OSError, ValueError, TypeError):
        print(
            json.dumps({"error": "LOCAL_ERROR", "action": "Check private file permissions/format"})
        )
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"error": "CANCELLED"}))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
