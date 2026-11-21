"""Bounded elastic controller facts; never read error files, messages or child streams."""

import json
import os
import subprocess
from types import SimpleNamespace


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("NORMAL_ELASTIC_JSON")
        value[key] = item
    return value


def _constant(_value):
    raise ValueError("NORMAL_ELASTIC_JSON")


NAME = "worker-elastic.json"
KEYS = {
    "schema",
    "source",
    "case",
    "phase",
    "failures",
    "first_global_rank",
    "python_trace_recovered",
}
ROW_KEYS = {
    "global_rank",
    "local_rank",
    "pid",
    "exitcode",
    "signal",
    "timestamp",
    "error_file_reported",
}


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def valid(value, context):
    """Exact bounded schema and authenticated source/phase/case, no unknown fields."""
    if context is None or context.phase != "worker" or type(value) is not dict:
        return False
    if set(value) != KEYS or (value["schema"], value["source"], value["case"], value["phase"]) != (
        "normal-elastic-failure-v1",
        context.source,
        context.case,
        "worker",
    ):
        return False
    if value["python_trace_recovered"] is not False:
        return False
    rows = value["failures"]
    if type(rows) is not list or not 1 <= len(rows) <= 2:
        return False
    for row in rows:
        if type(row) is not dict or set(row) != ROW_KEYS:
            return False
        if (
            not all(
                (
                    integer(row["global_rank"], 0, 1),
                    integer(row["local_rank"], 0, 1),
                    integer(row["pid"], 1, 2**32 - 1),
                    integer(row["exitcode"], -(2**31), 2**32 - 1),
                    integer(row["timestamp"], 0, 2**40),
                    type(row["error_file_reported"]) is bool,
                )
            )
            or row["exitcode"] == 0
            or row["global_rank"] != row["local_rank"]
        ):
            return False
        expected = -row["exitcode"] if -255 <= row["exitcode"] < 0 else None
        if row["signal"] is not None and not integer(row["signal"], 1, 255):
            return False
        if row["signal"] != expected:
            return False
    ranks = [r["global_rank"] for r in rows]
    return (
        ranks == sorted(set(ranks))
        and integer(value["first_global_rank"], 0, 1)
        and value["first_global_rank"]
        == min(rows, key=lambda r: (r["timestamp"], r["global_rank"]))["global_rank"]
    )


def project(error, context):
    """Project only exact pinned Torch types, ignoring every raw/private attribute."""
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, ProcessFailure

    if type(error) is not ChildFailedError or context is None or context.phase != "worker":
        return None
    failures = vars(error).get("failures")
    if type(failures) is not dict or not 1 <= len(failures) <= 2:
        return None
    rows = []
    for rank, failure in failures.items():
        if not integer(rank, 0, 1) or type(failure) is not ProcessFailure:
            return None
        fields = vars(failure)
        code = fields.get("exitcode")
        if type(code) is not int:
            return None
        rows.append(
            {
                "global_rank": rank,
                "local_rank": fields.get("local_rank"),
                "pid": fields.get("pid"),
                "exitcode": code,
                "signal": -code if -255 <= code < 0 else None,
                "timestamp": fields.get("timestamp"),
                "error_file_reported": type(fields.get("error_file")) is str
                and fields["error_file"] != "<N/A>",
            }
        )
    rows.sort(key=lambda r: r["global_rank"])
    value = {
        "schema": "normal-elastic-failure-v1",
        "source": context.source,
        "case": context.case,
        "phase": "worker",
        "failures": rows,
        "first_global_rank": min(rows, key=lambda r: (r["timestamp"], r["global_rank"]))[
            "global_rank"
        ],
        "python_trace_recovered": False,
    }
    return value if valid(value, context) else None


def persist(value, context):
    """Best effort create-only fsync; observer errors cannot replace the exception."""
    try:
        if not valid(value, context) or context.root.is_symlink() or context.root.is_junction():
            return
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
        if len(raw) > 4096:
            return
        with (context.root / NAME).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:  # pylint: disable=broad-exception-caught
        pass


def read(context):
    try:
        if context is None or context.phase != "worker":
            return None
        path = context.root / NAME
        if (
            path.is_symlink()
            or path.is_junction()
            or context.root.is_symlink()
            or context.root.is_junction()
        ):
            return None
        with path.open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            return None
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
        return value if valid(value, context) else None
    except BaseException:  # pylint: disable=broad-exception-caught
        return None


def observe(context, operation):
    """Supplement outer failure without masquerading as a rank Python primary."""
    try:
        return operation()
    except BaseException as error:
        try:
            code = (
                vars(error).get("returncode")
                if type(error) is subprocess.CalledProcessError
                else None
            )
            if type(code) is int and code != 0:
                value = read(context)
                if value is not None:
                    error._normal_elastic = value  # pylint: disable=protected-access
        except BaseException:  # pylint: disable=broad-exception-caught
            pass
        raise


def public(error, source, case):
    """Validate again before kernel publication, including exact worker provenance."""
    if type(error) is not subprocess.CalledProcessError:
        return None
    value = vars(error).get("_normal_elastic")
    context = SimpleNamespace(phase="worker", source=source, case=case)
    return value if valid(value, context) else None


def invoke(operation):
    """Keep the original success, SystemExit and exception identities unchanged."""
    from normal_launch_diagnostics import normal_context

    try:
        return operation()
    except BaseException as error:
        try:
            context = normal_context()
            persist(project(error, context), context)
        except BaseException:  # pylint: disable=broad-exception-caught
            pass
        raise


def main(argv=None):
    """The identical real torchrun main and argv, not a replacement launcher."""
    from torch.distributed.run import main as torchrun_main

    return invoke(lambda: torchrun_main(argv))


def cli():
    try:
        return main()
    except BaseException as error:
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

        if type(error) is ChildFailedError:
            return 1  # Same CLI failure status, without formatting private messages.
        raise


if __name__ == "__main__":
    raise SystemExit(cli())
