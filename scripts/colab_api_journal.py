"""Atomic, secret-free runtime journal with an OS-released single-writer lock."""

import contextlib
import json
import os
import tempfile
from pathlib import Path

import colab_api_client as client

STATES = {
    "create_unknown",
    "creating",
    "create_failed",
    "ready",
    "mismatch",
    "absent_unknown",
    "delete_unknown",
    "delete_acknowledged",
    "deleted",
}


@contextlib.contextmanager
def locked(path):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt  # pylint: disable=import-outside-toplevel

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # pylint: disable=import-outside-toplevel,import-error

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise client.ApiError("JOURNAL_BUSY") from None
        try:
            yield target
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate(raw):
    if not isinstance(raw, dict) or set(raw) - {
        "schema",
        "runtimeId",
        "requestId",
        "name",
        "runtimeSpec",
        "state",
        "operation",
        "failure",
    }:
        raise client.ApiError("INVALID_JOURNAL")
    client.validate_runtime_id(raw.get("runtimeId"))
    client.validate_request_id(raw.get("requestId"))
    if (
        type(raw.get("schema")) is not int  # pylint: disable=unidiomatic-typecheck
        or raw["schema"] != 1
        or raw.get("name") != "runtimes/" + raw["runtimeId"]
        or raw.get("runtimeSpec") != client.SPEC
        or raw.get("state") not in STATES
    ):
        raise client.ApiError("INVALID_JOURNAL")
    if "operation" in raw:
        client.operation_name(raw["operation"])
    if "failure" in raw:
        failure = raw["failure"]
        if not isinstance(failure, dict) or set(failure) - {"code", "reason"}:
            raise client.ApiError("INVALID_JOURNAL")
        if (
            type(failure.get("code")) is not int  # pylint: disable=unidiomatic-typecheck
            or not 0 <= failure["code"] <= 16
            or ("reason" in failure and failure["reason"] not in client.REASONS)
        ):
            raise client.ApiError("INVALID_JOURNAL")
    return raw


def load(path):
    try:
        return validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        raise client.ApiError("INVALID_JOURNAL") from None


def save(path, value, *, initial=False):
    validate(value)
    target = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if initial:
            try:
                os.link(temporary, target)
            except FileExistsError:
                raise client.ApiError("JOURNAL_EXISTS") from None
        else:
            os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
