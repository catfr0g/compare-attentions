"""Explicit one-T4 allocation; a durable journal prevents blind retries."""

import math
import time
import uuid

import colab_api_client as client
import colab_api_journal as journal

DELETE_STATES = {"delete_unknown", "delete_acknowledged", "deleted"}


def create(path, token, *, execute=False):
    if not execute:
        return {"execute": False, "runtimeSpec": dict(client.SPEC), "action": "create_one_runtime"}
    with journal.locked(path) as target:
        if target.exists():
            raise client.ApiError("JOURNAL_EXISTS", action="Use reconcile; never repeat POST")
        rows = client.specs(client.get(client.BASE + "runtimespecs", token))
        if not any(row["eligible"] is True and row["key"] == client.SPEC for row in rows):
            raise client.ApiError("T4_STANDARD_NOT_ELIGIBLE")
        runtime_id = "t4-" + uuid.uuid4().hex
        state = {
            "schema": 1,
            "runtimeId": runtime_id,
            "requestId": str(uuid.uuid4()),
            "name": "runtimes/" + runtime_id,
            "runtimeSpec": dict(client.SPEC),
            "state": "create_unknown",
        }
        journal.save(target, state, initial=True)
        try:
            raw = client.manage(
                "POST", "runtimes", token, request_id=state["requestId"], runtime_id=runtime_id
            )
        except OSError:
            raise client.ApiError(
                "TRANSPORT_ERROR", action="Use reconcile; POST outcome unknown"
            ) from None
        apply_operation(state, raw)
        journal.save(target, state)
        if state["state"] == "create_unknown":
            read_runtime(state, token)
        journal.save(target, state)
        return state


def validate_poll(wait_timeout, poll_interval):
    for value, lower, upper in ((wait_timeout, 0, 300), (poll_interval, 1, 30)):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise client.ApiError("INVALID_POLL_BOUNDS")


def reconcile(path, token, *, wait_timeout=0, poll_interval=2):
    validate_poll(wait_timeout, poll_interval)
    deadline = time.monotonic() + wait_timeout
    with journal.locked(path) as target:
        state = journal.load(target)
        while True:
            if "operation" in state and state["state"] not in DELETE_STATES:
                apply_operation(state, client.operation(state["operation"], token))
            if state["state"] not in ("creating", "create_failed"):
                read_runtime(state, token)
            journal.save(target, state)
            remaining = deadline - time.monotonic()
            if (
                state["state"] in ("ready", "deleted", "mismatch", "create_failed")
                or remaining <= 0
            ):
                return state
            time.sleep(min(poll_interval, remaining))
            if time.monotonic() >= deadline:
                return state


def apply_operation(state, raw):
    if not isinstance(raw, dict):
        raise client.ApiError("INVALID_OPERATION")
    name = client.operation_name(raw.get("name"))
    done = raw.get("done", False)
    if not isinstance(done, bool) or ("operation" in state and state["operation"] != name):
        raise client.ApiError("INVALID_OPERATION")
    if (not done and ("error" in raw or "response" in raw)) or (
        "error" in raw and "response" in raw
    ):
        raise client.ApiError("INVALID_OPERATION")
    state["operation"] = name
    state["state"] = "create_unknown" if done else "creating"
    if "error" in raw:
        error = raw["error"]
        if (
            not isinstance(error, dict)
            or type(error.get("code")) is not int  # pylint: disable=unidiomatic-typecheck
            or not 1 <= error["code"] <= 16
        ):
            raise client.ApiError("INVALID_OPERATION")
        failure = {"code": error["code"]}
        details = error.get("details", [])
        if isinstance(details, list):
            for detail in details:
                if (
                    isinstance(detail, dict)
                    and detail.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"
                    and detail.get("reason") in client.REASONS
                ):
                    failure["reason"] = detail["reason"]
        state.update(state="create_failed", failure=failure)


def read_runtime(state, token):
    try:
        raw = client.get(client.BASE + state["name"], token)
    except client.ApiError as exc:
        if exc.payload == {"error": "HTTP_ERROR", "status": 404}:
            if state["state"] in ("delete_acknowledged", "deleted"):
                state["state"] = "deleted"
            elif state["state"] != "delete_unknown":
                state["state"] = "absent_unknown"
            return
        raise
    if raw.get("name") != state["name"] or raw.get("runtimeSpec") != client.SPEC:
        if state["state"] in DELETE_STATES:
            raise client.ApiError("RUNTIME_MISMATCH")
        state["state"] = "mismatch"
    elif state["state"] not in DELETE_STATES:
        state["state"] = "ready"


def stop(path, token, *, execute=False):
    if not execute:
        state = journal.load(path)
        return {
            "execute": False,
            "name": state["name"],
            "action": "delete_unassign_not_pause",
            "warning": "Ephemeral runtime data will be lost",
        }
    with journal.locked(path) as target:
        state = journal.load(target)
        if state["state"] in DELETE_STATES:
            raise client.ApiError(
                "DELETE_ALREADY_ATTEMPTED", action="Use reconcile; never repeat DELETE"
            )
        read_runtime(state, token)
        if state["state"] != "ready":
            journal.save(target, state)
            raise client.ApiError("RUNTIME_NOT_VERIFIED")
        state["state"] = "delete_unknown"
        journal.save(target, state)
        try:
            response = client.manage("DELETE", state["name"], token)
        except OSError:
            raise client.ApiError(
                "TRANSPORT_ERROR", action="Use reconcile; DELETE outcome unknown"
            ) from None
        if response != {}:
            raise client.ApiError("INVALID_DELETE_RESPONSE")
        state["state"] = "delete_acknowledged"
        journal.save(target, state)
        read_runtime(state, token)
        journal.save(target, state)
        return state
