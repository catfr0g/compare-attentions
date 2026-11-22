"""Synthetic management protocol: no Google credentials or cloud calls."""

# pylint: disable=import-outside-toplevel
import importlib
import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
SPEC = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}


def test_create_plan_zero_writes_and_create_persisted_before_post(tmp_path, monkeypatch):
    core = importlib.import_module("colab_api_runtime")
    path = tmp_path / "journal.json"
    calls = []
    monkeypatch.setattr(
        core.client, "get", lambda *_a: {"runtimeSpecs": [{"key": SPEC, "eligible": True}]}
    )

    def manage(method, name, _token, **kw):
        saved = json.loads(path.read_text())
        assert saved["state"] == "create_unknown"
        assert saved["requestId"] == kw["request_id"]
        assert saved["runtimeId"] == kw["runtime_id"]
        calls.append((method, name))
        return {"name": "operations/one", "metadata": {"connectionInfo": {"token": "SECRET"}}}

    monkeypatch.setattr(core.client, "manage", manage)
    plan = core.create(path, None, execute=False)
    assert plan["execute"] is False
    assert not list(tmp_path.iterdir())
    result = core.create(path, "FAKE", execute=True)
    assert result["state"] == "creating"
    assert len(calls) == 1
    assert "SECRET" not in path.read_text() + json.dumps(result)
    with pytest.raises(core.client.ApiError, match="JOURNAL_EXISTS"):
        core.create(path, "FAKE", execute=True)
    assert len(calls) == 1


@pytest.mark.parametrize("interrupt", [TimeoutError, KeyboardInterrupt])
def test_interrupted_create_reconciles_get_only_without_double_post(
    tmp_path, monkeypatch, interrupt
):
    core = importlib.import_module("colab_api_runtime")
    path = tmp_path / "journal.json"
    calls = []

    def get(url, _token):
        calls.append(("GET", url))
        if url.endswith("runtimespecs"):
            return {"runtimeSpecs": [{"key": SPEC, "eligible": True}]}
        return {
            "name": json.loads(path.read_text())["name"],
            "runtimeSpec": SPEC,
            "connectionInfo": {"token": "SECRET", "url": "SECRET"},
        }

    def post(*_a, **_kw):
        calls.append(("POST", "runtimes"))
        raise interrupt("SECRET")

    monkeypatch.setattr(core.client, "get", get)
    monkeypatch.setattr(core.client, "manage", post)
    with pytest.raises((core.client.ApiError, KeyboardInterrupt)):
        core.create(path, "FAKE", execute=True)
    saved = json.loads(path.read_text())
    assert saved["state"] == "create_unknown"
    assert "SECRET" not in path.read_text()
    with pytest.raises(core.client.ApiError, match="JOURNAL_EXISTS"):
        core.create(path, "FAKE", execute=True)
    result = core.reconcile(path, "FAKE")
    assert result["state"] == "ready"
    assert "SECRET" not in json.dumps(result) + path.read_text()
    assert sum(method == "POST" for method, _ in calls) == 1
    assert calls[-1] == ("GET", core.client.BASE + saved["name"])


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("pending", "creating"),
        ("done", "ready"),
        ("wrong-name", "mismatch"),
        ("wrong-spec", "mismatch"),
        ("error", "create_failed"),
        ("bad-done", "INVALID_OPERATION"),
        ("bad-name", "INVALID_OPERATION_NAME"),
    ],
)
def test_operation_done_failure_and_ready_requires_exact_get(tmp_path, monkeypatch, mode, expected):
    core = importlib.import_module("colab_api_runtime")
    path = tmp_path / "journal.json"
    gets = []
    operation = {
        "name": "operations/one",
        "done": mode != "pending",
        "metadata": {"connectionInfo": {"token": "SECRET"}},
        "response": {"connectionInfo": {"url": "SECRET"}},
    }
    if mode == "pending":
        operation.pop("response")
    if mode == "error":
        operation["error"] = {
            "code": 8,
            "message": "SECRET",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "QUOTA_EXCEEDED",
                    "metadata": {"x": "SECRET"},
                }
            ],
        }
        operation.pop("response")
    if mode == "bad-done":
        operation["done"] = 1
    if mode == "bad-name":
        operation["name"] = "operations/../SECRET"

    def get(url, _token):
        gets.append(url)
        if url.endswith("runtimespecs"):
            return {"runtimeSpecs": [{"key": SPEC, "eligible": True}]}
        saved = json.loads(path.read_text())
        return {
            "name": "runtimes/foreign" if mode == "wrong-name" else saved["name"],
            "runtimeSpec": {**SPEC, "accelerator": "L4"} if mode == "wrong-spec" else SPEC,
            "connectionInfo": {"token": "SECRET"},
        }

    monkeypatch.setattr(core.client, "get", get)
    monkeypatch.setattr(core.client, "manage", lambda *_a, **_kw: operation)
    if expected.startswith("INVALID"):
        with pytest.raises(core.client.ApiError, match=expected):
            core.create(path, "FAKE", execute=True)
    else:
        result = core.create(path, "FAKE", execute=True)
        assert result["state"] == expected
        assert (len(gets) == 2) == (mode in ("done", "wrong-name", "wrong-spec"))
        if mode == "error":
            assert result["failure"] == {"code": 8, "reason": "QUOTA_EXCEEDED"}
        monkeypatch.setattr(core.client, "operation", lambda *_a: operation)
        assert core.reconcile(path, "FAKE")["state"] == expected
    assert "SECRET" not in path.read_text()


def seeded_journal(tmp_path):
    core = importlib.import_module("colab_api_runtime")
    path = tmp_path / "journal.json"
    state = {
        "schema": 1,
        "runtimeId": "one-t4",
        "requestId": str(uuid.uuid4()),
        "name": "runtimes/one-t4",
        "runtimeSpec": SPEC,
        "state": "ready",
    }
    core.journal.save(path, state, initial=True)
    return core, path


@pytest.mark.parametrize(
    "readback,expected",
    [(404, "deleted"), (200, "delete_acknowledged"), (503, "delete_acknowledged")],
)
def test_stop_exact_target_ack_and_readback(tmp_path, monkeypatch, readback, expected):
    core, path = seeded_journal(tmp_path)
    calls = []

    def get(url, _token):
        calls.append(("GET", url))
        if len(calls) > 1 and readback != 200:
            raise core.client.ApiError("HTTP_ERROR", status=readback)
        return {
            "name": "runtimes/one-t4",
            "runtimeSpec": SPEC,
            "connectionInfo": {"token": "SECRET"},
        }

    def delete(method, name, _token):
        assert json.loads(path.read_text())["state"] == "delete_unknown"
        calls.append((method, core.client.BASE + name))
        return {}

    monkeypatch.setattr(core.client, "get", get)
    monkeypatch.setattr(core.client, "manage", delete)
    before = path.read_bytes()
    assert core.stop(path, None, execute=False)["execute"] is False
    assert path.read_bytes() == before
    assert not calls
    if readback == 503:
        with pytest.raises(core.client.ApiError):
            core.stop(path, "FAKE", execute=True)
        assert json.loads(path.read_text())["state"] == expected
    else:
        assert core.stop(path, "FAKE", execute=True)["state"] == expected
    assert calls[:2] == [
        ("GET", core.client.BASE + "runtimes/one-t4"),
        ("DELETE", core.client.BASE + "runtimes/one-t4"),
    ]
    assert "SECRET" not in path.read_text()
    with pytest.raises(core.client.ApiError, match="DELETE_ALREADY_ATTEMPTED"):
        core.stop(path, "FAKE", execute=True)
    assert sum(method == "DELETE" for method, _ in calls) == 1
    if readback == 404:
        assert core.reconcile(path, "FAKE")["state"] == "deleted"


@pytest.mark.parametrize("wrong", ["foreign", "spec", "timeout"])
def test_stop_refuses_foreign_runtime_and_ambiguous_delete_no_retry(tmp_path, monkeypatch, wrong):
    core, path = seeded_journal(tmp_path)
    posts = []

    def get(*_a):
        return {
            "name": "runtimes/foreign" if wrong == "foreign" else "runtimes/one-t4",
            "runtimeSpec": {**SPEC, "accelerator": "L4"} if wrong == "spec" else SPEC,
        }

    def delete(*_a, **_kw):
        posts.append("DELETE")
        raise core.client.ApiError("TRANSPORT_ERROR")

    monkeypatch.setattr(core.client, "get", get)
    monkeypatch.setattr(core.client, "manage", delete)
    with pytest.raises(core.client.ApiError):
        core.stop(path, "FAKE", execute=True)
    if wrong == "timeout":
        assert json.loads(path.read_text())["state"] == "delete_unknown"
        monkeypatch.setattr(
            core.client,
            "get",
            lambda *_a: (_ for _ in ()).throw(core.client.ApiError("HTTP_ERROR", status=404)),
        )
        assert core.reconcile(path, "FAKE")["state"] == "delete_unknown"
        with pytest.raises(core.client.ApiError, match="DELETE_ALREADY_ATTEMPTED"):
            core.stop(path, "FAKE", execute=True)
        assert posts == ["DELETE"]
    else:
        assert not posts


def test_delete_unknown_mismatch_cannot_unlock_a_second_delete(tmp_path, monkeypatch):
    core, path = seeded_journal(tmp_path)
    state = core.journal.load(path)
    state["state"] = "delete_unknown"
    core.journal.save(path, state)
    monkeypatch.setattr(
        core.client, "get", lambda *_a: {"name": "runtimes/foreign", "runtimeSpec": SPEC}
    )
    with pytest.raises(core.client.ApiError, match="RUNTIME_MISMATCH"):
        core.reconcile(path, "FAKE")
    assert core.journal.load(path)["state"] == "delete_unknown"
    with pytest.raises(core.client.ApiError, match="DELETE_ALREADY_ATTEMPTED"):
        core.stop(path, "FAKE", execute=True)


def test_bounded_operation_polling_is_get_only(tmp_path, monkeypatch):
    core, path = seeded_journal(tmp_path)
    state = core.journal.load(path)
    state.update(state="creating", operation="operations/one")
    core.journal.save(path, state)
    clock = [0.0]
    calls = []

    def sleep(delay):
        assert 0 < delay <= 2
        clock[0] += delay

    def operation(name, _token):
        calls.append(name)
        return {"name": name, "done": len(calls) >= 3}

    monkeypatch.setattr(core.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core.time, "sleep", sleep)
    monkeypatch.setattr(core.client, "operation", operation)
    monkeypatch.setattr(
        core.client, "get", lambda *_a: {"name": state["name"], "runtimeSpec": SPEC}
    )
    monkeypatch.setattr(
        core.client, "manage", lambda *_a, **_kw: pytest.fail("write during polling")
    )
    assert core.reconcile(path, "FAKE", wait_timeout=5, poll_interval=2)["state"] == "ready"
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(
        core.client, "operation", lambda name, _t: calls.append(name) or {"name": name}
    )
    assert core.reconcile(path, "FAKE", wait_timeout=3, poll_interval=2)["state"] == "creating"
    assert len(calls) == 2
    for timeout, interval in ((-1, 2), (301, 2), (float("nan"), 2), (True, 2), (1, 0), (1, 31)):
        with pytest.raises(core.client.ApiError, match="INVALID_POLL_BOUNDS"):
            core.reconcile(path, "FAKE", wait_timeout=timeout, poll_interval=interval)


def test_fixed_write_transport_validates_identifiers_before_call(monkeypatch):
    import colab_api_client as client

    calls = []
    monkeypatch.setattr(client, "request_json", lambda url, **kw: calls.append((url, kw)) or {})
    request = str(uuid.uuid4())
    client.manage("POST", "runtimes", "FAKE", request_id=request, runtime_id="one-t4")
    assert calls[0][0] == client.BASE + f"runtimes?requestId={request}&runtimeId=one-t4"
    assert json.loads(calls[0][1]["data"]) == {"runtimeSpec": SPEC}
    client.manage("DELETE", "runtimes/one-t4", "FAKE")
    assert calls[1][1]["method"] == "DELETE"
    client.operation("operations/abc-123", "FAKE")
    assert calls[2][0] == "https://colaboratory.googleapis.com/v1/operations/abc-123"
    for name in (
        "operations/../x",
        "operations/a%2fb",
        "operations//x",
        "operations/a?x=y",
        " operations/a",
    ):
        with pytest.raises(client.ApiError):
            client.operation(name, "FAKE")
    for name in (
        "runtimes/../x",
        "runtimes/X",
        "runtimes/a_b",
        "runtimes/a?b",
        "runtimes/foreign/path",
    ):
        with pytest.raises(client.ApiError):
            client.manage("DELETE", name, "FAKE")
    for request_id in ("bad", str(uuid.uuid1()), request.upper()):
        with pytest.raises(client.ApiError):
            client.manage("POST", "runtimes", "FAKE", request_id=request_id, runtime_id="one-t4")
    assert len(calls) == 3
