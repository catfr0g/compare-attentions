"""Offline MCP transport contract; no Google or credentials."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


@pytest.mark.parametrize(
    "prefix",
    [
        b": keepalive\r\n\r\n",
        event({"jsonrpc": "2.0", "method": "notifications/progress"}),
        b"id: unused\ndata:\n\n",
    ],
)
def test_sse_ignores_notifications_until_matching_terminal(prefix):
    transport = module()
    terminal = {"jsonrpc": "2.0", "id": 3, "result": {"ok": True}}
    assert transport.read_reply(io.BytesIO(prefix + event(terminal)), "text/event-stream", 3) == {
        "ok": True
    }


@pytest.mark.parametrize(
    "body",
    [
        b"data: broken\n\n",
        event({"jsonrpc": "2.0", "id": 3, "error": {"message": "SECRET"}}),
        event({"jsonrpc": "2.0", "id": 5, "result": {}}),
        b"x" * (1024 * 1024 + 1),
    ],
    ids=["bad-json", "rpc-error", "wrong-id", "too-large"],
)
def test_sse_fails_closed_without_remote_error_leak(body):
    transport = module()
    with pytest.raises(transport.ProbeError, match="^MCP_INVALID_RESPONSE$"):
        transport.read_reply(io.BytesIO(body), "text/event-stream", 3)


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module():
    assert importlib.util.find_spec("colab_api_mcp") is not None, "transport not implemented"
    import colab_api_mcp  # pylint: disable=import-outside-toplevel

    return colab_api_mcp


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(json.dumps(payload).encode() if payload is not None else b"")
        self.status = status
        self.headers = {"Content-Type": "application/json", **(headers or {})}


def test_handshake_session_header_and_tool_calls():
    transport = module()
    seen = []

    def open_request(request, timeout):
        assert timeout == 30
        body = json.loads(request.data)
        seen.append((request, body))
        assert request.full_url == "https://colaboratory.googleapis.com/mcp"
        if body["method"] == "initialize":
            return Response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}},
                },
                headers={"MCP-Session-Id": "SECRET_SESSION"},
            )
        assert request.get_header("Mcp-session-id") == "SECRET_SESSION"
        assert request.get_header("Mcp-protocol-version") == "2025-11-25"
        if body["method"] == "notifications/initialized":
            return Response(None, status=202)
        return Response(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"structuredContent": {"name": "runtimes/demo/sessions/probe"}},
            }
        )

    client = transport.MCP("SECRET_TOKEN", opener=open_request)
    client.initialize()
    assert client.call("create_session", {"parent": "runtimes/demo", "sessionId": "probe"}) == {
        "name": "runtimes/demo/sessions/probe"
    }
    assert [body["method"] for _, body in seen] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert "SECRET" not in repr(client)


def test_network_error_is_sanitized_without_retry():
    transport = module()
    seen = []

    def fail(*args, **kwargs):
        seen.append((args, kwargs))
        raise OSError("SECRET https://private")

    with pytest.raises(transport.ProbeError, match="^MCP_CALL_AMBIGUOUS$"):
        transport.MCP("SECRET_TOKEN", opener=fail).initialize()
    assert len(seen) == 1


def test_json_reply_deadline_is_bounded(monkeypatch):
    transport = module()
    clock = iter([0, 0, 40, 80])
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(clock))

    class SlowStream(io.BytesIO):
        def read(self, _size=-1):
            pytest.fail("unbounded read used")

    with pytest.raises(transport.ProbeError, match="^MCP_INVALID_RESPONSE$"):
        transport.read_reply(
            SlowStream(b'{"jsonrpc":"2.0","id":3,"result":{}}'), "application/json", 3
        )


@pytest.mark.parametrize(
    "initial",
    [
        {"protocolVersion": "2026-07-28", "capabilities": {"tools": {}}},
        {"protocolVersion": "2025-11-25", "capabilities": {}},
    ],
)
def test_unsupported_handshake_stops_without_tool_execution(initial):
    transport = module()
    seen = []

    def open_request(request, timeout):
        seen.append((request, timeout))
        return Response({"jsonrpc": "2.0", "id": 1, "result": initial})

    with pytest.raises(transport.ProbeError, match="^MCP_UNSUPPORTED_PROTOCOL$"):
        transport.MCP("fixture-token", opener=open_request).initialize()
    assert len(seen) == 1


@pytest.mark.parametrize(
    "result",
    [{"isError": True, "content": [{"text": "SECRET"}]}, {"content": [{"text": "SECRET"}]}],
)
def test_unstructured_tool_output_fails_closed(result):
    transport = module()

    def open_request(request, timeout):
        assert timeout == 30
        body = json.loads(request.data)
        return Response({"jsonrpc": "2.0", "id": body["id"], "result": result})

    client = transport.MCP("fixture-token", opener=open_request)
    client._initialized = True  # pylint: disable=protected-access
    with pytest.raises(transport.ProbeError, match="^MCP_TOOL_FAILED$"):
        client.call("create_session", {})


def test_json_response_matches_only_requested_id():
    transport = module()
    response = io.BytesIO(b'{"jsonrpc":"2.0","id":3,"result":{"ok":true}}')
    assert transport.read_reply(response, "application/json", 3) == {"ok": True}
    with pytest.raises(transport.ProbeError, match="MCP_INVALID_RESPONSE"):
        transport.read_reply(io.BytesIO(response.getvalue()), "application/json", 4)
