"""Bounded, non-retrying Google MCP transport for the fixed readiness probe."""

import http.client
import json
import time
import urllib.error
import urllib.request

ENDPOINT = "https://colaboratory.googleapis.com/mcp"
VERSION = "2025-11-25"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProbeError("MCP_REDIRECT_REJECTED")


class MCP:
    """One legacy-protocol connection; secrets remain in memory only."""

    def __init__(self, token, *, opener=None):
        self._token = token
        self._open = opener or urllib.request.build_opener(NoRedirect()).open
        self._session = None
        self._counter = 0
        self._initialized = False

    def _post(self, method, params, *, notification=False):
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            self._counter += 1
            body["id"] = self._counter
        headers = {
            "Authorization": "Bearer " + self._token,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._initialized:
            headers["MCP-Protocol-Version"] = VERSION
        if self._session:
            headers["MCP-Session-Id"] = self._session
        request = urllib.request.Request(ENDPOINT, json.dumps(body).encode(), headers)
        try:
            with self._open(request, timeout=30) as response:
                if notification:
                    if response.status != 202 or response.read(1):
                        raise ProbeError("MCP_INVALID_RESPONSE")
                    return {}
                if response.status != 200:
                    raise ProbeError("MCP_CALL_AMBIGUOUS")
                result = read_reply(response, response.headers.get("Content-Type", ""), body["id"])
                if method == "initialize":
                    session = response.headers.get("MCP-Session-Id")
                    if session is not None:
                        if (
                            not session
                            or len(session) > 4096
                            or any(not 33 <= ord(c) <= 126 for c in session)
                        ):
                            raise ProbeError("MCP_INVALID_RESPONSE")
                        self._session = session
                return result
        except (OSError, ValueError, http.client.HTTPException):
            raise ProbeError("MCP_CALL_AMBIGUOUS") from None

    def initialize(self):
        result = self._post(
            "initialize",
            {
                "protocolVersion": VERSION,
                "capabilities": {},
                "clientInfo": {"name": "compare-attention-readiness", "version": "1"},
            },
        )
        if (
            result.get("protocolVersion") != VERSION
            or not isinstance(result.get("capabilities"), dict)
            or not isinstance(result["capabilities"].get("tools"), dict)
        ):
            raise ProbeError("MCP_UNSUPPORTED_PROTOCOL")
        self._initialized = True
        self._post("notifications/initialized", {}, notification=True)

    def call(self, name, arguments):
        if not self._initialized:
            raise ProbeError("MCP_NOT_INITIALIZED")
        result = self._post("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError", False) is not False or not isinstance(
            result.get("structuredContent"), dict
        ):
            raise ProbeError("MCP_TOOL_FAILED")
        return result["structuredContent"]


LIMIT = 1024 * 1024


class ProbeError(Exception):
    """Only controlled, secret-free error codes cross the CLI boundary."""


def _terminal(value, request_id):
    if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
        raise ValueError
    if "id" not in value and isinstance(value.get("method"), str):
        return None
    if (
        not isinstance(value.get("id"), int)
        or isinstance(value.get("id"), bool)
        or value["id"] != request_id
        or "error" in value
        or not isinstance(value.get("result"), dict)
    ):
        raise ValueError
    return value["result"]


def _chunks(stream):
    start, size = time.monotonic(), 0
    while time.monotonic() - start < 30:
        chunk = stream.read1(min(65536, LIMIT + 1 - size))
        if time.monotonic() - start >= 30:
            break
        if not chunk:
            return
        size += len(chunk)
        if size > LIMIT:
            break
        yield chunk
    raise ValueError


def _sse_values(stream):
    pending, data = b"", []
    for chunk in _chunks(stream):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            text = line.decode("utf-8").rstrip("\r")
            if text.startswith("data:"):
                data.append(text[5:].removeprefix(" "))
            elif text == "":
                payload, data = "\n".join(data), []
                if payload:
                    yield json.loads(payload)


def read_reply(stream, content_type, request_id):
    """Read matching JSON/SSE under 1 MiB and a response time budget."""
    try:
        media = content_type.split(";", 1)[0].strip()
        if media == "application/json":
            result = _terminal(json.loads(b"".join(_chunks(stream))), request_id)
            if result is not None:
                return result
        elif media == "text/event-stream":
            for value in _sse_values(stream):
                result = _terminal(value, request_id)
                if result is not None:
                    return result
        raise ValueError
    except (ValueError, UnicodeError, RecursionError):
        raise ProbeError("MCP_INVALID_RESPONSE") from None
