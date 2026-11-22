"""Exercise urllib HTTP/SSE on loopback only; never contact Google."""

import http.server
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import colab_api_mcp as mcp  # noqa: E402  # pylint: disable=wrong-import-position


@pytest.mark.parametrize("sse", [False, True])
def test_real_loopback_transport_handshake_and_redirect_rejection(sse):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # pylint: disable=invalid-name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if body["method"] == "initialize":
                result = {"protocolVersion": mcp.VERSION, "capabilities": {"tools": {}}}
            else:
                assert self.headers["MCP-Session-Id"] == "ephemeral-secret"
                assert self.headers["MCP-Protocol-Version"] == mcp.VERSION
                if body["method"] == "notifications/initialized":
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                result = {"structuredContent": {"name": "fixture-only"}}
            reply = {"jsonrpc": "2.0", "id": body["id"], "result": result}
            raw = json.dumps(reply).encode()
            if sse:
                raw = b": keepalive\n\ndata: " + raw + b"\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if body["method"] == "initialize":
                self.send_header("MCP-Session-Id", "ephemeral-secret")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # pylint: disable=invalid-name
            seen.append("redirect-request")
            self.send_response(302)
            self.send_header("Location", "/should-not-follow")
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    local = "http://127.0.0.1:" + str(server.server_port)
    opener = urllib.request.build_opener(mcp.NoRedirect())

    def loopback_only(request, timeout):
        assert request.full_url == mcp.ENDPOINT
        return opener.open(
            urllib.request.Request(local, request.data, dict(request.header_items())),
            timeout=timeout,
        )

    try:
        client = mcp.MCP("fixture-token", opener=loopback_only)
        client.initialize()
        assert client.call("create_session", {"parent": "fixture"}) == {"name": "fixture-only"}
        with pytest.raises(mcp.ProbeError, match="^MCP_REDIRECT_REJECTED$"):
            opener.open(local + "/redirect", timeout=3)
        assert [item["method"] for item in seen[:-1]] == [
            "initialize",
            "notifications/initialized",
            "tools/call",
        ]
        assert seen[-1] == "redirect-request"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
