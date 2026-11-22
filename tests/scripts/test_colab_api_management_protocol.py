"""Exercise real subprocess create/status/delete over local HTTP only."""

import http.server
import json
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}


def test_real_cli_management_protocol_local_http(tmp_path):
    # pylint: disable=too-many-locals,too-many-statements
    calls = []
    remote = {"name": None, "exists": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def reply(self, body, status=200):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):  # pylint: disable=invalid-name
            calls.append(("GET", self.path))
            if self.path == "/v1beta/runtimespecs":
                self.reply({"runtimeSpecs": [{"key": SPEC, "eligible": True}]})
            elif self.path == "/v1/operations/fixture-operation":
                self.reply(
                    {
                        "name": "operations/fixture-operation",
                        "done": True,
                        "metadata": {"proxyToken": "SECRET"},
                        "response": {"connectionInfo": {"token": "SECRET"}},
                    }
                )
            elif self.path == "/v1beta/" + remote["name"] and remote["exists"]:
                self.reply(
                    {
                        "name": remote["name"],
                        "runtimeSpec": SPEC,
                        "connectionInfo": {"url": "SECRET", "token": "SECRET"},
                    }
                )
            else:
                self.reply({"error": {"message": "SECRET"}}, 404)

        def do_POST(self):  # pylint: disable=invalid-name
            calls.append(("POST", self.path))
            parsed = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            assert parsed.path == "/v1beta/runtimes"
            assert set(query) == {"requestId", "runtimeId"}
            assert uuid.UUID(query["requestId"][0]).version == 4
            assert json.loads(self.rfile.read(int(self.headers["Content-Length"]))) == {
                "runtimeSpec": SPEC
            }
            remote.update(name="runtimes/" + query["runtimeId"][0], exists=True)
            self.reply({"name": "operations/fixture-operation", "metadata": {"token": "SECRET"}})

        def do_DELETE(self):  # pylint: disable=invalid-name
            calls.append(("DELETE", self.path))
            assert self.path == "/v1beta/" + remote["name"]
            remote["exists"] = False
            self.reply({})

    token = tmp_path / "fake-token.json"
    token.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "access_token": "FAKE",
                "scopes": ["https://www.googleapis.com/auth/colaboratory"],
                "expires_at": time.time() + 3600,
            }
        )
    )
    journal = tmp_path / "runtime.json"
    harness = (
        "import sys,runpy,urllib.parse;sys.path.insert(0,sys.argv.pop(1));"
        "import colab_api_client as c;original=c.request_json;base=sys.argv.pop(1);"
        "c.request_json=lambda url,**kw:original(base+urllib.parse.urlsplit(url).path+"
        "('?' + urllib.parse.urlsplit(url).query "
        "if urllib.parse.urlsplit(url).query else ''),**kw);"
        "runpy.run_module('colab_api',run_name='__main__')"
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:

            def cli(*args):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        harness,
                        str(ROOT / "scripts"),
                        f"http://127.0.0.1:{server.server_port}",
                        "--token-file",
                        str(token),
                        "--journal",
                        str(journal),
                        *args,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=20,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert "SECRET" not in result.stdout + result.stderr
                return json.loads(result.stdout)

            assert cli("--execute", "create")["state"] == "creating"
            assert cli("status")["state"] == "ready"
            count = len(calls)
            assert cli("stop")["execute"] is False
            assert len(calls) == count
            assert cli("--execute", "stop")["state"] == "deleted"
            assert cli("reconcile")["state"] == "deleted"
        finally:
            server.shutdown()
            thread.join(timeout=5)
    state = json.loads(journal.read_text())
    assert state["state"] == "deleted"
    assert "SECRET" not in journal.read_text()
    assert [method for method, _ in calls] == [
        "GET",
        "POST",
        "GET",
        "GET",
        "GET",
        "DELETE",
        "GET",
        "GET",
    ]
    assert (
        calls[1][1]
        == f"/v1beta/runtimes?requestId={state['requestId']}&runtimeId={state['runtimeId']}"
    )
