"""HTTP boundary tests with synthetic provider data, not account eligibility."""

# Test-first imports and pytest fixture names are intentional.
# Schema literals intentionally repeat across independent subprocess/projection tests.
# pylint: disable=import-outside-toplevel,redefined-outer-name,duplicate-code
import http.server
import importlib
import io
import json
import sys
import threading
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def client():
    return importlib.import_module("colab_api_client")


def test_specs_strict_boolean_and_key():
    core = client()
    rows = [
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
            "eligible": value,
        }
        for value in (True, False, 1, "true", None)
    ]
    rows += [
        {"key": {"variant": "VARIANT_CPU", "accelerator": "T4"}, "eligible": True},
        {"key": {"variant": "VARIANT_GPU", "accelerator": "T4"}, "eligible": True},
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "UNKNOWN"},
            "eligible": True,
        },
    ]
    result = core.specs({"runtimeSpecs": rows}, t4_only=True)
    assert len(result) == 1
    assert result[0]["eligible"] is True


def test_runtime_allowlist_never_connection_info():
    core = client()
    raw = {
        "name": "runtimes/id",
        "version": "v1",
        "runtimeSpec": {
            "variant": "VARIANT_GPU",
            "accelerator": "T4",
            "shape": "SHAPE_STANDARD",
            "unknown": "SECRET",
        },
        "connectionInfo": {"url": "SECRET", "token": "SECRET"},
        "unknown": "SECRET",
    }
    assert "SECRET" not in json.dumps(core.runtime(raw))
    assert set(core.runtime(raw)) == {"name", "version", "runtimeSpec"}


def test_http_error_only_allowlisted_reason_no_body_or_url():
    core = client()
    body = json.dumps(
        {
            "error": {
                "message": "SECRET",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "SERVICE_DISABLED",
                    }
                ],
            }
        }
    )

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError("https://SECRET", 403, "SECRET", {}, io.BytesIO(body.encode()))

    with pytest.raises(core.ApiError) as error:
        core.request_json("https://colaboratory.googleapis.com/v1beta/runtimes", opener=fail)
    assert error.value.payload == {
        "error": "HTTP_ERROR",
        "status": 403,
        "reason": "SERVICE_DISABLED",
    }
    assert "SECRET" not in str(error.value)


def test_partial_http_response_never_exposes_body():
    import http.client

    core = client()

    def fail(*_args, **_kwargs):
        raise http.client.IncompleteRead(b"PRIVATE_PARTIAL_BODY")

    with pytest.raises(core.ApiError, match="TRANSPORT_ERROR") as error:
        core.request_json("https://colaboratory.googleapis.com/v1beta/runtimes", opener=fail)
    assert "PRIVATE" not in str(error.value)


def test_api_forbids_foreign_hosts_and_paths_before_credentials():
    core = client()
    for url in (
        "http://colaboratory.googleapis.com/v1beta/runtimes",
        "https://evil.test/v1beta/runtimes",
        "https://colaboratory.googleapis.com@evil.test/",
    ):
        with pytest.raises(core.ApiError):
            core.get(url, "SECRET")


@pytest.fixture
def local_http():
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # pylint: disable=invalid-name
            calls.append((self.path, self.headers.get("Authorization")))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/exfil")
                self.end_headers()
                return
            if self.path == "/invalid":
                body = b"PRIVATE_NOT_JSON"
            else:
                body = json.dumps(
                    {
                        "runtimeSpecs": [
                            {
                                "key": {
                                    "variant": "VARIANT_GPU",
                                    "accelerator": "T4",
                                    "shape": "SHAPE_STANDARD",
                                },
                                "eligible": True,
                            }
                        ]
                    }
                ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", calls
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_local_http_headers_and_redirect_cannot_exfiltrate(local_http, capsys):
    core = client()
    base, calls = local_http
    raw = core.request_json(base + "/specs", headers={"Authorization": "Bearer PRIVATE"})
    assert len(core.specs(raw, t4_only=True)) == 1
    with pytest.raises(core.ApiError) as error:
        core.request_json(base + "/redirect", headers={"Authorization": "Bearer PRIVATE"})
    assert error.value.payload == {"error": "HTTP_ERROR", "status": 302}
    assert calls == [("/specs", "Bearer PRIVATE"), ("/redirect", "Bearer PRIVATE")]
    with pytest.raises(core.ApiError, match="INVALID_RESPONSE"):
        core.request_json(base + "/invalid")
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.out + captured.err


# Minimal public discovery projection, fetched 2026-11-05, revision 20261104:
# https://colaboratory.googleapis.com/$discovery/rest?version=v1beta
# No account response or eligibility is represented by this schema fixture.
PUBLIC_SCHEMA = {
    "Key": {"variant": "string", "accelerator": "string", "shape": "string"},
    "RuntimeSpec": {"key": "Key", "eligible": "boolean"},
    "Runtime": {
        "name": "string",
        "version": "string",
        "runtimeSpec": "Key",
        "connectionInfo": "ConnectionInfo",
    },
    "specs_path": "v1beta/runtimespecs",
    "method": "GET",
    "scope": "https://www.googleapis.com/auth/colaboratory.readonly",
}


def test_subprocess_specs_over_local_http(tmp_path, local_http):
    """Run the real CLI, replacing only HTTPS transport with a local fixture."""
    import subprocess
    import time

    base, calls = local_http
    root = Path(__file__).resolve().parents[2]
    token = tmp_path / "token.json"
    token.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "access_token": "SYNTHETIC",
                "expires_at": time.time() + 3600,
                "scopes": ["https://www.googleapis.com/auth/colaboratory.readonly"],
            }
        ),
        encoding="utf-8",
    )
    harness = (
        "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); "
        "import colab_api_client as c; original=c.request_json; "
        "base=sys.argv.pop(1); c.request_json=lambda url,**kw:original(base+'/specs',**kw); "
        "runpy.run_module('colab_api',run_name='__main__')"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            str(root / "scripts"),
            base,
            "--token-file",
            str(token),
            "t4-eligible",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runtimeSpecs"][0]["key"]["accelerator"] == "T4"
    assert "SYNTHETIC" not in result.stdout + result.stderr
    assert calls == [("/specs", "Bearer SYNTHETIC")]


def test_public_discovery_contract_binding():
    core = client()
    import colab_api_auth

    assert (
        core.BASE + "runtimespecs"
        == "https://colaboratory.googleapis.com/" + PUBLIC_SCHEMA["specs_path"]
    )
    assert colab_api_auth.READONLY == PUBLIC_SCHEMA["scope"]
    raw_key = {name: "value" for name in PUBLIC_SCHEMA["Key"]}
    assert set(core.key(raw_key)) == set(PUBLIC_SCHEMA["Key"])
    raw = {
        "name": "runtimes/id",
        "version": "v",
        "runtimeSpec": raw_key,
        "connectionInfo": {"url": "PRIVATE_URL"},
    }
    assert set(core.runtime(raw)) == set(PUBLIC_SCHEMA["Runtime"]) - {"connectionInfo"}
