"""Real loopback HTTP exercises the production urllib request path with synthetic data."""

# Pytest injects the fixture with the same name as the provider.
# pylint: disable=redefined-outer-name

import base64
import importlib
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
transfer = importlib.import_module("colab_api_transfer")


@pytest.fixture
def server():
    store, calls = {}, []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # pylint: disable=invalid-name
            name = unquote(urlsplit(self.path).path.split("/api/contents/", 1)[1])
            calls.append(("GET", name, dict(self.headers)))
            if name == "redirect":
                self.send_response(302)
                self.send_header("Location", "https://evil.invalid/TOKEN")
                self.end_headers()
                return
            if name not in store:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "type": "file",
                        "format": "base64",
                        "content": base64.encodebytes(store[name]).decode(),
                    }
                ).encode()
            )

        def do_PUT(self):  # pylint: disable=invalid-name
            name = unquote(urlsplit(self.path).path.split("/api/contents/", 1)[1])
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("PUT", name, value.get("chunk")))
            data = base64.b64decode(value["content"])
            store[name] = (
                store.get(name, b"") if value.get("chunk") not in (None, 1) else b""
            ) + data
            if name == "ambiguous":
                self.close_connection = True
                return
            self.send_response(201)
            self.end_headers()
            self.wfile.write(b'{"type":"file"}')

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    safe_opener = urllib.request.build_opener(transfer.NoRedirect())

    def local_opener(request, **kwargs):
        # Tests only: redirect the already-validated Google URL to this fake server.
        parsed = urlsplit(request.full_url)
        request.full_url = (
            f"http://127.0.0.1:{httpd.server_port}" + parsed.path + "?" + parsed.query
        )
        return safe_opener.open(request, **kwargs)

    contents = transfer.Contents(
        {"url": "https://test.google.com/?signature=SIGNED", "token": "PROXY"}, opener=local_opener
    )
    try:
        yield contents, store, calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_loopback_chunk_upload_private_readback_root_and_no_bearer(server, tmp_path):
    contents, store, calls = server
    store["content/marker"] = b"nonce"
    assert contents.discover_root("marker", "nonce") == "content/"
    contents.probe_chunks("content/probe", lambda _value: None)
    archive = tmp_path / "zip"
    archive.write_bytes(bytes(range(256)) * 100)
    journal = []
    contents.upload_archive("content/stage", archive, journal.append, chunk_size=1024)
    assert store["content/stage"] == archive.read_bytes()
    secret = b'HF_TOKEN="synthetic-private"\n'
    contents.write_private("content/private/env", secret, journal.append)
    assert store["content/private/env"] == secret
    assert "synthetic-private" not in json.dumps(journal)
    assert all("Authorization" not in item[2] for item in calls if item[0] == "GET")
    assert all(
        item[2]["X-Colab-Runtime-Proxy-Token"] == "PROXY" for item in calls if item[0] == "GET"
    )


def test_redirect_never_followed(server):
    contents, _store, calls = server
    with pytest.raises(ValueError):
        contents.read("redirect")
    assert len(calls) == 1


def test_lost_put_response_is_not_retried_and_no_secret_in_error(server, tmp_path):
    contents, store, calls = server
    source = tmp_path / "zip"
    source.write_bytes(b"archive")
    with pytest.raises(ValueError, match="CONTENTS_STATE_UNKNOWN") as error:
        contents.upload_archive("ambiguous", source, lambda _value: None)
    assert store["ambiguous"] == b"archive"
    assert len([c for c in calls if c[0] == "PUT"]) == 1
    assert "SIGNED" not in str(error.value)
    assert "PROXY" not in str(error.value)
