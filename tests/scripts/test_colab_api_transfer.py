"""Synthetic transport contracts; never use account credentials."""

# The synthetic transport implements the production interface verbatim.
# pylint: disable=unused-argument,too-few-public-methods

import base64
import importlib
import io
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def module():
    return importlib.import_module("colab_api_transfer")


def test_chunk_upload_and_unsupported_probe(tmp_path):
    store, events = {}, []

    class FakeContents(module().Contents):
        def __init__(self, append=True):
            self.append = append

        def request(self, path, *, data=None, content=True):
            if data is None:
                if path not in store:
                    raise FileNotFoundError
                return {
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(store[path]).decode(),
                }
            chunk = data.get("chunk")
            value = base64.b64decode(data["content"])
            store[path] = (store.get(path, b"") if self.append and chunk != 1 else b"") + value
            return {"type": "file"}

    transport = FakeContents()
    transport.probe_chunks("probe", events.append)
    source = tmp_path / "archive.zip"
    source.write_bytes(b"abcdefghij")
    transport.upload_archive("stage.zip", source, events.append, chunk_size=4)
    assert store["stage.zip"] == source.read_bytes()
    assert [e["chunk"] for e in events if e.get("file") == "archive"] == [1, 2, -1]
    with pytest.raises(ValueError, match="TARGET_EXISTS"):
        transport.upload_archive("stage.zip", source, events.append)
    with pytest.raises(ValueError, match="CHUNK_UPLOAD_UNSUPPORTED"):
        FakeContents(False).probe_chunks("bad-probe", events.append)


@pytest.mark.parametrize("domain", ["runtime.googleusercontent.com", "runtime.colab.dev"])
def test_proxy_preserves_signed_query_without_bearer(domain):
    calls = []

    class Reply(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def opener(req, **_kwargs):
        calls.append(req)
        return Reply(
            json.dumps(
                {"type": "file", "format": "base64", "content": base64.encodebytes(b"ok").decode()}
            ).encode()
        )

    contents = module().Contents(
        {"url": f"https://{domain}/proxy/?sig=SECRET", "token": "PROXY"},
        opener=opener,
    )
    assert contents.read("content/marker") == b"ok"
    req = calls[0]
    assert req.get_header("Authorization") is None
    assert req.get_header("X-colab-runtime-proxy-token") == "PROXY"
    assert urlsplit(req.full_url).path == "/proxy/api/contents/content/marker"
    assert parse_qs(urlsplit(req.full_url).query)["sig"] == ["SECRET"]


@pytest.mark.parametrize(
    "url",
    [
        "http://x.google.com/",
        "https://evil.com/",
        "https://runtime.colab.dev.attacker.invalid/",
        "https://runtime-colab.dev/",
        "http://runtime.colab.dev/",
        "https://x.google.com.evil.com/",
        "https://u:p@x.google.com/",
        "https://x.google.com:444/",
        "https://x.google.com/#x",
    ],
)
def test_untrusted_connection_rejected(url):
    with pytest.raises(ValueError, match="INVALID_PROXY"):
        module().Contents({"url": url, "token": "TOKEN"})
