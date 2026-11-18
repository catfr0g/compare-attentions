"""Synthetic HF HTTP responses; real SDK constructors and encoders remain active."""

import base64
import io
import json
from urllib.parse import unquote, urlsplit

import httpx2 as httpx
import requests


class HFHTTPFixture:
    """Block all network, accept only the synthetic repository and exact SDK routes."""

    def __init__(self, monkeypatch, *, corrupt=False):
        self.files = {}
        self.commits = 0
        self.encoded_files = 0
        self.corrupt = corrupt
        self.revision = "1" * 40
        self.versions = {self.revision: {}}

        def httpx_send(_client, request, **_kwargs):
            return self.hub(request)

        def requests_send(_session, request, **_kwargs):
            return self.download(request)

        monkeypatch.setattr(httpx.Client, "send", httpx_send)
        monkeypatch.setattr(requests.Session, "send", requests_send)

    def hub(self, request):
        path = unquote(request.url.path)
        assert path.startswith("/api/models/engineering/mqar"), "SYNTHETIC_HTTP_SCOPE"
        if request.method == "GET" and "/tree/" in path:
            revision = path.split("/tree/", 1)[1].split("/", 1)[0]
            files = self.versions[revision]
            response = [
                {"type": "file", "oid": "f" * 40, "size": len(raw), "path": name}
                for name, raw in files.items()
            ]
        elif request.method == "GET":
            revision = path.split("/revision/", 1)[1] if "/revision/" in path else self.revision
            response = {
                "id": "engineering/mqar",
                "private": True,
                "sha": revision,
                "siblings": [{"rfilename": name} for name in self.versions[revision]],
            }
        elif request.method == "POST" and "/preupload/" in path:
            payload = json.loads(request.read())
            response = {
                "files": [
                    {"path": row["path"], "uploadMode": "regular", "shouldIgnore": False}
                    for row in payload["files"]
                ]
            }
        elif request.method == "POST" and "/commit/" in path:
            records = [json.loads(line) for line in request.read().splitlines()]
            for row in records:
                if row["key"] == "file":
                    value = row["value"]
                    assert value["encoding"] == "base64"
                    self.files[value["path"]] = base64.b64decode(value["content"])
                    self.encoded_files += 1
                elif row["key"] == "deletedFile":
                    del self.files[row["value"]["path"]]
                else:
                    assert row["key"] == "header"
            self.commits += 1
            self.revision = f"{self.commits + 1:040x}"
            self.versions[self.revision] = dict(self.files)
            response = {
                "commitUrl": "https://huggingface.co/engineering/mqar/commit/" + self.revision,
                "commitOid": self.revision,
                "commitMessage": "engineering",
            }
        else:
            raise AssertionError("UNEXPECTED_SYNTHETIC_HF_ROUTE")
        return httpx.Response(200, json=response, request=request)

    def download(self, request):
        parsed = urlsplit(request.url)
        assert parsed.hostname == "huggingface.co"
        prefix = "/engineering/mqar/resolve/"
        assert unquote(parsed.path).startswith(prefix), "SYNTHETIC_IMMUTABLE_GET"
        revision, name = unquote(parsed.path)[len(prefix) :].split("/", 1)
        raw = self.versions[revision][name]
        if self.corrupt and name.endswith(".pt"):
            raw = raw[:-1] + bytes([raw[-1] ^ 1])
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        response.raw = io.BytesIO(raw)
        response._content = raw
        response.request = request
        return response
