"""Exercise the installed SDK HTTP serializer with only Session.send intercepted.

This fixture uses synthetic credentials only. It stores no headers or URLs.
"""

from __future__ import annotations

import datetime
import enum
import json
import types

import requests
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kaggle_object import KaggleObject


def _json_value(value):
    if isinstance(value, KaggleObject):
        # This is synthetic SERVER response encoding, not client to_dict(): the
        # installed SDK's outgoing duration serializer drops timedelta.days.
        return {
            field.json_name: _json_value(getattr(value, field.private_field_name))
            for field in value._fields
            if getattr(value, field.private_field_name) is not None
            and getattr(value, field.private_field_name) != field.default_value
        }
    if isinstance(value, datetime.timedelta):
        seconds = value.days * 86400 + value.seconds
        return f"{seconds}.{value.microseconds:06d}".rstrip("0").rstrip(".") + "s"
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, types.SimpleNamespace):
        return {
            key.split("_")[0] + "".join(part.title() for part in key.split("_")[1:]): _json_value(
                child
            )
            for key, child in vars(value).items()
        }
    if isinstance(value, list):
        return [_json_value(child) for child in value]
    return value


class HttpFixture:
    """Keep real request encoding, auth attachment, HTTP errors and response decoding."""

    def __init__(self, transport, monkeypatch):
        self.transport = transport
        self.serialized = []
        self.original = transport.call
        self.client = KaggleHttpClient(env=KaggleEnv.PROD, api_token="SYNTHETIC_ONLY_NOT_AUTH")
        self.client._init_session()
        self.client._session.trust_env = False
        monkeypatch.setattr(self.client._session, "send", self.send)
        monkeypatch.setattr(transport, "call", self.call)
        self.pending = None
        self.outputs = {}
        monkeypatch.setattr(requests.Session, "get", self.get_output)

    def get_output(self, _session, url=None, **_kwargs):
        # Session.get is bound on Session; the fixture replaces only network I/O.
        url = _session if url is None else url
        prefix = "https://synthetic.invalid/output/"
        assert url.startswith(prefix)
        name = url.removeprefix(prefix)
        response = requests.Response()
        response.status_code = 200
        response._content = self.outputs[name]
        response._content_consumed = True
        return response

    def send(self, prepared, **_settings):
        service, method, request, response_type = self.pending
        assert prepared.method == "POST"
        assert prepared.headers["Authorization"] == "Bearer SYNTHETIC_ONLY_NOT_AUTH"
        body = json.loads(prepared.body)
        assert body == request.__class__.to_dict(request)
        # Store only public operation names and encoded field names, not headers.
        self.serialized.append((method, sorted(body)))
        try:
            if method == "ListKernelSessionOutput":
                result = types.SimpleNamespace(
                    files=[
                        types.SimpleNamespace(
                            file_name=name, url="https://synthetic.invalid/output/" + name
                        )
                        for name in sorted(self.outputs)
                    ],
                    next_page_token="",
                )
            else:
                result = self.original(service, method, request, response_type)
        except requests.HTTPError as error:
            return error.response
        if isinstance(result, requests.Response):
            return result
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(_json_value(result)).encode()
        return response

    def call(self, service, method, request, response_type):
        assert self.pending is None
        self.pending = service, method, request, response_type
        try:
            return self.client.call(service, method, request, response_type)
        finally:
            self.pending = None
