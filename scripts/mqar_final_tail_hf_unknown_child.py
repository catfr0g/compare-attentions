"""Existing audited HF publisher through real SDK HTTP, including a lost commit ACK."""

# ruff: noqa: E402 -- mask and authenticated paths must precede SDK imports.
import os
import sys

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
sys.dont_write_bytecode = True
from pathlib import Path

source, science, sink = map(Path, sys.argv[1:4])
sys.path[:0] = [
    str(source),
    str(science / "operator"),
    str(science / "src"),
    str(science / "tools"),
    str(science / "operator/inherited"),
]
import json
import socket

from mqar_final_tail_http_fixture import FixtureHTTP
from mqar_final_tail_package import check
from mqar_final_tail_tracking import publish_final
from pytest import MonkeyPatch
from quality_services import HFTransport

check(science)
patch = MonkeyPatch()
http = FixtureHTTP(patch, sink)
http.install()
http.fail_commit = True


def forbidden(*_args, **_kwargs):
    raise AssertionError("NO_NETWORK")


patch.setattr(socket.socket, "connect", forbidden)
values = {"HF_TOKEN": "hf_FIXTURE_ONLY", "HF_REPO": "fixture/repo"}
transport = HFTransport(values)
file = sink / "fixture-result.json"
file.write_text(json.dumps({"scope": "ENGINEERING_TRANSPORT_ONLY_NOT_MODEL_SCORES"}))
try:
    publish_final(sink / "journal", "fixture-unknown/FINAL", {file.name: file}, transport)
except RuntimeError:
    pass
else:
    raise AssertionError("MISSING_LOST_ACK")
try:
    publish_final(sink / "journal", "fixture-unknown/FINAL", {file.name: file}, transport)
except ValueError:
    pass
else:
    raise AssertionError("UNKNOWN_ACK_RETRIED")
proof = publish_final(
    sink / "journal", "fixture-unknown/FINAL", {file.name: file}, transport, revision=http.revision
)
assert proof["state"] == "verified" and proof["revision"] == http.revision
assert sum("/commit/" in call[1] for call in http.calls) == 1
assert any("/resolve/" in call[1] for call in http.calls)
assert all(call[2] for call in http.calls)
print(
    json.dumps(
        {"scope": "REAL_SDK_HTTP_UNKNOWN_ACK_GET_ONLY_ENGINEERING_FIXTURE", "commit_requests": 1}
    )
)
