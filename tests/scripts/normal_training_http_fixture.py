"""No-network actual SDK HTTP fixture for fresh tiny NORMAL saved-main engineering."""

import base64
import json
import os
import sys
from pathlib import Path

assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
for cache_key in (
    "HF_HOME",
    "CLEARML_CACHE_DIR",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "UV_CACHE_DIR",
):
    assert Path(os.environ[cache_key]).is_relative_to(Path(os.environ["TMPDIR"]))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normal_training_http_inherited import FixtureHTTP  # noqa: E402


class NormalHTTP(FixtureHTTP):
    """Persistence is a fake remote service, not a scientific oracle or SDK replacement."""

    def __init__(self, monkeypatch, scratch):
        import threading

        self._state_lock = threading.RLock()
        scratch.mkdir(parents=True, exist_ok=True)
        super().__init__(monkeypatch, scratch)
        self.events, self.revisions = {}, {}
        state = scratch / "state.json"
        if state.is_file():
            value = json.loads(state.read_bytes())
            self.tasks, self.events = value["tasks"], value["events"]
            self.files = {name: base64.b64decode(raw) for name, raw in value["files"].items()}
            self.revisions = {
                revision: {name: base64.b64decode(raw) for name, raw in files.items()}
                for revision, files in value["revisions"].items()
            }
            self.head = value["head"]
            self.calls = value["calls"]

    def save(self):
        with self._state_lock:
            return self._save()

    def _save(self):
        if os.environ.get("RANK") == "1":
            # Rank one owns no service, including this simulated remote store.
            return
        value = {
            "tasks": self.tasks,
            "events": self.events,
            "files": {name: base64.b64encode(raw).decode() for name, raw in self.files.items()},
            "head": self.head,
            "calls": self.calls,
            "revisions": {
                revision: {name: base64.b64encode(raw).decode() for name, raw in files.items()}
                for revision, files in self.revisions.items()
            },
        }
        pending = self.scratch / "state.json.pending"
        pending.write_text(json.dumps(value), encoding="utf-8")
        pending.replace(self.scratch / "state.json")
        super().save()

    def hf(self, request):
        with self._state_lock:
            return self._hf(request)

    def _hf(self, request):
        from urllib.parse import unquote

        import httpx2 as httpx

        path = unquote(request.url.path)
        if "/commit/" in path:
            self.revision = format(len(self.revisions) + 2, "040x")
            result = super().hf(request)
            for row in [json.loads(line) for line in request.content.splitlines()][1:]:
                if row["key"] == "deletedFile":
                    self.files.pop(row["value"]["path"], None)
            self.revisions[self.revision] = dict(self.files)
        elif "/tree/" in path:
            revision = path.split("/tree/", 1)[1].split("/", 1)[0]
            files = self.revisions.get(revision, self.files)
            self.calls.append((request.method, path, bool(request.headers.get("authorization"))))
            assert request.headers.get("authorization") == "Bearer hf_FIXTURE_ONLY"
            result = httpx.Response(
                200,
                json=[
                    {"type": "file", "path": name, "oid": "a" * 40, "size": len(raw)}
                    for name, raw in files.items()
                ],
            )
        elif "/revision/" in path:
            revision = path.rsplit("/", 1)[-1]
            self.calls.append((request.method, path, bool(request.headers.get("authorization"))))
            assert revision in self.revisions
            result = httpx.Response(
                200, json={"id": "fixture/repo", "sha": revision, "private": True, "siblings": []}
            )
        else:
            result = super().hf(request)
        self.save()
        return result

    def requests_send(self, request, **kwargs):
        with self._state_lock:
            return self._requests_send(request, **kwargs)

    def _requests_send(self, request, **kwargs):
        if "/resolve/" in request.url:
            import io
            from urllib.parse import unquote, urlparse

            import requests

            path = unquote(urlparse(request.url).path)
            revision, name = path.split("/resolve/", 1)[1].split("/", 1)
            assert request.headers.get("Authorization") == "Bearer hf_FIXTURE_ONLY"
            self.calls.append((request.method, path, True))
            response = requests.Response()
            response.status_code = 200
            response._content = self.revisions[revision][name]
            response.raw = io.BytesIO(response._content)
            self.save()
            return response
        response = super().requests_send(request, **kwargs)
        self.save()
        return response

    def clearml(self, endpoint, payload, request):
        if endpoint == "events.get_task_events":
            rows = self.events.get(payload["task"], [])
            offset = int(payload.get("scroll_id") or "0")
            page = rows[offset : offset + 7]
            return {"events": page, "total": len(rows), "scroll_id": str(offset + len(page))}
        if endpoint == "events.add_batch":
            for event in payload["events"]:
                if event.get("type") == "training_stats_scalar":
                    self.events.setdefault(event["task"], []).append(event)
        if endpoint == "tasks.get_all" and payload.get("id"):
            return {"tasks": [self.tasks[key] for key in payload["id"] if key in self.tasks]}
        value = super().clearml(endpoint, payload, request)
        if endpoint == "tasks.create":
            self.tasks[value["id"]]["type"] = payload["type"]
        return value


def install_failures(private, fixture):
    """Inject only I/O boundaries around actual producers on fresh synthetic runs."""
    from clearml import Logger, Task

    from compare_attention.train.backup import BackupTracker

    config = private.parent / "failure-mode.json"
    if not config.is_file():
        return
    mode = json.loads(config.read_bytes())["mode"]
    monkeypatch = fixture.monkeypatch
    original_scalar = Logger.report_scalar
    result_started = [False]
    fired = [False]

    def fail():
        if not fired[0]:
            fired[0] = True
            raise LookupError("CANARY_PRIVATE_CREDENTIAL https://private.invalid/token")

    def scalar(self, *args, **kwargs):
        value = original_scalar(self, *args, **kwargs)
        if kwargs.get("title") == "result":
            result_started[0] = True
            if mode == "result_log":
                fail()
        return value

    monkeypatch.setattr(Logger, "report_scalar", scalar)
    for name in ("flush", "close"):
        original = getattr(Task, name)

        def boundary(self, *args, _name=name, _original=original, **kwargs):
            value = _original(self, *args, **kwargs)
            if result_started[0] and mode == _name:
                fail()
            return value

        monkeypatch.setattr(Task, name, boundary)
    original_read = fixture.clearml

    def read(endpoint, payload, request):
        if endpoint == "events.get_task_events" and mode in {"readback", "diagnostic"}:
            fail()
        return original_read(endpoint, payload, request)

    fixture.clearml = read
    if mode == "receipt_write":
        monkeypatch.setattr(BackupTracker, "record_scalar_readback", lambda *_: fail())
    if mode == "identity":
        original_identity = BackupTracker.record_identity

        def identity(self, value):
            if value["status"] == "completed":
                fail()
            return original_identity(self, value)

        monkeypatch.setattr(BackupTracker, "record_identity", identity)
    if mode == "diagnostic":

        def broken_sink(*_):
            raise OSError("CANARY_SECONDARY_DIAGNOSTIC")

        monkeypatch.setattr(BackupTracker, "record_failure", broken_sink)


def main():
    from pytest import MonkeyPatch

    arguments = sys.argv[1:]
    private = Path(arguments[arguments.index("--private") + 1])
    fixture = NormalHTTP(MonkeyPatch(), private.parent / "http")
    import faulthandler

    trace = (private.parent / ("fixture-trace-" + os.environ.get("RANK", "host") + ".log")).open(
        "w"
    )
    faulthandler.dump_traceback_later(35, repeat=True, file=trace)
    fixture.install()
    install_failures(private, fixture)
    import runpy

    package = Path(arguments[arguments.index("--package") + 1])
    try:
        try:
            runpy.run_path(str(package / "worker.py"), run_name="__main__")
        except SystemExit as result:
            return result.code
        return 0
    finally:
        fixture.save()
        faulthandler.cancel_dump_traceback_later()
        trace.close()


if __name__ == "__main__":
    raise SystemExit(main())
