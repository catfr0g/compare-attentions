"""No-network SDK transport emulator. All identities/results are engineering fixtures."""

import base64
import io
import json
import socket
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx2 as httpx
import requests


class FixtureHTTP:
    """Intercept prepared HTTP, retaining real SDK serialization and Authorization."""

    def __init__(self, monkeypatch, scratch):
        self.calls, self.tasks, self.files = [], {}, {}
        self.scratch = Path(scratch)
        self.head, self.revision = "1" * 40, "2" * 40
        self.monkeypatch = monkeypatch
        self.project = "3" * 32
        self.fail_commit = False

    def install(self):
        from huggingface_hub import close_session, set_client_factory

        def no_network(*_args, **_kwargs):
            raise AssertionError("NETWORK_FORBIDDEN_ENGINEERING_FIXTURE")

        self.monkeypatch.setattr(socket.socket, "connect", no_network)
        self.monkeypatch.setattr(socket.socket, "connect_ex", no_network)

        close_session()
        set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(self.hf)))
        self.monkeypatch.setattr(requests.sessions.Session, "send", self.requests_send)

    def hf(self, request):
        path = unquote(request.url.path)
        self.calls.append((request.method, path, bool(request.headers.get("authorization"))))
        assert request.headers.get("authorization") == "Bearer hf_FIXTURE_ONLY"
        if "/tree/" in path:
            rows = [
                {"type": "file", "path": n, "oid": "a" * 40, "size": len(v)}
                for n, v in self.files.items()
            ]
            return httpx.Response(200, json=rows)
        if "/preupload/" in path:
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "files": [
                        {"path": f["path"], "uploadMode": "regular", "shouldIgnore": False}
                        for f in payload["files"]
                    ]
                },
            )
        if "/commit/" in path:
            records = [json.loads(line) for line in request.content.splitlines()]
            assert records[0]["value"]["parentCommit"] == self.head
            for row in records[1:]:
                if row["key"] == "file":
                    self.files[row["value"]["path"]] = base64.b64decode(row["value"]["content"])
            self.head = self.revision
            if self.fail_commit:
                raise RuntimeError("SECRET_CANARY_NOT_REAL")
            return httpx.Response(
                200,
                json={
                    "commitOid": self.revision,
                    "commitUrl": "https://huggingface.co/fixture/repo/commit/" + self.revision,
                },
            )
        return httpx.Response(
            200, json={"id": "fixture/repo", "sha": self.head, "private": True, "siblings": []}
        )

    def requests_send(self, request, **_kwargs):
        path = unquote(urlparse(request.url).path)
        self.calls.append((request.method, path, bool(request.headers.get("Authorization"))))
        self.save()
        response = requests.Response()
        response.status_code, response.url, response.request = 200, request.url, request
        if "/resolve/" in path:
            assert request.headers.get("Authorization") == "Bearer hf_FIXTURE_ONLY"
            name = path.split("/resolve/", 1)[1].split("/", 1)[1]
            response._content = self.files[name]
        elif "fixture-files" in request.url:
            name = path.lstrip("/")
            if request.method in {"POST", "PUT"}:
                from email.parser import BytesParser

                body = request.body.read() if hasattr(request.body, "read") else request.body
                if "multipart/" in request.headers.get("Content-Type", ""):
                    message = BytesParser().parsebytes(
                        ("Content-Type: " + request.headers["Content-Type"] + "\r\n\r\n").encode()
                        + body
                    )
                    part = message.get_payload()[0]
                    name = (
                        part.get_filename() or part.get_param("name", header="content-disposition")
                    ).lstrip("/")
                    body = part.get_payload(decode=True)
                self.files[name] = body
                response._content = b"OK"
            else:
                response._content = self.files[name]
            response.headers["Content-Length"] = str(len(response._content))
        else:
            raw = request.body or b"{}"
            if isinstance(raw, bytes):
                raw = raw.decode()
            if path.endswith("events.add_batch"):
                payload = {"events": [json.loads(line) for line in raw.splitlines() if line]}
            else:
                payload = json.loads(raw) if raw and raw.startswith("{") else {}
            data = self.clearml(path.rsplit("/", 1)[-1], payload, request)
            response._content = json.dumps(
                {
                    "meta": {
                        "result_code": 200,
                        "result_msg": "OK",
                        "endpoint": {
                            "name": path.rsplit("/", 1)[-1],
                            "requested_version": "2.20",
                            "actual_version": "2.20",
                        },
                    },
                    "data": data,
                }
            ).encode()
        response.raw = io.BytesIO(response._content)
        return response

    def clearml(self, endpoint, payload, request):
        if endpoint == "auth.login":
            body = {
                "exp": int(time.time()) + 36000,
                "iat": int(time.time()),
                "identity": {"user": "fixture", "tenant": "fixture"},
                "api_version": "2.20",
            }

            def encode(value):
                return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

            return {"token": encode({"alg": "HS256"}) + "." + encode(body) + ".c2lnbmF0dXJl"}
        assert request.headers.get("Authorization", "").startswith("Bearer ")
        if endpoint == "projects.get_by_id":
            return {"project": {"id": self.project, "name": "compare-attention"}}
        if endpoint == "projects.get_all":
            return {"projects": [{"id": self.project, "name": "compare-attention"}]}
        if endpoint == "tasks.create":
            import hashlib

            identifier = hashlib.sha256(str(self.scratch).encode()).hexdigest()[:32]
            self.tasks[identifier] = {
                **payload,
                "id": identifier,
                "project": self.project,
                "status": "created",
                "type": "inference",
                "execution": {"artifacts": []},
                "hyperparams": {},
                "configuration": {},
                "tags": [],
                "system_tags": [],
            }
            return {"id": identifier}
        if endpoint == "tasks.get_all":
            return {
                "tasks": [
                    {k: v for k, v in task.items() if not k.startswith("_fixture_")}
                    for task in self.tasks.values()
                ]
            }
        if endpoint == "tasks.get_by_id":
            return {
                "task": {
                    k: v
                    for k, v in self.tasks[payload["task"]].items()
                    if not k.startswith("_fixture_")
                }
            }
        if endpoint == "events.get_task_events":
            assert payload["event_type"] == "training_stats_scalar"
            assert payload["order"] == "asc" and payload["batch_size"] == 10000
            rows = self.tasks[payload["task"]].get("_fixture_scalar_events", [])
            offset = int((payload.get("scroll_id") or "fixture-0").split("-")[-1])
            page = rows[offset : offset + 2]
            return {
                "events": page,
                "total": len(rows),
                "scroll_id": "fixture-" + str(offset + len(page)),
            }
        if endpoint in {"tasks.edit", "tasks.set_hyper_params"}:
            task = self.tasks[payload["task"]]
            task.update({k: v for k, v in payload.items() if k != "task"})
            return {"updated": 1, "fields": {}}
        if endpoint == "tasks.get_hyper_params":
            return {"params": []}
        if endpoint in {"tasks.started", "tasks.completed", "tasks.stopped"}:
            self.tasks[payload["task"]]["status"] = {
                "tasks.started": "in_progress",
                "tasks.completed": "completed",
                "tasks.stopped": "stopped",
            }[endpoint]
            return {"updated": 1, "fields": {}}
        if endpoint == "events.add_batch":
            for event in payload["events"]:
                if event.get("type") == "training_stats_scalar":
                    self.tasks[event["task"]].setdefault("_fixture_scalar_events", []).append(event)
                    self.tasks[event["task"]].setdefault("last_metrics", {}).setdefault(
                        event["metric"], {}
                    )[event["variant"]] = {
                        "metric": event["metric"],
                        "variant": event["variant"],
                        "value": event["value"],
                        "min_value": event["value"],
                        "max_value": event["value"],
                    }
            return {"added": len(payload["events"]), "errors": 0}
        if endpoint == "tasks.add_or_update_artifacts":
            self.tasks[payload["task"]]["execution"]["artifacts"] = payload["artifacts"]
            return {"added": len(payload["artifacts"]), "updated": 0}
        if endpoint == "tasks.set_requirements":
            return {"updated": 1, "fields": {}}
        if endpoint == "tasks.failed":
            return {"updated": 1, "fields": {}}
        if endpoint in {
            "tasks.ping",
            "debug.ping",
            "users.get_current_user",
            "workers.register",
            "workers.unregister",
            "workers.status_report",
        }:
            return {}
        raise AssertionError("Unimplemented fixture HTTP endpoint: " + endpoint)

    def save(self):
        (self.scratch / "http-calls.json").write_text(json.dumps(self.calls))
