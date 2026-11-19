"""Pagination negatives below installed SDK DTO/encoder/HTTP transport."""

import base64
import io
import json
import os
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import training_completion as completion


def fixture(monkeypatch, pages):
    monkeypatch.setenv("CLEARML_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("CLEARML_DISABLE_VAULT_SUPPORT", "1")
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "0")
    import requests
    from clearml import Task
    from clearml.backend_api.session import Session

    # SDK caches this class setting at import; transport remains intercepted.
    # Restore it after this explicitly online synthetic HTTP fixture.
    monkeypatch.setattr(Session, "_offline_mode", False)
    requests_seen = []

    def send(_session, request, **_kwargs):
        assert urlsplit(request.url).hostname == "fixture.invalid"
        endpoint = request.url.rsplit("/", 1)[-1]
        payload = json.loads(request.body or b"{}")
        if endpoint == "auth.login":

            def encode(value):
                return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

            data = {
                "token": encode({"alg": "HS256"})
                + "."
                + encode(
                    {
                        "exp": int(time.time()) + 3600,
                        "iat": int(time.time()),
                        "api_version": "2.20",
                        "identity": {"user": "synthetic", "tenant": "synthetic"},
                    }
                )
                + ".c3ludGhldGlj"
            }
        else:
            assert endpoint == "events.get_task_events"
            assert request.headers.get("Authorization", "").startswith("Bearer ")
            assert payload["task"] == "synthetic"
            assert payload["event_type"] == "training_stats_scalar"
            assert payload["order"] == "asc" and payload["batch_size"] == 10000
            requests_seen.append(payload)
            assert len(requests_seen) <= len(pages), "pagination exceeded finite fixture"
            data = pages[len(requests_seen) - 1]
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(
            {
                "meta": {
                    "result_code": 200,
                    "result_msg": "OK",
                    "endpoint": {
                        "name": endpoint,
                        "requested_version": "2.20",
                        "actual_version": "2.20",
                    },
                },
                "data": data,
            }
        ).encode()
        response.raw = io.BytesIO(response._content)
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    session = Session(
        api_key="SYNTHETIC_KEY", secret_key="SYNTHETIC_SECRET", host="https://fixture.invalid"
    )
    observed = SimpleNamespace(
        reload=lambda: None,
        get_status=lambda: "completed",
        get_parameters_as_dict=lambda: {"General": {}},
    )
    monkeypatch.setattr(Task, "get_task", lambda **_: observed)
    monkeypatch.setattr(Task, "_get_default_session", lambda: session)
    return requests_seen


def row(iteration):
    return {
        "metric": "train",
        "variant": "loss",
        "iter": iteration,
        "value": 1.0,
        "task": "synthetic",
    }


def test_real_sdk_unsampled_pagination_keeps_every_raw_row(monkeypatch):
    pages = [
        {"events": [row(1)], "total": 2, "scroll_id": "first"},
        {"events": [row(2)], "total": 2, "scroll_id": "second"},
    ]
    requests_seen = fixture(monkeypatch, pages)
    value = completion.TrainingServiceTracker._production_readback("synthetic")
    assert value["events"] == [row(1), row(2)]
    assert value["total"] == 2
    assert [r.get("scroll_id") for r in requests_seen] == [None, "first"]


@pytest.mark.parametrize(
    "kind",
    ["empty", "total_drift", "overshoot", "cursor", "duplicate", "float_total", "bool_total"],
)
def test_actual_sdk_paginated_response_rejects_incomplete_or_coerced_metadata(monkeypatch, kind):
    pages = [
        {"events": [row(1)], "total": 2, "scroll_id": "first"},
        {"events": [row(2)], "total": 2, "scroll_id": "second"},
    ]
    if kind == "empty":
        pages[1]["events"] = []
    elif kind == "total_drift":
        pages[1]["total"] = 1
    elif kind == "overshoot":
        pages[0]["total"] = 0
    elif kind == "cursor":
        pages[0]["scroll_id"] = None
    elif kind == "duplicate":
        pages[1]["events"] = [row(1)]
    elif kind == "float_total":
        pages[0]["total"] = 2.0
    elif kind == "bool_total":
        pages[0]["total"] = True
    fixture(monkeypatch, pages)
    with pytest.raises(ValueError):
        completion.TrainingServiceTracker._production_readback("synthetic")
