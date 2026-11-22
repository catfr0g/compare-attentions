"""Journal integrity and exact eligibility regression checks."""

# pylint: disable=import-outside-toplevel
import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def seed():
    import colab_api_client as client

    return {
        "schema": 1,
        "runtimeId": "one-t4",
        "requestId": str(uuid.uuid4()),
        "name": "runtimes/one-t4",
        "runtimeSpec": client.SPEC,
        "state": "create_unknown",
    }


def test_journal_alias_path_cannot_bypass_single_writer(tmp_path):
    import colab_api_client as client
    import colab_api_journal as journal

    path = tmp_path / "one.json"
    (tmp_path / "nested").mkdir()
    alias = tmp_path / "nested/../one.json"
    with journal.locked(path):
        with pytest.raises(client.ApiError, match="JOURNAL_BUSY"):
            with journal.locked(alias):
                pytest.fail("alias bypassed lock")


def test_journal_atomic_no_clobber_and_single_writer(tmp_path):
    import colab_api_client as client
    import colab_api_journal as journal

    path = tmp_path / "one.json"
    state = seed()
    with journal.locked(path):
        journal.save(path, state, initial=True)
        with pytest.raises(client.ApiError, match="JOURNAL_BUSY"):
            with journal.locked(path):
                pytest.fail("second writer acquired lock")
        with pytest.raises(client.ApiError, match="JOURNAL_EXISTS"):
            journal.save(
                path,
                {**state, "runtimeId": "different", "name": "runtimes/different"},
                initial=True,
            )
    assert json.loads(path.read_text()) == state
    with journal.locked(path):
        assert journal.load(path) == state
    assert not list(tmp_path.glob(".runtime-*"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtimeId", "Bad"),
        ("requestId", "bad"),
        ("name", "runtimes/foreign"),
        ("runtimeSpec", {}),
        ("state", "unapproved"),
        ("operation", "operations/../one"),
        ("connectionInfo", {"token": "SECRET"}),
        ("failure", {"code": 8, "message": "SECRET"}),
    ],
)
def test_untrusted_journal_rejected_before_lookup(tmp_path, monkeypatch, field, value):
    import colab_api_runtime as runtime

    path = tmp_path / "one.json"
    path.write_text(json.dumps({**seed(), field: value}))
    monkeypatch.setattr(runtime.client, "get", lambda *_a: pytest.fail("invalid journal lookup"))
    monkeypatch.setattr(
        runtime.client, "operation", lambda *_a: pytest.fail("invalid operation lookup")
    )
    with pytest.raises(runtime.client.ApiError):
        runtime.reconcile(path, "FAKE")


@pytest.mark.parametrize(
    "eligible,override",
    [
        (1, {}),
        ("true", {}),
        (False, {}),
        (True, {"shape": "SHAPE_HIGHMEM"}),
        (True, {"accelerator": "L4"}),
        (True, {"variant": "VARIANT_CPU"}),
    ],
)
def test_create_exact_standard_t4_and_strict_true_before_journal(
    tmp_path, monkeypatch, eligible, override
):
    import colab_api_runtime as runtime

    path = tmp_path / "one.json"
    monkeypatch.setattr(
        runtime.client,
        "get",
        lambda *_a: {
            "runtimeSpecs": [{"key": {**runtime.client.SPEC, **override}, "eligible": eligible}]
        },
    )
    monkeypatch.setattr(
        runtime.client, "manage", lambda *_a, **_kw: pytest.fail("ineligible allocation")
    )
    with pytest.raises(runtime.client.ApiError, match="T4_STANDARD_NOT_ELIGIBLE"):
        runtime.create(path, "FAKE", execute=True)
    assert not path.exists()
