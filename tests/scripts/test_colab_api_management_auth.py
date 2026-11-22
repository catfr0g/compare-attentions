"""Full-scope OAuth is explicit and never inferred from readonly."""

# pylint: disable=import-outside-toplevel
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


@pytest.mark.parametrize("granted", [True, False])
def test_management_login_scope_explicit_and_missing_fails(tmp_path, monkeypatch, granted):
    import urllib.parse

    import colab_api_auth as auth

    monkeypatch.setattr(
        auth, "load_client", lambda _p: {"client_id": "fake", "client_secret": "fake"}
    )
    monkeypatch.setattr(auth, "receive_code", lambda *_a: "fake")
    observed = []
    monkeypatch.setattr(auth.webbrowser, "open", lambda url: observed.append(url) or True)
    monkeypatch.setattr(
        auth,
        "exchange",
        lambda _f: {
            "access_token": "fake",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": auth.MANAGE if granted else auth.READONLY,
        },
    )
    target = tmp_path / "management.json"
    if granted:
        assert auth.login(None, target, management=True)["scopes"] == [auth.MANAGE]
    else:
        with pytest.raises(auth.ApiError, match="SCOPE_REQUIRED"):
            auth.login(None, target, management=True)
        assert not target.exists()
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(observed[0]).query)["scope"] == [auth.MANAGE]


def test_login_result_only_whitelisted_scopes(tmp_path, monkeypatch):
    import colab_api_auth as auth

    monkeypatch.setattr(
        auth, "load_client", lambda _p: {"client_id": "fake", "client_secret": "fake"}
    )
    monkeypatch.setattr(auth, "receive_code", lambda *_a: "fake")
    monkeypatch.setattr(auth.webbrowser, "open", lambda _u: True)
    monkeypatch.setattr(
        auth,
        "exchange",
        lambda _f: {
            "access_token": "fake",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": auth.MANAGE + " SECRET",
        },
    )
    result = auth.login(None, tmp_path / "management.json", management=True)
    assert result["scopes"] == [auth.MANAGE]
    assert "SECRET" not in json.dumps(result)


def test_management_scope_status_and_access(tmp_path):
    import colab_api_auth as auth

    path = tmp_path / "token.json"
    auth.save_token(
        path,
        {
            "type": "authorized_user",
            "access_token": "FAKE",
            "expires_at": time.time() + 3600,
            "scopes": [auth.MANAGE, "other"],
        },
    )
    status = auth.status(path)
    assert status["scope_valid"] is True
    assert status["management_valid"] is True
    assert status["scopes"] == [auth.MANAGE]
    assert auth.access_token(path, None, required_scope=auth.MANAGE) == "FAKE"
    auth.save_token(
        path,
        {
            "type": "authorized_user",
            "access_token": "FAKE",
            "expires_at": time.time() + 3600,
            "scopes": [auth.READONLY],
        },
        overwrite=True,
    )
    assert auth.status(path)["management_valid"] is False
    with pytest.raises(auth.ApiError, match="SCOPE_REQUIRED"):
        auth.access_token(path, None, required_scope=auth.MANAGE)
    assert auth.access_token(path, None) == "FAKE"


def test_management_refresh_scope_preserved_and_reduction_rejected(tmp_path, monkeypatch):
    import colab_api_auth as auth

    path = tmp_path / "token.json"
    previous = {
        "type": "authorized_user",
        "access_token": "OLD",
        "refresh_token": "FAKE",
        "expires_at": 0,
        "scopes": [auth.MANAGE],
        "client_id": "fake",
    }
    auth.save_token(path, previous)
    monkeypatch.setattr(
        auth, "load_client", lambda _p: {"client_id": "fake", "client_secret": "fake"}
    )
    raw = {"access_token": "NEW", "token_type": "Bearer", "expires_in": 3600}
    monkeypatch.setattr(auth, "exchange", lambda _f: raw)
    assert auth.access_token(path, None, required_scope=auth.MANAGE) == "NEW"
    assert json.loads(path.read_text())["scopes"] == [auth.MANAGE]
    auth.save_token(path, previous, overwrite=True)
    raw["scope"] = auth.READONLY
    with pytest.raises(auth.ApiError, match="SCOPE_REQUIRED"):
        auth.access_token(path, None, required_scope=auth.MANAGE)
    assert json.loads(path.read_text())["access_token"] == "OLD"
