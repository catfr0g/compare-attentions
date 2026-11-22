"""OAuth protocol checks use only generated temporary private files."""

# pylint: disable=import-outside-toplevel
import importlib
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def auth():
    return importlib.import_module("colab_api_auth")


def private_client(tmp_path):
    core = auth()
    path = tmp_path / "desktop.json"
    path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "test-client",
                    "client_secret": "SECRET",
                    "project_id": "test-project",
                    "auth_uri": core.AUTH_URI,
                    "token_uri": core.TOKEN_URI,
                    "redirect_uris": ["http://localhost"],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "endpoint,accepted",
    [
        ("https://accounts.google.com/o/oauth2/auth", True),
        ("https://accounts.google.com/o/oauth2/v2/auth", True),
        ("https://attacker.invalid/o/oauth2/auth", False),
    ],
)
def test_desktop_client_google_authorization_endpoint_compatibility(tmp_path, endpoint, accepted):
    core = auth()
    path = private_client(tmp_path)
    data = json.loads(path.read_text())
    data["installed"]["auth_uri"] = endpoint
    path.write_text(json.dumps(data))
    if accepted:
        assert core.load_client(path)["auth_uri"] == endpoint
    else:
        with pytest.raises(core.ApiError, match="INVALID_DESKTOP_CLIENT"):
            core.load_client(path)


def test_private_path_guard_and_atomic_no_overwrite(tmp_path):
    core = auth()
    target = tmp_path / "private/token.json"
    core.save_token(target, {"access_token": "SECRET"})
    assert json.loads(target.read_text())["access_token"] == "SECRET"
    with pytest.raises(core.ApiError, match="TOKEN_EXISTS"):
        core.save_token(target, {"access_token": "new"})
    for path in (
        Path(__file__).resolve().parents[2] / "local/token.json",
        Path(__file__).resolve().parents[2] / ".env",
    ):
        with pytest.raises(core.ApiError, match="PRIVATE_PATH_REQUIRED"):
            core.private_path(path)


def test_status_offline_only_metadata_and_scope_fail_closed(tmp_path):
    core = auth()
    target = tmp_path / "token.json"
    core.save_token(
        target,
        {
            "type": "authorized_user",
            "access_token": "SECRET",
            "refresh_token": "SECRET",
            "expires_at": time.time() + 300,
            "scopes": [core.READONLY],
        },
    )
    status = core.status(target)
    assert status["scope_valid"] is True
    assert "SECRET" not in json.dumps(status)
    for scopes in ([], ["other"]):
        core.save_token(
            target,
            {
                "type": "authorized_user",
                "access_token": "SECRET",
                "expires_at": time.time() + 300,
                "scopes": scopes,
            },
            overwrite=True,
        )
        with pytest.raises(core.ApiError, match="SCOPE_REQUIRED"):
            core.access_token(target, None)


def test_refresh_protocol_and_scope_reduction_rejected(tmp_path, monkeypatch):
    core = auth()
    target = tmp_path / "token.json"
    client = private_client(tmp_path)
    token = {
        "type": "authorized_user",
        "refresh_token": "REFRESH",
        "access_token": "OLD",
        "expires_at": 0,
        "scopes": [core.READONLY],
        "client_id": "test-client",
    }
    core.save_token(target, token)
    requests = []

    def exchange(url, **kwargs):
        requests.append((url, kwargs))
        return {"access_token": "NEW", "expires_in": 3600, "token_type": "Bearer"}

    monkeypatch.setattr(core, "request_json", exchange)
    assert core.access_token(target, client) == "NEW"
    assert requests[0][0] == core.TOKEN_URI
    assert b"grant_type=refresh_token" in requests[0][1]["data"]
    assert json.loads(target.read_text())["refresh_token"] == "REFRESH"
    core.save_token(target, token, overwrite=True)
    monkeypatch.setattr(
        core,
        "request_json",
        lambda *_a, **_k: {
            "access_token": "NEW",
            "expires_in": 3600,
            "scope": "other",
            "token_type": "Bearer",
        },
    )
    with pytest.raises(core.ApiError, match="SCOPE_REQUIRED"):
        core.access_token(target, client)
    assert json.loads(target.read_text())["access_token"] == "OLD"


# Complete OAuth protocol is one vertical fixture: state, browser, callback, exchange.
@pytest.mark.parametrize("bad_state", [False, True, "nonascii", "duplicate", "denied"])
def test_loopback_login_pkce_state_and_secret_redaction(tmp_path, monkeypatch, capsys, bad_state):
    # pylint: disable=too-many-locals,too-many-statements
    import base64
    import hashlib
    import threading
    import urllib.parse
    import urllib.request

    core = auth()
    client_path = private_client(tmp_path)
    target = tmp_path / "token.json"
    observed = {}
    threads = []

    def browser(url):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        observed.update({key: values[0] for key, values in query.items()})
        assert observed["code_challenge_method"] == "S256"
        assert observed["scope"] == core.READONLY
        assert observed["access_type"] == "offline"
        assert "SECRET" not in url

        def callback():
            state = "WRONG" if bad_state else observed["state"]
            if bad_state == "nonascii":
                state = "\u0441\u0435\u043a\u0440\u0435\u0442"
            elif bad_state in ("duplicate", "denied"):
                state = observed["state"]
            fields = {"state": state, "code": "PRIVATE_CODE"}
            if bad_state == "denied":
                fields["error"] = "PRIVATE_ERROR"
            callback_url = observed["redirect_uri"] + "?" + urllib.parse.urlencode(fields)
            if bad_state == "duplicate":
                callback_url += "&state=WRONG"
            try:
                with urllib.request.urlopen(callback_url, timeout=5) as response:
                    observed["callback_status"] = response.status
            except urllib.error.HTTPError as error:
                observed["callback_status"] = error.code

        thread = threading.Thread(target=callback)
        thread.start()
        threads.append(thread)
        return True

    def exchange(url, **kwargs):
        assert not bad_state
        assert url == core.TOKEN_URI
        fields = urllib.parse.parse_qs(kwargs["data"].decode())
        verifier = fields["code_verifier"][0]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(
            b"="
        )
        assert challenge.decode() == observed["code_challenge"]
        assert fields["code"] == ["PRIVATE_CODE"]
        assert fields["redirect_uri"] == [observed["redirect_uri"]]
        return {
            "access_token": "PRIVATE_ACCESS",
            "refresh_token": "PRIVATE_REFRESH",
            "token_type": "Bearer",
            "scope": core.READONLY,
            "expires_in": 3600,
        }

    original_receive = core.receive_code
    monkeypatch.setattr(
        core,
        "receive_code",
        lambda server, state, _deadline: original_receive(server, state, time.monotonic() + 0.3),
    )
    monkeypatch.setattr(core.webbrowser, "open", browser)
    monkeypatch.setattr(core, "request_json", exchange)
    if bad_state:
        expected = "OAUTH_DENIED" if bad_state == "denied" else "OAUTH_STATE_INVALID"
        with pytest.raises(core.ApiError, match=expected):
            core.login(client_path, target)
        assert not target.exists()
    else:
        assert core.login(client_path, target)["authenticated"] is True
        assert json.loads(target.read_text())["access_token"] == "PRIVATE_ACCESS"
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    captured = capsys.readouterr()
    assert "PRIVATE_" not in captured.out + captured.err
    assert "SECRET" not in captured.out + captured.err


def test_loopback_idle_socket_is_bounded_and_errors_are_silent(capsys):
    import socket

    core = auth()
    with core.LoopbackServer(("127.0.0.1", 0), core.http.server.BaseHTTPRequestHandler) as server:
        server.timeout = 0.05
        with socket.create_connection(server.server_address, timeout=1) as connection:
            accepted, _address = server.get_request()
            assert 0 < accepted.gettimeout() <= 2
            accepted.close()
            assert connection.fileno() >= 0
        server.handle_error(None, ("127.0.0.1", 1))
        with socket.create_connection(server.server_address, timeout=1) as slow:
            slow.sendall(b"GET /callback?code=PRIVATE ")
            before = time.monotonic()
            with pytest.raises(core.ApiError, match="OAUTH_TIMEOUT"):
                core.receive_code(server, "state", time.monotonic() + 0.05)
            assert time.monotonic() - before < 1
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("token_type", [None, 1, [], {}])
def test_invalid_token_type_response_is_sanitized(token_type):
    core = auth()
    with pytest.raises(core.ApiError, match="INVALID_TOKEN_RESPONSE"):
        core.exchanged(
            {
                "access_token": "PRIVATE",
                "expires_in": 3600,
                "scope": core.READONLY,
                "token_type": token_type,
            },
            {},
        )


def test_login_missing_client_fails_before_browser(tmp_path, monkeypatch):
    core = auth()
    monkeypatch.setattr(core.webbrowser, "open", lambda *_a: pytest.fail("browser opened"))
    with pytest.raises(core.ApiError, match="CLIENT_REQUIRED"):
        core.login(tmp_path / "absent.json", tmp_path / "token.json")
