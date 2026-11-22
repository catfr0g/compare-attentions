"""Private Desktop OAuth credentials and offline metadata; no Google SDK."""

import base64
import hashlib
import hmac
import http.server
import json
import os
import secrets
import tempfile
import time
import urllib.parse
import webbrowser
from pathlib import Path

from colab_api_client import ApiError, request_json

READONLY = "https://www.googleapis.com/auth/colaboratory.readonly"
MANAGE = "https://www.googleapis.com/auth/colaboratory"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
REPO = Path(__file__).resolve().parents[1]


def private_path(path):
    if path is None:
        raise ApiError(
            "CLIENT_REQUIRED", action="Provide --client-file: private Desktop OAuth JSON"
        )
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(REPO) or resolved.name == ".env":
        raise ApiError("PRIVATE_PATH_REQUIRED", action="Keep OAuth files outside the repository")
    return resolved


def read_json(path, error):
    resolved = private_path(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError):
        action = (
            "Run auth-login with --client-file pointing to a private Desktop JSON"
            if error == "AUTH_REQUIRED"
            else "Provide valid private Desktop JSON --client-file"
        )
        raise ApiError(error, action=action) from None


def load_client(path):
    raw = read_json(path, "CLIENT_REQUIRED").get("installed", {})
    if (
        not isinstance(raw, dict)
        or not all(
            isinstance(raw.get(name), str) and raw[name]
            for name in ("client_id", "client_secret", "project_id")
        )
        or raw.get("auth_uri") not in (AUTH_URI, "https://accounts.google.com/o/oauth2/auth")
        or raw.get("token_uri") != TOKEN_URI
    ):
        raise ApiError("INVALID_DESKTOP_CLIENT")
    return raw


def save_token(path, token, *, overwrite=False):
    target = private_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise ApiError("TOKEN_EXISTS", action="Use --overwrite explicitly")
    descriptor, temporary = tempfile.mkstemp(prefix=".oauth-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o600)
            json.dump(token, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # Hard-link provides atomic no-clobber on both NTFS and POSIX.
            try:
                os.link(temporary, target)
            except FileExistsError:
                raise ApiError("TOKEN_EXISTS") from None
    except OSError:
        raise ApiError("TOKEN_WRITE_FAILED") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_token(path):
    token = read_json(path, "AUTH_REQUIRED")
    if token.get("type") != "authorized_user":
        raise ApiError("INVALID_TOKEN_TYPE")
    return token


def valid_scope(token, required_scope=READONLY):
    scopes = token.get("scopes", [])
    return (
        isinstance(scopes, list)
        and required_scope in (READONLY, MANAGE)
        and (required_scope in scopes or (required_scope == READONLY and MANAGE in scopes))
    )


def status(path):
    token = load_token(path)
    scopes = token.get("scopes", [])
    expiry = token.get("expires_at")
    return {
        "present": True,
        "scope_valid": valid_scope(token),
        "management_valid": valid_scope(token, MANAGE),
        "scopes": [s for s in (READONLY, MANAGE) if isinstance(scopes, list) and s in scopes],
        "expires_at": expiry if type(expiry) in (float, int) else None,
        "refresh_available": bool(token.get("refresh_token")),
    }


def exchanged(raw, previous, *, required_scope=READONLY):
    scopes = raw.get("scope", " ".join(previous.get("scopes", [])))
    if not isinstance(scopes, str):
        raise ApiError("SCOPE_REQUIRED")
    token = {**previous, "scopes": scopes.split()}
    if not valid_scope(token, required_scope):
        raise ApiError("SCOPE_REQUIRED", action="Repeat auth-login granting Colab readonly scope")
    token_type = raw.get("token_type")
    if not isinstance(token_type, str):
        raise ApiError("INVALID_TOKEN_RESPONSE")
    if (
        not isinstance(raw.get("access_token"), str)
        or not raw["access_token"]
        or token_type.lower() != "bearer"
        or type(raw.get("expires_in")) not in (int, float)
        or not 0 < raw["expires_in"] <= 86400
    ):
        raise ApiError("INVALID_TOKEN_RESPONSE")
    token.update(access_token=raw["access_token"], expires_at=time.time() + raw["expires_in"])
    if isinstance(raw.get("refresh_token"), str) and raw["refresh_token"]:
        token["refresh_token"] = raw["refresh_token"]
    return token


def exchange(fields):
    return request_json(
        TOKEN_URI,
        data=urllib.parse.urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


def access_token(path, client_path, *, required_scope=READONLY):
    token = load_token(path)
    if not valid_scope(token, required_scope):
        raise ApiError("SCOPE_REQUIRED", action="Repeat auth-login granting Colab readonly scope")
    expiry = token.get("expires_at", 0)
    if (
        type(expiry) in (int, float)
        and expiry > time.time() + 60
        and isinstance(token.get("access_token"), str)
        and token["access_token"]
    ):
        return token["access_token"]
    client = load_client(client_path)
    if token.get("client_id") != client["client_id"]:
        raise ApiError("CLIENT_MISMATCH")
    if not isinstance(token.get("refresh_token"), str) or not token["refresh_token"]:
        raise ApiError("AUTH_REQUIRED", action="Repeat auth-login: refresh token absent")
    refreshed = exchanged(
        exchange(
            {
                "grant_type": "refresh_token",
                "refresh_token": token["refresh_token"],
                "client_id": client["client_id"],
                "client_secret": client["client_secret"],
            }
        ),
        token,
        required_scope=MANAGE if valid_scope(token, MANAGE) else required_scope,
    )
    save_token(path, refreshed, overwrite=True)
    return refreshed["access_token"]


class LoopbackServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(2)
        return connection, address

    def handle_error(self, request, client_address):
        # Never let stdlib traceback/logging expose callback request data.
        pass


def receive_code(server, state, deadline):
    received = {}

    class Callback(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # pylint: disable=redefined-builtin
            pass

        def do_GET(self):  # pylint: disable=invalid-name
            self.connection.settimeout(2)
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != "/callback":
                self.send_error(404)
                return
            query = urllib.parse.parse_qs(parsed.query)
            states = query.get("state", [])
            valid = (
                time.monotonic() < deadline
                and len(states) == 1
                and states[0].isascii()
                and hmac.compare_digest(states[0], state)
            )
            if not valid:
                received["error"] = "OAUTH_STATE_INVALID"
            elif "error" in query:
                received["error"] = "OAUTH_DENIED"
            elif len(query.get("code", [])) != 1 or not query["code"][0]:
                received["error"] = "OAUTH_CALLBACK_INVALID"
            else:
                received["code"] = query["code"][0]
            self.send_response(200 if "code" in received else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(b"Authorization callback received. Return to the terminal.")

    server.RequestHandlerClass = Callback
    while not received and time.monotonic() < deadline:
        server.timeout = min(1, max(0.01, deadline - time.monotonic()))
        server.handle_request()
    if "code" not in received:
        raise ApiError(received.get("error", "OAUTH_TIMEOUT"))
    return received["code"]


def login(client_path, token_path, *, overwrite=False, no_browser=False, management=False):
    client = load_client(client_path)
    token_path = private_path(token_path)
    if token_path.exists() and not overwrite:
        raise ApiError("TOKEN_EXISTS", action="Use --overwrite explicitly")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    state = secrets.token_urlsafe(32)
    deadline = time.monotonic() + 300
    with LoopbackServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler) as server:
        redirect = f"http://127.0.0.1:{server.server_port}/callback"
        url = (
            AUTH_URI
            + "?"
            + urllib.parse.urlencode(
                {
                    "client_id": client["client_id"],
                    "redirect_uri": redirect,
                    "response_type": "code",
                    "scope": MANAGE if management else READONLY,
                    "access_type": "offline",
                    "prompt": "consent",
                    "state": state,
                    "code_challenge": challenge.decode(),
                    "code_challenge_method": "S256",
                }
            )
        )
        if no_browser or not webbrowser.open(url):
            print(url, flush=True)
        code = receive_code(server, state, deadline)
    token = exchanged(
        exchange(
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": client["client_id"],
                "client_secret": client["client_secret"],
                "redirect_uri": redirect,
                "code_verifier": verifier,
            }
        ),
        {"type": "authorized_user", "client_id": client["client_id"], "scopes": []},
        required_scope=MANAGE if management else READONLY,
    )
    save_token(token_path, token, overwrite=overwrite)
    return {
        "authenticated": True,
        "scopes": [s for s in (READONLY, MANAGE) if s in token["scopes"]],
    }
