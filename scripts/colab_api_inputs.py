"""Headless private input transport. No credentials enter code, URLs or receipts."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from colab_api_client import NoRedirect, request_json
from colab_package import checksum
from colab_services import KEYS


def publish_archive(archive, sha256, values, *, api_factory=None, download=None):
    """One immutable prefix; verify private pinned metadata AND downloaded bytes."""
    archive = Path(archive)
    if checksum(archive) != sha256 or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("ARCHIVE_SHA_MISMATCH")
    prefix = "launch-inputs/" + sha256 + "/"
    path = prefix + "colab-package.zip"
    try:
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ.pop(key, None)
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            if api_factory is None or download is None:
                from huggingface_hub import HfApi, hf_hub_download  # pylint: disable=import-outside-toplevel

                api_factory, download = api_factory or HfApi, download or hf_hub_download
            repo, token = values["HF_REPO"], values["HF_TOKEN"]
            api = api_factory(endpoint="https://huggingface.co", token=token)
            head = api.model_info(repo, revision="main")
            if head.private is not True or not re.fullmatch(r"[a-f0-9]{40}", head.sha):
                raise ValueError
            listing = api.list_repo_files(repo, revision=head.sha, repo_type="model")
            if any(name.startswith(prefix) for name in listing):
                raise ValueError("PREFIX_EXISTS")
            commit = api.upload_file(
                path_or_fileobj=str(archive),
                path_in_repo=path,
                repo_id=repo,
                repo_type="model",
                revision="main",
                parent_commit=head.sha,
                create_pr=False,
                commit_message="Frozen Colab launch input",
            )
            revision = commit.oid
            if not re.fullmatch(r"[a-f0-9]{40}", revision):
                raise ValueError
            info = api.model_info(repo, revision=revision)
            files = api.get_paths_info(repo, [path], revision=revision, repo_type="model")
            if (
                info.private is not True
                or len(files) != 1
                or files[0].path != path
                or files[0].size != archive.stat().st_size
                or getattr(files[0].lfs, "sha256", None) != sha256
            ):
                raise ValueError
            copied = download(
                repo_id=repo,
                filename=path,
                revision=revision,
                repo_type="model",
                token=token,
                endpoint="https://huggingface.co",
                force_download=True,
            )
            if checksum(copied) != sha256 or checksum(archive) != sha256:
                raise ValueError
            return {
                "verified": True,
                "revision": revision,
                "sha256": sha256,
                "bytes": archive.stat().st_size,
                "path": path,
            }
    except Exception:  # pylint: disable=broad-exception-caught
        raise ValueError("HF_INPUT_PUBLICATION_UNVERIFIED") from None


def validate_receipt(receipt):
    if (
        set(receipt) != {"verified", "revision", "sha256", "bytes", "path"}
        or receipt["verified"] is not True
        or not re.fullmatch(r"[a-f0-9]{40}", receipt["revision"])
        or not re.fullmatch(r"[a-f0-9]{64}", receipt["sha256"])
        or type(receipt["bytes"]) is not int
        or receipt["bytes"] <= 0
        or receipt["path"] != "launch-inputs/" + receipt["sha256"] + "/colab-package.zip"
    ):
        raise ValueError("INVALID_INPUT_RECEIPT")
    return receipt


def downloader_code(receipt):
    """Stdlib child reads only a separately delivered private file; never prints it."""
    validate_receipt(receipt)
    return (
        """import hashlib, json, os, urllib.request, urllib.parse
from pathlib import Path
class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urllib.parse.urlsplit(newurl)
        allowed = ('huggingface.co', 'hf.co', 'amazonaws.com', 'cloudfront.net', 'xethub.hf.co')
        if (p.scheme != 'https' or p.username is not None or p.password is not None
                or p.port not in (None, 443) or p.fragment
                or not any(p.hostname == h or (p.hostname or '').endswith('.' + h)
                           for h in allowed)):
            raise RuntimeError('HF_REDIRECT_REJECTED')
        return urllib.request.Request(newurl)  # NO headers, never forward HF bearer.
values = {}
for line in Path('/content/compare-attention-private/kaggle-env.env').read_text().splitlines():
    key, sep, value = line.partition('=')
    if sep and key in ('HF_TOKEN', 'HF_REPO'):
        values[key] = json.loads(value)
receipt = """
        + repr(receipt)
        + """
EXPECTED_SHA256 = receipt['sha256']
url = ('https://huggingface.co/' + values['HF_REPO'] + '/resolve/'
       + receipt['revision'] + '/' + receipt['path'])
request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + values['HF_TOKEN']})
opener = urllib.request.build_opener(SafeRedirect())
archive = Path('/content/colab-package.zip')
partial = archive.with_suffix('.partial')
if archive.exists() or partial.exists():
    raise RuntimeError('ARCHIVE_TARGET_EXISTS')
try:
    hasher, size = hashlib.sha256(), 0
    with opener.open(request, timeout=60) as response, partial.open('xb') as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            size += len(chunk)
            if size > receipt['bytes']:
                raise RuntimeError('ARCHIVE_SIZE_MISMATCH')
            hasher.update(chunk)
            output.write(chunk)
    if size != receipt['bytes'] or hasher.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('ARCHIVE_SHA_MISMATCH')
    partial.rename(archive)
finally:
    partial.unlink(missing_ok=True)
del values, request, url
"""
    )


def render_env(values):
    """Controlled JSON-quoted dotenv; interpolation stays disabled in the consumer."""
    if any(not isinstance(values.get(key), str) or not values[key].strip() for key in KEYS):
        raise ValueError("INVALID_PRIVATE_INPUT")
    if any(ord(c) < 32 or ord(c) == 127 for key in KEYS for c in values[key]):
        raise ValueError("INVALID_PRIVATE_INPUT")
    return "".join(
        key + "=" + json.dumps(values[key], ensure_ascii=False) + "\n" for key in KEYS
    ).encode("utf-8")


class Contents:
    """Documented Jupyter Contents API through the Google runtime proxy only."""

    def __init__(self, connection, *, opener=None):
        try:
            url, token = connection["url"], connection["token"]
            parsed = urllib.parse.urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
                or parsed.query
                or parsed.fragment
                or not any(
                    (parsed.hostname or "").endswith("." + domain)
                    for domain in ("googleusercontent.com", "googleapis.com", "google.com")
                )
                or not isinstance(token, str)
                or not token
                or any(ord(c) < 33 or ord(c) > 126 for c in token)
            ):
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise ValueError("INVALID_PROXY") from None
        self._url = url.rstrip("/") + "/api/contents/"
        self._token = token
        self._open = opener or urllib.request.build_opener(NoRedirect()).open

    def request(self, path, *, data=None):
        if path.startswith("/") or any(p in ("", ".", "..") for p in path.split("/")):
            raise ValueError("INVALID_CONTENTS_PATH")
        url = self._url + urllib.parse.quote(path, safe="/")
        return request_json(
            url + ("?content=1&format=base64" if data is None else ""),
            headers={
                "X-Colab-Runtime-Proxy-Token": self._token,
                "Content-Type": "application/json",
            },
            data=None if data is None else json.dumps(data).encode(),
            method="GET" if data is None else "PUT",
            opener=self._open,
        )

    def read(self, path):
        model = self.request(path)
        try:
            if model.get("type") != "file" or model.get("format") != "base64":
                raise ValueError
            return base64.b64decode(model["content"], validate=True)
        except (ValueError, KeyError, TypeError):
            raise ValueError("INVALID_CONTENTS_RESPONSE") from None

    def write(self, path, data):
        if len(data) > 65536:
            raise ValueError("SMALL_FILES_ONLY")
        model = self.request(
            path,
            data={
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            },
        )
        if model.get("type") != "file" or self.read(path) != data:
            raise ValueError("CONTENTS_READBACK_FAILED")

    def discover_root(self, name, nonce):
        """Match an MCP-written NONSECRET /content marker, never guess a secret sink."""
        matches = []
        for prefix in ("", "content/"):
            try:
                if self.read(prefix + name) == nonce.encode("ascii"):
                    matches.append(prefix)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        if len(matches) != 1:
            raise ValueError("CONTENTS_ROOT_UNVERIFIED")
        return matches[0]
