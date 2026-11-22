"""Secret-safe Jupyter Contents transport; no retries, no Google bearer forwarding."""

# Keep all proxy boundary checks together for auditability.
# pylint: disable=too-many-boolean-expressions

import base64
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject redirects before sending credentials elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("PROXY_REDIRECT_REJECTED")


class Contents:
    """Keep the complete provider connection in memory only."""

    def __init__(self, connection, *, opener=None):
        try:
            url, token = connection["url"], connection["token"]
            parsed = urllib.parse.urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
                or parsed.fragment
                or not parsed.hostname
                or not any(
                    parsed.hostname.endswith("." + domain)
                    for domain in ("google.com", "googleusercontent.com", "colab.dev")
                )
                or any(ord(c) < 33 or ord(c) > 126 for c in url)
                or not isinstance(token, str)
                or not token
                or any(ord(c) < 33 or ord(c) > 126 for c in token)
            ):
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise ValueError("INVALID_PROXY") from None
        self._base = parsed
        self._token = token
        self._open = opener or urllib.request.build_opener(NoRedirect()).open

    def request(self, path, *, data=None, content=True):
        """One bounded request; error bodies and headers are never exposed."""
        if (
            not isinstance(path, str)
            or path.startswith("/")
            or any(part in ("", ".", "..") for part in path.split("/"))
            or "\\" in path
            or any(ord(c) < 32 for c in path)
        ):
            raise ValueError("INVALID_CONTENTS_PATH")
        query = self._base.query
        if data is None:
            query += ("&" if query else "") + (
                "content=1&format=base64" if content else "content=0"
            )
        url = urllib.parse.urlunsplit(
            (
                self._base.scheme,
                self._base.netloc,
                self._base.path.rstrip("/") + "/api/contents/" + urllib.parse.quote(path, safe="/"),
                query,
                "",
            )
        )
        request = urllib.request.Request(
            url,
            data=None if data is None else json.dumps(data).encode(),
            headers={
                "X-Colab-Runtime-Proxy-Token": self._token,
                "Content-Type": "application/json",
            },
            method="GET" if data is None else "PUT",
        )
        try:
            with self._open(request, timeout=60) as response:
                if response.status not in (200, 201):
                    raise ValueError("CONTENTS_UNVERIFIED")
                raw = response.read(262145)
                if len(raw) > 262144:
                    raise ValueError("CONTENTS_RESPONSE_TOO_LARGE")
                model = json.loads(raw)
                if not isinstance(model, dict):
                    raise ValueError("CONTENTS_INVALID_RESPONSE")
                return model
        except urllib.error.HTTPError as error:
            if error.code == 404 and data is None:
                raise FileNotFoundError("CONTENTS_ABSENT") from None
            raise ValueError("CONTENTS_HTTP_FAILED") from None
        except (OSError, http.client.HTTPException):
            raise ValueError("CONTENTS_STATE_UNKNOWN") from None
        except (ValueError, TypeError):
            raise ValueError("CONTENTS_UNVERIFIED") from None

    def absent(self, path):
        """Refuse unknown existing targets, including empty files."""
        try:
            self.request(path, content=False)
        except FileNotFoundError:
            return
        raise ValueError("TARGET_EXISTS")

    def put(self, path, value, *, chunk=None):
        """One write, deliberately never retried on an ambiguous response."""
        model = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(value).decode("ascii"),
        }
        if chunk is not None:
            model["chunk"] = chunk
        if self.request(path, data=model).get("type") != "file":
            raise ValueError("CONTENTS_WRITE_UNVERIFIED")

    def probe_chunks(self, path, record):
        """Empirically distinguish LargeFileManager from overwrite-only managers."""
        self.absent(path)
        for number, value in ((1, b"chunk-one/"), (2, b"two/"), (-1, b"end")):
            record({"phase": "probe_chunk_pending", "chunk": number})
            self.put(path, value, chunk=number)
        if self.read(path) != b"chunk-one/two/end":
            raise ValueError("CHUNK_UPLOAD_UNSUPPORTED")

    def upload_archive(self, path, source, record, *, chunk_size=1024 * 1024):
        """Bounded memory and durable pending-before-write; final -1 runs hooks."""
        if not 1 <= chunk_size <= 1024 * 1024:
            raise ValueError("INVALID_CHUNK_SIZE")
        self.absent(path)
        with source.open("rb") as stream:
            number, total = 1, 0
            value = stream.read(chunk_size)
            while value:
                following = stream.read(chunk_size)
                chunk = number if number == 1 or following else -1
                record(
                    {
                        "phase": "chunk_pending",
                        "file": "archive",
                        "chunk": chunk,
                        "bytes_before": total,
                    }
                )
                self.put(path, value, chunk=chunk)
                total += len(value)
                value, number = following, number + 1
            if number == 2:  # Single-chunk uploads still need final hooks.
                record(
                    {
                        "phase": "chunk_pending",
                        "file": "archive",
                        "chunk": -1,
                        "bytes_before": total,
                    }
                )
                self.put(path, b"", chunk=-1)
            if number == 1:
                raise ValueError("EMPTY_ARCHIVE")

    def write_private(self, path, value, record):
        """No size/hash/credential values enter the journal."""
        if not value or len(value) > 65536:
            raise ValueError("INVALID_PRIVATE_INPUT")
        self.absent(path)
        record({"phase": "private_write_pending"})
        self.put(path, value)
        if self.read(path) != value:
            raise ValueError("PRIVATE_READBACK_FAILED")
        record({"phase": "private_readback_verified"})

    def discover_root(self, name, nonce):
        """Read only a known owned nonsecret marker; never list remote roots."""
        matches = []
        for prefix in ("", "content/"):
            try:
                if self.read(prefix + name) == nonce.encode("ascii"):
                    matches.append(prefix)
            except FileNotFoundError:
                pass
        if len(matches) != 1:
            raise ValueError("CONTENTS_ROOT_UNVERIFIED")
        return matches[0]

    def read(self, path):
        """Only small marker/private files, never the full archive."""
        model = self.request(path)
        try:
            if model.get("type") != "file" or model.get("format") != "base64":
                raise ValueError
            encoded = model["content"]
            if not isinstance(encoded, str):
                raise ValueError
            # FileContentsManager uses base64.encodebytes (RFC2045 line wrapping).
            encoded = encoded.replace("\n", "").replace("\r", "")
            return base64.b64decode(encoded, validate=True)
        except (KeyError, ValueError, TypeError):
            raise ValueError("CONTENTS_INVALID_RESPONSE") from None
