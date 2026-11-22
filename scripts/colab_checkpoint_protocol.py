"""Secret-free checkpoint identities and atomic local proof files (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def checksum(path):
    """Hash streamed bytes, never deserialize a training snapshot."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    """Canonical secret-free metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def receipt_path(result):
    """Proof lives beside, not inside, the fresh trainer result sink."""
    result = Path(result)
    return result.parent / (result.name + ".hf-checkpoint-receipt.json")


def identity(package, result):
    """Bind transport to approved package, source and prepared data identity."""
    package, result = Path(package), Path(result)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", result.name):
        raise ValueError("Unsafe checkpoint run name")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    value = {key: manifest[key] for key in ("source_sha256", "data_fingerprint")}
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()):
        raise ValueError("Invalid frozen checkpoint identity")
    return {
        "schema_version": 1,
        "run": result.name,
        "package_manifest_sha256": checksum(package / "manifest.json"),
        **value,
    }


def atomic_json(path, value):
    """Publish a flushed local proof without partially replacing its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".hf-proof-", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
